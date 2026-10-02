# 2026-10-02 ホスト停止の調査とサービス復旧

ホスト再起動後に、ログ調査と OpenCode / EXL3 / MCP の復旧を実施した。
時刻はすべて日本時間。今回の主な証拠は **ホストRAMの逼迫、swap飽和、
継続的なページング・回収**である。GPUリセットやストレージ障害、
OOM killの記録は見つからなかった。

## 停止前の状況

- 前回boot: `d153b2f7e7d24ec2b1a2500f740f51be`。
  最終journalは20:29:33の `Under memory pressure, flushing caches.`。
  正常なshutdown記録はなく、次のbootは20:37:16に始まった。
- 19:40の時点で8GiB swapが99.98%使用されていた。
  大規模変換の開始前から余裕が少なかった点に注意する。
- 20:08頃からR9700でQwen3.8 Flash Next OrcaRouterの変換が実行された。
  入力は約336GiBのsafetensors、Engramは320,001,536行、K=5。
- 20:20のsysstatではswap使用率99.99%、major fault約5,778回/秒、
  swap out約339ページ/秒。anonymous pagesは約88.34GiB。
  20:21のOpenCode記録でも、RAM約109GiB中101GiB使用、available約8GiBだった。
- 同時に動いていたV620 APIは、以前の配置監査でhost PSS約44.6GiB、
  Engram 32,640,156,672 bytesをmlockしていた。
  大規模量子化コンテナにはRAM・swap上限が設定されていなかった。
- V620 `43:00.0` に18:52:53～20:29:27の間、GPU queue/runlistの
  oversubscription警告が1,352件あった。性能低下の兆候ではあるが、
  GPU reset/ring timeoutの記録はなかった。
- OOM kill、systemd-oomdによるkill、kernel panic、hung task、NVMe/ZFS
  I/O errorの記録は見つからなかった。ZFS poolは正常だった。

主因は、既に大きなRAM常駐量がある状態で変換負荷を重ねたことによる
メモリ圧迫と考えられる。再起動前の継続的なプロセス別メモリ記録はないため、
増加したallocationやmmapの内部まで特定したわけではない。
ZFS ARCの上限はカーネル自動値（物理RAM−1GiB）だったが、停止直前の
統計はanonymous memory優勢で、ARC単独を原因とは断定できない。

調査したOpenCode sessionは `ses_f0518fcebffelFQlNImhkZZ2Ha`。
変換開始時のcommitは `0c70ff5`。変換はEngram工程で中断し、
出力 `ngram_embedding.safetensors` は約9.52GBの未完成ファイルだった。
この復旧作業では大規模変換を再開していない。

## 復旧した構成

### EXL3 / OpenCode / LibreChat

以前の `rocm-exl3-api.service` とloopback proxyはtransient unitだったため、
ホスト再起動でunit定義が失われていた。V620コンテナも `restart=no` だった。

ローカルの永続user unit
`~/.config/systemd/user/rocm-exl3-api.service` を作成し、enableした。
`Linger=yes` を確認済み。起動時に既存V620コンテナをstartし、
既存 `runs/opencode-api/run_live.py` を新しい `start_service.py` から実行する。

- TP2/RCCL、V620×2、batch=3、共有context 786432 tokens（768Ki）
- 固定MTP1、K5/V4、GPU budgets 28.5/27.5GiB、xhigh
- Engram単一RAM owner / mlock、既存expert配置、Vision、timing footer
- loopbackはlauncher内蔵bridgeで `127.0.0.1:3953` に公開
- コンテナIPを起動ごとに解決し、古い固定IPに依存しない
- 再起動ごとに新しい短いrun tagを作り、過去のsocket/reportと衝突させない
- `Restart=always`、20秒backoff、600秒あたり5回の起動制限

復旧時に一度、長いrun tagがAF_UNIX socketの108-byte制限を超えたため、
短いtagと事前長チェックへ修正した。内蔵bridgeのcancel済みcopy taskもawaitする。
これらはローカルlauncherの変更で、推論カーネルの変更ではない。

正常起動run: `api-20261002-204615-1cfb`。
`/health` はHTTP 200、`/props` はcontext 786432 / slots 3を返した。
生成確認は `enable_thinking=false` を指定した単発のprobeで `OK` / `stop`。
本番の既定thinkingはxhighのまま。

OpenCode本体は再起動後に既に動いていたため、その会話を停止せず接続先を復旧した。
約256,004 tokensの既存会話が再投入され、KV cacheを作り直している最中だった。
この負荷中はhealth/propsにも約28秒、短い生成確認にも約36秒かかった。
スタック採取では実際にprefillを実行しており、ホストの再フリーズとは区別できた。

LibreChatとMongoDBは自動復帰しており、LibreChatのHTTP 200とhealthy状態を確認した。
共有Docker networkの `exl3:3953` 接続名は維持している。

### MCP / Firecrawl

FirecrawlのMCP frontendは起動していたが、backendの各コンテナは停止していた。
既存volumeを維持して、API、Playwright、Redis、RabbitMQ、NuQ Postgres、
FoundationDBを復旧した。これらのcompose設定と実コンテナを
`restart: unless-stopped` に変更し、one-shotのFoundationDB initは `restart=no` を維持した。

MCPクライアントが送る `domainTools` / `toolDetail` をbackend検索schemaが拒否する
互換性問題も見つかった。Firecrawl側でこの2項目だけをoptional metadataとして受け入れ、
検索処理に渡す前に除去した。検索filterは保持する。

ローカル変更先:

- `~/firecrawl/docker-compose.yaml`
- `~/firecrawl/apps/api/src/controllers/v2/types.ts`
- `~/firecrawl/apps/api/src/__tests__/snips/v2/types-validation.test.ts`

API imageのTypeScript build、runtime schema、liveness/readiness、
MCP initialize/tools/list、scrape、search、domain filter付きsearchを確認した。
`opencode mcp list` でもfirecrawl / huggingface / openai-docsの3件がconnected。

## メモリ上限

停止している大規模量子化用コンテナだけに次の上限を設定した:

```bash
docker update --memory 32g --memory-swap 34g rocm-exl3-r9700-conv
```

RAM 32GiB + swap 2GiBという意味で、次の起動から適用される。
上限に達すれば変換がコンテナ内で失敗する可能性がある。
これはホスト全体への影響を抑える設定であり、変換器自身のRAM使用量を
改善したという意味ではない。大規模変換の正常完了は別途検証が必要。

20:59の復旧確認時はavailable RAM約31GiB、swap使用4KiB。
復旧中に新しいmemory-pressure/OOM/GPU-resetのjournal記録はなかった。

## ローカル確認コマンド

```bash
systemctl --user status rocm-exl3-api.service firecrawl-mcp.service
opencode mcp list
curl --max-time 55 http://127.0.0.1:3953/health
docker ps
journalctl -b -1 --since '2026-10-02 19:30:00' --no-pager
```

詳細な新runログは
`~/datapool/rocm-exl3-rdna2/runs/opencode-api/api-20261002-204615-1cfb.log`。
認証情報を除いた復旧時点の一覧は
`~/datapool/rocm-exl3-rdna2/runs/recovery-20261002/status.json` に保存した。
