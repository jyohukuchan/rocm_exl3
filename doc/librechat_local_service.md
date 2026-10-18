# LibreChat local service — 2026-10-18

このホストの既存LibreChatとQwen3.8 Flash Nextサービスを、PCから利用するための運用メモ。
一般向けの新規インストール手順ではない。

## 接続先と認証

- LAN内のPC: <http://192.168.0.66:3080>
- サーバー自身: <http://127.0.0.1:3080>
- 既存アカウントの認証情報: `/home/homelab1/coding-local/librechat/.local/login.json`（mode 600）。パスワードやトークンはGitへ保存しない。
- 別ネットワークからは既存のSSH接続で `ssh -N -L 3080:127.0.0.1:3080 homelab1@<SSH接続先>` と転送し、PCの <http://localhost:3080> を開く。

LibreChatの `compose.local.yaml` に `192.168.0.66:3080:3080` を追加し、
既存の `127.0.0.1:3080:3080` も維持した。変更前ファイルは同ディレクトリの
`.local/compose.local.yaml.before-lan-20261018-121155` に保存した。
LANアドレスが変わった場合はポートのbind先も更新する。
認証設定、モデル選択、MCP設定、会話データは維持。
ルーターのポート転送設定は変更していない。

## 起動と停止

```bash
systemctl --user start rocm-exl3-api.service
docker compose -f /home/homelab1/coding-local/librechat/compose.local.yaml up -d

systemctl --user status rocm-exl3-api.service
curl http://127.0.0.1:3953/health
curl http://127.0.0.1:3080/readyz
```

systemdがactiveでもモデルのロード中には推論できない。`/health` がHTTP 200を
返してからチャットする。起動時点の既存構成はV620×2 TP2/RCCL、batch 3、
共有768Ki cache、MTP1、RAM Engram、vision有効、xhigh。
実際のモデルパスは既存ランチャーの
`/work/models/qwen38-flash-next-orcarouter-exl3-3.05bpw-original-vision`。

GPUを空けるには、利用中の生成がないことを確認して次を実行する。

```bash
systemctl --user stop rocm-exl3-api.service
```

Web UIも止める場合は、データを残す `stop` を使う。

```bash
docker compose -f /home/homelab1/coding-local/librechat/compose.local.yaml stop
```

## 今回の確認

- 停止していた既存 `rocm-exl3-api.service` を起動。検証終了時の再起動回数0。
- LibreChatとMongoDBのDocker healthcheck正常。
- ホストからLANアドレス宛ての画面取得、`/readyz`、既存アカウントのログインがHTTP 200。
- LibreChatの通常チャットAPIとSSEを経由し、Qwenが「接続できました」と回答。
- 別の物理PCでのブラウザ操作、Office連携、MCP実行、画像入力は今回の検証対象外。

検証スクリプトと結果は
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/librechat-start-20261018/`
に保存した。推論サービスは既存のenabled設定、LibreChatは既存の
`restart: unless-stopped` を維持している。

## 軽量Web検索（2026-10-18追加）

画面を再読み込みして新しいチャットを開くと、`web-search` を含む4種のMCPが
既定で利用可能になる。一般検索は既存SearXNGのBing経由で行う。
見出し・URL・短い抜粋のみ取得し、本文取得やrerankは行わない。
本文が必要な場合は別途Firecrawlを使う。
これは標準Web検索ボタンとは別のMCP機能。

実際のLibreChatチャットで検索1回と出典URL付き回答を確認済み。
設定、再起動、エンジン選択の理由は
[軽量検索のREADME](../rocm_tools/web_search_mcp/README.md)を参照。

## MCPの自動選択（2026-10-18）

既存モデルspecの `mcpServers` に `web-search`、`firecrawl`、`huggingface`、
`openai-docs` を設定。`preset.promptPrefix` で用途別の選択方針を指定した。
通常の会話や簡単な計算はツールなし、一般検索は軽量検索、指定URLの本文は
Firecrawl、モデル情報はHugging Face、OpenAI公式仕様はOpenAI Docsを優先する。
有効化されていても毎回ツールを実行するわけではない。

新しいチャットでは手動のMCP選択が不要。既存チャットの保存済み選択は残るため、
設定反映後は画面を再読み込みして新規チャットで利用する。
変更はLibreChatのローカルYAMLのみで、モデル推論サービスの再起動は不要。

ブラウザ実操作でMCPメニューに触れず新規チャットを3件送信し、4種のMCPが
自動選択されたリクエストを確認。計算はツールなし、公式ページ検索は軽量検索、
指定URLの要約はFirecrawlを選択した。検索結果だけで足りるURL案内で不要な
本文取得を控える指示も追加した。AIの判断による選択なので常に同じ経路に
なることを保証するものではない。
