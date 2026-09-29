# Qwen3.8 Flash Next / V620×2 MTP

2026-09-29。ユーザーの新しいMTP有効化指示に基づき、以前の投機生成無しのAR結果をbaselineとして保存し、MTPを実行・調整した。電力方針はbatch1がprefill auto / drafting・verification・decode profile_peak / 終了後auto、batch>1を許可する構成では推論中profile_peak。これはエコ目的で、約400Wでの冷却能力不足への対策ではない。

## 固定条件と実装

- 本体は既存EXL3 3.05bpw（head5/ngram5）、モデルrevision `69e33439ae950f17bcbe95c98f117d80f759ab6d`。Engram CPU RAM常駐。本体量子化・重みは変更していない。
- MTPはパック内の3bit重み。MTP本体とmixer patchは計1,037,230,632bytes（約0.966GiB）、新規モデル取得無し。実ロードの追加torch割当は約1.1GiB。MTPを共有lm_headと同じcuda:1へ先にロードする。
- nativeは `/work/lib-qwen38-reduction`、SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`。nofile soft65536、HSA_ENABLE_SDMA=0、V62043→03の2枚、layer split。TPは今回未実装。
- target Cacheは実際のbatch数に合わせたslot数と、候補長上限ぶんのmax_historyを確保する。既定16slotにMTP履歴を追加すると不要なVRAMを消費するため、batch1は1slot、batch2は2slotを使う。
- 既存のMTP本体・draft/verify/rollback実装で動作したため、推測だけで数式やGPU kernelは変更していない。OpenCodeの読取り監査も実証バグを検出せず。stream_tapの意味は上流でも経験的な説明が残るが、本パックで採用率と実生成を確認。
- `rocm_tools/rdna2/power_policy.py`をOpenCodeに実装委任しrootレビュー。MTPはtarget verifyより前にdraft計算があるため、その入口前にpeakへ切り替える。ACKのGPU数、初回auto、例外/接続終了、同期時間とRPC時間の分離等を強化。
- `rocm_tools/rdna2/qwen_mtp_run.py`もOpenCode実装をrootがレビュー。固定入力hash・語彙境界・cache容量、burst-aware進捗計測、warmup区別、配置/Engram RAM監査、正常終了、失敗JSON、power制御を統合。CPU323tests成功、batch2の実GPU finite smokeも成功。AR比較でもMTP重みを常駐させ、同じ履歴容量・配置条件を使う。

## 正常動作・状態検証

| 検証 | 結果 |
|---|---|
| 日本語説明・Python生成、MTP4 | 各128生成、target83/draft324回の検査対象forwardが全finite。初回JITを含むため速度には不採用 |
| 8K入力＋256生成、batch1/MTP4 | 正常終了、target82/draft308回全finite、accepted179/rejected129、Swap0、power auto→peak→auto |
| 2K入力＋256生成×2、batch2/MTP2 | 正常終了、target94/draft180回全finite、両job accepted166/rejected14、Swap0、推論中peak維持 |
| 32K入力＋256生成、batch1/MTP4 | 正常終了、target82/draft260回全finite、accepted191/rejected69、Swap0。torch peak約27.96/24.57GiB |

32K初回はEngram RAM事前guardで停止（必要31128MiB＋reserve2048に対しMemAvailable32471MiB）。ZFS ARC約63GiBが別途回収可能な109GiBホストであることを確認し、再試行のprocessだけEXL3_HOST_MEM_RESERVE_MB=0とした。OS/ZFS設定は不変更。後続の比較も同じprocess設定を記録して実施。推論faultやOOMを無視したものではない。

## 2K入力の候補長・confidence比較

日本語の長い説明とPython実装の2課題、各warmup1＋測定3、256生成、greedy。全設定で同一入力ID/hash、cache miss。AR/固定1/2/4/動的4(.4)が40job、固定3/動的4(.6/.8)が24job、計64jobを正常終了。最初の40jobは同一process、追加24jobは別processであり、allocator/checkpoint履歴まで完全一致とはしない。

| 設定 | 日本語decode tok/s | コードdecode tok/s | 日本語全体秒 | コード全体秒 |
|---|---:|---:|---:|---:|
| AR | 36.10 | 36.07 | 13.628 | 13.866 |
| 固定1 | 38.81 | 47.70 | 13.620 | 12.635 |
| 固定2 | 38.32 | 56.36 | 14.270 | 11.481 |
| 固定3 | 37.16 | 58.10 | 13.521 | 11.489 |
| 固定4 | 30.81 | 56.79 | 15.568 | 11.537 |
| 動的・上限4 / confidence0.4 | 39.64 | 55.69 | 13.785 | 12.297 |
| 動的・上限4 / confidence0.6 | 40.52 | 58.77 | 13.438 | 11.248 |
| 動的・上限4 / confidence0.8 | 39.83 | 56.94 | 13.860 | 11.441 |

各値は3回中央値。decodeは従来のengine time_generate基準。全体秒はprefill・生成・終了処理込み。終了時のRAM返却等のコストに揺れがあり、decode速度向上をそのままリクエスト全体の改善率と呼ばない。greedyでの結果で、sampling条件や入力が変われば採用率も変わる。

固定4は日本語でARより遅くなった。候補を増やすと後半が棄却されるコストが増えるため。動的上限4/confidence0.6では平均候補数が日本語1.76・コード3.35となり、今回の2課題を両立する候補として選定した。0.6と0.8の小差を普遍的な優越とは主張しない。

## 正式8K比較

再利用CLIで同一の12入力（2課題×warm1＋測定5）をAR/MTPで比較中。candidateは上限4・dynamic・confidence0.6、同一cache slot1/history4、MTP重みは両modeで常駐。結果は完了後に追記。

## 再現

以下はホスト側supervisor。専用Unix socket helperを一時起動し、container内のCLIを実行、終了時に両GPUをautoへ戻す。

```bash
python3 /home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38-mtp/run_cli.py \
  --tag my-mtp-run --mode mtp --prompts formal8k-prompts.json \
  --batch 1 --draft 4 --dynamic --confidence 0.6
```

`--tag`は新しい名前を使う。batch>1の入力は同じtimedフラグでバッチを組めるように用意する。`--arc-guard-override`は上記の回収可能RAMを確認した診断用であり、通常の既定値にはしない。内部CLIは `python -m rocm_tools.rdna2.qwen_mtp_run --help`。`--validate-finite`は検証用でGPU同期を加えるため性能値と混ぜない。

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38-mtp`。smoke/validate各JSON、`sweep-mtp.json`、`tune-mtp.json`、`formal8k-*`、OpenCode結果とCPUテストログを保持。既存ARのみのPhase5結果は[こちら](qwen38_v620_results.md)。
