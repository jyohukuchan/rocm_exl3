# V620×2 Tensor Parallel

既存のQwen3.8 Flash Next 3.05bpwパックと、同パック内のMTP重み（主部3bit、input 4bit、attention/shared 5bit）を使う。自前量子化MTP3/5へのoverrideは使用しない。レイヤー分割の起動方法は引き続き利用可能。

## 実装

- 本家 `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` のQwen TP関連Python処理を移植。
- HIPでは `tp_backend="nccl"` をpure Torch/RCCL backendへ振り分ける。CUDA native `pg_*` は呼ばず、FP32通信をFP32のまま行う。all-reduce/broadcast/uneven・部分参加gather対応。
- GDN等をテンソル分割、MoEを既定のexpert分割で実行。QSAは層ごとに片方のGPUが所有する。本家同様、全モジュールを一律に半分にする構成ではない。
- PLE/Engramは単一owner。親の表読み込みを遅延し、ownerだけがRAMに32,640,156,672 bytesの表をロードする。共有メモリへの表複製やdisk streamingへの切り替えは行わない。
- MTPは出力側GPU1に配置し、本体のembedding/lm_headをTP経由で共有する。MTPの動的候補長上限4、confidence 0.6を維持。
- batch=1の電力設定はprefill auto、draft/verification/decode profile_peak、終了後auto。TPの切り替え前同期は各workerのGPUコンテキストで実行する。

## 現時点の検証

Dense 8B、MoE 30B、Qwen3.8でTP2の自然な日本語生成を確認。Qwen3.8では既存3bit MTP生成も確認した。

Qの初回TPで出力が破綻した原因は、GatedRMSNormのTP export/importが`gate_activation`を落とし、sigmoidを既定SiLUに変えていたこと。本家の該当依存修正を取り込み、同じ入力の通常生成とMTP生成は自然な同一出力へ復旧した。比較したteacher-forced全vocab logitsはD/M各58位置、Q 27位置でtop-1一致、全値finite。Qのlogits relative L2は0.0201。これは全入力での完全一致を保証する試験ではない。

## 性能（暫定）

V620×2、batch=1、D/Mは2048入力、Qは8192入力、それぞれ256生成。自然な日本語/コードの固定token IDs、各課題warmup1+測定5。中央値。プロファイラなし。nativeは`lib-qwen38-reduction`（SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`）。

| モデル/課題 | LS prefill tok/s | TP prefill tok/s | LS decode tok/s | TP decode tok/s |
|---|---:|---:|---:|---:|
| Qwen3-8B 4bpw 日本語 | 1136.4 | 1999.7 | 56.96 | 48.98 |
| Qwen3-8B 4bpw コード | 1130.0 | 1985.5 | 57.42 | 49.04 |
| Qwen3-30B-A3B 3bpw 日本語 | 907.9 | 1587.0 | 68.23 | 39.48 |
| Qwen3-30B-A3B 3bpw コード | 924.6 | 1613.1 | 67.97 | 39.25 |
| Qwen3.8 + MTP3 / 8K 日本語 | 305.5 | 473.0 | 36.59 | 36.81 |
| Qwen3.8 + MTP3 / 8K コード | 296.4 | 463.2 | 40.81 | 42.56 |

prefillはengineの入力処理時間、decodeは最初のtoken配送後の観測レートを使う。単発の遅い反復も除外していない（D TP日本語33.52 tok/s、LSコード43.95 tok/sの反復あり）。Q 8Kの入力処理から256生成終了までの中央値は日本語33.95→25.08秒、コード34.15→23.71秒。prefill改善が全体の短縮に効いている。engineの`time_generate`だけを使う従来decode指標ではQ日本語39.24→38.57、コード55.38→58.53 tok/sであり、配送・最終処理も含む上表と混同しない。MTP採用率中央値はLS/TPで日本語54.6%/52.6%、コード78.7%/83.5%。生成経路の分岐により出力や採用率が変わるため、実測の速度差をすべて通信・演算だけの差とは解釈しない。

Q 8K+64の別検証ではtarget57回/draft154回、rank0/1で3175/3431回の有限値チェックに合格。8K+256の全反復前後もEngram 7,968,789ページすべてのRAM常駐を`mincore`で確認。プロセス全体VmSwapだけで表の退避を推測しない。

MoEの2 broadcastをTorch/RCCL coalescingでまとめる候補は、実通信の正しさは合格したがlatencyが約84→116µs（1row）に悪化したため不採用。差分と測定はartifactに保存し、本体は既存の逐次broadcastを維持。GPUトレースでは両rankに各655回のRCCL kernelと約6000回の演算kernelを確認した。約1.22秒の混合prefill/decode窓で演算が約952ms重なり、両GPUの並列実行を実測した。計測kernelはすべて指定窓内にあり、プロファイラ付き実行も通常終了した。これはプロファイラなしの速度表とは別の確認である。小さいMoEのdecode等、TPが遅い条件ではレイヤー分割を選べる。

## 記録

artifact: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/tp2`。

- `rccl-backend-rank{0,1}.json`: 実RCCL、strided FP32/非参加NaN/順序変更/uneven subset gather。
- `d-logits-initial.json`, `m-logits-initial.json`, `q-logits-gatefix.json`: 同じ量子化重みのLS/TP比較。
- `q-tp-smoke-gatefix.json`, `q-tp-mtp-smoke-gatefix.json`: 修正後の生成確認。
- `d-{ls,tp}-2k-r1.json`: 固定入力の速度比較。
- 初回Q破綻結果、入力special-token設定を誤った初回D結果は合格データに使用していない。

## 長時間実行・後始末

AR連続実行では、旧コードが8リクエスト完了後のキャッシュdefragでGPU書込faultを起こした。原因は、rotation指示をpinned共有メモリからGPUへ非同期転送した直後にCPU側の完了応答を返し、親が次入力でarenaを上書きできたこと。`mp_rotate_cache_pages`の最後だけでworkerのcurrent streamを同期し、転送とrotationが完了してから応答するよう修正した。トークンごとの同期は追加していない。

有効な別permutationで安全に上書きする実GPU回帰試験は旧3/3破損→修正3/3保持。同じ8K+256×12 ARリクエストも修正前2回とも同じ位置で失敗、同期だけを追加した版は全件完走した。MTP利用時もこの修正を使う。

worker生成途中の失敗、既に停止したrankへのdispatch、同一Modelの再ロードを検証した。全所有workerを回収して再ロードでき、前後20位置のlogitsは完全一致した。既に停止したrankをGPU collectiveへ入る前に検出し、cleanup側のエラーも報告する。

最終コードでは以下も合格した。

- batch2: 8K+256を日本語/コード同時、MTP固定2。target131回/draft242回のfinite検査、Engram全ページRAM、peak維持→終了auto。
- batch1: 32K+256を日本語/コード各1件、MTP固定4。target193回/draft636回のfinite検査、Engram全ページRAM、指定のauto/peak切り替え。Torchのrank別allocated peakは約26.1/27.4GiB。
- CPU suite: 591 tests + 53 subtests合格。

## 起動条件と再現

推奨値は `doc/qwen38_v620_tp_config.json`。既存LS用のMTP設定も保存してある。JSONは自動読込されないのでrunner/APIに明示して渡す。

このコンテナでは `LD_PRELOAD=libhsa-runtime64.so` と、host ROCmを先頭にした既存 `LD_LIBRARY_PATH` を使う。絶対pathの `.so.1` 指定ではTorch同梱HSAも追加ロードされた。bare nameなら全rankがhost HSA1.21だけをロードする。二重ロードはprofilerのSDK初期化衝突を起こしたが、上記ARのdefrag競合とは独立した問題だった。

TPに必要なAPI引数は次のとおり。Cacheはロード前に構築し、MTP headはGPU1へ先にロードする（完全な実例は `rocm_tools/rdna2/tp_run.py`）。

```python
config.infer_params.ngram_stream_from_disk = False
draft.load(device="cuda:1", max_chunk_size=2048)
model.load(tensor_p=True, tp_backend="nccl", tp_output_device="cuda:1",
           use_per_device=[28, 28], max_chunk_size=2048)
```

ROCmでは`tp_backend="nccl"`を指定する。`native` backendのCUDA専用通信を使う設定ではない。電力切替前の同期は各workerへdispatchするadapterを使う。親プロセスだけのCUDA同期では子rankのコンテキストを同期できない。

このworkspaceの検証を再実行するhost側コマンド例（モデル再取得は不要、未使用のtagを指定）。helperの起動・電力設定の復帰まで含む。

```bash
python3 /home/homelab1/datapool/rocm-exl3-rdna2/runs/tp2/run_tp_single_hsa.py \
  --tag q-tp-local-retest --source /src \
  --model /work/models/qwen38-flash-next-exl3-3.05bpw \
  --execution tp --mode mtp \
  --prompts /work/runs/qwen38-mtp/formal8k-prompts.json
```

`rocm-exl3-v620-pair`コンテナを使い、`/src`をrepo、`/work`をdata rootとする。R9700はこのTP splitに含めない。
