# V620×2 decode 分解・改善

Subsequent context/batch work uses the bounds-fixed native and additional opt-in batch settings. See [context and MTP window measurements](qwen38_v620_context_batch.md) and [RDNA padded-row bounds fix](rdna2_multirow_bounds_fix.md). Native paths and measurements in this report describe the earlier configuration.

2026-09-30。**std MoE routerを両GPUで計算して通知を省き、通常decodeを約10%改善した。** MTPは日本語がほぼ横ばい、コードはengine基準約12%・配送/終了処理込み約5%向上。ただしMTPの出力と採用率も変化したため、全差分を通信だけの効果とは解釈しない。

以後のKVはKey5bit/Value4bit。元の配布済みMTP3パック、Engramの単一RAM表、batch1のprefill auto / draft・verify・decode peak / idle auto、batch>1の推論中peakを維持する。

## 採用した変更

`EXL3_TP_REPLICATE_ROUTER=1`をPython起動前に指定する。std routerをrouted expertを持つ各TP rankへ複製し、選択したexpert indicesと重みの2 broadcast/層を省く。expertの分担と最後の出力集約は維持する。194→98 collective/本体forwardを実測した。GPU0だけを待たせる通知を減らす代わりに、小さいrouterを重複計算する方式。

generic defaultはoff、今回のQwen/V620推奨設定はon。対象外routerは理由を表示して既存経路を使う。router本体とdecode用transposeをVRAM計画へ計上し、routed shardが空のrankには不要なrouterをロードしない。推論後の各workerで比較したTorch allocated増分はGPU0約238.27MiB、GPU1約1.78MiB、計約240MiB。親プロセスのCUDA0=0という値をrank0のVRAMとは扱わない。

設定は`qwen38_v620_tp_config.json`。レイヤー分割用の`qwen38_v620_mtp_config.json`もK5/V4を選択するが、router複製はTP用。これらのJSONは自動読込されない。

## 計測器なしの速度

同一Qwen3.8 Flash Next EXL3 3.05bpw、8K入力＋256生成、日本語/コードそれぞれwarmup1＋測定5、固定token IDs、中央値。比較した有効/無効は同じsource snapshot `113fc57`、同じnative SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`、単一host HSA。有効測定の後にも無効条件を再測定して再現を確認した。

| モード/課題 | 無効 engine decode | 有効 engine decode | 変化 | 無効 prefill | 有効 prefill |
|---|---:|---:|---:|---:|---:|
| AR 日本語 | 31.31 | 34.48 | +10.1% | 465.35 | 464.17 |
| AR コード | 31.32 | 34.54 | +10.3% | 460.39 | 459.02 |
| MTP 日本語 | 38.44 | 38.52 | +0.2% | 451.19 | 452.74 |
| MTP コード | 52.94 | 59.29 | +12.0% | 446.00 | 447.04 |

単位tok/s。engine decodeは既存の`time_generate`基準。配送・最後の後処理まで含む観測decodeは、AR日本語29.41→32.42、ARコード27.15→29.55、MTP日本語35.92→36.74、MTPコード42.48→44.46。MTPコードの観測値の改善は約4.7%であり、上表の12%と混同しない。prefill差は約0.4%以内。

MTP採用率（timed合算）は日本語53.28→51.39%、コード78.14→81.72%。MTPの全文token列は両条件で変化した。VRAM見積りによりTP配置計画も変わるため、採用率変化の原因をrouterの通信削減だけに帰属させない。ARの測定対象10件、前の基準とwarmupを含めた12件の生成token列はすべて一致。

## 実時間のdraft / 本体検証

GPU同期やGPU profilerを追加せず、既存のdecode窓に2箇所の`perf_counter`を加えた。言語ごとに新規processのwarmup1＋代表1件。生成途中約64token（MTP burstの実token数で割る）を測定した。通常のnone controlはAR64tokenが約2.04秒で、clean測定の約31.6tok/sと近い。

| 代表区間 | draft 無効→有効 ms/出力token | 本体検証 無効→有効 ms/出力token |
|---|---:|---:|
| 日本語 | 2.48 → 2.32 | 24.50 → 22.11 |
| コード | 1.98 → 2.07 | 14.02 → 14.07 |

この条件では本体検証が約88～91%。draft層だけを速くしても全体への寄与は限られる。コードのこの短い代表区間では改善していない。正式benchmarkはコードの前に日本語jobを走らせており、calibration履歴も異なるため、この1件を5回中央値の代用にはしない。

日本語のさらに短い16token窓では、生成prefix・候補長・採用数が両条件で一致し、draft2.63→2.61、本体検証24.86→23.44ms/出力tokenだった。棄却された候補IDまで同一だったとの主張ではない。

## GPU処理別の内訳

OpenCodeが実装したtask/phase hooksを、検証用HIP event adapterから利用した。日本語MTPの65→81token、16出力、target8回・借用head18回を両条件で捕捉。通常終了と全hook復帰を確認し、graph capture中はeventを挿入しない。CPUのEngram stage/gatherも別記録。GPU0/1の時間は重なり、以下は**計測付きのstream区間時間**（launch待ちやcollective依存を含む）であって、通常のwall時間や純kernel時間の割合ではない。

| 有効時の主な処理 | GPU0 ms/出力token | GPU1 ms/出力token |
|---|---:|---:|
| 本体MoE（共有expert含む） | 9.89 | 9.70 |
| 本体Gated Residual | 4.01 | 4.00 |
| 本体GDN | 3.10 | 3.04 |
| 本体QSA（projection/indexer/attention） | 2.38 | 1.22 |
| 本体通信・同期 | 4.14 | 5.29 |
| 本体その他（head/embedding/PLE/dispatch等） | 1.39 | 1.70 |
| MTP draft（共有headを含む） | 0.92 | 2.98 |

通信・同期のstream時間は無効時GPU0約9.80、GPU1約7.64ms/出力token。通信回数の減少は確実だが、削除した呼出しの計測負荷も消えるため、時間の改善率は上のclean benchmarkで判断する。EngramのCPU stageは有効時約0.027ms/出力token、その中のRAM gather約0.011ms。今回の窓ではRAM lookupが大きな待ち要因ではなかった。

別の正常終了したAR ROCprofiler traceでも、6144 layer呼出しすべてのGR/HC境界と通信順序を検査してGDN/MoE/QSA/残差/通信へ分離した。量子化Linearのkernel時間にはdequantと行列積が融合しており、「dequantだけの時間」とは呼ばない。

PCIeは両方Gen4 x16。小payload all-reduceは10KB/40KBとも約60µsだった。今回、省いた通知は1行なら80B＋20B/層であり、帯域不足だけでは説明できない。頻繁な通信呼出しとGPU間の到着差・待ち合わせが改善可能な一因だった。**PCIe固有の往復遅延、RCCL内部処理、peer待ちの完全な分離はしていない。**

## QSAの2GPU化を検討した結果

K5/V4でQ24/KV2を各GPU Q12/KV1へ分割する単体試験を実施。1行はbit一致、3/5行はrelative L2約0.00035で参照比較に合格。GPU中央値はfull→遅い方halfで1行144.8→134.0µs、3行225.9→150.9µs、5行352.8→191.3µs。単一host threadから両GPUを起動した実時間は約204µs/呼出しだった。

indexer/projection/TP集約を含まない値で、通常1行decodeの利得は小さい。MTPの複数行検証には余地があるが、追加indexer処理等を含む実モデルでの効果は未検証のためhead分割は採用しなかった。今回2GPUへ移したのはrouter計算であり、EngramのRAM表は複製していない。

## 数値・回帰・終了処理

- 固定AR continuationの全語彙logits96位置はすべてbit一致、top1 96/96。元実装とのrouter1152サンプルも選択/重みの差0、複製した両rank間も一致。
- MTPのrouterは2/3/4/5行、255/1792/2048行prefillで両rank一致。最初は試験側の誤った1行必須条件でexit1だったが、ARで1行確認済みとして条件修正後の再試験は正常終了。
- K5/V4の実cacheはtarget12層＋draft1層。QSA raw_k/pooledはFP16、GDN recurrent stateは元のFP32/BF16のまま。8K cacheでtargetの実配列は100270080 bytes。
- 有効時batch2：各8K+256、固定draft2、target128/draft236 forwardのfinite検査。32K+256 JA/code：固定draft4、target211/draft708。いずれもRAM全ページ、cache bit、normal exit、power復帰を確認。
- CPU suite661 tests＋63 subtests、compile/diff検査合格。通常推論、clean benchmark、none/host timers、HIP event capturesは正常終了。
- ROCprofilerはCSV終了処理が2秒を超えたため診断だけjoin猶予120秒へ変更しAR captureに成功した。一方、詳細MTP captureは推論完了後のHSA終了処理でSIGSEGVとなったため採用せず、HIP eventsへ切替えた。driver/OSやproductionの終了待ちは変更していない。Torchが通常ロードするSDK依存と、外部`rocprofv3` captureは区別する。

## 再実行と記録

host側の検証済みrunner例（未使用tagを指定）。明示的な`--replicate-router`で選択する。native/modelは再ダウンロードしない。

```bash
python3 /home/homelab1/datapool/rocm-exl3-rdna2/runs/tp-decode-opt/run_bench.py \
  --tag q-tp-selected-retest --source /src \
  --model /work/models/qwen38-flash-next-exl3-3.05bpw \
  --execution tp --mode mtp --replicate-router \
  --prompts /work/runs/qwen38-mtp/formal8k-prompts.json
```

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/tp-decode-opt`。

- `ar-replicated-paired-comparison.json`, `mtp-replicated-paired-comparison.json`: 入力/native/モデル一致を検査したclean比較。
- `teacher-replicated-comparison.json`, `q-tp-mtp-router-agreement-r2*`: 数値/両rank routing。
- `q-tp-mtp-phase-wall-{ja,code}-{baseline,replicated}-phase-wall.json`: 軽いCPU wall分解。
- `q-tp-{ar,mtp}-event-spans-*-task-summary.json`: 排他的HIP stream区間、借用headのMTP分類、194/98通信検査。`event_span_capture.py`は診断用adapterでproduction modelを変更しない。
- `ar-kernel-task-decomposition.json`: 正常終了したAR kernel traceの層境界検査・分解。
- `qsa-head-split-probe.json`: QSAの単体分割試験。`q-tp-replicated-kv54-{batch2,32k}*`: 回帰試験。

各runにsource snapshot/manifest、実行command、実worker監査、power helperの復帰結果を保存。旧FP16 KVレポートを新K5/V4の証拠へ書き換えていない。
