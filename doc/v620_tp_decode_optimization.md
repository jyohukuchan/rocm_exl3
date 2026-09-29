# V620×2 decode 分解・改善（進行中）

K5/V4 KVを以後の比較条件とする。既存配布パックのMTP3、Engramの単一RAM表、batch1のprefill auto / draft・verify・decode peak / idle autoを維持する。最適化の採用判定は未完了。

## 確認済みの基準

同じQwen3.8 Flash Next EXL3 3.05bpw、8K入力＋256生成、固定token IDs、日本語/コードそれぞれwarmup1＋測定5。native SHA256は`12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`、単一host HSA。プロファイラなしの中央値。

| 構成 | 日本語prefill | コードprefill | 日本語engine decode | コードengine decode |
|---|---:|---:|---:|---:|
| TP・AR | 464.62 | 460.36 | 31.31 | 31.34 |
| TP・MTP | 451.30 | 446.08 | 38.56 | 53.15 |
| Layer split・MTP | 295.95 | 288.41 | 38.26 | 53.56 |

単位はtok/s。engine decodeは既存の`time_generate`基準。配送・終了処理を含む観測decodeはTP AR29.20/26.74、TP MTP35.46/41.52、LS MTP34.20/37.99。生成途中64～128tokenの配送時刻から計算した中央値はTP AR31.64/31.63、TP MTP37.15/56.81、LS MTP40.07/58.32。MTPの境界は実際のburst終了token数を使う。MTP採用率（timed合算）はTP53.28%/78.14%、LS49.21%/79.14%。入力が同じでも出力や採用率が変わるので、MTP速度差を通信差だけと解釈しない。

本体12層とdraft1層の実cacheが`CacheLayer_qsa_quant`、K5/V4であることを検査済み。QSAのraw_k/pooledはFP16、GDN recurrent stateは元の型を維持。8K+64のJA/code MTP finite検査、本体61/draft167 forward、両rank計7076 module checks、Engram全ページRAM常駐、正常cleanupとauto復帰に合格。AR/MTP/LSの上表の全jobも正常終了。過去のFP16 KVによる32K/batch2検証をK5/V4の証拠には使わない。

## decode専用traceから分かったこと

`q-tp-ar-kv54-decode-probe-r2`はJAの最初のtimed jobの64→128token、64 forwardだけを捕捉。prefill/最終EOS処理を含まず、両rankのkernelは110656/107968件。正常終了・電力復帰を確認した。

各rankに12416 RCCL kernel、すなわち194 kernel/forwardがある。MoEは各層でexpert選択indicesと重みを2回broadcastし、48層で96回となる。最初の改善候補は小さいrouterを両GPUで同一計算して、この通知を省く方式。expert本体の分担と最後の集約は維持する。追加のVRAMと数値一致を確認してから性能を判定する。

RCCL kernelの時間にはpeer待ちが含まれる。両GPUのkernelを実行順で対応付けると所属layerは全件一致したが、片方のkernelが終了してから他方が実行する例もあり、重なる時間を「純粋なPCIe転送時間」とは呼べない。PCIeは両方Gen4 x16。以前の小payload all-reduceは10KB/40KBとも約60µsであり、帯域だけでなく起動・同期の固定費を調べる必要がある。

このtrace窓は約4.58秒で、通常実行の同区間約2.02秒より大幅に遅い。GPU kernelにも単発の大きな外れ値がある。従ってprofiled kernel時間の合計を通常decodeの寄与率へ単純換算しない。現在、none controlと詳細task/phase markersで追加確認中。GPU kernelはHIP runtime Correlation_IdからCPU rangeへ対応付け、非同期GPU実行時刻をCPU rangeへ単純に当てはめない。量子化GEMVのkernel時間を「dequantのみの時間」とも呼ばない。

初回traceは、CSV出力に約14秒かかるのにworker終了待ちが2秒で、出力中のSIGTERMにより正常終了しなかった。診断bootstrapだけ終了猶予120秒にしてr2は成功した。productionの終了待ちや推論処理は変更していない。

## 数値比較と未完項目

固定AR continuation256tokenをJA/codeで与え、各48位置・計96位置の全語彙logitsを保存した。元実装のrouter記録はrank1の48層・1152サンプル、rank0はrouterなし。候補では同じ継続tokenでlogitsを比較し、両rankのexpert選択/重みも照合する。これは速度測定ではない。

### Router複製の実機結果（2026-09-30）

`EXL3_TP_REPLICATE_ROUTER=1`をPython起動前に設定すると、std routerをrouted expertを持つ各rankへ複製し、expert選択indices/重みのbroadcastを省く。expertの分担と最後の集約は維持。既定は無効で、対象外routerは理由を表示して既存経路を使う。必要なrouterとdecode用transposeをVRAM計画へ計上し、routed shardが空のrankには不要なrouterをロードしない。

固定続きの96位置で全語彙logitsはbit単位まで一致し、1152件のrouterサンプルも元実装・両rank間で一致した。MTPでは2/3/4/5行および255/1792/2048行prefillのrouter結果が両rankで一致し、finite/RAM/cleanup検査も合格。最初の試験はMTPでも1行処理が出るとの誤った必須条件で終了コード1になったが、1行はAR試験で確認済み。条件を修正した再試験は正常終了した。

同じ最終コード・同じ入力で、無効化したARを再測定して比較した。各言語5回中央値。

| AR | 無効 decode | 有効 decode | 変化 | 無効 prefill | 有効 prefill |
|---|---:|---:|---:|---:|---:|
| 日本語 | 31.31 | 34.48 | +10.1% | 465.35 | 464.17 |
| コード | 31.32 | 34.54 | +10.3% | 460.39 | 459.02 |

decodeはengine基準、単位tok/s。生成途中の配送レートでも+10.4%/+10.1%、終了処理込みでは+10.2%/+8.9%。測定対象10件の出力token列はすべて一致し、前の基準とwarmupを含む12件でも一致。推論後の各workerのTorch allocatedを比較すると、増分はGPU0約238.27MiB、GPU1約1.78MiB、合計約240MiB。親プロセスだけのCUDA0=0という値をrank0のVRAMと取り違えない。

QSA head分割も単体で検討した。K5/V4、Q24/KV2から各GPU Q12/KV1へ分けた結果は、1行でbit一致、3/5行でrelative L2約0.00035。GPU kernel中央値は1行144.8→134.0µs、3行225.9→150.9µs、5行352.8→191.3µs（右は遅い方のhalf）。indexer/projection/TP集約を除く値で、単一host threadから両GPUを起動した実時間は約204µs/呼出しだった。通常1行decodeの利得は小さく、MTP検証には余地があるが、実モデルへのhead分割は現時点で採用していない。

未完: 詳細task/phase分解、MTPでの速度比較、最終設定・長文/batch回帰検証。ARの改善と数値一致は上記の範囲で確認済み。

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/tp-decode-opt`。
基準は`q-tp-ar-kv54-baseline`, `q-tp-mtp-kv54-baseline`, `q-ls-mtp-kv54-baseline`と各`-detailed-summary.json`。数値参照は`q-tp-kv54-teacher-baseline.json` / `.logits.pt`。固定sourceとmanifest、実行command、電力helperの復帰記録を同じdirectoryに保存している。
