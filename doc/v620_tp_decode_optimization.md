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

未完: 詳細task/phase分解、router複製候補の実機数値/速度比較、QSA等の片側処理をさらに2GPUへ分ける妥当性の評価、最終設定・回帰検証。改善を確認したとの主張はまだ行わない。

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/tp-decode-opt`。
基準は`q-tp-ar-kv54-baseline`, `q-tp-mtp-kv54-baseline`, `q-ls-mtp-kv54-baseline`と各`-detailed-summary.json`。数値参照は`q-tp-kv54-teacher-baseline.json` / `.logits.pt`。固定sourceとmanifest、実行command、電力helperの復帰記録を同じdirectoryに保存している。
