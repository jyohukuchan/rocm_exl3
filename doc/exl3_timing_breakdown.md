# EXL3の処理時間内訳: V620 / R9700

計測: 2026-09-28〜29（JST）。48計測区間、計1536 decode tokensを監査。
対象は前回の [R9700/V620比較](r9700_vs_v620.md) と同じ2モデル、同じ入力token列。
計測ツールの実装はOpenCode Go/qwen3.8-flashへ委任し、Codexがレビュー修正・GPU実行・監査。
推論の高速化を目的とするkernel変更は加えていない。

## 分類と単位

- **量子化Linear関連（dequant_related）**: EXL3重み復元、Hadamard、および復号と積和が融合した
  GEMV/GEMM/MoE kernel。量子化Q/K/V/out projectionもここに含め、Attentionとは二重計上しない。
  **復号だけの時間ではない**。融合kernel内の復号・積和・変換は、このtraceから分離できない。
- **Attention関連**: QK・softmax・AVのpaged attention、split/combine、KV更新、RoPE/QK norm。
- **その他GPU**: 通常のGEMM、routing、block norm、activation、sampling、GPU copy/fill、
  sort/histogram、cooperative runtime supportなど。prefillの通常GEMMはここに入り、
  「その他」がそのまま無駄・overheadを意味するわけではない。
- 各値は**GPU kernel start/end時刻による実行時間の合計、3回の中央値**。
  CPU時間や待ち時間を混ぜない。decodeはms/token、prefillは入力全体のms/prompt。
  独立に計算した中央値なので列の和と合計時間の中央値は厳密一致するとは限らない。

## Decode: ms/token

| モデル | GPU | 入力 | 量子化Linear関連 | Attention関連 | その他GPU |
|---|---|---:|---:|---:|---:|
| Qwen3-8B EXL3 4bpw | V620 | 2048 | 12.33 | 2.26 | 0.40 |
| Qwen3-8B EXL3 4bpw | R9700 | 2048 | 10.02 | 1.47 | 0.37 |
| Qwen3-8B EXL3 4bpw | V620 | 8192 | 12.42 | 5.75 | 0.41 |
| Qwen3-8B EXL3 4bpw | R9700 | 8192 | 10.05 | 3.37 | 0.37 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 2048 | 7.08 | 2.51 | 1.51 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 2048 | 5.95 | 1.43 | 1.16 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 8192 | 7.12 | 5.60 | 1.53 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 8192 | 5.94 | 2.72 | 1.13 |

生成97〜160番目の64 tokensを1区間とし、各条件3回。
その時点の平均KV長は入力長+約128で、前回の256-token decodeの中央付近に相当する。

8KでのGPU時間比（V620時間 / R9700時間）:

| モデル | 量子化Linear関連 | Attention関連 | その他GPU |
|---|---:|---:|---:|
| Qwen3-8B EXL3 4bpw | 1.24× | 1.70× | 1.11× |
| Qwen3-30B-A3B EXL3 3bpw | 1.20× | 2.06× | 1.35× |

GPU側の帯域比以上の差は主にAttentionに出ている。量子化Linear関連は約1.20〜1.24倍で、
公称帯域比1.25倍に近い。一方Attentionは約1.71〜2.06倍。
2K→8Kで量子化Linear時間はほぼ一定、増分は主にAttentionとなる。
これをV620の最適化不足や、改善可能量そのものとは解釈しない。

### Decodeの量子化Linear関連をさらに分ける（8K、ms/token）

| モデル | GPU | 復号＋積和などの融合kernel | 単独Hadamard kernel | 単独重み復元kernel |
|---|---|---:|---:|---:|
| Qwen3-8B EXL3 4bpw | V620 | 11.967 | 0.455 | 0.000 |
| Qwen3-8B EXL3 4bpw | R9700 | 9.646 | 0.404 | 0.000 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 6.455 | 0.668 | 0.000 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 5.343 | 0.599 | 0.000 |

単独復元kernelが0でも、復号が無料という意味ではない。decodeでは復号がGEMVに融合している。
Hadamardも融合kernel内に含まれる部分があるため、上の単独Hadamard列は全変換時間ではない。
MoEのその他GPUではroutingが大きく、8KでV620約0.920ms/token、R9700約0.592ms/tokenだった。

## Prefill: ms/prompt

| モデル | GPU | 入力 | 量子化Linear関連 | Attention関連 | その他GPU |
|---|---|---:|---:|---:|---:|
| Qwen3-8B EXL3 4bpw | V620 | 2048 | 59.22 | 219.54 | 1504.85 |
| Qwen3-8B EXL3 4bpw | R9700 | 2048 | 49.77 | 31.41 | 251.83 |
| Qwen3-8B EXL3 4bpw | V620 | 8192 | 200.00 | 2862.47 | 6151.35 |
| Qwen3-8B EXL3 4bpw | R9700 | 8192 | 167.13 | 362.42 | 1014.44 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 2048 | 536.34 | 275.77 | 996.13 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 2048 | 289.91 | 33.86 | 638.34 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 8192 | 2063.01 | 3676.84 | 3956.02 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 8192 | 1102.24 | 405.54 | 2513.46 |

prefillは入力投入から最初のtokenまでを記録。Qwen3-8Bの8Kでは、通常GEMM等を含む
その他GPUがV620約6151ms、R9700約1014msと大きい。Attentionも2862→362msとなる。

MoE prefillのR9700には既存reconstruct+hgemm回避策を使っているため、V620と融合の境界が違う。
V620の融合quant GEMMに含まれる演算が、R9700では復元kernelと通常GEMMへ分かれる場合がある。
**MoE prefillの量子化Linear列だけを「純粋なdequant性能比」と比較しない。**

## CPU・待ち・計測負荷を混同しないための対照

同一harness・同一prompt・同一生成区間を、profilerなしでも実行した。
GPU内のkernel合計と、end-to-end wall時間は異なる。下表はms/token。

| モデル | GPU | 入力 | profilerありのwall | profilerなしのwall | wall増加 | 計測中のkernel外区間 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B EXL3 4bpw | V620 | 2048 | 19.39 | 17.44 | 11.2% | 4.41 |
| Qwen3-8B EXL3 4bpw | V620 | 8192 | 23.01 | 20.95 | 9.8% | 4.45 |
| Qwen3-8B EXL3 4bpw | R9700 | 2048 | 15.52 | 13.57 | 14.4% | 3.65 |
| Qwen3-8B EXL3 4bpw | R9700 | 8192 | 17.43 | 15.52 | 12.3% | 3.66 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 2048 | 17.83 | 14.39 | 23.9% | 6.73 |
| Qwen3-30B-A3B EXL3 3bpw | V620 | 8192 | 20.74 | 17.53 | 18.3% | 6.50 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 2048 | 14.64 | 10.25 | 42.9% | 6.08 |
| Qwen3-30B-A3B EXL3 3bpw | R9700 | 8192 | 15.34 | 11.54 | 33.0% | 5.55 |

- profilerによりdecodeのwall時間は約10〜43%増えた。GPU kernel時間表は**計測下の値**であり、
  profilerありのwallを通常速度として採用しない。
- kernel外区間は各windowの `marker wall − union(GPU kernel intervals)`。
  CPUの投入・待ち・profiler負荷に加え、未記録のDMA/転送も入りうる。純粋なCPU計算時間でも、
  全GPUがidleだった時間でもない。memory-copy traceは無効で、SDMA時間は未計測。
- GPU copy kernel (`__amd_rocclr_copyBuffer`) は実際のGPU kernelなのでその他GPUに含む。
- 重なったkernel区間はGPU busy unionで一度だけ数え、kernel時間の単純和との差も保存。
  この表の各列の中央値を強制的に加算してwallに合わせてはいない。

## 入力・記録の監査

- FP16 KV、batch1、chunk2048、cache8704、各contextのwarmup1、timed3。
- 前回benchの36行の乱数生成順を全て再生し、入力SHA256を全行照合した。
  同じmodel/GPU比較で同じpromptを使い、prefix cache hit=0。
- 4条件×2context×3反復×2stage = **48閉区間**。全区間にGPU kernel記録があり、
  重複・負の時刻・欠落・区間境界をまたぐkernelはなし。markerとharness wallの差は0.1%未満。
- decodeのkernel数は8Bが660/token、30B-A3Bが1068/token。各区間で64token分を確認。
  Attention split/combine/KV-updateは36層または48層×64回で、記録落ちの検査にも利用。
- 各GPU内のprofile/controlでは**全24 timed jobsの生成token列が一致**。
  GPU間は同じ入力だが生成列はD5/6、M4/6ケース一致。厳密な生成列一致を品質基準にはしないが、
  MoE expert選択やcache局所性まで完全同一と断定しない。相違ケースも除外していない。
- 分類はkernel名からsourceの役割を確認し、全ての名前を3区分へ割当。
  復号と積和が融合している部分を推測で分割していない。

## プロファイラの制約と回避方法

前回のGPU trace欠落/SDK error16に対し、SDKを先にpreloadし、TritonのLLVM symbolを
一時的なRTLD_DEEPBINDで読み込むことでGPU dispatch timestampを取得した。
Torch/HIP/HSAや推論kernelを別versionへ更新していない。

基本起動は `LD_PRELOAD=librocprofiler-sdk.so:libhsa-runtime64.so`、
`rocprofv3 --kernel-trace --marker-trace --output-format csv`。
bootstrapはTritonを先に読み込んでから、前回と同じR9700の比較adapter等を呼ぶ。
profilerのpause/resumeとROCtx範囲でload/JIT/warmupを除き、区間境界のみ同期する。

Dense両GPUとR9700 MoEのprofile、および全controlは通常終了した。
**V620 MoEだけは、処理とモデル解放後のHSAキュー破棄でSIGSEGVが残る。**
gdbでは `RuntimeTearDown → hsa_shut_down → GpuAgent/AqlQueue destructor →
hsa_signal_store_screlease` を確認した。初期のfull runはCSV保存前に落ちたため不採用。

採用したV620 MoE runは、全ROIとmodel.unloadを完了しJSONを保存した時点で待機させ、
SIGTERMでprofilerに保存を要求した。SDKは保存完了後もsignal handler内に残るため、
**`tool finalization` 完了ログを確認してから当該processをSIGKILLで停止（exit137）**した。
これは正常終了ではなく、終了不具合が直ったとは扱わない。記録済み48区間の監査、
生成列/control一致、checksum確認を経たデータのみを上の表に使っている。
停止操作は全計測区間の外で行い、未完了runを成功扱いするための `os._exit` は使わない。

## ファイルと再現

artifact root: `/path/to/rocm-exl3-data`。

- `runs/timing-breakdown/final-breakdown.json`: 全値、監査、CSV/JSON SHA256。
- `runs/timing-breakdown/{d,m}-{v620,r9700}-summary.json`: window別/全kernel別内訳。
- `runs/timing-breakdown/*-control.json`: observerなし対照。
- `runs/timing-breakdown/m-v620-profile-flushed{.json,-trace/,-process-status.json}`: 明示停止した採用run。
- `runs/timing-breakdown/process-status.json`: 最初のmatrixの実exit code。失敗runも保持。
- `runs/timing-breakdown/pilot-m-gdb.log`: 終了時のbacktrace。
- `rocm_tools/rdna2/profile_stages.py`: 入力再生とstage計測。--no-roctxで対照。
- `rocm_tools/rdna2/summarize_rocprof.py`: CPUのみでCSVを分類・区間監査・集計。
- 外部bootstrap/adapterの源泉はartifact root内。`run_timing_breakdown.py` が通常matrix、
  `run_profile_flush.py` がV620 MoE用の明示停止。後者は古いready PIDの再使用を拒否するため、
  再実行時は出力名を新しくする。既存データを上書きして再利用しない。

CPU回帰136件成功。速度最適化ではなく計測用ツールを追加し、結果を記録した。
