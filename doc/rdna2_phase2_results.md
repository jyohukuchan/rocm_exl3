# V620 / gfx1030 Phase 0–2 検証結果

実施日: 2026-09-28。対象は単一V620での起動・品質確認・速度改善。
実装は `opencode-delegate` 経由の `opencode-go/qwen3.8-flash` (OpenCode 2.0.12)、
レビュー、実機試験、コミットはCodexが担当。

## 環境と再現範囲

- 作業repo: `/path/to/rocm_exl3`。
- artifact root: `/path/to/rocm-exl3-data`。以下の `runs/` はこの配下。
- container: `rocm-exl3-rdna2`、repoを `/src`、artifact rootを `/work` にmount。
- GPU: V620、gfx1030、72 CU、約31.98 GiB。PCI `0000:43:00.0`、
  UUID `GPU-08b2ddcbd6e6b36c`、renderD128。GPU処理は直列実行。
  別V620の既存workloadとR9700は計算に使っていない。
- Torch `2.12.0+rocm7.2` (HIP `7.2.53211`)、Triton `3.7.0`、
  host HIP compiler/ROCr `7.14`、Python 3.12、Transformers `5.17.0`。
  これは検証済みの混在runtimeであり、任意のROCm環境での動作保証ではない。
- BF16参照: Transformers eager attention / native BF16 / CPU。
  8BについてはBF16単一V620生成も別途確認。30B BF16は約65GB RSSを使用。
- `environment.json`、`container-reproduction.json`、`python-environment-final.txt` に詳細。

| ID | checkpoint | 固定HF revision |
|---|---|---|
| D | turboderp/Qwen3-8B-exl3 / 4.0bpw | `1fd66d10f8fbdf071a0ff35842a2d1bf0df94b45` |
| D参照 | Qwen/Qwen3-8B / BF16 | `b968826d9c46dd6066d109eabc6255188de91218` |
| M | turboderp/Qwen3-30B-A3B-exl3 / 3.0bpw | `45c353c522dedaf4fb68be60423d611b5ff71e26` |
| M参照 | Qwen/Qwen3-30B-A3B / BF16 | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |

各model directoryの `source_manifest.json` にHFファイル情報を保存。
全24 safetensorsのサイズと全内容SHA256を固定revisionのHF LFS情報と照合済み。
結果は `runs/model-sha256-verification.json`。
公開量子化checkpointを利用し、量子化器のV620移植は行っていない。
実際のtrellis headerを調査すると、Dは4bit standard 252個＋6bit head 1個、
Mは3bit standard 18624個＋6bit head 1個だった。mul1 checkpointではないため、
将来用の3bit/8bit mul1はGEMV/GEMM合成試験でも別途検証した。
全tensor形状・bit幅・codebook inventoryは `runs/model-storage-inventory.json`。

## 実装

1. `2056e14`: gfx1030のbuild targetと64KiB LDS制限、WMMA相当の
   SIMT/fdot2 matrix primitiveを追加。118 sourceのfull build、import、
   FP16/BF16/int8 primitive、FP16連続累積、GEMV/GEMMを確認。
   `b5dbf96` で3bit/8bit mul1の合成GEMV/GEMMケースも追加し実機成功。
   prefillは既存forkの経路を維持できたため、計画時の全経路rocBLAS置換は不要だった。
2. `a39745f`: Qwen3-8BのFP16 MLP overflowを修正。
   Python corpusのlayer 2でgate/upは有限でも積が約9.9万になりInfが発生。
   up projectionのSVHを1/8、down projectionを8倍にする相殺スケーリングを
   deferred load完了後の初回forwardで適用する。clipはしない。
   HIP/gfx1030、対応するEXL3 MLPに限定し、bias/LoRA等の未検証構成は除外。
   `EXL3_ROCM_MLP_RANGE_BALANCE=0` で無効化可能。
   12 GPUチェックで反復、再ロード、72個の旧inner解放、全1024位置の有限性を確認。
3. `47e5458`: BC decode attentionのGQA tileを実際の4/8 headに合わせ、
   検証済みのpointerとsplit_lenにAOT `:16` alignment hintを付ける。
   HIP/gfx1030、FP16 KV、q_len=1、head_dim=128、q_heads=32、kv_heads=4/8、
   非QSA・非gate構成に限定。cache pointerが非整列なら既存設定へ戻す。
   内部bufferは登録前に整列を検査し、C++側のgridとprogram数が一致する。
   `EXL3_ROCM_GQA_TUNE=0` で無効化可能。
4. 固定token manifest、teacher-forced top-1比較、長さ/cache hit/有限性/終了を
   検査するbenchmark、各tokenのITL測定を `rocm_tools/rdna2/` に追加。

## 品質

英語・日本語・Pythonの8ケース、2009入力tokensから1024位置を選択。
source/EXL3で全ケースのtokenizer一致を確認。比較する位置・語彙151669を固定。
生成token列の完全一致やlogitsの厳密一致を合否条件にはしていない。

| 比較 | D: 8B 4bpw | M: 30B-A3B 3bpw |
|---|---:|---:|
| 最適化後decode経路 vs BF16 source | 970/1024 = 94.73% | 935/1024 = 91.31% |
| 最適化後 vs 同じEXL3重みの変更前decode経路 | 1021/1024 = 99.71% | 1024/1024 = 100% |
| 変更前bulk vs decode経路 | 1022/1024 = 99.80% | 1005/1024 = 98.14% |

BF16との差には量子化差が含まれる。移植・最適化による差とは分けて判断する。
受入目安はsource比D>=90%、M>=80%、同一EXL3最適化前後>=99%。
Mのbulk/decode差は低margin位置でtop-1が揺れるため、測定前に定めたKLD基準へ切替:

- `KL(P_bulk || P_chunk1)`、全151669語彙、1024位置、float64 log-softmax。
- mean **0.00165564 nats**、p99 **0.01485584**、max **0.33310937**。
- 事前基準mean<=0.01、p99<=0.05を満たす。全位置が小誤差という意味ではない。
- 不一致位置の参照top1/top2 logit margin中央値は0.015625。
- `runs/m-quality-kld-bulk-vs-chunk1.json` に位置別結果と生logitsのSHA256。

`runs/{d,m}-candidate-chunk1-optimized.json`、
`runs/{d,m}-optimized-vs-{baseline,bf16}.json` が通常経路の最終精度結果。
D/Mとも8K自然文混合入力＋256トークン生成の全256 forwardで有限logits、
cache hitなし、OOM/hangなし、通常終了を確認する独立試験を実施。
これは毎回同期する品質試験であり、速度値には使わない。

## 性能

速度はtok/s、中央値。主表は各モデルの最初の正式36-job測定をそのまま掲載する。

| モデル | 入力 | prefill 前→後 | decode 前→後 | decode倍率 | ITL p95後 (ms) | Torch peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| 8B 4bpw | 512 | 1123.5 → 1131.3 | 35.8 → 58.5 | 1.63× | 17.17 | 5.10 |
| 8B 4bpw | 2048 | 1147.5 → 1155.3 | 34.0 → 57.6 | 1.69× | 17.46 | 5.24 |
| 8B 4bpw | 8192 | 904.1 → 906.2 | 17.6 → 47.8 | 2.71× | 20.96 | 5.24 |
| 30B-A3B 3bpw | 512 | 473.6 → 471.5 | 52.3 → 70.6 | 1.35× | 14.19 | 11.74 |
| 30B-A3B 3bpw | 2048 | 974.7 → 967.9 | 50.1 → 69.2 | 1.38× | 14.50 | 11.82 |
| 30B-A3B 3bpw | 8192 | 743.6 → 742.5 | 29.8 → 56.9 | 1.91× | 17.61 | 11.89 |

prefillは全条件で変更前の約99.3–100.7%を維持。主対象Mの2K目標
（prefill>=200、decode>=20 tok/s）を満たす。全72 jobsが指定長・cache missを満たし通常終了。
Dは各group内5反復でTorch peak allocationが一定。Mも512/8Kは一定、
2Kはprefillで最大4.46 MiB、decodeで最大3.59 MiBの非単調な変動がある。
継続増加は観測していない。値はallocatorの割当量であり、
driver/contextを含むカード全体のVRAM使用量とは異なる。

最初の正式測定では短文decodeのspreadがD 512=6.8%、2K=5.2%、
M 512=5.5%、2K=6.7%となったため再測定した。8K decodeはD 4.7%、M 0.2%。
baselineにもD 8K prefill=12.8%、2K decode=5.3%のばらつきがあるため、
prefillの1%未満の差を改善とは主張しない。主表の反復を除外・差替えしていない。

同一入力列を再生して512/2Kを各5回追加測定した結果:

| モデル | decode512 | spread | decode2K | spread |
|---|---:|---:|---:|---:|
| D | 58.4 tok/s | 0.2% | 57.5 tok/s | 0.1% |
| M | 70.7 tok/s | 0.2% | 69.1 tok/s | 0.4% |

両モデルとも中央値の改善が再現した。`repeat_short_contexts.py` は省略した8Kの
乱数生成も消費し、元の正式測定と全入力SHA256が一致することを検査する。
cache容量も8704 tokensのまま。`runs/{d,m}-bench-short-repeat.json` と
`runs/short-repeat-audit.json` に追加48 jobsと通常終了の確認を保存。

追加測定ではDの2K prefillの1回だけTTFTが通常約1.80秒から3.24秒へ伸びた
（5回中央値1.80秒、spread45%）。この反復は除外していない。
元の正式5回はspread0.5%で、継続的なprefill退化は認めないが、単発遅延の原因は
特定できていない。CPU負荷/自動clockの記録は残し、温度や他processが原因と
断定しない。ここで報告する値はwarm中央値で、最大レイテンシ保証ではない。


条件: 単一V620、batch=1、FP16 KV、max_chunk_size=2048、
seed=1234、各group warmup 1＋timed 5回、decode出力256。
512/2048/8192の入力は各反復で別の乱数token列。
変更前後の対応する入力SHA256が一致し、全jobでprefix cache hit=0。
投機生成なし。prefillの独立jobは出力1、decode jobは出力256。
ITLは最初のtokenを除く実測interval（各run 255、各group 1275）を集約したp95。
TPOTはjob全体平均として別保存し、ITLと混同しない。

正式測定中のjunction温度最大はD 94℃、M 90℃、電力最大は253W/219W。
clock固定やdriver変更は行っていない。CPU process RSSの観測最大は
D 2.83 GiB、M 2.58 GiB。PSS snapshotはD約2.78 GiB、M約2.58 GiB、
swapは0。RSS/PSSはhost側使用量でありGPUの全割当量ではない。
`runs/{d,m}-bench-telemetry.jsonl` と `runs/{d,m}-ram-snapshot.json` に保存。

正式baselineは `runs/{d,m}-bench-final.json`（歴史的な名前で、最適化前）。
正式最適化後は `runs/{d,m}-bench-optimized.json`。
初期の約5 tok/sというcold/診断値を速度改善倍率の分母には使わない。
試作の8192入力/64出力/2反復では、Dはbase17.6、alignmentのみ42.8、
狭いtileのみ26.1、両方47.2 tok/sとなり、採用理由を個別A/Bで確認した。

## 検証と限界

- CPU回帰90件成功。長文attentionはkv_heads=8/4で各30条件成功
  （8K境界、16K、sliding windowを含む）。WMMA/GEMV/GEMM probeは独立参照と比較。
- 既定BC attentionの初回停止は別GPU probeの同時実行と重なっていた。
  単独warm再実行と新しいTriton cacheでのcold再実行は正常。
  `EXL3_BC_ATTN=0` やsync-before-loadは運用の必須条件ではない。
- fresh-cacheのD smokeはprefill約12.73秒、生成約32.49秒（計約45秒）を要した。
  load/JITを含む初回体感とwarm throughputは分けて扱う。
- PyTorch profilerはCPUイベントのみ、rocprofv3はAPI登録error16で失敗した。
  GPU eventがないことをGPU時間0とは解釈しない。計画の「上位3 GPU kernel」
  の時間割合は未取得で、CPU traceと独立kernel/E2E A/Bへ調査手法を変更した。
  詳細は `runs/profiling-limitations.json`。
  後続調査では読み込み順等を調整して [GPU時間内訳](exl3_timing_breakdown.md) を取得した。
  上記はPhase 2当時の制約を記録したもの。
- gfx1030全モデル、他RDNA2機種、全batch/cache精度への一般化は未検証。
  今回の成果はD/Mの単一V620経路。2GPU、Qwen3.8-Flash-Next、PLE RAM、TP2は
  [Phase 3以降](rdna2_port_plan.md)の対象。

## 再実行

既存containerは停止せず待機状態で残す。GPU workloadを1つずつ実行する。

```bash
cd /path/to/rocm_exl3
python3 -m unittest discover -s rocm_tools/rdna2/tests

docker exec -e MAX_JOBS=12 rocm-exl3-rdna2 \
  python setup.py build_ext --build-lib /work/lib --build-temp /work/build

docker exec rocm-exl3-rdna2 timeout -k 5 900 python -u \
  rocm_tools/rdna2/bench.py -m /work/models/qwen3-30b-a3b-exl3-3bpw \
  --contexts 512 2048 8192 --new-tokens 256 --warmup 1 --repeats 5 \
  --max-chunk-size 2048 --json-out /work/runs/m-bench-repeat.json

docker exec rocm-exl3-rdna2 timeout -k 5 480 python -u \
  rocm_tools/rdna2/collect_top1.py --manifest /work/runs/m-manifest.json \
  --backend exl3 -m /work/models/qwen3-30b-a3b-exl3-3bpw \
  --execution chunked --chunk-size 1 -o /work/runs/m-quality-repeat.json
```

比較CLI、参照生成、環境変数の詳細は [harness README](../rocm_tools/rdna2/README.md)。
生logit取得の補助scriptはartifact rootの `capture_quality_logits.py`、
`compare_full_kld.py`、長文有限性試験は `validate_long_generation.py`。
OpenCodeのbrief/result/eventsは `runs/01c-gfx1030`〜`runs/06-attention-tune` に保存。
新規containerを同じホストで再構成する場合は、`container-reproduction.json` の
base image ID、mount、device、環境変数を再現し、
`python-environment-final.txt` のpackage versionを使う。
追加した主要packageはaccelerate 1.15.0、psutil 7.2.2、setuptools 81.0.0、
container内git 2.43。`/work/lib` はmount先なのでimageだけでは拡張を含まない。
`runs/build-identity-final.json` がsource/binaryの対応を記録する。

CLI timeoutだけではdaemon taskは停止しないので、`opencode api session.active` と
`session.interrupt` で実際の状態を確認する。
