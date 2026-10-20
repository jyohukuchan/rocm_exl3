# XeStrata調査からのEXL3最適化取り込み

2026-10-21。[調査](xestrata_exl3_optimization_review.md)で見つかった方式を、rocm_exl3の既存配置・演算に合わせて実装した。
XeのSYCL実装コードは転記していない。

## 動作の変更

- **Q2量子化修正**: K=2のcost配列を既存のglobal scratchへ置く。hostとkernelの両方を変更し、
  L=256で66,240 bytesを要求していたLDSを704 bytesにする。deviceのLDS上限も起動前に検査する。
- **CPUのN-major配置**: AVX2のCPU expertは、N tileごとにK tileを連続配置する。
  tile内のビット列、入力量子化、積和順序、Hadamard、biasは変更しない。
  AVX-512の既定は従来のband配置。scalar fallbackもN-majorを読み、
  明示的にN-majorを指定したAVX-512機ではAVX2ループ用の入力も準備する。
- **CPU/GPU間の配置管理**: arenaへの初回コピーとexpert swapは共通のpacking関数を使う。
  streaming prefillではGPU上でcanonical配置へ復元する。N-majorはK=8も対象。
  checkpointの保存形式は変更しない。
- **GPU比較経路**: 圧縮wordをwave内で共有するGEMVと、論理K分割を固定して物理wave数を変えるGEMVを追加。
  対象はmultirow GEMVと、その構造を使う通常のm=1経路。
  FP16復号、FragB配置、FP32の積和順序とreduction順序を維持する。
  既定の専用カーネルを別に保持し、比較設定を無効にしたときの実行コードへ影響させない。
- **複数GPU向けビルド修正**: `PYTORCH_ROCM_ARCH` のsemicolon区切りを受け付け、
  コンパイル時だけでなくdevice linkにも全targetを渡す。
  以前はfat objectをコンパイルしても、リンク時に見えているGPUだけが選ばれる問題があった。

GPU候補は既定では有効にしない。V620/R9700の測定でword共有の明確な改善がなく、
狭い出力では遅くなったため。物理wave数も大きくするほど速いわけではない。
最初の実装で発生した既定経路の約5%の低下は、比較用のruntime分岐・論理分割loopを
既定のカーネルから分離して解消した。2048→6144、4bit、m=1の旧拡張との比較は、
V620が29.347→29.348 µs、R9700が21.962→22.205 µs。後者の約1.1%の差は小さい。
両GPUで旧・新拡張の入力と出力のSHA256が一致した。

## 数値検証と実モデル

同じfat binaryをV620 (`gfx1030`) とR9700 (`gfx1201`) で実行した。
Torch 2.12.0+rocm7.2、HIP compilerはROCm core-7.14を使用。

| 検証 | 結果 |
| --- | --- |
| Q2、L=256/160とK=3/4の回帰 | 両GPUで4件ずつ成功。循環state、独立に計算したCB2復号値、再実行時の一致を確認 |
| CPU配置・GPUへの復元 | 両GPUで4件ずつ成功。3/5/8bit、1/3行、1/4 worker、projection offsetを確認 |
| word共有・物理wave数・graph再生 | 両GPUで8件ずつ成功。3/4/6/8bit、m=1/3、16論理分割に対する4/16物理waveでbit一致 |
| scalar CPU fallback | 4件成功。canonical/N-majorでbit一致 |
| 実モデルのCPU併用decode | Qwen3.5-35B-A3B、4.5bpw mul1、全40層の128/256 expertをCPUへ配置。入力1 tokenから8 tokenを正常生成 |

実モデルは24 worker、N-major、R9700、実際のKV容量4096 tokenで確認した。
smokeでの完走確認であり、end-to-end速度改善や教師KLの測定ではない。

64 tokenのprefillではcanonical/N-majorの両配置で完了待ちが続き、試験を停止した。
原因は確定しておらず、GPUへのexpert streamingと同期の切り分けは未完了。
streamingを外す比較も途中で停止したため、結果として扱わない。
このため**複数tokenのCPU/GPU併用prefillを検証済みとはしない**。
追加の実機検証は保留し、以降の作業は軽いコード・文書確認に限定した。

## 設定

詳細は [環境変数](env_vars.md) を参照。

| 設定 | 意味 |
| --- | --- |
| `EXL3_MOE_CPU_TILE_ORDER=auto` | 既定。AVX2=N-major、AVX-512=band、scalar=canonical |
| `EXL3_MOE_CPU_TILE_ORDER=0/1/2` | canonical / band / N-majorを明示指定 |
| `EXL3_MOE_CPU_SWIZZLE=0` | どのISAでもcanonicalに戻す |
| `EXL3_GEMV_WORD_TILE=1` | GPU圧縮word共有を有効化。既定0 |
| `EXL3_GEMV_PHYSICAL_WARPS=4/8/16` | 論理K分割を維持して物理wave数だけ指定。既定は従来値 |

CPU配置はworker起動前、GPU設定はgraph capture前に指定する。
GPUの選択値はkernel引数としてcaptureされ、後から環境変数を変えても既存graphの計算条件は変わらない。

## CPU単体の速度

3995WXのAVX2、合成trellis、H=2048、FF=512、5 expert×40層を巡回。
canonical/N-majorを同一processで交互に測り、出力はbit一致した。
以下は40層の巡回1回あたりの中央値。CPU affinityも両配置で揃えた。

| 条件 | canonical ms | N-major ms | 時間短縮 |
| --- | ---: | ---: | ---: |
| 24 worker、affinity指定なし、10巡×3測定 | 16.321 | 15.144 | 7.2% |
| 24 worker、CPU 0〜23、50巡×3測定 | 9.745 | 7.906 | 18.9% |
| 48 worker、CPU 0〜47、50巡×3測定 | 11.817 | 7.131 | 39.7%（ばらつき大） |

24 worker・affinity指定ありの測定範囲はcanonical 9.609〜9.831 ms、N-major 7.117〜8.110 ms。
48 workerはcanonical 9.121〜12.569 ms、N-major 6.338〜9.647 msと大きく変動した。
48 workerの改善率を代表値として推奨しない。hostは他の作業と共有されている。
これはCPU expert単体の結果であり、LLM全体のdecode速度の改善率ではない。
runtimeがworker数やCPU affinityを自動変更する機能は追加していない。

## 再現用の入口

拡張をビルドし、そのbuild-libを`PYTHONPATH`の先頭に置く。

```sh
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH='gfx1030;gfx1201' MAX_JOBS=8 \
    python setup.py build_ext --build-temp /path/to/build --build-lib /path/to/lib

PYTHONPATH=/path/to/lib:$PWD python -m pytest -q \
    tests/test_rocm_q2_quantize.py tests/test_moe_cpu_layout.py tests/test_rocm_gemv_word_tile.py

PYTHONPATH=/path/to/lib:$PWD taskset -c 0-23 \
    python -m rocm_tools.bench_moe_cpu_layout --threads 24 --iters 50

PYTHONPATH=/path/to/lib:$PWD python -m rocm_tools.bench_gemv_word_tile \
    --k 2048 --n 6144 --bits 4 --rows 1
```

GPU測定は同じ入力と複数の重みを使用し、captureしたgraphでPythonのenqueue費用を避ける。
6巡で走査順を反転し、各条件の全サンプルと中央値をJSONで出力する。
実モデル・MTP・CPU併用の採用判断では、対象モデルの固定token列でも比較する必要がある。

公開する測定要約と全サンプルは
[xestrata-backport.json](../benchmarks/2026-10-21/xestrata-backport.json) に保存した。
測定は他の作業とCPU/GPUを共有する環境で行っており、専有条件での再測定はしていない。

## 後で行う実機検証

- CPU/GPUを他の作業が使わない時間帯で、64 token prefillの完了待ちを切り分ける。
- 実モデルのexpert swapとstreaming prefillを含む長時間動作を確認する。
- 実モデルのCPU併用・MTPについて、同一token列と設定で速度を比較する。
- AVX-512、CUDA、Windowsでの実行は今回未確認。
