# Qwen Image 2.1のEXL3実装と実行手順

2026年10月19日。R9700単体で公式通常版・TurboのEXL3 4bit変換、保存・別プロセスでの再読み込み、生成・編集を検証し、[初期目標](qwen_image_21_exl3_goal.md)を完了した。これは各モデル6固定ケースの初期品質確認であり、全入力の品質保証ではない。

その後、**V620単体（gfx1030）でも通常版・Turboの公開5bit checkpointによる
1024×1024生成を確認した**。V620での検証範囲・実行方法・診断時間は
[V620検証記録](qwen_image21_v620.md)を参照。以下の速度・NMSEはR9700での測定値。

## 実機結果

各モデルの224個のblock Linearを4bit化し、block外の投影・modulation・norm等を高精度で保持した。DiT safetensorsはそれぞれ **3,766,566,208 bytes（約3.77 GB、BF16の26.5%）**。スケール・高精度保持層込みの実効精度は **4.235 bit/weight**。text encoderとVAEはBF16のままで、この保存容量には含まない。

1024平方、batch 1、CFG 1、prefix KV cache有効。両方式ともDiTをGPUに常駐させ、text encoderだけをモジュール単位でCPU offloadした。VAEは256pxタイル・192px stride。別のGPUジョブがない状態でwarmup 1回＋計測3回、時間は中央値。

| 指標 | 通常版 BF16 | 通常版 EXL3 | Turbo BF16 | Turbo EXL3 |
|---|---:|---:|---:|---:|
| denoising（秒） | 33.50 | 37.06（1.106倍） | 6.67 | 7.47（1.120倍） |
| 生成全体（秒） | 35.91 | 39.50 | 9.11 | 10.02 |
| ピークallocated（GiB） | 15.16 | 5.42 | 15.16 | 5.42 |
| ピークreserved（GiB） | 16.28 | 6.76 | 16.29 | 6.76 |
| denoising steps | 40 | 40 | 8 | 8 |

生成全体はtext encoding・転送・VAEを含み、初期モデルロードとPNG保存を含まない。allocatedは約64.3%減。保存容量40%以下、denoising 1.3倍以内、VRAM削減の目標を両モデルとも達成した。GPU全体のメモリ消費とは区別する。

[通常版の集計](../benchmarks/2026-10-19/qwen-image21-base-results.json)、[Turboの集計](../benchmarks/2026-10-19/qwen-image21-turbo-results.json)、[28件の生reportとチェックサム](../benchmarks/2026-10-19/qwen-image21-reports/index.json)を保存した。各モデル・精度でbenchmarkの4回の出力は画素hashが一致し、別プロセスで先に生成した同条件のキャッシュ済みembeddingの画像とも一致した。[再読み込み監査](../benchmarks/2026-10-19/qwen-image21-replay-audit.json)を参照。

両モデルとも写真風2048平方、文字、構図、単一参照編集、RGBA、2枚参照の6ケースが完走した。最終latentとVAE出力は全件finiteで、224層すべてが全denoising stepでEXL3として実行された。2048平方のEXL3ピークallocatedは約6.50 GiB（キャッシュ済みembedding使用、text encodingを含まない）。

[通常版の目視判定](../benchmarks/2026-10-19/qwen-image21-base-quality-review.json)と[Turboの目視判定](../benchmarks/2026-10-19/qwen-image21-turbo-quality-review.json)では、HELLOの文字、物体の左右配置、船体の緑への編集、狐の透過、複数参照の物体・色を確認した。RGBAの耳・脚・折り目、船の形状、複数参照の配置などは変わる。特にTurboの複数参照では背景が青い布から灰色に変わり、物体の大きさ・配置にも差がある。事前のケース別条件は満たすが、BF16との画素一致や細部・背景の保存を保証するものではない。

## 実装済みの基盤

- `EXL3DiffusionLinear` はEXL3の重みを列方向に分割して復元し、入力がBF16なら重み側をBF16に変換してGEMMを実行する。入力活性値をFP16へ落とさない。復元用バッファは既定256 MiB以内で、全層の高精度重みを保持しない。実モデルの最大Linearに必要なFP16とBF16の復元領域は計192 MiBで、この上限なら1回のGEMMで扱える。
- `quantize_linear` は採取した入力からFP32のHessian近似を作り、既存のEXL3量子化器で変換する。空の校正や非有限値、無校正fallbackはエラーとして扱う。
- `ActivationCapture` は指定denoising stepの各層入力から一定数の行を採取する。CLIは初期・中間・終盤を採取し、校正付き実行を性能ベンチマークと区別する。
- checkpointは層単位のEXL3 safetensorsと高精度保持層に分けて保存する。変換は層単位で再開でき、完了したcheckpointだけを読み込める。読み込みでは元の高精度DiTを構築・ロードせず、meta parameterに高精度保持層を割り当てる。導出される非永続bufferは通常通り初期化する。
- CLIはBF16生成、text encodingの分離・再利用、校正採取、変換、EXL3読み込みに対応する。Turboのsampling sigmasは元のpipeline設定から保持する。

R9700環境でGPUとCPUの[計10件のテストが成功した](../benchmarks/2026-10-19/qwen-image21-tests.txt)。対象はFP16/BF16の分割復元、FP16範囲を超えるBF16入力、state保存・復元、実量子化とholdout入力、step別校正、小さなQwenImage21Transformerの7層変換・checkpoint再読み込み・forward、実行再開時の設定検証、比較条件とアロケータ設定の検証。同じcheckpointの再読み込み後の出力一致も確認した。**小さなモデルのテストは、公式7B重みの画質を証明するものではない。**

校正は各モデル専用の3入力で行い、通常版はstep 0・20・39、Turboは0・4・7から採取した。評価入力は別に固定した。実Linear形状は4096×4096、4096×12288、12288×4096。層ごとの変換結果は[通常版](../benchmarks/2026-10-19/qwen-image21-base-conversion.json)と[Turbo](../benchmarks/2026-10-19/qwen-image21-turbo-conversion.json)に保存した。

## 環境

重みrevisionと依存の固定値は [sources manifest](../rocm_tools/qwen_image21_sources.json) に保存している。ROCm版PyTorchとこのリポジトリの拡張を先に用意し、Diffusionの追加依存をインストールする。

```bash
python -m pip install --no-build-isolation '.[diffusion]'
```

測定環境では既存のR9700用コンテナを使い、Diffusersだけを `/work/qwen-image21/python` に追加している。この環境では `PYTHONPATH=/work/qwen-image21/python:/work/lib-r9700-opt:/src` を指定する。`/work/lib-r9700-opt` を落とすと既存のROCm拡張が見えず、未対応のJITビルドへ進むため注意する。このパスはローカル環境固有であり、一般のインストール手順の一部ではない。

測定環境はPyTorch 2.12.0+rocm7.2、HIP compiler 7.14.60850、Diffusers commit `b800f828f3951c4d206e154411bcbfa693cd322e`。コンテナID、依存、Python実装hash、native拡張SHAは[環境記録](../benchmarks/2026-10-19/qwen-image21-environment.json)、全Python package versionは[一覧](../benchmarks/2026-10-19/qwen-image21-python-packages.json)に保存した。native拡張は既存の検証済みR9700用バイナリを再利用し、今回の機能でnativeコードを変更していない。そのバイナリのビルド元commitは不明なので、環境をゼロから完全に再構築した検証とは区別する。

## 公開EXL3 checkpoint

[通常版](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-EXL3)と[Turbo](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-Turbo-EXL3)を別リポジトリで公開し、それぞれ3・4・5bitを収録している。モデルカードにライセンス、推論方法、NMSE、比較画像を記載。公開用の通知メタデータを加えているが、評価済みcheckpointのテンソルpayloadは変更していない。[公開手順・検証記録](qwen_image21_hf/README.md)を参照。

## 重みの取得

リポジトリルートで実行する。既定の保存先は `~/datapool/ai_models/safetensors/`。指定revisionの取得が完了すると `exl3_source.json` が作られ、後続コマンドはこれを確認する。

```bash
python rocm_tools/qwen_image21.py download --variant base
python rocm_tools/qwen_image21.py download --variant turbo
```

以下の例では `MODEL`、`RUN` を実際のパスに置き換える。コンテナではモデルルートが `/models/safetensors`、作業先が `/work/qwen-image21`。ホスト上ではそれぞれdatapoolの対応ディレクトリを使う。

## BF16基準と校正

生成全体のBF16計測には、プロンプトを直接渡す。CLIはコンポーネント単位のCPU offloadとVAE tilingを使用する。`--warmup 1 --repeats 3` で初回を分離して3回計測できる。

```bash
python rocm_tools/qwen_image21.py generate --variant base \
  --model MODEL --prompt 'A blue ceramic teapot and two oranges on a wooden kitchen table, soft morning window light, realistic product photography.' \
  --output RUN/bf16 --warmup 1 --repeats 3
```

校正は独立した実行にする。`--capture RUN/calibration` を付け、warmup=0、repeats=1にする。複数プロンプトや編集条件で採取するときは、それぞれ別の校正ディレクトリに保存する。品質評価には別の入力を使う。

## 変換とEXL3推論

```bash
python rocm_tools/qwen_image21.py convert --variant base --model MODEL \
  --calibration RUN/calibration-01 RUN/calibration-02 \
  --bits 4 --output RUN/exl3-4bit
```

同じ出力先・校正・sourceで再実行すると、完了済みの層をチェックサムで検証して再開する。高精度に保持するのは現在、Transformer block外の投影・modulation・norm等。層ごとの追加精度調整は実モデルの画質確認に応じて実装する。

EXL3推論では、まず公式text encoderでembeddingを作る。テキストエンコーダとDiTを別プロセスで扱うため、DiTのGPU常駐条件で比較できる。

```bash
python rocm_tools/qwen_image21.py encode --variant base --model MODEL \
  --prompt 'A small lighthouse on a rocky coast at sunrise.' --output RUN/encoding
python rocm_tools/qwen_image21.py generate --variant base --model MODEL \
  --encoding RUN/encoding --exl3 RUN/exl3-4bit --output RUN/generated
```

同じ `--encoding` を使って `--exl3` なしで実行すると、BF16 DiTのGPU常駐比較になる。画像編集ではencodeとgenerateの両方に同じ `--images` を渡す。入力画像の内容とresize後の内容を検証し、異なる参照画像のembeddingを誤用しない。CFGを有効にする場合はencodeに `--negative-prompt`、generateに `--cfg` を指定する。

プロンプトを直接渡すEXL3推論では、DiTを常駐させて高精度text encoderだけをモジュール単位でCPU offloadする経路を用意している。BF16にも `--offload text-encoder` を指定すれば同じ配置方針で比較できる。この配置はTurboのBF16校正3ケースで動作確認済み。上表の全体計測もこの経路を用いた。text encoderを丸ごとGPUへ転送する既定のBF16経路とは区別する。

Turboは `--variant turbo` と対応するモデル・校正・checkpointを使う。`--steps` は指定せず、保存された8-step scheduleを使う。通常版の校正やcheckpointの流用はsource検証で拒否する。

校正用3ケースと評価用6ケースは [固定ケース](../rocm_tools/qwen_image21_cases.json) に分けている。編集の評価ではBF16で生成した参照画像を両方式へ渡し、参照画像自体の違いを比較に混ぜない。

## 固定ケースの実行と比較

`qwen_image21_suite.py` は各処理を別プロセスで実行し、出力とログを指定rootに保存する。再開時はsource、プロンプト、seed、解像度、参照画像、sampling設定、step数、KV cacheの使用を照合する。設定が異なる既存出力は上書きせずエラーにする。

```bash
python rocm_tools/qwen_image21_suite.py calibrate --variant base --model MODEL --root RUN
python rocm_tools/qwen_image21_suite.py convert --variant base --model MODEL --root RUN
python rocm_tools/qwen_image21_suite.py evaluate --variant base --model MODEL --root RUN
python rocm_tools/qwen_image21_suite.py benchmark --variant base --model MODEL --root RUN
python rocm_tools/qwen_image21_compare.py --variant base --model MODEL --root RUN --output RUN/base/comparison.json
```

Turboにも対応するモデルパスと `--variant turbo` で同じ手順を実行する。`evaluate` はプロンプトembeddingを共有したBF16/EXL3の両方式で6ケースを生成する。`--precision bf16` や `--case text` で段階的に実行できるが、最終比較には全ケースが必要。参照元のphotoとcompositionをBF16で生成してから編集ケースへ進む。

`benchmark` はcompositionを対象に、両方式をtext-encoder offloadでwarmup 1回＋計測3回実行する。開始時にGPUに別の大きな割当があれば停止する。校正・変換・他のGPUジョブが完了してから実行する。品質用の生成と、独立して測る性能ベンチマークは区別される。

生成CLIでは `EXL3_EXPANDABLE_SEGMENTS=0` を既定にし、EXL3のimportだけでアロケータ設定が変わることを防ぐ。明示したPyTorch allocator設定は保持し、EXL3の自動設定へ明示的にopt-inする場合はBF16側にも同じinitializerを適用する。設定はreportに保存する。既存の通常版BF16基準6件は純粋なDiffusers経路で既定のPyTorch allocatorを使っており、その等価設定を[監査記録](../benchmarks/2026-10-19/qwen-image21-allocator-audit.json)付きで事後追記した。画像・時間の値は変更していない。上表の独立した性能比較は両方式とも同じ現行CLIで新規に測定した。

比較CLIは出力PNGと記録したpixel hash、最終latent/VAEのfinite検証、224層の実EXL3呼び出しを照合し、重み容量・時間・VRAM・画像差分を集計する。白背景へ合成したRGB差とalpha差は診断値であり、画質の自動合格条件ではない。目視判定を渡さない場合の出力JSONはmanual reviewがpendingになるため、固定ケースのcriteriaに沿って画像を見て判断する。

目視後は `--reviews RUN/base-exl3-review.json` で判定を添付できる。reviewの各caseには `verdict`、`observation`、`bf16_pixels_sha256`、`exl3_pixels_sha256` を記載する。画像hashが違う判定は拒否し、未確認ケースが残ればpendingを維持する。全件確認後の `quality_screen_passed` は明示的な目視判定を反映する値で、自動の画質判定ではない。

## 3bit・5bitの実測結果

2026年10月19日。通常版・Turboそれぞれの3bitと5bit、計4モデルを作成した。各モデルの224個のblock Linearを指定bit数で変換し、高精度保持部分と校正データは対応する4bitと同じ。全checkpointの保存ファイルのチェックサムと、3・4・5bit間の高精度保持部分の同一性を確認した。

NMSEは「文字」「構図」「2枚参照」の3ケース（各1 seed、1702/1703/1706）、1024平方、batch 1、CFG 1、prefix KV cache有効で比較した。通常版は40step、Turboは公式保存済み8step。各モデルのBF16軌道上で全stepの対象画像トークンを比較し、3ケースの誤差二乗和／BF16出力二乗和を主スコアとする。4bitもこの同じ3ケースで集計し、以前の1ケース・6ケース集計とは区別する。

| モデル | 3bit NMSE ↓ | 4bit NMSE ↓ | 5bit NMSE ↓ |
|---|---:|---:|---:|
| 通常版 | 0.002586973 | 0.000790211 | 0.000211085 |
| Turbo | 0.003792089 | 0.001894632 | 0.000219289 |

各モデルの対応するBF16への誤差であり、通常版とTurboの生成品質を直接順位付けする値ではない。

| Linearのbit数 | DiT実効bpw | DiT safetensors容量（GB、十進） |
|---|---:|---:|
| 3 | 3.254020 | 2.894151 |
| 4 | 4.234933 | 3.766566 |
| 5 | 5.215846 | 4.638981 |

容量は通常版・Turboで共通。実効bpwにはスケールと高精度保持層を含む。text encoder・VAEはこの容量に含まない。

| モデル／ケース | 3bit NMSE ↓ | 4bit NMSE ↓ | 5bit NMSE ↓ |
|---|---:|---:|---:|
| 通常版／文字 | 0.001591202 | 0.000446119 | 0.000142797 |
| 通常版／構図 | 0.002085636 | 0.000618676 | 0.000181886 |
| 通常版／2枚参照 | 0.004672557 | 0.001508764 | 0.000347635 |
| Turbo／文字 | 0.001600494 | 0.000521166 | 0.000120647 |
| Turbo／構図 | 0.002201254 | 0.000521980 | 0.000143489 |
| Turbo／2枚参照 | 0.009668966 | 0.006106987 | 0.000489717 |

3・4・5bitで基準側の出力エネルギーとBF16画像hashが一致することを確認した。さらに新規の3bit・5bitモデルで各3ケース、計12枚を独立した生成軌道で出力し、最終latentとVAE出力が全件finite、全224層が全stepでEXL3として実行されたことを確認した。

2枚参照の目視所見：

- 通常版3bit：赤い船体・赤い球・青い立方体は識別できるが、参照の白い帆が失われ、黄色いピラミッドが船の背後へ移る。物体の大きさ・配置も大きく変化する。
- 通常版5bit：白い帆を含む赤い船、赤い球、青い立方体を維持する。BF16同様に黄色いピラミッドも残る。船の大きさ・配置には差があるが、3bitで見られた白い帆の欠落はない。
- Turbo3bit：白い帆と赤い船体、赤い球、青い立方体は保たれる。BF16の青い背景が灰色に変わり、球と立方体が大きくなって配置も変わる。立方体の右端は見切れる。BF16にある黄色いピラミッドは省略される（プロンプトでは要求していない）。
- Turbo5bit：白い帆と赤い船、赤い球、青い立方体を保持。3bit/4bitで灰色になった背景もBF16と同様の青い布として残り、黄色いピラミッドも見える。物体の大きさ・位置や船内形状には小差がある。

[全数値・条件・容量](../benchmarks/2026-10-19/qwen-image21-multibit-nmse.json)、[30件のNMSE／生成report](../benchmarks/2026-10-19/qwen-image21-multibit-reports/index.json)、[画像hash付きの目視記録](../benchmarks/2026-10-19/qwen-image21-multibit-review.json)を保存した。新しいbit数別パスとNMSE関連のCPUテスト7件も成功した。

checkpointはローカルの `~/datapool/rocm-exl3-rdna2/qwen-image21/{base,turbo}-exl3-{3,5}bit/` に保存した。今回の3bit・5bit評価範囲は上記3固定ケースであり、4bitの6ケース・2048px検証とは別の記録。3bit・5bitの専用速度ベンチマークは行っていない。

## bit数を指定した変換・評価

suiteと比較CLIは `--bits 3` / `--bits 5` 等を受け付ける。各モデルで既存の専用BF16校正を共用し、校正入力・層別seed・高精度保持箇所を変えずにbit数を比較できる。

```bash
python rocm_tools/qwen_image21_suite.py convert --variant base --bits 3 --model MODEL --root RUN
python rocm_tools/qwen_image21_suite.py nmse --variant base --bits 3 --model MODEL --root RUN \
  --case text --case composition --case multi_ref
python rocm_tools/qwen_image21_suite.py evaluate --variant base --bits 3 --model MODEL --root RUN \
  --precision exl3 --case text --case composition --case multi_ref
```

5bitは `--bits 5`、Turboは対応するMODELと `--variant turbo` を指定する。checkpointは `RUN/base-exl3-3bit/` 等、画像は `RUN/base/exl3-3bit/` 等、NMSEは `RUN/base/nmse/exl3-3bit/` 等へ保存される。BF16参照とembeddingは共有し、既存4bitの `exl3/` ディレクトリは維持する。`benchmark --bits N` と `qwen_image21_compare.py --bits N` も同じ配置に対応するが、比較CLIによる従来の全件集計には全6画像ケースと専用benchmarkが必要。NMSEの部分ケース集計はその実行の `summary.json` を使用する。

## 計測値の読み方

`report.json` はsampling設定、実step数、cache mode、各Linearの形状、GPU、依存version、生成時間とVRAMを保存する。

- `pipeline_seconds`：プロンプトを直接渡す場合はtext encoding・offload転送・VAEを含む。embeddingを渡す場合はtext encodingを含まない。初期ロードとPNG保存は含めない。
- `denoising_ms`：最初のTransformer呼び出し直前から最後のscheduler step直後までのGPUイベント時間。CPU offloadを使う場合、最初のDiT転送等もこの範囲に入り得る。
- `transformer_ms`：各呼び出しのイベント時間。最初のprefillと以降のcached呼び出しを区別できる。
- `calibration_instrumented=true`：CPUへのサンプル転送などを含む診断用の値。性能比較には使用しない。
- `measurement_purpose`：quality、benchmark、calibration、diagnosticを区別する。品質確認中に他の計算が並行していても、その時間を独立したbenchmarkへ混ぜない。
- `finite_final_latents`、`finite_vae_output`、`alpha`：非有限値は生成をエラーにし、出力画像のalpha分布を記録する。RGBAの完全一致を要求するものではない。
- `peak_allocated_bytes` と `peak_reserved_bytes`：プロセスのPyTorch allocatorで計測した値。GPU全体の使用量とは異なる。

## BF16軌道上の出力NMSE

量子化方式の数値的な忠実度は、BF16の生成軌道上でDiTの直接出力を比較する。BF16と候補モデルへ同じlatent・時刻・テキスト・参照画像条件を渡し、候補の予測はschedulerへ渡さない。prefix KV cacheは各モデルが同じ条件から独立に作るため、候補側のprefix計算に由来する誤差も含む。prefill時のテキスト・参照画像トークンを除外し、全stepの対象画像トークンだけを測る。

`NMSE = sum((candidate - BF16)^2) / sum(BF16^2)`。小さいほどBF16に近い。FP32で差と二乗を計算し、和はFP64で蓄積する。分母がゼロまたは非有限値が出た場合はエラーにする。step別・case別の値と分子・分母を保存し、suiteの主スコアは全case・step・対象要素にわたる分子と分母の総和の比とする。case別NMSEの単純平均ではなく、2048pxケースの要素数やBF16出力エネルギーも重みに影響する。通常版とTurboは別々に集計し、異なるschedule同士の値をそのまま量子化方式の優劣に使わない。

既存の評価用embedding・参照画像を使って実行する（先に `evaluate` のBF16生成を完了する）。初期対応はCFG=1、各case 1 seed、全step。校正入力と評価入力は別である。

```bash
python rocm_tools/qwen_image21_suite.py nmse --variant base --model MODEL --root RUN
python rocm_tools/qwen_image21_suite.py nmse --variant base --model MODEL --root RUN \
  --case composition --nmse-self
```

Turboは `--variant turbo` と対応モデルを指定する。任意のEXL3 checkpointは `--checkpoint PATH`、評価ケースの選択は `--case text` 等で指定できる。出力先は既定で `RUN/VARIANT/nmse/exl3/`、自己比較は `bf16-self/`。既存の実験と混ぜないよう、再実行時は新しい `--nmse-output PATH` を指定する。集計は `summary.json`、各caseの設定・source・実装hash・step別NMSEは `CASE/report.json` に保存する。`--nmse-self` は同じBF16モデルのforwardを独立したcacheで再実行し、計測経路の誤差下限を確認する。

単一生成CLIでも `generate --encoding ENCODING --nmse-exl3 CHECKPOINT` または `--nmse-self` を使用できる。NMSEは診断専用で、速度ベンチマーク・校正との同時計測、EXL3をteacherにする実行、CFG>1、複数repeatは拒否する。比較時はBF16と候補DiTを同時にGPUへ載せるため、通常のEXL3推論よりVRAMを使う。時間とVRAMはNMSE計測処理込みであり、通常推論の性能値として扱わない。

実機のBF16自己比較は通常版・TurboともcompositionケースでNMSE **0**。初回NMSEを測った7ケースすべてで、計測を挟んだBF16画像の画素hashは既存のBF16基準と一致した。

追加したNMSEの計算・target tokenの切り出し・独立cache・自己比較・非有限値拒否を含め、GPU/CPUの計12テストが成功した。NMSE reportを通常の品質・性能測定へ再利用することも拒否する。

2026年10月19日の初回EXL3 4bit実測では、1024平方のcompositionケース（seed 1703）のNMSEは通常版 **0.0006186758313913627**、Turbo **0.0005219796840151518**。Turboは2048平方のphoto、文字、composition、単一編集、RGBA、2枚参照の全6ケースも計測し、全要素・stepの総和による集計は **0.0013028038331149188**。この初回記録の通常版はcompositionの1ケースのみで、6ケース集計とは区別する。画像の品質確認済みケース数とNMSEの計測済みケース数も別である。[数値とケース別条件](../benchmarks/2026-10-19/qwen-image21-nmse.json)、[元reportの保存先とチェックサム](../benchmarks/2026-10-19/qwen-image21-nmse-reports/index.json)を参照。

現在の変換CLIは `--bits 1` ～ `--bits 8` の整数を指定できるが、公式両モデルについて、4bitは6固定ケース、3bit・5bitは上記3固定ケースで実測済み。固定suiteも `--bits` を受け付け、既定値は4。これは量子化対象Linearのbit数。スケール・高精度保持層込みの実効bpwは、3bitで約3.254、4bitで約4.235、5bitで約5.216となる。小数bpwを指定した層別自動配分はこの画像生成CLIにはない。

## VAE分割とRGBAの見た目

現在は公式VAEの既定値である256pxタイル・192px strideを両方式に使用する。分割復元は非分割復元の近似であり、境界付近の薄い筋がBF16画像にも見られる。512pxの同一latentを用いたVAE単体の比較でも境界に差が出た。これをEXL3の量子化誤差として扱わず、両方式でVAE設定をそろえる。メモリと画質の兼ね合いを変える大きなタイルや非分割復元は、この比較の計測条件には含めていない。

RGBAの透明部分にはRGB値が残る。RGBだけを表示すると背景色や境界差が強く見えるため、見た目はalphaを合成した画像とalphaマップで確認する。比較CLIのRGB差も白背景への合成後に計算する。

## テスト

GPU拡張と固定したDiffusersが利用できる環境で実行する。

```bash
python -m pytest -q tests/test_diffusion_exl3_linear.py tests/test_diffusion_exl3_checkpoint.py tests/test_qwen_image21_suite.py tests/test_qwen_image21_nmse.py
```
