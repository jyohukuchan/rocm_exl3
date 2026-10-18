# Qwen Image 2.1のEXL3実装と実行手順

2026年10月19日。開発中。両7Bモデルの品質・性能に関する完了条件は[目標](qwen_image_21_exl3_goal.md)で管理する。

## 実装済みの基盤

- `EXL3DiffusionLinear` はEXL3の重みを列方向に分割して復元し、入力がBF16なら重み側をBF16に変換してGEMMを実行する。入力活性値をFP16へ落とさない。復元用バッファは既定128 MiB以内で、全層の高精度重みを保持しない。
- `quantize_linear` は採取した入力からFP32のHessian近似を作り、既存のEXL3量子化器で変換する。空の校正や非有限値、無校正fallbackはエラーとして扱う。
- `ActivationCapture` は指定denoising stepの各層入力から一定数の行を採取する。CLIは初期・中間・終盤を採取し、校正付き実行を性能ベンチマークと区別する。
- checkpointは層単位のEXL3 safetensorsと高精度保持層に分けて保存する。変換は層単位で再開でき、完了したcheckpointだけを読み込める。読み込みでは元の高精度DiTを構築・ロードせず、meta parameterに高精度保持層を割り当てる。導出される非永続bufferは通常通り初期化する。
- CLIはBF16生成、text encodingの分離・再利用、校正採取、変換、EXL3読み込みに対応する。Turboのsampling sigmasは元のpipeline設定から保持する。

R9700で6件のテストが成功した。対象はFP16/BF16の分割復元、FP16範囲を超えるBF16入力、state保存・復元、実量子化とholdout入力、step別校正、小さなQwenImage21Transformerの7層変換・checkpoint再読み込み・forward。同じcheckpointの再読み込み後の出力一致も確認した。**小さなモデルのテストは、公式7B重みの画質を証明するものではない。**

公式通常版のBF16では1024平方・40stepの最初の生成が完走した。校正用プロンプトの青いティーポットと2個のオレンジを目視確認し、224個のLinearからstep 0・20・39で入力を採取した。形状は4096×4096、4096×12288、12288×4096。サンプルの最大絶対値は137。この実行は校正処理を含むため速度比較には使用しない。

## 環境

重みrevisionと依存の固定値は [sources manifest](../rocm_tools/qwen_image21_sources.json) に保存している。ROCm版PyTorchとこのリポジトリの拡張を先に用意し、Diffusionの追加依存をインストールする。

```bash
python -m pip install --no-build-isolation '.[diffusion]'
```

測定環境では既存のR9700用コンテナを使い、Diffusersだけを `/work/qwen-image21/python` に追加している。この環境では `PYTHONPATH=/work/qwen-image21/python:/work/lib-r9700-opt:/src` を指定する。`/work/lib-r9700-opt` を落とすと既存のROCm拡張が見えず、未対応のJITビルドへ進むため注意する。このパスはローカル環境固有であり、一般のインストール手順の一部ではない。

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

プロンプトを直接渡すEXL3推論では、DiTを常駐させて高精度text encoderだけをモジュール単位でCPU offloadする経路を用意している。BF16にも `--offload text-encoder` を指定すれば同じ配置方針で比較できる。この経路の公式7Bモデルでの検証は進行中。text encoderを丸ごとGPUへ転送する既定のBF16経路とは区別する。

Turboは `--variant turbo` と対応するモデル・校正・checkpointを使う。`--steps` は指定せず、保存された8-step scheduleを使う。通常版の校正やcheckpointの流用はsource検証で拒否する。

校正用3ケースと評価用6ケースは [固定ケース](../rocm_tools/qwen_image21_cases.json) に分けている。編集の評価ではBF16で生成した参照画像を両方式へ渡し、参照画像自体の違いを比較に混ぜない。

## 計測値の読み方

`report.json` はsampling設定、実step数、cache mode、各Linearの形状、GPU、依存version、生成時間とVRAMを保存する。

- `pipeline_seconds`：プロンプトを直接渡す場合はtext encoding・offload転送・VAEを含む。embeddingを渡す場合はtext encodingを含まない。初期ロードとPNG保存は含めない。
- `denoising_ms`：最初のTransformer呼び出し直前から最後のscheduler step直後までのGPUイベント時間。CPU offloadを使う場合、最初のDiT転送等もこの範囲に入り得る。
- `transformer_ms`：各呼び出しのイベント時間。最初のprefillと以降のcached呼び出しを区別できる。
- `calibration_instrumented=true`：CPUへのサンプル転送などを含む診断用の値。性能比較には使用しない。
- `peak_allocated_bytes` と `peak_reserved_bytes`：プロセスのPyTorch allocatorで計測した値。GPU全体の使用量とは異なる。

## テスト

GPU拡張と固定したDiffusersが利用できる環境で実行する。

```bash
python -m pytest -q tests/test_diffusion_exl3_linear.py tests/test_diffusion_exl3_checkpoint.py
```
