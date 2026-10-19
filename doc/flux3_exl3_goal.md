# FLUX.3 Image Klein EXL3対応の完了条件

対象はユーザー指定の`black-forest-labs/flux-3-image-klein`。
Qwen Imageと同様に、このプロジェクトから量子化・保存・再読み込み・画像生成・
編集・精度比較を実行できる状態を目指す。重みの新規公開は今回の依頼範囲に含めない。
元モデルがRGB出力のため、QwenのRGBAケースは適用しない。

## 完了時に必要な証拠

- 固定したモデルrevision・Diffusers revisionと使用環境の記録。
- FLUXのimage・reference・text・joint各blockを対象にした校正とEXL3変換。
  3・4・5bitの実モデルを作り、別プロセスで再読み込みする。
- 公式の16ステップflow-mapスケジュール、`target_timestep`、text encoderの
  8 hidden layers・80トークン単位padding、FP32 VAEを保持する。
- 短い要求には公式PUPを使い、外部LLMで展開済みのcaptionも受け付ける。
  テキストと画像参照のPUP経路を実行確認する。
- R9700でBF16とEXL3の1024pxテキスト生成、文字、構図、単一参照編集、
  2枚参照、2048px生成を実行・目視確認する。
- 固定入力で3・4・5bitのBF16-relative予測NMSEを数値として記録する。
  BF16の同じlatent・時刻・条件で候補を呼び、候補からschedulerを更新しない。
  BF16自己比較で数値的な下限も確認する。
- BF16/EXL3の同条件・warmupありの生成速度とVRAMを比較する。
- V620単体でEXL3のテキスト生成と画像参照を確認し、必要な互換性修正を行う。
  V620×2の既存LLMサーバーは停止したままにする。
- 小規模FLUXグラフの変換・再読み込み・16ステップ生成テスト、およびQwenの
  関連テストが通る。実モデル検証と小規模テストを区別する。
- 実行手順、対応範囲、実測値、制約を文書化し、commit・pushする。

## 完了監査（2026-10-19）

指定したdistilled checkpointのEXL3対応と実機検証を完了した。
実行方法・測定条件・数値は[結果とCLI](flux3_exl3.md)に記載した。

| 完了条件 | 確認した証拠 |
|---|---|
| source・環境固定 | [モデル・PUP・Diffusers pins](../rocm_tools/flux3_sources.json)、[R9700環境とnative SHA](../benchmarks/2026-10-19/flux3/r9700-environment.json)、[V620環境](../benchmarks/2026-10-19/flux3/v620-environment.json) |
| 校正・3/4/5bit変換 | 各258層、同一校正manifestと同一の高精度保持state。777個のsafetensorsをchecksum検査した。[全モデル・report監査](../benchmarks/2026-10-19/flux3/reports/index.json)、[校正入力hash](../benchmarks/2026-10-19/flux3-initial/calibration-audit.json) |
| 実ファイルの別プロセス読み込み | 精度ごとの生成・NMSE・benchmarkを独立したCLIプロセスで実行。全層checksumと構成を読み込み時に検証し、期待するEXL3実行回数を確認した。生reportは上記indexの各archiveに保存 |
| 公式のsampling recipe | 全44画像実行で16ステップ、shift 6.93の現在時刻とtarget時刻列を確認。8 hidden layers・80単位paddingのtext encoder、FP32 VAEを共用 |
| PUPと展開済みcaption | V620の[テキストPUP](../benchmarks/2026-10-19/flux3-initial/pup-v620-text.json)、[参照画像PUP](../benchmarks/2026-10-19/flux3-initial/pup-v620-edit.json)、[PUPから生成まで](../benchmarks/2026-10-19/flux3-initial/v620-4bit-pup-full.json.gz)。R9700の[公式PUP経由の編集](../benchmarks/2026-10-19/flux3/pup-validation.json)も完走・目視確認。固定captionの経路は全比較に使用 |
| R9700品質確認 | BF16と4bitを各6ケース（2048px含む）、3bitと5bitを各3ケースの1024pxで確認。[4bitの目視記録](../benchmarks/2026-10-19/flux3/4bit-visual-review.json)、[3・5bitの目視記録](../benchmarks/2026-10-19/flux3/multibit-visual-review.json) |
| NMSE | 3bit 0.006519001098、4bit 0.001697549296、5bit 0.000467956625。各bitで3ケース、全target要素・全ステップの総和比。teacherのlatent・予測・conditioning hashが一致し、BF16自己比較は0。[比較JSON](../benchmarks/2026-10-19/flux3/comparison.json) |
| 速度・VRAM | 同条件でwarmup 1＋計測3の中央値。BF16と3/4/5bit、計4reportを照合し、全精度で計測中の画像hashが一致。[比較表](../benchmarks/2026-10-19/flux3/comparison.md) |
| V620 | 3/4/5bitの生成、4bitの単一参照・2枚参照を実行。PUPの未対応FLA BF16命令をTorch referenceへ切替。query分割でAttention一時領域を抑えた。[初期実機記録](../benchmarks/2026-10-19/flux3-initial/) |
| 小規模・回帰テスト | [R9700 5件](../benchmarks/2026-10-19/flux3/r9700-tests.txt)、[V620 3件](../benchmarks/2026-10-19/flux3/v620-tests.txt)が成功。Fluxの条件分岐校正・変換・再読み込み・16ステップ生成・NMSE、query分割の比較、既存Qwenのcheckpointとcaptureを含む |
| リソースと公開範囲 | V620のLLMサービスと検証用コンテナは停止済み。コード・結果・手順をcommit/pushし、モデル重みの新規Hub公開は行っていない |

これは固定ケースによる機能・精度確認であり、全入力の画質保証ではない。
V620の2048px、3/5bitの参照画像、複数GPUの画像推論、CUDAは今回の実機確認に含めない。
既存のnative拡張を再利用したため、クリーンビルドの再現確認とも区別する。
