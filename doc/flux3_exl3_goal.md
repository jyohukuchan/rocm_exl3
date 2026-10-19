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

## 現在の証拠

小規模FLUXとQwenの関連テストが成功した。
[checkpointテスト](../benchmarks/2026-10-19/flux3-initial/checkpoint-tests.txt)、
[16ステップ生成・NMSEテスト](../benchmarks/2026-10-19/flux3-initial/flow-tests.txt)。

R9700で3件の実BF16校正生成を確認し、text-only 228層とreferenceあり258層の入力を採取した。
[校正入力ファイルのSHA256とreport一覧](../benchmarks/2026-10-19/flux3-initial/calibration-audit.json)。
画像では青いティーポットと2個のオレンジ、そのティーポットだけを緑へ変える編集を目視確認した。

V620のPUPでFLAのBF16 dot命令がLLVMコンパイルに失敗したため、該当4関数を
TransformersのPyTorch referenceへ切り替えた。
[テキスト入力](../benchmarks/2026-10-19/flux3-initial/pup-v620-text.json)と
[画像参照入力](../benchmarks/2026-10-19/flux3-initial/pup-v620-edit.json)が完走した。

実モデルのEXL3変換・評価、NMSE、速度・VRAM比較とV620のEXL3生成は未完了。
この目標は継続中で、小規模テストやBF16生成だけでは完了とはしない。
