# Qwen Image 2.1 EXL3 on V620

## 検証条件

2026-10-19、単体の Radeon PRO V620 32GB（gfx1030）で公開済みEXL3
checkpointを検証した。R9700で変換した重みをそのまま使用し、V620用の
再量子化は行っていない。既存のV620×2 LLMサーバーは停止したまま、
1枚だけを公開する独立コンテナで実行した。画像生成の2 GPU分散は対象外。

PyTorch 2.12.0+rocm7.2、HIP compiler 7.14.60850、Transformers 5.17.0、
Diffusers commit `b800f828f3951c4d206e154411bcbfa693cd322e`。
V620用の既存ネイティブ拡張を再利用した。ビルド元commitは不明であり、
クリーンビルドの検証ではない。
[環境・拡張SHA256](../benchmarks/2026-10-19/qwen-image21-v620/environment.json)を参照。

EXL3 DiTはGPUに常駐させ、BF16 text encoderだけをモジュール単位で
CPU offloadする。VAEもBF16、256pxタイル・192px stride。プロンプトを直接
渡すため、各ケースでtext encoderから実行し、embeddingを使い回していない。
元のdense DiTを含まないコンポーネントディレクトリと公開checkpointを使用する。

## 生成結果

| モデル | Linear bits | ケース | steps | denoising秒 | 生成全体秒 | ピークallocated GiB |
|---|---:|---|---:|---:|---:|---:|
| 通常版 | 5 | composition | 40 | 411.38 | 470.33 | 10.03 |
| Turbo | 5 | composition | 8 | 82.78 | 137.97 | 10.03 |
| Turbo | 3 | composition | 8 | 82.77 | 99.73 | 8.41 |
| Turbo | 4 | 2枚参照のmulti_ref | 8 | 142.99 | 299.06 | 23.03 |

**4ケースとも完走し、推論コードの修正は不要だった。** 最終latentとVAE出力がfinite、
全224 EXL3層が全ステップで実行されたことをreportで検査した。
[集計とコマンド・生reportのSHA256](../benchmarks/2026-10-19/qwen-image21-v620/results.json)、
[画像hashに対応する目視記録](../benchmarks/2026-10-19/qwen-image21-v620/visual-review.json)を保存した。

2枚参照ではピークreservedが30.66 GiBで、テキストだけの場合より大きい。
通常版3・4bitのフルパイプライン、2048px、2 GPU分散は今回の確認範囲に含めない。
3・4・5bitの分割復元自体は後述の小規模テストで確認した。

通常版・Turboの5bit画像は、R9700と同様に赤い球が左、青い立方体が右、
中央に黄色い角錐という構図を保持した。表面や輪郭には細かな差があり、
VAEタイルの継ぎ目も見える。画素単位の一致ではない。
Turbo 3bitも同様の構図を保持した。4bitの2枚参照では白い帆の赤い船、
赤い球、青い立方体が出力され、R9700の4bit出力に近かった。
背景がBF16の青い布から灰色になる差はR9700でも既に観察している。

## 実行方法

GPUに対応するROCm PyTorch・toolchainを用意し、拡張のビルド先を
`gfx1030`に指定する。R9700用の`gfx1201`指定と区別する。
依存の固定値とcheckpointのダウンロードは
[モデルカードの手順](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-Turbo-EXL3#inference)
と[エンジンのビルド手順](reproduce_v620.md#2-build)を参照。

```bash
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1030 MAX_JOBS=8 \
  python -m pip install --no-build-isolation '.[diffusion]'
python -c 'import exllamav3_ext'

# 実際のV620のUUIDを指定する。MODELは公式のencoder/VAE等のディレクトリ。
ROCR_VISIBLE_DEVICES=GPU-YOUR-V620-UUID \
python rocm_tools/qwen_image21.py generate \
  --variant turbo --model MODEL --exl3 CHECKPOINT/5bit \
  --prompt 'A red sphere on the left and a blue cube on the right, sitting on a pale gray surface, a small yellow pyramid between them, clean studio rendering.' \
  --width 1024 --height 1024 --seed 1703 --output OUTPUT
```

通常版は`--variant base`と通常版のコンポーネント・checkpointを使う。
通常版は既定40ステップ、Turboは公式設定の8 sigmasを使用し、
`--steps`で変更しない。

今回のローカル環境では`PYTHONPATH=/work/qwen-image21/python:/work/lib:/src`。
`/work/lib`がV620用拡張で、R9700検証時の`/work/lib-r9700-opt`とは異なる。
新しい検証コンテナに不足していた`requests`を追加した。
通常の依存インストールではDiffusersの依存として導入される。

## 分割復元テスト

公開している3・4・5bitをFP16/BF16で確認した。
テストは128列ずつの復元と全列復元の一致、BF16でFP16の値域を超える入力、
保存・復元後の一致を確認する。

```bash
python -m pytest -q tests/test_diffusion_exl3_linear.py -k sliced_linear
```

[結果](../benchmarks/2026-10-19/qwen-image21-v620/linear-tests.txt): 6 passed。
この小規模テストと、実モデルの生成結果は区別する。

## 結果の読み方

実モデルの実行は1024×1024、batch 1、CFG 1、prefix KV cache有効。
各ケース1回、warmupなしの動作診断であり、中央値を求めた速度ベンチマークではない。
カーネルキャッシュの状態を揃えていないため、bit数による速度差の比較には使わない。
生成全体の時間はtext encoding・転送・denoising・VAEを含み、初期モデルロードと
PNG保存を除く。allocatedはPyTorchのピーク割り当て量で、GPU全体の使用量ではない。

V620とR9700で画素hashの一致は保証しない。公開済みのBF16-relative NMSEは
R9700で測定した値であり、この互換性検証でV620のNMSEを再測定したものではない。
GPUをまたぐ画素差を量子化NMSEとして扱わない。
