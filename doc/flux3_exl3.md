# FLUX.3 Image Klein EXL3

対象は[black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)の
distilledモデル。[固定revision](../rocm_tools/flux3_sources.json)を使用する。
16ステップのflow-map samplingで、現在時刻に加えて移動先の`target_timestep`を渡す。
text encoderはQwen3-VL-4Bの8 hidden layersを結合し、80トークン単位で右paddingする。
DiTとtext encoderはBF16、VAEはFP32。出力はRGB。

**実モデルのEXL3検証は進行中。** 現在、小規模FLUXグラフの24層EXL3変換・
保存・再読み込み・16ステップ生成・NMSE測定に加え、R9700とV620で公式モデルを
量子化した4bit checkpointの1024px生成を確認した。
V620では単一参照編集、公式PUPからEXL3生成までの経路も確認した。
3・5bit、NMSEと制御した速度比較、残りの品質ケースは検証中。
完了条件は[作業目標](flux3_exl3_goal.md)を参照。

## 環境と取得

ROCm版PyTorchと、このGPU向けのネイティブ拡張を先に用意する。
今回の実行環境はQwen Imageと共通で、Diffusersはcommit
`b800f828f3951c4d206e154411bcbfa693cd322e`、Transformers 5.17.0。
R9700は`PYTORCH_ROCM_ARCH=gfx1201`、V620は`gfx1030`を指定して拡張をビルドする。

```bash
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=8 \
  python -m pip install --no-build-isolation '.[diffusion]'
python -c 'import exllamav3_ext'

python rocm_tools/flux3.py download --output MODEL
python rocm_tools/flux3.py download --component upsampler --output PUP
```

`MODEL`と`PUP`は実際の保存先に置き換える。モデルのHubアクセス権限は事前に必要。
ダウンロードは固定revisionを使い、完了時に`exl3_source.json`を保存する。
EXL3生成だけを行う環境では`download --components-only`で元のdense DiTを省略できる。
校正・BF16基準・NMSEにはdense DiTが必要。

モデルの利用条件は取得した`LICENSE.md`・`NOTICE.md`と上流のモデルカードに従う。
エンジンのMITライセンスを重みに適用しない。

## 生成とプロンプト展開

短い要求は公式PUPで展開する。CLIはPUPをGPUから解放してからDiTを読み込み、
18.8 GBのPUPと画像モデルを同時にGPU常駐させない。

```bash
python rocm_tools/flux3.py generate --model MODEL --upsampler PUP \
  --prompt 'A red sphere on the left and a blue cube on the right.' \
  --seed 1703 --width 1024 --height 1024 --output OUTPUT
```

外部LLMで展開済みの詳細なscene captionは`--caption '...'`または
`--caption-file caption.txt`で渡す。短い要求を未展開のまま渡す用途には使わない。
公式PUPを単独で実行し、展開結果を保存することもできる。

```bash
python rocm_tools/flux3.py upsample --model PUP \
  --prompt 'A red sphere on the left and a blue cube on the right.' --output CAPTION
python rocm_tools/flux3.py generate --model MODEL \
  --caption-file CAPTION/caption.txt --output OUTPUT
```

画像編集・複数参照は`--images reference.png`または複数パスを指定する。
PUPと生成には同じ参照画像を渡す。参照画像は公式Diffusersの前処理を使い、
16px単位のcropを行う。`--boxes boxes.json`はPUPへのlayout指定で、
上流の0～1座標形式を使う。展開済みcaptionにはそのままbox tableを含められる。

V620（gfx10）では、PUPが使用するQwen3.5のFLA BF16カーネルでLLVMの未対応命令が
発生したため、該当する4関数をTransformersのPyTorch reference実装へ切り替える。
インストール先のTransformersや他のLLMサーバーのコードは変更しない。

V620のDiT Attentionは、queryを既定1024行ずつ計算して一時領域を抑える。
各queryはすべてのkeyへattendし、近似的なwindow制限は行わない。
R9700はnative SDPAを使用する。NMSEではteacherとcandidateに同じ設定を適用する。
単独の`generate --attention-query-chunk 0`でnativeへ、正の値で明示的なquery分割へ
切り替えられる。通常経路と分割経路の出力比較テストも行っている。

## 校正・変換・EXL3生成

[固定ケース](../rocm_tools/flux3_cases.json)は校正と評価を分けている。
全方式で同じcaption、seed、BF16参照画像を使用する。

```bash
python rocm_tools/flux3_suite.py calibrate --model MODEL --root RUN
python rocm_tools/flux3_suite.py convert --model MODEL --root RUN --bits 3 4 5
python rocm_tools/flux3_suite.py evaluate --model MODEL --root RUN \
  --precision exl3 --bits 4
```

対象はimage・image_cond・textのper-mode blocksとjoint single blocksのLinear。
公式モデルは計258層あり、テキストだけの入力では参照画像用30層が呼ばれない。
部分的な校正を統合し、全対象層に実際の入力が揃った場合だけ変換する。
block外の入力投影・時刻埋め込み・modulation・出力投影・normは高精度で保持する。
途中の変換は同一source・bit数・校正manifestでのみ再開できる。

単独のEXL3生成は`generate`へ`--exl3 RUN/exl3-4bit`を追加する。
重みは列方向に復元して入力と同じBF16/FP16で行列演算する。
復元用workspaceは既定256 MiBで、dense DiT全体をGPUへ展開しない。
EXL3の全対象層について、参照画像の有無に応じた実行回数を検査する。

## BF16比較とNMSE

```bash
python rocm_tools/flux3_suite.py evaluate --model MODEL --root RUN --precision bf16
python rocm_tools/flux3_suite.py nmse --model MODEL --root RUN --bits 3 4 5
python rocm_tools/flux3_suite.py nmse-self --model MODEL --root RUN
python rocm_tools/flux3_suite.py benchmark --model MODEL --root RUN --precision bf16
python rocm_tools/flux3_suite.py benchmark --model MODEL --root RUN --precision exl3 --bits 4
```

NMSEはBF16の軌道上の同じlatent・時刻・移動先時刻・text/reference条件で候補を呼び、
`sum((candidate - BF16)^2) / sum(BF16^2)`を求める。出力はtarget画像トークンのみ。
差分・二乗はFP32、集計はFP64。候補の出力はschedulerに渡さない。
既定はtext・composition・multi_refの3ケースで、全要素・全ステップの総和比を報告する。
知覚品質のスコアではないので、生成画像の確認も併用する。

benchmarkはcompositionのwarmup 1回＋3回計測。初期ロード、PUP、PNG保存を除き、
text encoding・参照画像のVAE encoding・denoising・VAE decodeを含む。
`--cached`はtext encodingだけを別計測にするため、同条件でのみ比較する。
校正・NMSEの時間は通常の生成速度と区別する。
集計は`RUN/results.json`、画像品質の出力は`RUN/quality/精度/ケース/`、
速度計測は`RUN/benchmark/精度/ケース/`へ保存する。checkpoint本体の
`RUN/exl3-4bit`等とは別のディレクトリで扱う。
各画像と生データはケース別の`report.json`に保存する。
