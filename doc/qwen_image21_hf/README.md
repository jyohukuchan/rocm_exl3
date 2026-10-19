# Qwen Image 2.1 EXL3 Hub publication

通常版とTurboを分け、各リポジトリに3bit・4bit・5bitのDiTを収録する。

- [Qwen-Image-2.1-EXL3](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-EXL3)
- [Qwen-Image-2.1-Turbo-EXL3](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-Turbo-EXL3)

名称は量子化元と形式を表す。GPUの検証範囲はモデルカード内の環境説明に記載する。
初回公開のcommit、`exl3-v1`タグ、検証済みファイル数は [publication.json](publication.json) に記録している。

公開後に[V620単体での動作を確認](../qwen_image21_v620.md)し、両リポジトリのmainへ
対応範囲と`gfx1030`のビルド指定を追記した。`V620_VALIDATION.json`を追加し、
`FILE_MANIFEST.json`のREADME・検証記録のチェックサムも更新している。
重みと`exl3-v1`タグは変更していない。[更新commitと認証なしの読み戻し検証](v620-publication.json)を参照。

## ライセンスとパッケージ

重みには元モデルの **Qwen Research License Agreement（2026-09-20）** を適用する。
研究・評価目的の非商用利用に限定され、商用利用は上流の別途許諾が必要。
エンジンのMITライセンスを重みに適用しない。

元モデルの固定revisionから取得したLICENSEを、rootと各bitディレクトリへそのままコピーする。
指定の帰属表示を `Notice` に記載し、モデルカードに `Built with Qwen` を明記する。
改変・出典・ライセンスの通知を各safetensorsのメタデータへ加え、`quantization.json` のchecksumを更新する。
原本の評価済みcheckpointは変更しない。すべてのテンソルpayloadがバイト単位で元の評価済みファイルと一致することを検証する。
未変更のtransformer configはそのままコピーする。

公開対象は推論に必要な225個のsafetensors／bit、config、manifest、ライセンス、モデルカード、NMSEと目視の記録。
text encoder・VAE・元のBF16 DiT・校正テンソル・私用ログは含めない。
カードには上流のtext encoder・VAE・processor・schedulerだけを取得する推論手順を載せる。
NMSE計測では別途元のBF16 DiTが必要になることも記載する。

## ローカルで準備

```bash
python rocm_tools/prepare_qwen_image21_hf_release.py \
  --variant base \
  --source-model /path/to/Qwen-Image-2.1 \
  --checkpoints-root /path/to/qwen-image21 \
  --output /path/to/new-staging/Qwen-Image-2.1-EXL3 \
  --repo-id jyohukuchan/Qwen-Image-2.1-EXL3
```

Turboは `--variant turbo`、対応するsource modelと公開先名を指定する。
このコマンドはローカル準備のみを行い、既存の出力先は上書きしない。
[モデルカードのテンプレート](README.template.md)を使用する。

準備後は公開用checkpointから生成を確認する。今回、両モデルの5bitで
元のBF16 DiTディレクトリを持たない構成からテキストを新規encodeして生成し、
評価済みcheckpointのcomposition画像と画素hashが一致することを確認した。
その結果を各パッケージの `VALIDATION.json` と `FILE_MANIFEST.json` に反映してから公開した。
ヘッダ変更のテンソル保存性・原本保護・不正checksum拒否のCPUテスト2件も成功した。

## 明示的に公開

```bash
HF_XET_HIGH_PERFORMANCE=1 python rocm_tools/publish_qwen_image21_hf_release.py \
  --folder /path/to/new-staging/Qwen-Image-2.1-EXL3 \
  --repo-id jyohukuchan/Qwen-Image-2.1-EXL3 \
  --result /path/to/base-published.json
```

このコマンドは公開リポジトリを作成してアップロードする。
既存リポジトリは既定では拒否する。`--resume` は公開済みの `FILE_MANIFEST.json` と
ローカルが一致する同一releaseの再開に限る。

ライセンス・モデルカードを先に送り、重みをbit別に送る。
アップロード後はリモートの全ファイル名・サイズとLFS SHA256／Git blob hashを照合し、
`exl3-v1` タグを作成する。認証なしのAPIでも公開状態とタグのcommitを確認する。
`exl3-v1` が既存の別commitを指す場合は検証に失敗し、既存タグを移動しない。
