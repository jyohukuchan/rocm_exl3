# FLUX.3 Image Klein EXL3のHugging Face公開

公開先: [jyohukuchan/FLUX.3-Image-Klein-EXL3](https://huggingface.co/jyohukuchan/FLUX.3-Image-Klein-EXL3)。
3・4・5bitを別フォルダに配置する。モデルカードには推論手順、NMSE、比較画像、
速度の計測条件、実機確認の範囲を記載する。PUP・text encoder・VAEは同梱しない。

2026-10-19に公開完了。[固定tag `exl3-v1`](https://huggingface.co/jyohukuchan/FLUX.3-Image-Klein-EXL3/tree/exl3-v1)
はrevision `620ad008afce6c90f2cae917f18d39e6ac4530a8`。
806ファイル・9,574,003,416 bytesについて、リモート全サイズ・全hashの一致と匿名での
tag/manifest読み戻しを確認した。[公開記録](../../benchmarks/2026-10-19/flux3-hf/publication.json)。

## ライセンス確認

2026-10-19に固定revisionの原文をHubから再取得し、取得済みファイルとの一致を確認した。

- 元モデル: `black-forest-labs/flux-3-image-klein@337337621b8526e120f2275d250242f4a6bedd75`
- [LICENSE.md原文](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/LICENSE.md)
  SHA256: `100a3a5af8f1a2943dca0d127671bcaa93ad894a63330825219f1a0a44451f22`
- [NOTICE.md原文](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/NOTICE.md)
  SHA256: `a12f7eb0cab87ed210ffb0921425589a1c88abded6ba0c4285b4d32cbaf5f35c`

FLUX Kommunity License v1.0の§2・§3に基づく非商用の改変・再配布として公開する。
§3(a)のライセンス添付とBFLから受領者への直接許諾の明示、§3(b)の指定帰属表示、
§3(c)の変更内容の明示と公式製品・承認済み製品との誤認防止を反映した。
元のライセンスに追加するモデル利用条件は設けない。
§10の説明に合理的に必要な参照の例外に基づき、元モデルの名前を説明に使用する。

§2(a)は、関連会社を含む年間換算総収益が500万米ドル未満のQualifying Userに対し、
生成物の商用利用とその生成に限定した本番運用を条件付きで認める。
§2(e)のフィルタリングまたは生成物のレビュー、法令上必要なAI表示等の条件も適用される。
商用再配布・学習・本番用途全般を一律に許可する条項ではない。
用途制限等を含む全文を各利用者が確認する必要がある。
エンジンのMITライセンスを重みに適用しない。

## 公開ファイルの作成と検証

```bash
python3 -m rocm_tools.prepare_flux3_hf_release \
  --source-model "$FLUX_SOURCE" --checkpoints-root "$FLUX_CHECKPOINTS" \
  --output "$FLUX_RELEASE" --repo-id jyohukuchan/FLUX.3-Image-Klein-EXL3
```

変数にはモデル、評価済み`exl3-{3,4,5}bit`の親ディレクトリ、新規staging先を指定する。
元のcheckpointは書き換えない。全777個のsafetensorsについて、元ファイルSHA256を
検査し、既存metadataと全tensor descriptorを維持して通知metadataを追加する。
保存後に全payload byteの一致を確認し、再配布ファイルのchecksumをmanifestへ反映する。
原文LICENSE.mdと変更を追記したNOTICE.mdをrootと各bitフォルダへ配置する。

公開前にはstagingの4bitを使い、dense DiTを含まないcomponentsとfresh text encodingで
1024px・seed 1703・16ステップの生成を実行した。228個のactive EXL3 Linearが全ステップで
動作し、latentとVAE出力はfinite。画像の画素SHA256は評価時の4bitと完全一致した。
この実行はcold動作確認であり、新しい速度benchmarkには使用しない。

[検証結果](../../benchmarks/2026-10-19/flux3-hf/validation.json)、
[実行report](../../benchmarks/2026-10-19/flux3-hf/staged-4bit-report.json.gz)。
同じ検証結果を`VALIDATION.json`としてstagingへ添付し、`FILE_MANIFEST.json`にも登録した。
テンソル保持・元checksum不一致拒否の小規模テストは2件成功した。

```bash
python3 -m rocm_tools.publish_flux3_hf_release \
  --folder "$FLUX_RELEASE" --repo-id jyohukuchan/FLUX.3-Image-Klein-EXL3 \
  --result "$FLUX_RESULT"
```

publisherはファイル一覧・ローカル全SHA256・モデルカード・送信先アカウントを検査する。
rootのライセンス等を先に送り、その後にbit別で転送する。各commitのLFSファイル数は500未満。
公開後、全ファイルのサイズとLFS SHA256またはGit blob SHA1を照合し、`exl3-v1`を作成する。
匿名アクセスによるtagとmanifestの読み戻しまで検証する。
既存repoへの再開は同一FILE_MANIFESTの場合に限り`--resume`で明示する。

検証用エンジンrevisionは`0d879e7a7b6aadd3c08508240ade6db87332bafe`。
今回追加した公開処理は推論コードや元の評価済み重みを変更しない。
