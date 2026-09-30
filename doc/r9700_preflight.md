# R9700 起動前診断

EXL3のビルド・モデルロード前に、ROCm版torch、選択GPU、LDS容量、ビルド対象と
fused-MoE設定を確認する読み取り専用コマンドです。

```bash
python3 rocm_tools/r9700_preflight.py
python3 rocm_tools/r9700_preflight.py --device 0 --json
```

torchは実機照会時だけimportします。EXL3・native extensionのimport、JIT、モデルロード、
GPU演算、設定変更、subprocess実行は行いません。torchによるデバイス照会を使用します。
`torch.version.hip`はtorchが報告するHIPビルド情報です。

## 結果の読み方

| 終了コード | 意味 |
|---|---|
| 0 | 診断結果を取得し、ブロッカーなし |
| 1 | 入力不正、torch/HIP不在、GPU照会・ファイル読込のエラー |
| 2 | LDS不一致や危険な設定などのブロッカー |

R9700/gfx1201の通常結果は`comparison_workarounds_required`です。
exit 0はモデルの推論成功・一般的なRDNA4対応を意味しません。
現在のR9700対応は[比較用の回避策](r9700_vs_v620.md)を使った範囲です。
その外部adapterは本リポジトリに同梱されていません。

- gfx1201のLDS期待値は実測の65536bytes（64KiB）。92160bytesを仮定したビルドは以前失敗しています。
- ビルド時は`PYTORCH_ROCM_ARCH=gfx1201`などで対象を指定します。
  `setup.py`では非空の`PYTORCH_ROCM_ARCH`が`GPU_ARCHS`より優先されます。
  環境変数はビルドの意図であり、インストール済みbinaryのtargetはこのコマンドでは`unknown`です。
- `EXL3_ROCM_RDNA4_FUSED_MOE`は現行gfx12 WMMA経路の強制を避けるため確認します。
  現行`_env_on`の規則ではunset、空文字、`0`、`false`、`False`がOFF。
  **`off`、`no`、`FALSE`はONとして扱われます。**
- 記録する環境変数は上述の3変数だけです。
- gfx1200にもRDNA4の未検証経路の注意点を表示します。build-target一覧への掲載は推論検証の証拠ではありません。

## オフライン照会

torch/GPUがない環境でも、次の形式のJSONを`--snapshot FILE`で診断できます。
環境変数の判定には実行時の上述3変数を使います。snapshotに秘密情報を追加する必要はありません。

```json
{
  "schema": "r9700-preflight-snapshot/v1",
  "torch_version": "2.12.0+rocm7.2",
  "hip_version": "7.2.53211",
  "device": {
    "index": 0,
    "name": "AMD Radeon AI PRO R9700",
    "gcn_arch_name": "gfx1201",
    "total_memory_bytes": 34208743424,
    "shared_memory_per_block": 65536
  }
}
```

```bash
python3 rocm_tools/r9700_preflight.py --snapshot snapshot.json --json
```

snapshotのdevice indexは省略時にそのまま使います。`--device`を指定する場合は記録と一致する必要があります。
feature suffix（例`gfx1201:sramecc+:xnack-`）は正規化時に除去します。
HIPがnullのsnapshotはブロッカーです。

## 検証（2026-10-01）

```bash
python3 -m unittest discover -s rocm_tools/tests -p test_r9700_preflight.py -v
```

CPUテスト50件が成功。実機のROCm containerでも通常診断、危険なfused-MoE強制、
無効device番号を確認しました。実機はR9700 / gfx1201 / LDS65536bytes、
torch2.12.0+rocm7.2、torch報告HIP7.2.53211です。
通常はexit0、強制時はexit2、無効deviceはexit1でした。モデル推論の測定ではありません。

実装本体と最初の45テストは、ローカルEXL3 APIに接続したOpenCode2.0.12が作成しました。
Codexがレビューを行い、残る環境変数優先順位・案内の修正、5件の回帰テスト、ドキュメントと実機確認を追加しました。
