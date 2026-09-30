# R9700 プリフライト (`rocm_tools/r9700_preflight.py`)

EXL3 を **ビルド・ロードする前**に、R9700 (gfx1201) の環境を診断する
読み取り専用CLI。環境診断であり、RDNA4推論一般の動作証明ではない
([r9700_vs_v620.md](r9700_vs_v620.md) の比較は外部adapterを要した。本ツールは
それらを同梱も適用もしない)。

- torch は **デフォルトで遅延importし、デバイスproperty読み取りだけ**に使う。
- exllamav3 の import、native extension の import/JIT、モデル読込、GPU kernel
  実行、subprocess 起動、インストール、環境変更、ファイル書き込みを一切行わない
  (出力は stdout のレポートのみ)。
- gfx1201 の実測 LDS = 64 KiB (65536) という expectation を報告し、
  unexpected / inadequate な値を診断する (setup.py `GPU_ARCH_SMEM` と
  `rocm_tools/hipcc_probe.sh` の根拠に対応)。
- ビルドターゲット環境 (`PYTORCH_ROCM_ARCH` / `GPU_ARCHS`) と実機 arch を
  比較するが、**環境設定はコンパイル済みバイナリのアーキテクチャの証拠では
  ない**と明示する。バイナリターゲット検証は常に `unknown` のまま。
- 記録する環境変数は build / fused-MoE 関連の許可リストのみ。他の環境変数や
  機密は一切記録しない。

## ステータスと exit code

| status | 意味 | exit |
|---|---|---|
| `comparison_workarounds_required` | gfx1201 の事実を収集。ready / general_supported ではない | 0 |
| `rdna4_unvalidated` | gfx1200 系の RDNA4。当リポジトリに実測記録なし | 0 |
| `not_r9700_supported_part` | port 対応アーキだが R9700 ではない | 0 |
| `outside_supported_matrix` | setup.py SUPPORTED_GPU_ARCHS 外 | 0 |
| `invalid_device_selection` / `invalid_snapshot` | 診断入力エラー (UTF-8でないsnapshot含む) | 1 |
| `torch_not_importable` / `not_rocm_torch` / `no_hip_device_visible` | HIP/torch が無い | 2 |
| `blocked_fused_moe_forcing` | gfx120x で `EXL3_ROCM_RDNA4_FUSED_MOE` が ON | 3 |

exit 0 は「レポートを収集できた」だけであり、推論が動く証明ではない。
`EXL3_ROCM_RDNA4_FUSED_MOE` の ON/OFF 判定は実行時と同じ `_env_on` 意味論
(設定値を strip し `""` / `0` / `false` / `False` 以外を ON) を再現する。
gfx120x で ON の場合、fused MoE の `rdna_wmma` は gfx12 で
`__builtin_trap()` に到達するため、ブロック扱い (有限 non-zero exit) になる。
なお `EXL3_SKIP_ROCM_VERSION_CHECK` (ROCm最小バージョンゲート) は `_env_on`
ではなく setup.py と同じ生の truthiness を再現する — 空でない文字列なら
(例 `"0"` でも) ゲートはスキップされ、`""` はスキップしない。

## 使い方

```bash
# 実機照会 (torch 遅延import、デバイスproperty読み取りのみ。ファイル書き込みなし)
python3 rocm_tools/r9700_preflight.py

# 別インデックスのGPU / 機械可読出力
python3 rocm_tools/r9700_preflight.py --device 1 --json

# 手で書いた (か他機で採取した) facts snapshotをCPU-onlyで再現診断
python3 rocm_tools/r9700_preflight.py --snapshot r9700.json --json

# 異常系の確認 (fused-MoE強制 = ブロック、exit 3)
python3 rocm_tools/r9700_preflight.py --snapshot bad.json; echo $?
```

## JSON スナップショット書式 (schema_version 1)

`--snapshot FILE` が読み取る形式。手で書く場合は以下の form に従う。
必須フィールド欠落・型違反・不正なindexはすべてドットパス付き問題リストで
exit 1 (不正UTF-8/不正JSONも同じく診断。`--json` 時は同じ内容が機械可読で
stdout に出る)。snapshot の書き出しオプションは意図的に持たない。

```json
{
  "kind": "r9700_preflight_snapshot",
  "schema_version": 1,
  "torch": {
    "importable": true,
    "version": "2.12.0+rocm7.2",
    "hip_version": "7.2.53211-…",
    "cuda_is_available": true,
    "device_count": 1
  },
  "rocm": {
    "version": "7.2.4",
    "parsed": [7, 2, 4],
    "version_file": "/opt/rocm/.info/version"
  },
  "devices": [
    {
      "index": 0,
      "name": "AMD Radeon AI PRO R9700",
      "gcn_arch_name": "gfx1201:sramecc+:xnack-",
      "total_memory": 34359738368,
      "shared_memory_per_block": 65536
    }
  ],
  "env": {
    "PYTORCH_ROCM_ARCH": "gfx1201",
    "EXL3_ROCM_RDNA4_FUSED_MOE": "0"
  }
}
```

- `gcn_arch_name` は torch 準拠の素文字列。`:sramecc+:xnack-` などの feature
  接尾辞は診断側で strip し、arch と features に分離して報告する。
- `torch.version` / `torch.hip_version` は null か空でない文字列。
  `hip_version: null` は「HIP無し」の正当な表記で、exit 2 の診断になる
  (スキーマエラーではない)。
- `total_memory` / `shared_memory_per_block` は bytes。torch build が
  property を持たない場合は `null` を許す (その場合 unchecked として警告)。
- `rocm` は省略/null可。`version` / `version_file` は null か空でない
  文字列、`parsed` は null か `[major, minor, patch]` の3つの非負整数
  (bool不可)。不正だと解析例外ではなくスキーマ診断になる。
- `schema_version` は整数 `1` (boolの `true` は拒否)。
- `env` は省略可。**snapshot の `env` が権威**で、無い場合は全関連変数
  「未設定」として扱う (ホスト環境は読み込まない。再現性のため)。

## テスト済み範囲 (CPU-only gates)

- `python3 -m unittest discover -s rocm_tools/tests -p test_r9700_preflight.py -v`
  — 67 tests, GPU・native extension・torch 実体不要 (offline facts / snapshot
  ファイル / sys.modules に注入した fake torch)。
- 主な回帰: gfx1201 feature接尾辞の正規化、64 KiB LDS一致 / 不足(32768) /
  過大(92160) / 未報告、`_env_on` 意味論の ON/OFF 表 (`no`・`FALSE` は ON)、
  gfx1200/gfx1201 の fused-MoE 強制ブロックと非gfx120xでの inert、
  gfx1200 の next steps が `PYTORCH_ROCM_ARCH=gfx1200` / `GPU_ARCH=gfx1200`
  を使う (gfx1201ピン留めを推奨しない。強制・非強制双方)、92160 (表の仮定) と
  65536 (gfx1201実測) の区別、存在しない引用文の混入禁止、
  hipcc_probe の位置づけ= fresh source compile 診断 (インストール済みbinary非照会)、
  非R9700・対応外arch、HIP無し (torch無し / CUDA build / デバイス0)、
  不正index・不正snapshot (ファイル無し / JSON崩れ / 不正UTF-8バイト列 /
  kind / schema_version=true / `torch.hip_version`型 / `rocm`型・`parsed`要素型)、
  `EXL3_SKIP_ROCM_VERSION_CHECK` の raw truthiness (`"0"`はスキップ・`""`は非スキップ)、
  build-env の複数書式 (`,` / 空白 / feature接尾辞 / `GPU_ARCHS` fallback)、
  text と JSON の exit code 一致、fake torch live照会での secrets 非記録、
  `--save-snapshot` のような書き込みmodeが存在しないこと (argparse拒否)。

## 正直な限界

- このツールは**環境の事実を照会して診断するだけ**だ。ビルドもしない、
  extension を import もしない、モデルも load も kernel も実行しない、
  ファイルも書かない。よってスループットや数値の保証は一切ない
  (向上するのは起動時の診断だけ)。
- 環境変数は「意図したビルドターゲット」を示すだけで、コンパイル済み
  バイナリのアーキテクチャ検証は `unknown` のまま。
  `rocm_tools/hipcc_probe.sh` は任意の**ソースコンパイル**診断 (freshな
  translation unit を setup.py風 flags で build する。副作用 — 中間物の
  書き出し、hipcc/torch 必要 — はそのツール側にあり、本プリフライトの
  外で別途実行するもの) であり、**インストール済み extension binary の
  アーキテクチャを照合しない**。バイナリターゲットの確認は別途行うこと。
- gfx12 WMMA を有効化しない。V620 (gfx1030) 限定の tune を他のカードに
  適用もしない。既知の benchmark 数値を持ち出して改善を主張もしない。
- gfx1200 は setup.py の表上 92160 假定だが、当リポジトリに実測記録は
  ない (実測済みは R9700 のみ)。診断は honest な unvalidated 扱いにとどめる。
- 実測 snapshot の env 記録は上記許可リストのみ。`rocminfo` 等は呼ばない
  (torch の device property と `$ROCM_PATH/.info/version` ファイル読み only)。
- `GPU_ARCH_SMEM` / `SUPPORTED_GPU_ARCHS` / `MIN_ROCM` は setup.py の
  mirror データ。setup.py 側が変わったら本ツール (とテスト) も更新が必要。
