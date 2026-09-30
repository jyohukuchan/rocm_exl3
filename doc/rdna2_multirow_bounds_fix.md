# RDNA multi-row GEMV の余り行読み越し修正

2026-09-30、V620 × 2 の Qwen3.8 Flash Next / EXL3 3.05bpw / MTP / K5/V4 ベンチマーク中に発見。修正 commit: `ab1bc11`。

## 問題

batch 4 の終了付近で実行中の job が減り、共有 MLP が `[1, 5, 2560]` を処理する際、GPU 0 が page-not-present fault で終了した。相手 rank の TCPStore エラーや RCCL timeout はその後に発生した。

`exl3_gemv_multirow_rdna.hip` は実際の行数 `m` に対して 1/2/4/8 行の template tile を選ぶ。5 行なら tile は 8 行になる。出力側には `rows_valid` の判定があったが、dot core は `A + r * lda` を無条件に読み、余りの行を読み越していた。3/5/6/7 行で問題になり得る。

`BC_GatedMLP::run_bszN_gr` の gate/up 用 scratch は `{2, num_tokens, width}` の密な配置であり、最後の行列の末尾に追加の読み取り余白を保証しない。GPU allocator が隣接領域もマップしている場合には表面化しないため、通常の数値比較だけでは検出できなかった。

この native ファイルは今回の Python 側 MoE 行数拡張では変更していなかった。従来の checkpoint 削除処理を使い、各 module と routed GEMM の完了を同期しても再現した。最後に成功した routed expert の計算と、fault が出た shared expert の境界をログで特定した。

## 修正

既存の `rows_valid` を共通 body から split-K helper、dot helper まで渡し、入力アドレスを作る前に次のように制限する。

```cpp
const int r_src = r < rows_valid ? r : rows_valid - 1;
```

余りの行は最後の有効行を読み、出力側では従来どおり保存しない。有効行の演算順序、reduction、graph の pointer patch、launch geometry は変更していない。

## 検証

- HIP VMM で scratch の直後に未マップの guard page を置いた。旧 native の 4 行は成功し、5 行・2 行列では **guard の先頭アドレスそのもの**で fault を再現した。
- 修正版は単一行列・複数行列の 1～8 行、計 16 ケースで guard を越えず、int32 view による参照との bit pattern 比較も一致した。
- 拡張した `rocm_tools/multirow_check.py` の 200 ケースが成功。有効行の 1 行ずつの計算との一致、および別経路との既定 tolerance 内の一致を確認した。
- 同じ実モデルの batch 4・固定 MTP 4 による cold / prefix reuse / eviction / reuse の 4 段階が正常終了した。各 reuse で 1,792 tokens が再利用され、古い checkpoint の削除後も動作した。K5/V4、2 GPU、Engram の単一 RAM 常駐、有限値を確認した。
- 修正版による batch 4・MTP 1～4 の一括測定も正常終了した。

検証範囲は V620 (`gfx1030`)。ここでの数値一致は native 演算の比較であり、cold prefill と prefix reuse の生成文が常に bit 単位で一致するという主張ではない。

## 使用バイナリと記録

既存バイナリを上書きせず、V620 コンテナ内の `/work/lib-context-mr-bounds` にビルドした。

```text
旧 SHA256: 12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a
新 SHA256: 57afa48c9a61b9e7ba2721917bf011096d7cc947e43b2a8444ee8350f0820718
PYTHONPATH=/work/lib-context-mr-bounds:/src
```

記録は `/home/homelab1/datapool/rocm-exl3-rdna2/runs/context-batch/`。

- `mr_guard_probe.py`, `guard-old-m5-multi-verdict.json`: 境界検出用の小さな再現テスト。
- `native-bits-gates.json`, `guard-fixed-bits-*.json`: 厳密な bit pattern 比較を含む guard 結果。
- `multirow-expanded-fixed.log`, `build-mr-bounds.log`: 数値テストとビルド。
- `reuse-legacy-native-debug.log`, `moe34-capture-*`: 修正前の shared MLP 内での停止箇所。
- `reuse-legacy-native-fixed.json`, `reuse-prune-native-fixed.json`: 修正後の実モデル検証。後者の checkpoint 削除最適化は別変更であり、この読み越しの原因ではない。
