# Qwen3.8 Flash Next: 3/2bpw 半々と DFlash2 の実現性

調査日: 2026-09-30。decode 改善作業完了後の追加調査。実装・再量子化・draft 重み取得は実施していない。

## 結論

容量面では有望。元の **3bpw 部分だけ**を重み数で半分ずつ 3bpw / 2bpw にすると、packed weight は約 **7.03 GiB** 減る。既存 MTP を外せば、合わせて約 **7.99 GiB** を代替 draft と追加状態に使える計算になる。ただし GPU ごとの配置、cache、workspace を含む実ロード保証ではない。

現時点の障害は容量よりも対応 draft と実装である。公式コレクションおよび Hugging Face の DFlash2 検索では Flash Next 用チェックポイントを確認できず、公開 Qwen3.8-27B 用は非互換。本家 ExLlamaV3 の DFlash2 も TP target を拒否する。まず対応 draft を確保し、layer split で成立性を検証してから TP に進む順序が妥当。先に量子化品質を落として容量だけ空ける理由はまだない。

## 半々の定義と容量

ユーザー指定は fractional な 2.5bit 形式ではなく、元の 3bpw 対象部分のパラメータ数を半々に配分する方式。既存 4/5bit 部分や RAM 上の Engram は変更しない。

現行モデルの全 7 safetensors のヘッダーから集計した値:

| 項目 | 値 |
| --- | ---: |
| target の 3bit パラメータ数 | 120,815,616,000 |
| その半分を 1bit 減らす容量 | 7,550,976,000 bytes = 7.0324 GiB |
| 現行 MTP の packed tensors | 1,024,102,952 bytes = 0.9538 GiB |
| MTP を置換する場合の重みだけの予算増 | 7.9862 GiB |

これは trellis payload の削減試算。モデル全体の平均が 2.5bpw になるという意味ではなく、実 VRAM 測定値でもない。

現在の `exllamav3/modules/multilinear.py` は同じ bundle の全 Linear が同じ `inner.K`（bit 数）であることを要求する。MoE の各 local expert を束ねる経路に任意の 2/3bit expert を混ぜるには、bit 別 grouping / dispatch の変更が必要。

初期案としては **48 MoE 層中 24 層を 2bit、24 層を 3bit** とし、各層内の expert / projection の bit 数を揃える。routed MoE だけで約 7.03125 GiB 節約でき、厳密な半々との差は約 1.17 MiB。どの層を下げるかは calibration で品質劣化を比較して決める。既存 3bit データの単純な bit 切り捨てではなく、公式元重みから再量子化する。

## DFlash2 のモデル互換性

[公式コレクション](https://huggingface.co/collections/incoai/dflash-2) と HF API の DFlash2 検索（保存時 195 件）では Flash Next 用を確認できなかった。これは調査範囲での結果で、未公開・別名配布まで存在しないと断定するものではない。

[公開 Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) は hidden size 5120、64 層 target 向けで、特徴取得層に 61 を含む。現行 Flash Next は hidden size 2560、48 層であり、そのまま接続できない。公開 draft は約 1.924B parameters、BF16 の重みだけで約 3.584 GiB。これは容量の参考値であって、Flash Next 用 draft の見積もりではない。

Flash Next 向けには別方式の [PixelML DeepSpec DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash) がある。DFlash2 ではなく、NVFP4 target 向けに学習された約 498M parameters の draft。共有 embedding/head を除く BF16 重みは約 0.928 GiB で、重みだけなら現行 MTP と近い。これを試す場合、2.5bpw 化が必須とは限らない。ただし EXL3 target の特徴・量子化差による採用率と engine adapter を検証する必要がある。

その作者の公平な MTP 比較では、集計 +3.87%、数学 +28.0%、コード +0.4%、会話 -5.9%。別ハードウェア・別エンジンの結果であり、V620 や日本語への速度予測には使えない。「外部 draft にすれば常に速い」とは言えない例になる。[作者の検証報告](https://github.com/PixelML/deepspec-qwen38-flash-next/blob/main/blog/README.md)

## ExLlamaV3 / ROCm / TP で必要な作業

[本家 DFlash2 実装](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/architecture/dflash2.py) は selector が top-k logits を必要とするため、`target.loaded_tp` に対して `NotImplementedError` を送出する。現行 ROCm fork の TP borrowed head は argmax 用であり、本家実装をそのまま移すだけでは TP2 対応にならない。

必要な作業は以下。

1. Flash Next 対応 draft の入手、特徴取得層・HC stream の扱い・token alignment の一致確認。未配布なら学習が別途必要で、現段階では所要時間を根拠付きで見積もれない。
2. 本家の DFlash2 architecture / selector / dynamic convolution / native top-k を移植し、HIP と gfx1030 で数値検証。RDNA3 の [ROCm 移植例](https://github.com/phoenixhaxor/exllamav3-rocm) は参考になるが、V620 動作を保証せず、WMMA 依存部分はそのまま使えない。
3. layer split で target 検証、reject 時の KV/GDN rollback、cache 境界、採用率・速度を確認。
4. TP vocab shard の top-k 候補を収集・統合して selector に渡す経路と、target 特徴取得を実装・検証。通信量だけでなく同期回数も測る。
5. 必要容量を確認してから mixed-bit 再量子化と品質検証。層単位配置なら既存 bundle 制約を守れる。expert 単位配置を選ぶなら追加の kernel dispatch 改修が必要。

調査に用いた本家ローカル参照 commit は `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`。現行作業ツリーには DFlash2 の architecture / native bindings はまだない。

## 高速化の条件と比較方法

現在の MTP の代表的な日本語計測では、出力 token 当たり draft 約 2.5ms、target verification 約 24.5ms。採用長と verification が同じなら、draft だけを無料にしても速度向上は約 10% に限られる。大きな改善には、より多くの token を一度の target 検証で採用できることが必要。

`時間 / 出力 token ≈ (draft block 時間 + target 検証時間) / 平均採用出力長`

block 8（draft 7 + 検証用 1）の場合、現行 native `MAX_BSZN=8` の境界内である一方、GDN rollback history を 4 → 7 に増やすだけで両 GPU 合計約 0.316 GiB 増える。draft 自身の KV、特徴 buffer、workspace、GPU ごとの偏りも別途必要。大きい block の性能は改めて測定する。

DFlash の検証が保持するのは採用した target の分布であり、2bit 化による target 自身の品質劣化を取り戻すものではない。また target の変更は draft 採用率にも影響し得る。

実験時は次の 4 条件を分ける。

| target の元 3bit 部分 | draft |
| --- | --- |
| 全て 3bit | 既存 MTP（基準） |
| 3/2bit 半々 | 既存 MTP |
| 全て 3bit | 対応 DFlash2 |
| 3/2bit 半々 | 対応 DFlash2 |

K5/V4、電力方針、入力、生成長を揃え、日本語・コードの prefill / decode、実出力速度、採用長、peak VRAM、品質を比較する。現時点では高速化倍率の数値予測はできない。

## 保存した根拠

ローカル資料: `/path/to/rocm-exl3-data/runs/dflash2-feasibility/`

- `local-bit-inventory.json`, `memory-scenario.json`: 重み集計と容量試算。
- `official-config.json`, `official-model-info.json`, `official-README.md`: 公開 27B DFlash2。revision `015e795645c74b1a0eeef3b570031fb62e769bc5`。
- `pixel-config.json`, `pixel-model-info.json`, `pixel-README.md`: Flash Next 向け別方式 draft。revision `9cd660f9050c92fedc88cbe547bd53af0392abe1`。
- `dflash2-search.json`, `inco_models.json`, `flash_search.json`: 配布物検索のスナップショット。

先行作業の測定・検証結果は [V620 TP decode 改善報告](v620_tp_decode_optimization.md) を参照。
