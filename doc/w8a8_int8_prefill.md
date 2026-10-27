# W8A8 INT8 プリフィル (feat/w8a8-int8-prefill)

## 目的

Radeon PRO V620 (gfx1030, RDNA2) x2 上での EXL3 プリフィル高速化。
RDNA2 には WMMA がなく、利用可能な INT8 ドット積は `udot4`
(unsigned のみ; `sudot4`/`sdot4` は gfx1030 に存在しない) である。
EXLR8 (FP8 テンソルコア向け) の発想を RDNA2 に移植する:
トレリス復号値を INT8 グリッドに載せ、活性を INT8 化し、
DP4A レートで GEMM する。

## 現状まとめ

- ベースライン (TP2, AR, 3.05bpw, fp16 パス): 512 tok で約 365 tok/s、
  2048 tok で約 571 tok/s、8192 tok で約 538 tok/s (first-delivery 基準)。
- rocBLAS int8 GEMM (int8 x int8 -> int32) は gfx1030 で動作する
  (correctness PASS)。fp16 対比 1.06x (m=128) / 1.16x (m=512) /
  1.23x (m=2048) / 1.47x (m=8192)、MoE expert 形状で 1.3〜1.6x。
  C が int32 (4B/elem) のため、m が大きいほどトラフィックが重くなる。
- mul1 トレリス復号値の INT8 (per-tensor) 丸めノイズ: SNR 約 40〜41.5 dB
  (rms 0.0078 対 weight std 0.8〜0.93)。重み側の誤差は小さい。
- K は最大でも数千のため int32 累積はオーバーフローしない
  (K * 127 * 127 < 2^31)。

## 設計 (Phase 1)

`LinearEXL3.reconstruct_hgemm` に `EXL3_INT8_PREFILL=1` で有効化する経路:

1. 入力 Hadamard (回転基底、非 fused パスと同じ)。
2. 内部トレリス重みを per-output-channel 対称 int8 グリッドへ丸め、
   スケールを svh のコピーに折り込み (`_w8` / `_svh8` を遅延キャッシュ)。
   エピローグは変更なし。
3. 活性を行ごと absmax / 127 で int8 化 (per-token)。
4. int8 GEMM (C は int32)。`ext.igemm_recon` (rocBLAS int8) があればそれを、
   なければビット正確なエミュレーション
   (`_exact_int32_gemm`: K=512 ブロック fp32 積 + fp64 累積、TF32 無効)。
5. 行スケールを戻し、出力 Hadamard (+ bias)。

対象は rows < 1024 の非 fused 域 (fused original-basis パスは rows >= 1024
で既に Hadamard を消しており、そちらは当面 fp16 のまま)。
グラフキャプチャ (`capture`) と sliced lm_head は fp16 フォールバック。
MoE (fused `exl3_moe` / batched tier) は対象外。

## 精度評価 (KL プロトコル)

(HF `jyohukuchan/qwen38-flash-next-bf16-logits`、21 ケース、
`compare.py` の macro case-mean KLD を主指標。詳細は後述。)

## 速度評価

(計測中。)

## 正直な注意点

- rocBLAS int8 の C(int32) ラウンドトリップはトラフィックを増やす。
  torch-op の量子化/逆量子化パスを素朴に足すと GEMM の勝ち分が消える。
  融合エピローグ (int32 + 行/列スケール + Hadamard + fp16 書き戻し) が
  なければ勝てない可能性が高い。
- 桁違いの本命は trellis-packed のまま DP4A するカスタム GEMM
  (upstream INT8 GEMV の GEMM 版、重み側の丸め誤差ゼロ) だが、
  Phase 2 以降の大型カーネル開発になる。

## 次の手順

- Phase 1 の KLD / 速度 A-B。
- 速度が出なければ Phase 2 (re-encode) 前に親に相談。
- 長期的には fused `exl3_moe` / `mgemm` の INT8 化。
