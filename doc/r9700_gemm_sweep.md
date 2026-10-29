# R9700 (gfx1201, RDNA4 WMMA) vs V620 (gfx1030, RDNA2 dot4): INT8/FP16 GEMM

問い: V620 の「MoE expert サイズで INT8 は FP16 の 1.0〜1.2x、
dense 大で ~1.8x」は、WMMA を持つ R9700 で大きく変わるか。
答え: **変わらない。むしろ torch 経路では悪化する。**
MoE 実 rows/expert 域で R9700 torch INT8/FP16 = 0.8〜1.2x、
dense torch では **0.6x (INT8 が遅い)**。WMMA 実働の naive Triton
でも MoE 域は 1.0x 前後、大 dense で 1.2〜1.6x に留まる。

GEMM-only、合成データ。スクリプト `eval/prof/r9700_gemm_sweep.py`
(ext 不要、torch + Triton のみ。同一スクリプトを両 GPU で実行)。
生 JSON: `eval/prof/r9700_gemm_sweep_{v620,r9700}.json`。

## 条件

- R9700: コンテナ `r9700-i8f8` (R9700 のみ可視)。V620: コンテナ
  `rocm-exl3-w8a8` (HIP_VISIBLE_DEVICES=1、V620 のみ)。
  1 プロセス 1 GPU (混載は segfault するため)。
- 計時は CUDA イベント中央値 (+min/max)。fp16/int8 交互実行。
- 形状: q30b expert gu (R,2048)x(2048,768) / d (R,768)x(768,2048)、
  Flash Next expert gu (R,2560)x(2560,640) / d (R,640)x(640,2560)
  (config 確認済: q30b hidden 2048 / moe_intermediate 768 / 128 experts top8、
  Flash Next hidden 2560 / moe_intermediate_size 640 / 512 experts top10)、
  dense 27B gate_up (M,5120)x(5120,17408) / down (M,17408)x(17408,5120)
  (Qwen3.8-27B config: hidden 5120 / intermediate 17408)。
- 実 rows/expert: q30b は T=512/2048/8192 → 32/128/512、
  Flash Next は 10/40/160。
- 公称ピーク: R9700 FP16 95.7 TFLOPS / INT8 766 TOPS (ベンダー製品頁。
  ただし後述の通り dense 実測 122 が 95.7 を上回るため、この 95.7 は
  dense GEMM 上限ではない。別資料の dense 換算 191/383 の方が実測と整合)。
  V620 FP16 40.55 TFLOPS / FP32 20.28 (ベンダー仕様)、INT8 ~81 TOPS
  (FP32 の 4x、dot4 前提の推定)。

## バックエンド可否 (ext を使わないという制約下)

| backend | V620 (gfx1030) | R9700 (gfx1201) |
|---|---|---|
| torch fp16 matmul (hipBLASLt) | OK | OK |
| torch._int_mm | 不可 (TensileLibrary_lazy_gfx1030.dat 無し。hipBLASLt heuristic が HIPBLAS_STATUS_INVALID_VALUE) | OK (ただし M>16 必須。M=8/16 は RuntimeError) |
| Triton tl.dot fp16 / int8 | OK | OK (下記 WMMA 確認) |
| torch fp8 e4m3 matmul | 不可 (addmm_cuda not implemented for Float8_e4m3fn) | 不可 (同左) |

WMMA 検証 (Triton コンパイル済み AMDGCN の grep):
R9700 の fp16 カーネルは `v_wmma_f32_16x16x16_f16`、
int8 カーネルは `v_wmma_i32_16x16x16_iu8` を実行 (真の WMMA)。
V620 は fp16 `v_dot2c_f32_f16` / int8 `v_dot4c_i32_i8` (WMMA/mfma 無し)。
torch 側は ISA を直接確認できないため backend 記録のみ
(hipBLASLt。R9700 で int8 が動く= gfx1201 Tensile lib あり)。

Triton 注意 (再現用): ポインタ加算型の K ループ
(`a_ptrs += BK*s_ak`) は本スタック (triton 3.7 + ROCm 7.2) で
multi-tile K が誤計算 (single-tile は正確)。base から再計算する形で
fp16 丸めレベル一致を確認。int8 `tl.dot` は `out_dtype=tl.int32` の
明示が必須 (既定 fp32 のため acc 型 assert で落ちる)。
全形状で Triton correctness スポットチェック True (fp16 は allclose、
int8 は torch._int_mm または float 直積との exact 一致)。

## 1. 単発 expert GEMM: INT8/FP16 比 (R9700 vs V620)

su = fp16時間 / int8時間。R9700-torch は torch._int_mm / torch.matmul、
triton は同一 naive カーネル同士の比。V620-ext は既存
`doc/moe_int8_breakdown.md` §B (ext igemm/hgemm) の参考値。

q30b gu (2048x768):

| R | R9700 torch | R9700 triton | V620 triton | V620 ext (§B) |
|---|---|---|---|---|
| 8 | — (M>16制約) | 0.98x | 1.34x | ~1.4x |
| 16 | — | 0.95x | 1.34x | ~1.4x |
| 32 | 0.95x | 0.97x | 1.11x | ~1.3x |
| 64 | 1.10x | 0.96x | 1.24x | ~1.3x |
| 128 | 1.14x | 0.96x | 1.22x | 1.05x |
| 256 | 1.16x | 0.98x | 1.25x | ~1.5x |
| 512 | 0.92x | 1.01x | 1.32x | ~1.2x |
| 1024 | 0.98x | 1.09x | 1.82x | ~1.1x |
| 2048 | 0.97x | 1.14x | 1.85x | 1.85x |
| 4096 | 0.92x | 1.21x | 1.81x | — |

q30b d (768x2048):

| R | R9700 torch | R9700 triton | V620 triton | V620 ext (§B) |
|---|---|---|---|---|
| 32 | 1.18x | 0.98x | 1.50x | 0.9〜1.0x域 |
| 128 | 1.15x | 0.98x | 1.47x | 同上 |
| 512 | 0.87x | 1.06x | 1.33x | 1.12x |
| 1024 | 0.90x | 1.07x | 1.46x | 1.28x |
| 2048 | 0.79x | 1.12x | 1.55x | 1.48x |
| 4096 | 0.81x | 1.17x | 1.41x | — |

(R=8/16 の R9700 torch は M>16 制約で欠測。R=64/256 も同傾向
1.16x/0.97x。V620 triton 全行 0.88〜1.55x。)

Flash Next gu (2560x640):

| R | R9700 torch | R9700 triton | V620 triton | V620 ext (§B) |
|---|---|---|---|---|
| 32 | 1.05x | 0.96x | 1.23x | 0.87〜0.93x域 |
| 40相当域 (R=32/64) | 1.05〜1.06x | 0.96x | 1.23x | 同上 |
| 128 | 1.16x | 0.97x | 1.23x | 1.11x |
| 160相当域 | — (R=128/256 で 1.16x/1.03x) | 0.97x | 1.23〜1.30x | ~1.1x推定 |
| 1024 | 1.13x | 1.07x | 1.51x | 0.83〜0.86x (崖) |
| 2048 | 1.03x | 1.11x | 2.58x | 1.09〜1.12x |
| 4096 | 1.00x | 1.19x | 1.79x | — |

Flash Next d (640x2560): R9700 torch 0.82〜1.05x (R=1024: 0.88x、
R=2048: 1.01x、R=4096: 0.82x)、R9700 triton 0.99〜1.19x、
V620 triton 0.71〜1.71x、V620 ext 0.91〜1.34x。

読み: R9700 の torch INT8 は MoE 実域 (R=10〜512) で 0.8〜1.2x、
R≳1024 でも 1.0x 前後 (down 大では 0.8x 台に悪化)。
WMMA 実働の naive Triton でも MoE 域は 1.0x 割れ〜等倍、
R=4096 でようやく 1.2x。V620 ext の「R=2048 で 1.85x」は
R9700 torch では再現しない (0.97x)。

## 2. dense 27B (INT8/FP16 比 + 到達率)

| 形状 | M | R9700 torch su | R9700 triton su | V620 triton su | R9700 fp16到達 | R9700 int8到達 | V620 fp16到達 |
|---|---|---|---|---|---|---|---|
| gate_up | 512 | 0.63x | 1.24x | 1.87x | 121.7 TF | 76.3 TOPS | 30.5 TF (75%) |
| gate_up | 2048 | 0.60x | 1.27x | 1.83x | 121.2 TF | 72.8 TOPS | 31.4 TF (77%) |
| gate_up | 8192 | 0.59x | 1.57x | 1.76x | 122.8 TF | 72.8 TOPS | 32.5 TF (80%) |
| down | 512 | 0.63x | 1.26x | 1.84x | 112.2 TF | 70.3 TOPS | 18.7 TF (46%) |
| down | 2048 | 0.61x | 1.36x | 1.85x | 118.9 TF | 72.7 TOPS | 21.0 TF (52%) |
| down | 8192 | 0.58x | 1.44x | 1.81x | 125.4 TF | 73.0 TOPS | 32.0 TF (79%) |

到達率の分母: V620 FP16 40.55。R9700 は二説併記:
fp16 122 は 95.7説の 128% (=上限超過。よって 95.7説は dense 上限
ではなく保守値か別条件の値) / 191説の 64%。
int8 73 は 383説の 19% / 766説の 10%。
(fp16 122 は CUDA イベントと wall-clock の二経路で一致確認:
同一プロセス再測定で event 2.87ms / wall 2.83ms。)

読み: dense で torch fp16 が 120TFLOPS 級に伸びる一方、
torch int8 は 73TOPS 止まりで **0.6x と逆転**。
V620 の dense-ideal 1.8x (ext) / 1.8x (triton) と真逆の絵。
naive Triton 同士なら R9700 でも int8 が 1.2〜1.6x 勝つ
(WMMA の素性は出る) が、torch の int8 経路 (hipBLASLt gfx1201)
は fp16 経路に全敗する。ハード上限ではなく利用可能 backend の成熟度
の問題とみるのが妥当 (ただし ext は gfx1201 未ビルドのため比較外)。

## 3. grouped 形 (B=8 均一、実 rows/expert。fp16 bmm 対 int8 expert-loop)

| case | r | R9700 fp16 bmm | R9700 int8 loop | 比 | V620 fp16 bmm | V620 int8 loop | 比 |
|---|---|---|---|---|---|---|---|
| q30b gu (T=2048) | 128 | 0.074ms (43TF) | 0.257ms (12.5TOPS, _int_mm) | 0.29x | 0.252ms | 0.529ms (triton) | 0.48x |
| q30b gu (T=8192) | 512 | 0.161ms (80TF) | 0.272ms (47TOPS, _int_mm) | 0.59x | 0.463ms | 0.685ms (triton) | 0.68x |
| flash gu (T=2048) | 40 | 0.065ms (16TF) | 0.253ms (4.1TOPS, _int_mm) | 0.26x | 0.195ms | 0.623ms (triton) | 0.31x |
| flash gu (T=8192) | 160 | 0.104ms (40TF) | 0.256ms (16TOPS, _int_mm) | 0.41x | 0.311ms | 0.631ms (triton) | 0.49x |

読み: expert-loop 型の INT8 は両 GPU で fp16-bmm の半分以下。
R9700 の int8 loop 時間は r に依らず ~0.25ms 定数
(=1 expert あたり ~30μs の launch オーバーヘッド支配)。
grouped を速くするには融合 grouped カーネルが必須で、
per-expert 呼び出しのままでは WMMA は寄与しない。

## 混雑・再現条件

- R9700 測定中: gsq_hessian が 24.5GB 常駐 (計算は idle、
  測定前後で GPU 使用率 4%。自プロセス以外に計算プロセス無し)。
  測定直前まで他 agent の attention bench
  (bench_attn_i8f8_cached.py) が動作していたため、それが終わり
  GPU 使用率 4% を確認してから開始。VRAM は最大でも dense 8192 の
  int32 出力 571MB + 入出力で 1.5GB 未満 (空き ~8GB 内)。
- V620 測定中: host GPU0/2 とも使用率 0%、他プロセス無し。
- 小 M (R≲128) は launch 支配で min/max の開きが大きい
  (例 R9700 q30b gu R=32: fp16 0.022〜0.196ms)。表は中央値。
  比率の結論は R≳256 の安定域でも同じ。
- R9700 はクロック立上りが遅い (idle SCLK 41MHz)。
  別プロセスの wall 10反復では 9.7ms (37TF) まで落ちるが、
  100反復 warm では 6.4ms、本 sweep (数分連続負荷) では 3.0ms
  (121TF) に収束。fp16/int8 は同一 hot 状態で交互計測のため
  比率への影響は無い。到達 TFLOPS は hot 時の値。

## 結論 (問いへの一文回答)

- MoE: R9700 でも結論は変わらない。実 rows/expert 域
  (q30b 32/128/512、Flash Next 10/40/160) の INT8/FP16 は
  torch で 0.8〜1.2x、WMMA 実働の naive Triton でも 1.0x 前後。
  V620 の 1.0〜1.2x と同じ絵であり、WMMA は MoE の律速
  (小行数・launch 支配) を解消しない。
- dense: むしろ逆転する。V620 の 1.8x (ext/triton) に対し、
  R9700 の torch INT8 は dense 大で **0.6x と FP16 に完敗**
  (fp16 120TFLOPS 級 vs int8 73TOPS)。hipBLASLt gfx1201 の
  int8 経路の成熟度の問題とみられる (naive Triton 同士では
  int8 が 1.2〜1.6x 勝つため、素子自体は生きている)。
  よって「R9700 なら MoE の INT8 化が報われる」にはならない。
  融合 grouped カーネル無しでは per-expert loop が 0.3〜0.6x と
  さらに悪化する点も V620 と共通。
