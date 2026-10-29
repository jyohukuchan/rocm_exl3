# MoE INT8 (W8A8) が prefill で速くならない理由の定量分解

対象ブランチ `feat/w8a8-fused-prefill`、計測スクリプトは
`eval/prof/moe_breakdown_*.py`、生 JSON 併置。
GPU は V620 (gfx1030, RDNA2) のみ使用 (HIP 2/R9700 不使用)。
原則コンテナ内 `HIP_VISIBLE_DEVICES=0` (ホスト V620 #0, 空きを確認)、
Flash Next の層分割ロードのみ `0,1` 使用。

## 要旨 (先に)

q30b (128 experts / top8) の MoE 1 層あたり実測 (fp16 実前進 vs INT8 grouped
実コードパス、疑似 s0・タイミングのみ。数値の正しさは問わない):

| chunk T | fp16/層 | int8/層 | 層 speedup |
|---|---|---|---|
| 512 | 21.53ms | 19.18ms | **1.12x** |
| 2048 | 37.73ms | 30.01ms | **1.26x** |
| 8192 | 102.05ms | 86.66ms | **1.18x** |

dense 等価 FLOPs の GEMM 単体では int8 が 1.2〜1.9x 勝つにもかかわらず、
層全体では 1.1〜1.3x に潰れる。内訳 (T=2048/層、実配分):

- 再構成 (trellis→fp16): ~4.2ms … **fp16 と共有** (両パスとも毎回 decode)
- 重み int8 量子化 (`quant_w8_slab`): ~5.9ms … **int8 固有**
- GEMM 合計 (gu×2+d): int8 13.34ms vs fp16 batched 15.28ms = **1.15x**。
  dense-ideal (1.79x) の約 1/3 しか出ないのが要因 (1)
- 活性 quant + 融合 epilogue + dquant + gather/scatter 等: ~6.8ms … **int8 固有**
  (GEMM の 55% 増し。要因 (2) で GEMM 勝ち −1.94ms を消去して余りある)
- fp16 側の残差 (前後 Hadamard 群 + tier 残務): ~18.3ms … **fp16 固有の税金**。
  int8 の層勝ち (−7.7ms) は、GEMM 差 −1.94ms ではなく実質この置換 (−11.45ms) 由来

つまり「GEMM が速いから層が速い」のではなく、
「fp16 の Hadamard 往復が重いので、融合 epilogue を持つ int8 が残差で勝つ」、
そしてその勝ちを int8 固有の固定費 (w-quant、index_add ループ) が食う構造。

## 測定条件と caveat

- MoE モデル (q30b-305-i/iii、flash-next-3.05bpw) に int8-grid (s0) が無い。
  `_int8_moe_state` は s0 必須のため、そのままでは grouped INT8 パスに
  入らない。そこで **全 expert に定数 s0=0.02、共有 suh=gate0 の物を仮置き**
  する monkeypatch で **実 grouped コードパス** (`_forward_int8_moe_grouped`、
  `reconstruct_batch` / `quant_w8_slab` / `igemm_batched` /
  `gu_epilog_had` / `d_epilog_had` / `dquant_rowwise` の実カーネル) を
  実行し、**時間だけ**を測った。数値は無効。スケール不整合で出力絶対値は
  膨らむが、カーネル形状・メモリ量・実行パスは実物と同一。
- 計測対象カーネルは `5a96436 fix: index fused-epilogue s0 by entry, not
  expert (OOB read)` **適用後のもの** (ソースで確認)。gu/d 両 epilogue とも
  per-entry 索引。
- grouped-vs-prototype (同 fake s0): maxabs 1.1〜1.5、mean 0.116。
  出力スケール (meanabs ~23、fake s0 のスケール不整合で膨張) に対する相対で
  **約 0.5〜0.8%**。修正前の OOB (maxabs=449、コミットメッセージ記載) とは
  桁違いに小さく、half 丸めの結合順序差の範囲。**タイミングには無関係**
  (どちらのパスも同一カーネル群を実行)。
- d-path に未解決の残差があるという指摘 (factorial128 ~6.2) について:
  本測定は fake-s0 条件下での grouped-vs-proto 相対 0.5〜0.8% を確認した
  に留め、真の s0 付き MoE モデルでの絶対精度検証は未実施 (要 re-encode)。
- Flash Next (512 experts / top10、重み約 52GB + ngram 32GB) は単一 V620
  (32GB) に載らず、層分割 (`use_per_device=[28,28]`/`[31,31]`) でも
  VRAM 不足でロード不可。TP2 は別プロセス worker のため in-process 計測と
  両立しない。よって Flash Next は **実モデル前進なし**。GEMM・ステージは
  Flash 形状の合成テンソルで計測、重み再構成のみ q30b 実測レートからの
  換算 (明記)。ルーター歪みは q30b 実測のみ + 行数マッピングで対応付け。
- 32768 tokens の全層前進は lm_head 出力 (32768×151936×2B ≈ 10GB) で OOM
  のため未測定。GEMM レベルでは rows/expert=2048 点まで測定済み。
- 時刻は CUDA イベント中央値 (通常 20〜40 反復、5 warmup)。クロックは
  ウォームアップで安定化 (re-verify 済みの 2 点を除き安定)。
- fp16 層内訳の「残差」は実前進 − (同条件の実測 GEMM + 実測 recon) であり、
  本番 batch-recon tier 内部の計装ではない。内容物は §A のとおり。

## A. MoE 1 層のステージ分解 (q30b 実測、ms/層)

実前進 (ランダム ids。表の B は skew 込み実測、T=512: B=128 / T=2048:
B≈162 (cmax=160) / T=8192: B≈472) と、均一配分時の合成ステージ計測の対応:

T=2048 の int8 層 30.01ms の内訳 (profile 中央値、48 層):
`recon_g/u/d` 3.37/3.22/3.28 (= trellis decode + w-quant 融合 tick)、
`igemm_g/u/d` 4.68/4.64/4.10、`gather` 0.18、`epigu` 0.43、`silu` 0.24、
`dquant` 0.20、`epid` 0.60、`accum_prep` 1.50、`indexadd` 2.23。
合計 28.67ms (実測 30.01ms との差は act-had 0.12 + act-quant 0.15 +
sort/bincount 0.3 + router 0.12 + dispatch 残差)。

3 サイズの stage 対応表 (profile 中央値 / 合成均一計測。単位 ms/層。
`recon_*` tick は decode+w-quant 融合。合成側の `recon`/`wquant` 分離値は
`moe_breakdown_stages_q30b{,_8192}.json` の均一条件 (B=128/128/512)):

| stage | T=512 prof | T=512 synth(均一) | T=2048 prof | T=2048 synth(均一) | T=8192 prof | T=8192 synth(均一) |
|---|---|---|---|---|---|---|
| recon_g/u/d (decode+wq) | 2.72/2.57/2.64 | (1.12+1.56)×3相当 | 3.37/3.22/3.28 | (1.12+1.56)×3相当 | 9.20/8.98/8.96 | 3.59+6.17 / 3.58+6.16 / 3.55+6.16 |
| igemm_g/u/d | 2.74/2.74/2.71 | 0.97/0.97/0.93 | 4.68/4.64/4.10 | 3.04/3.10/2.60 | 13.13/13.10/11.84 | 11.67/12.06/10.92 |
| act had + torch-quant | — (0.23) | 0.054+0.181 | — (0.27) | 0.119+0.152 | — (1.40) | 0.782+0.621 |
| gather / sort / bincount | 0.11+0.31 | 0.053+0.17+0.14 | 0.18+0.30 | 0.137+0.15+0.16 | 0.48+0.44 | 0.271+0.23+0.20 |
| epigu / silu / dquant / epid | 0.15/0.05/0.09/0.21 | 0.081/0.020/0.155/0.208 | 0.43/0.24/0.20/0.60 | 0.368/0.055/0.149/0.481 | 1.44/0.77/0.73/1.91 | 1.386/0.672/0.652/1.785 |
| accum_prep / indexadd | 0.27/0.99 | — /1.08 | 1.50/2.23 | — /1.60 | 4.70/7.29 | — /7.03 |
| 層合計 (prof) | 17.98 (実 19.18) | — | 28.67 (実 30.01) | — | 82.52 (実 86.66) | — |

router/topk は全層合計で T=512: 3.07ms、2048: 5.73ms、8192: 15.25ms
(0.06〜0.32ms/層)。層全体の 1% 未満であり、要因分解上は無視できる。

fp16 層 (実測 21.53/37.73/102.05ms/層) の同条件 GEMM 比較 (T=2048 中央層、
実配分 B=155/cmax=160、合成テンソル):
int8 `igemm_batched` gu 4.70ms / d 3.93ms 対
fp16 `hgemm_batched` (padded、本番相当) gu 5.55ms / d 4.18ms、
参考 `hgemm_exact_group` gu 11.28ms / d 5.73ms。
すなわち **同パッド条件の GEMM だけ見ても int8 は gu 1.18x / d 1.06x**
(射影あたり。層合計は gu×2+d で int8 13.34 vs fp16 15.28ms = 1.15x)。
T=8192: int8 gu 14.02/d 12.52ms 対 fp16 gu 17.08/d 12.84ms (1.22x/1.03x、
層合計 40.6 vs 47.0ms = 1.16x)。

同等 FLOPs の dense GEMM との対比 (T=2048、q30b 等価 N=18432):
dense int8 3.53ms vs dense fp16 6.21ms = **1.76x (dense-ideal)**。
同 FLOPs の MoE int8 GEMM 合計 8.74ms (均一) / 13.4ms (実配分)。
FLOPs は等しいのに MoE は 2.5〜3.8x 遅い。これが要因 (1) の GEMM 本体分。

## B. 要因 (1): rows/expert (合成 GEMM、単発)

実 expert 形状での単発 `hgemm_recon` vs `igemm_recon` (中央値):

q30b gu (2048×768): R=8:1.41x、16:1.43x、32:1.30x、64:1.30x、128:1.05x、
256:1.51x、512:1.18x、1024:1.13x、2048:1.85x。
q30b d (768×2048): R≤256 で 0.89〜1.00x (int8 が勝てない)、512:1.12x、
1024:1.28x、2048:1.48x。
flash-next gu (2560×640): R≤64 で 0.87〜0.93x (int8 が遅い)、
128:1.11x、256:1.58x、512:1.51x、**1024:0.83〜0.86x (3 反復で再現する
タイルの崖)**、2048:1.09〜1.12x。
flash-next d (640×2560): 0.91〜1.34x (1024:1.34x、2048:1.29x)。

- 小行数 (R≤256) は起動支配。例: q30b gu R=32 で fp16 0.066ms /
  int8 0.051ms (到達 TFLOPS は fp16 1.5 / int8 2.0 程度に留まる)。
  R=8/16 では両者とも 0.15〜0.22ms に張り付き、DP4A の素子性能が出ない。
- **安定して 1.3x を超えるのは R≳1024〜2048 (単発)**。q30b の実 rows/expert
  (T×8/128) は T=512→32、2048→128、8192→512、32768→2048。
  すなわち T=2048 でも rows/expert=128 で speedup ≈1.0x。
  flash-next (T×10/512) は 10/40/160/640 で、T=8192 でも 160
  (単発 su ≈1.1x 程度)。
- バッチ (B=8 full slabs) では int8 単一カーネルが fp16
  (`hgemm_exact_group` ≒ expert ループと同等) に対し 1.4〜5.4x 勝つが、
  これは「fp16 側がバッチ化されていない」見かけの勝ち (下記)。本番 fp16
  (`hgemm_batched`) との比較は §A の 1.1〜1.2x が実態。
- 例外: q30b gu B=8/cmax=160 で int8 0.84〜0.86x (3 反復で再現)。
  `igemm_batched` の M=160 付近の弱点。実配分の cmax=160 と一致するため、
  実層の gu で踏む (T=2048 の int8 gu 4.70ms 対 fp16 5.55ms = 1.18x に
  留まる一因)。

dense-ideal (等価 FLOPs dense GEMM): q30b 1.21x (T=512) / 1.79x (2048) /
1.84x (8192)。flash-next 1.47x / 1.53x / 1.89x。
MoE の GEMM-only 到達点 (1.1〜1.2x) との差が、要因 (1) の定量値。

## C. 要因 (2): 変換オーバーヘッド (活性 quant + epilogue)

T=2048 q30b (実配分、/層): w-quant ~5.9ms + act-had 0.12 + act-quant 0.15 +
epigu 0.43 + dquant 0.20 + epid 0.60 ≈ **7.4ms** 対 int8 GEMM 13.4ms。
**GEMM の 55% 増し**。理想 (GEMM のみ、均一条件で 1.17x) → 変換込みでは
(13.34+7.4)/15.28 = 1.36x のコスト、すなわち GEMM 勝ち分 (−1.94ms) を
完全に相殺して余りある。層全体で 1.26x 勝つのは §A の fp16 残差
(Hadamard 群) の置換によるものであり、GEMM のおかげではない。

活性 quant の実装差: 入力側は torch 多パス (amax/div/round/clamp/convert、
0.15ms) で、down 側の融合 Triton (`dquant_rowwise`、0.20ms) と大差ない。
エピローグ融合 (`gu/d_epilog_had`) は各 0.4〜0.6ms で安い。
重いのは `quant_w8_slab` (B=162 slabs 計 ~5.9ms、重みバイト数支配で T 非依存
の固定費) と `index_add_` ループ (B 回逐次、2.2ms。fp16 本番の fused gather
に対する int8 固有のオーバーヘッド)。

## D. 要因 (3): ルーティング歪み (q30b 実テキスト)

`pride_prejudice_mod.txt` 先頭 (T=2048、48 層、他テキスト+オフセットでも
再現。ランダム対照 CV=0.085):

- per-expert 件数: 中央層 top5 = 1807/1805/1421/1306/1296 (全 16384 件中)、
  ゼロ expert 57/128。中央値 6.5、99%ile 1469。
- 層集約 (中央値): max 2003 (T=512: 490、T=8192: 7935)、CV 2.4〜2.5、
  B=164 (512: 92.5、8192: 471)、cmax=159〜160、padded 26080
  (512: 13651、8192: 75360)。均一比 padded: 512: 3.3x、2048: 1.6x、
  8192: 1.15x (cap=160 分割で均される)。
- GEMM への影響 (同 T、同中央層、合成テンソル):
  T=2048: int8 gu 3.09→4.70ms (1.52x)、d 2.57→3.93ms (1.53x);
  fp16-batched も gu 3.70→5.55ms (1.50x)、d 2.79→4.18ms (1.50x)。
  **パディングは両パス共通の税金**。T=8192: int8 gu 12.12→14.02 (1.16x)、
  fp16 gu 14.86→17.08 (1.15x)。
- 層全体 (grouped INT8 実コード、均一 sel vs 実 sel、同 y):
  T=512: 12.82→17.71ms (**1.38x**)、T=2048: 21.82→30.52ms (**1.40x**)。
  T=8192 は in-script 計測が OOM (モデル 19GB 常駐下の 472-entry slab)。
  GEMM レベルでは 1.15〜1.16x。
- 歪みは int8 固有ではないが、int8 側の per-entry 逐次処理
  (`index_add_` ループ、B 増で増大) には余計に効く。

flash-next の実配分は未測定 (VRAM 制約、§caveat)。rows/expert 対応のみ:
T=512→10、2048→40、8192→160。T=8192 でも q30b の T=2048 (128) 並み。

## E. Waterfall 帰属 (q30b、/層)

T=2048、実配分。fp16 実測 37.73ms → int8 実測 30.01ms (Δ −7.7ms):

1. fp16 GEMM (batched padded、実配分): 15.28ms (gu×2+d =
   2×5.55+4.18)。dense-ideal 1.79x なら −6.74ms のところ、MoE 行数では
   1.15x (int8 13.34ms) で −1.94ms に縮小。
   → **要因 (1): −4.8ms 相当の勝ち消し** (GEMM 単体の理想比)。
   (内訳: 小行数化で 1.79x→1.15x。M=160 の int8 崖を含む)
2. 歪み分離: 均一なら int8 GEMM 8.74ms のところ実配分 13.34ms (+4.60ms)、
   fp16 も 10.18→15.28ms (+5.10ms)。両パス共通のパディング税。
   → **要因 (3): 層全体で均一 21.8ms → 実 30.5ms の 1.40x** (grouped 実測)。
3. 変換: w-quant 5.9 + act-quant 系 0.5 + 融合 epi/dquant 1.2 ≈ +7.4ms
   (int8 GEMM の 55% 増し)。recon 本体 ~4.2ms は fp16 と共有。
   → **要因 (2): GEMM 勝ち分を単独では消去**
   (13.34+7.4=20.7 vs fp16 GEMM 15.28 で 1.36x コスト)。
4. 相殺項: fp16 残差 18.25ms (前後 Hadamard 群 + tier 残務) → int8 側の
   対応物 ~6.80ms (gather/epi/silu/dquant/accum/indexadd/router 等)。
   → **−11.45ms。int8 の層勝ちは実質すべてここ**。

検算: 37.73 − 1.94 (GEMM) + 5.67 (w-quant 正味) − 11.45 (残差置換)
= 30.01 ✓。

T=512: fp16 21.53 → int8 19.18 (Δ −2.35)。recon 3.3 + wquant 4.7 (固定費
8.0ms が層の 42%) + int8 GEMM 6.8 vs fp16 (GEMM 7.8 + 残差 10.4) で薄勝ち。
(実配分 B=79 の同条件 GEMM 比較、§D。均一なら int8 2.9 vs fp16 3.5ms。)
T=8192: fp16 102.05 → int8 86.66 (Δ −15.4)。recon 10.7 + wquant 18.5
(固定費 29ms、層の 34%) が重いが、fp16 残差の大きさで勝ち残る。

要因の固有/共有整理:
- int8 固有: w-quant、act-quant、融合 epilogue の dequant 演算、
  per-entry `index_add_` ループ、M=160 付近の `igemm_batched` 崖。
- 共有: trellis decode (recon_batch)、router/topk、sort/gather の機構、
  パディング (cap 分割含む) の行数増。
- fp16 固有 (int8 が消す側): 入力 pre-Hadamard + 出力 post-Hadamard 群
  (特に (rows, 2048) 形状での `had_r_128` の崖: 16384×2048 で 2.05ms 対
  16384×768 で 0.07ms。29x/2.7 倍要素。本番は `had_r_128_batch` のため
  参考値)。

## 再構成・Hadamard 時間 (実測)

実 s0 モデル (27B int8grid dense) での trellis decode (T 非依存、重み依存):

| 形状 | recon→fp16 | +quant→int8 (計) | quant 単独 |
|---|---|---|---|
| 5120×10240 | 0.126ms | 0.333ms | 0.372ms |
| 5120×6144 | 0.089ms | 0.154ms | 0.105ms |
| 6144×5120 | 0.089ms | 0.154ms | 0.105ms |
| 5120×17408 | 0.498ms | 1.100ms | 0.624ms |

quant は recon の約 1〜3x (重みが大きいほど比率増)。MoE では expert ごと
ではなく `reconstruct_batch` で B slab 一括 (q30b T=2048 均一: 1.1ms/射影、
B=128)。T 固定費ではなく **B (エントリ数) 固定費** である点に注意
(歪みで B が増えると recon も増える。T=8192 実配分で 9.0ms/射影)。

Hadamard (活性側 `had_r_128`、実 suh):
27B 5120 幅で T=2048: 0.12ms。q30b MoE 層 (H=2048): T=512: 0.054ms、
2048: 0.119ms、8192: 0.782ms。重み側 Hadamard は両パスとも独立カーネルを
持たず、int8 は融合 epilogue 内 (epigu 0.43/epid 0.60ms @T=2048)、
fp16 は post-had 群 (上記残差の一部)。

dense 比較 (同等 FLOPs、T=2048): dense-ideal GEMM int8 3.53 / fp16 6.21ms
(1.76x)。同 FLOPs の MoE int8 GEMM は 8.74 (均一) / 13.4ms (実配分)。
**FLOPs 等価でも MoE 分割で 2.5〜3.8x 遅い**。

## Flash Next (合成、均一配分。B=512 固定 (E=512))

| T | cmax | wquant gu+d | igemm gu/d | hgemm_batched gu/d | epigu/epid | indexadd |
|---|---|---|---|---|---|---|
| 512 | 10 | 5.89+5.87 | 3.79/3.73 | 3.81/3.50 | 0.087/0.311 | 5.42 |
| 2048 | 40 | 5.91+5.87 | 7.53/7.55 | 10.63/8.14 | 0.450/0.732 | 4.79 |
| 8192 | 160 | 5.94+5.84 | 19.03/14.04 | 18.77/14.31 | 1.457/2.781 | 11.75 |

- T=2048: gu 1.41x、d 1.08x。wquant 固定費 11.8ms が GEMM 勝ち分
  ((10.63−7.53)+(8.14−7.55)=3.7ms) を上回る。indexadd 4.8ms (B=512 逐次)。
- T=8192: gu 0.99x (勝ちなし)、d 1.02x。wquant 11.8ms + indexadd 11.7ms。
- recon は換算: q30b 実測レート ~400GB/s (fp16 出力換算) で gu/d 各
  ~4.2ms、計 ~12.6ms/層 (要実測、±30% 精度)。
- 実配分 (skew) は未測定。q30b 同様の CV を仮定すれば padded 1.2〜1.6x。

## 帰属まとめ (主張との対応)

1. **rows/expert**: 確認。単発 GEMM が安定 1.3x 超は R≳1024〜。実 rows/expert
   (q30b: 32/128/512、flash: 10/40/160) では 0.9〜1.2x。等 FLOPs dense
   1.8x との差が定量値。ただしバッチ化で 1.1〜1.2x は出る (0 ではない)。
2. **変換コスト**: 確認。GEMM の 55% 増し (T=2048 q30b)。単独では GEMM 勝ち
   を消去する。ただし層勝敗を決める主項は fp16 側 Hadamard 残差との置換。
3. **歪み**: 確認。実テキストで CV≈2.4、padded 1.15〜3.3x (T 小ほど大)、
   層時間 1.4x (T=512/2048)。本質は両パス共通のパディング税 + int8 の
   per-entry 逐次処理への波及。

## 再現 (スクリプト一覧、新規のみ)

- `eval/prof/moe_breakdown_gemm.py`: B 単発+バッチ+dense-ideal (モデル不要)
- `eval/prof/moe_breakdown_recheck.py`: 外れ値 3 反復 re-verify
- `eval/prof/moe_breakdown_layer.py`: fp16/int8/stages (q30b 単 GPU)
- `eval/prof/moe_breakdown_prof.py`: grouped 内 profile 行出力
- `eval/prof/moe_breakdown_routing.py`: 実テキスト sel 取得+統計+同条件 GEMM
- `eval/prof/moe_breakdown_gemm_from_json.py`: routing JSON から GEMM のみ
  (モデル不要、OOM 回避用)
- `eval/prof/moe_breakdown_synth_fn.py`: Flash 形状合成ステージ (モデル不要)
- `eval/prof/moe_breakdown_recon.py` / `recon2.py`: 実 s0 再構成コスト
- `eval/prof/moe_breakdown_equiv.py`: grouped-vs-proto (fake s0)
- JSON: `moe_breakdown_{fp16,int8}_q30b.json`、
  `moe_breakdown_routing_q30b_{512,2048,8192}.json`、
  `moe_breakdown_stages_q30b{,_8192}.json`、`moe_breakdown_synth_fn.json`、
  `moe_breakdown_recon_27b.json`

## commit

- 測定時点 HEAD: `be0a512`〜`c773745` (間の差分は docs のみ。
  カーネル・測定対象コードは同一)。worktree に他者の未 commit 変更なしを確認

