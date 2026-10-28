# W8A8 INT8 プリフィル (feat/w8a8-int8-prefill)

## 目的

Radeon PRO V620 (gfx1030, RDNA2) x2 上での EXL3 プリフィル高速化。
RDNA2 には WMMA がなく、利用可能な INT8 ドット積は `udot4`
(unsigned のみ; `sudot4`/`sdot4` は gfx1030 に存在しない) である。
EXLR8 (FP8 テンソコア向け) の発想を RDNA2 に移植する:
トレリス復号値を INT8 グリッドに載せ、活性を INT8 化し、
DP4A レートで GEMM する。

## 結論 (先に)

- 精度: 達成。修正後の W8A8 パスは通常 3.05bpw パスと KL で区別不能
  (macro KLD 0.1264 対 0.1286、差は候補パスノイズ ~0.015 以下)。
  INT8 グリッド自体 (重み・活性とも) の誤差はほぼゼロだった。
  当初の +0.058 後退の原因はグリッドではなく、
  スケールを fp16 の svh コピーに折り畳んだ丸め誤差であり、
  fp32 エピローグスケールで解消した。Phase 2 (re-encode) の精度上の
  根拠はなくなった。
- 速度: 未達。V620x2・TP2・AR で 512 tok 0.83x / 2048 tok 0.97x /
  8192 tok 1.00x。GEMM 単体は 1.3〜2.7x 速いが、
  呼び出しごとの量子化オーバーヘッドと非対象域 (MoE・fused 域) が
  勝ち分を消す。Phase 2 (約 18.5h の re-encode) は速度の根拠もないため
  実施しない (親の判断どおり、速度が出ない場合は re-encode 前に相談、
  という条件が満たされなかった)。

## 設計 (Phase 1)

`LinearEXL3.reconstruct_hgemm` に `EXL3_INT8_PREFILL=1` で有効化する経路:

1. 入力 Hadamard (回転基底、非 fused パスと同じ)。
2. 内部トレリス重みを per-output-channel 対称 int8 グリッドへ丸め
   (呼び出しごとに再構築; 全 dense 線形の保持は 28GB/rank モデルの
   横で数 GB になるためキャッシュしない)。
   スケールは fp32 のままエピローグで掛ける (下記「svh8 事件」参照)。
3. 活性を行ごと absmax / 127 で int8 化 (per-token)。
4. int8 GEMM (C は int32、exact 累積)。`ext.igemm_recon`
   (rocBLAS int8 GEMM_EX) があればそれを、なければビット正確な
   エミュレーション (`_exact_int32_gemm`: K=512 ブロック fp32 積 +
   fp64 累積、TF32 無効)。
5. 行スケール・列スケールを戻し、元の svh で出力 Hadamard (+ bias)。

対象は rows < 1024 の非 fused 域 (fused original-basis パスは rows >= 1024
で既に Hadamard を消しており、そちらは当面 fp16 のまま)。
グラフキャプチャ (`capture`) と sliced lm_head は fp16 フォールバック。
MoE (fused `exl3_moe` / batched tier) は対象外。
`EXL3_INT8_ABLATE=w|a` で重み側・活性側の単独誤差を分離できる
(評価用。`s` は本パスに統合済みのため廃止)。

## バックエンド: `igemm_recon`

`exllamav3_ext/quant/exl3_igemm.cu` (+ `.cuh`、`bindings.cpp` 登録、
`rocm/cuda_shim/cublas_v2.h` に `CUDA_R_8I`/`CUDA_R_32I`/
`CUBLAS_COMPUTE_32I`/`CUBLAS_GEMM_DEFAULT` を追加)。
row-major C = A @ B (int8 x int8 -> int32)、呼び方は
`hgemm_recon` と同じ (c を埋める out-param 式)。
`m.def("igemm_recon", ...)`。

- 正確性: GPU 上で CPU int64 参照とビット一致を確認
  (m,k,n) = (32,64,48)、(513,2560,5120)、(256,640,256)。
  `rocm_tools/rdna2/tests/test_igemm_recon_gpu.py`
  (`EXL3_GPU_TESTS=1` で実行)。
- GEMM 単体速度 (V620、fp16 `hgemm_recon` 対比):
  m=512,k=2560,n=5120 で 2.68x (2.53ms 対 6.79ms)、
  m=512,k=2560,n=2560 で 1.27x、m=2048 で 1.41x。
  ただし C が int32 (4B/elem) のため m が大きいほど
  トラフィックが重くなる。
- tp_run では 44 形状で engage (`backend=igemm` ログで確認)。

## 精度評価 (KL プロトコル)

HF `jyohukuchan/qwen38-flash-next-bf16-logits` (21 ケース、32,359 位置。
参照エンジン = exllamav3 BF16 重み、FP32 ロジット) に対する
`compare.py` の macro case-mean KLD を主指標とする。
候補ダンプは 1 ケース 1 フォワード (バルク、チャンクなし、temperature 1)、
行 i = トークン i 消費後の生 FP32 ロジット。BF16 参照の再ダンプはしない。
比較は macro case-mean (KLD・top1 とも。位置重み付き平均も併記)。

エンジン整合性チェック (通常 EXL3 3.05bpw パスを自前ダンプ):
21 ケース macro KLD 0.1286 / top1 0.8929 (XeStrata 3.05bpw 公開値
0.1291 / 0.9016 と一致)。wiki2-scored (truncated `[55:2103]`)
macro KLD 0.1540 / top1 0.8925 (turboderp 3.05bpw 公開値
0.1559 / 0.8920 と一致。候補パスノイズ実績 ~0.015 に対して差は 0.002)。

| 候補 | macro KLD | Δ | top1 (macro) | top1 (位置重み) |
|---|---|---|---|---|
| 通常 3.05bpw (W8A8 off) | 0.1286 | — | 0.8929 | 0.9009 |
| W8A8 emu (svh8 折り畳み) | 0.1864 | +0.0578 | 0.8692 | 0.8954 |
| W8A8 重みグリッドのみ | 0.1293 | +0.0007 | 0.8924 | 0.9007 |
| W8A8 活性グリッドのみ | 0.1252 | -0.0034 | 0.8930 | 0.9010 |
| W8A8 fp32 スケール配置 (emu) | 0.1264 | -0.0022 | 0.8921 | (同左系) |
| W8A8 修正パス (igemm, 最終) | 0.1264 | -0.0022 | 0.8921 | (同左系) |

読み方: 重みグリッド単独 (+0.0007) も活性グリッド単独 (-0.0034、
ノイズ範囲内) も通常パスと区別不能。両方合わせた emu が +0.058
悪化したのはグリッドの相互作用ではなく、スケール配置の問題だった
(次節)。修正パスは通常パスと区別不能であり、W8A8 化による精度 loss は
実質ゼロ。Phase 2 (int8-exact encoder) で取り除ける誤差は残っていない。

### svh8 事件: +0.058 の正体

最初の実装は重みスケールを svh の fp16 コピー (`svh8 = (svh*s).half()`)
に折り畳み、エピローグを通常パスと同一にしていた。
`EXL3_INT8_ABLATE=w/a` の分離で両グリッドが無罪と判明したため、
「全 int8 GEMM + fp32 でスケールを戻し + 元の svh」
(`EXL3_INT8_ABLATE=s`) を試したところ macro 0.1264 に回復。
犯人は系統的な per-channel ゲインバイアス
(fp16 丸め ~2^-11 が 48 層で複利-compound) だった。
教訓: スケールは fp32 のまま GEMM の外で掛ける。Hadarmard 側の svh は
オリジナルを使う。現在のデフォルトパスはこの構成。

なお乖離が出たのは 257/513 トークンのケースのみ
(129 トークン以下は small-row 別経路、1024 以上は fused 域で
どちらも fp16 のため全ダンプでビット一致)。

## 速度評価

V620 x2、TP2、AR、`use-per-device 28 28`、frozen random prompt
(3 長 x 4 反復、最初は warmup)。中央値 (timed 3 反復)。

| prompt 長 | baseline (fp16) | W8A8 igemm (初期) | W8A8 修正パス (最終) | 比率 (最終/baseline) |
|---|---|---|---|---|
| 512 (first-delivery) | 364.6 tok/s | 295.9 | 301.3 | 0.83x |
| 2048 (first-delivery) | 570.8 tok/s | 547.6 | 551.5 | 0.97x |
| 8192 (first-delivery) | 537.5 tok/s | 526.9 | 537.5 | 1.00x |
| 512 (time_prefill) | 373.5 tok/s | 301.1 | 306.7 | 0.82x |
| 2048 (time_prefill) | 576.6 tok/s | 552.2 | 556.0 | 0.96x |
| 8192 (time_prefill) | 538.7 tok/s | 528.0 | 538.6 | 1.00x |

内訳 (m=512,k=2560,n=5120、1 engaged call あたり):
a-quant 0.34ms + w-quant 0.93ms + igemm 3.13ms + i32*scale 0.11ms
= 4.51ms 対 hgemm 10.04ms。GEMM 部は 2.2x 速いが、
呼び出しごとの量子化 (特に重みの fp32 amax + round、約 0.93ms x
約 200 calls/プリフィル) が固定費としてのしかかる。
短い prompt ほど固定費比率が高く、512 で 0.83x まで落ちる。
8192 では GEMM 支配になるが MoE・attention・fused 域が fp16 のまま
残るため 1.00x で頭打ち。

## 正直な注意点

- この W8A8 パスは現状では速くならない。GEMM 単体の勝ち (1.3〜2.7x)
  は本物 (DP4A が効いている) だが、エンドツーエンドでは
  (a) per-call 量子化の固定費、(b) MoE experts (FLOP の約半分) が
  非対象、(c) rows>=1024 fused 域が非対象、の三重苦で相殺される。
- 勝つには (a) 重み量子化のキャッシュ/永続化
  (OOM で断念した `_w8` キャッシュの再設計)、(b) MoE の INT8 化、
  (c) reconstruct+quant+GEMM の融合カーネル、が必要。
  いずれも Phase 2 とは別口の大型開発である。
- 精度面は解決済み: 修正パスは通常パスと区別不能
  (macro Δ -0.002、ノイズ範囲内)。re-encode で精度が上がる余地はない。
- 65/129 トークン forward は small-row 別経路のため int8 対象外、
  1024 トークン以上は fused 域のため対象外。
  本パスの恩恵域は 256〜1023 行の dense 線形のみ。

## 次の手順 (親への提案)

1. Phase 2 (int8-exact encoder + Flash Next の re-encode、約 18.5h) は
   中止を推奨: 速度の根拠がなく、精度の改善余地もない。
2. 速度を本気で取るなら、MoE の INT8 化か、trellis-packed のまま
   DP4A するカスタム GEMM (upstream INT8 GEMV の GEMM 版) が本命。
   どちらも大型カーネル開発であり、別タスクとして見積もるべき。
3. 本ブランチは実験記録として保持。`EXL3_INT8_PREFILL` はデフォルト
   off のまま残し、本番挙動に影響しない。

---

## ラウンド 2: エンコード時 INT8 化 + 小規模 MoE 検証 (feat/w8a8-codebook)

ラウンド 1 の結論 (ランタイム量子化の固定費が勝ち分を消す) を受け、
EXLR8 の encode-time 着想だけを採用する。Flash Next の re-encode 前に
小さな MoE で形式と速度の両方を検証するのが目的である。
EXLR8 の多幅ストレージ (K2/K3/K4) と TP シャーディング則は対象外。

### フォーマット設計

(a) INT8-exact codebook: Viterbi 探索の内部でコードブック出力を
int8 グリッド `T = q * s0` に丸める。`s0` はエンコード時に決まる
per-tensor 定数 (`amax / 127`) で、モデル内に `*.s0` テンソルとして
保存する。ランタイムは重みスケールを計算しない (`s0` を `suh`/`svh`
または fp32 エピローグに折り畳み)。`s0 = 0` の旧モデルは従来動作の
まま (ビット同一を確認済み)。
プレーンカーネルと dense 特殊化カーネルの両方に丸めを実装した。
特殊化側は `s0 > 0` で例外にしていたが、9 箇所の decode 呼び出し
(`decode_3inst`/`decode_3inst_2`/LUT/state/出力書き込み) を wrap して
対応し、ガードを撤去した。

(b) 共有 hidden-side 回転 (MoE gate/up 用): 層ごとに 1 本の入力側符号
ベクトルを routed experts 間で共有し、トークンごとの expert 入力を
1 回だけ int8 化して active experts 全員で使い回す。per-expert の
中間側符号・スケールは EXLR8 と同様に格納ベクトルへ折り畳む。
設定フラグをテンソル/モデル config に保存し、旧モデルは従来動作。

### 検証モデル選定

Qwen3.5-35B-A3B (Flash Next と同族) ではなく Qwen3-30B-A3B を使う。
Qwen3.5 の BF16 (~70GB) は V620x2 (64GB) に載らず、exllamav3 の
2-GPU 分割読み込みも層 15 で失敗する (autosplit の上限チェックが
断片化に弱い)。Qwen3-30B-A3B は同族 (`qwen3_moe`、128 experts /
top-8) で BF16 (61GB) と既存 3bpw EXL3 が手元にあり、
コーパスの再トークナイズも通った。LFM2.5-8B-A1B / Gemma E2B/E4B は
実体が dense で MoE 検証に使えない。

量子化は ~3bpw (`-b 3.05 --hq -hb 5 -cb mul1`) で 3 通り:
(i) 通常 EXL3、(ii) `--int8-grid`、(iii) `--int8-grid --share-input-su`。

### カーネルゲート (新 `.so`、V620)

- `sweep_s0_shapes.py`: 全 (K=1..7 × plain/mcg/mul1) × 両 leg
  (plain / optimized) で fp16 往復偏差 `0.0`。`s0=0` は旧 `.so` (r2)
  と `qt_bitwise.py` 全行ビット同一。
- よってランタイム側の丸め実装は正しい。残る精度リスクは
  `s0` 丸めが `proxy_err` / 状態遷移に与える影響のみで、KLD で測る。

### 11:59 クラッシュ記録 (10/28) → 原因判明: 稼働中 `.so` への上書きコピー

2 本の convert (i: 層 7、ii: 層 2) が SIGILL (`invalid opcode` in
`exllamav3_ext`) で死亡。原因は自カーネルではなく、round 2 の拡張
ビルド (`-j12`) の成果物を `w8a8_lib/` へ `cp` したことが直接の引き金
(確定。以下すべて 10/28 の実測):

- 新 `.so` のリンク完了が 11:52:15 (`build/lib.../` の mtime)、
  `w8a8_lib/exllamav3_ext.cpython-312-x86_64-linux-gnu.so` への
  コピー完了が 11:59:23 (mtime)。
- カーネルログ: `11:59:23 traps: python[666913] trap invalid opcode
  ... in exllamav3_ext...so[2bf5ad1,...]`、続いて `11:59:43` に
  もう 1 本も**同一ファイルオフセット `2bf5ad1`** で fault。
  1 本目の fault がコピー完了と同秒、2 本が同一オフセットで
  20 秒差というのは、混成テキストページの特徴である。
- 機序: 2 本の convert は旧 `.so` を `PROT_EXEC` で mmap したまま
  実行中だった。`cp` は同一 inode を truncate-in-place するため、
  既フォールト頁は旧内容・未フォールト頁は新内容で読まれ、
  旧コードからのジャンプ先がゴミになる → `invalid opcode`。
  `mv` (rename) なら旧 inode が生き残るため起きなかった。
- AVX-512 コード生成説は否定: `setup.py` に `-march=native` はなく
  (hipcc 既定は generic x86-64、自動ベクトル化で AVX-512 は出ない)。
  `cpu/moe_mul1.cpp` の `detect_isa()` は Linux では
  `__builtin_cpu_supports` で正しくガードされ Zen2 では Avx2 tier に
  落ちる。`all_reduce_cpu_avx512.cpp` の呼び出し側も
  `is_avx512_supported()` ガード付き。真に AVX-512 命令が実行されたなら
  2 本が同一オフセット・同一秒で落ちる説明がつかない。
- 再発防止: ジョブ稼働中は `.so` を直接 `cp` 上書きしない。
  一時ファイル名に書いてから `mv` (atomic rename) で配置するか、
  ジョブ停止後に配置する。ビルドと convert の同時実行自体も避ける
  (load avg 18〜23 の高負荷も一因の可能性)。
副被害: host GPU0 に所有者なしの 26GB 使用中が残留
(KFD にプロセスなし、reclaim されず)。同 GPU 使用中の他者ジョブが
あるため `gpureset` は打たず、以降の作業は host GPU1
(コンテナ内 `HIP_VISIBLE_DEVICES=1`) で継続する。

### convert (i) の OOM 死と再開 (10/28、ラウンド 3)

- 親の 14:10 JST 確認の直後、**14:11:04** に docker memory cgroup
  (上限 56 GiB) が (i) を OOM-kill (host PID 707431 = コンテナ内
  57079、anon-rss 58 GB)。進行は層 29/48 (ログ 9064 行、
  `layers.29 ... experts.31.down_proj` で停止)。
- RSS は層ごとの feedback (`rss:`) によると単調増加ではなく
  鋸歯状: 例えば層 15 で 39.9 GB → 層 16 で 15.0 GB → 層 17 で
  46.2 GB → 層 22 で 53.6 GB → 層 23 で 19.8 GB → 層 29 で 10.4 GB。
  (`malloc_trim()` 後の報告値なので、山は実保持・谷は解放成功を
  意味する。) OOM は floor が高い時の層遷移スパイクで発生する模様
  (層 29 の floor 10.4 GB の直後の遷移で anon 58 GB に急騰し kill)。
  断片化由来のスパイクであり、真性リークではない可能性が高い。
  (`/work/rss-i.log` に 120 秒毎の sampler も記録中だが、層内位相を
  拾うため参考値。判定はログの `rss:` を使う。)
- 同一コマンドで 14:16 JST に再開 (コンテナ PID 58275、
  `build-q30b-i/ckpt` から層 29 を resume)。設定は一切変えていない
  (比較可能性のため `-cr/-cc` 既定のまま)。
- 教訓: 56 GiB 上限下では 30B 級の convert 単独でも OOM しうる。
  2 本同時実行は厳禁 (前例: (ii) が 12:52 に anon 34.6 GB で kill)。
  コンテナ上限の引き上げは不可 (ホスト RAM 逼迫) のため、
  OOM 死してもチェックポイントから resume すれば進行する
  (再開後は層あたり約 4 分)。再 OOM のたびに同一コマンドで再開する。
  `-cr` 縮小は (i) の比較条件を変えるためゲート判定と要相談
  (最終手段)。
- 15:35 JST、層 47・norm・lm_head まで完了した直後の compile 相で
  再 OOM (anon 58 GB)。出力 dir は未作成のままだった。
  直後に同一コマンドで 3 度目の resume → floor が低い状態で
  チェックポイント (層 47 後) から即 compile に入り成功。
  `/work/models/q30b-exl3-305-i` 完成 (12 GB、shard 2 本。
  既存 `qwen3-30b-a3b-exl3-3bpw` と同サイズ)。
  結論: OOM は確率的スパイクであり、resume 繰り返しで完走できる。
- (iii) を 15:45 JST に開始 (fresh `/work/build-q30b-iii`、
  `--int8-grid --share-input-su`、他条件は (i) と同一)。
  ログは `/work/convert-iii.log`。

### 精度評価 (TODO: ダンプ後に記入)

- BF16 参照: llama.cpp `dump_logits` (`-ngl 20`) で 21 ケース全成功、
  非有限要素 0。語彙幅は llama 側パディングで 151936 だが、
  `compare.py` は valid 列だけ参照するためそのまま比較可能。
- 候補: exllamav3 `dump` で (i)/(ii)/(iii) を単一 GPU ダンプ予定。
- クロスエンジン・ノイズフロア未測定 (通常 EXL3 候補を llama でも
  取るか、短ケースで exllamav3-vs-llama 比較が必要)。

| 候補 | macro KLD vs BF16 | 備考 |
|---|---|---|
| (i) 通常 EXL3 | TODO | ノイズフロアの基準 |
| (ii) int8-grid | TODO | |
| (iii) +共有回転 | TODO | ゲート対象 |

### 速度評価 (TODO: モデル完成後に記入)

`bench_prefill.py` (単一 GPU、ケース毎に新規ランダムプロンプト、
KV 再利用なし) で 512/2048/8192 tok/s を (i) fp16 vs INT8 で比較予定。
`EXL3_INT8_MOE_PROFILE=1` で内訳 (dense int8 / MoE int8 / 量子化固定費)
を取得する。

| 長さ | (i) fp16 | INT8 | 倍率 |
|---|---|---|---|
| 512 | TODO | TODO | TODO |
| 2048 | TODO | TODO | TODO |
| 8192 | TODO | TODO | TODO |

### ゲート判定 (TODO)

(iii) が (i) と KLD ノイズ範囲内で同等 かつ 2048+ で約 1.2x 以上なら
Flash Next の re-encode (~18.5h) を提案・実施。満たさなければ中止し
理由を報告する。
