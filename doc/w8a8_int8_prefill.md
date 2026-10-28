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
- (iii) も層 12→13 遷移で OOM 死 (16:42 JST、anon 58 GB)。
  同一コマンド + `-r` で resume → 層 13 から継続。
  層 27〜31 で floor 54 GB 張り付きのまま遷移を耐え、
  層 31 で floor 14 GB に低下。その後順調に層 47・norm まで進むも、
  (i) と同じく lm_head 相で OOM 死 (19:05 JST、anon 58 GB)。
  再度 resume → 低 floor から compile のみ実行し成功。
  `/work/models/q30b-exl3-305-iii` 完成 (19:1x JST)。
  パターン確定: lm_head 相は毎回 OOM するため、
  head までの resume + compile のみの resume の 2 段階が必要。

### s0 ロード不具合と修正 (ラウンド 3、commit `3b2f1fd`)

(iii) 完成直後のベンチが `AssertionError: s0 must be positive` で
全滅した (全 s0 モデルが推論ロード不可 — round 2 はロード試験を
していなかった)。格納値は全 18,625 テンソルとも正であり、
原因は読み側: 推論ロードは `model_ls.py` が全モジュールを
`begin_deferred_load` で括り、s0 取得 (`opt(".s0", "cpu")`) が
遅延読み (後で埋まる未充填プレースホルダ) になっていた。
`LinearEXL3.__init__` は `s0.item()` を即時消費するため 0 を読み、
assert に落ちた。修正は 1 行:
`load_exl3` の s0 取得に `no_defer=True` を追加
(su/sv と同じ扱い。即時消費するテンソルは全て `no_defer` が付く、
という既存則どおり)。MoE expert も同一経路のため同時解決。
修正後は (iii) のロード・INT8 engage を確認済み。

### 精度評価 (記入済み)

- BF16 参照: llama.cpp `dump_logits` (`-ngl 20`) で 21 ケース全成功、
  非有限要素 0 (32,317 位置)。語彙幅は llama 側パディングで
  151936 だが、比較は共有先頭列のみ参照。
- 候補ダンプ: 自作 `/work/kl-q30b/dump_exl3_logits.py`
  (単一 GPU、1 ケース 1 フォワード、バルク・チャンクなし、
  生 FP32 ロジット、temperature 1)。(i)/(iii) とも 21 ケース成功、
  非有限要素 0。比較は自作 `/work/kl-q30b/compare_kld.py`
  (macro case-mean KLD + top1 macro/位置重み)。
- (ii) は未実施 (下記ゲートが速度で既決のため ablation 中止)。

| 候補 | macro KLD vs BF16 | top1 macro | top1 位置重み | 備考 |
|---|---|---|---|---|
| (i) 通常 EXL3 | 0.0524 | 0.9133 | 0.9310 | 基準 |
| (iii) +共有回転 | 0.0455 | 0.9201 | 0.9329 | ΔKLD -0.0069 |
| (iii) vs (i) 直接 | 0.0674 | 0.8996 | 0.9185 | 参考値 |

読み方: (iii) は BF16 参照に (i) より微接近しており、
形式変更による系統的劣化の証拠はない。ただし直接乖離 0.0674 が
両者の参照乖離 (0.045〜0.052) と同程度なのは、両候補の誤差方向が
直交気味 (独立な量子化ノイズ) であることを示す。
- ノイズフロア: **未測定**。exllamav3 BF16 (61GB) の 2-GPU ロードが
  `use_per-device [28,28]` + `max_chunk_size=128` でも
  `Insufficient VRAM in split for model and cache` で 2 回失敗し断念。
  参考: ラウンド 1 のクロスエンジン・ノイズ実績 ~0.015 (別モデル)。
  厳密な「ノイズ範囲内」主張は同モデル floor なしでは未検証だが、
  全ペア乖離が同程度 (0.045〜0.067) で方向に系統性なし。

### 速度評価 (記入済み)

`bench_prefill.py` (単一 GPU = host GPU1、ケース毎に新規ランダム
プロンプト、KV 再利用なし、warmup 2 + best-of-3)。

| 長さ | (i) fp16 | (iii) fp16 | (iii) INT8 | 倍率 INT8/(i) |
|---|---|---|---|---|
| 512 | 467 | 467 | 157 | **0.34x** |
| 2048 | 908 | 926 | 556 | **0.61x** |
| 8192 | 896 | 891 | 932 | **1.04x** |

(best 秒: (i) 1.096/2.256/9.139、INT8 3.253/3.686/8.793。
(iii) fp16 は (i) と一致 = s0 形式自体は速度中立。)
- engage: dense int8 は rows<1024 のみ (512 で 3 形状のみ engage、
  2048/8192 は fused 域で fp16 のまま)。MoE int8 は全 48 層で engage。
- 内訳 (`EXL3_INT8_MOE_PROFILE=1`): 共有回転 quant は ~0.1ms/層で
  成功しているが、expert 側が 512 行で約 57〜62ms/層
  (117〜128 segs、中央値 ~0.48ms/seg) → 48 層で約 3 秒。
  すなわち 128 本の小規模 per-expert GEMM の起動・gather 固定費が
  支配的で、fused fp16 MoE カーネル (約 20ms/層相当) の約 3 倍遅い。
  8192 行では expert 約 86〜88ms/層で fp16 とほぼ並び、dense int8 の
  寄与もないため 1.04x で頭打ち。
- 結論: MoE INT8 化は現実装では逆効果 (0.34〜0.61x)。
  勝つには per-seg 分割をやめる融合カーネルが必要で、別途大型開発。

### ゲート判定: 不通過 (no-go)

- 精度: 通過相当 ((iii) は (i) 対比で中立〜微良、ΔKLD -0.007)。
  ただし同モデル noise floor は未測定のため厳密な主張は留保。
- 速度: **不通過** (2048 で 0.61x、8192 で 1.04x。要求約 1.2x 以上に
  遠く及ばず)。
- よって Flash Next の re-encode (~18.5h) は**提案しない**。
  精度面の改善余地も (ラウンド 1 の結論どおり) ない。
- (ii) ablation はゲート既決のため実施しない。
  本ブランチは実験記録として保持。`EXL3_INT8_PREFILL` 系は
  デフォルト off のまま残し、本番挙動に影響させない。

---

## ラウンド 4: プロファイリングと融合 INT8 MoE (feat/w8a8-fused-prefill、進行中)

### 計測方針の変更: rocprofv3 不可 → CUDA イベント計測

- `rocprofv3 --kernel-trace` は本コンテナでハングする。
  自明な matmul (`python3 -c "import torch..."`) でも終了せず、
  別試行 (`prof_one.py` 2048、`/tmp/tiny_hip.py`) も 0% CPU のまま
  進行しなかった。SDK 初期化後の futex 待機で固まる模様。
- 代替としてイベント計測ハーネスを `eval/prof/` に追加
  (ライブラリ無改修。`torch.cuda.Event` による device time):
  - `prof_one.py`: 起動確認用 (warmup + N 回 full prefill)。
  - `agg_trace.py`: (rocprofv3 用だったが未使用。将来用に保持)
  - `stage_profile.py`: ブロック/attention/MLP(MoE) の forward を
    モンキーパッチで包み、層別・部位別の device time を集計。
    exllamav3 の Module は torch.nn ではないため hook 方式は不可、
    前方/後方ラッパ + イベント対で計測する。
  - `kernel_microbench.py`: 実テンソルでの単体計測
    (`reconstruct` / `reconstruct_had_slice` / `hgemm_recon` /
    `igemm_recon` / `had_r_128` / 量子化 / エピローグ)。
- 既知の落とし穴: コンテナ内 `HIP_VISIBLE_DEVICES=0` での初回
  ロードは約 4.6GB 読込後に進行停止 (ユーザ空間で 2 スレッドが
  空転、IO・GPU とも無変化。`kill -9` はコンテナ内 PID で行う)。
  ラウンド 3 の bench は `--device 1` で成功しているため、
  以降の計測は device 1 で行う。

### プロファイリング結果 (fp16 パス、単一 V620、(i) モデル)

`eval/prof/stage_profile.py` (forward のモンキーパッチ + CUDA イベント、
warmup 除く 3 回平均) の結果。壁時間からの tok/s はベースライン
(450/890/865) と一致し、計測の妥当性を確認済み。

| 長さ | 層合計 | attention | MoE (mlp) | norm 等 | 壁時間 | tok/s |
|---|---|---|---|---|---|---|
| 512 | 1115.6ms | 100.0 (9%) | 1012.7 (**91%**) | 3.0 | 1.128s | 454 |
| 2048 | 2263.9ms | 469.4 (21%) | 1786.5 (**79%**) | 8.0 | 2.308s | 887 |
| 8192 | 8999.7ms | 4175.3 (46%) | 4781.6 (**53%**) | 42.9 | 9.184s | 892 |

結論: 512/2048 では MoE が支配的 (9 割/8 割)。
8192 では attention が 46% を占め、MoE だけ速くしても
1.2x はぎりぎり (MoE 1.5x で全体 1.19x の計算)。

`eval/prof/kernel_microbench.py` (実テンソル単体、中央値):

| 操作 | 形状 | 時間 | 備考 |
|---|---|---|---|
| reconstruct (trellis 復号) | expert 2048x768 | 0.039ms | 復号は安い。律速ではない |
| hgemm fp16 | M=128,K=2048,N=768 | 0.241ms | int8 対比 1.14x に留まる |
| igemm int8 | 同上 | 0.211ms | 小 M・小 N では DP4A が効かない |
| 活性量子化 (amax+除算+丸め) | M=128,K=2048 | 0.146ms | expert 毎の固定費 |
| int32 エピローグ (scale) | M=128,N=768 | 0.057ms | — |
| reconstruct | dense 2048x151936 (lm_head) | 1.9ms | — |
| hgemm fp16 | M=512/2048,K=2048,N=151936 | 16.5/64.9ms | — |
| igemm int8 | 同上 | 7.8/30.9ms | **2.1x**。大 M・大 N では DP4A が効く |

読み: 現 MoE-int8 (per-expert 3 GEMM、約 0.48ms/seg) の病因は
復号ではなく、小規模 GEMM の効率と起動回数である。
128 experts x 3 launches = 384 launches/層が約 60ms/層を生む。
fused fp16 MoE は約 20〜37ms/層 (上表 1012/48=21.1、1786/48=37.2)。

### 設計: グループ化 INT8 MoE (v1)

- 1 層あたり GEMM を 2 発に集約する。gate/up は入力行を共有する
  ため重みを N 方向に結合 (N=1536) し 1 発、down を 1 発。
  トークンは expert 別に事前整列 (既存の order/counts を再利用)、
  パディング行は入力 0 埋め (int8 GEMM の 0 行は厳密に 0 を出す)。
- 重み復号は当面 per-expert のまま (`_recon_int8` を slab へ直書き。
  約 0.039ms x 384 = 15ms/層の上限。律速になれば
  `reconstruct_batch` 化 = v2)。
- 新規ネイティブは `igemm_batched` 1 本のみ
  (`cublasGemmStridedBatchedEx`、int8→int32。`hgemm_batched`
  と同型)。エピローグは slab 上のベクトル演算 + 既存
  `had_r_128_batch` で融合不要にする。
- 試算 (2048 tok): MoE FLOP/層 = 154GFLOP。fp16 fused は約
  4.2TFLOPS 実効。グループ int8 が 6TFLOPS に届けば MoE
  約 26ms/層 → forward 約 1.78s → 約 1150tok/s = **1.30x**。
  8192 では MoE 1.5x でも全体 1.19x のため、要実測。

### グループ化 v1 の結果 (正確性は達成、速度は未達)

- 正確性: per-expert 原型との比較でビット一致を確認
  (`factorial128`: 8/128 experts、パディング有無・重複選択有無の
  全条件で maxabs=0)。ただし到達までに要したデバッグ知見:
  - `tensor / python_float` と `tensor / tensor` では丸めが
    1LSB 異なる場合がある (スカラー除算は逆数乗算カーネルに
    なる模様)。重み量子化は原型と同じスカラー除算に統一した。
  - `reconstruct_batch` 自体は B=1〜128 で `reconstruct` と
    ビット一致 (無罪)。`igemm_batched` も全形状で一致 (無罪)。
  - 犯人は重み量子化の除算カーネル選択だった
    (expert 43 の up 重みが .5 境界に当たった)。
- 速度 (EXL3_INT8_MOE_PROFILE=1 付きの参考値。同期が入るため
  端末速度より遅めに出る): 512 で 216tok/s (原型 157 よりは速いが
  fp16 467 の 0.46x)、2048 で 454tok/s (原型 556 より遅い!)、
  8192 で OOM (gate エピローグで 2.2GB 確保失敗。
  実ルータは歪んでおり層 0 で cmax=378 / 平均 128。
  8192 では cmax が 1000 を超え slab が爆発 + 断片化)。
- 内訳 (2048/layer): recon_batch 本体は 1.1ms と速いが、
  量子化が律速。テンソル除算版 16.7ms x3、スカラー除算
  ループ版でも 9〜10ms x3 (128 発の小規模カーネルが非効率)。
  igemm は 2.5ms x3 で良好。すなわち「復号は安いが
  int8 化が高い」という構図。融合量子化カーネルが必要。
- v2 方針: (a) expert を 512 行単位に分割 (slab 上限。
  OOM 根治 + パディング削減)、(b) 除算+丸め+clamp+pack を
  1 パス化した融合量子化カーネル (Triton、目標 1ms)、
  (c) エピローグの一時物削減 (in-place 積算)。

### v2 結果と真の律速の特定

- v2 (cap=512): 512 で 216→398tok/s、2048 で 454→634tok/s。
  原型は上回るが fp16 (467/908) には届かず。8192 の OOM は解消。
- 位相内訳 (2048/layer): recon 8ms + igemm 8ms は良いが、
  エピローグ+had+silu+活性量子化+accum が 30ms 超。
  torch の要素演算が実効 64GB/s しか出ていない
  (ブロードキャスト積算・amax・変換の多パスが病的に遅い)。
  すなわち GEMM でも復号でもなく「エピローグのメモリ回り」が律速。
- v3 方針: エピローグ+had と活性量子化を Triton に融合
  (既存 `rocm_py/moe_prefill.py` の Hadamard を拡張)。
  ただし native had は半精度で途中丸めするため、
  fp32 通しの融合カーネルとは最終 1LSB 級の差が出る
  (より高精度側。KLD で中立性を検証する)。
- 副次知見: 実ルータは強く歪む (2048 で cmax=374/平均 128)。
  パディング削減のため cap 既定を 512→160 に下げた
  (634→793tok/s @2048)。

### v3 結果 (融合エピローグ)

- 融合カーネル 3 本 (`rocm_py/moe_int8.py`):
  `gu_epilog_had` (int32→scale→had 5 パス→1)、
  `dquant_rowwise` (amax+量子化 5 パス→2)、
  `d_epilog_had` (同 5 パス→1)。
  いずれも単体で torch 参照と一致 (had 融合は
  native の半精度途中丸めを省くため最終 1LSB 級の差。
  より高精度側であり KLD で中立性を検証する)。
- 速度 (iii, grouped, 単一 V620):
  512 で 441tok/s (fp16 467 の 0.94x)、
  2048 で 978tok/s (fp16 908 の 1.08x。初の fp16 超え)、
  8192 で 898tok/s (fp16 896 と互角)。
  目標 (1.2x @2048/8192、512 非悪化) には未達:
  2048 で +20%、512 で +6% が足りない。
- 残り時間の使い道: (a) gather マップのベクトル化
  (slice-copy ループ ~2ms/layer 削減。実装済み、未計測)、
  (b) 8192 は attention が 46% のため MoE だけでは届かず、
  attention 側 (INT8 qkv + コア) が必須。
  (c) KLD による精度検証 (実施中: cand-iii-v3)。
- KLD 状態 (10/28 15:45Z 時点):
  `EXL3_INT8_PREFILL=1 EXL3_INT8_MOE_GROUPED=1` で
  `/work/models/q30b-exl3-305-iii` → `/work/kl-q30b/cand-iii-v3`
  をダンプ中 (プロセス生存確認、env 正当)。
  完了後に `compare_kld.py --refdir /work/kl-q30b/ref-llama
  --canddir /work/kl-q30b/cand-iii-v3` を実行し、
  (iii) 0.0455 との差を記録すること。
- 注意: ダンプの stdout リダイレクト先 `/tmp/kld.log`
  (コンテナ内) が `ls` で見えない。プロセスは生存・env 正当で
  出力は `/work/kl-q30b/cand-iii-v3/*.npy` に向かうため、
  進捗は outdir のファイル数で監視すること
  (`ls /work/kl-q30b/cand-iii-v3 | wc -l`、全 21 件で完了)。

### attention フェーズ計画 (MoE 完成後の次段。未着手)

- 背景: 8192 での attention 占有 46% のため 8192 ゲート
  (1.2x) には attention 側の高速化が必須。
- (1) attention バケットの三分離 (2048/8192):
  q/k/v/o 射影 vs コア (QK^T/softmax/PV。gfx1030 では
  `modules/attention_fn` の Triton パスと想定。どちらか確認) vs
  RoPE/norm/KV 更新。コアは causal FLOPs ≒ 2·n²·d·h_q/層
  (d=128、h_q=32) から到達 TFLOPS を算出し fp16 ピーク
  約 41TFLOPS と比較する。
- (2) 射影: q/k/v/o を dense INT8 パスに寄せる
  (大 M で igemm 2.1x の実績)。
- (3) コア: 効率が低ければ RDNA2 向け調整 HIP
  flash-attention (packed fp16 / v_dot2) と SageAttention 型
  INT8 QK^T (DP4A、ブロック単位 Q/K スケール、K 平均平滑化、
  PV は fp16 維持) を検討。causal ブロック skip は維持。
- 精度: INT8 attention の KLD は MoE 変更と分離して測定する。

### Flash Next 優先度プローブ計画 (attention カーネル投資の前に実施。未着手)

- 背景: 実目標 Qwen3.8 Flash Next は Qwen3-30B-A3B と構造が違う:
  48 層 = 線形 attention (Gated DeltaNet、n に対し線形。
  K head 16 / V head 48 ×128) 36 層 + full attention
  (q 24 head、kv 2 head、head_dim 256、partial rotary 0.25、
  QSA indexer 4 head×128。prefill が sparse/top-k か要確認)
  12 層 + MoE (512 experts top-10 + shared、moe_intermediate 640、
  hidden 2560) + 層 2 の PLE/ngram embedding。
  二次 attention cost は Qwen3-30B の約 37%
  (12×24×256 対 48×32×128) 一方 MoE/token は約 1.4 倍。
  ラウンド 1 では Flash Next prefill が 2048→8192 で
  571→538tok/s と attention 増加が小さい。
- 方針: 既存 Flash Next EXL3
  (`/work/models/qwen38-flash-next-orcarouter-exl3-3.05bpw-original-vision`。
  速度目的なので orcarouter 重みで可。ラウンド 1 と同じく 2×V620)
  に対し `stage_profile` (fp16 パス) を 2048/8192 で実行し、
  MoE / full-attention / GDN 線形 attention / indexer / PLE /
  other に bucket 化する (`architecture/qwen3_next.py` は
  TransformerBlock 下に GatedDeltaNet/Attention を置く構成で
  既存フック方針が使えることを確認済み)。
- 目的: 優先度付け (Flash Next では attention INT8 の寄与が
  小さく GDN カーネルが律速かもしれない)。

### レジューム記録 (10/28 16:00Z 頃。中断時はここから再開)

- ブランチ `feat/w8a8-fused-prefill`、legacy-fork に push 済み。
- v3 + gather ベクトル化の再ベンチ未実施
  (コマンドは評価節参照。GPU が KLD ダンプで占有中だった)。
- KLD: `cand-iii-v3` ダンプ完了待ち → `compare_kld.py` 実行。
- その後: attention 三分離 (2048/8192) → 射影 INT8 化 →
  コア方針決定 → Flash Next stage_profile → ゲート判定。
