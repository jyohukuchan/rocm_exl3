# INT8 QK / FP8 PV プリフィル attention (opt-in)

SageAttention2 型の量子化 attention を、既存の fp16
プリフィル経路の opt-in 代替として追加した。QK^T を INT8、
P・V を FP8 (E4M3) で計算し、蓄積は fp32 で行う。
デフォルトは無効であり、既存の量子化 KV キャッシュ
(`exllamav3/cache/quant.py`、`EXL3_QC_STAGING`、`_fns_qc` 系)
およびデフォルト fp16 パスの動作は変えない。

## 使い方

```bash
EXL3_ATTN_INT8FP8=1            # 新モードでプリフィル (デフォルト 0 = 従来通り)
EXL3_ATTN_INT8FP8_ABLATE=nosmooth,per-tensor-k,fp16pv  # ablation (カンマ区切り)
```

- 対象はプリフィル (`q_len > 16`、fp16、causal/非causal) のみ。
  ページドキャッシュ付きとキャッシュなし (bulk。K/V を一時ページに
  畳んで同一カーネルで実行) の両方に対応。デコード (`q_len <= 16`)、
  量子化キャッシュ経由 (`q_cache`)、varlen、sliding-window、
  softcap、sinks 付きの呼び出しは従来カーネルにフォールバックする。
- スイッチはプロセス内で動的に読むため、同一プロセス・同一セッションで
  fp16 ベースラインと新モードの A/B ができる。
- 公正な fp16 ベースラインとの比較には `EXL3_HGEMM_FP16_OUT=1` を併用する
  (GEMM の fp32 出力税を除く。`doc/w8a8_int8_prefill.md` Plan E 参照)。

## 実装場所

- `exllamav3/modules/attention_fn/int8fp8.py` — 本体。
  環境変数の読み替え (`int8fp8_enabled()`、`int8fp8_ablate()`)、
  torch リファレンス (`sdpa_int8fp8_ref`)、Triton カーネル
  (`_i8f8_prefill_kernel`、`_i8f8_block`)、ページドラッパー
  (`paged_attn_triton_prefill_int8fp8`)、dispatch 関数
  (`fn_triton_paged_attn_prefill_int8fp8`)。
- `exllamav3/modules/attention_fn/dispatch.py` — opt-in フックのみ追加。
  無効時は新関数が即 `None` を返し、従来の探索順・ヒント・キャッシュ更新は
  ビット単位で不変。
- `tests/test_attn_int8fp8.py` — 維持テスト。
- `eval/prof/capture_qkv_27b.py` — 27B 実 Q/K/V の採取。

## 設計 (a) on-the-fly: キャッシュは触らない

新しいキャッシュ層型は作らない (将来の選択肢 (b) として残す)。
fp16 キャッシュを読み、呼び出しごとにカーネル内で量子化する。
計算量だけが減り、メモリ・帯域の削減はない。

### K 平滑化とキャッシュの関係

キャッシュがあると単一のグローバルな K 平均はトークン追加で不安定に
なる。そこで平均はブロック単位 (K ブロック = `BLOCK_N` トークン) で
持ち、softmax 不変性 (`q·μ` は行内でキーに依らず一定) を使って
`q·μ` バイアスを INT8 ドットの後に足し戻す。K を保存側で平滑化して
持つ必要はなく、呼び出し側で完結する。無効化は `nosmooth`。

### スケールの粒度

- Q: クエリ行ごと absmax / 127。
- K: K ブロックごと (平滑化後の absmax / 127)。
  `per-tensor-k` ではウィンドウ全体の kv ヘッドごとの単一スケールに
  落とす (ablation 用。ホスト側で 1 パス追加のため低速)。
- P: クエリ行ごと max / 448 (E4M3)。
- V: K ブロック内のチャネル (head-dim 列) ごと max / 448 (E4M3)。
- P@V の蓄積は fp32 (fp16 蓄積では FP8 のレンジが潰れる。
  SageAttention2++ の注意点どおり)。

`fp16pv` では P@V のみ fp16 のままにして QK-INT8 化の単独誤差を分離できる。

### V620 (gfx1030) と R9700 (gfx1201) の扱い

- V620 には FP8 も WMMA もない。`PV_MODE 1` (INT8 プロキシ PV) で
  カーネル機構を検証し、FP8 の精度主張は torch リファレンス
  (`torch.float8_e4m3fn` ラウンドトリップ、正確な E4M3 グリッド) で行う。
- R9700 では `PV_MODE 2` (実 `tl.float8e4nv`) を使う。
  Triton が gfx1201 で `tl.dot` を何に落とすか
  (int8xint8→int32、fp8e4xfp8e4→fp32 の AMDGCN と `v_wmma` の有無) は
  速度評価節に記録する。

## 精度

条件: Qwen3.8-27B (`qwen38-27b-exl3-305-int8grid`)、
`EXL3_HGEMM_FP16_OUT=1` の公正 fp16 ベースライン対比。
新モードはプリフィル (bulk / チャンクド) のみ、デコードは従来通り。

### KLD (cases-v1、9 ケース、65〜1025 rows)

`eval/prof/dense_kld_dump.py` で両モードをダンプし、
`kl-q30b/compare_kld.py` で fp16 参照→候補の KL
(macro case-mean、温度 1)。新モード発動は dispatch カウンタで確認済
(16 full-attention 層すべて。`window_size=-1` をウィンドウなしと
扱う修正の前は全層辞退して KLD=0.0 になっていた)。

| 候補 (対 fp16 attention) | macro KLD | macro top1 | 位置重み top1 |
|---|---|---|---|
| 新モード default | 0.0009 | 0.986 | 0.992 |
| + `nosmooth` | 0.0007 | 0.989 | 0.992 |
| + `fp16pv` | 0.0006 | 0.995 | 0.995 |

読み方: 量子化ギャップ (同系 3.05bpw 対 BF16 で 0.0767〜0.1286) の
約 1/85、INT8-GEMM の delta (0.0025) の約 1/3。
Plan D/E の neutral ゲート (mean 0.0009 / top1 99.5%) と同等で neutral。
`fp16pv` が最良 = 誤差の主因は PV の FP8 化 (QK の INT8 化ではない)。
`smooth` の有無はこの長さでは中立 (下の単体評価と整合)。

### 単体評価 (torch リファレンス対 fp16 SDPA、bottom-right マスク)

| 条件 | default rel max | fp16pv rel max | 備考 |
|---|---|---|---|
| ランダム、Tq64/Tk1024、8/2 頭、D256、causal/非causal | <0.08 (実測 ~0.03) | <0.02 | `tests/test_attn_int8fp8.py` |
| 27B 形状、Tq128/Tk4096、24/4 頭、D256、causal | <0.08 (実測 ~0.04) | — | 同上 |
| 実 Q/K/V (27B の 16 full-attention 層、512 rows 採取) | worst-rel 0.046 (layer 12) | meanabs 0.0118→0.0045 | `capture_qkv_27b.py` + 同テスト |

実データ 16 層: default の worst-rel 0.046 (出力スケール ~50 に対する
maxabs 2.31)、平均 meanabs 0.0118。`fp16pv` で meanabs は 0.0045 まで
下がる (PV が支配的)。`nosmooth` (worst 0.038、mean 0.0125)、
`per-tensor-k` (worst 0.055、mean 0.0132) と比べ、per-block + smoothing
の default が平均誤差で最良。worst-element は外れ値チャネル依存で
層により変動する。

### 長文 KV (実 K をタイリングして 8K/32K/128K、クエリ 128 行)

128 行クエリ over 長文 KV での新モード対 fp16:

| Tk | default rel | fp16pv rel | nosmooth rel | kernel-pv1 rel |
|---|---|---|---|---|
| 8192 | 0.0266 | 0.0219 | 0.0263 | 0.0211 |
| 32768 | 0.0266 | 0.0219 | 0.0263 | 0.0222 |
| 131072 | 0.0266 | 0.0219 | 0.0263 | 0.0226 |

長さで誤差は発散しない (行ごと・ブロックごとの適応スケールは
長さ不変のため)。注意: タイリング反復のため同一 worst-element が
並ぶ点は割り引いて見ること。カーネル (pv1 プロキシ) も同等。
128K ページド (262MB×2) でもインデックス異常なし。

### 長文脈 end-to-end (needle / 32K、V620、Generator 経路)

`eval/prof/needle_attn_32k.py`: チャンクドプリフィル (2048、fp16 ページド
キャッシュ) + greedy デコード。デコードは設計どおり両モードとも fp16。
`EXL3_HGEMM_FP16_OUT=1`。V620 単体 (32K コンテキスト+キャッシュで
約 20GB 使用)。

| ケース | fp16 | 新モード | 一致 |
|---|---|---|---|
| 32K needle (30 decode) | passkey 取得 ○ | passkey 取得 ○ | 先頭 60 文字同一、top1 22/30 |
| 2K needle (80 decode) | passkey 取得 ○ | passkey 取得 ○ | 先頭 200 文字同一、top1 80/80 |

32K の 30 デコード位置の内訳: top1 不一致 8/30 はテンプレート連鎖
(`<|im_start|>` 等のランブリング) でのカスケードで、意味内容
(passkey 本体) は両モード同一。2K では 80/80 完全一致。
デコード logit の KLD は未計測扱いとする (比較ヘルパーが +inf
パディング列で全行 nan→0 化したため。top1 を主指標とする)。

壁時間 (参考、デコード fp16 共通のためプリフィル差は薄まる):
32K total 127.7s (fp16) / 139.4s (新モード)、2K total 9.7s / 9.5s
(fp16 が常に先行=コールド実行のため、厳密な A/B ではない)。

## 速度 (R9700)

条件: gfx1201 (RDNA4、32SM、34GB) を `ROCR_VISIBLE_DEVICES` で単独可視化した
使い捨てコンテナ (同一イメージ `rocm-exl3-phase2-env:tested`) で、
`eval/prof/bench_attn_int8fp8.py` (B=1、Tq=1024、24/4 頭、D256、causal)。
fp16 ベースライン = リポジトリ内 Triton プリフィル。
同一プロセス内で交互に 7 反復×3 ラウンド、中央値を報告。
測定中の R9700 はアイドル (利用率 0%、他プロセスなし) を確認して実行。
VRAM は 2GB 未満。

| Tk | fp16 | i8f8-pv0 (QK INT8 + PV fp16) | i8f8-pv2/auto (QK INT8 + PV FP8) |
|---|---|---|---|
| 8192 | 4.82ms | 28.77ms (0.17x) | 35.42ms (0.14x) |
| 32768 | 19.12ms | 115.21ms (0.17x) | 140.85ms (0.14x) |

(各 3 ラウンドの中央値、ばらつき ±1〜2%。131072 も同傾向 (77.3 / 463 / 408ms)。
正確性は同条件で rel ~0.01〜0.04 (ランダムデータ)。)

内訳の読み方: fp16→pv0 の差が QK の INT8 化+量子化オーバーヘッド、
pv0→pv2 の差が PV の FP8 化。PV を FP8 にするとむしろ遅くなる
(28.8→35.4ms @8K)。

### Triton が gfx1201 で何を生成するか

コンパイルキャッシュの AMDGCN を確認 (`TRITON_CACHE_DIR` の
`_i8f8_prefill_kernel.amdgcn`):

- int8×int8→int32 の `tl.dot` → `v_wmma_i32_16x16x16_iu8` (128 発)。
- fp8e4×fp8e4→fp32 の `tl.dot` → `v_wmma_f32_16x16x16_fp8_fp8` (64 発)。
- 対照: fp16 の `tl.dot` → `v_wmma_f32_16x16x16_f16` (192 発)。

ISA への lowering はできている (Triton 3.7、ROCm 7.2)。
それでも 6〜7 倍遅い。GCN 行数は fp16 5,987 行に対し pv0 18,841 行、
pv2 36,183 行で、ドット以外の per-tile 処理
(fp32 リダクション: 平均・absmax、int/fp8 変換、dequant 乗算、
`sanitize_overflow` チェック等) が支配的。`_window_gscale` 等の
ホスト側コストは含まないカーネル単体での比較である。

結論: on-the-fly (a) は精度検証の手段であり、速度は出ない。
高速化には (b) 事前量子化キャッシュ (per-tile 量子化の除去) や
maskless fast path 付きの手書き HIP WMMA カーネルが必要。
本ブランチでは (a) を精度確立までとし、(b) は残件とする。

### 27B end-to-end (R9700): 不可 (報告)

R9700 上で 27B (`qwen38-27b-exl3-305-int8grid`) のロードは成功するが、
最初の 512-token forward で segfault (exit 139) する。
`exllamav3_ext` の HIP カーネル (または torch 演算) に gfx1201 対応が
ないためとみられる。よって end-to-end 27B プリフィルの R9700 計測は
行わず、層レベル計時 + 投影で代用する (以下)。
V620 側の Generator (チャンクドプリフィル+キャッシュ) での 32K
end-to-end 比較は精度節の needle 項を参照 (速度の主張ではなく
一致性の確認として)。

投影 (層レベル計時から): full-attention 16 層×上表の差分は、
現カーネルでは end-to-end でも遅くなる方向 (attention 占有率が高い
長文ほど悪化)。Plan E の手法 (Q 項 fit) に現カーネル比を代入しての
精密な外挿は、(b)/(HIP カーネル) 後に行う。

## 残件

- (b) 新キャッシュ層型 (K を平滑化 INT8 + ブロック平均・スケール、
  V を FP8 + スケールで保存し、カーネルが直接読む)。
  メモリ・帯域削減はこちらで初めて効く。
- HIP WMMA カーネル (Triton 3.7 は gfx1201 で WMMA を生成するが、
  per-tile 量子化の elementwise が支配的で 6〜7 倍遅い。maskless
  化・融合の手書きカーネルが必要)。
- デコード対応 (現状は fp16 にフォールバック)。
- num_splits (長文のグリッド量子化対策) の移植。
- R9700 での 27B end-to-end: `exllamav3_ext` (または torch 演算) が
  gfx1201 で segfault するため不可。R9700 対応ビルド後に再試行。
- デコード logit 比較ヘルパーの +inf パディング列対応
  (現状は KLD が全行 nan→0 化するため top1 のみ報告)。
