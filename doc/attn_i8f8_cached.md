# INT8-K / FP8-V パックドKVキャッシュ + キャッシュ読みプリフィル (design (b))

Round 1 (on-the-fly: `doc/attn_int8_fp8.md`) ではカーネルがタイルごとに
K/V を再量子化していた。これが R9700 で fp16 の 0.17x まで遅い主因だった。
本ドキュメントの design (b) では、トークンがキャッシュに入る時点で
K を INT8 (平滑化+per-block scale)、V を FP8 E4M3 (+per-block-per-channel
scale) に量子化して保持し、プリフィルカーネルはそれを直接読む。
タイル内再量子化は消える。

有効化は2段式 (どちらもデフォルト無効。既存の fp16 / 量子化キャッシュ経路
および round-1 on-the-fly は不変):

```bash
EXL3_CACHE_I8F8=1    # model_init の Cache 層を CacheLayer_i8f8 にする
EXL3_ATTN_INT8FP8=1  # プリフィルで INT8-QK/FP8-PV カーネルを使う (round-1 と共通)
```

`EXL3_CACHE_I8F8=1` のみ (カーネルスイッチなし) でも動作は正しい。
その場合は fp16 デクアント経路 (デコードと同等) で遅い。
`EXL3_I8F8_DEBUG=1` でキャッシュカーネルの起動ごとに形状と累積回数を出す。

## キャッシュ形式 (`exllamav3/cache/i8f8.py`)

`CacheLayer_i8f8(CacheLayer)`。P ページ×256 トークン、量子化ブロック
32 トークン (ページあたり 8 ブロック)、H kv ヘッド、D head_dim:

| テンソル | dtype | 形状 | 内容 |
|---|---|---|---|
| `k_q` | int8 | (P, 256, H, D) | 平滑化後 K の absmax/127 |
| `k_sc` | fp32 | (P, 8, H) | ブロックごと K スケール |
| `k_mu` | fp16 | (P, 8, H, D) | ブロックごと K 平均 |
| `v_q` | uint8 | (P, 256, H, D) | FP8 E4M3 ビット列 |
| `v_sc` | fp16 | (P, 8, H, D) | ブロック・チャネルごと V スケール |
| `k_stage`/`v_stage` | fp16 | (P, 32, H, D) | 未量子化の末尾行 |
| `stage_len` | int32 | (P,) | 末尾行数 (0–31) |

粒度は round-1 と完全一致 (Q: 行ごと /127、K: 32ブロックごと平滑化
absmax/127、P: 行ごと /448、V: ブロック内チャネルごと absmax/448、
蓄積 fp32)。フルブロックの数値は round-1 と一致する。

### 部分ブロックの扱い: fp16 ステージング

32 トークンブロックは 32 行そろって初めて量子化できる。
到着チャンクは任意長なので、各ページの最新部分ブロックは fp16 のまま
`*_stage` に保持する。追記時は「ステージ行 + 新行」を fp16 のまま結合し、
完成した 32 ブロックだけベクトル化 torch パスで一括量子化し、余り
(<32 行) をステージに戻す。追記は fp16 からの再量子化なので誤差は複利化
しない。1 追記あたり最大 31 行の再量子化で償却可能。
ブロックはページローカルかつ 32 アライン (256 % 32 == 0) のため、
追記が触るのは先頭ページのステージと完成ブロックのみ。
プロンプトキャッシュの `copy_page` は量子化ブロックとステージを
まとめて複写する (共有ページは同値プレフィクスが前提)。

デコード (`q_len <= 16`) や対象外形状では `get_kv` が fp16 にデクアント
(全ページ + ステージ上書き) して既存 fp16 カーネルで実行する。
正しいが全サイズ一時領域を使う (quant の level-2 相当)。高速路は
プリフィルカーネルのみ。

## カーネル (`exllamav3/modules/attention_fn/i8f8_cached.py`)

プリフィル専用 (`q_len > 16`)。Q は呼び出しごとに1回量子化
(行ごと absmax/127)。内側ループはブロック単位ロードのみ:

- QK^T: INT8 WMMA ドット → スケール + q·μ バイアス → fp32 オンライン softmax
- P を行ごと FP8 化 → V (FP8) との FP8 WMMA ドット → fp32 加算

K/V のリダクション・変換・sanitize はループ内にない
(書き込み時に `update_kv_direct` で償却済み)。
`<32` の末尾はステージ fp16 行から round-1 と同一のブロック計算
(`int8fp8._i8f8_block` を再利用) で処理するため、フルブロックの数値は
round-1 の torch リファレンスと一致する (R9700 実測 maxabs=0.00024)。

Triton 実装。gfx1030 では FP8 実ドットが codegen できないため
int8 プロキシ (`PV_MODE 1`) に自動フォールバックする
(round-1 と同じ)。
`nosmooth` / `per-tensor-k` は cached 形式と両立しないため辞退し、
fp16 デクアント路に落ちる (`fp16pv` は `PV_MODE 0` で対応)。

## dispatch 配線 (`dispatch.py`)

- 層が `CacheLayer_i8f8` (かつ fp16 Q 等) なら `update_kv_direct`
  で先に書き込み、`i8f8_cache` 付き `AttnArgs` でキャッシュカーネルへ。
- 成功すればそれで確定。辞退時は `get_kv` デクアント + 通常 fp16 路
  (k/v は無傷、キャッシュ書き込み済みなので正しい)。
- i8f8 層では round-1 フックは走らない (k_cache=None を新規行のみと
  誤認するため)。
- 後段 `update_kv` は本層では no-op (書き込みは direct 済み)。
- QSA 層を持つモデルでは `attn.py cache_layer_type` が i8f8 要求を
  fp16-planes の QSA キャッシュに退避する (既存動作不変)。

## GCN/codegen (27B 形状 B=1/Tq=1024/24/4/256/Tk=4096, causal)

R9700 (gfx1201) ネイティブコンパイル。`v_wmma` は実行される WMMA 命令の
静的出現数 (ドット種別は別途確認: QK → `v_wmma_i32_16x16x16_iu8`、
PV → `v_wmma_f32_16x16x16_fp8_fp8`)。

| カーネル | 総行 | v_wmma | 備考 |
|---|---|---|---|
| fp16 prefill | 5987 | 192 | 基準 (round-1 報告 6.0K と一致) |
| round-1 pv0/pv1/pv2 | 18841/26518/36183 | 192 | タイル内再量子化 (18.8K/36.2K と一致) |
| cached pv0/pv1/pv2 | 37100/33774/27121 | 240 | 再量子化なし |

cached は静的行数では round-1 を上回ることがある (3 ループ + 末尾呼び
出しの展開) が、動的にはタイルあたり dot+scale+bias+softmax+PV のみを
実行する。R9700 本命の pv2 では round-1 (36183) より小さい (27121)。
gfx1030 では SIMT 低下 (`v_dot`/`v_fma`) で機能確認用。
masked 領域のベクトル gather 版は LLVM SimplifyCFG をクラッシュさせた
(gfx1030) ため、タイル単位スカラーロード + スコアマスクのみの形に
直してある。masked ループ上限を `t_full` ではなく `n_hi` にしたのも
性能上必須だった (早期クエリが全タイルを無駄に計算していた)。

## R9700 速度 (attention カーネルのみ、idle 時、中央値、競合あり)

27B 形状 (24/4/256)。`cached` = カーネルのみ (pure reader)、
`cached-update` = 追記 1 回分の量子化書き込み。end-to-end =
両者の和。gsq_hessian (Flash Next) が同 GPU で 24.5GB 占有しながら
計算も回していたため、A/B 交互 × 3 ラウンドの中央値と広がりを報告する
(外れ値は競合による。round1 の 32K/8K ラウンドに +200ms 級の
スパイクあり)。

| Tk | Tq | fp16 | round-1 (対fp16) | cached (対fp16) | update | e2e (対fp16) |
|---|---|---|---|---|---|---|
| 8192 | 2048 | 8.81ms | 64.42 (0.14x) | 20.54 (0.43x) | 6.28 | 26.82 (0.33x) |
| 8192 | 8192 | 19.99 | 146.37 (0.14x) | 48.57 (0.41x) | 25.69 | 74.26 (0.27x) |
| 32768 | 2048 | 38.53 | 275.59 (0.14x) | 102.34 (0.38x) | 6.47 | 108.81 (0.35x) |
| 32768 | 8192 | 138.50 | 962.10 (0.14x) | 337.98 (0.41x) | 24.94 | 362.92 (0.38x) |
| 131072 | 2048 | 158.90 | 1141.99 (0.14x) | 415.07 (0.38x) | 6.43 | 421.50 (0.38x) |
| 131072 | 8192 | 627.32 | 4333.54 (0.14x) | 1608.10 (0.39x) | 25.72 | 1633.82 (0.38x) |

cached は round-1 の約 2.7–3.1 倍だが fp16 の 0.38–0.43x にとどまる。
内訳: update は Tq=2048 で 6.3–6.5ms、Tq=8192 で 24.9–25.7ms
(トークンあたり約 3.1µs/層で一定。注意: e2e の Tq=8192 行では
update が支配的になる)。
fp16 との rel は 0.027–0.044 (下の精度節参照)。

### なぜ fp16 に勝てないか: ドットレートの実測

生の `tl.dot` マイクロベンチ (gfx1201、同一タイル):

| dtype | 中央値 | 実効レート |
|---|---|---|
| fp16 | 15.3µs | 8.8 TMAC/s |
| int8 | 33.4µs | 4.0 TMAC/s |
| fp8 e4m3 | 32.7µs | 4.1 TMAC/s |

この Triton lowering では INT8/FP8 WMMA ドットが fp16 の約半速。
QK と PV の 2 ドットが律速のため、量子化 attention は原理的に
fp16 の約半分が上限で、実測 0.38–0.43x と整合する
(変換・softmax・P 量子化のオーバーヘッド分だけ下がる)。
HIP で書き直しても同じ WMMA 命令を使う限りレートは変わらないため、
速度目的の HIP 化は見送った。勝てるのはメモリ (次節) である。

### メモリ/token/層 (24/4/256)

| | 式 | B/token |
|---|---|---|
| fp16 | H·2D·2B | 4096.0 |
| cached | H·(2D + D/8 + 4/32) | 2176.5 (0.531x) |
| 既存 quant 4bit 参考 | — | 約 0.3x (別形式) |

内訳/頭: K int8 256B + μ fp16 16B + scale 0.125B + V fp8 256B +
V scale 16B = 544.125B。ステージングはページあたり 2·32·H·D·2B
(フルキャッシュ時は無視できる)。

## 精度

- 単体テスト (`tests/test_attn_i8f8_cached.py`, 7 件): 書き込みの
  fp16 ラウンドトリップ、ブロック量子化の完全一致、odd チャンク・
  ページ跨ぎ・1 トークン追記のカーネル対 round-1 リファレンス
  (maxabs < 0.02)、dispatch の on/off とデコードフォールバック、
  実 27B Q/K/V キャプチャ、copy_page/storage_size。V620 と R9700 で緑。
- R9700 カーネル対 round-1 torch リファレンス (27B 形状、pv2 実 FP8):
  maxabs=0.00024。フルブロックは round-1 と一致する (設計通り)。
- 実 27B Q/K/V キャプチャ (4 層): cached 対 fp16 maxabs 0.12–0.21、
  rel 0.022–0.036 (R9700 pv2。V620 プロキシでも同等 0.023–0.029)。
- 27B E2E needle (V620、32K passkey + 2K、3 アーム、HGEMM_FP16_OUT=1):
  全アーム passkey 取得。cached カーネル実行回数 272 (32K) / 32 (2K)
  でフォールバックなしを確認。
  - 32K デコード KLD (共通 finite 語彙に条件付け): round1-vs-cached
    mean=0.000336 max=0.0055 top1=1.0。fp16-vs-round1 mean=3.023、
    fp16-vs-cached mean=3.021 top1=0.733 (デコード履歴が 18 手で分岐
    した後の値。分岐前の同一履歴区間では fp16-vs-round1 0.000808、
    fp16-vs-cached 0.000332)。
  - 2K: round1-vs-cached mean=0.000975 top1=1.0。
  - round-1 報告 (cases-v1 KLD macro 0.0009) と同等。cached は
    round-1 と E2E で一致し、fp16 との差は round-1 と同程度
    (むしろやや小さいが誤差範囲)。
- 注意: V620 の E2E は int8 プロキシ PV であり、厳密な E4M3 の検証は
  R9700 のカーネル対リファレンスとマイクロベンチの fp16 対比
  (rel 0.027–0.044) が担う。V620 E2E は形式・dispatch・丸めの検証。

## 27B on-R9700 状態 (ext 再ビルド)

round-1 報告の segfault (exit 139) は ext に gfx1201 イメージが無い
ためだった (旧 .so は gfx1030 のみ)。`PYTORCH_ROCM_ARCH="gfx1030
gfx1201"` で再ビルドし、同一名に原子 mv (使用中プロセスなしを確認。
走行中プロセスは旧 inode を保持するため安全)。新 .so (89MB) は
V620 と R9700 の両方で import 可、R9700 で 8B モデルの load+forward
を確認 (segfault 解消)。
27B E2E (13.6GB) は同 GPU の Flash Next ジョブ (24.5GB 占有) と
重なり空き 9.7GB に載らないため未実施。層レベル計測 + 射影に留める:
prefill attention が全計算の支配項であり、上表の通り cached e2e は
fp16 の約 0.38x (0.27–0.43x)。27B が載れば `needle_i8f8_cached.py`
と `dense_kld_dump.py` + `EXL3_CACHE_I8F8=1` でそのまま測れる。

## Round 3: HIP gfx1201 WMMA プロトタイプ (結論: fp16 に届かず)

親タスクの指示 (INT8/FP8 が fp16 より速いのは given として HW レートの
再検証に時間を使わず、HIP で書いて cached attention が fp16 より
速くなるか試す) に従い、Triton を経由しない HIP 直書きの fused
プロトタイプを書き、R9700 で検証した。作業本体は
`eval/prof/r3_hip/` (`hip_i8f8_attn.hip` カーネル + torch 結合、
`bench_r3_hip.py` 正確性・速度ベンチ、`wmma_probe.hip` /
`probe2.hip` / `probe3.hip` レイアウト・素レート特定用プローブ)。

### gfx1201 WMMA レイアウトの特定

RDNA4 (gfx1201) の WMMA は gfx11 と異なり half-size フラグメント
(A/B 各レーン 8 要素: fp16 `half8` / int8 `int8x8` / fp8 `f8x8`、
C は `float8` / `int32x8`) で、既存 `rdna_wmma.hip.h` は gfx12 を
`__builtin_trap()` している (未対応)。ホスト hipcc の単体プローブで
R9700 実機・非対称行列・CPU 参照照合により以下を確定した
(3 dtype 共通、gfx11 式の配置):

- A: レーン L が行 `L%16`、k スライス `8*(L/16)+[0..8)`。
- B: レーン L が列 `L%16`、同 k スライス。
- C: 行 `L%16`、列 `8*(L/16)+[0..8)`。
- オペランド順序は (B, A, C) (第 1 引数が B 側)。
- fp16: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, a, c)`。
- int8: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kb, true, qa, ...)`
  (signed x signed、フラグはコンパイル時定数必須)。
- fp8: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, pq, ...)`。

### E4M3 は 0x38 = 1.0 (実効バイアス 7)

torch `.to(torch.float8_e4m3fn)` と gfx1201 WMMA は一致する
(tiny=2^-6、max 0x7E=448、サブノーマル単位 2^-9)。
当初の「fp8 WMMA が真値の 4 倍を返す」はバイアス 8 の誤読だった
(0x48 は 4.0)。スケール補正は不要。自作 RNE 変換器 `f32_to_e4m3`
は torch と 6 値域 x 65536 点でビット一致 (0 mismatch)。
P (非負) 専用。torch は有限 ~460 超で 0x7F (NaN) を返す quirk が
あるが P/sp は 448 を超えないため到達しない。

### 素レート (レジスタ常駐 WMMA ループ、chrono、best-of-7)

fp16 154–176 TOPS、int8 351–378 TOPS、fp8 364–365 TOPS。
**直書き WMMA では int8/fp8 ≒ 2.3x fp16** であり、Triton lowering
の約 0.5x とは逆転する。ドット律速は外せることが確定した
(1 命令あたり 8192 ops、4 独立アキュムレータ、CSE 回避のため
初期値 offset)。

### HIP プロトタイプの設計

`CacheLayer_i8f8` の packed テンソル + block_table + stage を直接読む
flash attention (online softmax)。1 ブロック = 16 Q 行 x 1 q-head の
単一 wave。QK^T は int8 WMMA、PV は fp8 WMMA、P 行量子化 (自作 RNE
E4M3)、q・μ バイアス、causal、末尾 <32 行はスカラー fp32 テール。
K/V タイルの LDS ステージングあり (素朴なストライド直読みは
8 倍オーバーフェッチで約 3 倍遅かった)。簡素化点: B=1、
head_dim=256 固定、GQA は head 毎ブロック (K/V は L2 共有で
Triton 版と同条件)、テールは fp32 (参照との差は量子化ノイズのみ)。

### 正確性

- E4M3 変換器 vs torch: 6 値域で 0 mismatch。
- カーネル vs round-1 torch 参照 (ランダム、奇数 Tq・テールあり・
  非 causal 含む): maxabs 0.00035–0.00061
  (同時計測の Triton cached 0.00043–0.00101 と同等以上)。
- 実 27B Q/K/V 4 層 vs fp16 SDPA: rel 0.022–0.036 (round-2 と一致)。
- `tests/test_attn_i8f8_cached.py` 7 件は R9700 でも緑
  (V620 でも緑を再確認)。`test_attn_int8fp8.py` + `test_kv_quant.py`
  68 件、`test_dense_int8.py` ALL_OK は緑のまま。

### 速度 (R9700、attention カーネルのみ、A/B 交互 x 3 ラウンド中央値)

Tq=2048/8192、27B 形状 (24/4/256)。`cached-tri` = round-2 Triton、
`cached-hip` = 本プロトタイプ。

| Tk | Tq | fp16 | cached-tri (対fp16) | cached-hip (対fp16) |
|---|---|---|---|---|
| 8192 | 2048 | 8.63ms | 20.76 (0.42x) | 47.99 (0.18x) |
| 8192 | 8192 | 20.00 | 48.41 (0.41x) | 113.66 (0.18x) |
| 32768 | 2048 | 38.28 | 96.09 (0.40x) | 212.04 (0.18x) |
| 32768 | 8192 | 137.70 | 335.67 (0.41x) | 747.97 (0.18x) |
| 131072 | 2048 | 158.13 | 411.64 (0.38x) | 886.07 (0.18x) |
| 131072 | 8192 | 620.50 | 1605.27 (0.39x) | 3344.04 (0.19x) |

fp16 / cached-tri は round-2 表と一致 (再現)。hip-vs-fp16 maxabs は
0.0003–0.0013 (1 点 0.052 も rel は 0.032 で範囲内)。
32K/2048 の round-2 に +200ms 級スパイクあり (同 GPU の競合。
中央値で吸収、round-1 の注意書きと同趣旨)。

### 結論: 融合しても fp16 に届かない

ドット単体は HIP 直書きで 2.3 倍にできたが、融合カーネル全体では
fp16 の 0.18x (Triton cached の半分以下) に留まった。主因は
プロトタイプのスケジューリングであって WMMA 自体ではない:
1 wave/block + 38.5KB LDS 占有で CU 常駐が 1 block に張り付き、
16 段の依存 WMMA チェーン・直列 exp/cvt/qμ のレイテンシが露出した
まま (実効 8.4 TFLOPS 対 WMMA ピーク約 350)。LDS 化・VGPR 削減等の
安い手は 1.27x / 微増に留まり、構造的と判断した。

勝つには Triton 並の multi-wave + パイプライン化 (= 本格 FA 最適化の
書き直し) が必要で、到達点の見込みは fp16 との parity 程度
(ドット 2.3 倍が同一オーバーヘッドを相殺する構図) のため、
Round 3 はここで区切る。dispatch 配線はしない
(Triton cached が速いため。現行の Triton がフォールバック兼本命)。
副産物の確定知見 (gfx12 WMMA レイアウト、E4M3 バイアス 7 の一致、
RNE 変換器、素レート) は将来の HIP 化の土台として残す。

## Round 4: fp16 最小差分の全 FP8 / INT8-QK プリフィル (結論: 0.90–1.04x、配線見送り)

方針転換: r2/r3 の内ループ作業 (per-32-block K 平均 μ + q·μ バイアス、
per-block-per-channel V による accumulator 全体の rescale、per-row P
スケール、per-tile K/V 再量子化) を全部捨て、調整済み fp16 Triton
prefill カーネル (`triton_paged._paged_attn_prefill_kernel` /
`_paged_attn_prefill_inner`、タイル/パイプ/split 選択は同一) を複写し、
内ループが「fp16 + スケール 1 乗算 + fp8 変換」になる最小差分だけを
加えた (`exllamav3/modules/attention_fn/r4_fp8.py`、差分は `[R4]` 標記)。

### スケール方式 (内ループに追加の reduction なし)

- **Variant A (all-FP8)**: Q/K/V/P すべて FP8 E4M3。Q は行ごと、
  K は kv-head ごと scalar (QK^T タイルへの寄与は fused 行係数
  `row_log2e = sq[row]·sk·sm·log2e` の単一 broadcast 乗算で、fp16 の
  `* qk_scale_log2e` と同数)、V は (head, channel) ごと frozen で
  出力タイルに最後に 1 回だけ適用 (block 不変なので online rescale
  と完全両立)、P は固定 1/448 (P∈[0,1]、reduction なし、逆数は最終
  正規化 `acc·sv/(l·448)` に折り込み)。
- **Variant B (INT8 QK + FP8 PV)**: K を call 内 frozen の単一平均 μ
  (head, channel) で平滑化 — 全キー共通なので q·μ は行定数で
  softmax 不変 → バイアス項なし。K は head ごと scalar、Q 行ごと。
- 量子化は pre-pass (`r4_quantize`、torch、カーネル外で 1 回) で、
  カーネルは pure reader。計時は kernel-only と pre-pass 込み (e2e)
  を分離報告する。QK_MODE constexpr で A/B 同一ソース。
- プロトタイプ範囲: prefill 専用、causal/非 causal、window/softcap/
  sinks/quant-cache/append なし (decline)。head_dim は pow2 のみ。

### Triton ノブ知見 (gfx1201)

- ドット単体 (QK^T GEMM C[4096,8192] K=256、wall 計時):
  fp16 17.5 / int8 22.7 (**1.30x**) / fp8 23.2 (**1.32x**) TMAC/s。
  r2 報告の「Triton int8/fp8 は fp16 の半速」は小タイル launch 支配の
  測定アーチファクトだった (0.03µs 表示の破綻を wall 計時で確認)。
  注意: HIP イベント計時は sub-ms で 0.000ms を返すことがある
  (同カーネルで wall 0.48ms に対し event 0.000ms を確認) ため、
  sub-ms 域は wall 併記が必須。10ms 以上ではイベントは正常。
- AMD ノブ (dot / full 共通傾向): `num_stages=1` が最良 (+5–8%)、
  `kpack=2` / `matrix_instr_nonkdim=16` / `waves_per_eu=2` /
  `schedule_hint=attention` は ±8% 以内。`num_stages=1` で shared
  OOM になる形状あり (128KB 要求など)。
- GCN (BM64/BN32/W8/S2、27B タイル): fp16 5987 行 vs r4 9768 行
  (+63%)、しかし `v_wmma` は **192 = 192 で同数**。fp8 operand の
  shuffle (`v_perm` +32) と変換が嵩む。WMMA 発行数が同一 =
  Triton は fp8/int8 の 2x ALU レートを引き出しておらず、1.3x は
  traffic (半バイト) 由来。内ループの追加コストは P の `×448 + cvt`
  が支配的。
- 副発見: **fp16・r4 とも BM=128 が速い**。D256 の既定 BM64 は
  Hopper 想定 (約 100KB smem) で、RDNA (64KB LDS) では undertune
  だった。fp16 既定 → BM128+BN64+S1 で 37.4→24.5ms (**1.52x**、
  32K/2K)。fp16 の BN64/S2 単体は shared OOM (73728 > 65536)、
  BM256 は fp16 OOM・r4 spill (0.3–0.5x) で不可。
  (本ラウンドでは fp16 既定を変更しない。親への再調整提案とする。)

### 成熟実装 check (timebox、見送り)

bench image (`rocm-exl3-phase2-env:tested`) に aiter / CK / flash-attn /
vllm / sglang は不在 (`/work` に aiter-hc の build 残骸のみ)。
重い導入が必要なため見送り。追加参照は torch 手計算 + fp16 Triton
(上の通り、それで足りた)。

### 精度

- `tests/test_attn_r4fp8.py` 10 件: V620 で 6 pass / 4 skip (カーネル
  件は FP8 WMMA 機で skip)、R9700 で 10 pass。
- カーネル対 fp16 (27B 形状サブセット): ≤32K rel 0.041–0.056、
  128K rel 0.044–0.051。r2/r3 (0.022–0.05) と同等。
- カーネル対 torch 参照: ≤32K rel 0.04–0.06 で一致。128K では
  **参照側のみ**乖離 (rel 0.16、両カーネルは fp16 と 0.044 で一致
  するため参照 quirk と判定。E4M3 subnormal 域での torch
  `.to(float8_e4m3fn)` 丸め差異の疑い — r3 報告の torch fp8 quirk
  (有限 ~460 超で NaN) と同系)。該当テストは ≤32K に pin した。
- 副産物: ROCm torch SDPA の GQA + causal が不正 (手計算基準で
  rel 1.5) のため、本テストの fp16 基準は手計算 (explicit mask)
  を使う。
- 27B cases-v1 KLD・32K E2E: R9700 空き 9.7GB < 27B 13.6GB
  (hessian 稼働中) のため未実施。VRAM に空きが出れば
  `bench_r4_fp8.py` の正確性節 + `dense_kld_dump.py` でそのまま測れる。

### 速度 (R9700、best-vs-best、カーネルのみ中央値、A/B 交互×3)

27B 形状 (24/4/256、causal)。fp16-best = BM128+BN64+S1、
r4-fp8 = BM128+BN64+S1、r4-int8 = BM128+BN64/S2。
pre-pass (3 反復中央値): 8K 2–3.6ms、32K 4–7ms、128K 15–20ms
(同 GPU の hessian と競合のためスパイクあり、中央値で吸収)。
e2e (= kernel + pre-pass) は全形状で fp16 に負ける。
小形状 (Tk1024/Tq128) では 1.2–1.38x が出たが launch 支配の差で、
大形状では消失する。

| Tk | Tq | fp16-best | r4-fp8 (対fp16) | r4-int8 (対fp16) |
|---|---|---|---|---|
| 8192 | 2048 | 5.9ms | 6.2 (0.95x) | 5.8 (1.01x) |
| 8192 | 8192 | 13.3 | 14.7 (0.91x) | 13.6 (0.98x) |
| 32768 | 2048 | 24.5 | 27.3 (0.90x) | 24.9 (0.99x) |
| 32768 | 8192 | 87.7 | 97.1 (0.91x) | 88.5 (0.99x) |
| 131072 | 2048 | 98.7 | 110.3 (0.90x) | 102.7 (0.96x) |
| 131072 | 8192 | 404 | 438 (0.93x) | 390 (**1.035x**、2 独立 run で再現) |

(参考: 同形状の fp16 既定 (BM64/BN32/S2) は 8.6 / 20.6 / 38.4 /
137 / 158 / 626ms。r4 既定タイルでも 0.91–0.93x だったが、fp16 側
再調整後の best-vs-best が上の表。)

### 結論: 配線見送り

r2 の 0.39x → **0.90–1.04x** に改善したが、fp16 を広く上回らない。
唯一の勝ち (int8、128K/8K、kernel-only +3.5%) は pre-pass (+20ms)
込みの e2e では負けるため、dispatch 配線はしない。
残る壁は構造的: 同一 WMMA 発行数 + fp8 周辺 (P cvt・shuffle) の
約 8% 固定オーバーヘッド。先は (a) 非ドット部の削減か (b) fp8 半
バイトが効く領域 (128K+ で int8 が勝ち始めた兆候) の深掘り。
どちらのためにも本ラウンドの資産 (`r4_fp8.py` + 3 bench + unit
10 件 + GCN/ノブ知見) が土台になる。

ファイル: `exllamav3/modules/attention_fn/r4_fp8.py` (本体+参照)、
`eval/prof/bench_r4_fp8.py` (正確性+速度表)、
`eval/prof/bench_r4_dotsweep.py` (ドット sweep+GCN)、
`eval/prof/bench_r4_knobs.py` (カーネル sweep+GCN/fp16sweep/verify)、
`tests/test_attn_r4fp8.py`。
