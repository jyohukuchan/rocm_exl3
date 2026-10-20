# XeStrataから取り込めるEXL3最適化の調査

調査日: 2026-10-21。実装変更・新規GPU測定は行わず、ローカルのコード、変更履歴、既存測定記録を照合した。

後続の取り込みと測定結果は [実装・検証記録](xestrata_exl3_backport.md) を参照。

比較したsnapshot:

- rocm_exl3: `7d79d4977df5423230c219a932eb6933d08af351` (`main`)
- xestrata-exl3/XeStrata: `ed74f95c89eb5b6b45629a28b19dd4825c4ab8d0` (`feature/integrated`)
- xestrata-exl3配下の `rocm_exl3-q2fix` と、保存済み `q2fix-k2.patch`

## 結論と優先順位

取り込む価値はある。最も具体的なのは**3995WXのAVX2 CPU expertにN-major圧縮タイル配置を追加すること**。
GPUでは**圧縮wordのwave内共有**と**デバイス・形状に応じたK分割の調整**が候補になる。
ただしrocm_exl3にも同種の最適化があり、XeStrataの高速化率をそのまま期待値にはできない。

| 優先度 | 候補 | 主な対象 | 判断 |
| --- | --- | --- | --- |
| 1 | CPUのN-major圧縮タイル配置 | CPU expertを併用する3995WX | 現行AVX2のstrideを直接改善できる。最初の試作に適する |
| 2 | GPUの圧縮wordをwave内で共有 | dense GEMV、MTPのmultirow、expert GEMV | 現行のFragB配置を保ったHIP実装で比較する価値がある |
| 3 | K分割と物理wave数の調整 | V620/R9700のdecode、小行数の行列積 | 既存split-Kの再調整。Xeの定数を直接コピーしない |
| 別枠 | K=2量子化のLDS超過修正 | 2bitを含むEXL3変換 | 移植作業で作られた修正が本体に未反映。性能改善より先に扱う不具合修正 |

## 1. CPU: N-major配置は現行AVX2と相性がよい

XeStrataの `ff47ff66` は、CPU cacheの圧縮重みをN-majorに変更した。
16×16 tile内部のビット列を変えず、tileの順序だけを変更する。
以下の添字に `16 * bits` 個のuint16を掛ける。

```text
canonical:  k_tile * n_tiles + n_tile
cpu_block: (n_tile / 8) * k_tiles * 8 + k_tile * 8 + n_tile % 8
cpu_nmajor: n_tile * k_tiles + k_tile
```

rocm_exl3にも8 N-tile単位のband配置はあるが、CPU hostはAVX-512BW対応時にしか適用しない。
3995WXで使用するAVX2経路はcanonicalのまま。
[avx2_tiles](../exllamav3/exllamav3_ext/cpu/moe_mul1.cpp) はN tileを外側、K tileを内側に走査し、
`row_stride = n_tiles * packed_size` ずつ読む。software prefetchは既にある。
例えばbits=5、N=2048なら1 tileの160 bytesを20 KiB先ごとに読み、N-majorなら連続160 bytesになる。

適用箇所:

- [MoeCpuMatrix](../exllamav3/exllamav3_ext/cpu/moe_mul1.h): `swz` の真偽だけでなく配置を識別する情報を持たせる。
- [CPU host](../exllamav3/model/moe_cpu_host.py): arenaへのrehome、expert swap時の更新、GPUへのstreaming時の逆変換。
- `avx2_tiles` とscalar fallback: N-majorの添字に対応する。
- 現行AVX-512向けband配置は、その演算ループに合わせて別の選択肢として残す。

Xe側の既存CPU測定はH=2048、FF=512、40層×5 expert、2 warmup＋20巡。
`docs/EXL3_CPU_LAYOUT.ja.md` の配置比較では24 workerでcanonical 17,694→N-major 11,700 µs/40層、
約34%短縮。48 workerでは17,010→12,798 µs、約25%短縮だった。
共有hostのノイズがあり、commit後の再測定では48 workerのcpu_blockとの差はばらつきに入る。
**rocm_exl3での性能も、併用decode全体の改善率も未測定**。

最小の検証は、同一重みのcanonical/N-majorでCPU出力をbit比較し、数十層分の重みを巡回させること。
1 expertだけのwarm反復ではcache効果を過大評価する。
GPUへ戻すバイト列、dynamic swap、streaming prefillも確認が必要。
出力Hadamardの128要素単位を変える必要はない。

## 2. GPU: 圧縮wordのwave内共有

Xe側の `57c9d0c3`、`dbadf9a7`、`1a5baf50` が参考になる。
`src/kernels/xe/exl3_word_tile.hpp` の `DistributedTile<Bits>` は、16×16 tileの圧縮wordを
laneあたり最大2個のレジスタへ読み、subgroup shuffleで必要なwordを配る。
tileあたりのuint32数は `8 * bits`。bits<=4なら1 bank、bits>4なら2 bankを使う。
5〜7bitの共有元bankは静的に決定し、8bitの曖昧なlookupは両bankを受信してから選ぶ。

rocm_exl3の [dq_dispatch](../exllamav3/exllamav3_ext/rocm/quant/exl3_dq_rdna.hip.h) は既にbit幅別に特殊化され、
laneが読むwordから複数stateを作って再利用している。ただしwordをwave全体で一度ずつ読む方式ではない。
例えば4bitの `dq8_aligned_4bits` にはlaneごとに2つのword loadがあり、tile全体には32 wordしかない。
これは論理的に重複する読み取りであり、**DRAM転送量がその倍率になるという意味ではない**。
coalescing/cacheで既に共有される部分があり、shuffleの追加コストとの比較が必要。

最初の適用先は、global memoryから直接復号している以下の共通部分がよい。

- [multirow GEMV](../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_multirow_rdna.hip): `exl3_gemv_dot_tile_direct_mr`
- [single-row GEMV](../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.hip.h): `exl3_gemv_dot_tile_direct`

Xeの16列×2 K-lane配置を移すのではなく、rocm_exl3の `FragB` と `fdot2` の対応を保ち、
wordの供給だけを変更すれば、既存の積和順序を保ちやすい。
大きなMのcooperative GEMMは圧縮BをLDSへstageしているので、最初から同じ変更を広げない。

Xe側の `dbadf9a7` の記録ではresident fixed-token decodeが37.11→49.13 tok/s、
CPU併用は29.80/29.93→30.66 tok/s。2 holdout、793位置で変更前後のlogitsがbit一致した。
これはXeの旧scalar寄り復号との比較であり、既に8値をまとめて復号するrocm_exl3との差ではない。

検証は代表bit幅3/4/6/8、m=1とMTP用m=3から開始し、既存の
[multirow_check.py](../rocm_tools/multirow_check.py)、
[mgemv_bitwise.py](../rocm_tools/mgemv_bitwise.py) を使う。
VGPR数、spill、VMEM命令数、shuffleのコストも比較し、kernel時間と固定token decodeの両方で判定する。

## 3. GPU: K分割の再調整

Xeの `d0a8b20d` は、固定した論理K分割とその加算順序を保ちつつ、実際に担当する物理wave数を変える。
出力tile数とdeviceの並列容量からwave数を選び、1 waveが担当するK tile数にも上限を設ける。
V620の既存dot単体測定ではin_proj_qkv 75.0→60.2 µs、in_proj_z 48.4→36.7 µsだった。

rocm_exl3には既にin-block split-Kがあり、
[exl3_gemv_splitk_warps](../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_rdna.hip) が4/8/16 waveを選ぶ。
現在の閾値は主にgfx1151の測定に基づき、K tile数、N tile数、matrix batch数で分岐する。
V620/R9700の能力を取得してこの選択へ反映する余地がある。
まず既存の `EXL3_GEMV_SPLITK_WARPS=4/8/16` で同一shapeを比較できる。

Xe側のR9700測定 `records/wave-span-sweep-r9700-ee7affba/results.json` では、
span=4で61.92/62.83、16で66.36/66.21、64で64.94/65.25 tok/s。
小さく分けるほど速いわけではない。span=16をrocm_exl3の既定値にする根拠にもならない。

注意点として、現行rocm_exl3はwave数を変えるとK区間とreduction順も変わる。
Xeの「物理waveだけを変更してbit一致する」性質は自動的には得られない。
まず既存設定の数値許容差とlogitsを評価し、bit一致を必要とする場合は論理分割を別に保持する設計が必要。
multirow、expert batch、wide lm_headは別々に測る。

## 重複するため取り込み不要なもの

| Xe側の改善 | rocm_exl3の現状 |
| --- | --- |
| CB2のhalf FMA (`5edb8f51`) | [codebook_rdna](../exllamav3/exllamav3_ext/rocm/quant/codebook_rdna.hip.h) が既に `__hfma` / `__hfma2` を使う |
| 符号なしbyte dot (`d8752094`) | [hip_compat](../exllamav3/exllamav3_ext/rocm/hip_compat.hip.h) の `__dp4a` は既に `__builtin_amdgcn_udot4` |
| 複数tokenで復号重みを再利用 | multirow GEMVは1回の `dq_dispatch` を最大8行の積和へ再利用済み |
| 復号後のLDS往復を避ける | direct GEMVは既にnative fragmentのまま `fdot2` で演算する |
| H128のbutterfly、起動融合 | Hadamard helper、入力／出力融合、graph対応が既にある。Xeの構造をそのまま移す理由は薄い |
| CPU整数内積／AVX2化 | CPU mul1は既にAVX2、AVX-512BW/VNNI/VBMIをdispatchする |
| RAM/cgroup読取りのhandle再利用・層間の集約 | Xeのhost容量管理に対する改善。今回確認したrocm CPU forwardに同じ毎層pollingは見当たらない |

CPUの整数内積は入力量子化を含む近似経路であり、GPUの正確なF16復号とのbit一致を意味しない。
Xe側にはaffine補正の差もあるので、レイアウト変更と数値式の変更を一緒に取り込まない。
またROCmの [GPU int8 GEMV](../exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_int8_rdna.hip) は現在stubだが、
XeのCPU整数内積をそのまま移せるものではなく、別の実装・精度評価が必要。

## 別枠: 移植作業で見つかったQ2量子化修正

XeStrata本体のSYCL最適化ではなく、そのEXL3 checkpointを作るためのrocm側workcopyにある修正。
本体の `quantize_rdna.hip` と `quantize_tiles_kernel_rdna.hip.h` には未反映だった。

K=2、L=256のplain quantizerは `edges=65536>>2=16384` とし、
cost配列2本で65,536 bytes、その他704 bytes、合計**66,240 bytesのdynamic LDS**を要求する。
R9700で `cudaFuncSetAttribute` がinvalid argumentになった既存記録がある。
`EXL3_QT_OPTIMIZED=0` でも再現しており、optimizedの選択ミスではない。

保存済み修正は以下の2点を同時に変更する。

1. hostのLDS計算でK=2のcost配列を数えない。
2. kernelのK=2 cost配列を既存のglobal `temp_costs` scratchへ向ける。

hostの要求サイズだけを減らすとLDS範囲外アクセスになるため、片方だけの移植は不可。
取り込む際はdeviceのLDS上限と要求サイズの確認も加えるとよい。
既存記録では修正版がV620で層25以降まで変換を進めたことを確認できるが、
この調査では完成checkpointの検証やR9700上の修正版再実行は行っていない。
取り込む際はK=2の小規模量子化と、K=3/4の既存経路を確認する。

## 取り込み時の進め方と根拠

CPU併用を改善したい場合はN-majorから、全GPU decodeを改善したい場合はword共有から着手する。
各変更は個別にON/OFFを比較し、同じ重み・token列・KV設定・expert配分・graph条件を使う。
CPUはwarm単体だけでなく複数層巡回、GPUはV620とR9700それぞれで測る。
MTPについては受理率や生成token数の変動も含め、kernel時間だけで採用しない。

参照したXeの一次資料は、上記snapshot内の `docs/EXL3_WORD_TILE.ja.md`、
`docs/EXL3_CPU_LAYOUT.ja.md`、各変更commitの本文と実装。
既存測定は専用データ領域 `xestrata-exl3/records/` の以下を照合した。

- `verify-gpu-routed/RESULT.ja.md`、`verify-gpu-dense-cost/RESULT.ja.md`
- `verify-half-fma/RESULT.ja.md`、`routed-word-tile/git-audit.json`
- `wave-span-sweep-r9700-ee7affba/results.json` と対応する測定script
- `quality-instruct/exl3-2.0-conversion-failure.json`、`quality-instruct/q2fix-k2.patch`
- `quality-instruct/conversion-2.0-live-follow-20261020.verification.json`

Xeの追加部分のheaderはLGPL-3.0-or-later、rocm_exl3本体のLICENSEはMIT。
今回の成果物は方式・適用箇所の調査であり、Xeの実装コードは転記していない。
