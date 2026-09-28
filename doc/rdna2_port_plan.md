# V620 / RDNA2 移植・性能改善・TP2 実施計画

作成: 2026-09-28。状態: 計画のみ。実装・モデルダウンロード・GPUベンチマークは未着手。

対象fork: `dd7a670065f37943f09a5eeb53818f38e9751472`。
本家TP参照: `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`。

## 目的と順序

ユーザー指定の順序を工程の依存関係とする。

1. 単一V620で動作させる。
2. prefill/decodeを測り、遅い部分を修正する。
3. V620×2のレイヤー分割を動かす。
4. 単一GPUで使った同一モデルを測り、分割による遅さを修正する。
5. Qwen3.8-Flash-Next EXL3約3bpw + PLE RAM常駐を動かし、速度を改善する。
6. 本家のモデル対応を取り込み、ROCmでTensor Parallel 2を実現する。

各段階で数値の正しさ、再現可能な起動、性能を確認する。初期はtext、batch=1、FP16 KV、HIP graph/MTP/投機生成なし。目標の変更は測定根拠とともに記録する。性能未達を黙って合格にしない。

## 使用モデル

| ID | モデル | 用途 |
| --- | --- | --- |
| D | turboderp/Qwen3-8B-exl3、4.0bpw | 小さなdense smoke test、matmul/attentionの切り分け。公開ファイル約5.21GB |
| M | turboderp/Qwen3-30B-A3B-exl3、3.0bpw | 単一GPU・2GPU・TP2の共通主検証モデル。公開ファイル約12.2GB |
| Q | turboderp/Qwen3.8-Flash-Next-exl3、3.05bpw_h5_ng5 | 最終対象。本体shards約48.74GiB、PLE約30.40GiB |

D→Mまでを単一GPU動作の完了条件にする。Mは1枚に十分載るため、同一重みを1枚/2枚で比較できる。Dの検証は短い診断工程に留める。GDN/QSA/PLEはQの工程で個別テストしてから全モデルを実行する。

ダウンロード時にHF revision、ファイルhash、config、実際のcodebookと各tensorのbit幅をmanifestへ固定する。EXL3旧checkpointのcodebookがQと同じとは仮定しない。Qに必要な3bit mul1と高bit dense/headは合成tensorでも先に検証する。

公開量子化重みを利用し、自前BF16→EXL3変換はこの6工程の必須経路に含めない。V620上の量子化器検証は別工程として追加可能（既存調査では追加5–15人日、ジョブ実時間は未測定）。

## 工程と工数

GPU/推論エンジン開発経験者1名相当、1人日約8時間の実作業見積もり。AI処理時間や暦日を意味しない。download/長時間benchmarkの待ち時間は別。必要な修正量はPhase 0–1の実行結果で更新する。

| Phase | 成果 | 工数 |
| --- | --- | ---: |
| 0 | 固定環境・計測基盤 | 1–2人日 |
| 1 | 単一V620でD/Mの正しい生成 | 3–6人日 |
| 2 | 単一V620のprefill/decode改善 | 3–8人日 |
| 3 | 同じD/Mを2GPUレイヤー分割で生成 | 2–4人日 |
| 4 | 2GPUの転送・配置・同期改善 | 2–5人日 |
| 5 | Qの起動・品質・速度改善 | 5–12人日 |
| 6 | D→M→QのTP2対応・評価 | 10–20人日 |

TP前まで16–37人日、TP込み26–57人日。性能修正が不要なら当該工程は短縮する。gfx1030対応Torch/Triton、QSA/GDN、TP通信が主要な不確定要素。TP native経路の全面移植まで必要ならPhase 6は再見積もりする。

## Phase 0: 環境と計測の固定

- V620のPCI BDF/UUIDとHIP indexを対応付け、片方のみを公開する。R9700は自動選択から外す。SMIとHIPの番号一致は仮定しない。
- 既存venv/containerを確認し、使える組み合わせを専用環境へ固定する。デフォルトpythonのCPU版Torchは使わない。driver/全体環境を無条件に更新しない。
- HIP SDK、Torch HIP、Triton、rocBLAS、kernel driverのversionとbuild flagsを保存する。gfx1030のGEMM、Triton JIT、copyを最小入力で実行する。
- benchmark対象GPUの他ジョブ・clock・温度・電力・VRAMを記録し、競合がある測定を採用しない。
- `rocm_tools/bench_model.py` を基にRDNA2用harnessを整備する。現行harnessにはmodel split/PLE指定がなく、decodeは128-token promptのみ、`os._exit(0)`でteardownを隠すため、そのまま全工程の合否判定には使わない。
- 明示device/split、ngram RAM、複数contextでのdecode、JSON出力、全device同期、エラー伝播を追加。通常終了のsmoke testも独立して置く。

成果物: 環境manifest、再現用setup/run手順、計測harness、GPU ID対応表。合格: 両V620を個別に指定でき、基本演算とTritonが正しい。

## Phase 1: 単一V620で動作

主対象: `setup.py`、`rocm_tools/hipcc_probe.sh`、`exllamav3_ext/rocm/`、`rocm_py/`。

- gfx1030のcapability/build分岐を追加。LDSは保守的64KiBと実device値を照合し、host/device両側で同じ制約を使用する。
- WMMAを呼ぶソース・instantiation・dispatchを分離し、RDNA2で到達しないだけでなくビルド時にも要求しない構成にする。
- decodeは既存fdot2 GEMV/mgemvを再利用。復号、SUH/SVH Hadamard、routing weight、FP16/FP32出力まで照合する。
- prefillは必要なmatrix/expertのみ復元してrocBLASへ渡す実装から開始する。短いprefill、multirow、BC/MultiLinear、MoE専用経路の取りこぼしをなくす。
- 全重みをFP16常駐へ置き換えず、復元bufferを制限・再利用する。
- Dで短い生成→MでMoE生成→512/2K/8K入力へ拡大する。

合格:

- FP32またはPyTorch参照とのkernel比較、layer出力比較が通る。許容誤差は既存テストと演算精度から事前に定義し、失敗を消す目的で緩めない。
- 3bit mul1、モデルが実際に使うcodebook/bit幅、Q向け高bit projectionが検証される。
- D/Mが8K入力+256-token生成を完了し、NaN/Inf、OOM、hangがない。終了・再ロードも正常。
- 同じseed/入力の数値誤差を記録。GPU間・reduction順序が変わる場合、全文字列の完全一致のみを判定基準にしない。

成果物: 単一GPU起動config、correctness結果、最初のprefill/decode baseline。

## Phase 2: 単一V620の速度改善

Mを主対象とし、Dも回帰確認に使う。

- prefillとdecodeを別profileに分け、累積時間上位3項目を抽出する。
- prefill候補: expertごとの小GEMM/launch乱発、dequantの重複、buffer確保、Triton tile/occupancy。
- decode候補: EXL3復号・Hadamard、split-K、MoE expert並列化、CPU側同期、kernel間launch待ち。
- grouped/batched expert実行、RDNA2用SIMD/fdot2 tiled GEMM、dequant融合はprofileで必要と判明したものから実装する。
- 演算reference→kernel A/B→同条件E2Eの順で採用判定し、最適化単位でcommitを分ける。

合格: 下記測定規約を満たし、Phase 1より性能を悪化させず、既知の大きな回避可能ボトルネックを解消する。実用目標の未達は原因・実測・追加見積もりを残して明示し、完了扱いしない。

## Phase 3: 2GPUレイヤー分割

主対象: `model/model.py` とautosplit関連、`util/device_copy.py`、cache/stateのdevice配置。

- Phase 2の同じD/M checkpoint・入力・cache設定を使う。
- 通常autosplitではM全体が1枚に載るため、split予算または配置指定で両GPUに確実にlayerを置く。loader log/各GPU VRAM/traceで実配置を確認する。
- 連続layer groupの概ね均等分割から開始し、余計なdevice往復を作らない。
- P2P copyの両方向を複数sizeで数値検証・計測し、必要なら既存host bounceを使う。PCIe転送帯域は測定する。
- layer境界でhidden states、cache、norm/embedding/head、stream/eventの整合を確認する。

合格: 同一モデルの1GPU/2GPUでlogits/PPLの差が許容範囲内。両GPUの使用を確認。30連続jobを完走し、warmup後のVRAMが継続増加しない。

成果物: 明示2GPU配置config、P2P/host-bounce結果、1GPU対2GPUの比較表。

## Phase 4: 2GPUの速度改善

- Phase 2の1GPU結果を固定比較対象にする。重み、入力token列、chunk、batch、KV精度を変えずに比較する。
- 転送回数、転送量、host staging、device同期、片側GPUの計算時間を測る。
- layer配分、embedding/head配置、buffer再利用、転送・同期位置を調整する。
- 初期の調査基準は「同じMの1GPUに対してprefill/decodeが10%以上遅い」。これは2GPUが2倍速くなる要求ではなく、余分なoverheadを調べる基準。
- 正しい最小転送・同期にしても10%以上の差が残る場合、計算/通信/host時間で説明し、最善構成と残る制約を報告する。2GPU化の価値は容量確保にもある。

成果物: 分割方式の性能差、推奨配置、Phase 5向けの余裕あるVRAM予算。

## Phase 5: Qwen3.8-Flash-Next

- Qのrevision `69e33439ae950f17bcbe95c98f117d80f759ab6d`を基準候補とし、実ファイルを確認して固定する。
- gfx1030上でQSA、GDN prefill/recurrent、gated residual、PLEの個別テストを先に通す。GDN/QSAでTriton JITまたは数値誤差が出れば、この段階で修正する。
- PLEは `--ngram_ram` を明示し、実際の `trellis_ram`、RSS、page fault、per-token gather/copyを確認する。
- 全routed expertを2 GPUに常駐。text、batch=1、MTP/vision/投機生成なしで512→2K→8K→32Kへ進む。
- 最初のload ceilingは1枚28GiB程度から測り、scratchやcache込みで各deviceに少なくとも1GiB程度の余裕を確保する。OOM時にcacheだけを黙って減らして別条件の結果にしない。
- PLEのEOS境界、複数job、prefix再利用、GDN状態reset、QSA選択長境界を検証する。
- 遅い場合はMoE、GDN、QSA、PLE gather/同期に分解。Phase 2/4の共通kernel変更はD/Mにも回帰確認する。
- Qの量子化品質とRDNA2移植誤差を分離する。同じEXL3重みのreconstruct参照・module参照を使い、利用可能なら本家CUDA logitsとも比較する。125BをFP16でGPUへ一括ロードする検証は前提にしない。

合格: 32K入力+256-token生成、30連続job、有限logits、参照に対する許容誤差、PLE RAM常駐、各GPUピークVRAM、正常終了を確認。品質は日本語/コードの固定課題と固定corpusのPPL/teacher-forced logitsで記録する。

性能の暫定実用目標: batch=1、8K入力時にdecode 10 tok/s以上、warm状態・prefix cache missの8K TTFT 60秒以内。これは実測予測やユーザー指定値ではなく、本計画の改善優先度を決める仮目標。32Kはまず完走・安定性を確認して別測定する。未達なら追加のprofile→修正→再測定を行い、物理的・実装的制約と到達値を明記する。

## Phase 6: 本家対応を取り込んだTP2

TPはモデル分割と通信backendの二層に分けて実装する。

### 6A: 本家差分の取り込み

- 参照commitからQwen3.8のTP関連差分と依存変更を列挙する。`supports_tp=True`の1行だけを移植しない。
- QSAはwhole-layer配置、PLEはrankごとのmodule複製という本家設計を起点とする。GDN、gated residual、MoE、cache/state export/importまで追う。
- 最新本家への無条件の全面rebaseを先に行わず、必要な変更を追跡可能なcommit単位で適用する。
- D/M/Qのレイヤー分割経路が変わらず動くことを確認する。

### 6B: ROCm通信

- `model/model_tp_backend.py` のNCCL backendはbroadcast/gather等で `TPBackendNative` に依存する。`-tpb nccl`だけでROCm化できるとは扱わない。
- 第一候補はtorch.distributedのROCm/RCCLを用いてall-reduce、broadcast、gather等を完結させるbackend。dtype、uneven split、非参加rank、stream順序を含めて検証する。
- `model_tp_cuda.py` のCUDA runtime呼び出し、`model_tp_shared.py` のhost registration、worker spawn/teardownをHIP対応へ分離する。
- 現backendのFP32→BF16 all-reduce変換は精度に影響するため、そのまま高速化として採用せず、FP32参照との差を検証する。
- gfx1030/RCCLで性能・機能が不足する場合にnative `pg_*`/shared-memory通信のHIP実装を追加する。最初からnative全カーネルの全面移植を必須とはしない。
- timeout、rank異常終了、shutdown、再起動も試験する。hang時は今回起動したworkerだけを停止する。

### 6C: D→M→Qの順に統合

- Dでcolumn/row parallelとcollectiveを確認。
- Mでexpert配置、router、weighted reductionを確認。
- Qで本家モデルTPを有効化し、PLE/RAM、GDN/QSA、各rankのcacheを確認。
- PLE module複製とtableの物理RAM複製を区別し、RSS/PSSで計測する。5bit tableがrankごとに約30.4GiBを占める場合、親processを含むピークを確認し、必要ならread-only shared memory/mmapへ分離する。RAMに載らずswapへ落ちる構成を合格にしない。
- 1GPU・2GPUレイヤー分割・TP2を同一D/M条件で比較し、Qはレイヤー分割対TP2で比較する。

合格: TP2で数値・安定性が通ること。速度向上はprefill/decodeそれぞれ測り、両GPUで演算・通信が実行されることをtraceで確認。TP2が遅ければcollective、rank同期、細粒度MoE、QSAの非分割部分を改善する。最終的にどの条件でTP2が有利かを明示し、性能が劣る条件ではレイヤー分割を選べる状態を維持する。

## 測定規約と「遅い」の判断

| 項目 | 条件 |
| --- | --- |
| 通常測定 | batch=1、入力512/2048/8192、出力256。Qは32768も追加 |
| decode | 128-token promptだけでなく2K/8K contextでも測る。Qは32Kも別枠 |
| 繰り返し | warmup 1回以上、timed 5回以上、中央値とばらつき。spread>5%なら競合/clockを調べ再計測 |
| 入力 | seed固定の同一token IDセットをA/B共通で使用。各反復はcache clearまたは別prompt、cached_tokens=0を確認 |
| 品質 | 固定の自然文/日本語/コードcorpus。ランダムtoken性能試験とは分離 |
| 記録 | prefill tok/s、TTFT、decode tok/s、TPOT p50/p95、VRAM peak、RAM RSS/PSS、GPU状態、commit/model/environment |
| 時間計測 | 全対象GPUの完了を保証し、load/JIT、prefill、decodeを分離。profile付き計測を通常速度と混ぜない |

- 同じ構成の最良版よりprefillまたはdecodeが10%以上退化したら修正対象。
- 同条件の代替実装が10%以上速ければ切替または原因調査対象。
- traceで不要copy/synchronize/allocation、CPU fallback、重複dequant等が総時間の20%以上を占めれば優先修正対象。
- 1GPU→レイヤー分割で10%以上遅くなったらPhase 4の調査対象。
- 採用する速度改善は、正確性合格に加え、E2Eで概ね5%以上かつ測定noiseを超える改善を反復確認する。大きなregressionがある場合はshape/context別dispatch等で解決する。
- Mの暫定実用目標は2K入力時prefill 200 tok/s以上・decode 20 tok/s以上。Qの目標はPhase 5に記載。どちらも実機未測定の仮目標で、初回baseline後に根拠を添えて妥当性を評価する。勝手に達成済みへ読み替えない。
- 他engine/GGUFとの比較は参考値として併記可能だが、異なる量子化形式・品質を同一条件のA/Bとして扱わない。

## 成果物・運用

- ソース・短い検証設定・再現手順はrepo管理。巨大な重み、profile、出力logは別artifact directoryに置く。
- `rocm_tools/rdna2/` に環境チェック、correctness/benchmark実行入口、各stageのconfigを順次追加する予定。
- 各stageに `manifest.json`、`metrics.json`、`correctness.json`、`summary.md` を保存し、single-V620 baselineを以降も比較に使う。
- 変更はbuild/数値検証が通る小さな単位でcommitする。各stage終了時、速度差、残課題、次の対象を報告する。
- profilingは既存benchとPyTorch/HIPツールを主体にする。Magpieが利用可能ならkernel analyze/compare等に使う。ExLlama向けの安定Magpie benchmark backendが存在するとは仮定しない。
- この計画は順次実行を前提とする。新たな外部サービス・モデル公開等は含まない。

## 参照

- [D: Qwen3-8B 4.0bpw](https://huggingface.co/turboderp/Qwen3-8B-exl3/tree/4.0bpw)
- [M: Qwen3-30B-A3B 3.0bpw](https://huggingface.co/turboderp/Qwen3-30B-A3B-exl3/tree/3.0bpw)
- [Q: Qwen3.8-Flash-Next 3.05bpw_h5_ng5](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)
- [本家Qwen3.8 TP対応参照](https://github.com/turboderp-org/exllamav3/blob/d3739fd393337b1ff4d6c2a342b12f0c87a9592f/exllamav3/architecture/qwen4_exp.py#L307-L309)
