# Qwen3.8 Flash Next / V620 pair — 実行・性能検証結果

2026-09-29。Qwen3.8 Flash Next EXL3 3.05bpwをV620×2のlayer split、PLE/Engram RAM、投機的デコーディング無しで実行。32K+256生成・有限logits、日本語/コード生成、36連続job×2電力設定で正常終了を確認した。8K入力時はautoでprefill388.7/decode24.6tok/s、profile_peakで374.9/36.1tok/s。以下に固定条件、検証経過、正式結果、再現用コマンドを保存する。

cached prefill/decodeのtop-1一致率97.75%前後は既知事項として残す。ユーザー指示に従い、明らかな実推論異常がなければ暫定99%閾値だけで進行を止めない。Tensor Parallelは後続Phase6であり、今回の実測はTPではない。

## 固定条件

- モデル: `turboderp/Qwen3.8-Flash-Next-exl3`、branch `3.05bpw_h5_ng5`、revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`。
- HF metadataの全ファイル合計: 85,139,442,313 bytes。本体EXL3 3.05bpw、head 5bit、ngram専用量子化。全サイズ・LFS SHA256照合済み。PLEはRAM常駐を実測確認済み。
- text-only、batch1、全routed expertはGPU常駐。draft model、MTP、n-gram draftingを使わない。
- 基準環境: `doc/v620_pair_config.json`。V620 UUID順43→03、`HSA_ENABLE_SDMA=0`。性能測定時のみ`profile_peak`、終了時に元のpolicyへ戻す。
- load budget `[28,28]` GiBを検証済み。32K+256時のboard sampled peakは約29.043/24.848GiBで、scratch/cache/driver込みでも各GPUに余裕を残す。
- 48層: GDN36、QSA12。GDN key/value heads16/48、head_dim128。QSA main head_dim256、indexer head_dim128、token budget2048。PLEは第2層前。

## 検証・改善の順序

1. GDN recurrent/chunked prefill、QSA選択・attention、gated residual/norm、PLEの数学参照との比較。正確性を速度測定の前提とする。
2. モデル全ファイルの取得・サイズ/SHA256検証、RAM実配置、2GPU層配置・再帰状態・indexer cacheの配置確認。
3. 短文生成から512/2K/8K/32Kへ拡大。32K+256token、有限logits、30連続job、VRAM/RSS/page fault、EOS境界・状態reset・prefix再利用を検証。
4. 固定の日本語/コード課題とcorpusで品質を記録。bulk/chunkedの一致は内部整合性検査であり、独立参照の代わりにはしない。量子化誤差と移植誤差を分けて評価する。
5. プロファイラ無しのTTFT/decode基準値を保存し、MoE/dequant、GDN、QSA、PLE gather/copyと同期へ分解。候補修正は独立の正確性確認後、同一条件でend-to-end再測定。共通kernel変更時には既存D/Mも回帰確認。

完了条件・暫定性能目標は`doc/rdna2_port_plan.md` Phase5を維持する。未達・未測定を完了へ読み替えない。Tensor Parallelは同計画Phase6で扱う。

## 調査の経過（以下の「未完了」「実行中」は各時点の記録）

- 元repo HEAD `9d32c69`、作業開始時clean。既存の2GPU D/M検証済み環境を再使用。
- 既存GDNテストの小規模参照一致2件・ビット再現性3件が実V620で成功（5 passed、19 deselected）。Qwen3.8全体の正確性は未証明。
- 重みの取得と専用検証harnessを準備中。現行top1 collectorは再帰モデルを拒否していたため、GDN状態とPLE履歴を正しく引き継ぐ検証経路が必要。
- artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38`。取得metadataは`model-source.json`、GPU小規模検証は`gdn-existing.log`。

### GDN prefillのRDNA2問題と候補比較

Qwen実形状（batch1、key/value heads16/48、head_dim128）のBF16 vendored FLA prefillは、LLVMの`Cannot select: intrinsic llvm.amdgcn.fdot2.bf16.bf16`でprocess abort（exit134）。native recurrent kernelの実行後に発生し、ログは`gdn-shape-probe.log`。

入力q/k/v/betaをFP32へ昇格したFLA経路は65tokenで成功。独立の逐次PyTorch参照に対して出力最大絶対誤差5.95e-5、状態1.94e-7。初回JIT約157.6秒はsteady推論時間とは分ける。

さらにnative recurrentとFP32 chunkを65/512/2048tokenで比較し、全て出力・状態の`atol=0.003, rtol=0.02`判定に合格。2048tokenでの最大絶対差は出力1.22e-4、状態9.54e-7。両GPUのpolicyをprofile_peakへ揃えた単体予備測定（7回、allocation/conversion/hostsync込み）は以下。

| tokens | native recurrent（中央値ms） | FP32 chunk（中央値ms） |
|---|---:|---:|
| 65 | 0.556 | 2.881 |
| 512 | 4.052 | 9.219 |
| 2048 | 16.206 | 36.133 |

`gdn-backend-compare-peak.json`にraw samples、`gdn-peak-process.json`にexit0と元policy（両auto）への復元結果を保存。native経路をRDNA2 prefillにも使用する案は有望だが、これはモデル全体の速度改善・品質合格の証拠ではない。FP32互換経路と専用検証harnessの実装はOpenCodeへ委任中、diffレビューと実機再試験は未完了。

### 実装・Engram検証の追加進捗

- 全モデルファイル取得完了。全サイズ一致、LFSファイルのSHA256をHF metadataと照合済み（`model-sha256-verification.json`, `complete=true`）。固定入力8件・評価位置1024の`manifest.json`も作成。
- `7aa1465`: gfx103xのGDN chunk BF16入力をFP32へ昇格し、BF16出力へ復元。CPU50testsと両V620の実入力比較をrootが再検証。元BF16入力の出力maxabs1.53e-5、状態1.94e-7（65token、逐次参照）。
- `58e62e0`: benchmarkの`--ngram-ram`、実テーブル所在/RSS/fault、GDN/PLE状態とQSA cache配置監査。CPU253tests。rootレビューでCPU/meta計算状態、空状態、欠けたQSA planeを拒否するよう強化。
- `7b417f9`: teacher-forced収集器の再帰状態持越/解放・PLE履歴検査。CPU全282tests成功。実モデルでのlogits収集は未完了。
- `43ad8cf`: RDNA2 GDN prefillをnativeへ切替、`EXL3_RDNA2_GDN_CHUNK=1`でFP32 chunk対照へ戻せる。rootレビューで`save_state=False`かつ既存状態ありの場合は従来の非破壊chunk経路を維持。関連CPU63tests。

Engramの独立検証:
- 合成K5 codecは両GPU×1/16/513行の6caseでCPU参照とbit一致（`ngram-codec-probe.json`）。
- 実テーブルは`trellis_ram`、CPU tensor計32,640,156,672bytes、ロード約10秒。実測RSS peak 34,079,104KiB。モデル全体を載せた際のRSS/VRAMとは別測定。
- EOSを含む1/3/17/129/513tokenを独立torch hash+codec参照と比較。513tokenの1,313,280要素中2要素のみ差があり、最大絶対差1.91e-6（FP16の1step以内）。その2要素はFP64計算を丸めた値がtorch参照側に一致。native側の丸め差の詳細原因は未断定。完全bit一致という当初の検査は不合格として保存し、許容幅と生値を`ngram-actual-probe-rounding.json`へ記録。
- 同じfast経路で一括と7token分割の結果は全case bit一致。prefetchを含むhash/状態境界の内部整合性検査であり、モデル全体の品質証明ではない。

新しい問題:
- GDN convも33token以上のTriton経路でBF16 dot命令のLLVM abortを再現（`conv33-repro.log`）。nativeへ限定した独立probeは33/64/257/2048tokenおよび40token historyの出力参照比較に合格、状態はbit一致（`conv-native-probe.json`）。RDNA2だけ長文nativeへ分岐する修正を委任中。
- `max_chunk_size=32`での最初の全体ロード試行はloaderのPAGE_SIZE=256制約で拒否。次の256chunk/native conv候補・予算[28,28]GiBは実VRAM不足で停止。まだ全体生成成功ではない。例外時の層/割当/最大transientを`load-oom-trace.py`で診断中。
- GPU前提テストは実機試験・rootレビューで参照側のimport/因果conv位置/状態clone/入力contiguous/QSA並び順等を修正中。最終固定版の全テスト成功は未確認。途中版の成功数を全合格とは扱わない。

### 最初の全体生成成功と割当エラーの切り分け

`load-oom-trace.json`ではlayer18でGPU0の実割当18.95GiB・空き12.51GiBなのに128MiB allocationが失敗し、続くGPU1も割当42MiB・空き31.73GiBで同じ失敗。単純なVRAM総量不足ではない。コンテナのmemory limitは無し、cgroup OOM event無し、nofile soft limitが1024だった。

プロセス内の`RLIMIT_NOFILE` soft limitだけを65536へ上げた候補で、同じ[28,28]GiB予算・Engram RAM・256chunk・cache4096のロードと2job生成が成功しexit0。途中の実FD数1610を確認。GPU0がlayer26で設定予算28GiBに達してGPU1へ移る正常なsplitも記録できた（その時点FD1496）。`first-smoke-nofile.json`はok=true、失敗0、spec_decode.enabled=false、RAM表32,640,156,672bytes・実配置監査OK。診断traceとnative conv一時overrideを含むこのrunの速度は性能値として採用しない。

`37a96d5`で長いconvのRDNA2 native分岐を正式実装。CPU29testsと、override無しの33/64/257/2048token・40token historyを独立参照と比較し成功、状態bit一致（`conv-production-probe.json`）。

以後、モデル全体は一時kernel override無しの正式コードで動作。プロセスのnofile soft limit65536は必要条件として継続する（現在のコンテナ既定は1024のままなので、実行コマンドで明示的に引き上げる）。

`first-bench-512.json`: auto policy、max_chunk2048、cache8704、生成32、warm1+測定2、prefill/decode計6jobs、失敗0・通常終了。prefill中央値259.0tok/s（spread4.4%）、decode24.0tok/s（spread1.2%）、ITL p50約40.50ms。この512tokenの予備値を8K/32Kや最終最適値へ外挿しない。ロード約44秒、両deviceのQSA cache各6、recurrent state21/16。transformer module数27/22はPLEを1module含むため、48decoder層の分割数とは異なる。

現在は`candidate-chunk1.json`/`candidate-chunk1-logits.f32`を生成する固定8case・1024位置のteacher-forced検証を実行中。長文、30job、独立モデル参照、最終profile/性能改善は未完了。

### 品質比較の不合格とshared expert gateの競合

`candidate-chunk1-v2.json`は8case/1024位置、2030chunk、全logits有限、state作成/解放各8、PLE履歴照合・release_failures/carry_notes無しでexit0。`candidate-bulk-v1.json`も同じ1024位置でexit0。ただし両者のtop1一致は832/1024=81.25%、KL(P_bulk||P_chunk1)平均0.36994nats、p99 3.73589natsで不合格（`chunk1-vs-bulk-top1.json`, `chunk1-vs-bulk-kld.json`）。従って上記512予備速度は正しいモデル演算としての性能合格値には使わず、修正後に測り直す。

rootによる切り分け:
- 同じ第1tokenの埋め込み/stream展開はbit一致。最初のdecoder層（PLEより前）から差が発生。
- GDN、hyper-connection、shared expert単体の差は小さいが、MoE合成出力でrms差0.01644（値のrms約0.0466）。
- 最初のMoE入力を完全固定し、選択expert10個とrouting weightsも一致させて比較。1/8/16rowおよびgeneric1rowで大差、64/183rowはEXL3 reconstruct参照に近い。個別expert Linearの高速/展開結果は近く、中間幅640の末尾切落とし仮説では説明できなかった。
- 誤差の99.9999817%がshared expertのgate値の差で説明できる（`moe-shared-gate-fit.json`）。実gateのFP64内積は-0.51480674、通常Linearも-0.51480669、sigmoidは0.37406739。融合projection kernelは0.20132904を返す。bias無し、pre/post_scale=1を確認（`shared-gate-probe.json`）。
- `reduction.cuh::block_reduce_sum_broadcast_f`はwarp0のshuffle-down集計後に全laneが`shared[0]=v`を書いていた。完全な合計を持つのはlane0だけで、他laneは部分和。RDNA2でこの競合により誤った値がbroadcastされる。
- モデル不要の再現: onesの内積を1にしたdim128でsigmoidが期待0.73106→実0.5、dim2560で期待0.73101→実0.68993。均一入力の1024/2048/4096は問題を隠す（`shared-gate-synthetic-baseline.json`）。

単一writer化＋直接projectionの数値回帰テストをOpenCodeへ委任中。native再build、回帰テスト、実モデル再検証は未完了。Pythonの参照projectionだけを差し込む`candidate-chunk1-gate-ref`介入実験を並行して実行中（品質原因の検証用であり運用/性能経路ではない）。

`0ec44fc`: GDN/conv/gated norm/hyperconnections/QSA/状態のGPU前提テスト最終44件がroot実機runで成功。これは上記の実重みMoE gate不具合を覆い隠す合格条件ではない。native binaryはまだ旧`26c326b6...3156f`で、修正後は新binary identityを保存し共通kernelのD/M回帰も必要。

### gate修正完了・長文検証・進行基準の更新

`1885e20`でbroadcast reductionをlane0だけの書込みへ修正。gfx1030 native buildはexit0、候補 `/work/lib-qwen38-reduction` のSHA256は `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`。旧 `/work/lib` は保持しており、実行時は候補をPYTHONPATHに明示する。新projection 8件とarchitecture 44件の実GPUテストは計52件成功。旧binaryでは新テスト7件が失敗し、回帰検出力を確認した。

同じ実モデルMoE入力・routingを使った独立reconstruct参照比較でも、1/8rowの最大絶対差は0.14969から2.575e-5、RMS差は0.016426から7.026e-6へ縮小（`moe-isolated-fixed-probe.json`）。これは明確なgate不具合の修正確認。

残る経路差（同一EXL3重み・固定1024位置）:
- cached prefill対native decodeは1001/1024=97.7539%一致、KL平均0.005919・p99 0.054472。
- uncached bulk対cached prefillは1014/1024=99.0234%一致。
- `EXL3_QKV_SLICE=0`はnative decode logitsのSHA256まで同一で、今回の差に影響しない。
- 選択位置のうち次tokenラベルを持つ1016位置のPPLはuncached bulk 3.79696、native decode 3.81172。全corpus PPLとは区別する（`sampled-position-ppl.json`）。

ユーザーは「実モデルの推論で明らかな異常値が発生しないなら問題を放置して先に進んで」と指示。上記の残差は保存し、暫定99%/KLD基準だけを理由に原因追跡を続けない。量子化品質や完全な経路同値が証明されたという意味ではない。

修正後の正式経路で8K/32K入力それぞれ256token生成に成功、全256logit行が有限・正常終了（`long-8192-native.json`, `long-32768-native.json`）。投機生成なし、prefix cache miss、Engram実テーブルCPU RAM 32,640,156,672bytes、Swap 0。32KのtorchピークはGPU0 30,172,561,920bytes / GPU1 25,656,937,472bytes、0.5秒サンプリングのboard VRAM最大は31,184,961,536 / 26,680,176,640bytes。各stepで有限性を検査するrunなので正式性能値にはしない。

正式性能測定はauto/profile_peak各36job（prefill/decode × 512/2048/8192入力 × warmup1+測定5）、生成256tokenを順次実行中。`formal-benchmarks-process.json`で終了状態・policy復元を管理。残る作業は測定結果の集計、全corpus NLLの記録、共有native変更のD/M回帰、起動条件の再現性確認。

### 再現用起動条件

コンテナ `rocm-exl3-v620-pair` はV620の2枚だけを公開し、`/src`が本repo、`/work`が `/home/homelab1/datapool/rocm-exl3-rdna2`。コンテナ既定の旧binary/nofileに依存せず、次のように明示する。GPU power policyは外から変更せず現在値を利用する（測定時のpolicyは結果と併記）。

```bash
docker exec -e PYTHONPATH=/work/lib-qwen38-reduction:/src \
  -e HSA_ENABLE_SDMA=0 -w /src rocm-exl3-v620-pair \
  bash -c 'ulimit -S -n 65536 && exec python rocm_tools/rdna2/bench.py \
    -m /work/models/qwen38-flash-next-exl3-3.05bpw \
    --use-per-device 28 28 --ngram-ram --mode bench \
    --contexts 512 2048 8192 --new-tokens 256 --warmup 1 --repeats 5 \
    --max-chunk-size 2048 --cache-tokens 8704 \
    --json-out /work/runs/qwen38/reproduce-bench.json'
```

`--ngram-ram`はEngramのCPU RAM常駐指定。投機デコーディング用のngram matchingとは別機能で、本harnessはdraft model無し・ngram matching無効を実体で検査する。32K検証はcache33280を使用。上の8704は512/2K/8K測定専用。

### 修正後の正式性能測定

`formal-auto-36.json` / `formal-peak-36.json`: 各36連続job、各条件warmup1+測定5、失敗0、全prefix cache miss、通常exit0。2 runの全入力token hash・モデルfingerprintが一致。batch1、EXL3 3.05bpw（head/ngram 5bpw）、Engram RAM、生成256token、投機生成なし、max_chunk2048、cache8704、予算[28,28]GiB。最後にGPU policyを両autoへ復元済み。集計は`formal-summary.json`。

| 入力token | prefill auto / peak (tok/s) | decode auto / peak (tok/s) | 終了処理込みdecode auto / peak (tok/s) |
|---|---:|---:|---:|
| 512 | 259.26 / 261.22 | 24.88 / 36.41 | 22.92 / 32.27 |
| 2048 | 413.81 / 398.23 | 24.80 / 36.10 | 22.84 / 31.54 |
| 8192 | 388.66 / 374.87 | 24.59 / 36.14 | 23.57 / 34.32 |

全て5回の中央値。decodeはgeneratorのtime_generateによる従来値、右列は最初のtoken受取から最後のtoken受取までの実測値で、最終iterate内の終了処理を含む。autoの最終token受取は0.4〜1.4秒かかり、この差を無視しない。8Kの実TTFT中央値はauto21.716秒/peak22.473秒。decode ITL p50/p95はauto40.38/40.96ms、peak27.50/27.81ms（各1275間隔）。

profile_peakで8K decodeは約47%向上、終了処理込みでも約46%向上。8K prefillは約3.5%低下。生成重視ではprofile_peakが有利だが、設定を恒久変更せず測定後autoへ戻した。2K auto prefillはspread5.1%、2K decode jobのTTFTは18.4%と揺れがあり、その値から小さい改善差を論じない。8K prefillのspreadはauto0.71%/peak0.57%程度、decodeはauto2.28%/peak0.3%程度。

両policyでboard VRAM sampled peakはGPU0約28.853GiB/GPU1約24.655GiB、torch peak約27.976/23.625GiB。終了時host RSSはauto38.37GiB/peak38.40GiB、推論区間major fault0。Engram表は30.40GiB、再帰checkpoint cacheの既定上限は別途4GiB。VRAM使用量はjobごとのsnapshotを保存し、増加はcontext/cacheと一時割当に対応する。

gpu_metricsのGPU1 activityはauto run全体で99%固定だったため、この値だけでGPU稼働割合を推定しない。clock/power/VRAMは変動しており生値を保存。測定中の他GPUコンテナはsleepのみ（`formal-process-inventory.json`）。終了時の待ち時間はキャッシュ整理/RAM返却の計測で切り分ける。

### 実生成と終了処理の追加確認

`chat-sanity-native.json`: モデル付属chat template、thinking無効、greedy、投機生成なし。日本語のVRAM/RAM説明は88token、順序保持の重複除去Python関数は91tokenでEOS正常終了。全179logit行が有限で、目視で明らかな崩壊・反復異常なし。生成関数は空リスト・重複・負数を含む4caseで期待値と一致（`chat-code-check.json`）。これは限定的な生成smokeであり総合能力評価ではない。初回は診断wrapperの引数名不一致でTypeErrorになったため、修正して再実行。失敗ログは`chat-sanity-harness-failure.*`に保持。

`idle-observed-auto-2k.json`: 2Kだけの追加12jobは0fail・exit0。`idle-observation.json`ではdecode最後のiterateが92.7〜343.3ms、そのうちqueue終了処理が51.8〜303.4msで、残り約40msは通常decodeと一致。RAM返却の`malloc_trim`とstranded checkpoint除去中の同処理が主な終了コストで、page defragは最大0.004ms未満。この追加runのhost RSSは34.36→35.24GiB、保持checkpointは最大0.759GiBで、正式36jobの約38.4GiBとは状態が異なる。従って正式runの0.4〜1.4秒の内訳をこのrunだけで断定しない。

2K追加prefill中央値412.5tok/s、spread21.4%で揺れを再確認。正式結果の小さなprefill差は8Kの同一入力比較を中心に評価する。終了時のメモリ返却を無効化する変更は加えず、現在の到達性能と終了コストを分けて記録する。

全corpus評価用に`3a14430`で`collect_nll.py`を追加。OpenCodeの実装をrootがレビューし、例外後の誤った位置対応を防止する処理と不完全artifact保存を強化。CPU294tests・compile成功。固定manifestの全次tokenラベル2022個を評価対象とし、1024選択位置のPPLと区別する。

### 全corpus評価・共通nativeの回帰確認

全corpus NLL追加時、最初のcase後にGPU page faultが2回再現した。損失計算をGPU FP64からCPU FP64へ移しても再現し、FP64計算だけが原因という仮説は否定。全ラベルの評価では最後の入力tokenに次tokenラベルが無いため、その最終forwardをharvestしない。以前のmanifestは最後の位置を選択しており、harvestの`.item()`が暗黙の同期を担っていた。評価collectorは未完了のGPU処理がある状態で次caseへ進み得た。

`c3183e8`: 再帰stateを返却する前に全使用GPUを同期するケース境界処理を追加。生成エンジンは無変更、各tokenでの追加同期ではない。同じCPU損失計算のまま境界同期だけを追加すると全8case完走・exit0。CPU294testsも成功。故障runのログと途中GPU coreは別名で保存し、故障済みの当該PIDだけを終了、GPU resetは実施していない。

`full-corpus-nll-native.json`: 全2022ラベル、mean NLL **1.33130964**、PPL **3.78599843**、complete=true、errors0。以前のnative decode結果と共通する1016位置はtop1 **1016/1016一致**（`full-nll-overlap-check.json`）。このrunはCPUでFP64損失を計算し、そのmetadata表示フィールドの追加前に起動したもの。PPLは本固定corpusに対する値であり、独立の未量子化モデルとの同等性は主張しない。

`dm-native-regression.json`: 新nativeで既存D（Qwen3-8B 4bpw）/M（Qwen3-30B-A3B 3bpw）も各1024位置を再収集し、両方とも従来bulk結果と1024/1024一致、通常exit0。新nativeへの変更で両モデルの測定対象位置に退行なし。

`prefix-reuse-native.json`: 同じ2048token入力を2回生成し、1回目cache miss、2回目1792token再利用。両方32token・全logits有限・生成ID列も一致・正常exit0。最初の試行はロード前のhost RAM guardで停止した（`prefix-reuse-memory-guard.*`）。109GiBホストでZFS ARCが約78GiBを占有し、MemAvailableが必要量を下回っていたため。他に大きなRSSのprocessがないことを確認し、この診断だけ`EXL3_HOST_MEM_RESERVE_MB=0`で事前guardを解除。ARCは実際に回収されて完走した。OS/ZFS設定は変更していない。

運用時はEngram約30.4GiBとその他のhost working set分のRAMを確保すること。既定guardはMemAvailableだけを見るため、ZFS ARCが大きいとロードを拒否し得る。上記guard解除はRAM量・回収可能cacheを確認した診断条件であり、正式72jobの速度値はguard解除無しのrun。再現コマンドは既定guardを維持する。

Phase5の到達点: 32K+256、連続36job×2条件、実生成、prefix再利用、全corpus記録、D/M回帰、新nativeの再現用起動条件を確認済み。8K実TTFT約22秒・decode24.6〜36.1tok/sで暫定実用目標を満たす。残る経路間top1差はユーザー指示で保留。TP2、未量子化モデルとの総合品質比較、全kernelの最適化完了は今回の結果に含めない。
