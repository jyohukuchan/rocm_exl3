# V620×2 レイヤー分割・高速化: Phase 3/4結果

2026-09-29。Qwen3-8B EXL3 4bpw（D）とQwen3-30B-A3B EXL3 3bpw（M）で完了。
既存のRDNA2移植済み演算とEXL3 autosplit APIを使用し、2GPU計測・配置監査・品質検証・profileを実装した。
実装はOpenCode Go/qwen3.8-flashへ委任し、Codexがレビュー修正・実機測定・監査を担当。
今回の範囲はbatch1の層分割。Qwen3.8-Flash-Next固有処理とTensor Parallelは計画のPhase5/6に残る。

## 採用した設定

- プロセス起動前に `HSA_ENABLE_SDMA=0`。直接GPU間コピーは維持し、明示的なhost bounceは0回。
- 推論中は両V620の `power_dpm_force_performance_level=profile_peak`。終了時に保存した元の値へ復元。
- D: `use_per_device=[2.7,4]` GiB、Transformer19/17層。M: `[6.1,8]` GiB、24/24層。
- FP16 KV、cache8704、chunk2048、batch1。embeddingは既存のprefer_cpu設定に従いCPU、norm/headは後段GPU。
- `doc/v620_pair_config.json` にmodel revision、UUID、API引数、runtime設定を保存。
- 専用container `rocm-exl3-v620-pair` はSDMA無効をデフォルト化した構成へ切替済み。両モデルの生成を再確認した。旧構成は `rocm-exl3-v620-pair-sdma-default` として停止状態で保持。

| 論理device | PCI | UUID |
|---|---|---|
| cuda:0 | 43:00.0 | GPU-08b2ddcbd6e6b36c |
| cuda:1 | 03:00.0 | GPU-76a08c022586fed6 |

両方gfx1030、約32GiB、PCIe16.0GT/s×16。SMI indexとHIP indexは異なるため、実行時のPCI/UUIDを確認した。
Torch2.12.0+rocm7.2、HIP7.2.53211、Triton3.7.0、host ROCr7.14 preload。
image `rocm-exl3-phase2-env:tested`（sha256:cb979a9dbf69fa98fc19a303826a3c118b6f27cc144d0a1181185a05363628dc）、
`/work/lib/exllamav3_ext.cpython-312-x86_64-linux-gnu.so` はPhase2と同じgfx1030 binary。演算kernelの追加変更はない。

## 品質・安定性

- 同一EXL3重みの1GPU参照に対し、D/Mのbulk・chunk1各1024位置、計4096比較が全てtop1一致。合格基準は99%以上、未達時にKLDで検証する方針を測定前に固定した。KLDへの切替は不要だった。
- 日本語・英語・Pythonの固定manifestを使用し、token/position/vocabの一致、全位置の回収、非有限値なしを検査。これはBF16元モデルとの100%一致を意味しない。
- 両モデルで固定corpusの8K入力＋256生成、全256 forwardのlogits有限、要求長完走、通常終了。
- 両モデルで36連続job、計72jobが通常終了。warm1＋timed5、prefill/decode×512/2K/8K。全36入力SHA256はPhase2の各モデルの入力streamと一致し、prefix hit0、投機生成なし。
- module/cacheの実deviceと、trace上の両GPUのAttention実行数を照合。decode各区間のsplit kernel数はGPUごとの層数×64と一致。
- 補助確認として、1GPU/2GPUの同条件生成列、observer有無、profile/controlの生成列も全て一致した。生成列の完全一致を品質の合格条件に置き換えてはいない。
- CPU回帰220件成功。新containerでも両GPUのFP16 GEMMと両モデルの4jobずつの生成を再確認し、全てexit0。

## 1GPUとの速度比較

単位t/s、5回の中央値。1GPUは固定したPhase2のauto基準、2GPUは上記の採用設定。重み・入力・KV・chunk・生成長256を一致させた。
| モデル | 入力 | 1GPU prefill | 2GPU prefill | 1GPU decode | 2GPU decode |
|---|---:|---:|---:|---:|---:|
| 8B 4bpw | 512 | 1131.3 | 1105.0 | 58.5 | 58.5 |
| 8B 4bpw | 2048 | 1155.3 | 1114.0 | 57.6 | 57.5 |
| 8B 4bpw | 8192 | 906.2 | 1269.0 | 47.8 | 47.8 |
| 30B-A3B 3bpw | 512 | 471.5 | 471.2 | 70.6 | 70.1 |
| 30B-A3B 3bpw | 2048 | 967.9 | 968.4 | 69.2 | 68.6 |
| 30B-A3B 3bpw | 8192 | 742.5 | 744.9 | 56.9 | 56.7 |

最終runの全12速度群のspreadは0.2〜2.2%。Dの8K prefillは約1.40倍、D/Mのdecodeは元の1GPU基準から約1%以内。
生成器の従来`time_first_token`は最初のdecode forwardの前に設定され、その初回処理が`time_generate`へ入る。比較用の既存指標を残し、
harnessに実観測の`first_token_wall_ms`と`decode_observed_tps`も追加した。8Kの実観測TTFT中央値はD約6.48秒、M約11.02秒。

### 同じruntime・policyでの対照と配置差

以下はprofile用入力の同一promptを再生した、プロファイラなしの対照。prefillは実際の最初のtokenまで、decodeは生成97〜160番目の64tokens。
全構成SDMA無効。採用配置は3反復、配置違いと1GPUは2反復。

| モデル | 構成 | 8K prefill ms | 8K decode ms/token |
|---|---|---:|---:|
| D | 2GPU auto | 6243.6 | 34.815 |
| D | 2GPU peak・採用配置 | 6562.3 | 20.932 |
| D | 2GPU peak・初期配置 | 6975.0 | 21.039 |
| D | 1GPU peak | 9730.6 | 21.498 |
| M | 2GPU auto | 10791.4 | 28.244 |
| M | 2GPU peak・採用配置 | 11020.3 | 17.634 |
| M | 2GPU peak・初期配置 | 11071.2 | 17.725 |
| M | 1GPU peak | 11333.1 | 17.838 |

Dの初期21/15層から19/17層への変更は、同じpeak設定の8K prefillで約6%改善し、decodeはほぼ同等。Mの26/22層と24/24層の差は小さく、メモリ配分が均等な24/24を採用した。
Dは複数prefill chunkのGPU実行が重なる。Mは`exllamav3/modules/block_sparse_mlp.py`のexpert_count.tolist()などprefill中のCPU readbackがあり、trace上の重なりも小さい。
この点はMのprefillが2GPUでほぼ同速となる説明と整合する。CPU readbackを除去した場合の改善量は未測定。

## 修正対象の切り分け

### 動作クロック

初期D splitのdecodeは約36〜37t/sへ低下。前段43カードのGFXCLKが約1.1〜1.3GHzへ落ち、後段は約2.4GHzだった。
GPU metrics v1.3をdriverの構造体と照合して取得した。通常のSMI/pp_dpm_sclkが0MHzを返すカードでは、この詳細値を使用した。
同じ短文試験のauto→high→profile_peak→autoは36.98→58.82→58.68→36.83t/s。最初の別high試験は改善せず、その結果も保持した。
採用したprofile_peakでは最終反復も速度が安定。層ごとのtorch.cuda.device context変更は+0.2%で不採用、host bounceも低下を解消しなかった。

**peakを待機中に固定しない。** 対象モデル未ロードの短時間測定で2枚合計の平均電力表示はauto約54W、peak約274W、復元後約66Wだった。温度・直前の負荷の影響を含む参考値。
全controllerは元policyを保存して終了時に復元し、現在は両方auto。

### SDMA経路の長い同期待ち

Mのauto・SDMA有効で同じ2K入力が約28/2.1/22秒となり、8K warmupで数分GPU同期を待つ現象を確認。GPU演算は止まり、kernel logにfault/resetなし。
待機後にpeakへ変更しても復帰しなかった。該当診断プロセスのみ停止しexit143を保持した。
SDMA無効では同一8jobを通常終了し、2K入力は2.14〜2.17秒、8K入力は10.76〜10.79秒。再びSDMA=1に戻すと28/2.1/22秒の待ちが再現し、180秒の上限でexit124となった。
そのため`HSA_ENABLE_SDMA=0`を採用。以降の品質、72job、長文、対照、移行確認は全て通常終了した。ドライバ内部の故障箇所までは断定していない。
追加のread-onlyコードレビューでは確定的な順序違反は見つからず、不要なdevice同期は追加していない。P2P自動判定の初回probeを原因とする推測は、実際の停止stackと時点が一致せず採用しなかった。

## 転送とGPU処理時間

層境界のhidden stateはFP32。KVがFP16であることとは別。Dは16KiB/token、Mは8KiB/tokenを0→1へ1回転送し、KVは層のdeviceに常駐。
2K prefillは2copy、8Kは5copy（prefill chunkと最初のdecode token）。Dは32/128MiB、Mは16/64MiB。
64-token decode区間は両モデルとも64copy、D1MiB・M512KiB。コピーAPIのhost呼出時間は約0.03ms/token。これはpure DMA時間ではない。
同期APIは64token区間で66回（samplerの64回＋計測終端の2回）。その待ちにはGPU計算完了待ちが含まれるため、CPU処理の無駄とはみなさない。

実サイズのFP32コピーを、毎回異なる入力と転送先のpoisonで検査し、staleデータが一致判定を通らないようにした。2方向×6サイズ×direct/host-bounce×13回の計312copyは全ビット一致。
以下はprofile_peak・SDMA無効、allocationと両GPU同期を含むeffective値。

| payload | 0→1 direct | 1→0 direct |
|---|---:|---:|
| 8KiB | 0.046ms | 0.049ms |
| 16KiB | 0.046ms | 0.049ms |
| 16MiB | 24.30GB/s | 25.04GB/s |
| 32MiB | 17.75GB/s | 16.88GB/s |

### 8K decodeのGPU kernel時間（ms/token）

| モデル | cuda:0 | cuda:1 | 量子化Linear関連 | Attention関連 | その他GPU |
|---|---:|---:|---:|---:|---:|
| D | 9.295 | 9.086 | 12.275 | 5.689 | 0.417 |
| M | 6.873 | 7.189 | 7.043 | 5.509 | 1.513 |

量子化Linear関連は復号と積和の融合kernelを含み、dequant単独時間ではない。GPU時間は計測下の値、3反復の独立した中央値であり、列の和は厳密一致しない場合がある。
全24閉区間・768decode tokensを監査し、欠落・区間跨ぎ・未分類kernelなし。agent_infoのPCIからAgent1=43、Agent2=03を確認。
プロファイラは特に8K prefillのGPU重なりに影響する。Dの8K prefillは無観測約6.56秒に対し計測下約8.71秒。GPU内訳をそのまま通常のwall時間へ足し合わせない。
copy CSVのDirectionにGPU→GPUをHOST_TO_DEVICEと記録する例があり、実endpointのAgent IDsを保存・使用した。CSVにないbyte数は捏造せず、alias観測で取得。

## VRAMと後続モデル向け予算

| モデル | Torch peak cuda:0 / cuda:1 GiB | board使用量の観測最大 GiB |
|---|---:|---:|
| D | 2.799 / 2.902 | 3.414 / 3.525 |
| M | 6.259 / 6.227 | 6.917 / 6.870 |

board値は同条件native control中の0.5秒pollで、driver/runtimeを含むが短いpeakを取りこぼしうる。Torch値はallocatorのpeak。
Dのallocatedはwarm後一定。Mは2K用bufferが各GPU128KiB増えた後一定。reservedは形状拡張時に増え、Mの8K prefillでは追加2MiBを確保したが、その後の全18decode jobでallocated/reservedとも一定。
Phase5は `use_per_device=[28,28]` GiB程度を開始点とし、各GPUに約4GiBの余地を残す。新モデルのcache/scratch/native割当は未検証で、実測したheadroomが1GiB程度以上あることを確認してから予算を詰める。

## 再実行

専用containerはSDMA無効・両UUID可視・既存binaryを設定済み。下記はホスト上で実行するM benchmarkの例。Dはmodel-dirとbudgetを設定JSONに合わせる。
```bash
(
  set -e
  v620_policy_43=/sys/bus/pci/devices/0000:43:00.0/power_dpm_force_performance_level
  v620_policy_03=/sys/bus/pci/devices/0000:03:00.0/power_dpm_force_performance_level
  v620_old_43=$(cat "$v620_policy_43")
  v620_old_03=$(cat "$v620_policy_03")
  restore_v620_policy() {
    printf "%s\n" "$v620_old_43" | sudo -n tee "$v620_policy_43" >/dev/null
    printf "%s\n" "$v620_old_03" | sudo -n tee "$v620_policy_03" >/dev/null
  }
  trap restore_v620_policy EXIT
  printf "profile_peak\n" | sudo -n tee "$v620_policy_43" "$v620_policy_03" >/dev/null
  docker exec -e HSA_ENABLE_SDMA=0 rocm-exl3-v620-pair timeout -k 5 1200 \
    python /src/rocm_tools/rdna2/bench.py \
    -m /work/models/qwen3-30b-a3b-exl3-3bpw --use-per-device 6.1 8 \
    --contexts 512 2048 8192 --new-tokens 256 --warmup 1 --repeats 5 \
    --cache-tokens 8704 --max-chunk-size 2048 --seed 1234 \
    --json-out /work/runs/two-gpu/recheck-m.json
)
```

Python APIではCacheを先に作り、設定JSONの`models.M.load_kwargs`を`model.load(**kwargs)`へ渡す。split時に`device=`を併用しない。
既存のmodel_init CLIを使う例では `-gs 6.1,8 -cs 8704 -chunk_size 2048` が対応する。budgetの実単位はGiB。

## プロファイラの制約と証跡

通常推論・quality・benchmark・controls・移行確認は全てexit0。SDKを付けた終了処理だけはHSA queue破棄でSIGSEGVが残る。
SDMA無効でも通常profile終了はexit139でCSV保存前に落ちたため不採用。採用traceは全ROI・model cleanup・JSON保存完了後にSIGTERMで保存を要求し、
`tool finalization`完了後も残った当該PIDのみ停止した（exit137）。これは正常終了とは報告しない。停止は計測区間外で、全区間を別途監査した。

Artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/two-gpu`。

- `phase3/`、`phase4/`: 各manifest.json / metrics.json / correctness.json / summary.md。
- `completion-evidence.json`, `final-pair-audit.json`: 要件別監査、主要artifact SHA256、24区間の整合検査。
- `*-bench-peak-sdma0.json`, `*-quality-*-peak-sdma0*.json`, `*-long-peak-sdma0.json`: 最終のraw測定・品質。
- `*-sdma0-final*.json` と `*-trace/`: 同条件のcontrol/observer/配置違い/1GPU/profile、raw CSVとsummary。
- `acceptance-sdma0-process-status.json`, `profiles-sdma0*-process-status.json`: 実command・終了code・policy復元。
- `sdma-triage.json`, `m-auto-policy-intervention.json`: 同期待ちの再現と明示停止。失敗・timeout・不採用runも保持。
- `peer-copy-fp32-sdma0.json`, `deployment-final.json`, `deployment-process-status.json`, `idle-policy-power.json`: 実copy、最終container、消費電力の参考値。
- 実行controller・bootstrap・peer probeはartifact rootの2階層上（`/work`直下）。再実行時は出力prefixを変え、古いready PIDを再使用しない。
