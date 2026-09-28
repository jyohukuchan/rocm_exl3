# V620×2 層分割: Phase 3/4（検証中）

2026-09-29。対象は単一GPUで検証したQwen3-8B EXL3 4bpw（D）と
Qwen3-30B-A3B EXL3 3bpw（M）。TPおよびQwen3.8大型モデルは後続段階。
この文書は途中経過であり、正式な品質・速度・連続実行の合格宣言ではない。

## SDMA無効・固定性能設定での合格結果（処理時間の最終監査は継続中）

推奨候補は **`HSA_ENABLE_SDMA=0` + 推論中のみ両V620を `profile_peak`**。
Dは `[2.7, 4]` GiBで19/17層、Mは `[6.1, 8]` GiBで24/24層。
同じ量子化checkpointの1GPU参照に対し、両モデルともbulk/decode各1024位置のtop1が100%一致。
両モデルで8K+256の全logits有限、36連続jobの通常終了を確認した。
全36入力SHA256は元の1GPU benchmarkと一致。FP16 KV/cache8704、batch1、chunk2048、生成256、seed1234、warm1+timed5。
1GPU欄はPhase2のauto基準、2GPU欄はprofile_peak・SDMA無効。同じpolicyでの対照も別途取得する。

| モデル | 入力 | 1GPU prefill t/s | 2GPU prefill t/s | 1GPU decode t/s | 2GPU decode t/s |
|---|---:|---:|---:|---:|---:|
| 8B 4bpw | 512 | 1131.3 | 1105.0 | 58.5 | 58.5 |
| 8B 4bpw | 2048 | 1155.3 | 1114.0 | 57.6 | 57.5 |
| 8B 4bpw | 8192 | 906.2 | 1269.0 | 47.8 | 47.8 |
| 30B-A3B 3bpw | 512 | 471.5 | 471.2 | 70.6 | 70.1 |
| 30B-A3B 3bpw | 2048 | 967.9 | 968.4 | 69.2 | 68.6 |
| 30B-A3B 3bpw | 8192 | 742.5 | 744.9 | 56.9 | 56.7 |

全12速度群のspreadは0.2〜2.2%以内。Dのallocated VRAMはwarm後一定。Mは2K向けbufferが各device128KiB増えた後、以降の全groupで一定。
最新の根拠は `runs/two-gpu/acceptance-sdma0-audit.json` と `acceptance-sdma0-process-status.json`。
通常の推論・品質・長文試験は全てexit0。GPU policyはcontrollerのfinallyで元のautoへ復元した。

### 速度低下と同期停止の切り分け

- 初期のauto設定ではD decodeが約36〜37t/sへ低下。GPU間コピーは数値一致し、転送API時間は小さい。
- GPUの詳細metricsで前段43カードのGFXCLK低下を確認。通常のSMI/pp_dpm_sclkは0MHz等を返し、このカードのコアクロック確認には使えなかった。
- 同じ短文試験のauto→high→profile_peak→autoは36.98→58.82→58.68→36.83t/s。最初の別high試験では改善しなかった事実も保持し、profile_peakを確認対象に選んだ。
- 層ごとのdevice context変更は+0.2%で不採用。host bounceも速度低下を解消しなかった。
- 別にSDMA有効・autoで長いGPU同期待ちが再現。Mの同じ2K入力で約28/2.1/22秒、8K warmupで長時間停止。再有効化した確認でも28/2.1/22秒となり、試験全体は180秒でtimeout(exit124)。
- SDMA無効では同じMの2K入力が2.14〜2.17秒、8K入力が10.76〜10.79秒で完了し、さらに上記の両モデル36job・品質試験を通過した。ドライバ内部の原因までは断定しない。
- `profile_peak`は待機時にも高クロックを維持するため、実行期間に限定し元の設定を保存・復元する。

生成器の従来`time_first_token`は最初のdecode forwardより前に設定され、初回token待ちが`time_generate`に入る。
既存比較用の値を残し、harnessに実際の`first_token_wall_ms`と`decode_observed_tps`も追加した。
初期runの大きな外れ値は削除していない。現在のSDMA無効の正式runでは全群が5%のばらつき基準内。

## 環境と再現

既存 `rocm-exl3-phase2-env:tested` imageから `rocm-exl3-v620-pair` containerを作成。
`/src` は本repo、`/work` は `/home/homelab1/datapool/rocm-exl3-rdna2`、
`/work/lib` は単一GPU試験と同じgfx1030 extension。
Torch 2.12.0+rocm7.2、HIP 7.2.53211。ROCR_VISIBLE_DEVICESの順序:

| 論理device | PCI | GPU UUID |
|---|---|---|
| cuda:0 | 43:00.0 | GPU-08b2ddcbd6e6b36c |
| cuda:1 | 03:00.0 | GPU-76a08c022586fed6 |

両カードともgfx1030、約32GiB。実行時のTorch propertiesでPCIとarchitectureを確認。
PCIeは両方16.0GT/s ×16、別root配下。短いFP16行列演算は両deviceで数値一致・正常終了。
SMIのindexはこの論理indexと異なるので流用しない。他workloadは停止していない。

## GPU間コピー

`pair_copy_probe.py` を使い、各方向について256B、4KiB、8KiB、4/8/16/64MiBを
直接 `.to()` と `cpu().to()` で比較。各13回（warmup3、timed10）、全コピーをCPU側の
独立した正解tensorと比較し、両方向・全sizeで一致。peer capabilityも両方向true。
各測定前後に両GPUを同期する。以下はallocation・同期を含む実効値で、DMA単独時間ではない。

| 方向 | size | direct中央値 | host bounce中央値 |
|---|---:|---:|---:|
| 0→1 | 8KiB | 0.070ms | 0.115ms |
| 1→0 | 8KiB | 0.066ms | 0.105ms |
| 0→1 | 16MiB | 14.64GB/s | 9.61GB/s |
| 1→0 | 16MiB | 19.49GB/s | 9.58GB/s |

直接コピーの数値検証に合格したため、既存の自動検出を使う。host bounceを強制する必要はない。

## 初期配置と長文検証

`model.load(use_per_device=..., max_chunk_size=2048)` の既存autosplit APIを使う。
budgetの単位はGiB。cache8704、FP16 KV、batch1、投機生成なし。
実module.deviceとcacheのk/v tensor.deviceを読み出して配置を確認した。

| モデル | budget GiB | cuda:0 / cuda:1の層数 | 8K+256検証のpeak allocated GB |
|---|---|---|---|
| D | [3, 4] | 21 / 15 | 3.224 / 2.890 |
| M | [6.5, 8] | 26 / 22 | 7.010 / 6.098 |

どちらも連続した層group、embeddingは既存のprefer_cpu設定に従いCPU、最初のTransformer層はcuda:0、最終norm/headはcuda:1。
KV cacheは対応するAttentionと同じdeviceに配置。短い32-token生成が両方で正常終了。
loadを含むcopy counterはdirect33、bounced0、probe1。

さらに既存の固定corpusから同じ8K入力を構成し256-token生成を実行。
両モデルとも全256 forwardのlogitsが有限、prefix hit0、要求長を生成して正常終了。
これは有限性・長文動作の検証であり、1GPUとのtop1/KLD品質比較は別途必要。
検証中は毎stepの有限性確認が同期を起こすため、その時間を速度として採用しない。

## 未完了の検証

- 1GPUとの固定1024-position top1比較（必要時はKLD）。
- 同一入力streamによるprefill/decode、30連続job以上、各GPUのwarm後VRAM推移。
- 転送回数・bytes・host staging・同期・各GPU計算時間の確認と、必要な調整。
- 初期配置と代替配置の性能差、推奨配置、後続大型モデル向けのVRAM予算。

## Artifacts

`/work/runs/two-gpu/`:

- `preflight.json`, `preflight.log`: 実deviceと行列演算。
- `topology.json`: sysfs上のPCIe linkとroot経路。
- `peer-copy.json`, `peer-copy.log`: 全転送sampleと数値比較。
- `{d,m}-load-probe.json`: module/cache実配置、初期load、短文生成。
- `{d,m}-long-initial.json`, `long-process-status.json`: 長文有限性と実process終了code。

検証用スクリプト `pair_preflight.py`, `pair_copy_probe.py`, `pair_load_probe.py`,
`validate_pair_long.py`, `run_pair_long.py` は `/work` 直下に保存。
