# V620×2 層分割: Phase 3/4（検証中）

2026-09-29。対象は単一GPUで検証したQwen3-8B EXL3 4bpw（D）と
Qwen3-30B-A3B EXL3 3bpw（M）。TPおよびQwen3.8大型モデルは後続段階。
この文書は途中経過であり、正式な品質・速度・連続実行の合格宣言ではない。

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
