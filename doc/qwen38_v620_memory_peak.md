# V620×2: VRAM最高段・コア可変の検証

2026-09-29。ユーザー依頼: VRAM・PCIeだけ最高クロックにし、コアはauto相当の可変制御を残せるかを調査し、速度と消費電力を比較する。

## 設定の可否

- VRAM: `auto`から`manual`へ移り、`pp_dpm_mclk`へ最高段index `3`を指定できた。両V620で1000MHzの段が選択された。
- コア: `pp_dpm_sclk`やOverDriveを変更していない。インストール済みドライバではmanualへの遷移はGFX/SOC周波数範囲を設定し直さないため、直前のautoの全周波数範囲を維持する。global modeの表示はmanualだが、コアの最低・最高を同値にする固定ではない。
- PCIe: このドライバのSienna Cichlid用`force_clk_levels`には`SMU_PCIE`処理がなく、未処理の種類も成功を返す。`pp_dpm_pcie`への書込み成功を「固定できた」証拠として扱えない。強制固定は今回確認できていない。
- PCIeの実リンクは設定前から両GPUで最高の16.0GT/s x16（Gen4 x16）。測定中も`current_link_speed`/`current_link_width`を約0.5秒間隔で保存し、最高状態を維持するかを確認する。`pp_dpm_pcie`の619MHz表示はテーブル値であり、内部LCLKを独立測定・固定した証拠ではない。

従って今回の条件名は「VRAM最高段＋コア可変」。PCIe最高固定が保証された設定とは表現しない。

## 実際の指定方法

既存設定を保存したうえで、今回のV620（PCI 43:00.0、03:00.0）に対して順次実施した操作:

```text
power_dpm_force_performance_level = auto（開始状態を確認）
power_dpm_force_performance_level = manual
pp_dpm_mclk = 3
```

profile_peakから直接manualへ移ると、以前のコア制限を引き継ぐ可能性がある。まずautoの範囲に戻してからmanualへ移る。PCIeやコアの段には書き込まない。終了時は両GPUを元のautoへ戻すfinally処理をrunnerに実装。

## 測定条件

`formal-mempeak-36`: Qwen3.8 Flash Next EXL3 3.05bpw、head/ngram5bpw、Engram RAM、V620×2 layer split、batch1、投機生成なし。入力512/2048/8192、生成256、warmup1+測定5、prefill/decode計36job。予算[28,28]GiB、chunk2048、cache8704、nofile65536、修正native SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`。

保存済みauto/profile_peakとの同一入力比較。電力はGPU報告値の時間積分から算出し、CPU/RAM/電源損失を含まない。decode電力は最初のtoken受取〜最後から2番目の受取（終了処理除外）。結果は以下に記載。

artifact root: `/path/to/rocm-exl3-data/runs/qwen38`。runner `run_memory_peak_benchmark.py`、状態/復元記録 `memory-peak-process.json`、driver source fingerprint `memory-peak-driver-evidence.json`、集計 `analyze_memory_peak.py`。

[Linux公式のクロック段指定仕様](https://docs.kernel.org/gpu/amdgpu/thermal.html#pp-dpm)と、インストール済み `/usr/src/amdgpu-6.16.13-2303411.24.04/amd/pm/swsmu/smu11/sienna_cichlid_ppt.c` のforce_clk_levels、`smu_v11_0.c` のset_performance_levelを照合した。

## 測定結果（完了）

36job・失敗0・全prefix cache miss・投機生成なし・通常exit0。auto/peak/mempeakの全36入力hashは一致。auto/peakは先の保存済みrunとの比較で、交互実行による比較ではない。設定後の全1172サンプルで両GPUのメモリ報告値は1000MHz、実PCIeリンクは16.0GT/s x16だった。

| 入力token | 設定 | prefill tok/s | prefill平均W | decode tok/s | decode平均W |
|---:|---|---:|---:|---:|---:|
| 512 | auto | 259.3 | 242.9 | 24.88 | 163.2 |
| 512 | VRAM最高段＋コア可変 | 263.2 | 254.3 | 26.57 | 171.3 |
| 512 | profile_peak | 261.2 | 407.0 | 36.41 | 378.3 |
| 2048 | auto | 413.8 | 259.2 | 24.80 | 162.4 |
| 2048 | VRAM最高段＋コア可変 | 414.3 | 273.3 | 26.17 | 170.2 |
| 2048 | profile_peak | 398.2 | 395.8 | 36.10 | 376.1 |
| 8192 | auto | 388.7 | 259.0 | 24.59 | 164.2 |
| 8192 | VRAM最高段＋コア可変 | 388.0 | 271.9 | 26.30 | 173.1 |
| 8192 | profile_peak | 374.9 | 396.2 | 36.14 | 379.2 |

速度は5回中央値、電力はGPU2枚合計の時間加重平均。decode速度は従来のtime_generate基準で、終了処理を含む実受取速度は8Kでauto23.57 / VRAM最高段25.24 / peak34.32tok/s。

8Kではautoに対してdecode **+6.9%**、decode電力 **+5.45%（+8.95W）**。prefill速度はほぼ同じで、prefill電力は約5%増。新設定のspreadは8K prefill 0.62%、decode 0.98%。

decodeのGPU消費エネルギーはauto6.628 / VRAM最高段6.550 / peak10.456J/token。autoとの差約1.2%を、測定ばらつきを超える効率改善とは断定しない。**decodeの電力効率をほぼ保ちつつ、少し速くする設定**という評価。8K入力＋256生成＋終了処理の全体では2.003→2.095Wh（+4.56%）で、prefillを含む全体の省エネルギー化にはなっていない。

profile_peakとの比較ではdecode平均電力を約54%減らせるが、速度は36.14→26.30tok/sとなる。VRAM最高段だけではprofile_peakの大きな速度向上を再現しない。残る差のコア周波数・省電力状態・SOC等への寄与分解は今回行っていない。

## クロックと復元の確認

- コアはGPU0の8K decodeで報告値500〜2345MHz、平均GFX報告値約900MHzと可変。GPU1の同区間は2425MHz報告で、auto時にも高いコア報告値が継続していた。こちらもSCLKの固定指定は行っていない。瞬時/平均クロックは稼働割合や省電力停止の影響を受けるため、異なる種類のセンサー値を混同しない。
- 全サンプルのPCIe Gen4 x16維持を確認したが、強制固定APIや内部LCLK固定が使えたとは主張しない。
- finallyで両GPUをautoへ復元済み。pp_power_profile_modeも開始時のBOOTUP_DEFAULTのまま。電圧・電力上限・BIOS・OS設定は変更していない。

結果は`memory-peak-power-comparison.json`。平均電力の算出法は[前の電力比較](qwen38_v620_power.md)と共通。補助のPCIe binary-decode欄2個はoffset誤りがあったため無効として正規化時に除去した（rawを別保存）。PCIe判定は独立したsysfsのspeed/width値のみで行い、時間・電力・メモリ/コアclockの値は変更していない。詳細は`memory-peak-telemetry-normalization.json`。
