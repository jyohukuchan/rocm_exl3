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

保存済みauto/profile_peakとの同一入力比較。電力はGPU報告値の時間積分から算出し、CPU/RAM/電源損失を含まない。decode電力は最初のtoken受取〜最後から2番目の受取（終了処理除外）。結果は測定完了後に追記。

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38`。runner `run_memory_peak_benchmark.py`、状態/復元記録 `memory-peak-process.json`、driver source fingerprint `memory-peak-driver-evidence.json`、集計 `analyze_memory_peak.py`。

[Linux公式のクロック段指定仕様](https://docs.kernel.org/gpu/amdgpu/thermal.html#pp-dpm)と、インストール済み `/usr/src/amdgpu-6.16.13-2303411.24.04/amd/pm/swsmu/smu11/sienna_cichlid_ppt.c` のforce_clk_levels、`smu_v11_0.c` のset_performance_levelを照合した。
