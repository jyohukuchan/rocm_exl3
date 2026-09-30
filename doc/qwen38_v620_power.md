# Qwen3.8 Flash Next / V620×2: autoとprofile_peakの消費電力

2026-09-29。正式ベンチマークで保存した約0.5秒間隔の電力ログを再集計。新しいGPU負荷実行や設定変更は行っていない。対象はEXL3 3.05bpw、Engram RAM、layer split、batch1、投機生成なし。各入力長・各処理・各policyでwarmupを除いた5回を比較し、全入力token hashの一致を確認。

**8K入力時の2枚合計平均は、prefill 259→396W（+53%）、decode 164→379W（+131%）、8K入力＋256生成のリクエスト全体では221→389W（+76%）。** GPU報告値を集計しており、CPU・RAM・電源変換損失などのシステム全体電力は含まない。

## 全入力長の結果

| 入力token | 区間 | auto平均W | profile_peak平均W | 増加W | 増加率 |
|---:|---|---:|---:|---:|---:|
| 512 | prefill | 242.9 | 407.0 | 164.1 | 67.6% |
| 512 | decode（終了処理除外） | 163.2 | 378.3 | 215.1 | 131.8% |
| 512 | 入力処理＋256生成＋終了処理 | 169.1 | 375.5 | 206.4 | 122.1% |
| 2048 | prefill | 259.2 | 395.8 | 136.6 | 52.7% |
| 2048 | decode（終了処理除外） | 162.4 | 376.1 | 213.7 | 131.6% |
| 2048 | 入力処理＋256生成＋終了処理 | 184.4 | 376.1 | 191.7 | 103.9% |
| 8192 | prefill | 259.0 | 396.2 | 137.2 | 53.0% |
| 8192 | decode（終了処理除外） | 164.2 | 379.2 | 215.0 | 131.0% |
| 8192 | 入力処理＋256生成＋終了処理 | 220.8 | 388.9 | 168.1 | 76.1% |

## 8K時のGPU別内訳

GPU0はPCI 43:00.0、GPU1は03:00.0。平均は同じ時間区間で積算した電力量÷秒数。

| 区間 | auto GPU0 / GPU1 W | peak GPU0 / GPU1 W |
|---|---:|---:|
| prefill | 135.8 / 123.1 | 201.8 / 194.4 |
| decode（終了処理除外） | 62.6 / 101.6 | 179.6 / 199.6 |
| 入力処理＋256生成＋終了処理 | 108.1 / 112.7 | 193.9 / 194.9 |

## 速度と消費エネルギー

既存の8K速度測定ではdecode 24.59→36.14tok/s（約47%向上）。今回切り出した定常decode区間のGPU消費エネルギーは6.628→10.456J/token（約58%増加）だった。速度向上だけでは平均電力の増加を相殺できていない。

8K入力＋256token生成＋終了処理の1リクエストあたりGPU消費エネルギーは、5回平均で2.003→3.253Wh（約62%増加）。prefillは388.66→374.87tok/sで約3.5%低下していたため、この条件ではprofile_peakのprefill側に速度上の利点は見られない。

## 集計方法と精度

- 元データ: artifact root `/path/to/rocm-exl3-data/runs/qwen38` の `formal-auto-36.json`, `formal-peak-36.json` と各 `*-telemetry.json`。実測間隔中央値はauto0.501265秒、peak0.501016秒、最大0.538070/0.504920秒。ロードとwarmupを除外。
- 電力源: `gpu_metrics` v1.3の`average_socket_power`（offset22のuint16、W）。ローカルのamdgpuドライバ `sienna_cichlid_ppt.c` におけるSMU `AverageSocketPower`からの転記、`kgd_pp_interface.h`の配置を確認。GPU activity値は区間選択に使っていない（GPU1のactivity99%固定という既知の観測上の制約を回避）。
- prefill: 専用prefill jobのwall開始から報告されたtime_prefillまで。内部のprefill開始はwall開始と完全同時ではないため境界に微小なずれがあり、1 token目のforwardと終了処理は対象外となる近似区間。
- decode: 最初のtoken受取から最後から2番目のtoken受取まで。256生成の各runから254個のITLを用い、最初のtokenと最終iterate全体を除外。最終iterateはdecodeと終了処理が混ざるため、両者を不正確に分割して含めない。
- リクエスト全体: decode jobのwall開始〜終了。prefill、全256生成、終了処理を含める。
- 各sample間を線形補間し、台形積分でJを算出。5回の合計J÷合計秒を平均Wとする。異なる処理区間を単純平均した値ではない。
- 8Kの区間ごとのsample数: autoはprefill210/decode102/request325、peakは218/70/300。512token prefillは約2秒と短く、各mode合計18〜20sampleのため8Kほど精密ではない。
- 8K各run平均Wの範囲: prefill auto257.4〜261.5/peak395.3〜397.0、decode auto163.4〜164.9/peak377.8〜380.7。境界を各0.5秒削った平均も、decode auto163.0/peak379.3Wで結論不変。
- 独立の単純sample平均との照合では、8Kの全区間で時間積分平均との差が0.83W未満。センサー自体の平均化遅延・校正誤差は評価していない。ワットメーターによるコンセント側測定ではない。

再集計スクリプトは同artifact rootの`analyze_power.py`、入力SHA256と全runの境界・GPU別J/Wを含む機械可読結果は`power-comparison.json`。

[LinuxのGPU metrics構造体定義](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h)（実際のlayoutはインストール済みドライバソースでも照合）。
