# V620×2: prefill auto / decode profile_peakの動的切替

2026-09-29。**現在のbatch1・逐次生成では可能。** 評価用wrapperでprefill完了後・最初のdecode前に両GPUをprofile_peakへ変更し、最後のdecode後・queue終了処理前にautoへ戻すことを実モデルで確認した。生成エンジンのファイルや既定設定は変更していない。

## 切替にかかる時間

以下はV620 **2枚合計・逐次書込み**の中央値。毎回sudoやshellを起動せず、一時的なhost側制御プロセスが固定の2GPUのsysfsだけを操作する。Unix socket通信はローカル限定、mode0600、許可コマンドはauto/profile_peakのみ。終了/切断時に元policyへ復元しsocketを削除する。

| 条件・区間 | auto→profile_peak | profile_peak→auto |
|---|---:|---:|
| アイドル時・sysfs書込みのみ（各10回） | 3.338ms | 3.260ms |
| アイドル時・通信込み（各10回） | 4.115ms | 4.031ms |
| 実推論時・sysfs書込みのみ（各5回） | 4.914ms | 3.413ms |
| 実推論時・通信込み（各5回） | **6.128ms** | **4.625ms** |
| 実推論時・GPU完了待ちも含む境界ブロック | 22.469ms | 4.641ms |

実推論の往復通信込みは中央値**10.753ms**、範囲9.857〜11.565ms。上りの通信込みは5.609〜6.670ms、下りは4.248〜4.896ms。実GPUクロックの厳密な収束時刻ではなく、書込みが返りpolicy readbackが一致し、制御応答を受け取るまでを測った値。

prefill終了時の境界ブロック22.469msのうち、GPU完了待ちが中央値16.307ms。これは未完了のprefill workを待っている時間で、通常のGeneratorにも最初のdecode前に`cuda_sync_active()`がある。従って22.5ms全体を「設定変更による追加遅延」と数えない。今回測った切替APIと通信の追加コストは上り約6ms・下り約5msで、前後のGPU待ちとは区別する。

## 実モデルの確認

Qwen3.8 Flash Next EXL3 3.05bpw、Engram RAM、V620 pair layer split、投機生成なし、batch1、予算[28,28]GiB、chunk2048、cache8704、nofile65536。native SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`。

保存済みformal-auto-36の全乱数入力列を再生成してhashを照合し、8K decode jobのwarmup1＋測定5の計6jobを選択した。全jobで8192入力・256生成・prefix cache miss・正常終了。warmupを除く中央値:

- 報告prefill時間（切替境界を含む）: **20.843秒**。
- 実TTFT: **20.873秒**。
- decode: **36.057tok/s**。既存の常時profile_peak測定36.14tok/sと近い値。
- 終了処理・autoへの復帰を含む実受取decode: **34.111tok/s**。
- リクエスト全体: **28.348秒**。
- 切替応答後から1 token目受取: **28.545ms**。追加のsleep/settling waitは挟んでいない。
- 継続decodeのITL中央値（65token目以降、最後を除く）: **27.408ms**。

2token目のITLは73.2〜78.3msで、既存の常時peak runの35.1〜41.4msより長かった。今回は別プロセスで8K jobだけを実行しており、先行job数・checkpoint/host allocator状態が一致しないため、この差を切替だけに帰属しない。総decode throughputは上記の通り維持している。rawの`prefill_excluding_switch_s`はGPU待ちを含む境界全体を引いた補助値であり、純粋なprefill演算時間/切替追加時間の結論には使用しない。

## 参考電力（完全に記録できた4job）

別の読み取り専用observerを途中から起動したため、最初の測定jobは欠測として除外。残る4jobのGPU2枚合計平均はprefill約262.7W、decode約362.9W、リクエスト全体約285.9W、1リクエスト約2.257Wh。CPU/RAM/電源損失は含まない。約0.5秒間隔の平均電力センサーを区間で積分しており、モード境界にはセンサーの平均化遅延もある参考値。主目的の切替時間・速度は全5回の結果。

## 適用範囲と復元

power_dpm_force_performance_levelはGPU全体への設定。複数requestのprefill/decodeを同じGPUで同時実行する場合、各requestが独立にpolicyを切り替える設計にはできない。現在のbatch1のようにフェーズが分離している場合は、1リクエストにつき前後2回の切替で済む。

一時helper/benchmarkは全て終了、両GPUとも元のautoへ復元済み。OS設定・sysfs権限・推論エンジン既定動作への恒久変更なし。本実験は自動切替を運用コードへ恒久導入したものではない。

artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38`。
- `power_switch_server.py`, `benchmark_phase_switch.py`, `run_phase_switch.py`: 実行用の診断スクリプト。
- `phase-switch-idle.json`, `power-switch-server.json`: 各sysfs書込み、readback、通信の生時間。
- `phase-switch-benchmark.json`, `phase-switch-summary.json`: stage別境界・速度・token interval。
- `phase-switch-process.json`: process exitと最終policy。
- `phase-switch-telemetry.json`, `phase-switch-power-summary.json`: 参考電力、欠測除外・区間定義。

[Linux公式の動的performance level仕様](https://docs.kernel.org/gpu/amdgpu/thermal.html#power-dpm-force-performance-level)。engine内のprefill判定と最初のdecode前の同期箇所もローカルsourceで照合した。
