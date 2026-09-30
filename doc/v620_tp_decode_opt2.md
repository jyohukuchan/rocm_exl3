# V620 TP2: コード生成を基準にしたdecode改善

この文書は最初の限定版の測定記録。現在の実装は [共通MoE経路の改善](v620_moe_common_paths.md) へ拡張され、gfx10/gfx11で標準有効、行数上限24となっている。

2026-09-30。MTPの将来の学習で採用率が上がることを想定し、現時点で採用率の高いコード生成を参照して本体側を改善する。元の3.05bpw target・配布済みMTP3・K5/V4・単一RAM Engram・電力方針を使用。

## 最新実装の基準

engine source `dff0b5b8d0965aecb7b3a4bbc27e3dec530b276c` を固定。native SHA256 `57afa48c9a61b9e7ba2721917bf011096d7cc947e43b2a8444ee8350f0820718`。code-only、batch1、8192入力+256生成、warm1+timed2、dynamic max4/confidence0.6。router複製・MoE row20・batched pruning・Engram mlock有効。ロード時のみ既存のhost reserve guardを0にする診断用overrideを記録し、実際のRAM常駐監査を維持。

終了処理込みdecode中央値53.014 tok/s（50.854–55.174）、採用率74.25%、conservative prefill474.372 tok/s。公開済み50.52 tok/sは日本語groupの後にcodeを測った記録であり、calibration履歴が異なる。今回の基準値との差をコード変更の改善とは扱わない。

## Draft / 本体検証の実時間

最初のtimed jobの出力67→134、67token/15iterateで計測。既存の区間境界同期を使うnone-controlへCPU wall時計のみ追加し、per-token GPU同期やGPU eventは追加しない。

| 区間 | ms/出力token | 2区間合計に対する比率 |
|---|---:|---:|
| MTP draft | 1.887 | 12.61% |
| 本体検証 | 13.078 | 87.39% |

1検証あたり平均4.47出力token。この短い区間の値はリクエスト全体平均や終了処理込み速度とは区別する。warm/timedとも通常baselineとの生成token列は同一、全監査/正常終了/電力復帰を確認。

## 本体GPU区間の内訳

同じ最初のtimed jobの出力67→102、35token/8target forwardをHIP eventsで捕捉。旧日本語profileの代わりに、最新コードでコード生成を測定。以下は排他的なGPU stream区間で、launch間の空き・collective依存待ち・計測負荷を含む。純kernel時間や通常wall時間の割合ではなく、GPU0とGPU1は重なるので合算しない。

| 本体処理 | GPU0 ms/出力token | GPU1 ms/出力token |
|---|---:|---:|
| MoE（共有expert含む） | 5.946 | 5.913 |
| Gated Residual | 2.121 | 2.084 |
| GDN | 1.610 | 1.565 |
| QSA projection/indexer/attention | 1.396 | 0.711 |
| 通信・同期 | 2.081 | 2.813 |
| その他（head/embedding/PLE等） | 0.672 | 0.756 |

98collective/target forwardを両rankで実測。Engram CPU stageは0.01285ms/出力、内包されるRAM gatherは0.00525ms/出力。量子化Linearは復号と行列積が融合しているので、MoE列を「dequantだけ」とは呼ばない。

## MoE multi-token実験

MoEの小行数経路は現在、各候補tokenについてgate/up/down・activation・copyを繰り返す。検証3–5行では複数候補をまとめ、position-preservingなローカルexpert indexをGPUで作ることにより、TP range filtering時のcooperative fallbackを避けつつnative mgemvのmulti-token経路を試す。

`EXL3_ROCM_MOE_MULTI_TOKEN=1` を明示して使うdefault-offの実験として実装。gfx1030のgated MoE・2–5行・最大128 expert slotに限定し、それ以外は既存row loopへ戻す。slot位置を維持してTPのlocal expert IDへ変換し、入力をslotごとに配置する。native bufferは全て実際のslot数のviewを渡し、必要ならmoduleごとの上限付きscratchを一度確保する。範囲外slotはゼロへ初期化し、sharded downの1呼び出しだけfused epilogueを無効にして、masked slotによる到着counterの不成立を避ける。

実重み1層、full/TP片側ずつ、行数1–5の15条件で検証済み。候補とcooperative referenceの最大relative L2は9.644e-5、最大絶対誤差/reference最大値は1.178e-4。出力は有限、全expertが範囲外の行は厳密なゼロ、A→B→Aのscratch再利用はbit一致。native呼び出しも3回、各引数のactive slot数を確認。環境変数は呼び出し前後で復帰する。

profile_peak、CUDA events、warmup5・5sample×5repeatの単層proxy中央値。router/shared expert/通信は含まない。

| 行数 | 既存 TP片側 ms | 候補 TP片側 ms | 傾向 |
|---|---:|---:|---|
| 2 | 0.147–0.148 | 0.173–0.178 | 遅くなる |
| 3 | 0.249–0.250 | 0.238–0.240 | 小幅短縮 |
| 4 | 0.361 | 0.302–0.303 | 約16%短縮 |
| 5 | 0.395–0.396 | 0.300–0.304 | 約23–24%短縮 |

実モデルTP2でも8K入力・32生成・warm1+timed1の有限値検査を完了し、target32 forward/draft88 forward、両rankのmodule出力検査、RAM/quant/power監査・正常終了を確認。速度測定にはこの検査付きrunを使わない。同一code-only入力の通常A/Bで採用を判断する。MTP重みの学習・交換は今回行わない。

### 通常推論のA/B（2026-09-30）

同一source `e04cd643232ab1643b85e733243fa051a979d9e6` を固定し、feature OFF→ON→ON→OFFの順に4run。各runはcode-only 8K入力+256生成、warm1+timed2で、各設定のtimed4jobを集計。

| 指標 | OFF | ON | 差 |
|---|---:|---:|---:|
| 終了処理込みdecode中央値 tok/s | 54.395 | 58.777 | +8.06% |
| engine decode中央値 tok/s | 55.603 | 61.851 | +11.24% |
| 終了処理込みdecode範囲 tok/s | 51.755–55.044 | 57.194–60.876 | |
| conservative prefill中央値 tok/s | 486.739 | 484.559 | -0.45% |
| draft採用率 | 74.25% | 77.36% | +3.11ポイント |
| 本体検証回数（timed4job合計） | 280 | 286 | |

モデル・MTP・native・入力・quant cache・router複製・RAM常駐・電力方針は共通。全run正常終了、監査/電力復帰を確認。OFF出力は公開済みsource基準とwarm/timed3jobとも同一。各設定を再測定した際の出力とdraft statsも再現する。

ON/OFFではwarmの生成token列は同一だが、timed2jobは9/13token目から分岐する。単層誤差が小さくても後続argmaxやdraft confidenceが変わり得るため、8.06%を量子化kernel単独の改善率とは扱わない。通常A/Bの出力厳密一致checkerは不一致を記録したまま保持する。

### 実入力の数値比較と採用範囲

別の診断runで、実推論中の各MoEへの同一入力・同一expert選択・同一routing weightを候補とcooperative referenceへ渡す。各層でcandidate出力を退避し、基準経路との比較後に復元してからモデルを継続。8K入力+32生成、warm1+timed1、48層×2rankの96比較を完了した。全比較は5行の検証入力。

最大relative L2は3.9123e-4（約0.039%）、最大絶対誤差/reference最大値は6.6285e-4。全出力有限、両rank全48層の比較漏れなし、環境変数/出力復帰、RAM/quant/power監査・正常終了を確認。全expertが範囲外の行はこの実入力sampleには含まれず、前述の単層試験で検証した。診断runの生成token列は、追加比較を入れない通常の32token検査runとwarm/timedとも同一。診断の速度は通常benchmarkへ混ぜない。

CPU828 tests+148 subtests、単層数値/shape/再利用試験、実モデル有限値、実入力全層比較、通常推論ABBAを根拠に、測定したbatch1 V620×2 TP2/MTP構成で使えるopt-in経路として採用する。`EXL3_ROCM_MOE_MULTI_TOKEN=1` で有効化。一般設定はdefault-offのままで、batch>1の速度や長文上限は今回追加検証していない。MTP重みは元の3bitのまま。

### 再現

環境・process limit・power helperは [既存の再現手順](reproduce_v620.md) と同じ。公開bundleのbatch1 promptsからcode groupだけを抽出すれば今回の入力を再現できる（日本語groupを先に実行しない）。

```python
import gzip, json
from pathlib import Path
with gzip.open("benchmarks/2026-09-30/prompts-8192-b1-r2.json.gz", "rt") as f:
    prompts = json.load(f)
prompts["prompts"] = [p for p in prompts["prompts"] if p["language"] == "code"]
Path("code-only-8k.json").write_text(json.dumps(prompts))
```

既存TP2 commandのpromptsをこのfile、`--batch-size 1 --draft-tokens 4 --dynamic-draft --draft-confidence 0.6 --cache-tokens 8704 --new-tokens 256 --max-chunk-size 2048 --use-per-device 28 28` に合わせる。各processを新しく開始し、flag 0→1→1→0でwarm1+timed2を比較する。全runで `EXL3_TP_REPLICATE_ROUTER=1`、`EXL3_ROCM_MOE_MGEMM_MAX_ROWS=20`、`EXL3_BATCH_RECURRENT_PRUNE=1`、`EXL3_NGRAM_MLOCK=1` は共通。今回のロード診断ではhost reserve guard override `EXL3_HOST_MEM_RESERVE_MB=0` を使い、実常駐監査を維持した。

設定の一時変更はprocess全体に及ぶ。今回のTP2はrankごとに別process、MTP draft/verifyも直列なのでrank間の環境変数競合はない。任意の別threadからのnative呼び出しとの並行利用は未対応。batch>1や長文の一般的な速度改善も、この単層proxyからは主張しない。

GatedResidualは既にfused実装なので、単純なfusion提案ではなく実kernel geometryから判断する。HIP graph再有効化は現行の主要MoE proxyを通らず、過去のflat測定もあるため最初の候補には選ばない。

## ローカル測定記録

集計・入力条件・native/model識別・単層試験・実入力96ケースの数値記録は [日付付きartifact](../benchmarks/2026-09-30/moe-multitoken.json) に保存。

ローカルの `runs/decode-opt2` にsource manifest、code-current-decomposition.json、code-gpu-task-summary.json、source review、候補試験を保存。実model報告は `runs/context-batch/decode-opt2-code-{clean,phase,events}*`、`decode-opt2-multi-{off,on}[-r2]*`、`decode-opt2-actual-numeric*`。これらの詳細診断ログは日付付き集計artifactとは別の調査記録。
