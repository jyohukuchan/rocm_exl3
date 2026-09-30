# V620 TP2: コード生成を基準にしたdecode改善

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

## 次の実験

MoEの小行数経路は現在、各候補tokenについてgate/up/down・activation・copyを繰り返す。検証3–5行では複数候補をまとめ、position-preservingなローカルexpert indexをGPUで作ることにより、TP range filtering時のcooperative fallbackを避けつつnative mgemvのmulti-token経路を試す。

候補はdefault-offの実験として準備中。単体の実重み・TP半分ずつのexpert範囲・全候補が範囲外の行・連続scratch再利用で数値確認を行い、その後同一code-only入力の実モデルA/Bで採用判断する。数値条件や本体速度を満たさなければ採用しない。MTP重みの学習・交換は今回行わない。

GatedResidualは既にfused実装なので、単純なfusion提案ではなく実kernel geometryから判断する。HIP graph再有効化は現行の主要MoE proxyを通らず、過去のflat測定もあるため最初の候補には選ばない。

## ローカル測定記録

`runs/decode-opt2` にsource manifest、code-current-decomposition.json、code-gpu-task-summary.json、source review、候補試験を保存。実model報告は `runs/context-batch/decode-opt2-code-{clean,phase,events}*`。これらは公開用の日付付きbenchmark一式とは別の調査記録。
