# 2026-10-02 V620 TP2 expert配置の実験

Qwen3.8 Flash Next EXL3 3.05bpwをV620×2で実行し、日本語coding/chatのdecode用にrouted expertの担当GPUを変更する。全expert・量子化済み重み・router列は保持し、pruningや再量子化は行わない。

## 現在の2GPU構成

- GDNはhead単位のTP2。QSAは層全体を片方のGPUで実行し、GPU0に7層、GPU1に5層。
- routed MoEは各expert全体を片方のGPUへ置くEP2。shared expertとLM headはチャネルTP。
- 元の3bit MTPはGPU1だけで実行。Engramは1つのmlock済みRAM owner。
- context 786,432、target GPU budget 28.5/27.5 GiB、K5/V4、batch 1、prefill chunk 2,048。
- thinking有効・xhigh・temperature 0、dynamic MTP最大4・confidence 0.6、Vision有効。
- batch 1の電力設定はprefill/idleがauto、draft/verify/decodeがprofile_peak。

## データと候補

公開データ: [tcclaviger/Qwen3.8-Flash-Next-expert-activation-map](https://huggingface.co/datasets/tcclaviger/Qwen3.8-Flash-Next-expert-activation-map)。
revision `54779caae07e215083a185f8ed41a1f230419c38`、`expert_map_sequences.jsonl.gz`のSHA256は`f562de5df73a0f22733c12256c28e9fa6bde22a014f0512c2b94c546de2347f6`。
2,751入力、28 domain、48 MoE層×512 experts。BF16のprefill+generationを集計したもので、今回の量子化済みdecodeの分布とは異なる。
route回数である`hits`を使用し、出力振幅を含むREAP `importance`を計算量の代用にはしない。
`code_gen/code_proj/code_files/code_doc/cot/dialogue/instruct/tools`の8 domainの層別hit shareを等重みで混合した。
元データはQwen Community License 1.0。データと由来する配置JSONはローカルの実験ディレクトリに保存し、MITのコードとは別に扱う。

[独自prompt](../benchmarks/2026-10-02/expert-placement-prompts.json)は、日本語coding 8件・chat 8件の配置作成用と、別のcoding 6件・chat 6件の速度評価用に分けた。
配置作成では7,806生成tokensを得て、両rankで同じglobal expert IDsを記録した。prefill末尾を除き、target decode/verifyの計算を含む各層3,015イベントを利用した。
reasoningのアルファベット等の文字のうち、Latin文字は84.94%（4,942/5,818）。tokenベースの言語判定ではない。英語reasoningと棄却されるMTP検証分も配置の計算量へ含めた。

| 候補 | 決め方 | 配置作成データの平均不均衡 |
|---|---|---:|
| 基準 | 連続expert IDの分配 | 0.08878 |
| public-frequency | 公開hit頻度をgreedyで均等化 | 0.08769 |
| own-frequency | 入力・カテゴリを等重みにした独自hit share 90% + 公開10%を均等化 | 0.08047 |
| own-cooccurrence | 同じforwardで一緒に使うexpertを考慮してswap | 0.05209 |

不均衡は各forwardのrank0 route回数と全体の半分との差の絶対値を、coding/chatと各入力に等重みとなるよう正規化して集計したもの。**この値の減少率は速度向上率ではない。** 各層のrank0 expert数238〜265を維持した。

## 実装と正しさ

`--tp-expert-order FILE.json`で、層keyごとのstorage slot→元expert IDの順列を読み込む。router logitsの列、top-kの選択順、tieの扱い、per-expert scalingは変更しない。
既存routing kernelの選択ID保存時にだけ、元ID→storage slotのint32 LUTを参照する。追加kernel起動・tokenごとのCPU転送はない。LUTは48層で96 KiB/rank。
TP exportでgate/up/downのexpertリストを並べ替え、各rankが自分のstorage slot範囲をimportする。shared expert・MTPには適用しない。
非std router、MoE tensor split、TP以外、未知の層key、重複・欠落IDは拒否する。古いnative extensionでmapped routingを要求した場合は再buildを要求する。

- V620両方、rows 1/5/32/512、random/tie logitsで16 GPU testsが成功。router scores、選択順、weightsは元経路とbit一致し、IDだけがLUT通り変わる。
- 実モデルのMoE層0/23/47、rows 1/5/8/32/128で、同一GPU上の配置変更前後の出力が全てbit一致。
- TPではFP32部分和の所属GPUが変わるため、最終生成文のbit一致までは保証しない。異なる生成やMTP採用率を含む実測として評価する。
- CPU regression: 871 passed、25 skipped、148 subtests。既存の警告2件。その後のfocused tests 45件も成功し、公開domainの等重み集計・profile失敗時のunloadも確認した。

## 測定

配置作成用のGPUコピー記録を完全に外し、同じnative buildを基準・全候補で使用する。API経由でwarmup後に評価用12入力を順番に実行し、最大512生成tokens、prefix cache hit 0を確認する。
生成tokensにはreasoningを含む。`output_tokens / decode_seconds`をカテゴリ内で合計して求め、MTP採用率も併記する。
最初のscreeningの後、最速候補と基準を異なるnonceで各2巡追加し、経時変化を確認する。

MTP有効で基準・最速のown-cooccurrenceを各3巡（各36 requests）実行した集計:

| 配置 | coding decode tok/s | chat decode tok/s | 全体 decode tok/s | MTP採用率（全体） |
|---|---:|---:|---:|---:|
| 基準 | 50.11 | 52.99 | 51.40 | 68.34% |
| own-cooccurrence | 51.74 | 53.79 | 52.66 | 69.19% |

同一入力・nonceどうしのdecode速度比の幾何平均は、coding **+3.31%**、chat **+1.81%**、全体 **+2.56%**。合計token数を合計decode時間で割る表とは重み付けが異なる。
1巡目だけではcoding +7.4%だったが、3巡では差が縮小した。MTP採用率や生成内容も変わるため、この差全てをtarget計算の改善とは扱わない。

MTPなしの対照では、draftをVRAMに残して配置を同じに保ち、Generatorへのdraft接続だけを外した。別の同一nonceで各12入力・最大256生成tokensを測定し、draft_tokens=0を確認した。

| 配置 | coding AR tok/s | chat AR tok/s | 全体 AR tok/s |
|---|---:|---:|---:|
| 基準 | 34.58 | 34.76 | 34.67 |
| own-cooccurrence | 34.50 | 34.93 | 34.71 |

同一入力の幾何平均比は全体 **+0.13%**、coding **−0.19%**、chat **+0.46%**。単一tokenのAR decodeに明確な改善は出ていない。
MTP有効時は多token検証を含む別の実行経路なので、ARの結果だけで「差は全て採用率による」とは断定できない。この実験では多token検証時間の変化と採用率・生成内容の変化を完全には分離していない。

現在のMTP運用では、3巡平均で全12入力の速度比が正だったown-cooccurrenceを採用する。改善は数%の範囲で、長文脈・batch 2以上の速度向上はこの測定から主張しない。
採用配置の768Kiメモリ確認では、3,072の全pageを初期化し、末尾2,266tokensだけをprefillして固定4-token MTPで32tokensを生成した。書き込み終端786,430、実vocab内の返却logitsは全て有限値。
VRAMピークはGPU0 **31.488 GiB**、GPU1 **31.182 GiB**で、各31.984 GiB以内。768Kiの実入力によるprefillや品質評価ではなく、全cacheの確保と末尾実行のメモリ検証である。
通常サービスはこの配置・768Ki・元の3bit MTPに戻す。

CLI既定では順列を適用せず、`--tp-expert-order`を指定した場合だけ有効になる。

集計・入力別速度・MTP採用率・配置hashは[実測JSON](../benchmarks/2026-10-02/expert-placement.json)に保存した。入力case単位のbootstrap区間は探索的な目安であり、12入力を超えた母集団の保証ではない。

## 再利用

native extensionを通常のROCm手順で再buildしてから利用する。
順列ファイルは`version: 1`、`num_experts: 512`、`orders: {実際のMoE層key: [全512 IDの順列]}`のJSON。
量子化モデル自体のファイルを書き換える必要はない。

```sh
# private API keyは環境変数EXL3_API_KEYに設定
python -m rocm_tools.rdna2.expert_placement_bench \
  --cases benchmarks/2026-10-02/expert-placement-prompts.json \
  --split eval --warmup --output eval.json

# 計測専用serverにはexamples/expert_profile_bootstrap.pyを先に読み込む。
# EXL3_TP_REPLICATE_ROUTER=1、EXL3_EXPERT_PROFILE_DIR=/path/to/profilesを設定。
# 例: python -c 'import runpy; runpy.run_path("examples/expert_profile_bootstrap.py"); runpy.run_module("rocm_tools.exl3_server.server", run_name="__main__")' 通常のTP引数...
# 同じbenchmark CLIを--split trainで実行し、通常停止で最終profileをflush。

# 元配置のroute captureから、prefillを除いたeventsとhitsを抽出
python -m rocm_tools.rdna2.expert_route_profile \
  --profiles /path/to/profiles --requests train.json \
  --output-events events.json --output-hits hits.npy

# 公開mapの8 domainを同じ方法で集計し、public-frequency配置を作る
python -m rocm_tools.rdna2.plan_expert_placement \
  --plan /path/to/profiles/plan.json --public-map expert_map_sequences.jsonl.gz \
  --save-frequency public.npy --output public-order.json

# 実際の2-rank Model.planをJSONで保存して使用
python -m rocm_tools.rdna2.plan_expert_placement \
  --plan plan.json --frequency hits.npy --events events.json --output order.json

# 実測したown-frequency / own-cooccurrenceを再現する場合
python - <<'PY'
import json
import numpy as np
from rocm_tools.rdna2.plan_expert_placement import event_matrix
with open("events.json") as f:
    events = json.load(f)
public = np.load("public.npy")
mixed = []
for layer, rows in enumerate(events):
    matrix, weights = event_matrix(rows)
    frequency = (matrix * weights[:, None]).sum(0)
    mixed.append(.9 * frequency / frequency.sum() + .1 * public[layer])
np.save("mixed.npy", np.stack(mixed))
PY
python -m rocm_tools.rdna2.plan_expert_placement --plan plan.json --frequency mixed.npy --events events.json --seed-frequency public.npy --output coocc-order.json

# 通常のTP起動引数に追加
# python -m rocm_tools.exl3_server.server ... -tp --tp-expert-order order.json
```

配置作成で使ったown-frequencyは、events.jsonの各入力・カテゴリを等重みにした層別hit shareとpublic.npyを0.9/0.1で混合した。raw hits.npyの単純正規化とは異なる。own-cooccurrenceはその行列を`--frequency`へ渡し、`--seed-frequency public.npy`も指定して、基準・own-frequency・public-frequencyの3つからswapを始めた。

route captureは`expert_route_profile.install/flush`をTP workerへdispatchする専用計測。Qwen3.8の48層・512 experts・top-k 10に限定する。
`install`はロード後、最初の入力のprefill前に一度だけ実行し、全入力終了後に`flush`をdispatchして最終caseを保存する。8tokensより長い初回prefillでcaseを識別し、max_rows既定4,096を超える場合は計測上限を増やす。
並べ替え済みの配置では元IDを記録しないため、capture時は`--tp-expert-order`を外す。記録にはGPUコピーが加わるので速度比較では必ず外す。
