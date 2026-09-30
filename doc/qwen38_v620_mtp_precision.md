# V620×2 / 公式MTPの自前3bit・5bit比較

2026-09-29。公式BF16からMTPだけを自前量子化して比較した。**今回の入力・設定では、5bit化による目に見える速度改善は確認できなかった。** 同じ手順の自前3bitに対し、日本語decode中央値は−2.1%、コードは−1.1%。終了処理込みの速度とリクエスト全体時間もほぼ同じで、追加VRAM約0.63GiBに見合う改善は見られない。既定の配布済み3bitと電力方針を維持した。

## 取得元と量子化

量子化済みの別モデルは取得していない。取得元は [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540)、revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`。28shardに分散した31個の `mtp.*` テンソルだけをHTTP Rangeで取得した。全てBF16、合計5,214,301,696bytes。共有embeddingとlm_headは既存targetから借りるため、取得・再量子化していない。

`prepare_mtp_source.py` はリビジョン、元のindex/config、各shardのヘッダー、各テンソルの位置・形状・SHA256を保存する。HTTP 206/Content-Range/長さを検査し、全shard取得へのfallbackはしない。保存した全31テンソルをrootが独立に走査し、取得時のhashと一致することを確認した。

- raw source: `models/qwen38-flash-next-mtp-bf16-official/mtp_source.safetensors`
- SHA256: `ee81e63ff6a518c2768bce9ad9e3bfe038fcb25b97efcb1338cee192d15ddce3`
- 標準の `util/convert_mtp.py` を使用。MTPは統合converterでも単体converterでも**校正データを使わない**。seed=0、mul1、out_scales=always、HQ有効。日本語校正データを追加した実験ではない。
- V620を1枚ずつ明示的に分離して3/5bitを並行変換。どちらも同じgfx1030 native、Torch、CPU threads=8。推論比較は変換完了後に行った。

既存の `mtp_bits=3` は均一3bitではなく、次のHQ構成だった。自前版にも同じHQ方針を適用した。

| 対象 | linear数 | 既存/自前3bit | 自前5bit |
|---|---:|---:|---:|
| routed experts・indexer projection | 1537 | 3 | 5 |
| MTP入力のfc_hidden/fc_embedding | 2 | 4 | 6 |
| attentionのQ/K/V/O・shared expert | 7 | 5 | 7 |

norm/router/mixer等の未量子化テンソルはそのまま。target本体3.05bpw、共有lm_head 5bit、Engram 5bitも固定した。

| 自前MTP | 変換時間 | safetensorsファイル容量 | SHA256 |
|---|---:|---:|---|
| 3bit HQ | 1117.76秒 | 1,037,933,517B | `71d2833d8e3572c0e9daf55e4088ae6cec9913a71178750219aafadf5fd2c9e8` |
| 5bit HQ | 916.76秒 | 1,684,451,364B | `1adb798cfd446c52160179ec2fd1bdcb31e2b98c465a5b353e3aa48c63a41f38` |

保存先はdata root配下の `models/qwen38-flash-next-mtp-self3-hq` と `models/qwen38-flash-next-mtp-self5-hq`。変換manifestをそれぞれ `conversion-report.json` に保存した。両変換processとも終了時VmSwap=0。

## 自前3bitを対照にした理由

配布済み3bitと、自前3bitはbit数が同じでも量子化結果が完全には一致しない。単体converterはseed=0固定、統合converterはmodule番号をseedにする。検査した符号列では、配布済みの入力projectionはseed=0、decoder blockはseed=1の期待値と一致した。自前3/5bitは両方seed=0で一致する。

この検査では `regularize()` の負のcodebook_scaleによる符号反転も考慮している。`sign-realization-check.json`、`seed-realization-audit.json`、`seed-realization-interpretation.json` に保存した。**bit数そのものの比較は自前3bit対自前5bitを主とし、配布済み3bitは現在の運用基準として併記する。**

## 検証と固定条件

- 両方6,203テンソル。全3,111個の浮動小数点テンソルがfinite。全1,546linearのbit mapを検査し、3bit版は既存と同じ、5bit版は各linearが+2bit。
- 既存パックに残る未量子化MTP重み19テンソルは、dtypeを揃えると公式重みと全一致。mixer補完ファイルの3テンソルもこの中に含む。
- 公式を標準loaderで読み込んだFP16作業重みと、nativeカーネルによる復元結果を16行列で比較。全てfiniteで、全16行列で5bit側の相対RMSEが低下。expert 9行列の中央値は自前3bit 13.192%→5bit 3.394%、fc_hiddenは19.965%→5.892%。相対RMSEは `||復元値−参照値||₂ / ||参照値||₂`。converterログの誤差ラベルとは独立に計算した。
- 実モデルsmokeは各版で日本語・コード各8K+128。自前3bitはtarget117/draft255回、自前5bitはtarget108/draft267回の検査forwardが全finite。自然な日本語説明とPythonコードを生成して通常終了。
- 正式測定はV620×2、layer split、Engram CPU RAM。batch=1、greedy、8,192入力+256生成。日本語説明/Python LRUCacheの2課題、各warmup1+timed5。以前の `formal8k-prompts.json` と同じ入力ID。
- MTPは上限4、dynamic、confidence=0.6。target cache slot1/history4、capacity8704、chunk2048、ロード予算[28,28]GiB。MTPはcuda:1、target embedding/headを共有。
- batch1はprefill auto、draft/verify/decode profile_peak、終了auto。全supervisorでhelper正常終了・auto復元を確認。RAM guardのoverrideは今回使用していない。
- nativeは `lib-qwen38-reduction`、SHA256 `12859e31a1bd03b61ef5a1ba6725d020dca3557e3c206dcd10f791662ae4767a`。nofile65536、HSA_ENABLE_SDMA=0。
- 3群でtarget fingerprint、native、全入力hash、cache容量、target全配置が一致。全cache miss、各256token。全jobで `rounds + accepted = 256`、`proposed = accepted + rejected` を独立再集計して一致。

正式測定の順序は配布済み3bit→自前5bit→自前3bit。新規2版は正式測定とは別processでfinite smokeを行い、同期を追加する検査を速度値に混ぜていない。配布済み3bitの再測定は前回と採用率が一致し、decode中央値も近い値だった。

## 結果

decodeは従来と同じengine時間基準の各5回中央値。採用率はtimed分のaccepted/proposedの合算。

| MTP | 日本語採用率 | 日本語decode tok/s | コード採用率 | コードdecode tok/s |
|---|---:|---:|---:|---:|
| 配布済み3bit HQ | 54.19% | 40.13 | 79.79% | 55.15 |
| 自前3bit HQ | 53.05% | 39.52 | 83.81% | 58.67 |
| 自前5bit HQ | 52.34% | 38.68 | 84.14% | 58.02 |

配布済み3bitに対するコード+5.2%だけを5bit化の効果と扱うことはできない。**自前3bitでもコード58.67 tok/sとなり、5bitをわずかに上回る。** 自前同士では日本語採用率−0.71ポイント、コード+0.32ポイント。小差であり、今回の条件で5bitが常に遅いという一般化もしない。

| 自前版・課題 | 最初の候補の採用率 | 平均候補長 | prefill秒 | 終了処理込みdecode tok/s | 全体秒 |
|---|---:|---:|---:|---:|---:|
| 3bit 日本語 | 59.83% | 1.577 | 26.763 | 36.82 | 33.741 |
| 5bit 日本語 | 58.67% | 1.713 | 26.795 | 36.56 | 33.587 |
| 3bit コード | 92.53% | 3.195 | 27.499 | 41.48 | 33.988 |
| 5bit コード | 94.69% | 3.566 | 27.618 | 41.60 | 33.799 |

prefill差は日本語+0.12%、コード+0.43%。終了処理込みdecodeは−0.69%/+0.30%、全体時間は−0.46%/−0.56%で、実用上ほぼ同じ。全体時間はprefillと終了時のRAM返却等を含む。各工程の中央値は同じ試行を指すとは限らないため、中央値同士を足して全体時間を復元しない。

生成時間を合算した速度 `sum(new_tokens−1)/sum(time_generate)` では日本語39.48→38.89 tok/s、コード57.33→58.55 tok/s。コードはこの集計では+2.1%、中央値では−1.1%と小差の向きも変わる。自前3bitのコード各値は50.33–61.42、5bitは57.22–60.30 tok/sだった。速度評価を有利な集計だけに切り替えない。

コードでは最初の候補の採用率は増えたが平均候補長も伸び、全体の採用率・速度はほぼ同等。draft/verify別の時間をこの実験では分離していないので、追加コストの内訳までは断定しない。日本語の低採用率は今回の5bit化では解消しなかった。BF16 MTPとの比較や日本語向け追加学習の効果は未検証。

この結果は同じ2課題の比較であり、日本語一般やすべてのコード課題に普遍化しない。また、日本語説明とPythonコードは言語だけでなく課題も異なる。

### メモリ

MTPロード後のtorch allocatedは3bitで1,179,183,104B、5bitで1,858,136,064B。**追加0.6323GiB**。重みpayload自体の増加は0.6021GiBで、割当単位等を含む実VRAMとは区別する。

正式測定のtorch peakは配布済み3bitがcuda:0/1=28.044/23.157GiB、5bitが28.044/23.789GiB。本体配置は変わっていない。推論processの観測時VmSwapも0だった。

## 保存先と再現

data root: `/path/to/rocm-exl3-data`

artifact root: `runs/qwen38-mtp5`。主な証拠は `comparison-summary.json`、3群の正式JSON、`self{3,5}-smoke8k.json`、各 `*-process.json` / `*-power.json`、source/output/weight reconstructionの各audit JSON。`summarize.py` が比較条件と候補数の帳尻を検査して再集計する。

正式5bit比較を別tagで再実行する例（ホストから）:

```bash
python3 /path/to/rocm-exl3-data/runs/qwen38-mtp5/run_cli.py \
  --tag my-self5-comparison --mode mtp \
  --mtp-model /work/models/qwen38-flash-next-mtp-self5-hq \
  --prompts formal8k-prompts.json --batch 1 --draft 4 --dynamic --confidence 0.6
```

`--mtp-model` を省くと従来の配布済み3bit。自前3bitは末尾を `mtp-self3-hq` にする。既定のtargetや運用設定は変更していない。

公式source取得は `rocm_tools/rdna2/prepare_mtp_source.py --help` を参照。実変換の呼出・設定・file hashはartifactの `quantize_one.py` と各モデルの `conversion-report.json` に保存した。同wrapperは既存の出力重みを上書きしないので、再変換には新しい出力ディレクトリを使う。

OpenCode Go `qwen3.8-flash` に取得用CLIと分離読み込みを実装委任し、rootが差分・CPUテスト・実取得・GPU検証を確認した。commits: `0a8c786`（MTP別directory）、`5470d29`（公式source抽出）。GPU kernelとMTPの数式はこの比較では変更していない。
