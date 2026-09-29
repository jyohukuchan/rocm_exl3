# Qwen3.8 Flash Next / V620 pair — 検証中

2026-09-29開始。未完了。目的は投機的デコーディング無しでQwen3.8 Flash NextをEXL3約3bpw、PLE/EngramをRAMに置いてV620×2で実行し、正確性と速度を検証・改善すること。

## 固定条件

- モデル: `turboderp/Qwen3.8-Flash-Next-exl3`、branch `3.05bpw_h5_ng5`、revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`。
- HF metadataの全ファイル合計: 85,139,442,313 bytes。本体EXL3 3.05bpw、head 5bit、ngram専用量子化。PLEはVRAMへ全量配置せずRAM常駐を実測確認する。
- text-only、batch1、全routed expertはGPU常駐。draft model、MTP、n-gram draftingを使わない。
- 基準環境: `doc/v620_pair_config.json`。V620 UUID順43→03、`HSA_ENABLE_SDMA=0`。性能測定時のみ`profile_peak`、終了時に元のpolicyへ戻す。
- 初期load budget `[28,28]` GiBは未検証。scratch/cache/driver込みの実VRAMと余裕を確認して決定する。
- 48層: GDN36、QSA12。GDN key/value heads16/48、head_dim128。QSA main head_dim256、indexer head_dim128、token budget2048。PLEは第2層前。

## 検証・改善の順序

1. GDN recurrent/chunked prefill、QSA選択・attention、gated residual/norm、PLEの数学参照との比較。正確性を速度測定の前提とする。
2. モデル全ファイルの取得・サイズ/SHA256検証、RAM実配置、2GPU層配置・再帰状態・indexer cacheの配置確認。
3. 短文生成から512/2K/8K/32Kへ拡大。32K+256token、有限logits、30連続job、VRAM/RSS/page fault、EOS境界・状態reset・prefix再利用を検証。
4. 固定の日本語/コード課題とcorpusで品質を記録。bulk/chunkedの一致は内部整合性検査であり、独立参照の代わりにはしない。量子化誤差と移植誤差を分けて評価する。
5. プロファイラ無しのTTFT/decode基準値を保存し、MoE/dequant、GDN、QSA、PLE gather/copyと同期へ分解。候補修正は独立の正確性確認後、同一条件でend-to-end再測定。共通kernel変更時には既存D/Mも回帰確認。

完了条件・暫定性能目標は`doc/rdna2_port_plan.md` Phase5を維持する。未達・未測定を完了へ読み替えない。Tensor Parallelは同計画Phase6で扱う。

## 現在の証拠

- 元repo HEAD `9d32c69`、作業開始時clean。既存の2GPU D/M検証済み環境を再使用。
- 既存GDNテストの小規模参照一致2件・ビット再現性3件が実V620で成功（5 passed、19 deselected）。Qwen3.8全体の正確性は未証明。
- 重みの取得と専用検証harnessを準備中。現行top1 collectorは再帰モデルを拒否していたため、GDN状態とPLE履歴を正しく引き継ぐ検証経路が必要。
- artifact root: `/home/homelab1/datapool/rocm-exl3-rdna2/runs/qwen38`。取得metadataは`model-source.json`、GPU小規模検証は`gdn-existing.log`。

### GDN prefillのRDNA2問題と候補比較

Qwen実形状（batch1、key/value heads16/48、head_dim128）のBF16 vendored FLA prefillは、LLVMの`Cannot select: intrinsic llvm.amdgcn.fdot2.bf16.bf16`でprocess abort（exit134）。native recurrent kernelの実行後に発生し、ログは`gdn-shape-probe.log`。

入力q/k/v/betaをFP32へ昇格したFLA経路は65tokenで成功。独立の逐次PyTorch参照に対して出力最大絶対誤差5.95e-5、状態1.94e-7。初回JIT約157.6秒はsteady推論時間とは分ける。

さらにnative recurrentとFP32 chunkを65/512/2048tokenで比較し、全て出力・状態の`atol=0.003, rtol=0.02`判定に合格。2048tokenでの最大絶対差は出力1.22e-4、状態9.54e-7。両GPUのpolicyをprofile_peakへ揃えた単体予備測定（7回、allocation/conversion/hostsync込み）は以下。

| tokens | native recurrent（中央値ms） | FP32 chunk（中央値ms） |
|---|---:|---:|
| 65 | 0.556 | 2.881 |
| 512 | 4.052 | 9.219 |
| 2048 | 16.206 | 36.133 |

`gdn-backend-compare-peak.json`にraw samples、`gdn-peak-process.json`にexit0と元policy（両auto）への復元結果を保存。native経路をRDNA2 prefillにも使用する案は有望だが、これはモデル全体の速度改善・品質合格の証拠ではない。FP32互換経路と専用検証harnessの実装はOpenCodeへ委任中、diffレビューと実機再試験は未完了。
