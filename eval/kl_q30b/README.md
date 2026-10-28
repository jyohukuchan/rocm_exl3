# KLD / prefill 評価ツール (kl-q30b)

ラウンド 3 で `/work/kl-q30b/` に作った評価スクリプトのコミット済みコピー。
コンテナ内 (`/src` = 本リポジトリ、`/work` = datapool マウント) で実行する。

## スクリプト

- `bench_prefill.py` — プリフィルスループット計測 (単一 GPU)。
  `bench_prefill.py --model DIR --out OUT.json [--lengths 512,2048,8192] [--reps 3] [--device 0] [--exl3-int8-prefill 0/1]`
- `dump_exl3_logits.py` — KLD ケースの FP32 ロジットダンプ (exllamav3)。
- `compare_kld.py` — 候補 vs llama.cpp BF16 参照の macro case-mean KLD + top1 一致率。
- `dump_bf16_noise.py` — ノイズフロア用 BF16 ソースダンプ (2-GPU split)。

## データ (コミット対象外、`/work` に残す)

- 入力トークン列・llama.cpp BF16 参照 (`ref-llama/`)・候補ダンプ (`cand-*/`)
  は計 ~50GB のためリポジトリに含めない。
- 参照: `/home/homelab1/datapool/rocm-exl3-rdna2/kl-q30b/`
  (`corpus/`、`ref-llama/`、`cand-i/`、`cand-iii/`、`bench-*.json`)。
