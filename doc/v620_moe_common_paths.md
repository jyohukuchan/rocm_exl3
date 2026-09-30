# MoE改善を共通経路へ適用

2026-09-30。前段の [batch1実験](v620_tp_decode_opt2.md) を共通dispatchへ広げた。ビルド、関連CPUテスト、代表的な実モデル推論で確認した。

## 変更内容

- native gfx10/gfx11ではmulti-token MoEを標準で有効化。`EXL3_ROCM_MOE_MULTI_TOKEN=0` で無効化できる。
- MoEの標準行数上限を24へ変更。2–24tokenを、nativeの128 expert slot上限以内のtoken単位のchunkへ分ける。top-k=10の20token検証なら12+8tokenとなり、gate/up/down呼び出しは60回から6回になる。
- TP、layer split、単一GPU、MTP検証、通常のbatch decodeが使う同じMoE経路に適用。gateのないMoEと、連続配置でない入力も扱う。
- 単一tokenと既存fallbackでも、活性化処理をバッファ全容量からtop-k分だけへ縮小。gate/up/activation/downのview容量を揃えてnativeのscratch契約を守る。
- nativeのweighted reductionを修正。範囲外expertのslotにも必要な到着通知を送り、tokenごとの固定slot幅を保ったまま無効slotを集計から除く。全expertが範囲外の場合もゼロを出す。
- 新nativeではTPでも出力回転・集約をdot kernelへ融合できるため、`EXL3_GEMV_FUSE_OUT` の一時切り替えと別kernelを省く。旧nativeは互換fallbackを使う。

大きなprefillは従来のbulk経路を使う。共有MLP等のnative行数上限はこの変更では広げない。RDNA4のMoE fallbackも従来どおりで、gfx12にこのPython dispatchを強制しない。

## 代表的な実モデルの結果

V620×2、TP2、Qwen3.8-Flash-Next EXL3 3.05bpw、元のpacked MTP3、K5/V4、Engram単一RAM table+mlockを使用。code-only 8192入力+256生成、dynamic draft max4/confidence0.6、GPU予算28/28GiB、chunk2048。batch4はwarm4job+timed4job、cache34816。batch1はcache8704。batch1のprefillはauto、draft/verify/decodeはprofile_peak、batch4推論はprofile_peak、終了時auto。

| batch4のaggregate decode | 改善前 | 共通経路適用後 | 差 |
|---|---:|---:|---:|
| 全decode区間 tok/s | 55.826 | 73.091 | +30.9% |
| 全jobが動く共通区間 tok/s | 57.359 | 76.295 | +33.0% |
| draft採用率 | 76.80% | 75.30% | -1.50ポイント |

batch1もwarm1+timed2を完了し、decode中央値60.156 tok/s、engine62.167 tok/s。前段の限定版58.777/61.851 tok/sと比べ、速度を維持して適用範囲を広げた結果と扱う。今回の小幅な差は前段の測定変動範囲と重なるため、batch1の追加改善率として確定しない。

全8jobが256tokenを生成し、読み取ったコード生成出力に明らかな異常は見られなかった。TP/RAM/cache監査、正常終了、電力設定の復帰を確認。生成文やdraft統計は丸め差で変わり得るので、速度差を単独kernelの改善率とは扱わない。この代表結果から、全modelや全経路に同じ改善率が出るとは主張しない。

CPU829 tests+148 subtestsを通過。その後に代表batch4で見つかったR1 down scratchの容量不整合は修正し、関連25 tests+15 subtestsと代表runの再実行で確認した。経路ごとの数値比較や前段の96ケース比較は追加実行していない。

nativeはgfx1030、118sources、ROCm SDK7.14でビルド成功。新native SHA256は `25b536e7486ca2ac9a00e274f4efdbbc7e7435d64e96a20c604a44b5533ca57f`。`EXL3_MGEMV_MASKED_REDUCE_SUPPORTED` がtrueであることを確認した。

集計・条件・source/native識別・正常終了の記録は [日付付きartifact](../benchmarks/2026-09-30/moe-common-paths.json) に保存した。

## 再現

[通常の環境・process limit・power helper手順](reproduce_v620.md) に従い、この変更を含むnativeを再ビルドする。公開bundleの `prompts-8192-b4-r3.json.gz` を展開し、`language == "code"` かつ `repeat <= 1` のpromptだけを使う。

`--batch-size 4 --cache-tokens 34816 --new-tokens 256 --draft-tokens 4 --dynamic-draft --draft-confidence 0.6 --max-chunk-size 2048 --use-per-device 28 28` とし、router複製・batched pruning・Engram mlockを維持する。multi-token/row capのoverrideを指定しなければ新しい標準設定になる。今回のロードには `EXL3_HOST_MEM_RESERVE_MB=0` を使い、実常駐監査を維持した。

改善前はsource `e04cd643232ab1643b85e733243fa051a979d9e6`、旧native SHA `57afa48c9a61b9e7ba2721917bf011096d7cc947e43b2a8444ee8350f0820718`、multi-token OFF、MoE row cap20。改善後は `9bc8251` の共通経路と新native、標準のmulti-token ON/row cap24。単に新nativeでflagをOFFにしても、R1の不要計算削減やnative修正は元に戻らない。
