# OpenCode API統合の検証（2026-10-01）

Qwen3.8 Flash Next EXL3 3.05bpwを、V620×2のTP2/RCCLでOpenCodeに接続するための実装と検証。
元の3bit MTP head、K5/V4 cache、Engramの単一RAM owner＋mlockを使う。
APIのcontextは32768、1候補のoutput上限は8192。より長いcontextのエンジン検証は別のbenchmark資料を参照。

## 実装

- protocol: OpenAI形式のfunction名・ID・JSON引数、assistant/tool履歴、reasoningを保持。
- SSE: incremental argumentsとfinish_reason=tool_calls。JSON文字列の引用符・改行・backslash・XML風文字列を保持。
- schema/filter: auto/none/required/指定名、最大8 parallel calls、json_object/json_schemaの実生成制約。
  各callを別matcherで制約し、Qwenのcall間改行を受理する。無制限JSON空白を使わない。
- Qwenのtemplate例とtool履歴を、生成するJSON parameter形式に合わせてメモリ上でrenderする。
- runtime: 検証済みTP/MTPのロード順、GPU配置、quantized cache、RAM residency、power policyを監査する。
- 共通AsyncGenerator終了処理で同期queueを解放し、raw completionの複数候補usageも合計する。

## 実行証拠

| 項目 | 結果 |
|---|---|
| CPU API/protocol/schema/runtime tests | 65 passed。実Qwen tokenizer＋LLGuidance、ASGI HTTPも使用 |
| 実モデルAPI checks | 18項目成功。最終source、引用符・backslash・nullを含むSSEも成功 |
| JSON/思考/autoの制約 | 数学の通常回答やuser要求と異なるenumを強制するテストで確認 |
| OpenCode direct connection | OpenCode 2.0.12、local rocm-exl3/qwen38 provider、127.0.0.1:3953/v1 |
| 実tool往復 | read×3、write×2、shell×1。6 callのAPI引数とclient引数が完全一致 |
| コード・テスト作成 | LRUCacheとunittestをOpenCodeが作成。実装827 bytes、テスト3588 bytes |
| サンプル実行 | OpenCodeとCodexの独立実行で14 tests passed |
| dynamic batch | 4同時text生成＋4同時schema付きtool生成を2回繰り返し、16 requestすべて完了 |

[サンプルと検証JSON](../examples/opencode_lru/README.md)に、生成ファイルのSHA256、
call ID、実行したtool、テスト件数を記録した。コードの本文はOpenCodeの成果物。
SSE修正後は、引用符を含む完全なソースが実ファイルへ渡ることもcall単位で比較した。

batch 4の継続検証は、RCCLの標準timeout=180秒を使用した。
初期のtimeout=45秒を強制した検証では、次のprompt群のprefillでcollective timeoutが1回発生した。
その直接原因は未確定で、180秒の試行では45秒を超える遅延は観測していない。
180秒設定での2回連続成功を、現在のHTTP動作確認条件として記録する。

機械可読な監査結果は[検証JSON](opencode_api_verification.json)に記録した。

## 再現

[起動とOpenCode設定](../rocm_tools/exl3_server/README.md)に従い、loopback APIを起動する。
[設定例](../examples/opencode.jsonc)のAPI keyを起動時のkeyに合わせ、対象projectへコピーする。
サンプル検証時はthinking=true / reasoning_effort=low、greedy sampling、dynamic MTP最大4tokensを使用。

```bash
EXL3_API_KEY=your-local-key python rocm_tools/exl3_server/verify_api.py \
  --output api-checks.json
# -ambs 4で起動したサーバーに対して:
EXL3_API_KEY=your-local-key python rocm_tools/exl3_server/verify_batch.py \
  --batch 4 --timeout 360 --output api-batch4.json
python -m unittest discover -s examples/opencode_lru -v
```

OpenCodeで再生成する場合は、隔離した作業directoryに設定とSPEC.mdを置き、
`opencode run --standalone --auto --model rocm-exl3/qwen38`で仕様の実装とテスト実行を指示する。
元のOpenCode実行では、API転送修正後に以前のsessionを再開して成功した。

## APIの範囲

tool定義はOpenCode標準12 toolsでcompileできることを実tokenizerでも検証した。
XMLの独立parameter形式で表せないroot-level oneOf/anyOf/allOf、依存関係などは400で拒否する。
parameter内のnested schemaはLLGuidance compilerに渡す。response_formatとactive tool choiceの同時指定は400。
Responses API、Anthropic Messages、embeddingsなどの追加protocolは提供しない。
実行したサーバーでは、batch1はprefill/idle=auto、draft/verify/decode=profile_peak。
configured batch>1はprofile_peakを保持し、終了時に元の設定へ戻す。
