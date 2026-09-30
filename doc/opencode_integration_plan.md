# OpenCode接続の実装・検証計画

2026-10-01。Qwen3.8-Flash-Next EXL3 3.05bpw＋既存MTP3をV620×2でHTTP提供し、OpenCodeから実際にファイル操作とコード作成を行う。

## 完了条件

- [ ] モデル固有のtool出力をOpenAI形式へ変換し、名前・ID・JSON引数を返す。
- [ ] assistant/toolの履歴、tool_call_id、名前、reasoningを保持して次の推論へ渡す。
- [ ] SSEで部分tool引数、複数call、finish_reason=tool_callsを正しく返す。
- [ ] tool_choice auto/none/required/指定名とparallel_tool_callsを扱う。
- [ ] thinkingの設定と、思考・本文・tool callの分離を実装する。
- [ ] JSON Schema形式保証を生成filterと検証へ接続する。
- [ ] usageとモデルcontext/output上限を正しく公開・設定し、容量超過や無効な入力を明示する。
- [ ] 検証済みTP2/RCCL、K5/V4、MTP3、Engram単一RAM owner＋mlock、電力切替をHTTPの実推論で確認する。
- [ ] SSEテキスト、切断キャンセル、動的batchとprefix再利用が新機能でも動くことを確認する。
- [ ] OpenCodeの専用provider設定で実際に接続する。
- [ ] OpenCodeがread/write/shell等を実行し、結果を再入力して完了できることを実証する。
- [ ] 小さなサンプルコードをOpenCodeに作らせ、テストを実行して成果物を残す。
- [ ] 手順・API制約・実行証拠とサンプルを文書化し、最終状態を項目ごとに監査する。

## 作業順

1. protocol/parser、schema/filter、runtime統合を独立したモジュールに分けて実装する。
2. HTTP handlerへ統合し、chunk境界・履歴・エラー・選択制御のCPUテストを行う。
3. loopbackのみでV620×2サーバーを起動し、実モデルでtoolとJSON生成を検証する。
4. 専用のOpenCode設定と隔離したサンプル作業ディレクトリを使い、実ツール往復とコード作成を行う。
5. 実行証拠を確認して完了条件をチェックする。

remote委譲は既に許可されたOpenCode Go/Qwen3.8 Flashだけを使い、利用制限時は通常Codex subagentへ切り替える。GPU作業はrootが逐次管理し、他のプロセスや既存OpenCodeの設定を変更しない。モデルの再量子化・学習、外部へのAPI公開は行わない。
