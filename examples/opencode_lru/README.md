# OpenCodeによる実APIサンプル（2026-10-01）

OpenCode 2.0.12が、V620×2のローカルEXL3 APIを使って生成したLRUCacheとunittestです。
Codexは成果物をレビューし、テストを別途実行しました。実装・テスト本文はOpenCodeの出力です。

```bash
cd examples/opencode_lru
python3 -m unittest -v
```

14テストが成功します。挿入、読込による順序変更、更新、eviction、欠損値と不正capacityを確認します。
実行したread/write/shellのcall IDと、APIの引数がクライアントで一致した証拠は`verification.json`にあります。
サーバーがtoolを実行する構成ではなく、OpenCodeがファイル操作・コマンド実行を行います。

再生成する場合は、[OpenCode設定例](../opencode.jsonc)を別の作業ディレクトリへコピーし、
APIを起動してからSPEC.mdに沿って実装・テスト実行を指示してください。
検証時はthinking=true、reasoning_effort=lowを使用しました。
