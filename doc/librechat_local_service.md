# LibreChat local service — 2026-10-18

このホストの既存LibreChatとQwen3.8 Flash Nextサービスを、PCから利用するための運用メモ。
一般向けの新規インストール手順ではない。

## 接続先と認証

- LAN内のPC: <http://192.168.0.66:3080>
- サーバー自身: <http://127.0.0.1:3080>
- 既存アカウントの認証情報: `/home/homelab1/coding-local/librechat/.local/login.json`（mode 600）。パスワードやトークンはGitへ保存しない。
- 別ネットワークからは既存のSSH接続で `ssh -N -L 3080:127.0.0.1:3080 homelab1@<SSH接続先>` と転送し、PCの <http://localhost:3080> を開く。

LibreChatの `compose.local.yaml` に `192.168.0.66:3080:3080` を追加し、
既存の `127.0.0.1:3080:3080` も維持した。変更前ファイルは同ディレクトリの
`.local/compose.local.yaml.before-lan-20261018-121155` に保存した。
LANアドレスが変わった場合はポートのbind先も更新する。
認証設定、モデル選択、MCP設定、会話データは維持。
ルーターのポート転送設定は変更していない。

## 起動と停止

```bash
systemctl --user start rocm-exl3-api.service
docker compose -f /home/homelab1/coding-local/librechat/compose.local.yaml up -d

systemctl --user status rocm-exl3-api.service
curl http://127.0.0.1:3953/health
curl http://127.0.0.1:3080/readyz
```

systemdがactiveでもモデルのロード中には推論できない。`/health` がHTTP 200を
返してからチャットする。起動時点の既存構成はV620×2 TP2/RCCL、batch 3、
共有768Ki cache、MTP1、RAM Engram、vision有効、xhigh。
実際のモデルパスは既存ランチャーの
`/work/models/qwen38-flash-next-orcarouter-exl3-3.05bpw-original-vision`。

GPUを空けるには、利用中の生成がないことを確認して次を実行する。

```bash
systemctl --user stop rocm-exl3-api.service
```

Web UIも止める場合は、データを残す `stop` を使う。

```bash
docker compose -f /home/homelab1/coding-local/librechat/compose.local.yaml stop
```

## 今回の確認

- 停止していた既存 `rocm-exl3-api.service` を起動。検証終了時の再起動回数0。
- LibreChatとMongoDBのDocker healthcheck正常。
- ホストからLANアドレス宛ての画面取得、`/readyz`、既存アカウントのログインがHTTP 200。
- LibreChatの通常チャットAPIとSSEを経由し、Qwenが「接続できました」と回答。
- 別の物理PCでのブラウザ操作、Office連携、MCP実行、画像入力は今回の検証対象外。

検証スクリプトと結果は
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/librechat-start-20261018/`
に保存した。推論サービスは既存のenabled設定、LibreChatは既存の
`restart: unless-stopped` を維持している。
