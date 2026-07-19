# dev2/dev3/dev4 実行時の自動rosbag収集

## 背景・目的

`make dev2`/`make dev3`/`make dev4` は複数台のAutowareを別ROS_DOMAIN_ID
(1..N) で並行起動する混走シミュレーションモード。現状、走行データの
rosbag収集は `docker exec` でコンテナに入り
`aichallenge/workspace/src/aichallenge_tools/record_run.sh` を手動実行する
運用になっている。

この手動ステップを廃止し、`make dev2`/`dev3`/`dev4` 実行と同時に
ego車(ROS_DOMAIN_ID=1)のrosbagを自動収集し、`make down` で
グレースフルに停止できるようにする。

## スコープ

- 録画対象は **ego(ROS_DOMAIN_ID=1)のみ**。2号車・3号車以降は録画しない。
  - 理由: `record_run.sh` のトピック一覧はego視点のv2x/overtake診断
    (`/v2x/vehicle_positions`, `/mpc/overtake_status` 等)を含んでおり、
    ego 1本の記録で混走・追い越し分析に必要なデータが揃うため。
- 録画トピック一覧・QoS override・mcap形式は既存の `record_run.sh` を
  そのまま流用する(重複実装しない)。
- `dev2`/`dev3`/`dev4` は同一Makefileレシピを共有しているため、3つとも
  同じ仕組みで自動録画する(dev3だけの特別扱いはしない)。

## アーキテクチャ

### 新規スクリプト: `aichallenge/utils/record_dev_bag.bash`

`record_run.sh` (トピック一覧の唯一のソースとして維持) を `setsid` で
新しいプロセスグループの子として起動するラッパー。

- 引数でタグを受け取り `record_run.sh <TAG>` を呼び出す
  (例: `record_dev_bag.bash dev3` → `record_run.sh dev3`)
- `EXIT`/`SIGINT`/`SIGTERM` を trap し、受信したら子プロセスグループに
  `SIGINT` を送って `ros2 bag record` を正常終了(mcap finalize)させる。
  `aichallenge/utils/record_rosbag.bash` の trap パターンを踏襲する。

### docker-compose.yml: `bag-recorder` サービス追加

- `autoware-base` アンカーを再利用(ボリューム/イメージ共通化。
  `autoware-command` サービスと同じ再利用パターン)。
- `command`:
  `exec /aichallenge/utils/record_dev_bag.bash ${BAG_TAG:-dev} > "${LOG_DIR:-/output}/rosbag.log" 2>&1`
  - `exec` によりラッパースクリプトがコンテナのPID1になるため、
    `docker compose down` のSIGTERMが直接ラッパーに届く。

### Makefile: `dev2 dev3 dev4` レシピ拡張

既存の3台分autoware起動ループの後に、ego(project 1)向けの
bag-recorder起動を追加する。

```
dev2 dev3 dev4: simulator
	@N=$(@:dev%=%); \
	echo "Start $$N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$$N via docker compose -p)"; \
	for p in $$(seq 1 $$N); do LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=$$p docker compose -p $$p up -d autoware; done; \
	LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=1 BAG_TAG=$@ docker compose -p 1 up -d bag-recorder; \
	echo "Recording ego rosbag (ROS_DOMAIN_ID=1) -> aichallenge/workspace/bag/run_$@_*"; \
	echo "To Stop: make down"
```

`make down` は既存の
`for p in 1 2 3 4; do docker compose -p $$p down --remove-orphans; done`
がそのまま project 1 の `bag-recorder` コンテナにも SIGTERM
(`stop_grace_period: 10s` 猶予)→SIGKILL を送るため、**Makefileの
`down` ターゲット自体は変更不要**。

## 命名規則(既存慣習の延長。新規パターンは導入しない)

- **rosbagディレクトリ名**: `record_run.sh` の `run_<TAG>_<timestamp>`
  規則をそのまま使い、TAGを実行したmakeターゲット名
  (`dev2`/`dev3`/`dev4`)にする。
  例: `aichallenge/workspace/bag/run_dev3_20260719_123456/`
- **プロセスログファイル名**: `simulator` サービスの `awsim.log` 等、
  `$LOG_DIR` 直下にサービス名ベースの `.log` を置く慣習に合わせ、
  `rosbag.log` を `$LOG_DIR` 直下(ホスト側 `output/<timestamp>/rosbag.log`)
  に置く。

## 出力先まとめ

| 内容 | コンテナ内パス | ホスト側パス |
|---|---|---|
| rosbag本体(mcap) | `/aichallenge/workspace/bag/run_<mode>_<ts>/` | `aichallenge/workspace/bag/run_<mode>_<ts>/` |
| 録画プロセスの標準出力/エラー | `$LOG_DIR/rosbag.log` | `output/<timestamp>/rosbag.log` |

## エッジケース

- `make dev3` を `make down` せずに再実行した場合、
  `docker compose up -d` は既存コンテナをそのまま使う
  (録画プロセスは継続、二重起動はしない)。
- `qos_override.yaml` は毎回 `record_run.sh` が上書き生成する
  (既存動作のまま変更しない)。

## 変更ファイル

1. 新規: `aichallenge/utils/record_dev_bag.bash`
2. 変更: `docker-compose.yml` (`bag-recorder` サービス追加)
3. 変更: `Makefile` (`dev2 dev3 dev4` レシピ拡張)

## テスト方針

- `make dev3` 実行後、`docker compose -p 1 ps` で `bag-recorder`
  コンテナが起動していることを確認。
- `aichallenge/workspace/bag/run_dev3_*/` に mcap ファイルが
  生成されていることを確認。
- `make down` 実行後、bagディレクトリの `metadata.yaml` が
  正しく書き出されており(rosbag2の正常終了の証跡)、
  `output/<timestamp>/rosbag.log` にSIGINT起因のクリーンな
  終了ログが残っていることを確認。
