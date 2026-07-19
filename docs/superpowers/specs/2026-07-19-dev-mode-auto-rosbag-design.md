# 複数車両シミュレーション実行時の自動rosbag収集

## 背景・目的

このリポジトリには複数台のAutowareを別ROS_DOMAIN_ID(1..N)で並行起動する
シミュレーション起動経路が2つある。

1. `make dev2`/`make dev3`/`make dev4` — ローカル開発用の混走モード
2. `run_parallel_submissions.bash` — 提出tar.gz(最大4件)を評価用に
   並行実行するモード

現状、走行データのrosbag収集はどちらの経路でも `docker exec` で
コンテナに入り `aichallenge/workspace/src/aichallenge_tools/record_run.sh`
を手動実行する運用になっている。

この手動ステップを廃止し、両方の起動経路でego車(ROS_DOMAIN_ID=1)の
rosbagを自動収集し、それぞれの停止操作(`make down` /
`run_parallel_submissions.bash down`)でグレースフルに停止できるように
する。録画の中身(スクリプト・compose定義)は両経路で共通化し、
呼び出し側だけをそれぞれの既存の慣習に合わせて拡張する。

## スコープ

- 録画対象はどちらの経路でも **ego(ROS_DOMAIN_ID=1)のみ**。
  2号車以降(dev2/3/4の2〜4号車、parallel submissionsのd2〜d4)は
  録画しない。
  - 理由: `record_run.sh` のトピック一覧はego視点のv2x/overtake診断
    (`/v2x/vehicle_positions`, `/mpc/overtake_status` 等)を含んでおり、
    ego 1本の記録で混走・追い越し分析に必要なデータが揃うため。
- 録画トピック一覧・QoS override・mcap形式は既存の `record_run.sh` を
  そのまま流用する(重複実装しない)。
- `dev2`/`dev3`/`dev4` は同一Makefileレシピを共有しているため、3つとも
  同じ仕組みで自動録画する(dev3だけの特別扱いはしない)。
- `run_parallel_submissions.bash` は `autoware-eval-base` を使い
  `/aichallenge` をホストマウントしない(evalイメージに焼き込み)構成だが、
  rosbagの保存先はdevモードと同じ `aichallenge/workspace/bag/` に統一する
  (`bag-recorder` サービス自体は常に dev用イメージ・`autoware-base` を
  使うため、どちらの経路から起動しても保存先は変わらない)。

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

### run_parallel_submissions.bash への統合

`bag-recorder` サービス・`record_dev_bag.bash` は上記と完全に共通のまま
再利用する。追加するのは `run_parallel_submissions.bash` の `main()` 内、
`autoware-d1..dN` 起動ループの**後**に1呼び出しだけ。

```bash
log_dir="/output/${run_id}/d1"
log "Starting bag-recorder (ego rosbag, ROS_DOMAIN_ID=1)"
LOG_DIR="${log_dir}" \
    ROS_DOMAIN_ID=1 \
    BAG_TAG="parallel_${run_id}" \
    docker compose up -d --force-recreate bag-recorder
```

- この script は `-p` を使わずデフォルトprojectで全サービスを起動する
  ため、`bag-recorder` も `-p` なしで起動する。
  `run_parallel_submissions.bash down` の
  `docker compose down --remove-orphans` がそのまま同じ
  グレースフル停止(SIGTERM→trap→SIGINT→mcap finalize)を行うため、
  **down分岐の変更も不要**。
- 他の `docker compose up -d` 呼び出しがすべて `--force-recreate` を
  使っている(同一script再実行時に確実に新しいコンテナへ切り替える
  ため)慣習に合わせ、`bag-recorder` も `--force-recreate` を付ける。
  (Makefile経路の`dev2/3/4`は`--force-recreate`を使っていないため付けず、
  それぞれの既存呼び出し方の慣習をそのまま踏襲する。)
- `LOG_DIR` はこのscriptの既存規則(`output/<run_id>/d<domain_id>/` に
  そのドメインの `autoware.log` 等を集約する)に合わせ、domain 1用の
  `output/<run_id>/d1/` を使う。よって `rosbag.log` は
  `output/<run_id>/d1/rosbag.log` に出力される
  (Makefile経路ではトップレベルの `output/<timestamp>/rosbag.log` に
  なるのに対し、こちらは呼び出し元scriptの既存の階層規則に従う)。
- `BAG_TAG` に `run_id` を含めることで、bagディレクトリ名
  (`run_parallel_<run_id>_<timestamp>/`)から
  `output/<run_id>/` 側の実行と突き合わせられるようにする。

## 命名規則(既存慣習の延長。新規パターンは導入しない)

- **rosbagディレクトリ名**: `record_run.sh` の `run_<TAG>_<timestamp>`
  規則をそのまま使い、TAGを呼び出し元に応じて変える。
  - Makefile経路: 実行したmakeターゲット名(`dev2`/`dev3`/`dev4`)。
    例: `aichallenge/workspace/bag/run_dev3_20260719_123456/`
  - `run_parallel_submissions.bash`経路: `parallel_<run_id>`。
    例: `aichallenge/workspace/bag/run_parallel_20260719-123456_20260719_123500/`
- **プロセスログファイル名**: 呼び出し元それぞれの既存の `.log` 配置慣習
  (サービス名ベースの `.log` を `$LOG_DIR` 直下に置く)をそのまま延長し、
  `rosbag.log` という名前にする。`$LOG_DIR` の値そのものは呼び出し元の
  既存規則通り(下表参照)。

## 出力先まとめ

| 内容 | コンテナ内パス | ホスト側パス(Makefile経路) | ホスト側パス(parallel_submissions経路) |
|---|---|---|---|
| rosbag本体(mcap) | `/aichallenge/workspace/bag/run_<TAG>_<ts>/` | `aichallenge/workspace/bag/run_dev3_<ts>/` | `aichallenge/workspace/bag/run_parallel_<run_id>_<ts>/` |
| 録画プロセスの標準出力/エラー | `$LOG_DIR/rosbag.log` | `output/<timestamp>/rosbag.log` | `output/<run_id>/d1/rosbag.log` |

## エッジケース

- `make dev3` を `make down` せずに再実行した場合、
  `docker compose up -d` は既存コンテナをそのまま使う
  (録画プロセスは継続、二重起動はしない)。
  `run_parallel_submissions.bash` 側は `--force-recreate` を使うため、
  再実行のたびに新しいTAG(`parallel_<新run_id>`)で録画し直される。
- `qos_override.yaml` は毎回 `record_run.sh` が上書き生成する
  (既存動作のまま変更しない)。共有の
  `aichallenge/workspace/bag/` に両経路の録画が集約されるため、
  同時に両方の経路を実行すると同じ `qos_override.yaml` を
  取り合う形になるが、内容は同一なので実害はない。

## 変更ファイル

1. 新規: `aichallenge/utils/record_dev_bag.bash`
2. 変更: `docker-compose.yml` (`bag-recorder` サービス追加)
3. 変更: `Makefile` (`dev2 dev3 dev4` レシピ拡張)
4. 変更: `run_parallel_submissions.bash` (`autoware-d1..dN` 起動後に
   `bag-recorder` 起動を1呼び出し追加)

## テスト方針

- `make dev3` 実行後、`docker compose -p 1 ps` で `bag-recorder`
  コンテナが起動していることを確認。
  `aichallenge/workspace/bag/run_dev3_*/` に mcap ファイルが
  生成されていることを確認。
  `make down` 実行後、bagディレクトリの `metadata.yaml` が
  正しく書き出されており(rosbag2の正常終了の証跡)、
  `output/<timestamp>/rosbag.log` にSIGINT起因のクリーンな
  終了ログが残っていることを確認。
- `./run_parallel_submissions.bash --submit <tar1> <tar2>` 実行後、
  `docker compose ps` で `bag-recorder` コンテナが起動していることを確認。
  `aichallenge/workspace/bag/run_parallel_<run_id>_*/` に mcap ファイルが
  生成されていることを確認。
  `./run_parallel_submissions.bash down` 実行後、同様に `metadata.yaml`
  と `output/<run_id>/d1/rosbag.log` のクリーンな終了ログを確認。
  注意: `bag-recorder` は常に `autoware-base`(`aichallenge-2025-dev`
  イメージ)を使うため、この経路単独で使う場合でも事前に
  `aichallenge-2025-dev` イメージがビルド済みである必要がある
  (`make autoware-build` 等)。
- 自動テストはこの機能の根幹(SIGINTが実際に`ros2 bag record`まで届き、
  mcapが正常にfinalizeされること)をサンドボックス制約により検証できない
  (Task 1参照)。したがって上記の手動統合テストは省略可能なオプションでは
  なく、本機能をマージ・運用する前に必ず一度は実行して結果を確認すること。
