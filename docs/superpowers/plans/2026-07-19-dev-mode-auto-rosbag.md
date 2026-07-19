# 複数車両シミュレーション実行時の自動rosbag収集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make dev`/`make dev2`/`make dev3`/`make dev4` と `run_parallel_submissions.bash` のすべての起動経路で、ego車(ROS_DOMAIN_ID=1)のrosbagを自動収集し、それぞれの停止操作でグレースフルに停止できるようにする。

**Architecture:** 新しいシグナルトラップ付きラッパースクリプト `record_dev_bag.bash` が既存の `record_run.sh`(トピック一覧の唯一のソース、変更しない)を子プロセスグループで起動し、SIGTERM/SIGINTを受けたら子プロセスグループにSIGINTを転送してmcapを正常finalizeさせる。これを新規docker-composeサービス `bag-recorder` から起動し、Makefileの`dev2/dev3/dev4`レシピと`run_parallel_submissions.bash`の両方から、それぞれの既存呼び出し慣習に合わせて呼び出す。

**Tech Stack:** Bash, Docker Compose, ROS 2 (rosbag2/mcap), GNU Make

**設計ドキュメント:** `docs/superpowers/specs/2026-07-19-dev-mode-auto-rosbag-design.md`

## Global Constraints

- 録画対象はどちらの経路でも **ego(ROS_DOMAIN_ID=1)のみ**。他ドメインは録画しない。
- 録画トピック一覧・QoS override・mcap形式は既存の `aichallenge/workspace/src/aichallenge_tools/record_run.sh` をそのまま流用し、変更しない。
- 新しい命名パターンは導入しない。既存の `run_<TAG>_<timestamp>` (bagディレクトリ) と `$LOG_DIR/<service>.log` (プロセスログ) の慣習をそれぞれ延長するだけ。
- `bag-recorder` サービスは常に `autoware-base` アンカー(dev用イメージ・`/aichallenge`ホストマウントあり)を使う。呼び出し元(Makefile経路か`run_parallel_submissions.bash`経路か)に関わらず、rosbagの保存先は常に `aichallenge/workspace/bag/` になる。
- Makefile経路の`dev2/dev3/dev4`は`--force-recreate`を使わない(既存の同レシピの慣習通り)。`run_parallel_submissions.bash`経路は他の`docker compose up -d`呼び出しと同じく`--force-recreate`を使う。

---

### Task 1: `record_dev_bag.bash` ラッパースクリプト

**Files:**
- Create: `aichallenge/utils/record_dev_bag.bash`
- Test: `aichallenge/utils/test_record_dev_bag.bash`

**Interfaces:**
- Produces: `aichallenge/utils/record_dev_bag.bash <TAG>` — 引数`TAG`(省略時`dev`)で`record_run.sh <TAG>`を新しいプロセスグループの子として起動し、`EXIT`/`SIGINT`/`SIGTERM`を受けたら子プロセスグループに`SIGINT`を送って正常終了させる。子スクリプトのパスは環境変数`RECORD_DEV_BAG_CHILD`で上書き可能(デフォルトは`/aichallenge/workspace/src/aichallenge_tools/record_run.sh`)。この上書き口はテスト用に導入する。
- Consumes: なし(このタスク単体で完結)。

このスクリプトはROS 2環境がなくても、trap・シグナル転送のロジック単体をmockの子スクリプトで検証できる。ROS 2固有のトピック録画自体は`record_run.sh`側の既存動作であり、このタスクではテストしない。

- [ ] **Step 1: 失敗するテストを書く**

`aichallenge/utils/test_record_dev_bag.bash` を作成する:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/record_dev_bag.bash"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

export MARKER="${TMPDIR}/marker"
MOCK_CHILD="${TMPDIR}/mock_child.bash"

cat > "${MOCK_CHILD}" <<'EOF'
#!/bin/bash
trap 'echo "child got SIGINT" > "${MARKER}"; exit 0' SIGINT
echo "child started with tag=$1"
while true; do sleep 0.1; done
EOF
chmod +x "${MOCK_CHILD}"

RECORD_DEV_BAG_CHILD="${MOCK_CHILD}" bash "${TARGET}" mytag &
WRAPPER_PID=$!

sleep 0.5
if ! kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    echo "FAIL: wrapper exited before signal was sent"
    exit 1
fi

kill -TERM "${WRAPPER_PID}"

for _ in $(seq 1 30); do
    kill -0 "${WRAPPER_PID}" 2>/dev/null || break
    sleep 0.1
done

if kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    echo "FAIL: wrapper did not exit within 3s of SIGTERM"
    kill -9 "${WRAPPER_PID}" 2>/dev/null || true
    exit 1
fi

if [ ! -f "${MARKER}" ] || ! grep -q "child got SIGINT" "${MARKER}"; then
    echo "FAIL: child did not receive forwarded SIGINT"
    exit 1
fi

echo "PASS: SIGTERM to wrapper forwarded as SIGINT to child, wrapper exited cleanly"
```

```bash
chmod +x aichallenge/utils/test_record_dev_bag.bash
```

- [ ] **Step 2: テストを実行し、失敗することを確認する**

Run: `bash aichallenge/utils/test_record_dev_bag.bash`
Expected: `record_dev_bag.bash: No such file or directory` のようなエラーで非ゼロ終了(`record_dev_bag.bash`がまだ存在しないため)。

- [ ] **Step 3: 最小実装を書く**

`aichallenge/utils/record_dev_bag.bash` を作成する:

```bash
#!/bin/bash
set -euo pipefail

TAG="${1:-dev}"
CHILD_SCRIPT="${RECORD_DEV_BAG_CHILD:-/aichallenge/workspace/src/aichallenge_tools/record_run.sh}"
PID=""

cleanup() {
    if [ -z "${PID}" ]; then
        return 0
    fi
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Rosbag recording cleanup... (PID/PGID=${PID})"
        kill -INT -- "-${PID}" 2>/dev/null || kill -INT "${PID}" 2>/dev/null || true
        wait "${PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT SIGINT SIGTERM

if command -v setsid >/dev/null 2>&1; then
    setsid bash "${CHILD_SCRIPT}" "${TAG}" &
else
    bash "${CHILD_SCRIPT}" "${TAG}" &
fi
PID=$!
wait "${PID}" || true
```

```bash
chmod +x aichallenge/utils/record_dev_bag.bash
```

- [ ] **Step 4: テストを実行し、パスすることを確認する**

Run: `bash aichallenge/utils/test_record_dev_bag.bash`
Expected:
```
child started with tag=mytag
PASS: SIGTERM to wrapper forwarded as SIGINT to child, wrapper exited cleanly
```
(exit code 0)

- [ ] **Step 5: シェル構文チェック**

Run: `bash -n aichallenge/utils/record_dev_bag.bash && echo OK`
Expected: `OK`

- [ ] **Step 6: コミット**

```bash
git add aichallenge/utils/record_dev_bag.bash aichallenge/utils/test_record_dev_bag.bash
git commit -m "feat(utils): add record_dev_bag.bash wrapper for graceful rosbag stop"
```

---

### Task 2: `docker-compose.yml` に `bag-recorder` サービスを追加

**Files:**
- Modify: `docker-compose.yml` (`autoware-command` サービス定義の直後、`autoware-simulator-evaluation` の直前に挿入)

**Interfaces:**
- Consumes: Task 1 の `aichallenge/utils/record_dev_bag.bash`。
- Produces: `bag-recorder` サービス。呼び出し例:
  `BAG_TAG=<tag> LOG_DIR=<dir> ROS_DOMAIN_ID=<id> docker compose [-p <proj>] up -d [--force-recreate] bag-recorder`
  コンテナ内では `exec /aichallenge/utils/record_dev_bag.bash ${BAG_TAG:-dev} > "${LOG_DIR:-/output}/rosbag.log" 2>&1` を実行する。

このタスクは宣言的なcompose設定の追加のみのため、「テスト」は `docker compose config` によるYAML妥当性・サービス列挙で代替する(Docker Compose自体にYAML構文検証機能があるため、これを利用する)。

- [ ] **Step 1: 変更前の状態を確認する(失敗させる)**

Run: `docker compose config --services`
Expected: 出力に `bag-recorder` が含まれない(現状のサービス一覧のみ)。

- [ ] **Step 2: `docker-compose.yml` を編集する**

`autoware-command` サービスの直後にこのブロックを追加する(既存の `autoware-command:` の次の行、`autoware-simulator-evaluation:` の前):

```yaml
  bag-recorder:
    <<: *autoware-base
    command: ["bash", "-lc", "exec /aichallenge/utils/record_dev_bag.bash ${BAG_TAG:-dev} > \"${LOG_DIR:-/output}/rosbag.log\" 2>&1"]

```

- [ ] **Step 3: 変更後の状態を確認する(パスさせる)**

Run: `docker compose config --services`
Expected: 出力に `bag-recorder` が追加されている(他の既存サービスも全て残っている)。

Run: `docker compose config --quiet && echo OK`
Expected: `OK` (YAML構文・変数展開エラーなし)。

- [ ] **Step 4: コミット**

```bash
git add docker-compose.yml
git commit -m "feat(compose): add bag-recorder service for automatic ego rosbag recording"
```

---

### Task 3: `Makefile` の `dev2 dev3 dev4` レシピ拡張

**Files:**
- Modify: `Makefile:78-82`

**Interfaces:**
- Consumes: Task 2 の `bag-recorder` サービス。
- Produces: `make dev2`/`make dev3`/`make dev4` が、既存のautoware起動ループの後に project 1 で `bag-recorder` を起動するようになる。`make down` 側の変更は不要(既存の `for p in 1 2 3 4; do docker compose -p $$p down --remove-orphans; done` がそのまま project 1 の `bag-recorder` にもグレースフル停止を及ぼす)。

- [ ] **Step 1: 変更前の状態を確認する(失敗させる)**

Run: `make -n dev3`
Expected(タイムスタンプ部分は実行時刻により変わる):
```
echo "Start AWSIM (SIM_MODE=dev3)"
LOG_DIR=/output/<timestamp> SIM_MODE="dev3" ROS_DOMAIN_ID=0 docker compose up -d simulator
N=3; \
echo "Start $N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$N via docker compose -p)"; \
for p in $(seq 1 $N); do LOG_DIR=/output/<timestamp> ROS_DOMAIN_ID=$p docker compose -p $p up -d autoware; done; \
echo "To Stop: make down"
```
(`bag-recorder` を起動する行がまだ存在しないことを確認する)

- [ ] **Step 2: `Makefile` を編集する**

`Makefile:78-82` の現在の内容:

```makefile
dev2 dev3 dev4: simulator
	@N=$(@:dev%=%); \
	echo "Start $$N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$$N via docker compose -p)"; \
	for p in $$(seq 1 $$N); do LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=$$p docker compose -p $$p up -d autoware; done; \
	echo "To Stop: make down"
```

これを次のように変更する:

```makefile
dev2 dev3 dev4: simulator
	@N=$(@:dev%=%); \
	echo "Start $$N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$$N via docker compose -p)"; \
	for p in $$(seq 1 $$N); do LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=$$p docker compose -p $$p up -d autoware; done; \
	LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=1 BAG_TAG=$@ docker compose -p 1 up -d bag-recorder; \
	echo "Recording ego rosbag (ROS_DOMAIN_ID=1) -> aichallenge/workspace/bag/run_$@_*"; \
	echo "To Stop: make down"
```

- [ ] **Step 3: 変更後の状態を確認する(パスさせる)**

Run: `make -n dev3`
Expected(タイムスタンプ部分は実行時刻により変わる):
```
echo "Start AWSIM (SIM_MODE=dev3)"
LOG_DIR=/output/<timestamp> SIM_MODE="dev3" ROS_DOMAIN_ID=0 docker compose up -d simulator
N=3; \
echo "Start $N-vehicle dev (autoware on ROS_DOMAIN_ID 1..$N via docker compose -p)"; \
for p in $(seq 1 $N); do LOG_DIR=/output/<timestamp> ROS_DOMAIN_ID=$p docker compose -p $p up -d autoware; done; \
LOG_DIR=/output/<timestamp> ROS_DOMAIN_ID=1 BAG_TAG=dev3 docker compose -p 1 up -d bag-recorder; \
echo "Recording ego rosbag (ROS_DOMAIN_ID=1) -> aichallenge/workspace/bag/run_dev3_*"; \
echo "To Stop: make down"
```

Run: `make -n dev2` と `make -n dev4` でも同様に `BAG_TAG=dev2`/`BAG_TAG=dev4` になっていることを確認する。

- [ ] **Step 4: コミット**

```bash
git add Makefile
git commit -m "feat(makefile): auto-start ego bag-recorder from dev2/dev3/dev4"
```

---

### Task 4: `run_parallel_submissions.bash` への統合

**Files:**
- Modify: `run_parallel_submissions.bash:131-141`

**Interfaces:**
- Consumes: Task 2 の `bag-recorder` サービス。
- Produces: `run_parallel_submissions.bash --submit ...` が、`autoware-d1..dN` 起動ループの後に(project指定なしのデフォルトプロジェクトで)`bag-recorder` を起動するようになる。`run_parallel_submissions.bash down` 側の変更は不要(既存の `docker compose down --remove-orphans` がそのままグレースフル停止を及ぼす)。

- [ ] **Step 1: 変更前の状態を確認する(失敗させる)**

Run: `bash -n run_parallel_submissions.bash && echo OK`
Expected: `OK` (現状でも構文エラーがないことを先に確認しておく)

Run: `grep -n "bag-recorder" run_parallel_submissions.bash || echo "NOT FOUND"`
Expected: `NOT FOUND` (まだ統合されていないことを確認する)

- [ ] **Step 2: `run_parallel_submissions.bash` を編集する**

`run_parallel_submissions.bash:131-141` の現在の内容:

```bash
    for ((domain_id = 1; domain_id <= vehicles; domain_id++)); do
        log_dir="/output/${run_id}/d${domain_id}"
        log "Starting autoware-d${domain_id}"
        LOG_DIR="${log_dir}" \
            RUN_MODE="awsim" \
            ROS_HOME="${log_dir}/.ros" \
            ROS_LOG_DIR="${log_dir}/ros_log" \
            docker compose up -d --force-recreate "autoware-d${domain_id}"
    done

    log "Started. Output: output/${run_id}/d1..d${vehicles}"
}
```

これを次のように変更する(ループの後、最後の`log`行の前に`bag-recorder`起動を追加):

```bash
    for ((domain_id = 1; domain_id <= vehicles; domain_id++)); do
        log_dir="/output/${run_id}/d${domain_id}"
        log "Starting autoware-d${domain_id}"
        LOG_DIR="${log_dir}" \
            RUN_MODE="awsim" \
            ROS_HOME="${log_dir}/.ros" \
            ROS_LOG_DIR="${log_dir}/ros_log" \
            docker compose up -d --force-recreate "autoware-d${domain_id}"
    done

    log_dir="/output/${run_id}/d1"
    log "Starting bag-recorder (ego rosbag, ROS_DOMAIN_ID=1)"
    LOG_DIR="${log_dir}" \
        ROS_DOMAIN_ID=1 \
        BAG_TAG="parallel_${run_id}" \
        docker compose up -d --force-recreate bag-recorder

    log "Started. Output: output/${run_id}/d1..d${vehicles}"
}
```

- [ ] **Step 3: 変更後の状態を確認する(パスさせる)**

Run: `bash -n run_parallel_submissions.bash && echo OK`
Expected: `OK`

Run: `grep -n "bag-recorder" run_parallel_submissions.bash`
Expected:
```
<行番号>:    log "Starting bag-recorder (ego rosbag, ROS_DOMAIN_ID=1)"
<行番号>:        docker compose up -d --force-recreate bag-recorder
```

Run: `./run_parallel_submissions.bash --help`
Expected: 既存の usage テキストがそのまま表示され、エラーなく終了する(このパスはdockerを呼ばないため、編集で壊れていないことの安全確認になる)。

- [ ] **Step 4: コミット**

```bash
git add run_parallel_submissions.bash
git commit -m "feat: auto-start ego bag-recorder from run_parallel_submissions.bash"
```

---

### Task 5: `Makefile` の `dev` レシピ拡張(単一車両モード)

**Files:**
- Modify: `Makefile:70-73`

**Interfaces:**
- Consumes: Task 2 の `bag-recorder` サービス。
- Produces: `make dev` が、既存の`autoware-simulator`起動の後に(project指定なしのデフォルトプロジェクトで)`bag-recorder`を起動するようになる。`make dev`は`ROS_DOMAIN_ID`を明示しないため`autoware-base`アンカーの既定値(`ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}`)によりegoが動く、`dev2/3/4`と同じ意味でのego。`make down`側の変更は不要(既存の`docker compose down --remove-orphans`(project指定なし)がそのままこの`bag-recorder`にもグレースフル停止を及ぼす)。

- [ ] **Step 1: 変更前の状態を確認する(失敗させる)**

Run: `make -n dev`
Expected(タイムスタンプ部分は実行時刻により変わる):
```
echo "Start AWSIM (SIM_MODE=dev)"
LOG_DIR=/output/<timestamp> SIM_MODE="dev" ROS_DOMAIN_ID=0 docker compose up -d simulator
echo "Start Autoware for AWSIM"
LOG_DIR=/output/<timestamp> RUN_MODE=awsim docker compose up -d autoware
echo "Start dev simulation (AWSIM + Autoware)"
echo "To stop: make down  (docker compose down --remove-orphans)"
```
(`bag-recorder` を起動する行がまだ存在しないことを確認する)

- [ ] **Step 2: `Makefile` を編集する**

`Makefile:70-73` の現在の内容:

```makefile
dev: SIM_MODE := dev
dev: simulator autoware-simulator
	@echo "Start dev simulation (AWSIM + Autoware)"
	@echo "To stop: make down  (docker compose down --remove-orphans)"
```

これを次のように変更する:

```makefile
dev: SIM_MODE := dev
dev: simulator autoware-simulator
	@echo "Start dev simulation (AWSIM + Autoware)"
	LOG_DIR=$(LOG_DIR) ROS_DOMAIN_ID=1 BAG_TAG=dev docker compose up -d bag-recorder
	@echo "Recording ego rosbag (ROS_DOMAIN_ID=1) -> aichallenge/workspace/bag/run_dev_*"
	@echo "To stop: make down  (docker compose down --remove-orphans)"
```

- [ ] **Step 3: 変更後の状態を確認する(パスさせる)**

Run: `make -n dev`
Expected(タイムスタンプ部分は実行時刻により変わる):
```
echo "Start AWSIM (SIM_MODE=dev)"
LOG_DIR=/output/<timestamp> SIM_MODE="dev" ROS_DOMAIN_ID=0 docker compose up -d simulator
echo "Start Autoware for AWSIM"
LOG_DIR=/output/<timestamp> RUN_MODE=awsim docker compose up -d autoware
echo "Start dev simulation (AWSIM + Autoware)"
LOG_DIR=/output/<timestamp> ROS_DOMAIN_ID=1 BAG_TAG=dev docker compose up -d bag-recorder
echo "Recording ego rosbag (ROS_DOMAIN_ID=1) -> aichallenge/workspace/bag/run_dev_*"
echo "To stop: make down  (docker compose down --remove-orphans)"
```

- [ ] **Step 4: コミット**

```bash
git add Makefile
git commit -m "feat(makefile): auto-start ego bag-recorder from dev"
```

---

## 手動統合テスト(自動テスト対象外・全経路共通)

実装完了後、以下を手動で確認する(設計ドキュメントの「テスト方針」節に準拠):

1. `make dev` 実行 → `docker compose ps` で `bag-recorder` コンテナが起動していることを確認 → `aichallenge/workspace/bag/run_dev_*/` に mcap ファイルが生成されることを確認 → `make down` 実行後、`metadata.yaml` が正しく書き出され、`output/<timestamp>/rosbag.log` にSIGINT起因のクリーンな終了ログが残ることを確認する。
2. `make dev3` 実行 → `docker compose -p 1 ps` で `bag-recorder` コンテナが起動していることを確認 → `aichallenge/workspace/bag/run_dev3_*/` に mcap ファイルが生成されることを確認 → `make down` 実行後、`metadata.yaml` が正しく書き出され、`output/<timestamp>/rosbag.log` にSIGINT起因のクリーンな終了ログが残ることを確認する。
3. `./run_parallel_submissions.bash --submit <tar1> <tar2>` 実行 → `docker compose ps` で `bag-recorder` コンテナが起動していることを確認 → `aichallenge/workspace/bag/run_parallel_<run_id>_*/` に mcap ファイルが生成されることを確認 → `./run_parallel_submissions.bash down` 実行後、同様に `metadata.yaml` と `output/<run_id>/d1/rosbag.log` のクリーンな終了ログを確認する。
