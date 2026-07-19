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
