#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/record_dev_bag.bash"

# --- Static check -------------------------------------------------------
# cleanup() must signal the child with SIGINT, never SIGTERM: ros2 bag
# record only finalizes its mcap gracefully on SIGINT (same contract as
# the existing aichallenge/utils/record_rosbag.bash it mirrors). This is
# checked statically because this sandbox cannot deliver a real SIGINT to
# a backgrounded child process (independently reproduced: `kill -INT --
# "-$PID"` and `kill -INT "$PID"` both hang here, even against a child
# that traps SIGINT, while `kill -TERM` to the same child is delivered
# immediately). Full delivery must be confirmed manually against the real
# Docker container, per the plan's manual integration test step.
CLEANUP_BODY="$(awk '/^cleanup\(\)/,/^}/' "${TARGET}")"
if ! grep -q 'kill -INT' <<<"${CLEANUP_BODY}"; then
    echo "FAIL: cleanup() does not send SIGINT to the child"
    exit 1
fi
if grep -q 'kill -TERM' <<<"${CLEANUP_BODY}"; then
    echo "FAIL: cleanup() must not send SIGTERM to the child (breaks graceful mcap finalize)"
    exit 1
fi
echo "PASS (static): cleanup() signals the child with SIGINT only"

# --- Dynamic check -------------------------------------------------------
# The wrapper must react to an external signal (SIGTERM, what docker
# stop/compose down actually send to the container) by running its trap
# and attempting to forward it to the child. We only assert the trap
# fires and issues the forwarding kill -- we do not wait for the child to
# exit, since that requires real SIGINT delivery, which this sandbox
# cannot provide (see above).
TMPDIR="$(mktemp -d)"
cleanup_tmp() { rm -rf "${TMPDIR}"; }
trap cleanup_tmp EXIT

MOCK_CHILD="${TMPDIR}/mock_child.bash"
cat > "${MOCK_CHILD}" <<'EOF'
#!/bin/bash
echo "child started with tag=$1"
while true; do sleep 0.1; done
EOF
chmod +x "${MOCK_CHILD}"

OUT="${TMPDIR}/wrapper.out"
RECORD_DEV_BAG_CHILD="${MOCK_CHILD}" bash "${TARGET}" mytag >"${OUT}" 2>&1 &
WRAPPER_PID=$!

sleep 0.5
if ! kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    echo "FAIL: wrapper exited before signal was sent"
    exit 1
fi

kill -TERM "${WRAPPER_PID}"
sleep 0.5

if ! grep -q "Rosbag recording cleanup" "${OUT}"; then
    echo "FAIL: wrapper did not run cleanup() after receiving SIGTERM"
    cat "${OUT}"
    exit 1
fi
echo "PASS (dynamic): wrapper's trap fired and attempted to forward the signal"

# Best-effort teardown: cleanup()'s own kill -INT cannot finish the child in
# this sandbox, so reap both processes directly rather than waiting on them.
CHILD_PID="$(pgrep -f "mock_child.bash mytag" | head -n1 || true)"
[ -n "${CHILD_PID}" ] && kill -9 "${CHILD_PID}" 2>/dev/null || true
kill -9 "${WRAPPER_PID}" 2>/dev/null || true
wait "${WRAPPER_PID}" 2>/dev/null || true
