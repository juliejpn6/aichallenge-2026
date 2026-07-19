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
trap 'echo "child got SIGTERM" > "${MARKER}"; exit 0' TERM
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

if [ ! -f "${MARKER}" ] || ! grep -q "child got SIGTERM" "${MARKER}"; then
    echo "FAIL: child did not receive forwarded SIGTERM"
    exit 1
fi

echo "PASS: SIGTERM to wrapper forwarded as SIGTERM to child, wrapper exited cleanly"
