#!/usr/bin/env bash
# =============================================================
# collect_perf_logs.sh
#   263節(2026-08-02、予選環境ギャップ分析の準備Phase 3): 走行後、
#   [PERF]系ログの比較分析(analyze_perf_gap.py)に必要な一式を
#   1つのtar.gzへまとめる。制御には一切関与しない、収集専用ツール。
#
#   使い方(run_autoware.bashの出力先ディレクトリを渡す):
#     bash collect_perf_logs.sh <run_dir> [出力先ディレクトリ]
#   例:
#     bash collect_perf_logs.sh /output/latest/d1
#     bash collect_perf_logs.sh output/20260802-082816/d1 /tmp/perfpkg
#
#   <run_dir>には run_autoware.bash が
#     exec > "${out_dir}/autoware.log" 2>&1
#   で全ノードのstdout/stderrを書き出す先を渡す(run_autoware.bash:29-30)。
#   [PERF]/[PERF-DT]/[PERF-SPIKE]/[PERF-RUSAGE]/[PERF-PLATFORM]は
#   いずれもmpc_controllerノードのstdoutへprint()されるため、全て
#   このautoware.log 1本に含まれる(ノード別に別ファイルを探す必要はない)。
#
#   予選環境そのものでこのスクリプトを実行できるとは限らない(運営が
#   ダウンロード形式でログを配布する可能性が高い)。その場合は配布された
#   autoware.log相当のファイルを<run_dir>直下に autoware.log という名前で
#   置いてから実行すれば、それ以降(config.yamlスナップショット・git情報・
#   PERF-PLATFORM抽出・tar.gz化)は同じ手順で使える。
# =============================================================
set -eo pipefail

RUN_DIR="${1:?使い方: collect_perf_logs.sh <run_dir> [出力先ディレクトリ]}"
OUT_DIR="${2:-.}"

LOG_FILE="${RUN_DIR}/autoware.log"
if [ ! -f "${LOG_FILE}" ]; then
    echo "エラー: ${LOG_FILE} が見つかりません。" >&2
    echo "  run_autoware.bashは <出力先>/autoware.log へ全ノードのstdout/stderrを" >&2
    echo "  書き出す(exec > \"\${out_dir}/autoware.log\" 2>&1、run_autoware.bash:29-30)。" >&2
    echo "  ローカルではリポジトリ直下の output/<timestamp>/d<id>/ または" >&2
    echo "  output/latest/d<id>/ (最新走行へのシンボリックリンク)を指定すること。" >&2
    echo "  予選環境からダウンロードしたログを使う場合は、<run_dir>直下に" >&2
    echo "  autoware.log という名前で配置してから再実行すること。" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="$(mktemp -d)"
PKG_NAME="perf_logs_${TS}"
PKG_DIR="${WORK_DIR}/${PKG_NAME}"
mkdir -p "${PKG_DIR}"

cp "${LOG_FILE}" "${PKG_DIR}/autoware.log"

# config.yamlスナップショット(この走行に使われた設定値を後から追跡できるように)。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/../config/config.yaml"
if [ -f "${CONFIG_PATH}" ]; then
    cp "${CONFIG_PATH}" "${PKG_DIR}/config_snapshot.yaml"
else
    echo "警告: config.yamlが見つかりません(${CONFIG_PATH})。スキップします。" >&2
fi

# gitコミットハッシュ+作業ツリーの汚れ具合(どのコードでの走行かを後から追える証跡)。
{
    echo "collected_at: $(date -Iseconds)"
    echo "run_dir: ${RUN_DIR}"
    if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
        echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
        echo "git_branch: $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
        echo "git_dirty (リポジトリルートからの相対パス):"
        git -C "${REPO_ROOT}" status --short
    else
        echo "git_commit: N/A (gitリポジトリが見つからない環境、例: 予選環境)"
    fi
} > "${PKG_DIR}/collection_info.txt"

# [PERF-PLATFORM]行だけを抜き出す(巨大なログ全体を読まずにプラットフォーム
# 構成(cgroup制限・governor・可用性マップ等)をすぐ確認できるように)。
if ! grep "\[PERF-PLATFORM\]" "${LOG_FILE}" > "${PKG_DIR}/perf_platform_lines.txt"; then
    echo "([PERF-PLATFORM]行なし: 263節Phase 1/2より前の計装で収集されたログの可能性)" \
        > "${PKG_DIR}/perf_platform_lines.txt"
fi

TAR_PATH="${OUT_DIR}/${PKG_NAME}.tar.gz"
tar -czf "${TAR_PATH}" -C "${WORK_DIR}" "${PKG_NAME}"
rm -rf "${WORK_DIR}"

echo "収集完了: ${TAR_PATH} ($(du -h "${TAR_PATH}" | cut -f1))"
echo "内訳:"
tar -tzvf "${TAR_PATH}"
