#!/usr/bin/env bash
# check_race_config.sh — config.yamlのレース値維持キー(CLAUDE.md §1.1)に
# コミット未反映の差分が残っていないかを検査する。
#
# 単体でも実行できる(手動実行・CIどちらも想定):
#   ./scripts/check_race_config.sh
#
# 設計方針: 「レース値」そのものをこのスクリプト内にハードコードしない
# (CLAUDE.md §0の「実測値・現在値はconfig.yamlを正とする」方針に従う)。
# 代わりに「git HEAD(直近コミット)からの差分が、レース値維持キーに
# 触れていないか」をチェックする——実験中に変更したまま復元し忘れている
# ケースを検出するのが目的であり、HEAD自体が必ず「レース値」だとは限らない
# (実験ブランチの途中コミットである可能性)点は運用上の限界として留意する。
#
# exit code: 0=問題なし(差分なし、または対象キー以外の差分のみ)
#            1=レース値維持キーに未コミットの差分あり(要確認)
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "[check_race_config] gitリポジトリ外で実行されました" >&2
  exit 2
fi
cd "$REPO_ROOT"

CONFIG_PATH="aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/config/config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "[check_race_config] config.yamlが見つかりません: $CONFIG_PATH" >&2
  exit 2
fi

# CLAUDE.md §1.1 レース値維持キー(トップレベルセクション名でgrepする、
# ネストしたキー名までは追わない簡易版——誤検知を避けるため広めに拾う)
RACE_KEY_SECTIONS=(
  "bicycle_model:" "mpc:" "v2x_obstacle_avoidance:" "lat_ttc:"
  "stuck_recovery:" "overtake:" "pit_lane:"
)

DIFF_OUTPUT="$(git diff -- "$CONFIG_PATH" 2>/dev/null)"

if [ -z "$DIFF_OUTPUT" ]; then
  echo "[check_race_config] OK: config.yamlに未コミットの差分はありません"
  exit 0
fi

echo "[check_race_config] config.yamlに未コミットの差分があります:"
echo "$DIFF_OUTPUT" | grep -E "^[+-]  " | head -50
echo ""
echo "[check_race_config] 変更されたトップレベルセクション:"
CHANGED_SECTIONS=()
for sec in "${RACE_KEY_SECTIONS[@]}"; do
  # diffの文脈行(セクション見出し)またはハンク内に該当セクション名が
  # 含まれるかを簡易チェック(厳密なYAMLパースは行わない、意図的に単純)
  if echo "$DIFF_OUTPUT" | grep -q "^[@ +-].*${sec}"; then
    CHANGED_SECTIONS+=("$sec")
    echo "  - $sec"
  fi
done

echo ""
echo "[check_race_config] mpc.debug_extra_actuator_delay_s / mpc.enable_diag_log の現在値(情報のみ、実験中は非デフォルトでも正常):"
grep -n "debug_extra_actuator_delay_s\|enable_diag_log" "$CONFIG_PATH" | sed 's/^/  /'

if [ ${#CHANGED_SECTIONS[@]} -gt 0 ]; then
  echo ""
  echo "[check_race_config] WARN: レース値維持キーを含むセクションに差分が残っています。"
  echo "  実験目的でまだ作業中なら問題ありません。復元を忘れている場合は"
  echo "  /restore-race-config を実行するか、'git diff -- $CONFIG_PATH' で確認してください。"
  exit 1
fi

echo ""
echo "[check_race_config] OK: 差分はありますが、レース値維持キーの対象セクション外のようです"
exit 0
