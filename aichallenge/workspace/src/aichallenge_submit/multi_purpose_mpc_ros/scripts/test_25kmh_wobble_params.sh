#!/bin/bash
# test_25kmh_wobble_params.sh
#
# 25km/h蛇行対策の候補パラメータをdev3走行中にライブ切替するヘルパー
# (2026-08-07)。config.yamlは一切変更せず、`ros2 param set`のみで
# /mpc_controllerへ投入する(走行終了=プロセス終了で自然に元へ戻る)。
#
# 候補パラメータ(いずれもベースライン: Q=[200000,1000000,200000,0.0],
# wp_id_offset=1, steer_low_pass_gain=0.35 からの差分):
#   cand1(コストバランス最適化): Q[0]=100000(QN不変)
#   cand2(先読み延長):           wp_id_offset=3
#   cand3(操作量平滑化の強化):    steer_low_pass_gain=0.25
#     ※ CLAUDE.md §3禁止リスト5番(確信度「高」): steer_low_pass_gainは
#       0.35から動かさないことが既存速度域(15-20km/h)で確定している。
#       25km/hでの再検証という位置づけであり、実行前に必ず人間の確認を
#       取ること。このスクリプトは値を投入するだけで、可否判断はしない。
#
# 使い方:
#   ./test_25kmh_wobble_params.sh <container> <mode>
#     container: docker container名 (例: 1-autoware-1, 2-autoware-1)
#     mode:      base | cand1 | cand2 | cand3
#
#   例:
#     ./test_25kmh_wobble_params.sh 2-autoware-1 cand1
#     ./test_25kmh_wobble_params.sh 3-autoware-1 base    # ベースラインへ復元
#
# オプション:
#   --node <name>   対象ROSノード名(既定: /mpc_controller)
#
set -euo pipefail

NODE="/mpc_controller"

# --node オプションの取り出し(位置引数の前後どちらでも受け付ける)
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)
            NODE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

if [[ $# -lt 2 ]]; then
    echo "使い方: $0 [--node <ros_node_name>] <container> <base|cand1|cand2|cand3>" >&2
    echo "例:     $0 2-autoware-1 cand1" >&2
    exit 1
fi

CONTAINER="$1"
MODE="$2"

# ベースライン値(全モード共通、変更対象外パラメータを必ずここへ戻す)
BASE_Q0=200000.0
BASE_QN0=1000000.0
BASE_WP_ID_OFFSET=1
BASE_STEER_LOW_PASS_GAIN=0.35

ROS_SRC="source /autoware/install/setup.bash 2>/dev/null; source /aichallenge/workspace/install/setup.bash 2>/dev/null"

set_param() {
    local name="$1" value="$2"
    echo "  -> ${name} = ${value}"
    docker exec "${CONTAINER}" bash -c "${ROS_SRC}; ros2 param set ${NODE} ${name} ${value}"
}

echo "=================================================================="
echo " container=${CONTAINER}  node=${NODE}  mode=${MODE}"
echo "=================================================================="

case "${MODE}" in
    base)
        echo "[BASE] 全パラメータをベースラインへ復元"
        set_param Q0 "${BASE_Q0}"
        set_param QN0 "${BASE_QN0}"
        set_param wp_id_offset "${BASE_WP_ID_OFFSET}"
        set_param steer_low_pass_gain "${BASE_STEER_LOW_PASS_GAIN}"
        ;;
    cand1)
        echo "[CAND1] コストバランス最適化: Q[0] 200000 -> 100000(QN不変)"
        set_param Q0 100000.0
        set_param QN0 "${BASE_QN0}"
        set_param wp_id_offset "${BASE_WP_ID_OFFSET}"
        set_param steer_low_pass_gain "${BASE_STEER_LOW_PASS_GAIN}"
        ;;
    cand2)
        echo "[CAND2] 先読み延長: wp_id_offset 1 -> 3"
        set_param Q0 "${BASE_Q0}"
        set_param QN0 "${BASE_QN0}"
        set_param wp_id_offset 3
        set_param steer_low_pass_gain "${BASE_STEER_LOW_PASS_GAIN}"
        ;;
    cand3)
        echo "[CAND3] 操作量平滑化の強化: steer_low_pass_gain 0.35 -> 0.25"
        echo "  !!! 注意: CLAUDE.md §3禁止リスト5番(確信度「高」)に該当するパラメータです。"
        echo "  !!! 既存速度域(15-20km/h)では0.2/0.5どちらも全指標悪化・0.35が局所最適と確定済み。"
        echo "  !!! 25km/hでの再検証という位置づけです。実行前に人間の承認を確認してください。"
        set_param Q0 "${BASE_Q0}"
        set_param QN0 "${BASE_QN0}"
        set_param wp_id_offset "${BASE_WP_ID_OFFSET}"
        set_param steer_low_pass_gain 0.25
        ;;
    *)
        echo "不明なmode: ${MODE}(base|cand1|cand2|cand3のいずれかを指定)" >&2
        exit 1
        ;;
esac

echo "=================================================================="
echo " 投入完了。現在値を確認します。"
echo "=================================================================="
docker exec "${CONTAINER}" bash -c "${ROS_SRC}; \
    echo -n 'Q0: '; ros2 param get ${NODE} Q0; \
    echo -n 'QN0: '; ros2 param get ${NODE} QN0; \
    echo -n 'wp_id_offset: '; ros2 param get ${NODE} wp_id_offset; \
    echo -n 'steer_low_pass_gain: '; ros2 param get ${NODE} steer_low_pass_gain"
