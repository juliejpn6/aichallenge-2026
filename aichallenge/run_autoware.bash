#!/bin/bash

mode="${1}"
id="${2:-${ROS_DOMAIN_ID:-0}}"
out_dir="${3:+${3}/d${id}}"
out_dir="${out_dir:-/output/$(date +%Y%m%d-%H%M%S)/d${id}}"

case "${mode}" in
"awsim")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true")
    ;;
"awsim-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false")
    ;;
"vehicle")
    opts=("simulation:=false" "use_sim_time:=false" "run_rviz:=false")
    ;;
"rosbag")
    opts=("simulation:=false" "use_sim_time:=true" "run_rviz:=true")
    ;;
*)
    echo "invalid argument (use 'awsim' or 'vehicle' or 'rosbag')"
    exit 1
    ;;
esac

export ROS_DOMAIN_ID=$id

mkdir -p "${out_dir}"
exec >"${out_dir}/autoware.log" 2>&1

cd "${out_dir}" || exit
# Persist ROS node logs under the run output directory (so autostart_orchestrator logs are collectible).
export ROS_HOME="${out_dir}/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "${ROS_LOG_DIR}"

# 2026-08-01追加(262節続報、判定基準改訂+work_cpu計装Phase 4、C4実験「双方向
#   アフィニティ」用): mpc_controllerは自分自身をos.sched_setaffinity(config.yamlの
#   mpc.cpu_affinity経由)で専有コアへ固定できるが、C3実測ではmpc_controller
#   *以外*の同居ノード(rviz2・ekf_localizer・autostart_orchestrator等)が
#   その専有コアへ自由に侵入できてしまう「一方通行」の隔離だった。これらの
#   ノードのソースコードには一切手を加えず、`ros2 launch`親プロセス自体を
#   taskset(AUTOWARE_OTHER_NODES_CPU_AFFINITY、既定未設定=従来通り無制限)で
#   起動することで対処する——taskset無しで起動された子プロセスは通常CPU
#   アフィニティを親から継承するため、mpc_controllerだけが後から自分の
#   アフィニティをsched_setaffinityで上書きし、他の全ノードは親のtaskset
#   範囲(mpc専有コアの補集合)に留まる、という双方向の隔離が実現する。
# 2026-08-05マージ(origin/main #236): set -m + trap + waitによるSIGINT/TERM
#   転送(終了処理改善)をtaskset分岐の両方に適用し、上記のCPUアフィニティ機構と
#   両立させた(set -mはバックグラウンド化した子へSIGINTがSIG_IGNされるのを防ぐ)。
set -m
if [ -n "${AUTOWARE_OTHER_NODES_CPU_AFFINITY:-}" ]; then
    taskset -c "${AUTOWARE_OTHER_NODES_CPU_AFFINITY}" \
        ros2 launch aichallenge_system_launch aichallenge_system.launch.xml "${opts[@]}" "domain_id:=$id" &
else
    ros2 launch aichallenge_system_launch aichallenge_system.launch.xml "${opts[@]}" "domain_id:=$id" &
fi
trap 'kill -INT $! 2>/dev/null' TERM INT
while kill -0 $! 2>/dev/null; do wait; done
