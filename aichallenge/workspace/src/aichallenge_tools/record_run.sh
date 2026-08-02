#!/usr/bin/env bash
# =============================================================
# record_run.sh
#   制御周期改善後の単独走行 rosbag を収集する（コンテナ内で実行）。
#   保存先 /aichallenge/workspace/bag/ はマウント領域なので
#   ホスト側 ~/aichallenge-racingkart/aichallenge/workspace/bag/ から analyze_bag.py で読める。
#   mcap 形式・/clock 併用・trajectory(BEST_EFFORT) は QoS 上書きで取りこぼし防止。
#
#   使い方（make dev で走行中に、別シェルでコンテナに入って実行）:
#     docker ps                       # コンテナ名を確認
#     docker exec -it <container> bash
#     bash /aichallenge/workspace/bag/record_run.sh [タグ]
#   → 1周走らせたら Ctrl-C で停止。
# =============================================================
set -eo pipefail

# ROS の setup.bash は未定義変数を参照するため、source 中だけ -u を無効化する
set +u
source /opt/ros/humble/setup.bash
source /aichallenge/workspace/install/setup.bash
set -u

BAG_DIR=/aichallenge/workspace/bag
mkdir -p "$BAG_DIR"
cd "$BAG_DIR"

# trajectory は BEST_EFFORT。録画側 subscription を publisher に合わせる。
cat > qos_override.yaml << 'EOF'
/planning/scenario_planning/trajectory:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 10
EOF

TAG="${1:-perffix}"
RUN_NAME="run_${TAG}_$(date +%Y%m%d_%H%M%S)"
echo "録画開始: ${BAG_DIR}/${RUN_NAME}  (Ctrl-C で停止)"

# ウェイポイント追従不良(操舵飽和/アンダー)を顕在化するため、操舵チェーンの各段を録る。
#  - control_cmd     : MPC目標操舵角(gain適用後) / 目標加速度
#  - control_cmd_raw : MPC生出力(gain適用前)
#  - actuation_cmd   : アクチュエータ最終指示(steer_cmd/accel/brake)
#  - actuation_status: アクチュエータ現在値(steer_status 等)
#  - steering_status : 実操舵角（指令に追従できているかの比較対象）
#  - imu_raw         : ヨーレート/横G（操舵に対し車両が回頭しているか＝アンダー判定）
#  - mpc/prediction  : MPC予測軌道（入口でMPCが描く軌道が膨らむ予測か曲がれる予測かを確認）
#  - mpc/ref_path    : MPC内部の参照経路（予測軌道との乖離＝先読み/求解の問題切り分け）
#  - ground_truth/*  : 真値の位置と衝突フラグ（localization誤差の切り分け・衝突地点特定）
# 真値(ground_truth)は提出コードでは使えないが、分析用ログとしては利用できる。
# カメラ・点群は除外。
# 混走(v2x)・追い越し診断(gate2)は分析に必須のため記録対象に含める:
#  - v2x/vehicle_positions : 他車の実位置（左右配置・検知有無の確認）
#  - mpc/overtake_status   : 追い越しFSM状態/選択側/left_free,right_free（STOPPING誤判定の切り分け）
# 2026-07-23追加(166節、GHOST-BLOCK breakthrough続報): EKFへの入力を録画してbagリプレイでの
#  センサフュージョン検証(proc_stddev_wz_c等)を可能にする。/localization/imu_gnss_poser/
#  pose_with_covariance(EKFのpose入力)は既存録画対象。以下2つを追加:
#  - localization/twist_estimator/twist_with_covariance : EKFのtwist入力(gyro_odometer出力)
#  - sensing/imu/imu_data : imu_corrector後の補正済みIMU(gyro_odometerの入力そのもの)
ros2 bag record -o "$RUN_NAME" \
  --storage mcap \
  --qos-profile-overrides-path qos_override.yaml \
  /clock \
  /control/command/control_cmd \
  /control/command/control_cmd_raw \
  /control/command/actuation_cmd \
  /vehicle/status/actuation_status \
  /vehicle/status/steering_status \
  /vehicle/status/velocity_status \
  /localization/kinematic_state \
  /localization/pose_with_covariance \
  /localization/imu_gnss_poser/pose_with_covariance \
  /localization/twist_estimator/twist_with_covariance \
  /localization/initial_pose3d \
  /sensing/gnss/pose_with_covariance \
  /sensing/imu/imu_raw \
  /sensing/imu/imu_data \
  /mpc/prediction \
  /mpc/ref_path \
  /mpc/overtake_status \
  /mpc/opponent_speed_map \
  /v2x/vehicle_positions \
  /planning/scenario_planning/trajectory \
  /awsim/ground_truth/localization/kinematic_state \
  /awsim/ground_truth/on_collision \
  /awsim/status \
  /aichallenge/pitstop/condition \
  /tf /tf_static
