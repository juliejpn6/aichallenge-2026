#!/bin/bash

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

# 車両数: 第1引数（既定 1）
vehicles="${1:-1}"

# 2026-07-25: 実際の予選/決勝環境ではwall-recoveryがoffと判明(ユーザー確認)。
#   eval.sh(評価環境相当)も既にoffになっており、devのみon→offの不一致があった。
#   ローカルでの挙動が独自STUCK/PUSH復帰ロジック(149〜173節)とAWSIM組み込みの
#   自動修正(壁衝突時、速度>0.5m/sで180°/s・1秒間)を混同しないよう、実環境に揃える。
$AWSIM_DIRECTORY/AWSIM.x86_64 \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles "${vehicles}" \
    --npcs 0 \
    --boosts 2 \
    --laps unlimited \
    --timeout unlimited \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar off

# Cameraを使う場合 : --camera cpu or gpu
# LiDARを使う場合 : --lidar cpu or gpu
# GPUがない場合 -headlessを末尾に追加
