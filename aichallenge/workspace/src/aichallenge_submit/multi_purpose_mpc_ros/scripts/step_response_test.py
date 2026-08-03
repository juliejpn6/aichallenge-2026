#!/usr/bin/env python3
"""step_response_test.py

ステアリング・ステップ応答テストツール(2026-08-03、初日測定キット)。
AWSIMアップデート(steer rate limit 0.8->0.6等)適用後、steer_rate_maxの
実測較正に使う「意図的なステップ操舵入力→実舵応答」を能動的に生成する。

既存のanalyze_actuator_delay.pyは走行中の自然な指令変化を受動的に観測する
手法だが、本ツールは既知の振幅・タイミングのステップ列を直接publishすること
で、エッジ法・FOPDTフィットのサンプル数と信頼性を確保する。

**重要**: 本ツールは/control/command/control_cmdへ直接publishするため、
実行前にmpc_controllerノードを停止すること(2つのpublisherが競合し、
どちらの指令が実際にAWSIMへ届くか不定になるため)。制御ロジック
(mpc_controller.py等)には一切変更を加えない、独立した検証専用ノード。

手順:
  1. `make dev`(または dev3 等)でAWSIM+bag-recorderを起動する
  2. mpc_controllerノードのみ停止する(例:
     `docker exec 1-autoware-1 bash -c "pkill -f mpc_controller"`)
  3. 本ツールをコンテナ内で実行する:
     `python3 step_response_test.py --speed 2.0 --step-deg 15.0`
  4. 完了後、bag-recorderを停止してrosbagを回収する
  5. `analyze_actuator_delay.py --mode steering`(ローカル、実測操舵角あり)
     または `--mode yawrate`(予選同等条件の検証)で解析する

ステップパターン: 0° -> +step -> 0 -> -step -> 0 を1サイクルとし、
n_cycles回繰り返す(正負両方向のエッジ・FOPDTサンプルを確保するため)。
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from autoware_auto_control_msgs.msg import AckermannControlCommand


class StepResponseTestNode(Node):
    def __init__(self, speed_mps, step_deg, step_period_s, n_cycles, publish_rate_hz):
        super().__init__('step_response_test_node')
        self._pub = self.create_publisher(
            AckermannControlCommand, '/control/command/control_cmd', 1)
        self._speed_mps = speed_mps
        self._step_rad = step_deg * 3.141592653589793 / 180.0
        self._step_period_s = step_period_s
        self._n_cycles = n_cycles
        self._publish_rate_hz = publish_rate_hz
        self._phases = self._build_phase_schedule()
        self._start_time = None
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)
        self.get_logger().info(
            f"[STEP-TEST] speed={speed_mps}m/s step={step_deg}deg "
            f"period={step_period_s}s cycles={n_cycles} rate={publish_rate_hz}Hz "
            f"total_duration={len(self._phases) * step_period_s:.1f}s")

    def _build_phase_schedule(self):
        """1サイクル=[0, +step, 0, -step]の4フェーズ。n_cycles回連結する。"""
        one_cycle = [0.0, self._step_rad, 0.0, -self._step_rad]
        return one_cycle * self._n_cycles

    def _on_timer(self):
        now = self.get_clock().now()
        if self._start_time is None:
            self._start_time = now
        elapsed_s = (now - self._start_time).nanoseconds / 1e9
        phase_idx = int(elapsed_s // self._step_period_s)
        if phase_idx >= len(self._phases):
            self.get_logger().info("[STEP-TEST] 完了、ノードを終了します")
            self._timer.cancel()
            rclpy.shutdown()
            return
        target_angle = self._phases[phase_idx]
        msg = AckermannControlCommand()
        msg.stamp = now.to_msg()
        msg.lateral.stamp = now.to_msg()
        msg.lateral.steering_tire_angle = target_angle
        msg.lateral.steering_tire_rotation_rate = 2.0
        msg.longitudinal.stamp = now.to_msg()
        msg.longitudinal.speed = self._speed_mps
        msg.longitudinal.acceleration = 0.0
        self._pub.publish(msg)


def main(argv=None):
    parser = argparse.ArgumentParser(description='ステアリング・ステップ応答テストツール')
    parser.add_argument('--speed', type=float, default=2.0, help='走行速度[m/s](既定2.0)')
    parser.add_argument('--step-deg', type=float, default=15.0, help='ステップ振幅[deg](既定15.0)')
    parser.add_argument('--period', type=float, default=3.0, help='各フェーズの持続時間[s](既定3.0)')
    parser.add_argument('--cycles', type=int, default=5, help='サイクル数(既定5、正負各5回分)')
    parser.add_argument('--rate', type=float, default=40.0, help='publish周波数[Hz](既定40.0)')
    args = parser.parse_args(argv)

    rclpy.init()
    node = StepResponseTestNode(
        speed_mps=args.speed, step_deg=args.step_deg, step_period_s=args.period,
        n_cycles=args.cycles, publish_rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
