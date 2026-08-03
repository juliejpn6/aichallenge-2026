#!/usr/bin/env python3
"""analyze_actuator_delay.py

アクチュエータ遅延(むだ時間L+一次遅れtau)の実測(2026-08-03、Phase3/PartC-3-4)。
操舵指令と実測操舵角からFOPDT(First-Order Plus Dead Time)モデルのL・tauを
フィットし、速度条件間(v_max=15/20km/h等)で系統的な差があるかを検証する。
tau=190msは15km/h運用時代に確定した値(208節)で、20km/h環境での意図的な
再検証は行われていなかったため、この検証で速度依存性の有無を直接確認する。
制御には一切関与しない、オフライン分析専用ツール。

データソース:
  - 操舵指令: rosbagの/control/command/control_cmd
    (analyze_steering_psd.read_steering_seriesを再利用、CDR手動パース)
  - **[steeringモード]** 実測操舵角: rosbagの/vehicle/status/steering_status
    (autoware_vehicle_msgs/msg/SteeringReport。ワークスペース固有パッケージで
    ホストのrclpyに存在しないため、CDRを手動パースする。実地検証済みレイアウト:
    4byte encapsulation + stamp.sec(i32) + stamp.nanosec(u32) +
    steering_tire_angle(f32) = 16byte固定長)。**予選rosbagには含まれない
    制約が判明済み**(大会運営の録画設定)、ローカル実験でのみ使用可能。
  - **[yawrateモード、2026-08-03追加、PhaseC-0-2]** 実測ヨーレート応答:
    rosbagの/localization/kinematic_state(nav_msgs/msg/Odometry、標準型)の
    twist.twist.angular.z。**予選rosbagにも含まれるため、予選環境の実効遅延
    (指示→車両応答)を直接測定できる唯一の手段**。steeringモードとは別の量
    (操舵角そのものの遅れ ではなく 操舵→ヨーレート応答という車両ダイナミクス
    込みの「ループ全体の実効遅延」)であることに注意。

手法:
  1. **FOPDTグリッドサーチ**: むだ時間L・時定数tauの候補格子上で、指令列を
     FOPDTモデル(離散オイラー法)で応答予測し、実測値とのRMS残差が最小となる
     (L, tau)を探す。
  2. **エッジ法(クロスチェック)**: 指令の急激な変化イベントを自動検出し、
     各イベントについて「指令変化開始→実測値がノイズ床を有意に超えて動き出す
     までの時間」を測定し、その分布の中央値を参考のむだ時間推定値とする。

較正プロトコル(PhaseC-0-2): 予選走行のたびに本ツールをyawrateモードで適用し
L_eff_予選を記録する。ローカルで同条件(delay=0)のL_eff_ローカルを測り、
delta = L_eff_予選 - L_eff_ローカル を算出、
`debug_extra_actuator_delay_s = delta` を投入して較正する。

使い方:
    python3 analyze_actuator_delay.py <bag1>:<label1> [<bag2>:<label2> ...] [--mode steering|yawrate]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_steering_psd import read_steering_series  # noqa: E402 (操舵指令、既存関数を再利用)

_STEERING_REPORT_FMT = '<4x iIf'  # encapsulation + stamp.sec + stamp.nanosec + steering_tire_angle

# steeringモード用の既定値([deg]系)
DEFAULT_L_CANDIDATES = np.arange(0.0, 0.401, 0.01)
DEFAULT_TAU_CANDIDATES = np.arange(0.05, 0.401, 0.01)
EDGE_THRESHOLD_DEG = 2.0  # [deg] この量を超える指令変化を「エッジ」とみなす
EDGE_MIN_GAP_S = 0.5  # [s] エッジ同士の最小間隔(近すぎるものは同一イベントとみなし片方採用)
RESPONSE_NOISE_FLOOR_DEG = 0.3  # [deg] 実測角がこの量を超えて動いたら「応答開始」とみなす

# yawrateモード用の既定値([deg]の指令 vs [deg/s]のヨーレート、スケールが違うため別定義)
YAWRATE_NOISE_FLOOR_DEGPS = 3.0  # [deg/s] ヨーレートがこの量を超えて変化したら「応答開始」とみなす


def read_steering_status_series(bag_path):
    """rosbagから(log_time_epoch_s, steering_tire_angle_rad)のリストを時刻昇順で
    返す(/vehicle/status/steering_status)。analyze_steering_psd.read_steering_series
    と同じCDR手動パース方針(48byte固定長のAckermannControlCommandとは別レイアウト、
    16byte固定長)。**予選rosbagには含まれない**(大会運営の録画設定)。"""
    from mcap.reader import make_reader
    series = []
    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(
                topics=['/vehicle/status/steering_status']):
            vals = struct.unpack(_STEERING_REPORT_FMT, message.data)
            angle = vals[2]
            series.append((message.log_time * 1e-9, angle))
    series.sort(key=lambda p: p[0])
    return series


def read_yaw_rate_series(bag_path):
    """rosbagから(log_time_epoch_s, yaw_rate_rad_s)のリストを時刻昇順で返す
    (/localization/kinematic_state、nav_msgs/msg/Odometry、標準型のため
    rclpy.serialization.deserialize_messageで読める。analyze_steering_psd.
    read_speed_seriesと同じ購読対象・同じデコード方式)。**予選rosbagにも
    含まれるため、予選環境の実効遅延を直接測定できる。**"""
    import rclpy.serialization
    from nav_msgs.msg import Odometry
    from mcap.reader import make_reader
    series = []
    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(
                topics=['/localization/kinematic_state']):
            msg = rclpy.serialization.deserialize_message(message.data, Odometry)
            series.append((message.log_time * 1e-9, msg.twist.twist.angular.z))
    series.sort(key=lambda p: p[0])
    return series


def fopdt_predict(cmd_t, cmd_v, grid, L, tau):
    """指令列(cmd_t, cmd_v)を、むだ時間L+一次遅れtauのFOPDTモデルで応答予測する
    (離散化はオイラー法)。gridは応答を評価する等間隔サンプリング時刻列。"""
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 0.025
    u = np.interp(grid - L, cmd_t, cmd_v)
    y = np.zeros_like(u)
    alpha = min(dt / max(tau, 1e-6), 1.0)
    for k in range(1, len(u)):
        y[k] = y[k - 1] + alpha * (u[k - 1] - y[k - 1])
    return y


def fit_fopdt(steering_cmd, steering_act, sample_hz=40.0,
              L_candidates=None, tau_candidates=None):
    """グリッドサーチでL(むだ時間)・tau(時定数)を推定する。"""
    if L_candidates is None:
        L_candidates = DEFAULT_L_CANDIDATES
    if tau_candidates is None:
        tau_candidates = DEFAULT_TAU_CANDIDATES

    cmd_t = np.array([p[0] for p in steering_cmd])
    cmd_v = np.degrees(np.array([p[1] for p in steering_cmd]))
    act_t = np.array([p[0] for p in steering_act])
    act_v = np.degrees(np.array([p[1] for p in steering_act]))

    t_lo = max(cmd_t[0], act_t[0]) + max(L_candidates)
    t_hi = min(cmd_t[-1], act_t[-1])
    if t_hi <= t_lo:
        return None
    grid = np.arange(t_lo, t_hi, 1.0 / sample_hz)
    if len(grid) < 10:
        return None
    act_i = np.interp(grid, act_t, act_v)

    best = None
    for L in L_candidates:
        for tau in tau_candidates:
            y_pred = fopdt_predict(cmd_t, cmd_v, grid, L, tau)
            resid = float(np.sqrt(np.mean((y_pred - act_i) ** 2)))
            if best is None or resid < best[2]:
                best = (float(L), float(tau), resid)
    return {'L_s': best[0], 'tau_s': best[1], 'resid_deg': best[2], 'n_samples': len(grid)}


def detect_edges(cmd_t, cmd_v, threshold_deg=EDGE_THRESHOLD_DEG, min_gap_s=EDGE_MIN_GAP_S):
    """指令の急激な変化(1周期あたりthreshold_deg超)をエッジとして検出する。
    min_gap_s未満で連続するエッジは最初の1個のみ採用する。"""
    edges = []
    last_edge_t = -1e9
    for i in range(1, len(cmd_t)):
        d = abs(cmd_v[i] - cmd_v[i - 1])
        if d > threshold_deg and (cmd_t[i] - last_edge_t) > min_gap_s:
            edges.append(i)
            last_edge_t = cmd_t[i]
    return edges


def estimate_delay_by_edges(steering_cmd, steering_act,
                             threshold_deg=EDGE_THRESHOLD_DEG,
                             noise_floor_deg=RESPONSE_NOISE_FLOOR_DEG,
                             window_s=0.6):
    """エッジ法: 各エッジ(指令の急変時刻)から、実測角がnoise_floor_deg超の
    変化を見せる最初の時刻までの遅延を測定し、その分布(件数・中央値・p90)を
    返す。"""
    cmd_t = np.array([p[0] for p in steering_cmd])
    cmd_v = np.degrees(np.array([p[1] for p in steering_cmd]))
    act_t = np.array([p[0] for p in steering_act])
    act_v = np.degrees(np.array([p[1] for p in steering_act]))

    edges = detect_edges(cmd_t, cmd_v, threshold_deg)
    delays = []
    for idx in edges:
        t0 = cmd_t[idx]
        baseline_mask = (act_t >= t0 - 0.1) & (act_t < t0)
        if not np.any(baseline_mask):
            continue
        baseline = float(np.mean(act_v[baseline_mask]))
        window_mask = (act_t >= t0) & (act_t <= t0 + window_s)
        window_t = act_t[window_mask]
        window_v = act_v[window_mask]
        moved = np.where(np.abs(window_v - baseline) > noise_floor_deg)[0]
        if len(moved) > 0:
            delays.append(float(window_t[moved[0]] - t0))
    if not delays:
        return {'n_edges': len(edges), 'n_delays': 0}
    delays = np.array(delays)
    return {
        'n_edges': len(edges), 'n_delays': len(delays),
        'median_delay_s': float(np.median(delays)),
        'p90_delay_s': float(np.percentile(delays, 90)),
        'mean_delay_s': float(np.mean(delays)),
    }


def analyze_bag(bag_path, label, sample_hz=40.0, mode='steering'):
    """mode='steering': 操舵指令→実測操舵角(ローカルのみ、tau=190msと同じ量)。
    mode='yawrate': 操舵指令→実測ヨーレート(予選でも測定可、L_effは車両
    ダイナミクス込みのループ全体の実効遅延で操舵角ベースのL・tauとは別の量)。"""
    steering_cmd = read_steering_series(bag_path)
    if mode == 'yawrate':
        response = read_yaw_rate_series(bag_path)
        noise_floor = YAWRATE_NOISE_FLOOR_DEGPS
    else:
        response = read_steering_status_series(bag_path)
        noise_floor = RESPONSE_NOISE_FLOOR_DEG
    fopdt = fit_fopdt(steering_cmd, response, sample_hz=sample_hz)
    edges = estimate_delay_by_edges(steering_cmd, response, noise_floor_deg=noise_floor)
    return {'label': label, 'bag_path': bag_path, 'mode': mode, 'fopdt': fopdt, 'edges': edges,
            'n_cmd': len(steering_cmd), 'n_act': len(response)}


def format_report(results, default_tau_s=0.19):
    lines = []
    mode = results[0]['mode'] if results else 'steering'
    title = ('アクチュエータ遅延(FOPDT L・tau、操舵角ベース)' if mode == 'steering'
             else 'ループ全体の実効遅延(FOPDT L_eff、ヨーレートベース)')
    lines.append(f'# {title} 実測レポート')
    lines.append('')
    if mode == 'yawrate':
        lines.append('**注意**: これは操舵角そのものの遅れ(tau)ではなく、操舵指令→ヨーレート応答'
                     'という車両ダイナミクス込みの「ループ全体の実効遅延」。'
                     'steeringモードのL・tauとは別の量として扱うこと。')
        lines.append('')
    lines.append(f'既定tau(config非依存デフォルト): {default_tau_s * 1000:.0f}ms')
    lines.append('')
    lines.append('| label | n_cmd | n_act | FOPDT L[ms] | FOPDT tau/tau_eff[ms] | resid | edge中央値[ms] | edge件数 |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for r in results:
        f = r['fopdt']
        e = r['edges']
        if f is None:
            l_str = tau_str = resid_str = 'N/A(データ不足)'
        else:
            l_str = f"{f['L_s']*1000:.0f}"
            tau_str = f"{f['tau_s']*1000:.0f}"
            resid_str = f"{f['resid_deg']:.3f}"
        edge_str = f"{e['median_delay_s']*1000:.0f}" if e.get('n_delays') else 'N/A'
        lines.append(f"| {r['label']} | {r['n_cmd']} | {r['n_act']} | {l_str} | {tau_str} | "
                     f"{resid_str} | {edge_str} | {e['n_edges']}(有効{e.get('n_delays', 0)}) |")
    lines.append('')
    if mode == 'steering':
        lines.append(f'既定tau={default_tau_s*1000:.0f}msからの乖離(FOPDT tau基準)が±15%を超える'
                     'ログがあれば、速度依存性の候補として要検証。')
    else:
        lines.append('較正プロトコル: delta = L_eff_予選 - L_eff_ローカル を算出し、'
                     'debug_extra_actuator_delay_s = delta を投入して較正する(PhaseC-0-2)。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='アクチュエータ遅延/実効遅延の実測')
    parser.add_argument('specs', nargs='+', help='bag:label の形式で複数指定')
    parser.add_argument('--sample-hz', type=float, default=40.0)
    parser.add_argument('--default-tau-s', type=float, default=0.19)
    parser.add_argument('--mode', choices=['steering', 'yawrate'], default='steering',
                         help='steering: 操舵角ベース(ローカルのみ)。'
                              'yawrate: ヨーレートベース(予選でも測定可、較正用)')
    args = parser.parse_args(argv)

    results = []
    for spec in args.specs:
        bag_path, label = spec.split(':', 1)
        print(f"分析中: {label} ({bag_path})", file=sys.stderr)
        results.append(analyze_bag(bag_path, label, sample_hz=args.sample_hz, mode=args.mode))

    print(format_report(results, default_tau_s=args.default_tau_s))
    return 0


if __name__ == '__main__':
    sys.exit(main())
