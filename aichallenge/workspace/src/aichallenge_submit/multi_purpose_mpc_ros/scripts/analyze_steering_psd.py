#!/usr/bin/env python3
"""analyze_steering_psd.py

263節続報(2026-08-02、蛇行/性能ギャップ分析Part B): OT状態(NORMAL/
STOPPING/OVERTAKING)別に操舵角のパワースペクトル密度(PSD)を比較し、
対戦車対応中の修正舵角増加(+43〜69%、263節本編のSTEER-XCORR分析)が
「既知の限界サイクル(0.6〜0.7Hz)の励起」なのか「新目標(offset変動)への
正当な過渡応答」なのかを周波数領域で判別する。制御には一切関与しない、
オフライン分析専用ツール。対策(offsetレート制限等)・フェイルセーフの
実装はこのツールのスコープ外——判別結果を人間が確認してから別途設計する。

データソース:
  - 操舵角: rosbagの/control/command/control_cmd
    (autoware_auto_control_msgs/msg/AckermannControlCommand.lateral.
    steering_tire_angle)。この型はワークスペース固有のカスタムメッセージで
    ホストのrclpyには存在しないため、CDR(Common Data Representation)を
    手動でパースする(実地検証済み: 48byte固定長、全フィールドint32/
    uint32/float32で4byte整列のため、struct.unpackで十分——rosbag2_pyや
    mcap_ros2のスキーマ自動デコード[ros2idlエンコーディング]は非対応
    だったため、この回避策を取った)。
  - 速度: rosbagの/localization/kinematic_state(nav_msgs/msg/Odometry、
    標準メッセージのためrclpy.serialization.deserialize_messageで読める)。
  - OT状態: autoware.logの[OT] state=...行(ROS時刻付き)。

時刻同期: rosbagメッセージのmessage.log_time(mcap記録時刻)は、
autoware.logのタイムスタンプと同じエポック秒スケールであることを実測で
確認して同期に用いる(AckermannControlCommandのメッセージ内部stamp
フィールドは小さい相対値[かつ非単調な箇所あり]でありOT状態ログとの
同期には使えないことも実地確認済み)。

使い方:
    python3 analyze_steering_psd.py <autoware.log> <rosbag.mcap> [--out-dir DIR]
"""
import argparse
import bisect
import re
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import signal

# AckermannControlCommandのCDRレイアウト(実地検証済み、48byte固定長):
#   header(4byte encapsulation) +
#   stamp(sec:i,nanosec:I) + lateral.stamp(sec:i,nanosec:I) +
#   lateral.steering_tire_angle(f) + lateral.steering_tire_rotation_rate(f) +
#   longitudinal.stamp(sec:i,nanosec:I) + longitudinal.speed(f) +
#   longitudinal.acceleration(f) + longitudinal.jerk(f)
_ACKERMANN_FMT = '<4x iIiIff iIfff'

MIN_SEGMENT_S = 8.0  # 限界サイクル周期(0.6-0.7Hz≈1.4-1.7s)の5倍相当の最低長
LIMIT_CYCLE_BAND = (0.5, 0.9)  # Hz、既知の限界サイクル0.6-0.7Hzを包含
# 過渡応答/目標変更由来の低周波帯パワーは、当初PSDの0.02-0.1Hz帯域として
# 定義する予定だったが、実測でこのアプローチが機能しないことが判明した
# (scipy.signal.welchの既定detrend='constant'がセグメント平均[DC成分]を
# 除去する仕様と、MIN_SEGMENT_Sに紐づく周波数分解能[df≈0.125Hz]が
# 帯域幅0.08Hzより粗いことが重なり、常に0.000000になる)。代わりに
# analyze()内で生の操舵角の標準偏差(std_rad、周波数非依存の総変動幅)を
# 参考指標として使う——詳細はanalyze()内のコメント参照。
OT_STATES = ('NORMAL', 'STOPPING', 'OVERTAKING')


def read_steering_series(bag_path):
    """rosbagから(log_time_epoch_s, steering_tire_angle_rad)のリストを
    時刻昇順で返す。"""
    from mcap.reader import make_reader
    series = []
    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(
                topics=['/control/command/control_cmd']):
            vals = struct.unpack(_ACKERMANN_FMT, message.data)
            angle = vals[4]
            series.append((message.log_time * 1e-9, angle))
    series.sort(key=lambda p: p[0])
    return series


def read_speed_series(bag_path):
    """rosbagから(log_time_epoch_s, speed_mps)のリストを時刻昇順で返す。"""
    import rclpy.serialization
    from nav_msgs.msg import Odometry
    from mcap.reader import make_reader
    series = []
    with open(bag_path, 'rb') as f:
        reader = make_reader(f)
        for _schema, _channel, message in reader.iter_messages(
                topics=['/localization/kinematic_state']):
            msg = rclpy.serialization.deserialize_message(message.data, Odometry)
            series.append((message.log_time * 1e-9, msg.twist.twist.linear.x))
    series.sort(key=lambda p: p[0])
    return series


def read_ot_state_series(log_path):
    """autoware.logから(epoch_timestamp, state)のリストを時刻昇順で返す
    (OT状態遷移。同一状態への重複遷移は保持したままでよい——状態割り当て側は
    直近の遷移だけを使うため)。"""
    series = []
    pattern = re.compile(r'\[(\d{10}\.\d+)\].*\[OT\] state=(\w+)')
    with open(log_path, errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                series.append((float(m.group(1)), m.group(2)))
    series.sort(key=lambda p: p[0])
    return series


def assign_states(sample_times, ot_series):
    """各サンプル時刻に「その時刻以前の直近のOT状態」を割り当てる。
    ot_seriesより前(最初の状態遷移より前)のサンプルはNoneとする。"""
    ot_times = [t for t, _ in ot_series]
    ot_states = [s for _, s in ot_series]
    result = []
    for t in sample_times:
        idx = bisect.bisect_right(ot_times, t) - 1
        result.append(ot_states[idx] if idx >= 0 else None)
    return result


def segment_by_state(series, states):
    """(time, value)の系列とサンプルごとの状態ラベルから、状態が連続する
    区間(セグメント)のリスト[(state, [(t, v), ...]), ...]を返す。"""
    segments = []
    cur_state = None
    cur_points = []
    for (t, v), s in zip(series, states):
        if s != cur_state:
            if cur_points:
                segments.append((cur_state, cur_points))
            cur_state = s
            cur_points = []
        cur_points.append((t, v))
    if cur_points:
        segments.append((cur_state, cur_points))
    return segments


def filter_min_length(segments, min_length_s=MIN_SEGMENT_S):
    """最低セグメント長未満を除外し、(採用セグメント, 除外件数, 除外合計秒数)
    を返す。"""
    kept = []
    excluded_count = 0
    excluded_duration = 0.0
    for state, pts in segments:
        duration = pts[-1][0] - pts[0][0]
        if duration >= min_length_s:
            kept.append((state, pts))
        else:
            excluded_count += 1
            excluded_duration += duration
    return kept, excluded_count, excluded_duration


def resample_uniform(pts, sample_hz):
    """不等間隔の(t, v)点列を、開始時刻起点のsample_hz等間隔グリッドへ
    線形補間する。"""
    times = np.array([p[0] for p in pts])
    vals = np.array([p[1] for p in pts])
    t0, t1 = times[0], times[-1]
    n = int((t1 - t0) * sample_hz)
    if n < 2:
        return None, None
    grid = t0 + np.arange(n) / sample_hz
    resampled = np.interp(grid, times, vals)
    return grid, resampled


def welch_psd(values, sample_hz, nperseg):
    freqs, pxx = signal.welch(values, fs=sample_hz, nperseg=min(nperseg, len(values)))
    return freqs, pxx


def band_power(freqs, pxx, band):
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(pxx[mask], freqs[mask]))


def analyze(log_path, bag_path, sample_hz=40.0, min_segment_s=MIN_SEGMENT_S):
    steering = read_steering_series(bag_path)
    speed = read_speed_series(bag_path)
    ot_series = read_ot_state_series(log_path)

    steering_states = assign_states([t for t, _ in steering], ot_series)
    speed_states = assign_states([t for t, _ in speed], ot_series)

    raw_segments = segment_by_state(steering, steering_states)
    segments, excluded_count, excluded_duration = filter_min_length(
        raw_segments, min_segment_s)

    nperseg = int(min_segment_s * sample_hz)
    per_state = {}
    for state in OT_STATES:
        state_segs = [pts for s, pts in segments if s == state]
        psds = []
        total_duration = 0.0
        all_values = []
        for pts in state_segs:
            grid, resampled = resample_uniform(pts, sample_hz)
            if grid is None:
                continue
            freqs, pxx = welch_psd(resampled, sample_hz, nperseg)
            psds.append(pxx)
            total_duration += pts[-1][0] - pts[0][0]
            all_values.append(resampled)
        if not psds:
            per_state[state] = None
            continue
        freqs_ref, _ = welch_psd(np.zeros(nperseg), sample_hz, nperseg)
        avg_pxx = np.mean(np.stack(psds), axis=0)
        lc_power = band_power(freqs_ref, avg_pxx, LIMIT_CYCLE_BAND)
        total_power = float(np.trapezoid(avg_pxx, freqs_ref))
        # 2026-08-02追加: LOW_FREQ_BAND(0.02-0.1Hz)でのPSD帯域パワーは、
        # scipy.signal.welchの既定detrend='constant'がセグメント平均(DC成分)を
        # 除去する仕様と、このサンプル長でのdf=sample_hz/nperseg(≈0.125Hz)が
        # 帯域幅0.08Hzより粗いことが重なり、常に0.000000になる(実測で確認、
        # PSDでは検出不能)。代わりに生の操舵角の標準偏差(周波数非依存の
        # 総変動幅、DC成分[平均オフセットのずれ]も含む)を「低周波/過渡応答」の
        # 参考指標として用いる。
        concatenated = np.concatenate(all_values) if all_values else np.array([])
        std_rad = float(np.std(concatenated)) if concatenated.size else None
        speed_vals = [v for (t, v), s in zip(speed, speed_states) if s == state]
        avg_speed = float(np.mean(speed_vals)) if speed_vals else None
        per_state[state] = {
            'freqs': freqs_ref, 'pxx': avg_pxx,
            'limit_cycle_power': lc_power,
            'total_power': total_power, 'num_segments': len(psds),
            'total_duration_s': total_duration, 'avg_speed_mps': avg_speed,
            'std_rad': std_rad,
        }

    return {
        'per_state': per_state,
        'excluded_count': excluded_count,
        'excluded_duration_s': excluded_duration,
        'nperseg': nperseg,
        'sample_hz': sample_hz,
    }


def discriminate(per_state):
    """NORMALを基準に、非NORMAL状態の限界サイクル帯パワー比・低周波帯パワー比
    から「励起仮説」「過渡応答」「判別不能」を報告する。対策の実装はしない、
    判別結果の文字列を返すのみ。"""
    normal = per_state.get('NORMAL')
    if normal is None or normal['limit_cycle_power'] <= 0:
        return {'verdict': '判別不能(NORMALのデータ不足)', 'ratios': {}}
    ratios = {}
    for state in ('STOPPING', 'OVERTAKING'):
        s = per_state.get(state)
        if s is None:
            ratios[state] = None
            continue
        lc_ratio = s['limit_cycle_power'] / normal['limit_cycle_power']
        std_ratio = (s['std_rad'] / normal['std_rad']
                     if normal['std_rad'] and normal['std_rad'] > 0 else None)
        ratios[state] = {'limit_cycle_ratio': lc_ratio, 'std_ratio': std_ratio}
    valid_lc_ratios = [r['limit_cycle_ratio'] for r in ratios.values() if r]
    if not valid_lc_ratios:
        verdict = '判別不能(STOPPING/OVERTAKINGのデータ不足)'
    elif all(r >= 2.0 for r in valid_lc_ratios):
        verdict = '励起仮説を支持(限界サイクル帯パワー比が2倍超)'
    elif all(r < 1.3 for r in valid_lc_ratios):
        verdict = '過渡応答仮説を支持(限界サイクル帯パワー比はNORMALと同水準)'
    else:
        verdict = '中間(明確な判別不能、両仮説とも部分的にしか支持されない)'
    return {'verdict': verdict, 'ratios': ratios}


def plot_psd(per_state, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 既定フォント(DejaVu Sans)は日本語グリフを持たず豆腐化するため、
    # 実行環境にNoto Sans CJK JPが入っていればそれを使う(無ければ既定のまま
    # =英語ラベルだけ読めて日本語部分が豆腐化する、実害はないため無理に
    # フォールバックしない)。
    ja_fonts = [f.name for f in fm.fontManager.ttflist if 'Noto Sans CJK JP' in f.name]
    if ja_fonts:
        matplotlib.rcParams['font.family'] = ja_fonts[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'NORMAL': 'tab:blue', 'STOPPING': 'tab:orange', 'OVERTAKING': 'tab:red'}
    for state in OT_STATES:
        s = per_state.get(state)
        if s is None:
            continue
        ax.semilogy(s['freqs'], s['pxx'], label=f"{state} (n={s['num_segments']}セグメント)",
                    color=colors.get(state))
    ax.axvspan(*LIMIT_CYCLE_BAND, color='gray', alpha=0.2, label='限界サイクル帯(0.5-0.9Hz)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [rad^2/Hz]')
    ax.set_xlim(0, 2.0)
    ax.set_title('操舵角PSD: OT状態別比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_report(result, verdict_info, log_path, bag_path):
    lines = []
    lines.append(f"# 蛇行PSD分析: {log_path} / {bag_path}")
    lines.append('')
    lines.append(f"サンプリング: {result['sample_hz']:.0f}Hzへ線形補間で等間隔化 "
                 f"(nperseg={result['nperseg']})")
    lines.append(f"最低セグメント長({MIN_SEGMENT_S}s)未満で除外: "
                 f"{result['excluded_count']}件 "
                 f"(合計{result['excluded_duration_s']:.1f}s)")
    lines.append('')
    lines.append('## 状態別バンドパワー')
    for state in OT_STATES:
        s = result['per_state'].get(state)
        if s is None:
            lines.append(f"{state}: データ不足(セグメント無し)")
            continue
        speed_str = (f"{s['avg_speed_mps']:.2f}m/s" if s['avg_speed_mps'] is not None
                     else 'N/A')
        std_str = f"{s['std_rad']:.4f}rad" if s['std_rad'] is not None else 'N/A'
        lines.append(
            f"{state}: セグメント数={s['num_segments']} "
            f"合計{s['total_duration_s']:.1f}s 平均速度={speed_str} "
            f"限界サイクル帯(0.5-0.9Hz)パワー={s['limit_cycle_power']:.6f} "
            f"操舵角std(周波数非依存の参考指標)={std_str} "
            f"全帯域パワー={s['total_power']:.6f}")
    lines.append('')
    lines.append('## 対NORMAL比')
    for state, r in verdict_info['ratios'].items():
        if r is None:
            lines.append(f"{state}: N/A")
        else:
            lines.append(f"{state}: 限界サイクル帯比={r['limit_cycle_ratio']:.2f} "
                         f"std比(参考)={_fmt_ratio(r['std_ratio'])}")
    lines.append('')
    lines.append(f"## 判別結論: {verdict_info['verdict']}")
    return '\n'.join(lines)


def _fmt_ratio(v):
    return f"{v:.2f}" if v is not None else 'N/A'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='OT状態別の操舵角PSD比較(蛇行=限界サイクル励起 vs 過渡応答の判別)')
    parser.add_argument('log', help='autoware.log(OT状態遷移の抽出元)')
    parser.add_argument('bag', help='rosbag(.mcap、操舵角・速度の抽出元)')
    parser.add_argument('--sample-hz', type=float, default=40.0,
                         help='等間隔リサンプリング後のサンプリングレート(既定40Hz)')
    parser.add_argument('--out-dir', default='.', help='PSDプロット(PNG)の出力先')
    args = parser.parse_args(argv)

    result = analyze(args.log, args.bag, sample_hz=args.sample_hz)
    verdict_info = discriminate(result['per_state'])
    print(format_report(result, verdict_info, args.log, args.bag))

    out_path = Path(args.out_dir) / 'steering_psd.png'
    plot_psd(result['per_state'], out_path)
    print(f"\nプロット保存: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
