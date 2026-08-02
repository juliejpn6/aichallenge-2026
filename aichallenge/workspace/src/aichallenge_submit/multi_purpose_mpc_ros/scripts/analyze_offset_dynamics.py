#!/usr/bin/env python3
"""analyze_offset_dynamics.py

offset動特性の定量分析(2026-08-03、Part1-B)。OT(追い越し)中のoffset指令(参照
ラインからの横方向オフセット目標)がどれだけ急峻に変化しているか、その変化が
物理的に滑らかに追従可能な範囲を超えているか、OT状態遷移がチャタリングしていないか
を実測ログから定量する。制御には一切関与しない、オフライン分析専用ツール。

データソース:
  - offset/OT状態: autoware.logの[OT] state=... offset=... 行(~1Hz)
  - ekf_ey: autoware.logの[LOC-XCHECK] wp=... ekf_ey=... 行(~4Hz)
  - 操舵角: rosbagの/control/command/control_cmd(analyze_steering_psd.pyと同じCDR手動パース)

使い方:
    python3 analyze_offset_dynamics.py --log <autoware.log> --bag <rosbag.mcap> --label <ラベル> [--out-dir DIR]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_steering_psd import read_steering_series  # noqa: E402

# d_off(空き側へ寄せる目標オフセット、config.yaml既定3.0m)をramp_time(既定2.5s)で
# ランプする設計のため、設計上の最大追従速度は d_off/ramp_time ≈ 1.2 m/s。
DESIGN_D_OFF_M = 3.0
DESIGN_RAMP_TIME_S = 2.5
DESIGN_MAX_RATE_MPS = DESIGN_D_OFF_M / DESIGN_RAMP_TIME_S
SHORT_DWELL_S = 3.0  # [s] この滞在時間未満のOT状態を「短期往復(チャタリング)」とみなす


def read_ot_series(log_path):
    """autoware.logから(t, offset, v_odom_or_none, wp_id, state)のリストを時刻昇順で
    返す。v_odomは[OT]行に直接含まれないため、u0(実行速度指令)を代替指標として使う。"""
    series = []
    pattern = re.compile(
        r'\[(\d{10}\.\d+)\].*\[OT\] state=(\w+) side=-?\d+ obs=\d+ fwd=\d+ wp_id=(\d+) '
        r'.*?offset=(-?[\d.]+) .*?u0=([\d.]+)')
    with open(log_path, errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                series.append({
                    't': float(m.group(1)), 'state': m.group(2),
                    'wp_id': int(m.group(3)), 'offset': float(m.group(4)),
                    'u0': float(m.group(5)),
                })
    series.sort(key=lambda r: r['t'])
    return series


def read_ekf_ey_series(log_path):
    series = []
    pattern = re.compile(r'\[(\d{10}\.\d+)\].*\[LOC-XCHECK\] wp=\d+ kappa=-?[\d.]+ ekf_ey=(-?[\d.]+)')
    with open(log_path, errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                series.append((float(m.group(1)), float(m.group(2))))
    series.sort(key=lambda p: p[0])
    return series


def compute_rate_stats(series):
    """offsetの変化率(m/s)分布・ステップ幅(非ゼロ変化量)分布を計算する。"""
    rates = []
    steps = []
    for i in range(1, len(series)):
        dt = series[i]['t'] - series[i - 1]['t']
        if dt <= 0:
            continue
        d_off = series[i]['offset'] - series[i - 1]['offset']
        if abs(d_off) < 1e-6:
            continue
        rates.append(d_off / dt)
        steps.append(d_off)
    return np.array(rates), np.array(steps)


def evaluate_trackability(rates, design_max_rate=DESIGN_MAX_RATE_MPS):
    """実測変化率のうち、設計上の最大追従速度(d_off/ramp_time)を超える割合を返す。"""
    if len(rates) == 0:
        return {'n': 0, 'exceed_ratio': None, 'design_max_rate_mps': design_max_rate}
    exceed = np.abs(rates) > design_max_rate
    return {
        'n': len(rates),
        'exceed_ratio': float(np.mean(exceed)),
        'design_max_rate_mps': design_max_rate,
        'p50_abs': float(np.median(np.abs(rates))),
        'p95_abs': float(np.percentile(np.abs(rates), 95)),
        'max_abs': float(np.max(np.abs(rates))),
    }


def analyze_ot_chattering(series, short_dwell_s=SHORT_DWELL_S):
    """OT状態の遷移頻度・滞在時間分布を計算し、短期往復(チャタリング)の件数を返す。"""
    if len(series) < 2:
        return {'n_transitions': 0, 'n_short_dwell': 0, 'dwell_times': []}
    segments = []
    cur_state = series[0]['state']
    cur_start = series[0]['t']
    for r in series[1:]:
        if r['state'] != cur_state:
            segments.append((cur_state, r['t'] - cur_start))
            cur_state = r['state']
            cur_start = r['t']
    segments.append((cur_state, series[-1]['t'] - cur_start))
    dwell_times = [d for _, d in segments]
    n_short = sum(1 for d in dwell_times if d < short_dwell_s)
    return {
        'n_transitions': len(segments) - 1,
        'n_short_dwell': n_short,
        'dwell_times': dwell_times,
        'mean_dwell_s': float(np.mean(dwell_times)) if dwell_times else None,
        'median_dwell_s': float(np.median(dwell_times)) if dwell_times else None,
    }


def analyze_log(log_path, bag_path, label):
    ot_series = read_ot_series(log_path)
    rates, steps = compute_rate_stats(ot_series)
    trackability = evaluate_trackability(rates)
    chattering = analyze_ot_chattering(ot_series)
    return {
        'label': label, 'log_path': log_path, 'bag_path': bag_path,
        'ot_series': ot_series, 'rates': rates, 'steps': steps,
        'trackability': trackability, 'chattering': chattering,
    }


def format_report(all_results):
    lines = []
    lines.append('# offset動特性の定量分析')
    lines.append('')
    lines.append(f'設計上の最大追従速度(d_off={DESIGN_D_OFF_M}m / ramp_time={DESIGN_RAMP_TIME_S}s) = '
                  f'{DESIGN_MAX_RATE_MPS:.2f} m/s')
    lines.append(f'短期往復(チャタリング)判定閾値: 滞在時間 < {SHORT_DWELL_S}s')
    lines.append('')
    lines.append('## offset変化率・追従可能性')
    lines.append('| label | n_events | 設計超過率 | p50\\|rate\\| | p95\\|rate\\| | max\\|rate\\| |')
    lines.append('|---|---|---|---|---|---|')
    for r in all_results:
        t = r['trackability']
        if t['n'] == 0:
            lines.append(f"| {r['label']} | 0 | N/A(offset変化なし) | - | - | - |")
            continue
        lines.append(f"| {r['label']} | {t['n']} | {t['exceed_ratio']*100:.1f}% | "
                      f"{t['p50_abs']:.2f} | {t['p95_abs']:.2f} | {t['max_abs']:.2f} |")
    lines.append('')
    lines.append('## OT状態遷移のチャタリング')
    lines.append('| label | n_transitions | n_short_dwell(<3s) | mean_dwell_s | median_dwell_s |')
    lines.append('|---|---|---|---|---|')
    for r in all_results:
        c = r['chattering']
        mean_s = f"{c['mean_dwell_s']:.2f}" if c['mean_dwell_s'] is not None else 'N/A'
        med_s = f"{c['median_dwell_s']:.2f}" if c['median_dwell_s'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {c['n_transitions']} | {c['n_short_dwell']} | {mean_s} | {med_s} |")
    return '\n'.join(lines)


def plot_offset_ey_steer(result, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    ja_fonts = [f.name for f in fm.fontManager.ttflist if 'Noto Sans CJK JP' in f.name]
    if ja_fonts:
        matplotlib.rcParams['font.family'] = ja_fonts[0]

    ot_series = result['ot_series']
    if not ot_series:
        return
    ekf_series = read_ekf_ey_series(result['log_path'])
    steer_series = read_steering_series(result['bag_path'])

    t0 = ot_series[0]['t']
    ot_t = [r['t'] - t0 for r in ot_series]
    ot_off = [r['offset'] for r in ot_series]
    ekf_t = [t - t0 for t, _ in ekf_series]
    ekf_v = [v for _, v in ekf_series]
    steer_t = [t - t0 for t, _ in steer_series]
    steer_v = [np.degrees(v) for _, v in steer_series]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(ot_t, ot_off, color='tab:orange')
    axes[0].set_ylabel('offset [m]')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ekf_t, ekf_v, color='tab:green')
    axes[1].set_ylabel('ekf_ey [m]')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steer_t, steer_v, color='tab:blue', linewidth=0.5)
    axes[2].set_ylabel('操舵角 [deg]')
    axes[2].set_xlabel('時刻 [s]')
    axes[2].grid(True, alpha=0.3)

    # OT状態遷移を縦線で重ねる
    prev_state = None
    for r in ot_series:
        if r['state'] != prev_state:
            for ax in axes:
                ax.axvline(r['t'] - t0, color='gray', linestyle=':', alpha=0.4)
            prev_state = r['state']

    fig.suptitle(f"{result['label']}: offset / ekf_ey / 操舵角 とOT遷移")
    fig.tight_layout()
    out_path = Path(out_dir) / f"offset_dynamics_{result['label']}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='offset動特性の定量分析')
    parser.add_argument('specs', nargs='+', help='log:bag:label の形式で複数指定')
    parser.add_argument('--out-dir', default='.', help='プロット(PNG)の出力先')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)

    all_results = []
    for spec in args.specs:
        log_path, bag_path, label = spec.split(':', 2)
        print(f"分析中: {label} ({log_path})", file=sys.stderr)
        all_results.append(analyze_log(log_path, bag_path, label))

    print(format_report(all_results))
    if not args.no_plot:
        for r in all_results:
            plot_offset_ey_steer(r, args.out_dir)
        print(f"\nプロット保存先: {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
