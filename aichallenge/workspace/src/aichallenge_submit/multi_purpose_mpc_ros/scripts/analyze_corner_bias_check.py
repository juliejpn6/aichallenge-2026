#!/usr/bin/env python3
"""analyze_corner_bias_check.py

質的観察(内巻き・低周波蛇行)の軽量定量化(2026-08-04、Q[e_psi]反復調整ラウンドA)。

背景: Q[e_psi]=1000000の走行で、ユーザーがwp180・wp252・wp340付近で「内巻き」
(コーナーで旋回ラインが内側へ寄りすぎる)・「蛇行」を目視確認した。既存の
analyze_corner_ringing.pyのコーナー出口振幅分析は「リンギング(2ピーク以上の
減衰振動)」を検出する枠組みであり、内巻きのような符号付きバイアス(振動ではなく
ライン逸脱そのもの)は検出できない可能性が高い(Gemini・別Claude両者の指摘と一致)。

analyze_corner_ringing.pyのread_hotspot_series/split_hotspot_laps(wp範囲汎用)を
再利用し、指定wp区間の**符号付き**ekf_ey統計(平均・最小・最大、絶対値ではない)を
周回別に出す。制御には一切関与しない、オフライン分析専用ツール。

使い方:
    python3 analyze_corner_bias_check.py --wp-ranges 175:185 247:257 335:345 \
        <log1>:<label1> [<log2>:<label2> ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_corner_ringing import read_hotspot_series, split_hotspot_laps  # noqa: E402


def analyze_signed_bias(log_path, wp_lo, wp_hi, label):
    rows = read_hotspot_series(log_path, wp_lo, wp_hi, margin_wp=0)
    laps = split_hotspot_laps(rows)
    lap_stats = []
    for lap in laps:
        ey_vals = np.array([r[2] for r in lap])
        lap_stats.append({
            'mean_ey': float(np.mean(ey_vals)),
            'min_ey': float(np.min(ey_vals)),
            'max_ey': float(np.max(ey_vals)),
        })
    if not lap_stats:
        return {'label': label, 'wp_range': (wp_lo, wp_hi), 'n_laps': 0}
    means = np.array([s['mean_ey'] for s in lap_stats])
    return {
        'label': label, 'wp_range': (wp_lo, wp_hi), 'n_laps': len(lap_stats),
        'mean_of_means': float(np.mean(means)), 'std_of_means': float(np.std(means)),
        'overall_min': float(min(s['min_ey'] for s in lap_stats)),
        'overall_max': float(max(s['max_ey'] for s in lap_stats)),
        'lap_stats': lap_stats,
    }


def format_report(results_by_range):
    lines = ['# コーナーバイアス検査(内巻き/符号付きekf_ey統計)', '']
    for (wp_lo, wp_hi), results in results_by_range.items():
        lines.append(f'## wp{wp_lo}-{wp_hi}')
        lines.append('| ログ | 周回数 | 平均(周回平均) | 周回間std | 区間内min | 区間内max |')
        lines.append('|---|---|---|---|---|---|')
        for r in results:
            if r['n_laps'] == 0:
                lines.append(f"| {r['label']} | 0 | - | - | - | - |")
                continue
            lines.append(f"| {r['label']} | {r['n_laps']} | {r['mean_of_means']:.3f}m | "
                         f"{r['std_of_means']:.3f}m | {r['overall_min']:.3f}m | "
                         f"{r['overall_max']:.3f}m |")
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='コーナーバイアス(内巻き)の軽量定量化')
    parser.add_argument('logs', nargs='+', help='log:label の形式で複数指定')
    parser.add_argument('--wp-ranges', nargs='+', required=True,
                         help='lo:hi 形式で複数指定(例: 175:185 247:257 335:345)')
    args = parser.parse_args(argv)

    ranges = []
    for r in args.wp_ranges:
        lo, hi = r.split(':')
        ranges.append((int(lo), int(hi)))

    specs = []
    for spec in args.logs:
        log_path, label = spec.split(':', 1)
        specs.append((log_path, label))

    results_by_range = {}
    for wp_lo, wp_hi in ranges:
        results = []
        for log_path, label in specs:
            print(f"分析中: {label} wp{wp_lo}-{wp_hi}", file=sys.stderr)
            results.append(analyze_signed_bias(log_path, wp_lo, wp_hi, label))
        results_by_range[(wp_lo, wp_hi)] = results

    print(format_report(results_by_range))
    return 0


if __name__ == '__main__':
    sys.exit(main())
