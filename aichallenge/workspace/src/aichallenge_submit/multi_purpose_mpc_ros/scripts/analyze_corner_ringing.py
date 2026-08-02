#!/usr/bin/env python3
"""analyze_corner_ringing.py

corner-exit ringingのアンサンブル分析(2026-08-03、Part1-A)。208節で発見された
「コーナー立ち上がり後のリンギング(6.10° vs 3.17°)」が、実際にはコース全体で
常時発生している蛇行(限界サイクル)とどう関係しているかを、複数周回のアンサンブル
平均で定量化する。制御には一切関与しない、オフライン分析専用ツール。

手法:
  1. waypoint CSV(kappa_radpm列)からコーナー区間・出口wp_idを機械的に検出する
     (|kappa|が閾値を超える連続区間の終端 = コーナー出口)。
  2. autoware.logの[LOC-XCHECK](~4Hz)からwp_id-時刻の疎な対応を取得し、線形補間で
     各コーナー出口を通過した時刻を周回ごとに特定する。
  3. rosbagの操舵角(40Hz、analyze_steering_psd.read_steering_seriesを再利用)から、
     各通過時刻を原点として前後の波形を切り出し、同一コーナーの複数周回分を
     アンサンブル平均する(ノイズが落ち、ringingの素性が読める)。
  4. アンサンブル波形から、通過直後の操舵角トレンド(移動平均)を差し引いた残差の
     ピークを検出し、振動周波数(隣接ピーク間隔の逆数)・対数減衰率(隣接ピーク振幅比
     の自然対数)・初期振幅・整定時間(振幅が初期の10%へ落ちるまで)を推定する。

データソース:
  - waypoint CSV: multi_purpose_mpc_ros/env/final_ver3/traj_mincurv.csv
    (s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2)
  - wp_id-時刻対応: autoware.logの[LOC-XCHECK] wp=... kappa=... ekf_ey=... v=... ot=...
  - 操舵角: rosbagの/control/command/control_cmd(analyze_steering_psd.pyと同じCDR手動パース)

使い方:
    python3 analyze_corner_ringing.py --waypoints <csv> <log1>:<bag1>:<label1> [<log2>:<bag2>:<label2> ...]
"""
import argparse
import bisect
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from analyze_steering_psd import read_steering_series  # noqa: E402

KAPPA_THR = 0.05  # [1/m] コーナー判定閾値(緩め、pass_block_kappa=0.10より小さく全コーナーを拾う)
MIN_CORNER_ARC_M = 3.0  # [m] これ未満の短い区間は接続ノイズとして除外
PRE_S = 1.0  # [s] 出口通過時刻より前に切り出す長さ
POST_S = 4.0  # [s] 出口通過時刻より後に切り出す長さ(既知のringing整定時間2.5s程度を包含)
SAMPLE_HZ = 40.0
LOWPASS_HZ = 2.0  # [Hz] 既知の限界サイクル(0.6-0.7Hz)より十分高く、センサ/制御ノイズより低いカットオフ
MIN_PEAK_HEIGHT_DEG = 0.03  # [deg] ピークとみなす残差の最小振幅(ノイズ床対策)
MIN_LAPS_FOR_ENSEMBLE = 2  # このラップ数未満のコーナーはアンサンブル対象から除外


def load_waypoint_kappa(csv_path):
    """waypoint CSVを手動パースする(pandasはnumpy ABI不整合で警告が出るため、
    csvモジュールで十分な軽量パースに留める)。"""
    import csv as csv_mod
    s_m, kappa = [], []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            s_m.append(float(row['s_m']))
            kappa.append(float(row['kappa_radpm']))
    return np.array(s_m), np.array(kappa)


def detect_corner_exits(kappa, min_corner_arc_m=MIN_CORNER_ARC_M, kappa_thr=KAPPA_THR,
                         wp_spacing_m=1.0):
    """|kappa|>thrの連続区間(コーナー)の終端wp_idを返す。周回コースのため
    末尾→先頭のラップアラウンドも1箇所だけ考慮する。min_corner_arc_m未満の
    短い区間(接続ノイズ)は除外する。"""
    in_corner = np.abs(kappa) > kappa_thr
    n = len(kappa)
    # ラップアラウンドを扱うため2周分に拡張してから中央の1周分を採用する
    ext = np.concatenate([in_corner, in_corner])
    transitions = np.diff(ext.astype(int))
    exits = np.where(transitions == -1)[0] + 1  # True->False、遷移後の最初のwp
    entries = np.where(transitions == 1)[0] + 1
    exits = exits[(exits >= n) & (exits < 2 * n)] % n if len(exits[exits < n]) < len(exits) else exits[exits < n]
    # 上の1行は「1周目の遷移が無い場合は2周目(=同じ形状)から拾う」ためのフォールバック。
    # min_corner_arc_m未満の区間を除外するには対応するentryとの距離を見る必要があるが、
    # 簡略化のため「隣接するexit同士がwp_spacing_m×2未満なら片方を除去」する処理に留める。
    exits = np.sort(np.unique(exits))
    if len(exits) == 0:
        return exits
    min_gap_wp = max(1, int(min_corner_arc_m / max(wp_spacing_m, 1e-6)))
    filtered = [exits[0]]
    for e in exits[1:]:
        if e - filtered[-1] >= min_gap_wp:
            filtered.append(e)
    return np.array(filtered)


def read_wp_time_series(log_path):
    """autoware.logから(t, wp_id)のリストを時刻昇順で返す([LOC-XCHECK]、~4Hz)。"""
    series = []
    pattern = re.compile(r'\[(\d{10}\.\d+)\].*\[LOC-XCHECK\] wp=(\d+)')
    with open(log_path, errors='replace') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                series.append((float(m.group(1)), int(m.group(2))))
    series.sort(key=lambda p: p[0])
    return series


def find_corner_crossings(wp_series, exit_wp, n_wp):
    """wp_seriesの中から、wp_idがexit_wpをまたいだ(通過した)時刻を全周回分検出し、
    線形補間で通過時刻を推定する。1周回1回だけカウントする(wp_idが単調増加する
    区間のみを見る、STUCK等でのwp後退は無視)。"""
    crossings = []
    for i in range(1, len(wp_series)):
        t0, wp0 = wp_series[i - 1]
        t1, wp1 = wp_series[i]
        # 周回のラップアラウンド(wp1 << wp0、例: 348->2)はスキップ対象外として個別に扱う
        if wp1 < wp0 and (wp0 - wp1) > n_wp * 0.5:
            wp1_adj = wp1 + n_wp
        else:
            wp1_adj = wp1
        if wp1_adj < wp0:
            continue  # 後退(STUCK等)はクロッシング検出対象外
        target = exit_wp
        if wp0 <= target < wp1_adj or (wp1_adj >= n_wp and wp0 <= target + n_wp < wp1_adj):
            span = wp1_adj - wp0
            if span <= 0:
                continue
            frac = (target - wp0) / span if wp0 <= target < wp1_adj else (target + n_wp - wp0) / span
            frac = min(max(frac, 0.0), 1.0)
            crossings.append(t0 + frac * (t1 - t0))
    return crossings


def extract_windows(steering_series, crossing_times, pre_s=PRE_S, post_s=POST_S,
                     sample_hz=SAMPLE_HZ):
    """各通過時刻を原点として[-pre_s, +post_s]の操舵角波形(deg)を共通グリッドへ
    線形補間して切り出す。データが不足するクロッシングはスキップする。"""
    times = np.array([p[0] for p in steering_series])
    vals = np.degrees(np.array([p[1] for p in steering_series]))
    grid = np.arange(-pre_s, post_s, 1.0 / sample_hz)
    windows = []
    for tc in crossing_times:
        t_lo, t_hi = tc - pre_s, tc + post_s
        if t_lo < times[0] or t_hi > times[-1]:
            continue
        idx_lo = bisect.bisect_left(times, t_lo)
        idx_hi = bisect.bisect_right(times, t_hi)
        if idx_hi - idx_lo < 10:
            continue
        w = np.interp(grid + tc, times, vals)
        windows.append(w)
    return grid, windows


def estimate_ringing(grid, ensemble_mean, sample_hz=SAMPLE_HZ, lowpass_hz=LOWPASS_HZ,
                      min_peak_height=MIN_PEAK_HEIGHT_DEG):
    """アンサンブル平均波形から振動パラメータを推定する。
    (1) ローパスフィルタ(cutoff=lowpass_hz、既知の限界サイクル0.6-0.7Hzより十分高く
        センサ/制御ノイズより低い)で高周波成分を除去し、(2) 線形トレンド(コーナー
        出口通過後の緩やかな定常操舵角変化)を差し引いた残差からピークを検出する。
    振動周波数(隣接ピーク間隔の逆数)・対数減衰率(隣接ピーク振幅比の自然対数の平均)・
    初期振幅・整定時間(残差が初期振幅の15%未満に落ちて以降そのまま戻らない最初の
    時刻)を求める。ピークが2個未満の場合はNoneを返す(判定不能)。"""
    post_mask = grid >= 0
    post_grid = grid[post_mask]
    post_wave = ensemble_mean[post_mask]
    if len(post_wave) < 10:
        return None

    nyq = sample_hz / 2.0
    b, a = butter(2, lowpass_hz / nyq, btype='low')
    filtered = filtfilt(b, a, post_wave)

    coeffs = np.polyfit(post_grid, filtered, 1)
    trend = np.polyval(coeffs, post_grid)
    residual = filtered - trend

    peak_idx, _ = find_peaks(residual, height=min_peak_height)
    if len(peak_idx) < 2:
        return None

    peak_times = post_grid[peak_idx]
    peak_amps = residual[peak_idx]
    periods = np.diff(peak_times)
    freq_hz = float(1.0 / np.mean(periods)) if len(periods) else None

    valid_ratios = [peak_amps[i] / peak_amps[i + 1] for i in range(len(peak_amps) - 1)
                    if peak_amps[i + 1] > 1e-6 and peak_amps[i] > peak_amps[i + 1]]
    log_decrement = float(np.mean([np.log(r) for r in valid_ratios])) if valid_ratios else None

    initial_amp = float(peak_amps[0]) if len(peak_amps) else None
    settle_time = None
    if initial_amp:
        for idx in range(len(residual)):
            if np.all(np.abs(residual[idx:]) < initial_amp * 0.15):
                settle_time = float(post_grid[idx])
                break

    return {
        'freq_hz': freq_hz, 'log_decrement': log_decrement,
        'initial_amp_deg': initial_amp, 'settle_time_s': settle_time,
        'n_peaks': len(peak_idx),
    }


def analyze_log(log_path, bag_path, waypoint_csv, label):
    s_m, kappa = load_waypoint_kappa(waypoint_csv)
    n_wp = len(kappa)
    wp_spacing = float(np.mean(np.diff(s_m))) if len(s_m) > 1 else 1.0
    exits = detect_corner_exits(kappa, wp_spacing_m=wp_spacing)

    wp_series = read_wp_time_series(log_path)
    steering_series = read_steering_series(bag_path)

    results = []
    for exit_wp in exits:
        crossings = find_corner_crossings(wp_series, int(exit_wp), n_wp)
        if len(crossings) < MIN_LAPS_FOR_ENSEMBLE:
            results.append({'exit_wp': int(exit_wp), 'n_laps': len(crossings), 'skipped': True})
            continue
        grid, windows = extract_windows(steering_series, crossings)
        if len(windows) < MIN_LAPS_FOR_ENSEMBLE:
            results.append({'exit_wp': int(exit_wp), 'n_laps': len(windows), 'skipped': True})
            continue
        stacked = np.stack(windows)
        ens_mean = stacked.mean(axis=0)
        ens_std = stacked.std(axis=0)
        params = estimate_ringing(grid, ens_mean)
        results.append({
            'exit_wp': int(exit_wp), 'n_laps': len(windows), 'skipped': False,
            'grid': grid, 'ens_mean': ens_mean, 'ens_std': ens_std,
            'params': params,
        })
    return {'label': label, 'log_path': log_path, 'bag_path': bag_path,
            'n_wp': n_wp, 'exits': [int(e) for e in exits], 'results': results}


def format_report(all_results):
    lines = []
    lines.append('# corner-exit ringingアンサンブル分析')
    lines.append('')
    for r in all_results:
        lines.append(f"## {r['label']}")
        lines.append(f"log: {r['log_path']}")
        lines.append(f"検出コーナー出口数: {len(r['exits'])}  wp_id一覧: {r['exits']}")
        lines.append('')
        lines.append('| exit_wp | n_laps | freq_hz | log_decrement | initial_amp_deg | settle_time_s |')
        lines.append('|---|---|---|---|---|---|')
        for res in r['results']:
            if res.get('skipped'):
                lines.append(f"| {res['exit_wp']} | {res['n_laps']}(不足) | - | - | - | - |")
                continue
            p = res['params']
            if p is None:
                lines.append(f"| {res['exit_wp']} | {res['n_laps']} | 判定不能(ピーク<2) | - | - | - |")
                continue
            freq = f"{p['freq_hz']:.3f}" if p['freq_hz'] else 'N/A'
            ld = f"{p['log_decrement']:.3f}" if p['log_decrement'] is not None else 'N/A'
            amp = f"{p['initial_amp_deg']:.2f}" if p['initial_amp_deg'] is not None else 'N/A'
            settle = f"{p['settle_time_s']:.2f}" if p['settle_time_s'] is not None else 'N/A(未整定)'
            lines.append(f"| {res['exit_wp']} | {res['n_laps']} | {freq} | {ld} | {amp} | {settle} |")
        lines.append('')
    return '\n'.join(lines)


def plot_top_corners(all_results, out_dir, top_n=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    ja_fonts = [f.name for f in fm.fontManager.ttflist if 'Noto Sans CJK JP' in f.name]
    if ja_fonts:
        matplotlib.rcParams['font.family'] = ja_fonts[0]

    for r in all_results:
        valid = [res for res in r['results']
                 if not res.get('skipped') and res['params'] is not None
                 and res['params']['initial_amp_deg'] is not None]
        valid.sort(key=lambda x: x['params']['initial_amp_deg'], reverse=True)
        for res in valid[:top_n]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(res['grid'], res['ens_mean'], color='tab:blue', label='アンサンブル平均')
            ax.fill_between(res['grid'], res['ens_mean'] - res['ens_std'],
                             res['ens_mean'] + res['ens_std'], color='tab:blue', alpha=0.2)
            ax.axvline(0, color='gray', linestyle='--', label='コーナー出口通過')
            ax.set_xlabel('出口通過からの時刻 [s]')
            ax.set_ylabel('操舵角 [deg]')
            ax.set_title(f"{r['label']} wp={res['exit_wp']} (n_laps={res['n_laps']})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = Path(out_dir) / f"ringing_{r['label']}_wp{res['exit_wp']}.png"
            fig.savefig(out_path, dpi=110)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='corner-exit ringingのアンサンブル分析')
    parser.add_argument('--waypoints', required=True, help='waypoint CSV(kappa_radpm列を含む)')
    parser.add_argument('logs', nargs='+',
                         help='log:bag:label の形式で複数指定(例: a.log:a.mcap:予選0801)')
    parser.add_argument('--out-dir', default='.', help='プロット(PNG)の出力先')
    parser.add_argument('--plot-top', type=int, default=3,
                         help='ラベルごとに初期振幅が大きい上位N個をプロット(既定3)')
    args = parser.parse_args(argv)

    all_results = []
    for spec in args.logs:
        log_path, bag_path, label = spec.split(':', 2)
        print(f"分析中: {label} ({log_path})", file=sys.stderr)
        all_results.append(analyze_log(log_path, bag_path, args.waypoints, label))

    print(format_report(all_results))
    if args.plot_top > 0:
        plot_top_corners(all_results, args.out_dir, top_n=args.plot_top)
        print(f"\nプロット保存先: {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
