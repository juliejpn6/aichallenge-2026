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


def analyze_log(log_path, bag_path, waypoint_csv, label, exclude_wp_range=None):
    """exclude_wp_rangeは(lo, hi)のタプル(両端含む)。指定した場合、その範囲に
    入るexit_wpは解析対象から完全に除外する(239-240節のwp269-282のような
    STUCK/巨大横偏差イベントは小振幅の定常限界サイクルとは別現象であり、
    同じ枠組みで混ぜて集計すると平均振幅・減衰率が歪むため)。"""
    s_m, kappa = load_waypoint_kappa(waypoint_csv)
    n_wp = len(kappa)
    wp_spacing = float(np.mean(np.diff(s_m))) if len(s_m) > 1 else 1.0
    exits = detect_corner_exits(kappa, wp_spacing_m=wp_spacing)

    excluded_exits = []
    if exclude_wp_range is not None:
        lo, hi = exclude_wp_range
        excluded_exits = [int(e) for e in exits if lo <= e <= hi]
        exits = np.array([e for e in exits if not (lo <= e <= hi)])

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
            'n_wp': n_wp, 'exits': [int(e) for e in exits], 'results': results,
            'excluded_exits': excluded_exits}


HOTSPOT_LINE_RE = re.compile(
    r'\[(\d{10}\.\d+)\].*\[LOC-XCHECK\] wp=(\d+) kappa=(-?[\d.]+) '
    r'ekf_ey=(-?[\d.]+) gnss_ey=(-?[\d.]+) v=(-?[\d.]+) ot=(\w+)')
STALL_SPEED_THR_MPS = 0.3  # [m/s] これ未満を「準停止」とみなす閾値(0803-04のwp285準停止v=1.39は
                            # 対象外、0803-03のwp281完全停止v=0.00は確実に拾う保守的な値)
STALL_MIN_DURATION_S = 0.5  # [s] これ以上継続した低速区間のみstallイベントとして計上する
HOTSPOT_LAP_GAP_S = 5.0  # [s] これを超える時刻の空白は周回間の切れ目とみなす


def read_hotspot_series(log_path, wp_lo, wp_hi, margin_wp=5):
    """[LOC-XCHECK]から wp_lo-margin 〜 wp_hi+margin の範囲の生時系列
    (t, wp, ekf_ey, v)を全周回分読み出す。ringing分析と異なり、フィルタ・
    トレンド除去・アンサンブル平均は一切行わない(239-240節のwp269-282の
    ようなSTUCK/巨大横偏差イベントは、平滑化すると実態が消えてしまうため)。"""
    lo_ext, hi_ext = wp_lo - margin_wp, wp_hi + margin_wp
    rows = []
    with open(log_path, errors='replace') as f:
        for line in f:
            m = HOTSPOT_LINE_RE.search(line)
            if not m:
                continue
            t_s, wp_s, _kappa_s, ekf_ey_s, _gnss_ey_s, v_s, _ot_s = m.groups()
            wp = int(wp_s)
            if lo_ext <= wp <= hi_ext:
                rows.append((float(t_s), wp, float(ekf_ey_s), float(v_s)))
    rows.sort(key=lambda row: row[0])
    return rows


def split_hotspot_laps(rows, gap_s=HOTSPOT_LAP_GAP_S):
    """時刻の大きなギャップ(周回間隔、既定5秒超)でrowsを周回単位に分割する。
    サンプル数が極端に少ない断片(3点未満、通過が浅く区間の一部しか
    かすらなかったケース)は除外する。"""
    if not rows:
        return []
    laps = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if cur[0] - prev[0] > gap_s:
            laps.append([])
        laps[-1].append(cur)
    return [lap for lap in laps if len(lap) >= 3]


def analyze_hotspot(log_path, wp_lo, wp_hi, label):
    """wp269-282のような既知ホットスポット専用の生データ分析。ringingの
    小振幅定常振動とは別現象(STUCK・巨大横偏差)であるとの判断(C-0-3)に
    基づき、一般ringing分析とは別の枠組みで、周回ごとの最大|ekf_ey|・
    stall(準停止/停止)イベント発生率を集計する。"""
    rows = read_hotspot_series(log_path, wp_lo, wp_hi)
    laps = split_hotspot_laps(rows)
    lap_stats = []
    for lap in laps:
        max_abs_ey = max(abs(r[2]) for r in lap)
        min_v = min(r[3] for r in lap)
        stall_dur = 0.0
        cur_start = None
        for (t, _wp, _ey, v) in lap:
            if v < STALL_SPEED_THR_MPS:
                if cur_start is None:
                    cur_start = t
                stall_dur = max(stall_dur, t - cur_start)
            else:
                cur_start = None
        lap_stats.append({
            'max_abs_ey': max_abs_ey, 'min_v': min_v, 'stall_dur_s': stall_dur,
            'is_stall_event': stall_dur >= STALL_MIN_DURATION_S,
        })
    n_laps = len(lap_stats)
    n_stall = sum(1 for s in lap_stats if s['is_stall_event'])
    max_ey_arr = np.array([s['max_abs_ey'] for s in lap_stats]) if lap_stats else np.array([])
    return {
        'label': label, 'wp_range': (wp_lo, wp_hi), 'n_laps': n_laps,
        'n_stall_events': n_stall,
        'stall_rate': (n_stall / n_laps) if n_laps else None,
        'p50_max_ey': float(np.percentile(max_ey_arr, 50)) if len(max_ey_arr) else None,
        'p95_max_ey': float(np.percentile(max_ey_arr, 95)) if len(max_ey_arr) else None,
        'max_max_ey': float(np.max(max_ey_arr)) if len(max_ey_arr) else None,
        'lap_stats': lap_stats,
    }


def format_hotspot_report(hotspot_results):
    lines = []
    lines.append('# wp269-282専用(既知ホットスポット)生データ分析')
    lines.append('')
    lines.append(f"stall判定: v<{STALL_SPEED_THR_MPS}m/sが{STALL_MIN_DURATION_S}s以上連続")
    lines.append('')
    lines.append('| ログ | n_laps | n_stall_events | stall_rate | p50\\|ekf_ey\\| | p95\\|ekf_ey\\| | max\\|ekf_ey\\| |')
    lines.append('|---|---|---|---|---|---|---|')
    for r in hotspot_results:
        if r['n_laps'] == 0:
            lines.append(f"| {r['label']} | 0 | - | - | - | - | - |")
            continue
        sr = f"{r['stall_rate'] * 100:.1f}%" if r['stall_rate'] is not None else 'N/A'
        p50 = f"{r['p50_max_ey']:.2f}" if r['p50_max_ey'] is not None else 'N/A'
        p95 = f"{r['p95_max_ey']:.2f}" if r['p95_max_ey'] is not None else 'N/A'
        mx = f"{r['max_max_ey']:.2f}" if r['max_max_ey'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {r['n_laps']} | {r['n_stall_events']} | {sr} | {p50} | {p95} | {mx} |")
    lines.append('')
    return '\n'.join(lines)


# Part C(Q×v_max実験)の合否基準(2026-08-03、C-0-3で固定)。実験条件間の比較が
# 印象論で流動しないよう、実験前に固定する。基準値はcorner_ringing_offset_dynamics_
# 20260803.mdのPart1-A実測(Q=700k、wp269-282混入込みの19コーナー平均: 初期振幅
# 6.04-7.17°、対数減衰率0.61-1.15)を参考にした暫定値であり、本ツールでのwp269-282
# 除外後の再集計(現行Q=700k条件の実測)はまだ実施していない。次回Part C実験で
# "現行"条件を本ツールにかけた時点で、除外後の実測ベースラインに合わせて更新する。
RINGING_PASS_MEAN_INITIAL_AMP_DEG = 6.0  # 以下ならPASS
RINGING_PASS_MEAN_LOG_DECREMENT = 0.90  # 以上ならPASS(収束が速い)
HOTSPOT_PASS_STALL_RATE = 0.0  # wp269-282でのstallイベント発生率、0%がPASS
HOTSPOT_PASS_P95_EY_M = 1.5  # p95|ekf_ey|がこれ未満ならPASS(237節の異常閾値=名目クリアランス30%減)


def evaluate_pass_criteria(all_results, hotspot_results):
    lines = []
    lines.append('# Part C 合否基準判定(固定基準、C-0-3・暫定値)')
    lines.append('')
    lines.append(
        f"- 一般ringing: 平均初期振幅 <= {RINGING_PASS_MEAN_INITIAL_AMP_DEG}deg かつ "
        f"平均対数減衰率 >= {RINGING_PASS_MEAN_LOG_DECREMENT}")
    lines.append(
        f"- wp269-282ホットスポット: stall_rate <= {HOTSPOT_PASS_STALL_RATE * 100:.0f}% かつ "
        f"p95|ekf_ey| < {HOTSPOT_PASS_P95_EY_M}m")
    lines.append('')
    lines.append('| ログ | 平均初期振幅 | 平均対数減衰率 | ringing判定 | stall_rate | p95\\|ekf_ey\\| | hotspot判定 |')
    lines.append('|---|---|---|---|---|---|---|')
    hotspot_by_label = {h['label']: h for h in hotspot_results}
    for r in all_results:
        valid = [res for res in r['results']
                 if not res.get('skipped') and res['params'] is not None
                 and res['params']['initial_amp_deg'] is not None
                 and res['params']['log_decrement'] is not None]
        if valid:
            mean_amp = float(np.mean([res['params']['initial_amp_deg'] for res in valid]))
            mean_ld = float(np.mean([res['params']['log_decrement'] for res in valid]))
            ringing_pass = (mean_amp <= RINGING_PASS_MEAN_INITIAL_AMP_DEG
                             and mean_ld >= RINGING_PASS_MEAN_LOG_DECREMENT)
            ringing_verdict = 'PASS' if ringing_pass else 'FAIL'
            amp_s, ld_s = f'{mean_amp:.2f}', f'{mean_ld:.2f}'
        else:
            amp_s, ld_s, ringing_verdict = 'N/A', 'N/A', '判定不能'
        h = hotspot_by_label.get(r['label'])
        if h and h['n_laps'] > 0:
            sr, p95 = h['stall_rate'], h['p95_max_ey']
            hotspot_pass = sr <= HOTSPOT_PASS_STALL_RATE and p95 < HOTSPOT_PASS_P95_EY_M
            hotspot_verdict = 'PASS' if hotspot_pass else 'FAIL'
            sr_s, p95_s = f'{sr * 100:.1f}%', f'{p95:.2f}'
        else:
            sr_s, p95_s, hotspot_verdict = 'N/A', 'N/A', '判定不能'
        lines.append(f"| {r['label']} | {amp_s} | {ld_s} | {ringing_verdict} | "
                      f"{sr_s} | {p95_s} | {hotspot_verdict} |")
    lines.append('')
    return '\n'.join(lines)


def format_report(all_results):
    lines = []
    lines.append('# corner-exit ringingアンサンブル分析')
    lines.append('')
    for r in all_results:
        lines.append(f"## {r['label']}")
        lines.append(f"log: {r['log_path']}")
        lines.append(f"検出コーナー出口数: {len(r['exits'])}  wp_id一覧: {r['exits']}")
        if r.get('excluded_exits'):
            lines.append(f"除外(ホットスポット範囲、別途wp269-282専用分析で扱う): {r['excluded_exits']}")
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
    parser.add_argument('--exclude-wp-range', type=int, nargs=2, default=[269, 282],
                         metavar=('LO', 'HI'),
                         help='この範囲のexit_wpを一般ringing分析から除外する'
                              '(既定269 282、239-240節の既知ホットスポット)。'
                              '除外したくない場合は0 0を指定する。')
    parser.add_argument('--hotspot-wp-range', type=int, nargs=2, default=[269, 282],
                         metavar=('LO', 'HI'),
                         help='wp269-282専用の生データ分析(stall/巨大偏差検出)の対象範囲(既定269 282)')
    parser.add_argument('--no-hotspot', action='store_true',
                         help='wp269-282専用の生データ分析セクションを省略する')
    parser.add_argument('--no-pass-criteria', action='store_true',
                         help='Part C合否基準判定セクションを省略する')
    args = parser.parse_args(argv)

    exclude_range = tuple(args.exclude_wp_range) if tuple(args.exclude_wp_range) != (0, 0) else None

    all_results = []
    hotspot_results = []
    for spec in args.logs:
        log_path, bag_path, label = spec.split(':', 2)
        print(f"分析中: {label} ({log_path})", file=sys.stderr)
        all_results.append(analyze_log(log_path, bag_path, args.waypoints, label,
                                        exclude_wp_range=exclude_range))
        if not args.no_hotspot:
            lo, hi = args.hotspot_wp_range
            hotspot_results.append(analyze_hotspot(log_path, lo, hi, label))

    print(format_report(all_results))
    if not args.no_hotspot:
        print(format_hotspot_report(hotspot_results))
    if not args.no_pass_criteria:
        print(evaluate_pass_criteria(all_results, hotspot_results))
    if args.plot_top > 0:
        plot_top_corners(all_results, args.out_dir, top_n=args.plot_top)
        print(f"\nプロット保存先: {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
