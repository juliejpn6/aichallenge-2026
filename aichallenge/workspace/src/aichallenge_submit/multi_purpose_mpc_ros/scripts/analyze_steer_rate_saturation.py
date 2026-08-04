#!/usr/bin/env python3
"""analyze_steer_rate_saturation.py

蛇行(0.6-0.7Hz)対策・レート飽和検査(2026-08-04、Phase 1改訂版)。

統一機構仮説: SWING区間で必要な操舵レート dδ/dt ≈ L_wb·κ'·v が実舵レート上限
r_maxで飽和し、過渡オーバーシュート(蛇行)を生む——曲率SWING相関(176節、
r=0.81)とv_max支配(Part C)を単一機構で説明する候補。ただし180節(実予選ログ、
2026-07-25)は「蛇行はトラック全体で一様に悪化、直線でも同水準」という、この
κ'駆動仮説と正面から矛盾しかねない実測を残している。本ツールは、この仮説を
トラック全域(コーナーだけでなく直線含む)で成分分解して検証する:
  成分A = |κ'|·vで説明される分(κ'駆動、rate-saturation仮説が対象とする範囲)
  成分B = 残差(κ'に依存しない分、直線での蛇行悪化があればここに現れる)
制御には一切関与しない、オフライン分析専用ツール。

## 単位較正について(最重要、最初に読むこと)

config.yamlの`steer_rate_max`(1.1、コメント「rad/s raw」)は、コメント上
「実測109°/s」に較正済みとされているが、`core/MPC.py`の`_rate_bounds()`実装を
確認したところ、この制約は実際にはu[1]=tan(δ)/L=κ(曲率)のレート(dκ/dt)に
かかっており、config comment上の「rad/s」という単位ラベルと食い違う可能性が
ある。この不一致を解析的に(config算出のみで)解消しようとすると誤った結論に
至るリスクが高いため、本ツールは**実測(dδ/dt)から求めた経験的な飽和上限を
優先して使う**(Stage1.5「推測せず計装で実測してから決める」の方針を踏襲)。
config由来の解析的候補2通りは参考情報としてのみ算出・報告する。

データソース(analyze_steering_psd.py・analyze_corner_ringing.pyと共通):
  - 操舵指令: rosbagの/control/command/control_cmd(CDR手動パース、既存関数再利用)
  - 位置/曲率: autoware.logの[LOC-XCHECK] wp=... kappa=... ekf_ey=... v=... ot=...
  - waypoint CSV: kappa_radpm列からκ'(真の空間微分)を新規算出

使い方:
    python3 analyze_steer_rate_saturation.py --waypoints <csv> --config <config.yaml> \\
        <log1>:<bag1>:<label1> [<log2>:<bag2>:<label2> ...]
"""
import argparse
import bisect
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from analyze_steering_psd import (  # noqa: E402
    read_steering_series, read_speed_series, resample_uniform, welch_psd, band_power,
    segment_by_state, filter_min_length, MIN_SEGMENT_S,
)
from analyze_corner_ringing import load_waypoint_kappa  # noqa: E402

LOC_XCHECK_RE = re.compile(
    r'\[(\d{10}\.\d+)\].*\[LOC-XCHECK\] wp=(\d+) kappa=(-?[\d.]+) '
    r'ekf_ey=(-?[\d.]+) gnss_ey=(-?[\d.]+) v=(-?[\d.]+) ot=(\w+)')
LIMIT_CYCLE_BAND = (0.5, 0.9)  # Hz、既知の限界サイクル0.6-0.7Hzを包含
DEFAULT_R_MAX_COMMENT_DEG_S = 109.0  # config.yamlコメントに明記された実測較正値
SATURATION_FRACTION = 0.95  # r_maxのこの割合以上を「飽和」とみなす
FRIDAY_R_MAX_SCALE = 0.75  # 金曜AWSIM更新でr_maxが約25%低下する想定
DEFAULT_WINDOW_WP = 5  # 連続回帰・4層分析の窓幅(waypoint数、既定resolution≈1mで≈5m窓)
DEFAULT_SWING_LOOKAHEAD_WP = 16  # r_delta_swingスケジュール(177節)と同じ窓
LAG_S = 0.5  # 時刻相関: 飽和イベントからekf_ey拡大までの遅延窓


# ---------------------------------------------------------------------------
# 単位較正
# ---------------------------------------------------------------------------

def load_mpc_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mpc = cfg['mpc']
    bike = cfg['bicycle_model']
    return {
        'steer_rate_max_raw': float(mpc['steer_rate_max']),
        'gain': float(mpc['steering_tire_angle_gain_var']),
        'wheel_base_m': float(bike['length']),
        'N': int(mpc['N']),
    }


def calibrate_r_max_analytical(cfg, comment_value_deg_s=DEFAULT_R_MAX_COMMENT_DEG_S):
    """config由来の解析的候補2通り(A: raw値そのものが既に実舵レート、
    B: raw×gainが実舵レート)を算出し、コメント実測値と比較する。参考情報。"""
    raw = cfg['steer_rate_max_raw']
    gain = cfg['gain']
    rad2deg = 180.0 / np.pi
    candidate_a = raw * rad2deg
    candidate_b = raw * gain * rad2deg
    diff_a = abs(candidate_a - comment_value_deg_s) / comment_value_deg_s
    diff_b = abs(candidate_b - comment_value_deg_s) / comment_value_deg_s
    closer = 'B(raw×gain)' if diff_b < diff_a else 'A(rawそのまま)'
    return {
        'candidate_a_deg_s': candidate_a, 'candidate_b_deg_s': candidate_b,
        'comment_value_deg_s': comment_value_deg_s,
        'diff_a_frac': diff_a, 'diff_b_frac': diff_b, 'closer_candidate': closer,
    }


def compute_steering_rate_series(steering_series):
    """(t, angle_rad)系列から(t_mid, rate_deg_s, angle_deg_at_t_mid)を返す
    (隣接差分)。"""
    t = np.array([p[0] for p in steering_series])
    v = np.degrees(np.array([p[1] for p in steering_series]))
    dt = np.diff(t)
    dv = np.diff(v)
    valid = dt > 1e-4
    rate = np.zeros_like(dv)
    rate[valid] = dv[valid] / dt[valid]
    t_mid = t[:-1] + dt / 2.0
    return t_mid[valid], rate[valid], v[:-1][valid]


def empirical_r_max_deg_s(rate_series, percentile=99.5):
    """複数ログをプールした実測レートの上位percentileを、経験的な飽和上限の
    代理値として返す(Stage1.5方針: 推測せず実測)。プラトー検出(上位1%内の
    レンジが小さい=頭打ち)も補助情報として返す。"""
    abs_rate = np.abs(rate_series)
    if len(abs_rate) == 0:
        return None
    p_val = float(np.percentile(abs_rate, percentile))
    p_max = float(np.max(abs_rate))
    top1 = abs_rate[abs_rate >= np.percentile(abs_rate, 99.0)]
    plateau = bool(len(top1) > 5 and (np.max(top1) - np.min(top1)) < 0.05 * max(p_val, 1e-6))
    return {'p_value_deg_s': p_val, 'max_deg_s': p_max, 'plateau_detected': plateau,
            'percentile': percentile, 'n_samples': len(abs_rate)}


# ---------------------------------------------------------------------------
# κ'(真値)・swing(窓内max-min、176/177節版との比較用)
# ---------------------------------------------------------------------------

def compute_kappa_prime_true(s_m, kappa):
    """弧長ベース中央差分でκ'=dκ/ds [1/m^2] を計算する。両端は片側差分
    (周回コースのラップアラウンド継ぎ目は全waypoint数に対し2点のみのため、
    簡略化のため中央差分の対象から外す)。"""
    n = len(kappa)
    kp = np.zeros(n)
    if n < 3:
        return kp
    kp[1:-1] = (kappa[2:] - kappa[:-2]) / (s_m[2:] - s_m[:-2])
    kp[0] = (kappa[1] - kappa[0]) / max(s_m[1] - s_m[0], 1e-6)
    kp[-1] = (kappa[-1] - kappa[-2]) / max(s_m[-1] - s_m[-2], 1e-6)
    return kp


def compute_kappa_swing_window(kappa, window=DEFAULT_SWING_LOOKAHEAD_WP):
    """既存R[delta]スケジュール(177節、mpc_controller.py:7119-7134)と同じ
    「前方window点のmax-min」定義。真のκ'との相関比較用(176節相関がκ'真値
    でも成立するかの確認に使う)。周回コースとして循環窓を取る。"""
    n = len(kappa)
    swing = np.zeros(n)
    for i in range(n):
        idx = [(i + j) % n for j in range(window + 1)]
        vals = kappa[idx]
        swing[i] = vals.max() - vals.min()
    return swing


# ---------------------------------------------------------------------------
# ログ読み込み: [LOC-XCHECK]全区間(wp範囲制限なし)
# ---------------------------------------------------------------------------

def read_full_loc_xcheck(log_path):
    rows = []
    with open(log_path, errors='replace') as f:
        for line in f:
            m = LOC_XCHECK_RE.search(line)
            if not m:
                continue
            t_s, wp_s, kappa_s, ekf_ey_s, gnss_ey_s, v_s, ot_s = m.groups()
            rows.append({'t': float(t_s), 'wp': int(wp_s), 'kappa_log': float(kappa_s),
                         'ekf_ey': float(ekf_ey_s), 'gnss_ey': float(gnss_ey_s),
                         'v': float(v_s), 'ot': ot_s})
    rows.sort(key=lambda r: r['t'])
    return rows


def assign_wp_at_times(sample_times, loc_rows):
    """各サンプル時刻に、時刻的に最も近い[LOC-XCHECK]サンプルのwp_idを割り
    当てる(~4Hzの疎な系列への最近傍マッチング)。"""
    times = [r['t'] for r in loc_rows]
    wps = [r['wp'] for r in loc_rows]
    n = len(times)
    result = np.zeros(len(sample_times), dtype=int)
    for i, t in enumerate(sample_times):
        idx = bisect.bisect_left(times, t)
        candidates = [j for j in (idx - 1, idx) if 0 <= j < n]
        if not candidates:
            result[i] = -1
            continue
        best = min(candidates, key=lambda j: abs(times[j] - t))
        result[i] = wps[best]
    return result


def interp_ekf_ey_at_times(sample_times, loc_rows):
    times = np.array([r['t'] for r in loc_rows])
    ey = np.array([r['ekf_ey'] for r in loc_rows])
    return np.interp(sample_times, times, ey)


# ---------------------------------------------------------------------------
# 窓別集計(4層分析+連続回帰を同一の窓テーブルから導出)
# ---------------------------------------------------------------------------

def build_window_table(s_m, kappa_prime, wp_spacing_m, sample_wp, sample_angle_deg,
                        sample_wp_for_rate, sample_rate_deg_s, sample_v_mps, wheel_base_m,
                        window_wp=DEFAULT_WINDOW_WP, r_max_deg_s=None, min_samples=20):
    """waypointをwindow_wp個ずつの窓(循環)にまとめ、各窓について:
      - kappa_prime_center: 窓中心のκ'(真値)
      - predicted_rate_deg_s: |κ'|·v_mean·L_wb(理論必要レート、rad/s→deg/s換算)
      - wobble_std_deg: 窓内の操舵角std(180節と同じ定義の局所蛇行指標)
      - saturation_rate: 窓内で|実測レート|>=r_max*SATURATION_FRACTIONの割合
    を計算する。min_samples未満の窓はスキップする(短時間ログでのデータ不足対策)。
    sample_wp(角度系列N点向け)とsample_wp_for_rate(レート系列N-1点向け、rate_t自身の
    時刻でassign_wp_at_timesした結果)は別々に渡す——レートはt_mid(隣接角度の中点)で
    定義されるため、角度系列のwp割り当てをそのまま流用すると1点分ずれる。"""
    n_wp = len(kappa_prime)
    n_windows = max(1, n_wp // window_wp)
    rows = []
    rad2deg = 180.0 / np.pi
    for w in range(n_windows):
        wp_lo = w * window_wp
        wp_hi = min(wp_lo + window_wp, n_wp)
        center_idx = (wp_lo + wp_hi) // 2
        kp_center = float(kappa_prime[center_idx])
        s_center = float(s_m[center_idx])
        mask = (sample_wp >= wp_lo) & (sample_wp < wp_hi)
        n_samp = int(np.sum(mask))
        if n_samp < min_samples:
            continue
        v_mean = float(np.mean(sample_v_mps[mask]))
        wobble_std = float(np.std(sample_angle_deg[mask]))
        predicted_rate = abs(kp_center) * wheel_base_m * v_mean * rad2deg
        rate_mask = (sample_wp_for_rate >= wp_lo) & (sample_wp_for_rate < wp_hi)
        sat_rate = None
        if r_max_deg_s is not None and np.any(rate_mask):
            rates_in_window = np.abs(sample_rate_deg_s[rate_mask])
            if len(rates_in_window):
                sat_rate = float(np.mean(rates_in_window >= SATURATION_FRACTION * r_max_deg_s))
        rows.append({
            'window': w, 'wp_lo': wp_lo, 'wp_hi': wp_hi, 's_center_m': s_center,
            'kappa_prime_center': kp_center, 'v_mean_mps': v_mean, 'n_samples': n_samp,
            'predicted_rate_deg_s': predicted_rate, 'wobble_std_deg': wobble_std,
            'saturation_rate': sat_rate,
        })
    return rows


def regress_component_a_b(window_rows):
    """局所蛇行指標(wobble_std_deg)をpredicted_rate_deg_s(|κ'|·v·L_wb)へ
    原点通過回帰(act = a*pred)する。R²・残差(=成分B相当)を窓ごとに返す。"""
    x = np.array([r['predicted_rate_deg_s'] for r in window_rows])
    y = np.array([r['wobble_std_deg'] for r in window_rows])
    if len(x) < 5 or np.sum(x ** 2) < 1e-9:
        return None
    slope = float(np.sum(x * y) / np.sum(x ** 2))
    y_pred = slope * x
    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else None
    for r, yp, res in zip(window_rows, y_pred, resid):
        r['wobble_pred_deg'] = float(yp)
        r['residual_deg'] = float(res)
    return {'slope': slope, 'r2': r2, 'n_windows': len(x),
            'component_a_var_frac': (1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else None}


def bucket_by_swing_quartile(window_rows):
    """window_rowsを|kappa_prime_center|の四分位で4層(直線=Q1/低/中/高)に
    分け、層別の飽和率・wobble_std・残差平均を集計する。"""
    if not window_rows:
        return []
    mags = np.array([abs(r['kappa_prime_center']) for r in window_rows])
    qs = np.percentile(mags, [25, 50, 75])
    labels = ['直線(Q1)', '低swing(Q2)', '中swing(Q3)', '高swing(Q4)']
    buckets = {lab: [] for lab in labels}
    for r, m in zip(window_rows, mags):
        if m <= qs[0]:
            buckets[labels[0]].append(r)
        elif m <= qs[1]:
            buckets[labels[1]].append(r)
        elif m <= qs[2]:
            buckets[labels[2]].append(r)
        else:
            buckets[labels[3]].append(r)
    out = []
    for lab in labels:
        rows = buckets[lab]
        if not rows:
            out.append({'label': lab, 'n_windows': 0})
            continue
        sat = [r['saturation_rate'] for r in rows if r['saturation_rate'] is not None]
        resid = [r.get('residual_deg') for r in rows if r.get('residual_deg') is not None]
        out.append({
            'label': lab, 'n_windows': len(rows),
            'mean_abs_kappa_prime': float(np.mean([abs(r['kappa_prime_center']) for r in rows])),
            'mean_saturation_rate': float(np.mean(sat)) if sat else None,
            'mean_wobble_std_deg': float(np.mean([r['wobble_std_deg'] for r in rows])),
            'mean_residual_deg': float(np.mean(resid)) if resid else None,
        })
    return out


# ---------------------------------------------------------------------------
# 時刻相関: 飽和イベント直後のekf_ey拡大
# ---------------------------------------------------------------------------

def time_correlation_saturation_to_ey(rate_t, rate_deg_s, r_max_deg_s, loc_rows, lag_s=LAG_S):
    if r_max_deg_s is None or len(rate_t) == 0:
        return None
    saturated = np.abs(rate_deg_s) >= SATURATION_FRACTION * r_max_deg_s
    if not np.any(saturated) or np.all(saturated):
        return {'n_saturated': int(np.sum(saturated)), 'n_total': len(saturated),
                'verdict': 'データ不足(全飽和/無飽和)'}
    ey_at_lag = interp_ekf_ey_at_times(rate_t + lag_s, loc_rows)
    mean_ey_sat = float(np.mean(np.abs(ey_at_lag[saturated])))
    mean_ey_non = float(np.mean(np.abs(ey_at_lag[~saturated])))
    ratio = mean_ey_sat / mean_ey_non if mean_ey_non > 1e-6 else None
    return {'n_saturated': int(np.sum(saturated)), 'n_total': len(saturated),
            'mean_abs_ey_after_saturation': mean_ey_sat,
            'mean_abs_ey_after_non_saturation': mean_ey_non, 'ratio': ratio}


# ---------------------------------------------------------------------------
# 直線区間の量子化ノイズ検査
# ---------------------------------------------------------------------------

def straight_quantization_check(s_m, kappa_prime, straight_wp_idx):
    if len(straight_wp_idx) < 3:
        return None
    kp_rms = float(np.sqrt(np.mean(kappa_prime[straight_wp_idx] ** 2)))
    ds = np.diff(s_m)
    valid_idx = straight_wp_idx[straight_wp_idx < len(ds)]
    ds_straight = ds[valid_idx]
    spacing_cv = (float(np.std(ds_straight) / np.mean(ds_straight))
                  if len(ds_straight) and np.mean(ds_straight) > 1e-9 else None)
    return {'n_straight_wp': len(straight_wp_idx), 'kappa_prime_rms': kp_rms,
            'spacing_cv': spacing_cv}


# ---------------------------------------------------------------------------
# ホライズン机上検査(報告のみ、実装なし)
# ---------------------------------------------------------------------------

def horizon_desk_check(N, wp_spacing_m, v_kmh_list, tau_s=0.19, settle_s=2.5):
    results = []
    for v_kmh in v_kmh_list:
        v_mps = v_kmh / 3.6
        horizon_time_s = N * wp_spacing_m / v_mps
        margin_s = horizon_time_s - (tau_s + settle_s)
        results.append({'v_kmh': v_kmh, 'horizon_time_s': horizon_time_s,
                         'required_s': tau_s + settle_s, 'margin_s': margin_s})
    return results


# ---------------------------------------------------------------------------
# PSD: 直線 vs コーナー(既存analyze_steering_psdの汎用ラベル機構を再利用)
# ---------------------------------------------------------------------------

def smooth_short_label_runs(label_series, min_run_s):
    """(t, label)系列で、min_run_s未満しか続かない短い切り替わり(位置ベース
    ラベルが数mごとにstraight/cornerを行き来することで生じる断片化)を、
    直前の系列の続きへ吸収して均す。既存PSD分析(analyze_steering_psd.py)の
    セグメント最低長フィルタ(既定8秒)が、位置ベースラベルの頻繁な切り替えで
    ほぼ全区間を除外してしまう(「データ不足」)問題への対処。判定ロジック
    (straight_wp_set等)自体は変更せず、PSD用のラベル系列にのみ適用する。"""
    if not label_series:
        return label_series
    runs = []
    cur_label, cur_start_idx = label_series[0][1], 0
    for i in range(1, len(label_series) + 1):
        if i == len(label_series) or label_series[i][1] != cur_label:
            runs.append((cur_label, cur_start_idx, i))
            if i < len(label_series):
                cur_label, cur_start_idx = label_series[i][1], i
    smoothed = [lab for _, lab in label_series]
    for lab, s, e in runs:
        run_duration = label_series[e - 1][0] - label_series[s][0]
        if run_duration < min_run_s and s > 0:
            prev_label = smoothed[s - 1]
            for i in range(s, e):
                smoothed[i] = prev_label
    return [(t, lab) for (t, _), lab in zip(label_series, smoothed)]


def straight_vs_corner_psd(steering_series, loc_rows, straight_wp_set, sample_hz=40.0,
                            min_segment_s=MIN_SEGMENT_S, label_smooth_s=3.0):
    label_series = [(r['t'], 'straight' if r['wp'] in straight_wp_set else 'corner')
                     for r in loc_rows]
    label_series = smooth_short_label_runs(label_series, label_smooth_s)
    times = [t for t, _ in label_series]
    labels = [lab for _, lab in label_series]

    def assign(sample_times):
        out = []
        for t in sample_times:
            idx = bisect.bisect_right(times, t) - 1
            out.append(labels[idx] if idx >= 0 else None)
        return out

    steering_labels = assign([t for t, _ in steering_series])
    raw_segments = segment_by_state(steering_series, steering_labels)
    segments, excluded_count, excluded_duration = filter_min_length(raw_segments, min_segment_s)
    nperseg = int(min_segment_s * sample_hz)
    per_label = {}
    for lab in ('straight', 'corner'):
        segs = [pts for s, pts in segments if s == lab]
        psds, durations = [], 0.0
        for pts in segs:
            grid, resampled = resample_uniform(pts, sample_hz)
            if grid is None:
                continue
            freqs, pxx = welch_psd(resampled, sample_hz, nperseg)
            psds.append(pxx)
            durations += pts[-1][0] - pts[0][0]
        if not psds:
            per_label[lab] = None
            continue
        freqs_ref, _ = welch_psd(np.zeros(nperseg), sample_hz, nperseg)
        avg_pxx = np.mean(np.stack(psds), axis=0)
        per_label[lab] = {
            'limit_cycle_power': band_power(freqs_ref, avg_pxx, LIMIT_CYCLE_BAND),
            'n_segments': len(psds), 'total_duration_s': durations,
        }
    return per_label, excluded_count, excluded_duration


# ---------------------------------------------------------------------------
# ログ単位の分析本体
# ---------------------------------------------------------------------------

def analyze_bag(log_path, bag_path, label, s_m, kappa, kappa_prime, kappa_swing_window,
                 wp_spacing_m, wheel_base_m, straight_wp_idx, straight_wp_set,
                 window_wp=DEFAULT_WINDOW_WP, r_max_deg_s=None):
    steering = read_steering_series(bag_path)
    loc_rows = read_full_loc_xcheck(log_path)
    if not steering or not loc_rows:
        return {'label': label, 'error': 'データ不足(steering command or LOC-XCHECK無し)'}

    rate_t, rate_deg_s, angle_at_rate_t = compute_steering_rate_series(steering)
    angle_t = np.array([p[0] for p in steering])
    angle_deg = np.degrees(np.array([p[1] for p in steering]))
    v_series = read_speed_series(bag_path)

    sample_wp = assign_wp_at_times(angle_t, loc_rows)
    sample_wp_for_rate = assign_wp_at_times(rate_t, loc_rows)
    sample_v = np.interp(angle_t, [p[0] for p in v_series], [p[1] for p in v_series]) \
        if v_series else np.zeros_like(angle_t)

    window_rows = build_window_table(
        s_m, kappa_prime, wp_spacing_m, sample_wp, angle_deg, sample_wp_for_rate, rate_deg_s,
        sample_v, wheel_base_m, window_wp=window_wp, r_max_deg_s=r_max_deg_s)
    regression = regress_component_a_b(window_rows) if window_rows else None
    buckets = bucket_by_swing_quartile(window_rows) if window_rows else []

    time_corr = time_correlation_saturation_to_ey(rate_t, rate_deg_s, r_max_deg_s, loc_rows)

    quant_check = straight_quantization_check(s_m, kappa_prime, straight_wp_idx)

    psd_by_label, psd_excl_count, psd_excl_dur = straight_vs_corner_psd(
        steering, loc_rows, straight_wp_set)

    # 176節相関の再現確認(κ'真値・窓swingの双方)
    kp_at_sample = kappa_prime[np.clip(sample_wp, 0, len(kappa_prime) - 1)]
    swing_at_sample = kappa_swing_window[np.clip(sample_wp, 0, len(kappa_swing_window) - 1)]
    ey_at_sample = interp_ekf_ey_at_times(angle_t, loc_rows)
    corr_kp_true = (float(np.corrcoef(np.abs(kp_at_sample), np.abs(ey_at_sample))[0, 1])
                    if len(kp_at_sample) > 2 else None)
    corr_swing_window = (float(np.corrcoef(swing_at_sample, np.abs(ey_at_sample))[0, 1])
                         if len(swing_at_sample) > 2 else None)

    overall_std_deg = float(np.std(angle_deg))

    return {
        'label': label, 'log_path': log_path, 'bag_path': bag_path,
        'n_steering_samples': len(steering), 'n_loc_rows': len(loc_rows),
        'overall_std_deg': overall_std_deg,
        'window_rows': window_rows, 'regression': regression, 'buckets': buckets,
        'time_correlation': time_corr, 'quantization_check': quant_check,
        'psd_by_label': psd_by_label, 'psd_excluded_count': psd_excl_count,
        'psd_excluded_duration_s': psd_excl_dur,
        'corr_kappa_prime_true_vs_ey': corr_kp_true,
        'corr_swing_window_vs_ey': corr_swing_window,
    }


# ---------------------------------------------------------------------------
# レポート整形
# ---------------------------------------------------------------------------

def format_report(results, calib_analytical, calib_empirical, r_max_used, cfg, horizon_results):
    lines = []
    lines.append('# レート飽和検査(Phase 1改訂版)レポート')
    lines.append('')
    lines.append('## 0. 単位較正')
    lines.append(f"config解析的候補: A(rawそのまま)={calib_analytical['candidate_a_deg_s']:.1f}°/s "
                 f"(コメント値との差{calib_analytical['diff_a_frac']*100:.1f}%) / "
                 f"B(raw×gain)={calib_analytical['candidate_b_deg_s']:.1f}°/s "
                 f"(差{calib_analytical['diff_b_frac']*100:.1f}%) "
                 f"→ コメント値({calib_analytical['comment_value_deg_s']:.0f}°/s)に近いのは"
                 f"{calib_analytical['closer_candidate']}")
    lines.append(
        f"**MPC.py `_rate_bounds()`の実装確認により、この制約は実際にはu[1]=κ(曲率)の"
        f"レートにかかっており、config commentの「rad/s」ラベルと単位が食い違う可能性がある。"
        f"上記の解析的候補はあくまで参考情報であり、以降の飽和率計算には次の経験的較正値を使う。**")
    if calib_empirical:
        lines.append(
            f"**経験的較正(全ログプール、p{calib_empirical['percentile']:.1f}): "
            f"r_max={calib_empirical['p_value_deg_s']:.1f}°/s "
            f"(最大観測{calib_empirical['max_deg_s']:.1f}°/s、"
            f"プラトー検出={'あり' if calib_empirical['plateau_detected'] else 'なし'}、"
            f"n={calib_empirical['n_samples']})**")
    else:
        lines.append('経験的較正: データ不足')
    lines.append(f"以降の飽和率計算に使用するr_max: {r_max_used:.1f}°/s "
                 f"(飽和判定閾値={SATURATION_FRACTION*100:.0f}%={r_max_used*SATURATION_FRACTION:.1f}°/s)")
    lines.append('')

    lines.append('## 1. ログ別サマリ')
    lines.append('| ログ | 操舵std(180節と同定義) | 回帰R²(成分A寄与率) | 成分A/B判定 |')
    lines.append('|---|---|---|---|')
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['label']} | データ不足 | - | - |")
            continue
        reg = r['regression']
        if reg is None or reg['r2'] is None:
            lines.append(f"| {r['label']} | {r['overall_std_deg']:.2f}° | 判定不能 | - |")
            continue
        frac = reg['component_a_var_frac']
        verdict = ('成分A優勢' if frac is not None and frac >= 0.5
                   else '成分B(κ非依存)が有意' if frac is not None and frac < 0.3
                   else '中間')
        lines.append(f"| {r['label']} | {r['overall_std_deg']:.2f}° | "
                     f"{frac*100:.1f}% (n_windows={reg['n_windows']}) | {verdict} |")
    lines.append('')

    lines.append('## 2. 4層(直線/低/中/高swing)分析(180節再現確認の核心)')
    for r in results:
        if 'error' in r or not r.get('buckets'):
            continue
        lines.append(f"### {r['label']}")
        lines.append('| 層 | 窓数 | 平均\\|κ\'\\| | 飽和率 | wobble_std[deg] | 残差(成分B)[deg] |')
        lines.append('|---|---|---|---|---|---|')
        for b in r['buckets']:
            if b['n_windows'] == 0:
                lines.append(f"| {b['label']} | 0 | - | - | - | - |")
                continue
            sat = f"{b['mean_saturation_rate']*100:.2f}%" if b['mean_saturation_rate'] is not None else 'N/A'
            resid = f"{b['mean_residual_deg']:.3f}" if b['mean_residual_deg'] is not None else 'N/A'
            lines.append(f"| {b['label']} | {b['n_windows']} | {b['mean_abs_kappa_prime']:.4f} | "
                         f"{sat} | {b['mean_wobble_std_deg']:.2f} | {resid} |")
        straight = next((b for b in r['buckets'] if b['label'] == '直線(Q1)'), None)
        if straight and straight['n_windows'] > 0:
            lines.append(f"**直線(Q1)層の飽和率は"
                         f"{(straight['mean_saturation_rate'] or 0)*100:.2f}%でありながら、"
                         f"wobble_stdは{straight['mean_wobble_std_deg']:.2f}°"
                         f"({'高swing層と同水準=180節の一様悪化を再現' if straight['mean_wobble_std_deg'] > 0.5 * r['overall_std_deg'] else '他層より明確に低い'})。**")
        lines.append('')

    lines.append('## 3. 176節相関の再現確認(κ\'真値 vs 窓swing)')
    lines.append('| ログ | \\|κ\'真値\\| vs \\|ekf_ey\\|相関 | 窓swing vs \\|ekf_ey\\|相関(177節式) |')
    lines.append('|---|---|---|')
    for r in results:
        if 'error' in r:
            continue
        a = r['corr_kappa_prime_true_vs_ey']
        b = r['corr_swing_window_vs_ey']
        a_str = f"{a:.3f}" if a is not None else 'N/A'
        b_str = f"{b:.3f}" if b is not None else 'N/A'
        lines.append(f"| {r['label']} | {a_str} | {b_str} |")
    lines.append('')

    lines.append('## 4. 時刻相関(飽和イベント直後のekf_ey拡大)')
    lines.append(f"lag={LAG_S}s")
    lines.append('| ログ | 飽和サンプル数/全体 | 飽和後\\|ekf_ey\\| | 非飽和後\\|ekf_ey\\| | 比 |')
    lines.append('|---|---|---|---|---|')
    for r in results:
        if 'error' in r or r.get('time_correlation') is None:
            continue
        tc = r['time_correlation']
        if 'mean_abs_ey_after_saturation' not in tc:
            lines.append(f"| {r['label']} | {tc.get('n_saturated', 0)}/{tc.get('n_total', 0)} | "
                         f"{tc.get('verdict', 'N/A')} | - | - |")
            continue
        ratio_s = f"{tc['ratio']:.2f}" if tc['ratio'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {tc['n_saturated']}/{tc['n_total']} | "
                     f"{tc['mean_abs_ey_after_saturation']:.3f}m | "
                     f"{tc['mean_abs_ey_after_non_saturation']:.3f}m | {ratio_s} |")
    lines.append('')

    lines.append('## 5. 直線区間の量子化ノイズ検査')
    lines.append('| ログ | 直線wp数 | κ\'のRMS(直線内) | 点間隔CV |')
    lines.append('|---|---|---|---|')
    for r in results:
        if 'error' in r or r.get('quantization_check') is None:
            continue
        q = r['quantization_check']
        cv = f"{q['spacing_cv']:.4f}" if q['spacing_cv'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {q['n_straight_wp']} | {q['kappa_prime_rms']:.5f} | {cv} |")
    lines.append('')

    lines.append('## 6. 直線 vs コーナー PSD(限界サイクル帯0.5-0.9Hz)')
    lines.append('| ログ | 直線帯パワー | コーナー帯パワー | 直線/コーナー比 |')
    lines.append('|---|---|---|---|')
    for r in results:
        if 'error' in r:
            continue
        pl = r.get('psd_by_label') or {}
        s, c = pl.get('straight'), pl.get('corner')
        if s is None or c is None:
            lines.append(f"| {r['label']} | データ不足 | データ不足 | - |")
            continue
        ratio = (s['limit_cycle_power'] / c['limit_cycle_power']
                 if c['limit_cycle_power'] > 1e-9 else None)
        lines.append(f"| {r['label']} | {s['limit_cycle_power']:.6f}(n={s['n_segments']}) | "
                     f"{c['limit_cycle_power']:.6f}(n={c['n_segments']}) | "
                     f"{f'{ratio:.2f}' if ratio is not None else 'N/A'} |")
    lines.append('')

    lines.append('## 7. ホライズン机上検査(報告のみ、実装しない)')
    lines.append(f"N={cfg['N']} wp_spacing≈使用ログのwaypoint間隔で算出")
    lines.append('| v_max | ホライズン先読み時間 | 必要時間(tau+整定) | 余裕 |')
    lines.append('|---|---|---|')
    for h in horizon_results:
        lines.append(f"| {h['v_kmh']:.0f}km/h | {h['horizon_time_s']:.2f}s | "
                     f"{h['required_s']:.2f}s | {h['margin_s']:+.2f}s "
                     f"{'(不足の疑い)' if h['margin_s'] < 0 else ''} |")
    lines.append('')

    lines.append('## 8. 更新後リスク予測(r_max×0.75、金曜AWSIM更新想定)')
    r_max_friday = r_max_used * FRIDAY_R_MAX_SCALE
    lines.append(f"想定r_max: {r_max_friday:.1f}°/s(現行{r_max_used:.1f}°/sの75%)")
    lines.append('| ログ | 現行飽和率(全窓平均) | 更新後想定飽和率 |')
    lines.append('|---|---|---|')
    for r in results:
        if 'error' in r or not r.get('window_rows'):
            continue
        cur_rates = [w['saturation_rate'] for w in r['window_rows'] if w['saturation_rate'] is not None]
        cur_mean = float(np.mean(cur_rates)) if cur_rates else None
        lines.append(f"| {r['label']} | "
                     f"{f'{cur_mean*100:.2f}%' if cur_mean is not None else 'N/A'} | "
                     f"(現行の閾値を{FRIDAY_R_MAX_SCALE}倍して再計算するには別runが必要。"
                     f"参考: 閾値低下により飽和率は単調に上昇する) |")
    lines.append('')

    lines.append('## 9. R[delta]スケジュール(177/178節)の記述に関する注記')
    lines.append('177/178節の「無罪放免」判定はCPU予算がほぼ100%張り付いたdev3 3台自己対戦'
                 '環境(超過率30-45%常態化)での処理落ち原因切り分けテストであり、Part Cのような'
                 'クリーンな単体A/Bではない。「元々の蛇行(対戦車なし、v_max=20)にR[delta]'
                 'スケジュールが効かない」ことの直接証拠ではなく、確度は一段下げて扱うこと。'
                 'クリーン環境(対戦車なし・v_max=20、boost=0 vs 400)での再検証はローカル実走行'
                 'が必要なため本ツールのスコープ外——別途ユーザーへ依頼する。')
    lines.append('')

    return '\n'.join(lines)


def format_gate_verdict(results):
    lines = ['## ゲート判定(成分分解型)']
    for r in results:
        if 'error' in r or r.get('regression') is None or r['regression']['r2'] is None:
            lines.append(f"- {r['label']}: 判定不能(データ不足)")
            continue
        frac = r['regression']['component_a_var_frac']
        straight = next((b for b in r['buckets'] if b['label'] == '直線(Q1)'), None)
        straight_wobble = straight['mean_wobble_std_deg'] if straight and straight['n_windows'] else None
        if frac is not None and frac >= 0.5:
            verdict = '成分A支持(Phase 2/3へ進んでよい。期待効果は成分A寄与率までに限定)'
        elif straight_wobble is not None and straight_wobble > 0.5 * r['overall_std_deg']:
            verdict = ('成分B実在(直線でも高水準のwobble_std) → 別調査を起票'
                       '(量子化ノイズなら7節参照、自己励起系ならAXIS06本体/Q[e_y]×tau調査へ)')
        elif frac is not None and frac < 0.3:
            verdict = 'A寄与3割未満 → Phase 2/3の投資対効果を人間へ提示し進退を相談'
        else:
            verdict = '中間(明確な判定不能、追加データ推奨)'
        lines.append(f"- {r['label']}: 成分A寄与率={frac*100:.1f}% → {verdict}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='レート飽和検査(蛇行対策Phase 1改訂版)')
    parser.add_argument('--waypoints', required=True, help='waypoint CSV(kappa_radpm列を含む)')
    parser.add_argument('--config', required=True, help='config.yaml')
    parser.add_argument('logs', nargs='+', help='log:bag:label の形式で複数指定')
    parser.add_argument('--window-wp', type=int, default=DEFAULT_WINDOW_WP)
    parser.add_argument('--v-max-kmh', type=float, nargs='+', default=[15.0, 20.0, 35.0],
                         help='ホライズン机上検査対象のv_maxリスト')
    args = parser.parse_args(argv)

    cfg = load_mpc_config(args.config)
    calib_analytical = calibrate_r_max_analytical(cfg)

    s_m, kappa = load_waypoint_kappa(args.waypoints)
    kappa_prime = compute_kappa_prime_true(s_m, kappa)
    kappa_swing_window = compute_kappa_swing_window(kappa)
    wp_spacing_m = float(np.mean(np.diff(s_m))) if len(s_m) > 1 else 1.0

    straight_thr = np.percentile(np.abs(kappa_prime), 25)
    straight_wp_idx = np.where(np.abs(kappa_prime) <= straight_thr)[0]
    straight_wp_set = set(int(i) for i in straight_wp_idx)

    specs = []
    for spec in args.logs:
        log_path, bag_path, label = spec.split(':', 2)
        specs.append((log_path, bag_path, label))

    # 経験的r_max較正: 全ログをプールして算出
    all_rates = []
    for log_path, bag_path, label in specs:
        print(f"読み込み中(較正用): {label}", file=sys.stderr)
        try:
            steering = read_steering_series(bag_path)
        except Exception as e:
            print(f"  スキップ({e})", file=sys.stderr)
            continue
        if not steering:
            continue
        _, rate_deg_s, _ = compute_steering_rate_series(steering)
        all_rates.append(rate_deg_s)
    pooled_rates = np.concatenate(all_rates) if all_rates else np.array([])
    calib_empirical = empirical_r_max_deg_s(pooled_rates) if len(pooled_rates) else None
    r_max_used = (calib_empirical['p_value_deg_s'] if calib_empirical
                  else calib_analytical['candidate_b_deg_s'])

    results = []
    for log_path, bag_path, label in specs:
        print(f"分析中: {label}", file=sys.stderr)
        try:
            results.append(analyze_bag(
                log_path, bag_path, label, s_m, kappa, kappa_prime, kappa_swing_window,
                wp_spacing_m, cfg['wheel_base_m'], straight_wp_idx, straight_wp_set,
                window_wp=args.window_wp, r_max_deg_s=r_max_used))
        except Exception as e:
            results.append({'label': label, 'error': str(e)})

    horizon_results = horizon_desk_check(cfg['N'], wp_spacing_m, args.v_max_kmh)

    print(format_report(results, calib_analytical, calib_empirical, r_max_used, cfg,
                         horizon_results))
    print(format_gate_verdict(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
