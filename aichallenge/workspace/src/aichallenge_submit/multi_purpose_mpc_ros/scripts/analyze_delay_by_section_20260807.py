#!/usr/bin/env python3
"""analyze_delay_by_section_20260807.py

トラックのセクション(等分割 or コーナー単位)ごとに実効遅延(L_eff)を
推定する(2026-08-07)。既存の`analyze_actuator_delay.py --mode yawrate`
は全区間平均のFOPDT L・tauしか出せないため、区間限定版を新設する。

**方式**: bagの操舵指令(steering_cmd)とヨーレート応答(yaw_rate)を、
対応するautoware.log(テキストログ)の[LOC-XCHECK] wp情報でタイムスタンプ
→wpへマッピングし、指定セクション内の時刻範囲のみを抽出してから
クロス相関で遅延(ラグ)を推定する。FOPDTグリッドサーチより少ない
サンプル数でも安定するクロス相関ベースの簡易推定を採用する
(セクション単位ではサンプル数が全体解析より少なくなるため)。

制御には一切関与しない、オフライン分析専用ツール。

使い方:
    python3 analyze_delay_by_section_20260807.py <bag_path> <log_path> \
        [--sections 8] [--corners]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_actuator_delay import read_yaw_rate_series  # noqa: E402
from analyze_steering_psd import read_steering_series  # noqa: E402

LOC_XCHECK_PAT = re.compile(
    r"\[(\d+\.\d+)\].*\[LOC-XCHECK\] wp=(\d+) kappa=(-?[\d.]+)"
)


def read_wp_series(log_path):
    """(timestamp, wp, kappa)のリストを時刻昇順で返す。"""
    rows = []
    for ln in open(log_path, errors="replace"):
        m = LOC_XCHECK_PAT.search(ln)
        if m:
            rows.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
    rows.sort(key=lambda r: r[0])
    return rows


def detect_corner_sections(wp_rows, kappa_threshold=0.08, min_width=3):
    """|kappa|>閾値の連続wp区間をコーナーとして抽出する(最初に見た値を使用)。"""
    seen = {}
    for _t, wp, k in wp_rows:
        if wp not in seen:
            seen[wp] = k
    wps = sorted(seen.keys())
    if not wps:
        return []
    in_corner = False
    start = None
    corners = []
    for wp in range(min(wps), max(wps) + 1):
        k = seen.get(wp, 0.0)
        if abs(k) > kappa_threshold and not in_corner:
            in_corner = True
            start = wp
        elif abs(k) <= kappa_threshold and in_corner:
            in_corner = False
            if wp - 1 - start + 1 >= min_width:
                corners.append((start, wp - 1))
    if in_corner and max(wps) - start + 1 >= min_width:
        corners.append((start, max(wps)))
    return corners


def equal_sections(wp_rows, n_sections):
    wps = [r[1] for r in wp_rows]
    lo, hi = min(wps), max(wps)
    step = (hi - lo + 1) / n_sections
    return [(int(lo + i * step), int(lo + (i + 1) * step) - 1) for i in range(n_sections)]


def time_windows_for_section(wp_rows, wp_lo, wp_hi, margin_s=0.5):
    """セクション内にいた時刻区間を連続塊ごとに返す(周回で複数回訪れる
    場合は複数区間になる)。前後にmargin_sだけ余裕を持たせる。
    wp_lo > wp_hi の場合は周回のラップアラウンド区間(例: wp340-40は
    wp>=340 or wp<=40)として扱う(トラックはcircular=true)。"""
    wrap = wp_lo > wp_hi
    windows = []
    cur_start = None
    prev_t = None
    for t, wp, _k in wp_rows:
        inside = (wp >= wp_lo or wp <= wp_hi) if wrap else (wp_lo <= wp <= wp_hi)
        if inside and cur_start is None:
            cur_start = t
        elif not inside and cur_start is not None:
            windows.append((cur_start - margin_s, prev_t + margin_s))
            cur_start = None
        prev_t = t
    if cur_start is not None:
        windows.append((cur_start - margin_s, prev_t + margin_s))
    return windows


def cross_correlation_lag(cmd_t, cmd_v, act_t, act_v, sample_hz=40.0,
                           lag_candidates_ms=None):
    """cmd/actをsample_hzで再サンプリングし、正規化相互相関が最大となる
    ラグ(ms)を返す。サンプル数が少なくても安定するよう候補ラグの
    グリッドサーチとする(FOPDTの2パラメータ同時推定より軽量)。"""
    if lag_candidates_ms is None:
        lag_candidates_ms = list(range(0, 310, 10))
    if len(cmd_t) < 5 or len(act_t) < 5:
        return None
    t_lo = max(cmd_t[0], act_t[0])
    t_hi = min(cmd_t[-1], act_t[-1])
    if t_hi - t_lo < 0.3:
        return None
    grid = np.arange(t_lo, t_hi, 1.0 / sample_hz)
    if len(grid) < 8:
        return None
    cmd_i = np.interp(grid, cmd_t, cmd_v)
    act_i = np.interp(grid, act_t, act_v)
    cmd_i = cmd_i - np.mean(cmd_i)
    act_i = act_i - np.mean(act_i)
    if np.std(cmd_i) < 1e-6 or np.std(act_i) < 1e-6:
        return None

    best_lag, best_corr = None, -2.0
    dt = 1.0 / sample_hz
    for lag_ms in lag_candidates_ms:
        shift = int(round((lag_ms / 1000.0) / dt))
        if shift >= len(cmd_i):
            continue
        a = cmd_i[:len(cmd_i) - shift] if shift > 0 else cmd_i
        b = act_i[shift:] if shift > 0 else act_i
        n = min(len(a), len(b))
        if n < 5:
            continue
        a, b = a[:n], b[:n]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9:
            continue
        corr = float(np.dot(a, b) / denom)
        if corr > best_corr:
            best_corr, best_lag = corr, lag_ms
    if best_lag is None:
        return None
    return {"lag_ms": best_lag, "corr": best_corr, "n": len(grid)}


def analyze_sections(bag_path, log_path, sections):
    """sections: [(wp_lo, wp_hi, label), ...]"""
    steering_cmd = read_steering_series(bag_path)
    yaw_rate = read_yaw_rate_series(bag_path)
    wp_rows = read_wp_series(log_path)
    cmd_t = [p[0] for p in steering_cmd]
    cmd_v = [np.degrees(p[1]) for p in steering_cmd]
    act_t = [p[0] for p in yaw_rate]
    act_v = [np.degrees(p[1]) for p in yaw_rate]  # rad/s -> deg/s

    results = []
    for wp_lo, wp_hi, label in sections:
        windows = time_windows_for_section(wp_rows, wp_lo, wp_hi)
        lags = []
        corrs = []
        total_n = 0
        for w_lo, w_hi in windows:
            c_idx = [i for i, t in enumerate(cmd_t) if w_lo <= t <= w_hi]
            a_idx = [i for i, t in enumerate(act_t) if w_lo <= t <= w_hi]
            if len(c_idx) < 5 or len(a_idx) < 5:
                continue
            sub_cmd_t = [cmd_t[i] for i in c_idx]
            sub_cmd_v = [cmd_v[i] for i in c_idx]
            sub_act_t = [act_t[i] for i in a_idx]
            sub_act_v = [act_v[i] for i in a_idx]
            r = cross_correlation_lag(sub_cmd_t, sub_cmd_v, sub_act_t, sub_act_v)
            if r is not None:
                lags.append(r["lag_ms"])
                corrs.append(r["corr"])
                total_n += r["n"]
        if lags:
            results.append({
                "label": label, "wp_range": (wp_lo, wp_hi),
                "corr_mean": float(np.mean(corrs)), "corr_min": float(np.min(corrs)),
                "n_windows": len(windows), "n_valid": len(lags),
                "lag_mean_ms": float(np.mean(lags)), "lag_std_ms": float(np.std(lags)),
                "n_samples": total_n,
            })
        else:
            results.append({
                "label": label, "wp_range": (wp_lo, wp_hi),
                "n_windows": len(windows), "n_valid": 0,
                "lag_mean_ms": None, "lag_std_ms": None, "n_samples": 0,
            })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="セクション別実効遅延の推定")
    parser.add_argument("bag_path")
    parser.add_argument("log_path")
    parser.add_argument("--sections", type=int, default=8, help="等分割数(既定8)")
    parser.add_argument("--corners", action="store_true", help="コーナー単位で解析する")
    args = parser.parse_args(argv)

    wp_rows = read_wp_series(args.log_path)
    if not wp_rows:
        print("wp情報が取得できませんでした", file=sys.stderr)
        return 1

    if args.corners:
        corners = detect_corner_sections(wp_rows)
        sections = [(lo, hi, f"corner_wp{lo}-{hi}") for lo, hi in corners]
    else:
        eq = equal_sections(wp_rows, args.sections)
        sections = [(lo, hi, f"section{i+1}_wp{lo}-{hi}") for i, (lo, hi) in enumerate(eq)]

    results = analyze_sections(args.bag_path, args.log_path, sections)
    print(f"{'区間':30s} {'n_valid':>8s} {'lag_mean_ms':>12s} {'lag_std_ms':>11s} {'n_samples':>10s}")
    for r in results:
        if r["lag_mean_ms"] is not None:
            print(f"{r['label']:30s} {r['n_valid']:8d} {r['lag_mean_ms']:12.1f} "
                  f"{r['lag_std_ms']:11.1f} {r['n_samples']:10d}")
        else:
            print(f"{r['label']:30s} {'データ不足':>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
