#!/usr/bin/env python3
"""analyze_ekf_artifact_check.py

蛇行(0.6-0.7Hz)対策・(A)排除検査(2026-08-04、AXIS06本体調査の第一歩)。

Gemini・別Claude両者の提案(仮説「Q[e_y]×tauの相互作用による自己励起的
限界サイクル」の検証設計)を受け、本命実験(Q[e_y]対数スイープ、要ローカル
実走行・1.5日)に着手する前に、既存ログのみで今すぐ判別できる別Claude提案の
(A)排除検査を実装する: 「蛇行は物理的な現象か、EKF推定固有のアーティファクト
か」——これが後者ならQ[e_y]をどう調整しても直らないため、最優先で排除すべき。

予選rosbagには`/awsim/ground_truth/*`(ローカルdev環境のみの診断トピック)が
含まれないため、真の意味でのground truth比較はできない。代わりに、autoware.log
の[LOC-XCHECK]が同時に記録しているekf_ey(EKF融合後の推定)とgnss_ey(生GNSS
測位、EKFのフィルタリングを経ていない独立した測定路)を比較する:
  - 両者が同水準の0.6-0.7Hz帯パワーを持てば、両方の独立した測定路で見えている
    「物理的な」現象である可能性が高い(EKF固有のアーティファクトなら、
    フィルタリングを経ないgnss_eyには出にくいはず)。
  - 位相ロック検査: wobbleがコース上の特定位置に固定された現象(=何らかの
    地形/測位トリガー)か、周回ごとにランダムな位相か(マージナル安定+
    ノイズ励起の自励振動に典型的)を、waypoint整列した周回間相関で調べる。

制御には一切関与しない、オフライン分析専用ツール。

使い方:
    python3 analyze_ekf_artifact_check.py <log1>:<label1> [<log2>:<label2> ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from analyze_steer_rate_saturation import read_full_loc_xcheck  # noqa: E402

LIMIT_CYCLE_BAND = (0.5, 0.9)
SAMPLE_HZ = 4.0  # [LOC-XCHECK]の実測レート(~4Hz)に合わせる
MIN_SEGMENT_S = 15.0  # 4Hzでの周波数分解能確保のため、既存PSDツールより長めの最低長
LAP_WRAP_MARGIN = 0.5  # wpが後退(前回一周のn_wp*0.5超)した箇所をラップ境界とみなす


def resample_uniform(t, v, sample_hz):
    t = np.asarray(t)
    v = np.asarray(v)
    n = int((t[-1] - t[0]) * sample_hz)
    if n < 2:
        return None, None
    grid = t[0] + np.arange(n) / sample_hz
    return grid, np.interp(grid, t, v)


def segment_continuous(t, gap_thr_s=2.0):
    """時刻の大きなギャップ(欠測・周回切れ目等)でセグメントに分割し、
    各セグメントの(開始index, 終了index)を返す。"""
    t = np.asarray(t)
    breaks = np.where(np.diff(t) > gap_thr_s)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(t)]])
    return list(zip(starts, ends))


def band_power_welch(values, sample_hz, band=LIMIT_CYCLE_BAND, min_segment_s=MIN_SEGMENT_S):
    nperseg = int(min_segment_s * sample_hz)
    if len(values) < nperseg:
        return None
    freqs, pxx = signal.welch(values, fs=sample_hz, nperseg=min(nperseg, len(values)))
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return None
    total = float(np.trapezoid(pxx, freqs))
    band_p = float(np.trapezoid(pxx[mask], freqs[mask]))
    # スペクトル集中度(狭帯域=鋭いピークほど1に近い): 帯域内最大値/帯域内平均
    peak_to_mean = float(np.max(pxx[mask]) / np.mean(pxx[mask])) if np.mean(pxx[mask]) > 0 else None
    return {'freqs': freqs, 'pxx': pxx, 'band_power': band_p, 'total_power': total,
            'peak_to_mean': peak_to_mean, 'n_samples': len(values)}


def compare_ekf_vs_gnss(loc_rows):
    """全周回を結合し、[LOC-XCHECK]の時刻ギャップ(欠測)で分割したセグメント
    ごとにPSDを取り、限界サイクル帯パワーをekf_ey・gnss_eyそれぞれ集計する。"""
    t = [r['t'] for r in loc_rows]
    ekf = [r['ekf_ey'] for r in loc_rows]
    gnss = [r['gnss_ey'] for r in loc_rows]
    segs = segment_continuous(t, gap_thr_s=1.0)

    ekf_bands, gnss_bands = [], []
    for s, e in segs:
        if e - s < 5:
            continue
        grid, ekf_r = resample_uniform(t[s:e], ekf[s:e], SAMPLE_HZ)
        if grid is None:
            continue
        _, gnss_r = resample_uniform(t[s:e], gnss[s:e], SAMPLE_HZ)
        r_ekf = band_power_welch(ekf_r, SAMPLE_HZ)
        r_gnss = band_power_welch(gnss_r, SAMPLE_HZ)
        if r_ekf:
            ekf_bands.append(r_ekf)
        if r_gnss:
            gnss_bands.append(r_gnss)

    def summarize(bands):
        if not bands:
            return None
        return {
            'mean_band_power': float(np.mean([b['band_power'] for b in bands])),
            'mean_peak_to_mean': float(np.mean([b['peak_to_mean'] for b in bands
                                                  if b['peak_to_mean'] is not None])),
            'n_segments': len(bands),
        }
    ekf_summary = summarize(ekf_bands)
    gnss_summary = summarize(gnss_bands)
    ratio = None
    if ekf_summary and gnss_summary and ekf_summary['mean_band_power'] > 1e-9:
        ratio = gnss_summary['mean_band_power'] / ekf_summary['mean_band_power']
    return {'ekf': ekf_summary, 'gnss': gnss_summary, 'gnss_to_ekf_ratio': ratio}


def split_laps(loc_rows, n_wp_hint=350):
    """wpの後退(周回境界)でラップに分割する(corner_ringing.pyの周回検出と
    同じ考え方)。"""
    laps = [[]]
    prev_wp = None
    for r in loc_rows:
        if prev_wp is not None and r['wp'] < prev_wp and (prev_wp - r['wp']) > n_wp_hint * 0.5:
            laps.append([])
        laps[-1].append(r)
        prev_wp = r['wp']
    return [lap for lap in laps if len(lap) >= 20]


def _detrend_moving_average(x, window):
    """xから自身の移動平均(周期境界扱い、window点)を差し引いた残差を返す。
    周回内の低周波成分(コーナリングに伴う緩やかなe_y変化=ベースライン)を
    除去し、蛇行帯の振動成分を残す。周回間で平均を取って引く方式(旧実装の
    バグ)とは異なり、各周回を独立に処理するため、位置固定の蛇行パターンが
    複数周回で共通していても消えない(位置固定パターンは「低周波の
    ベースライン」ではなく「振動成分」に属するため、この移動平均には
    ほぼ乗らずに残る)。"""
    n = len(x)
    kernel = np.ones(window) / window
    # 周期境界(周回はループ)としてconvolveする
    x_ext = np.concatenate([x[-window:], x, x[:window]])
    smoothed_ext = np.convolve(x_ext, kernel, mode='same')
    smoothed = smoothed_ext[window:window + n]
    return x - smoothed


def lap_phase_lock_check(loc_rows, n_wp=350, detrend_window=40):
    """周回をwp軸へ整列(共通グリッド0..n_wp-1へ線形補間)し、各周回を個別に
    移動平均でデトレンド(ベースライン除去、_detrend_moving_average参照)した
    上で、周回間の相関を見る。位置固定のトリガーがあれば周回間相関は正に
    有意、ノイズ駆動の自励振動ならほぼ0になる。
    2026-08-04修正: 旧実装は周回間の平均(lap_mean)を各周回から差し引いて
    いたが、蛇行が全周回で同じ位置に固定して現れる場合、まさにその共通成分が
    lap_meanに現れてしまい差し引きで消える(=位置固定パターンを検出できない
    設計バグ、合成データのテストで発覚)。個別デトレンドへ変更しこれを修正。"""
    laps = split_laps(loc_rows, n_wp_hint=n_wp)
    if len(laps) < 3:
        return {'n_laps': len(laps), 'verdict': 'データ不足(周回数<3)'}

    grid = np.arange(n_wp)
    aligned = []
    for lap in laps:
        wp = np.array([r['wp'] for r in lap])
        ey = np.array([r['ekf_ey'] for r in lap])
        order = np.argsort(wp)
        wp_sorted, ey_sorted = wp[order], ey[order]
        wp_u, idx_u = np.unique(wp_sorted, return_index=True)
        if len(wp_u) < n_wp * 0.5:
            continue
        ey_u = ey_sorted[idx_u]
        aligned.append(np.interp(grid, wp_u, ey_u, left=np.nan, right=np.nan))
    if len(aligned) < 3:
        return {'n_laps': len(aligned), 'verdict': 'データ不足(整列後周回数<3)'}

    detrended = []
    for a in aligned:
        if np.any(np.isnan(a)):
            filled = np.where(np.isnan(a), np.nanmean(a), a)
        else:
            filled = a
        detrended.append(_detrend_moving_average(filled, detrend_window))
    residuals = np.stack(detrended)

    corrs = []
    n_laps = residuals.shape[0]
    for i in range(n_laps):
        for j in range(i + 1, n_laps):
            a, b = residuals[i], residuals[j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(valid) < n_wp * 0.5:
                continue
            if np.std(a[valid]) < 1e-6 or np.std(b[valid]) < 1e-6:
                continue
            c = float(np.corrcoef(a[valid], b[valid])[0, 1])
            corrs.append(c)
    if not corrs:
        return {'n_laps': n_laps, 'verdict': 'データ不足(相関算出不可)'}
    mean_corr = float(np.mean(corrs))
    verdict = ('位置固定トリガーの疑い(周回間相関が有意に正)' if mean_corr > 0.3
               else 'ランダム位相(ノイズ駆動の自励振動と整合)' if abs(mean_corr) < 0.15
               else '中間(判定不能)')
    return {'n_laps': n_laps, 'n_pairs': len(corrs), 'mean_lap_to_lap_corr': mean_corr,
            'verdict': verdict}


def format_report(results):
    lines = ['# (A)排除検査: 蛇行は物理現象かEKF推定アーティファクトか', '']
    lines.append('## 1. ekf_ey vs gnss_ey 限界サイクル帯(0.5-0.9Hz)パワー比較')
    lines.append('gnss_eyは生GNSS測位(EKFフィルタを経ない独立測定路)。両者が同水準なら'
                 '物理現象、gnss_eyだけ低ければEKF起因の疑い。')
    lines.append('| ログ | ekf帯パワー | gnss帯パワー | gnss/ekf比 | ekfピーク鋭さ | gnssピーク鋭さ |')
    lines.append('|---|---|---|---|---|---|')
    for r in results:
        c = r['ekf_vs_gnss']
        if c['ekf'] is None or c['gnss'] is None:
            lines.append(f"| {r['label']} | データ不足 | データ不足 | - | - | - |")
            continue
        ratio = f"{c['gnss_to_ekf_ratio']:.2f}" if c['gnss_to_ekf_ratio'] is not None else 'N/A'
        lines.append(f"| {r['label']} | {c['ekf']['mean_band_power']:.5f}(n={c['ekf']['n_segments']}) | "
                     f"{c['gnss']['mean_band_power']:.5f}(n={c['gnss']['n_segments']}) | {ratio} | "
                     f"{c['ekf']['mean_peak_to_mean']:.2f} | {c['gnss']['mean_peak_to_mean']:.2f} |")
    lines.append('')

    lines.append('## 2. 周回間位相ロック検査(wp整列ekf_ey残差の周回間相関)')
    lines.append('相関が有意に正なら位置固定トリガー(地形/測位起因)、ほぼ0ならランダム位相'
                 '(マージナル安定+ノイズ励起の自励振動と整合)。')
    lines.append('| ログ | 周回数 | ペア数 | 平均周回間相関 | 判定 |')
    lines.append('|---|---|---|---|---|')
    for r in results:
        p = r['phase_lock']
        corr = f"{p['mean_lap_to_lap_corr']:.3f}" if 'mean_lap_to_lap_corr' in p else 'N/A'
        pairs = p.get('n_pairs', '-')
        lines.append(f"| {r['label']} | {p['n_laps']} | {pairs} | {corr} | {p['verdict']} |")
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='(A)排除検査: 蛇行の物理性 vs EKF起因の判別')
    parser.add_argument('logs', nargs='+', help='log:label の形式で複数指定')
    parser.add_argument('--n-wp', type=int, default=350)
    args = parser.parse_args(argv)

    results = []
    for spec in args.logs:
        log_path, label = spec.split(':', 1)
        print(f"分析中: {label}", file=sys.stderr)
        loc_rows = read_full_loc_xcheck(log_path)
        if not loc_rows:
            results.append({'label': label, 'ekf_vs_gnss': {'ekf': None, 'gnss': None},
                             'phase_lock': {'n_laps': 0, 'verdict': 'データ不足'}})
            continue
        results.append({
            'label': label,
            'ekf_vs_gnss': compare_ekf_vs_gnss(loc_rows),
            'phase_lock': lap_phase_lock_check(loc_rows, n_wp=args.n_wp),
        })

    print(format_report(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
