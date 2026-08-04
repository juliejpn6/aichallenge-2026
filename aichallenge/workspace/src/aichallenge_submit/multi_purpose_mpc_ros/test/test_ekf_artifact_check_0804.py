"""Unit tests for scripts/analyze_ekf_artifact_check.py (2026-08-04)。

背景: Gemini・別Claude両者提案の「(A)排除検査」(蛇行は物理現象かEKF推定
アーティファクトか)の実装。既存ログ7本全てで、ekf_ey/gnss_ey帯パワー比が
1.06-1.09(ほぼ同水準)、周回間位相相関が-0.15〜-0.20(位置固定トリガーなし)
という結果を得た。純粋な計算ロジック(ラップ分割・位相ロック検査・帯パワー)を
合成データでテストする。
"""
import os
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

from analyze_ekf_artifact_check import (  # noqa: E402
    band_power_welch, split_laps, lap_phase_lock_check, segment_continuous,
)


# --- ①非矛盾性: band_power_welchは既知の正弦波で正しい帯域に検出する ---

def test_band_power_welch_detects_known_oscillation_frequency():
    sample_hz = 4.0
    t = np.arange(0, 60, 1.0 / sample_hz)
    signal_in_band = np.sin(2 * np.pi * 0.65 * t)  # 限界サイクル帯(0.5-0.9Hz)内
    signal_out_band = np.sin(2 * np.pi * 1.8 * t)  # 帯域外
    r_in = band_power_welch(signal_in_band, sample_hz, min_segment_s=15.0)
    r_out = band_power_welch(signal_out_band, sample_hz, min_segment_s=15.0)
    assert r_in is not None and r_out is not None
    assert r_in['band_power'] > r_out['band_power']


def test_band_power_welch_returns_none_for_short_series():
    r = band_power_welch(np.zeros(3), sample_hz=4.0, min_segment_s=15.0)
    assert r is None


# --- ②非矛盾性: segment_continuousは時刻ギャップで正しく分割する ---

def test_segment_continuous_splits_on_gap():
    t = np.concatenate([np.arange(0, 10, 0.25), np.arange(15, 25, 0.25)])
    segs = segment_continuous(t, gap_thr_s=2.0)
    assert len(segs) == 2


# --- ③非矛盾性: split_lapsはwp後退(周回境界)で正しく分割する ---

def test_split_laps_detects_wraparound():
    rows = []
    t = 0.0
    for lap in range(3):
        for wp in range(0, 350, 10):
            rows.append({'t': t, 'wp': wp, 'ekf_ey': 0.0, 'gnss_ey': 0.0})
            t += 0.25
    laps = split_laps(rows, n_wp_hint=350)
    assert len(laps) == 3


def test_split_laps_ignores_short_fragments():
    """非冗長性②: 極端に短い断片(20点未満)は周回として数えない。"""
    rows = [{'t': i * 0.25, 'wp': i, 'ekf_ey': 0.0, 'gnss_ey': 0.0} for i in range(5)]
    laps = split_laps(rows, n_wp_hint=350)
    assert len(laps) == 0


# --- ④位相ロック検査: 既知の合成データ(位置固定 vs ランダム位相)で正しく判別 ---

def _make_lap(n_wp, offset_pattern, noise_seed=0):
    rng = np.random.RandomState(noise_seed)
    rows = []
    for wp in range(n_wp):
        ey = offset_pattern(wp) + rng.normal(0, 0.01)
        rows.append({'t': wp * 0.25, 'wp': wp, 'ekf_ey': ey, 'gnss_ey': ey})
    return rows


def test_phase_lock_detects_position_fixed_trigger():
    """同じwp位置(例: wp=100付近)に毎周回ピークが立つ合成データなら、
    周回間相関は正に有意になるはず。"""
    n_wp = 350

    def pattern(wp):
        return 1.0 * np.exp(-((wp - 100) ** 2) / (2 * 15 ** 2))  # wp=100に固定ピーク

    rows = []
    for lap in range(5):
        lap_rows = _make_lap(n_wp, pattern, noise_seed=lap)
        rows.extend(lap_rows)
    result = lap_phase_lock_check(rows, n_wp=n_wp)
    assert result['mean_lap_to_lap_corr'] > 0.3
    assert '位置固定' in result['verdict']


def test_phase_lock_detects_random_phase():
    """周回ごとにピーク位置がランダムに動く合成データなら、周回間相関は
    ほぼ0になるはず(実データ0801-01等の-0.15〜-0.20相当のパターンを再現)。"""
    n_wp = 350
    rng = np.random.RandomState(42)
    rows = []
    for lap in range(6):
        peak_pos = rng.uniform(0, n_wp)

        def pattern(wp, peak_pos=peak_pos):
            return 1.0 * np.exp(-((wp - peak_pos) ** 2) / (2 * 15 ** 2))

        rows.extend(_make_lap(n_wp, pattern, noise_seed=lap + 100))
    result = lap_phase_lock_check(rows, n_wp=n_wp)
    assert abs(result['mean_lap_to_lap_corr']) < 0.3  # 位置固定ほど強くない


def test_phase_lock_insufficient_laps_reports_data_shortage():
    rows = _make_lap(350, lambda wp: 0.0, noise_seed=0)
    result = lap_phase_lock_check(rows, n_wp=350)
    assert 'データ不足' in result['verdict']
