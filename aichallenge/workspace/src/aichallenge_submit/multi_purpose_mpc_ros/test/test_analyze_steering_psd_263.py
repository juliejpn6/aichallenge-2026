"""263節続報(2026-08-02、蛇行/性能ギャップ分析Part B): analyze_steering_psd.py
の単体テスト。

rosbag読み取り部分(read_steering_series/read_speed_series)はmcap/rclpy/
nav_msgs依存かつ実データが無いと意味のある検証ができないため、ここでは
セグメント分割・状態割り当て・バンドパワー計算・判別ロジックという純粋な
部分だけを対象にする(rosbag読み取り自体は263節本編でのB-1データソース確認・
B-3実行で実データに対して動作確認済み)。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np

import analyze_steering_psd as asp  # noqa: E402


# ---------------------------------------------------------------------------
# 状態割り当て・セグメント分割
# ---------------------------------------------------------------------------

def test_assign_states_uses_most_recent_transition():
    ot_series = [(10.0, 'NORMAL'), (20.0, 'OVERTAKING'), (30.0, 'STOPPING')]
    states = asp.assign_states([5.0, 15.0, 25.0, 35.0], ot_series)
    assert states == [None, 'NORMAL', 'OVERTAKING', 'STOPPING']


def test_assign_states_exact_transition_time_uses_new_state():
    ot_series = [(10.0, 'NORMAL'), (20.0, 'OVERTAKING')]
    states = asp.assign_states([20.0], ot_series)
    assert states == ['OVERTAKING']


def test_segment_by_state_splits_on_state_change():
    series = [(0.0, 0.1), (1.0, 0.2), (2.0, 0.3), (3.0, 0.4)]
    states = ['A', 'A', 'B', 'B']
    segments = asp.segment_by_state(series, states)
    assert [s for s, _ in segments] == ['A', 'B']
    assert segments[0][1] == [(0.0, 0.1), (1.0, 0.2)]
    assert segments[1][1] == [(2.0, 0.3), (3.0, 0.4)]


def test_segment_by_state_merges_consecutive_same_state():
    series = [(0.0, 0.1), (1.0, 0.2), (2.0, 0.3)]
    states = ['A', 'A', 'A']
    segments = asp.segment_by_state(series, states)
    assert len(segments) == 1
    assert segments[0][0] == 'A'


def test_filter_min_length_excludes_short_segments():
    segments = [
        ('A', [(0.0, 0.0), (5.0, 0.0)]),      # 5s、除外(<8s)
        ('B', [(0.0, 0.0), (10.0, 0.0)]),     # 10s、採用
    ]
    kept, excluded_count, excluded_duration = asp.filter_min_length(
        segments, min_length_s=8.0)
    assert [s for s, _ in kept] == ['B']
    assert excluded_count == 1
    assert excluded_duration == 5.0


def test_filter_min_length_keeps_all_when_all_long_enough():
    segments = [('A', [(0.0, 0.0), (9.0, 0.0)])]
    kept, excluded_count, excluded_duration = asp.filter_min_length(
        segments, min_length_s=8.0)
    assert len(kept) == 1
    assert excluded_count == 0
    assert excluded_duration == 0.0


# ---------------------------------------------------------------------------
# リサンプリング・バンドパワー
# ---------------------------------------------------------------------------

def test_resample_uniform_produces_correct_grid_length():
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]
    grid, resampled = asp.resample_uniform(pts, sample_hz=10.0)
    assert grid is not None
    assert len(grid) == int(4.0 * 10.0)
    assert abs(grid[0] - 0.0) < 1e-9
    # 線形補間により t=2.5 付近の値が2.5に近いこと(区分的に線形なため)。
    idx = int(2.5 * 10.0)
    assert abs(resampled[idx] - 2.5) < 0.5


def test_resample_uniform_returns_none_for_too_short_segment():
    pts = [(0.0, 0.0), (0.01, 1.0)]
    grid, resampled = asp.resample_uniform(pts, sample_hz=10.0)
    assert grid is None
    assert resampled is None


def test_band_power_known_flat_spectrum():
    """一定パワー1.0のスペクトルなら、帯域幅と一致するパワーになること。"""
    freqs = np.linspace(0, 10, 101)  # 0.1Hz刻み
    pxx = np.ones_like(freqs)
    p = asp.band_power(freqs, pxx, (2.0, 4.0))
    assert abs(p - 2.0) < 0.05  # 帯域幅2.0Hz × パワー1.0 ≈ 2.0


def test_band_power_empty_band_returns_zero():
    freqs = np.array([0.0, 0.1, 0.2])
    pxx = np.array([1.0, 1.0, 1.0])
    assert asp.band_power(freqs, pxx, (5.0, 6.0)) == 0.0


# ---------------------------------------------------------------------------
# 判別ロジック(励起仮説 vs 過渡応答仮説)
# ---------------------------------------------------------------------------

def _mock_state(limit_cycle_power, std_rad=1.0):
    return {'limit_cycle_power': limit_cycle_power, 'std_rad': std_rad}


def test_discriminate_supports_excitation_hypothesis_when_ratio_high():
    per_state = {
        'NORMAL': _mock_state(1.0),
        'STOPPING': _mock_state(2.5),
        'OVERTAKING': _mock_state(3.0),
    }
    result = asp.discriminate(per_state)
    assert '励起仮説' in result['verdict']


def test_discriminate_supports_transient_response_when_ratio_low():
    """263節続報Part B実測(0802ログ)で実際に得られたケース: 限界サイクル帯
    比はSTOPPING=0.10・OVERTAKING=0.78とNORMAL以下——励起仮説を支持しない。"""
    per_state = {
        'NORMAL': _mock_state(1.0),
        'STOPPING': _mock_state(0.10),
        'OVERTAKING': _mock_state(0.78),
    }
    result = asp.discriminate(per_state)
    assert '過渡応答仮説' in result['verdict']


def test_discriminate_reports_ambiguous_when_ratios_mixed():
    per_state = {
        'NORMAL': _mock_state(1.0),
        'STOPPING': _mock_state(0.5),   # 過渡応答寄り
        'OVERTAKING': _mock_state(2.5),  # 励起寄り
    }
    result = asp.discriminate(per_state)
    assert '中間' in result['verdict']


def test_discriminate_handles_missing_normal_data():
    result = asp.discriminate({'NORMAL': None})
    assert '判別不能' in result['verdict']


def test_discriminate_computes_std_ratio():
    per_state = {
        'NORMAL': _mock_state(1.0, std_rad=0.5),
        'STOPPING': _mock_state(0.1, std_rad=1.0),
        'OVERTAKING': _mock_state(0.8, std_rad=1.5),
    }
    result = asp.discriminate(per_state)
    assert abs(result['ratios']['STOPPING']['std_ratio'] - 2.0) < 1e-9
    assert abs(result['ratios']['OVERTAKING']['std_ratio'] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# CDRレイアウトの検証(既知バイト長との整合性)
# ---------------------------------------------------------------------------

def test_ackermann_cdr_format_matches_verified_byte_length():
    """実データ(0802予選ログのrosbag)で実測したメッセージバイト長(48byte)と
    フォーマット文字列のcalcsizeが一致すること(263節本編B-1で実地検証済み)。"""
    import struct
    assert struct.calcsize(asp._ACKERMANN_FMT) == 48
