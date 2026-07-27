"""Unit tests for the cheap_ok pre-filter threshold relax fix (2026-07-14,
フローチャートで洗い出したギャップ①).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
decision logic is mirrored here (verbatim transcription of the shipped lines):

    _thr = (self._ot_min_gap - self._ot_gap_hys) if _was_ot else self._ot_min_gap
    if (_fwd_vopp is not None and _fwd_vopp < self._opp_obstacle_speed):
        _thr = min(_thr, self._along_min_width)   # was: self._along_lane_need
    _left_ok = (_left_free is not None and _left_free >= _thr)
    _right_ok = (_right_free is not None and _right_free >= _thr)

Bug: this "cheap" pre-filter (gating whether `_plan_pass` is even invoked) still
capped the stopped/slow-opponent threshold at along_lane_need(1.85m) after 59節
relaxed `_plan_pass`'s OWN internal k_corner veto and min-width check to
along_min_width(1.45m). A corridor of 1.5-1.84m would be rejected HERE (lr=0),
before `_plan_pass` was ever called, silently neutering the 59節 relaxation.
"""
import pytest

MIN_GAP = 2.5
GAP_HYS = 0.5
OBSTACLE_SPEED = 1.67
ALONG_MIN_WIDTH = 1.45


def cheap_ok_thr(was_ot, fwd_vopp, min_gap=MIN_GAP, gap_hys=GAP_HYS,
                  obstacle_speed=OBSTACLE_SPEED, along_min_width=ALONG_MIN_WIDTH):
    thr = (min_gap - gap_hys) if was_ot else min_gap
    if fwd_vopp is not None and fwd_vopp < obstacle_speed:
        thr = min(thr, along_min_width)
    return thr


def test_retroactive_1_5m_corridor_now_passes_the_cheap_gate():
    """遡及検証: 実測1.5〜1.84m級のcorridor(旧閾値1.85mでは弾かれていた)が、
    新閾値(along_min_width=1.45m)では通過することを確認する。"""
    thr = cheap_ok_thr(was_ot=False, fwd_vopp=1.0)  # 停止/低速車
    assert thr == pytest.approx(1.45)
    left_free = 1.6
    assert left_free >= thr  # 新閾値では通過(旧1.85mでは弾かれていた)


def test_moving_car_threshold_unaffected_regression():
    """回帰: 走行車(vopp>=obstacle_speed)への基準(2.5/2.0)は無変更。"""
    thr_enter = cheap_ok_thr(was_ot=False, fwd_vopp=10.0)
    thr_maintain = cheap_ok_thr(was_ot=True, fwd_vopp=10.0)
    assert thr_enter == pytest.approx(2.5)
    assert thr_maintain == pytest.approx(2.0)


def test_boundary_exactly_at_obstacle_speed_does_not_relax_regression():
    """境界値: vopp==obstacle_speedちょうどは「未満」条件を満たさないため緩和されない
    (既存の`<`厳密比較を維持)。"""
    thr = cheap_ok_thr(was_ot=False, fwd_vopp=OBSTACLE_SPEED)
    assert thr == pytest.approx(MIN_GAP)  # 緩和されず通常基準のまま


def test_boundary_just_below_obstacle_speed_relaxes():
    """境界値: obstacle_speedをわずかに下回ると緩和が適用される。"""
    thr = cheap_ok_thr(was_ot=False, fwd_vopp=OBSTACLE_SPEED - 0.01)
    assert thr == pytest.approx(ALONG_MIN_WIDTH)


def test_fwd_vopp_none_does_not_relax_regression():
    """回帰: fwd_vopp=None(対象車速度不明)では緩和条件に入らない。"""
    thr = cheap_ok_thr(was_ot=False, fwd_vopp=None)
    assert thr == pytest.approx(MIN_GAP)
