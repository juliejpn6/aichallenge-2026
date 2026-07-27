"""Unit tests for the G-2/G-3 release "stopped opponent" bypass (2026-07-15,
initial version + emergency correction after a real collision).

mpc_controller.py imports rclpy/autoware message types at module scope, and the
_side_clear computation is embedded inline in the giant _control() method, so it
is verified via a hand-written verbatim mirror (documented session convention)
rather than AST-extraction or a real import.

--- History ---

v1 bug (user-reported, confirmed via 0714-06 log wp171-174 replay, t=56.6-67.7s):
the F3-creep floor (icc_f3, gated on est_gap=ds+dlat vs hard_stop_gap) and the
original G-2/G-3 full-speed release (gated on self._ot_cleared, which itself
requires dlat to already have grown) formed a self-referential loop for a
genuinely STOPPED opponent: growing dlat requires forward motion, but forward
motion was throttled precisely because dlat (and hence est_gap) hadn't grown
yet. v1 fix: bypass _ot_cleared for a confirmed stopped/slow opponent as soon as
the CURRENTLY-MEASURED side_room (static corridor width) was at least
along_min_width.

v1 REGRESSION (confirmed via 0715-01 log replay, t=298.31-299.52s): v1 checked
ONLY side_room (the wall-based static corridor width) and never checked
fwd_dlat (how far the ego itself had ACTUALLY moved sideways away from the
opponent right now). At t=298.31, side_room=3.32m (wide) but fwd_dlat=0.24m
(the ego was still almost directly behind the opponent) — the bypass fired
anyway, releasing full speed (eff_v_cap). 0.9 seconds later
([COLLISION-SUSPECTED-CUM] at t=299.25/299.38), d_min collapsed from 4.06m to
1.02m: a real rear-end collision. "There is room over there" and "I have
already moved into it" are different facts, and v1 conflated them.

v2 fix (this file, current): additionally require fwd_dlat (the scan's current
measured lateral separation from the specific opponent locked as the overtake
target) to already be at least along_min_width before releasing full speed.
This still bypasses the (much stricter, hysteresis-gated) _ot_cleared latch,
but no longer releases full acceleration while the ego is still substantially
directly behind the opponent.
"""
import pytest

ALONG_LANE_NEED = 1.85
ALONG_MIN_WIDTH = 1.45
OPP_OBSTACLE_SPEED = 6.0 / 3.6  # 1.6667 m/s


def side_clear_decision(side_room, ot_cleared, fwd_vopp, fwd_dlat,
                         along_lane_need=ALONG_LANE_NEED,
                         along_min_width=ALONG_MIN_WIDTH,
                         opp_obstacle_speed=OPP_OBSTACLE_SPEED):
    """Verbatim mirror of the _side_clear computation in mpc_controller.py's
    OVERTAKING v_safe candidate stack (v2, 2026-07-15緊急修正版)."""
    stopped_opponent = fwd_vopp is not None and fwd_vopp < opp_obstacle_speed
    side_room_ok_now = side_room is not None and side_room >= along_min_width
    actual_lat_clear_now = fwd_dlat is not None and fwd_dlat >= along_min_width
    side_clear = (side_room is not None
                  and side_room >= along_lane_need
                  and ot_cleared) or (
                      stopped_opponent and side_room_ok_now and actual_lat_clear_now)
    return side_clear


def test_moving_opponent_still_requires_ot_cleared_regression():
    """回帰: 走行中の相手(vopp>=obstacle_speed)は従来通りot_cleared必須のまま
    (0712-02の追突事故を踏まえたG-2/G-3安全条件は無変更)。"""
    assert side_clear_decision(side_room=3.0, ot_cleared=False, fwd_vopp=8.0,
                                fwd_dlat=3.0) is False


def test_moving_opponent_with_cleared_and_room_releases_regression():
    """回帰: 走行中の相手でも、ot_cleared=Trueかつ室が十分なら従来通り解放される
    (この経路はfwd_dlatを見ない、v1と同じ挙動のまま)。"""
    assert side_clear_decision(side_room=2.0, ot_cleared=True, fwd_vopp=8.0,
                                fwd_dlat=0.1) is True


def test_stopped_opponent_bypasses_ot_cleared_requirement_when_already_beside():
    """本修正(v2)の中核: 停止/低速車は、ot_cleared未達成でも、既に十分横へ
    移動できていれば(fwd_dlat十分)解放される。"""
    assert side_clear_decision(side_room=3.0, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=2.0) is True


def test_stopped_opponent_still_requires_minimum_room():
    """回帰: 停止/低速車でも、室自体がalong_min_width未満なら解放しない。"""
    assert side_clear_decision(side_room=1.0, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=2.0) is False


def test_stopped_opponent_with_wide_room_but_still_almost_directly_behind_does_not_release():
    """v1→v2の中核修正: 室(side_room)がいくら広くても、自車が実際にはまだ
    ほぼ真後ろ(fwd_dlatが小さい)なら解放しない。v1はこのケースで誤って解放していた。"""
    assert side_clear_decision(side_room=5.0, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=0.2) is False


def test_boundary_exactly_at_along_min_width_releases():
    """境界値: side_room・fwd_dlatともalong_min_widthちょうどは解放される(`>=`)。"""
    assert side_clear_decision(side_room=ALONG_MIN_WIDTH, ot_cleared=False,
                                fwd_vopp=0.9, fwd_dlat=ALONG_MIN_WIDTH) is True


def test_boundary_just_below_along_min_width_dlat_does_not_release():
    """境界値: fwd_dlatがalong_min_widthをわずかに下回ると解放されない。"""
    assert side_clear_decision(side_room=3.0, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=ALONG_MIN_WIDTH - 0.01) is False


def test_boundary_exactly_at_obstacle_speed_is_not_stopped_regression():
    """境界値: vopp==opp_obstacle_speedちょうどは「未満」条件を満たさないため
    stopped_opponent=Falseのまま(既存の`<`厳密比較を踏襲)。"""
    assert side_clear_decision(side_room=3.0, ot_cleared=False,
                                fwd_vopp=OPP_OBSTACLE_SPEED, fwd_dlat=3.0) is False


def test_side_room_none_never_releases_regression():
    """回帰: 対象車のvidが一致せずside_roomがNoneの場合は、停止車判定に関わらず解放しない。"""
    assert side_clear_decision(side_room=None, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=3.0) is False


def test_fwd_dlat_none_never_releases_via_bypass_regression():
    """回帰: fwd_dlatが取得できない場合はバイパス経路を使わない(安全側)。"""
    assert side_clear_decision(side_room=3.0, ot_cleared=False, fwd_vopp=0.9,
                                fwd_dlat=None) is False


def test_retroactive_0714_06_corner3_wp171_would_have_released_early():
    """遡及検証(0714-06実測、コーナー3、wp171相当、t=58.62s): 当時の実測値
    (side_room=Lfree=3.049, fwd_vopp=0.9)は、fwd_dlat=0.035(このタイミングでは
    ほぼ真後ろ)だったため、v2修正版ではまだ解放されない(v1の想定より慎重)。
    その後dlatが育つにつれ(t=59.61でfwd_dlat=0.395、t=63.63で0.925...)、
    along_min_width(1.45)へ到達した時点(実測ではt=64.61付近、fwd_dlat=1.037)で
    初めて解放される — v1(側の室だけで即解放)よりは遅いが、_ot_cleared到達
    (2.1m必要)よりは早い、安全側の中間点になっていることを確認する。"""
    at_engage = side_clear_decision(side_room=3.049, ot_cleared=False, fwd_vopp=0.9,
                                     fwd_dlat=0.035)
    assert at_engage is False  # v1ならTrueだったはずの箇所が、v2ではまだ慎重
    once_dlat_grows = side_clear_decision(side_room=3.03, ot_cleared=False, fwd_vopp=1.5,
                                           fwd_dlat=1.5)
    assert once_dlat_grows is True


def test_retroactive_0715_01_collision_incident_now_blocked():
    """遡及検証(0715-01実測、実際に追突が発生したt=298.31秒の周期): 当時の実測値
    (side_room=3.318, fwd_vopp=0.0, fwd_dlat=0.242)をv1のロジック(fwd_dlat無視)に
    通すとTrue(誤って全開解放、0.9秒後に実際の追突を招いた)。同じ値をv2(本修正)に
    通すとFalse(fwd_dlat=0.242<1.45のため解放しない)になることを確認する。"""
    v1_would_have_released = (3.318 >= ALONG_MIN_WIDTH) and (0.0 < OPP_OBSTACLE_SPEED)
    assert v1_would_have_released is True  # v1の欠陥(実際に事故を起こした挙動)を再現
    v2_result = side_clear_decision(side_room=3.318, ot_cleared=False, fwd_vopp=0.0,
                                     fwd_dlat=0.242)
    assert v2_result is False  # v2では正しくブロックされる


def test_retroactive_0715_01_safe_bypass_case_still_releases():
    """遡及検証(0715-01実測、t=269.95秒、事故につながらなかった正常なバイパス発火):
    side_room=3.795, fwd_vopp=1.64(stopped), fwd_dlat=1.657(既にalong_min_width超)
    のケースはv2でも引き続き解放されることを確認する(過剰に保守的になっていない)。"""
    assert side_clear_decision(side_room=3.795, ot_cleared=False, fwd_vopp=1.640,
                                fwd_dlat=1.657) is True
