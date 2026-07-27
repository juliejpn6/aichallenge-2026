"""Unit tests for the closing-rate-derived dynamic engage distance (2026-07-15,
ユーザー提案: 固定距離での判断は「追いつくタイミング」を無視しており不適切).

mpc_controller.py imports rclpy/autoware message types at module scope, and the
_close_enough computation is embedded inline in the giant _control() method, so
it is verified via a hand-written verbatim mirror (documented session
convention) rather than AST-extraction or a real import.

Bug (design gap, not a crash): the near-proximity engage gate used a FIXED
distance (engage_max_dist=6.0m) regardless of how fast the ego was actually
closing on the opponent. For a genuinely stopped opponent (closing≈v_pot≈
4.17m/s), 6.0m of remaining distance is only ~1.4s away — not enough lead time
to complete the lateral maneuver (t_lateral=3.0s) before getting dangerously
close, which is the root mechanism behind 67節's F3-creep self-reference loop
and a contributing factor in 68節's near-collision. For a barely-faster
opponent (closing near opp_min_closing), 6.0m was needlessly cautious (plenty
of time to spare).

Fix: replace the fixed threshold with
    engage_dist = closing_est * t_lateral + pass_clear
(derived from "time-to-reach = ds/closing <= t_lateral+margin" rearranged to a
distance), reusing five already-existing named constants (v_pot, t_lateral,
pass_clear, opp_min_closing, fwd_max_consider) — zero new parameters. The
result is clamped between the original engage_max_dist (floor — never engage
LATER than before) and fwd_max_consider (ceiling — never try to engage a car
we don't even track as forward traffic).
"""
import pytest

V_POT = 4.1667  # m/s (15km/h potential speed)
T_LATERAL = 3.0
PASS_CLEAR = 3.0
OPP_MIN_CLOSING = 0.7
ENGAGE_MAX_DIST = 6.0
FWD_MAX_CONSIDER = 20.0


def engage_dist_dynamic(fwd_vopp, v_pot=V_POT, t_lateral=T_LATERAL,
                         pass_clear=PASS_CLEAR, opp_min_closing=OPP_MIN_CLOSING,
                         engage_max_dist=ENGAGE_MAX_DIST,
                         fwd_max_consider=FWD_MAX_CONSIDER):
    """Verbatim mirror of the _engage_dist_dynamic computation added to
    mpc_controller.py's cheap_ok gate (2026-07-15)."""
    closing_est = (v_pot - fwd_vopp) if fwd_vopp is not None else v_pot
    closing_est = max(closing_est, opp_min_closing)
    return min(fwd_max_consider, max(engage_max_dist,
                                      closing_est * t_lateral + pass_clear))


def test_stopped_opponent_engages_much_farther_than_old_fixed_distance():
    """本修正の中核: 相手が完全停止(vopp=0)の場合、closing≈v_potとなり、
    従来の固定6.0mよりずっと遠く(約15.5m)からエンゲージ評価が始まる。"""
    dist = engage_dist_dynamic(fwd_vopp=0.0)
    assert dist == pytest.approx(V_POT * T_LATERAL + PASS_CLEAR, abs=0.01)
    assert dist > ENGAGE_MAX_DIST * 2  # 従来値の2倍以上


def test_barely_closing_opponent_does_not_regress_below_old_fixed_floor():
    """回帰: closingがopp_min_closing付近(ほぼ追いつけない速さ)まで小さい場合、
    動的距離は従来の固定値(6.0m)を下回らない(フロアとして機能する=退化しない)。"""
    dist = engage_dist_dynamic(fwd_vopp=V_POT - OPP_MIN_CLOSING)  # closing==opp_min_closing
    assert dist == pytest.approx(ENGAGE_MAX_DIST)


def test_moderate_closing_falls_between_floor_and_stopped_case():
    """中間ケース: closingが中程度なら、距離もフロアと停止車ケースの間になる。"""
    dist = engage_dist_dynamic(fwd_vopp=2.0)  # closing = 4.1667-2.0 = 2.1667
    assert ENGAGE_MAX_DIST < dist < engage_dist_dynamic(fwd_vopp=0.0)


def test_ceiling_is_fwd_max_consider_regression():
    """回帰: closingが極端に大きくても(理論上あり得ないが)、fwd_max_consider(20m)を
    超えない — そもそも走査対象外の相手に適用してはならないため。"""
    dist = engage_dist_dynamic(fwd_vopp=0.0, t_lateral=100.0)  # 意図的に極端な値
    assert dist == FWD_MAX_CONSIDER


def test_fwd_vopp_none_uses_full_v_pot_as_closing():
    """回帰: vopp不明の場合はv_pot全体をclosingとみなす(既存の他ロジックと同じ
    「不明時は保守的に速い相手扱い」ではなく「保守的に速く追いつく前提」)。"""
    dist = engage_dist_dynamic(fwd_vopp=None)
    assert dist == pytest.approx(V_POT * T_LATERAL + PASS_CLEAR, abs=0.01)


def test_retroactive_0715_01_collision_episode_would_have_engaged_much_earlier():
    """遡及検証(0715-01実測、追突事故が起きたエピソード): t=297.48秒時点で
    d_min=9.01m・vopp=0.0(完全停止)。旧固定値(6.0m)では9.01>6.0のためまだ
    エンゲージ対象外だったが、新しい動的距離(vopp=0.0→約15.5m)では既に
    エンゲージ範囲内(9.01<15.5)になっていたことを確認する。これにより
    オフセット確立により多くの距離・時間的余裕が生まれ、68節で修正した
    fwd_dlatチェックが自然に満たされるまでの猶予が増える。"""
    old_close_enough = 9.01 <= ENGAGE_MAX_DIST
    assert old_close_enough is False
    new_dist = engage_dist_dynamic(fwd_vopp=0.0)
    new_close_enough = 9.01 <= new_dist
    assert new_close_enough is True


def test_retroactive_0714_06_corner3_engages_before_old_fixed_point():
    """遡及検証(0714-06実測、コーナー3、t=55.55秒付近): vopp=1.5・d_min=6.97mの
    時点は旧固定値(6.0m)ではまだエンゲージ対象外だったが、新しい動的距離
    (closing=4.1667-1.5=2.667 → dist=2.667*3+3=11.0m)では既に対象になり、
    より早い段階から側選択・オフセット準備が始まることを確認する。"""
    dist = engage_dist_dynamic(fwd_vopp=1.5)
    assert dist == pytest.approx(2.6667 * T_LATERAL + PASS_CLEAR, abs=0.01)
    assert 6.97 <= dist  # 旧固定値6.0mでは対象外だったが、新方式では対象内
