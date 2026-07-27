"""Unit tests for the safety_margin_override ramp fix (2026-07-14, 0714-01 事象A対策).

mpc_controller.py の該当ブロック(_control()内、オフセットランプ直後)をミラーリングして
検証する。ROS型(rclpy/self._mpc等)に依存するため実コードの直接importは行わない。

ミラー対象(実装と逐語同一のロジック):
    _margin_target = (self._ot_safety_margin if self._ot_state == "OVERTAKING"
                       else self._ot_margin_full)
    _margin_rate = abs(self._ot_margin_full - self._ot_safety_margin) / max(
        self._ot_ramp_time, 1e-3)
    _margin_step = _margin_rate * dt
    self._ot_margin_cur += float(np.clip(
        _margin_target - self._ot_margin_cur, -_margin_step, _margin_step))
    _is_ramping = abs(_margin_target - self._ot_margin_cur) >= 1e-3
"""
import numpy as np
import pytest

SAFETY_MARGIN_OVERTAKE = 0.8   # [m] 実装既定値(safety_margin_overtake)
MARGIN_FULL = float(2.30 / np.sqrt(2))  # [m] width=2.30 から算出されるmodel.safety_margin実測値(実装同様float化)
RAMP_TIME = 0.5                 # [s] 既存_ot_ramp_time既定値(オフセットランプと共通)
DT = 0.025                      # [s] 40Hz制御周期


def margin_step(margin_cur, state, margin_full=MARGIN_FULL,
                 safety_margin_overtake=SAFETY_MARGIN_OVERTAKE,
                 ramp_time=RAMP_TIME, dt=DT):
    margin_target = safety_margin_overtake if state == "OVERTAKING" else margin_full
    margin_rate = abs(margin_full - safety_margin_overtake) / max(ramp_time, 1e-3)
    step = margin_rate * dt
    new_cur = margin_cur + float(np.clip(margin_target - margin_cur, -step, step))
    is_ramping = bool(abs(margin_target - new_cur) >= 1e-3)
    return new_cur, is_ramping


def test_transition_to_stopping_does_not_snap_instantly():
    """0714-01再現: OVERTAKING(0.8m収束済み)からSTOPPINGへ遷移した直後の1周期で、
    marginが即座にフル値へジャンプしないこと(従来はNone代入で即時フルへ飛んでいた)。"""
    new_cur, is_ramping = margin_step(SAFETY_MARGIN_OVERTAKE, "STOPPING")
    assert new_cur < MARGIN_FULL - 0.5   # フル値(≈1.626m)へは程遠い
    assert new_cur > SAFETY_MARGIN_OVERTAKE  # ただし緩み始めてはいる
    assert is_ramping is True


def test_transition_converges_to_full_margin_after_ramp_time():
    """ramp_time(0.5s)相当のステップを繰り返せば、最終的にフルマージンへ収束する(回帰: 復帰自体は止まらない)。"""
    cur = SAFETY_MARGIN_OVERTAKE
    is_ramping = True
    for _ in range(int(RAMP_TIME / DT) + 5):
        cur, is_ramping = margin_step(cur, "STOPPING")
    assert cur == pytest.approx(MARGIN_FULL, abs=1e-6)
    assert is_ramping is False


def test_overtaking_direction_also_ramps_symmetrically():
    """OVERTAKINGへ再エンゲージした直後(フルマージンから)も同じ速さで縮小する(対称性、非冗長性の根拠)。"""
    new_cur, is_ramping = margin_step(MARGIN_FULL, "OVERTAKING")
    assert new_cur > SAFETY_MARGIN_OVERTAKE + 0.5  # まだ縮小の途中
    assert new_cur < MARGIN_FULL
    assert is_ramping is True


def test_no_overshoot_when_close_to_target():
    """目標との差がステップ幅より小さい場合はクリップにより目標でぴったり止まる(オーバーシュートしない)。"""
    near_target = MARGIN_FULL - 1e-4
    new_cur, is_ramping = margin_step(near_target, "STOPPING")
    assert new_cur == pytest.approx(MARGIN_FULL, abs=1e-6)
    assert is_ramping is False


def test_already_settled_at_overtaking_target_is_not_ramping():
    """OVERTAKING中、既にsafety_margin_overtakeへ収束済みなら_is_ramping=Falseのまま
    (毎周期[MARGIN-RAMP]ログが出続けない=エッジトリガーの回帰確認)。"""
    _, is_ramping = margin_step(SAFETY_MARGIN_OVERTAKE, "OVERTAKING")
    assert is_ramping is False


def test_normal_state_targets_full_margin_same_as_stopping():
    """NORMAL状態もSTOPPINGと同じくフルマージンを目標にする(3箇所の代入サイトの整合性)。"""
    new_cur_stopping, _ = margin_step(1.0, "STOPPING")
    new_cur_normal, _ = margin_step(1.0, "NORMAL")
    assert new_cur_stopping == pytest.approx(new_cur_normal)
