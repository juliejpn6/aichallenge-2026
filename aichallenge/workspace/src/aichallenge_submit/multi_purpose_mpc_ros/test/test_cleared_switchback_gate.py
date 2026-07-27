"""Unit tests for the switchback/_ot_cleared alpha-gate fix (2026-07-14, 0714-02 事象③).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
logic is mirrored here (verbatim transcription), matching the existing style of
test_ot_offset_ramp.py.

Real per-cycle ORDER in mpc_controller.py's _control() (confirmed by line number):
  1. engage/switchback decision block (may reset _ot_alpha=0.0 / _ot_cleared=False)
  2. offset-ramp block: `_a_target = 0.0 if self._ot_cleared else 1.0`, then ramps
     `_ot_alpha` toward it using the CURRENT (this-cycle-start) `_ot_cleared`.
  3. Fix-2 cleared-recompute block (`if fwd_dlat < clear_lat_reacquire: cleared=False
     elif fwd_dlat >= clear_lat_release (...): cleared=True`) — this is what THIS
     session's fix gates on `self._ot_alpha >= 1.0 - 1e-3`, using the alpha value
     that step 2 JUST updated this same cycle.

Bug (0714-02 wp175-176実測): fwd_dlat is a side-agnostic absolute distance, so right
after a switchback (branch A) it can already exceed clear_lat_release even though the
car has not yet actually moved onto the new side (alpha freshly reset to 0). The old
code re-latched `_ot_cleared=True` within 1-2 cycles, which immediately reversed the
just-started ramp-up (`_a_target=0` once cleared), pinning alpha near 0 and defeating
the side switch. The fix requires the ramp to have actually completed (alpha≈1) before
`_ot_cleared` can become True again.
"""
import pytest

RAMP_TIME = 0.5


def a_target(ot_state, ot_side, ot_cleared):
    if ot_state == "OVERTAKING" and ot_side != 0:
        return 0.0 if ot_cleared else 1.0
    return 0.0


def ramp_step(alpha, target, dt, ramp_time=RAMP_TIME):
    step = dt / max(ramp_time, 1e-3)
    delta = max(-step, min(step, target - alpha))
    return max(0.0, min(1.0, alpha + delta))


def cleared_recompute(ot_state, fwd_dlat, fwd_ds, ot_alpha, cleared,
                       clear_lat_reacquire=1.6, clear_lat_release=2.1,
                       fwd_min_lat_sep=1.8, clear_ds_beside=1.0,
                       gate_on_alpha=True):
    """mpc_controller.py の Fix-2 recompute ブロックの逐語ミラー。
    gate_on_alpha=Falseで2026-07-14修正前の(バグのある)挙動を再現できる。"""
    if ot_state == "OVERTAKING":
        if fwd_dlat is not None:
            if fwd_dlat < clear_lat_reacquire:
                cleared = False
            elif (fwd_dlat >= clear_lat_release
                  or (fwd_dlat >= fwd_min_lat_sep and fwd_ds is not None
                      and fwd_ds <= clear_ds_beside)):
                if (not gate_on_alpha) or ot_alpha >= 1.0 - 1e-3:
                    cleared = True
    else:
        cleared = False
    return cleared


def run_cycle(alpha, cleared, ot_side, fwd_dlat, fwd_ds, dt=0.025, gate_on_alpha=True):
    """1制御周期分: 実装と同じ順序(オフセットランプ→Fix-2再計算)で1ステップ進める。"""
    target = a_target("OVERTAKING", ot_side, cleared)
    alpha = ramp_step(alpha, target, dt)
    cleared = cleared_recompute("OVERTAKING", fwd_dlat, fwd_ds, alpha, cleared,
                                 gate_on_alpha=gate_on_alpha)
    return alpha, cleared


def test_switchback_with_fix_ramps_to_full_before_reclearing():
    """0714-02 wp175-176再現(修正後): switchback直後(alpha=0, cleared=False)から、
    fwd_dlatが既に閾値を超えていても、alphaが実際に1.0まで完全にランプし終えるまで
    clearedはFalseのまま維持され、offset-returnが即座に反転を潰さない。alphaが1.0に
    達して初めてclearedがTrueになり、その後は(section 50の設計通り)offset-returnで
    alphaが再び下降し始める — これは意図された挙動であり、バグではない。"""
    alpha, cleared = 0.0, False
    fwd_dlat = 2.5  # switchback直後から既にclear_lat_release(2.1)を超えている(側非依存の絶対値)
    peak_alpha_before_clear = 0.0
    for _ in range(int(RAMP_TIME / 0.025) - 1):  # ramp_time未満の間
        alpha, cleared = run_cycle(alpha, cleared, ot_side=-1,
                                    fwd_dlat=fwd_dlat, fwd_ds=3.0)
        assert cleared is False  # ランプ完了前はclearedへ昇格しない
        peak_alpha_before_clear = max(peak_alpha_before_clear, alpha)
    assert peak_alpha_before_clear < 1.0 - 1e-3  # まだ完全には寄り切っていない
    # 更に回せば、alphaが1.0へ到達しclearedがTrueになる(その後はoffset-returnでalphaが下降)
    reached_full_before_clear = False
    for _ in range(3):
        prev_cleared = cleared
        alpha, cleared = run_cycle(alpha, cleared, ot_side=-1,
                                    fwd_dlat=fwd_dlat, fwd_ds=3.0)
        if cleared and not prev_cleared:
            reached_full_before_clear = (alpha == pytest.approx(1.0, abs=1e-3))
            break
    assert reached_full_before_clear  # cleared=Trueに転じた瞬間、alphaは確かに1.0まで到達していた


def test_switchback_without_fix_reclears_almost_immediately_regression_demo():
    """修正前(gate_on_alpha=False)の挙動を再現し、バグが実在したことを示す回帰デモ:
    switchback直後1周期で早くもcleared=Trueへ戻り、alphaがほぼ0(0.05前後)に
    張り付いたままオフセット復帰ロジックに反転を潰されてしまう。"""
    alpha, cleared = 0.0, False
    alpha, cleared = run_cycle(alpha, cleared, ot_side=-1, fwd_dlat=2.5, fwd_ds=3.0,
                                gate_on_alpha=False)
    assert cleared is True   # 修正前は即座にclearedへ戻ってしまう
    assert alpha == pytest.approx(0.05, abs=1e-6)  # 1周期分(dt/ramp_time=0.05)しか進めていない
    # 以降も cleared=True のままなので a_target=0 に固定され、alphaはこれ以上伸びない
    alpha, cleared = run_cycle(alpha, cleared, ot_side=-1, fwd_dlat=2.5, fwd_ds=3.0,
                                gate_on_alpha=False)
    assert alpha < 0.05  # 伸びるどころか0へ戻り始める


def test_small_fwd_dlat_still_blocks_clearing_regardless_of_gate():
    """回帰: fwd_dlatがclear_lat_reacquire未満なら、alphaの値に関わらずclearedはFalseのまま
    (ゲートを追加しても既存の「まだ近い」判定は変えていないことの確認)。"""
    cleared = cleared_recompute("OVERTAKING", fwd_dlat=1.0, fwd_ds=3.0,
                                 ot_alpha=1.0, cleared=True)
    assert cleared is False


def test_alpha_exactly_at_gate_threshold_allows_clearing():
    """境界値: alpha=1.0-1e-3(ゲート境界ちょうど)ではclearedへ昇格できる。"""
    cleared = cleared_recompute("OVERTAKING", fwd_dlat=2.5, fwd_ds=3.0,
                                 ot_alpha=1.0 - 1e-3, cleared=False)
    assert cleared is True


def test_alpha_just_below_gate_threshold_blocks_clearing():
    """境界値: alpha=1.0-1e-3をわずかに下回るとclearedへ昇格できない。"""
    cleared = cleared_recompute("OVERTAKING", fwd_dlat=2.5, fwd_ds=3.0,
                                 ot_alpha=1.0 - 2e-3, cleared=False)
    assert cleared is False


def test_normal_engage_without_switchback_is_unaffected_regression():
    """回帰: switchbackを経ない通常エンゲージでは、alphaがramp_time(0.5s)以内に
    1.0へ到達するため、fwd_dlatが閾値を超える現実的なタイミング(通常は並走に数秒
    かかる)であれば、ゲート追加後もclearedは従来通り成立する。"""
    alpha, cleared = 0.0, False
    for _ in range(int(RAMP_TIME / 0.025) + 5):  # ramp_time超過分だけ回してalpha=1.0に収束させる
        alpha, cleared = run_cycle(alpha, cleared, ot_side=1, fwd_dlat=1.0, fwd_ds=3.0)
    assert alpha == pytest.approx(1.0, abs=1e-3)
    assert cleared is False  # まだfwd_dlatが小さい(1.0<1.6)ので未クリア
    alpha, cleared = run_cycle(alpha, cleared, ot_side=1, fwd_dlat=2.5, fwd_ds=3.0)
    assert cleared is True  # alphaは既に1.0収束済みなので即クリアされる(従来と同じ体感速度)
