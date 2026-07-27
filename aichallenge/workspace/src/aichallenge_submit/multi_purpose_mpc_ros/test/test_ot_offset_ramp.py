"""Unit tests for the `_ot_cleared`-gated offset-ramp relaxation (50節, 2026-07-14
初版 + 71節/2026-07-15 縦方向完了確認の追加).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
decision logic (`_a_target` selection + the alpha ramp-step formula) is mirrored
here rather than imported. See test_plan_pass_kcorner.py for the AST-extraction
alternative used where the target function has no ROS-dependent free variables;
`_a_target`'s computation is inline in `_control()` (a 600+ line method with many
ROS-typed locals), so full extraction is impractical. The mirror below is a
verbatim transcription of the lines actually shipped in mpc_controller.py:

    _offset_return_ok = self._ot_cleared and _scan.get("fwd_ds") is None
    _a_target = 0.0 if _offset_return_ok else 1.0
    _ramp_step = dt / max(self._ot_ramp_time, 1e-3)
    self._ot_alpha += clip(_a_target - self._ot_alpha, -_ramp_step, _ramp_step)
    self._ot_alpha = clip(self._ot_alpha, 0.0, 1.0)

--- History ---

v1 (50節): `_a_target = 0.0 if ot_cleared else 1.0`. `_ot_cleared` itself is
gated purely on lateral separation (fwd_dlat) — it never checks whether the
opponent has actually been passed longitudinally. Confirmed via 0715-02 log
replay (real collision, t=452-460s: cleared=True the whole time while fwd_dlat
shrank back from 3.63m to 1.89m as the offset returned toward centerline) and
0715-03 log replay (a milder recurrence: a forced LAT-TTC C2_cleared giveup at
wp109, t=34.21, causing a moderate brake rather than a collision).

v2 (71節, this file): additionally require `fwd_ds is None` — i.e., no forward
car is currently tracked at all. `_scan_traffic` only ever populates `fwd_ds`
for cars with `0 < ds` (strictly ahead of the ego, see mpc_controller.py line
~1624), so `fwd_ds is None` is a reliable proxy for "the opponent we were
passing is no longer ahead of us" (genuinely passed, or simply no traffic).
The `_ot_cleared` latch itself is UNCHANGED (still used as-is for the G-2/G-3
ICC release and the LAT-TTC B_cleared bypass, both of which are fine to fire
once merely alongside with good lateral separation — only the offset-return
trigger needed the extra longitudinal check).
"""
import pytest

RAMP_TIME = 0.5


def a_target(ot_state, ot_side, ot_cleared, fwd_ds=None):
    """mpc_controller.py 該当箇所の逐語ミラー(71節でfwd_ds引数を追加)。"""
    if ot_state == "OVERTAKING" and ot_side != 0:
        offset_return_ok = ot_cleared and fwd_ds is None
        return 0.0 if offset_return_ok else 1.0
    return 0.0


def ramp_step(alpha, target, dt, ramp_time=RAMP_TIME):
    step = dt / max(ramp_time, 1e-3)
    delta = max(-step, min(step, target - alpha))
    return max(0.0, min(1.0, alpha + delta))


def run_ramp(alpha0, ot_state, ot_side, ot_cleared, cycles, dt=0.1, fwd_ds=None):
    alpha = alpha0
    for _ in range(cycles):
        alpha = ramp_step(alpha, a_target(ot_state, ot_side, ot_cleared, fwd_ds), dt)
    return alpha


def test_cleared_true_but_fwd_car_still_present_keeps_full_offset():
    """本修正(v2)の中核: clearedがTrueでも、まだ前方に対象車が(横並び中で)
    追跡されている間(fwd_ds is not None)は、オフセットを中央へ戻さない。
    v1(fwd_dsを見ない)ではここでalphaが0へ向かい始め、0715-02実測で
    実際の追突につながっていた。"""
    alpha = run_ramp(1.0, "OVERTAKING", 1, True, cycles=5, dt=0.1, fwd_ds=6.5)
    assert alpha == pytest.approx(1.0)  # v1ならここで<1.0になっていたはず


def test_cleared_true_and_no_forward_car_ramps_offset_back_toward_zero():
    """0713-06 wp136/wp243相当 + 71節: clearedがTrueかつ前方車を見失って
    いる(fwd_ds is None、=縦に抜き終えた)場合のみ、alphaが減少に転じる。"""
    alpha = run_ramp(1.0, "OVERTAKING", 1, True, cycles=5, dt=0.1, fwd_ds=None)
    assert alpha < 1.0
    assert alpha == pytest.approx(0.0, abs=0.05)  # 5*0.1=0.5s=ramp_time分で0近くまで戻る


def test_cleared_false_maintains_full_offset_regression():
    """回帰: cleared=Falseなら従来通りalpha=1.0を維持する(fwd_dsの値に関わらず)。"""
    alpha = run_ramp(1.0, "OVERTAKING", 1, False, cycles=5, dt=0.1, fwd_ds=None)
    assert alpha == pytest.approx(1.0)


def test_fresh_engage_ramps_up_to_full_offset_regression():
    """回帰: エンゲージ直後(cleared=False)は従来通り0から1.0へランプアップする。"""
    alpha = run_ramp(0.0, "OVERTAKING", -1, False, cycles=5, dt=0.1, fwd_ds=6.0)
    assert alpha == pytest.approx(1.0)


@pytest.mark.parametrize("ot_state,ot_side,ot_cleared", [
    ("OVERTAKING", 0, True),   # side未確定(エンゲージ前)
    ("STOPPING", 1, False),    # 追従中
    ("NORMAL", 0, False),      # 通常走行
])
def test_a_target_is_zero_outside_active_overtaking_regression(ot_state, ot_side, ot_cleared):
    """回帰: OVERTAKING中かつside!=0以外は、clearedの値に関わらず常にa_target=0.0。"""
    assert a_target(ot_state, ot_side, ot_cleared, fwd_ds=None) == 0.0


def test_alpha_never_exceeds_valid_range_during_relaxation():
    """境界値: 緩和中もalphaは[0,1]の範囲を割らない。"""
    alpha = 1.0
    for _ in range(50):  # 十分に長く回してオーバーシュートが無いか確認
        alpha = ramp_step(alpha, a_target("OVERTAKING", 1, True, fwd_ds=None), dt=0.1)
        assert 0.0 <= alpha <= 1.0
    assert alpha == pytest.approx(0.0)


def test_cleared_flicker_back_to_false_resumes_ramp_up():
    """clearedが再度Falseへ戻れば(側変更やヒステリシス解除等)、alphaは1.0へ再上昇する。
    switchback時に_ot_clearedをリセットする対処(50-3節で発見した副次バグの修正)が
    正しく機能する前提となるシナリオ。"""
    alpha = run_ramp(1.0, "OVERTAKING", 1, True, cycles=5, dt=0.1, fwd_ds=None)  # まず0近くへ
    assert alpha < 0.1
    alpha = run_ramp(alpha, "OVERTAKING", -1, False, cycles=5, dt=0.1, fwd_ds=6.0)  # 反転+再エンゲージ相当
    assert alpha == pytest.approx(1.0)


def test_retroactive_0715_02_collision_episode_no_longer_returns_early():
    """遡及検証(0715-02実測、実際に追突した episode): t=452〜460秒はcleared=True
    継続中もfwd_ds(6.9〜13.0m台)は終始有効な値を持っていた(相手がまだ前方に
    追跡されていた)。v1ロジックではこの間ずっとalpha->0(オフセット中央復帰)へ
    向かっていたが、v2では前方車が追跡され続けている限りalpha=1.0を維持し、
    幅寄せに相当する動きを起こさないことを確認する。"""
    alpha_v1 = run_ramp(1.0, "OVERTAKING", -1, True, cycles=10, dt=0.1, fwd_ds=None)
    assert alpha_v1 < 1.0  # v1相当(fwd_ds無視)を再現: 早期に戻り始めてしまう
    alpha_v2 = run_ramp(1.0, "OVERTAKING", -1, True, cycles=10, dt=0.1, fwd_ds=7.0)
    assert alpha_v2 == pytest.approx(1.0)  # v2: 前方車追跡中は維持


def test_retroactive_0715_03_giveup_episode_fwd_ds_was_present():
    """遡及検証(0715-03実測、t=34.21秒、C2_clearedによる強制giveup直前): この時点
    でも対象車はまだ追跡中(fwd_wp=109相当)だったと推定される。v2ロジックであれば
    この場面でもオフセット復帰を保留し、急ブレーキに至る前段の幅寄せ挙動自体を
    未然に防げていた可能性が高いことを、同じミラー関数で確認する。"""
    alpha = run_ramp(1.0, "OVERTAKING", 1, True, cycles=3, dt=0.1, fwd_ds=2.3)
    assert alpha == pytest.approx(1.0)
