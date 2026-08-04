"""Unit tests for the actuator_gain correction (2026-08-03、緩和策①).

背景: Part C 2×2実験(Q×v_max)で蛇行の支配因子はv_max(Q[e_y]ではない)と確定した後、
`analyze_actuator_delay.py`のtau実測(FOPDT tau=160ms、v_max/Qに依存せず一定)+ゲイン実測
(`estimate_gain_continuous`新設、通常走行域|指令振幅|0-30°で4条件すべて0.63-0.78、
既知値0.67と整合)により、蛇行はtau(時定数)の変化ではなく、速度・Qに依存しない
一定のゲイン不足(delta_actualが指令deltaへゲイン1ではなく約0.67-0.7でしか収束しない)
が原因である可能性が高いと判明した。

対処: BicycleModelのdelta_actual状態の連続時間ダイナミクスを
d(delta_actual)/dt = (delta - delta_actual)/tau から
d(delta_actual)/dt = (actuator_gain*delta - delta_actual)/tau へ変更し、
定常状態でdelta_actual→actuator_gain*deltaに収束するようにする。
actuator_gain=1.0(既定)は旧モデルと数学的に完全一致することを保証する
(下位互換、本ファイルで検証)。設計根拠: design_docs/axis06_gain_correction_design_20260803.md。

spatial_bicycle_models.py はrclpy非依存のため、実モジュールを直接importし、実クラス
BicycleModelに対して実メソッドを数値的に検証する(既存のtest_delta_actual_state_
augmentation_201.pyと同じ方針)。
"""
import numpy as np
import pytest

from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.reference_path import Waypoint


class _RefPathStub:
    def __init__(self):
        self.waypoints = [Waypoint(0.0, 0.0, 0.0, 0.0)]
        self.resolution = 1.0
        self.segment_lengths = np.array([1.0])


def make_car(tau=0.19, gain=1.0):
    return BicycleModel(reference_path=_RefPathStub(), length=1.0, width=0.5,
                         Ts=0.1, actuator_lag_tau_s=tau, actuator_gain=gain)


# ---------------------------------------------------------------------------
# ①下位互換性: actuator_gain既定値・gain=1.0は旧モデル(gain概念なし)と完全一致
# ---------------------------------------------------------------------------

def test_default_actuator_gain_is_1_when_unspecified():
    """mpc_controller.py/path_constraints_provider.pyのcreate_car()は
    config.yamlのbicycle_model.actuator_gain(既定1.0)を渡す設計だが、
    直接BicycleModel()を呼んだ場合の既定値も1.0であること。"""
    car = BicycleModel(reference_path=_RefPathStub(), length=1.0, width=0.5, Ts=0.1)
    assert car.actuator_gain == pytest.approx(1.0)


@pytest.mark.parametrize("v_ref,kappa_ref,delta_s,tau", [
    (4.0, 0.1, 1.0, 0.19),
    (1.5, -0.2, 0.6, 0.055),
    (5.5, 0.0, 0.999, 0.0),
])
def test_gain_1_reproduces_no_gain_model_exactly_in_linearize(v_ref, kappa_ref, delta_s, tau):
    """actuator_gain=1.0で明示指定した場合と、gain引数を省略した場合(既定1.0)とで、
    linearize()の出力(f, A, B)が完全一致すること。"""
    car_explicit = make_car(tau=tau, gain=1.0)
    car_default = BicycleModel(reference_path=_RefPathStub(), length=1.0, width=0.5,
                                Ts=0.1, actuator_lag_tau_s=tau)
    f1, A1, B1 = car_explicit.linearize(v_ref, kappa_ref, delta_s)
    f2, A2, B2 = car_default.linearize(v_ref, kappa_ref, delta_s)
    np.testing.assert_allclose(A1, A2, atol=1e-12)
    np.testing.assert_allclose(B1, B2, atol=1e-12)
    np.testing.assert_allclose(f1, f2, atol=1e-12)


def test_gain_1_matches_precomputed_pre_gain_formula():
    """gain=1.0の場合、b_2/b_4が導入前の式(delta_s*alpha / alpha)と一致すること
    (2026-08-03のactuator_gain追加前のコードと数値的に完全一致することの保証)。"""
    tau, v_ref, kappa_ref, delta_s = 0.055, 4.0, 0.1, 1.0
    car = make_car(tau=tau, gain=1.0)
    f, A, B = car.linearize(v_ref, kappa_ref, delta_s)
    alpha = min(1.0, max(0.0, delta_s / (tau * v_ref)))
    assert B[1, 1] == pytest.approx(delta_s * alpha)
    assert B[3, 1] == pytest.approx(alpha)
    assert A[1, 3] == pytest.approx(delta_s * (1 - alpha))
    assert A[3, 3] == pytest.approx(1 - alpha)


# ---------------------------------------------------------------------------
# ①非矛盾性(核心): actuator_gain!=1.0はb_2/b_4のみをスケールし、a_2/a_4は不変
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gain", [0.67, 0.5, 1.3])
def test_gain_scales_only_b2_and_b4_leaves_a2_a4_unchanged(gain):
    tau, v_ref, kappa_ref, delta_s = 0.19, 4.0, 0.1, 1.0
    car_gain1 = make_car(tau=tau, gain=1.0)
    car_gainX = make_car(tau=tau, gain=gain)

    f1, A1, B1 = car_gain1.linearize(v_ref, kappa_ref, delta_s)
    fX, AX, BX = car_gainX.linearize(v_ref, kappa_ref, delta_s)

    # a_2[3](delta_actual_kの係数)・a_4(1-alpha)はgainの影響を受けない
    assert AX[1, 3] == pytest.approx(A1[1, 3])
    assert AX[3, 3] == pytest.approx(A1[3, 3])
    # b_2[1]・b_4[1]はgain倍される
    assert BX[1, 1] == pytest.approx(B1[1, 1] * gain)
    assert BX[3, 1] == pytest.approx(B1[3, 1] * gain)
    # f、他の行は無変更
    np.testing.assert_allclose(fX, f1, atol=1e-12)
    assert AX[0, 0] == pytest.approx(A1[0, 0])


def test_v_ref_zero_gain_does_not_matter_delta_actual_still_frozen():
    """v_ref==0ではalpha=0なのでb_2/b_4は既に0であり、gainを掛けても0のまま
    (ゼロ除算・異常値なし)。"""
    car = make_car(tau=0.055, gain=0.67)
    f, A, B = car.linearize(v_ref=0.0, kappa_ref=0.1, delta_s=1.0)
    assert B[1, 1] == pytest.approx(0.0)
    assert B[3, 1] == pytest.approx(0.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(f))


# ---------------------------------------------------------------------------
# ①非矛盾性: get_spatial_derivatives()もgainを反映する
# ---------------------------------------------------------------------------

def test_spatial_derivatives_reflects_gain():
    car = make_car(tau=0.055, gain=0.67)
    state = [0.0, 0.0, 0.0, 0.0]
    deriv = car.get_spatial_derivatives(state, input=[4.0, 0.1], kappa=0.0)
    s_dot = 4.0
    expected = (0.67 * 0.1 - 0.0) / (0.055 * s_dot)
    assert deriv[3] == pytest.approx(expected)


def test_spatial_derivatives_gain_1_matches_pre_gain_formula():
    car = make_car(tau=0.055, gain=1.0)
    state = [0.0, 0.0, 0.0, 0.0]
    deriv = car.get_spatial_derivatives(state, input=[4.0, 0.1], kappa=0.0)
    s_dot = 4.0
    expected = (0.1 - 0.0) / (0.055 * s_dot)
    assert deriv[3] == pytest.approx(expected)


def test_gain_causes_steady_state_delta_actual_to_undershoot_command():
    """定常応答の直感的確認: gain<1では、delta_actualがdeltaへ完全収束せず、
    gain*deltaに漸近すること(オイラー積分で数ステップ回して確認)。"""
    car = make_car(tau=0.1, gain=0.67)
    delta_actual = 0.0
    delta_cmd = 0.3
    dt = 0.01
    for _ in range(2000):  # 十分に長い時間積分し定常状態へ収束させる
        d = (car.actuator_gain * delta_cmd - delta_actual) / car.actuator_lag_tau_s
        delta_actual += d * dt
    assert delta_actual == pytest.approx(0.67 * delta_cmd, rel=1e-3)
