"""Unit tests for the curvature-dependent EKF lateral-error correction
("State Sanitizer", 220節続報, AXIS07, 2026-07-28)。

背景: AXIS07(211節)で、EKFの横方向位置推定(ekf_ey)と独立な生GNSS測位から
計算した横方向位置(gnss_ey)の間に、曲率に比例した系統誤差があることが判明した
(|kappa| vs |ekf_ey-gnss_ey|回帰、n=774、r=0.702、Δe_y≈0.772×|kappa|+0.016)。

ユーザー提案により、ローカリゼーション層(EKF/GNSS融合)には一切触れず、
mpc_controller.py側の`t2s()`(EKFから受け取った状態をMPCのx0へ変換する直前の
1レイヤー)で、このバイアスを直接差し引く「State Sanitizer」を実装した。

実装前に、当初の想定(|kappa|ベースの絶対値回帰をそのまま符号反転して適用)が
誤りであることが分かった: 符号付き回帰(実ログ0728-02、n=2101)を行ったところ
diff(ekf_ey-gnss_ey) = -0.747*kappa - 0.012(r=-0.703)であり、正しい補正方向は
e_y += +slope*kappa + intercept(引くのではなく足す)であることが確認された。
既定係数はより大きなサンプル(211節、n=774)由来の0.772/0.016を採用し、
0728-02の符号付き回帰(0.747/0.012)と近い値であることを交差検証した。

core/spatial_bicycle_models.pyはrclpy非依存のため、test_delta_actual_state_
augmentation_201.pyと同じ方針(_RefPathStub + 実BicycleModel/t2s()を直接数値
検証、モックやAST抽出ではない)を踏襲する。
"""
import os
import re

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.reference_path import Waypoint

_MC_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_MC_SRC_PATH) as _f:
    _MC_SRC = _f.read()

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG = _f.read()


class _RefPathStub:
    """test_delta_actual_state_augmentation_201.pyと同一パターン。waypointの
    kappaを外から指定できるよう拡張。"""
    def __init__(self, kappa=0.0):
        self.waypoints = [Waypoint(0.0, 0.0, 0.0, kappa)]
        self.resolution = 1.0
        self.segment_lengths = np.array([1.0])


def make_car(kappa=0.0, use_correction=False, slope=0.772, intercept=0.016):
    return BicycleModel(
        reference_path=_RefPathStub(kappa=kappa), length=1.0, width=0.5, Ts=0.1,
        use_curvature_bias_correction=use_correction,
        curvature_bias_slope=slope, curvature_bias_intercept=intercept)


# ---------------------------------------------------------------------------
# 1) デフォルト値・非侵襲性の確認
# ---------------------------------------------------------------------------

def test_default_is_disabled():
    car = make_car(kappa=0.3)
    assert car.use_curvature_bias_correction is False
    assert car.curvature_bias_slope == pytest.approx(0.772)
    assert car.curvature_bias_intercept == pytest.approx(0.016)


def test_flag_false_leaves_e_y_unchanged_regardless_of_kappa():
    """④遡及効果: 既定(false)では、非ゼロkappaのwaypointでも従来通りe_yが変化しない
    (回帰: 既存のtest_delta_actual_state_augmentation_201.py群の前提を壊さない)。"""
    for kappa in [-0.3, -0.05, 0.0, 0.1, 0.35]:
        car = make_car(kappa=kappa, use_correction=False)
        new_state = car.t2s(reference_waypoint=car.current_waypoint,
                             reference_state=car.temporal_state)
        # car.temporal_state == waypoint位置なので、補正前のe_yは常に0
        assert new_state.e_y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2) 補正式の数値検証(符号・大きさ)
# ---------------------------------------------------------------------------

def test_flag_true_applies_slope_times_kappa_plus_intercept():
    slope, intercept = 0.772, 0.016
    for kappa in [-0.3, -0.05, 0.1, 0.35]:
        car = make_car(kappa=kappa, use_correction=True, slope=slope, intercept=intercept)
        new_state = car.t2s(reference_waypoint=car.current_waypoint,
                             reference_state=car.temporal_state)
        expected = 0.0 + slope * kappa + intercept  # 補正前のe_yは0
        assert new_state.e_y == pytest.approx(expected)


def test_correction_sign_matches_verified_regression():
    """220節続報で実ログ(0728-02, n=2101)から検証した符号付き回帰
    diff(ekf_ey-gnss_ey) = -0.747*kappa - 0.012 を踏まえ、正しい補正方向は
    「引く」ではなく「足す」(e_y_corrected = e_y_ekf - diff = e_y_ekf + 0.747*kappa
    + 0.012)であることを固定化する。もし符号を誤ってe_y -= slope*kappa+intercept
    としてしまうと、正のkappaに対して誤差を倍加させる方向になる。"""
    kappa = 0.25
    car_correct_sign = make_car(kappa=kappa, use_correction=True,
                                 slope=0.747, intercept=0.012)
    new_state = car_correct_sign.t2s(reference_waypoint=car_correct_sign.current_waypoint,
                                      reference_state=car_correct_sign.temporal_state)
    # 正しい符号(足す)であれば、正のkappaに対してe_yは正方向に補正される
    assert new_state.e_y > 0.0
    assert new_state.e_y == pytest.approx(0.747 * kappa + 0.012)


def test_zero_kappa_applies_intercept_only():
    car = make_car(kappa=0.0, use_correction=True, slope=0.772, intercept=0.016)
    new_state = car.t2s(reference_waypoint=car.current_waypoint,
                         reference_state=car.temporal_state)
    assert new_state.e_y == pytest.approx(0.016)


# ---------------------------------------------------------------------------
# 3) 他の状態量への非干渉性(①非矛盾性)
# ---------------------------------------------------------------------------

def test_correction_does_not_affect_e_psi_or_delta_actual():
    car = make_car(kappa=0.3, use_correction=True)
    car.spatial_state.delta_actual = 0.05
    new_state = car.t2s(reference_waypoint=car.current_waypoint,
                         reference_state=car.temporal_state)
    assert new_state.e_psi == pytest.approx(0.0)
    assert new_state.delta_actual == pytest.approx(0.05)


def test_correction_scales_linearly_with_kappa():
    """補正式が線形(smoothstep等の非線形要素を含まない、単純な線形回帰式のまま)
    であることを確認する回帰テスト。"""
    slope, intercept = 0.772, 0.016
    car_a = make_car(kappa=0.1, use_correction=True, slope=slope, intercept=intercept)
    car_b = make_car(kappa=0.2, use_correction=True, slope=slope, intercept=intercept)
    e_y_a = car_a.t2s(reference_waypoint=car_a.current_waypoint,
                       reference_state=car_a.temporal_state).e_y
    e_y_b = car_b.t2s(reference_waypoint=car_b.current_waypoint,
                       reference_state=car_b.temporal_state).e_y
    assert (e_y_b - e_y_a) == pytest.approx(slope * 0.1)


# ---------------------------------------------------------------------------
# 4) mpc_controller.py / config.yaml側の配線(ソーステキスト検証)
# ---------------------------------------------------------------------------

def test_config_yaml_declares_params_with_correct_defaults():
    """223節続報: use_curvature_bias_correctionは実走行検証のためtrue/falseを
    行き来するライブトグル(debug_extra_actuator_delay_s/use_savgol_kappaと同じ
    運用)のため、config.yaml側は値そのものではなくキーの宣言・型のみを確認する。
    Pythonデフォルト(False)自体はtest_default_is_disabledで別途固定的に検証済み。"""
    assert re.search(r"^\s*use_curvature_bias_correction:\s*(true|false)\s*(#.*)?$",
                      _CFG, re.MULTILINE)
    assert re.search(r"^\s*curvature_bias_slope:\s*0\.772\s*(#.*)?$", _CFG, re.MULTILINE)
    assert re.search(r"^\s*curvature_bias_intercept:\s*0\.016\s*(#.*)?$", _CFG, re.MULTILINE)


def test_create_car_passes_curvature_bias_kwargs():
    assert "use_curvature_bias_correction=bool(" in _MC_SRC
    assert '"use_curvature_bias_correction", False' in _MC_SRC
    assert "curvature_bias_slope=float(" in _MC_SRC
    assert "curvature_bias_intercept=float(" in _MC_SRC


def test_ros2_param_declared_and_live_updatable():
    assert 'self.declare_parameter(\n                "use_curvature_bias_correction", self._car.use_curvature_bias_correction)' in _MC_SRC
    assert 'param.name == "use_curvature_bias_correction"' in _MC_SRC
    assert "self._car.use_curvature_bias_correction = bool(param.value)" in _MC_SRC
    assert 'param.name == "curvature_bias_slope"' in _MC_SRC
    assert 'param.name == "curvature_bias_intercept"' in _MC_SRC


def test_startup_log_present():
    assert '"[CONFIG] use_curvature_bias_correction: "' in _MC_SRC


def test_param_declaration_happens_after_car_is_constructed():
    """self._carがdeclare_parameter呼び出し時点で既に存在することを、__init__内での
    実行順序(self._initialize()を先に呼び、その後にself._setup_parameters_callback()
    を呼ぶ)で確認する。メソッドの「定義」順はPythonの呼び出し順を意味しないため、
    __init__本体内の呼び出し順のみを検証する(_initialize/_setup_parameters_callback
    それぞれの定義がファイル中のどこにあるかは無関係)。"""
    idx_call_1 = _MC_SRC.index("self._initialize()")
    idx_call_2 = _MC_SRC.index("self._setup_parameters_callback()")
    assert idx_call_1 < idx_call_2
    # 両メソッドの中に、それぞれ期待する処理が実在することも併せて確認する。
    assert "self._car = create_car(self._reference_path)" in _MC_SRC
    assert 'self.declare_parameter(\n                "use_curvature_bias_correction"' in _MC_SRC
