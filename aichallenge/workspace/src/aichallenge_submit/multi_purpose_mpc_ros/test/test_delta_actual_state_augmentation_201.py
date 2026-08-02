"""Unit tests for the delta_actual state augmentation (nx=3->4) reimplementation
(2026-07-27, 201節続報, AXIS06).

背景: 198節で同じ状態拡張(delta_actual, 一次遅れでアクチュエータの実舵角を模擬)を
実装したが、tauに環境の実測フル遅延(130ms/190ms)を使い、既存Q/R重み体系(何日も
ローカル実環境=実アクチュエータ遅延130-140ms込みで経験的にチューニングされてきた)
との整合性を崩して悪化したため撤回した(199節)。

ユーザーの指摘(201節): 「ローカルの140ms分は既存の重みが既に暗黙に吸収済みのはず、
予選環境との差分60msだけを対処すべき」という198節撤回時の洞察は、199節では
wp_id_offset(距離ベース)へ適用されたが、delta_actual(状態拡張)へは未適用だった。
199節のwp_id_offset拡張は同じ「差分のみ」の考え方を距離ベースの機構で実装したが、
量子化が粗く実効性がなかった(200節)。delta_actualは連続値の状態拡張なので量子化の
問題を原理的に持たないが、tau=55ms(差分のみ)は今度はMPCのホライズンステップが
表す実時間(delta_s/v_ref、典型的に0.1-0.4s)より常に小さく、alphaが常に1.0へ
飽和して旧3状態モデルと数値的に区別できないと判明した(202節)。

202節続報の方針転換: tauは「アクチュエータ自体の物理応答遅れ」を表すもので、これは
ローカルでも予選でも同じAWSIMシミュレータの同じ車両モデルを使っている以上、環境に
依らず一定(130ms)のはずである。予選環境特有の追加55-60msは、アクチュエータの
特性ではなく環境固有のパイプライン遅延であり、debug_extra_actuator_delay_s(196節、
別機構)が扱う範囲であってtauの役割ではない。そこでtauの既定値を130ms(ローカルの
実測フル遅延)へ変更し、Q/Rをこのモデルに対して再チューニングする方針とした。

spatial_bicycle_models.py はrclpy非依存のため、実モジュールを直接importし、実クラス
BicycleModel/SimpleSpatialStateに対して実メソッドを数値的に検証する
(test_ego_wp_id_wall_aware.pyと同じ方針、モックやAST抽出ではない)。

実装時に発見・修正したバグ: 当初、e_psi行がdelta_actual_k(更新前の値)をそのまま
使っていたため、tau=0でも旧3状態モデルと一致しない(1周期分余分な遅れが入る)
不具合があった。delta_actual_{k+1}(今回ステップで更新された後の値)をe_psi行へ
代入し直す形へ修正済み(本ファイルのtest_tau_zero_*系テストで検証)。
"""
import numpy as np
import pytest

from multi_purpose_mpc_ros.core.spatial_bicycle_models import (
    BicycleModel, SimpleSpatialState)
from multi_purpose_mpc_ros.core.reference_path import Waypoint


class _RefPathStub:
    def __init__(self):
        self.waypoints = [Waypoint(0.0, 0.0, 0.0, 0.0)]
        self.resolution = 1.0
        self.segment_lengths = np.array([1.0])


def make_car(tau):
    return BicycleModel(reference_path=_RefPathStub(), length=1.0, width=0.5,
                         Ts=0.1, actuator_lag_tau_s=tau)


# ---------------------------------------------------------------------------
# ①非矛盾性: SimpleSpatialStateの拡張(members/デフォルト値)
# ---------------------------------------------------------------------------

def test_simple_spatial_state_has_delta_actual_member():
    s = SimpleSpatialState()
    assert s.members == ['e_y', 'e_psi', 't', 'delta_actual']
    assert s.delta_actual == 0.0
    assert len(s) == 4


def test_simple_spatial_state_delta_actual_explicit_value():
    s = SimpleSpatialState(e_y=1.0, e_psi=0.2, t=0.0, delta_actual=0.05)
    assert s.delta_actual == pytest.approx(0.05)
    # SpatialState.__getitem__は整数indexでも常にリストを返す(既存仕様、s[0:2]等の
    # スライス構文と挙動を揃えるため)。
    assert s[3] == pytest.approx([0.05])


def test_bicycle_model_n_states_is_4():
    car = make_car(tau=0.055)
    assert car.n_states == 4
    assert len(car.spatial_state) == 4


def test_default_tau_is_190ms_when_unspecified():
    """mpc_controller.py/path_constraints_provider.pyのcreate_car()はtau引数を
    渡していないため、この既定値がそのまま使われる(config.yaml非依存、201節)。
    202/203節続報を経てtau=190msを確定・AXIS06クローズ(208節)。213節でtau=240ms
    (Test A、持続直線蛇行対策)を試したが、実測で持続直線std 3.17→4.40°・コーナー
    立ち上がり後std 6.10→7.14°と両方悪化したため190msへ復元した。"""
    car = BicycleModel(reference_path=_RefPathStub(), length=1.0, width=0.5, Ts=0.1)
    assert car.actuator_lag_tau_s == pytest.approx(0.19)


# ---------------------------------------------------------------------------
# ①非矛盾性(核心): tau=0(無効化)は旧3状態モデルと数値的に完全一致する
# ---------------------------------------------------------------------------

def _old_model_ABf(v_ref, kappa_ref, delta_s):
    """撤去前(3状態)のlinearize()と同一の計算(回帰確認用に再現)。"""
    a_1 = np.array([1, delta_s, 0])
    a_2 = np.array([-kappa_ref ** 2 * delta_s, 1, 0])
    b_1 = np.array([0, 0])
    b_2 = np.array([0, delta_s])
    if v_ref == 0:
        a_3 = np.array([0, 0, 1])
        b_3 = np.array([0, 0])
        f = np.array([0.0, 0.0, 0.0])
    else:
        a_3 = np.array([-kappa_ref / v_ref * delta_s, 0, 1])
        b_3 = np.array([-1 / (v_ref ** 2) * delta_s, 0])
        f = np.array([0.0, 0.0, 1 / v_ref * delta_s])
    A = np.stack((a_1, a_2, a_3), axis=0)
    B = np.stack((b_1, b_2, b_3), axis=0)
    return f, A, B


@pytest.mark.parametrize("v_ref,kappa_ref,delta_s", [
    (4.0, 0.1, 1.0),
    (1.5, -0.2, 0.6),
    (5.5, 0.0, 0.999),
])
def test_tau_zero_reproduces_old_3state_model_exactly(v_ref, kappa_ref, delta_s):
    car = make_car(tau=0.0)
    f, A, B = car.linearize(v_ref, kappa_ref, delta_s)
    f_old, A_old, B_old = _old_model_ABf(v_ref, kappa_ref, delta_s)

    # 旧モデルの3x3/3x2部分と一致すること
    np.testing.assert_allclose(A[:3, :3], A_old, atol=1e-9)
    np.testing.assert_allclose(B[:3, :], B_old, atol=1e-9)
    np.testing.assert_allclose(f[:3], f_old, atol=1e-9)

    # delta_actualはe_psi行に寄与せず(4列目=0)、delta_actual自体は指令へ瞬時追従
    assert A[1, 3] == pytest.approx(0.0, abs=1e-9)
    assert A[3, 3] == pytest.approx(0.0, abs=1e-9)
    assert B[3, 1] == pytest.approx(1.0, abs=1e-9)


def test_tau_zero_delta_actual_row_and_col_are_otherwise_zero():
    car = make_car(tau=0.0)
    f, A, B = car.linearize(v_ref=4.0, kappa_ref=0.1, delta_s=1.0)
    # delta_actual列(3列目)はe_y/t行に影響しない
    assert A[0, 3] == pytest.approx(0.0)
    assert A[2, 3] == pytest.approx(0.0)
    # delta_actual行(3行目)はvから影響を受けない
    assert B[3, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ①非矛盾性: 一般のtauでのalpha計算・e_psi行への正しい代入
# ---------------------------------------------------------------------------

def test_general_tau_alpha_and_e_psi_coupling():
    tau = 0.055
    v_ref, kappa_ref, delta_s = 4.0, 0.1, 1.0
    car = make_car(tau=tau)
    f, A, B = car.linearize(v_ref, kappa_ref, delta_s)

    alpha = min(1.0, max(0.0, delta_s / (tau * v_ref)))
    assert A[1, 3] == pytest.approx(delta_s * (1 - alpha))
    assert B[1, 1] == pytest.approx(delta_s * alpha)
    assert A[3, 3] == pytest.approx(1 - alpha)
    assert B[3, 1] == pytest.approx(alpha)
    # e_psi行の他の係数は旧モデルと不変
    assert A[1, 0] == pytest.approx(-kappa_ref ** 2 * delta_s)
    assert A[1, 1] == pytest.approx(1.0)


def test_alpha_is_clipped_to_1_when_step_time_exceeds_tau():
    """delta_s/(tau*v_ref) > 1 (1ステップでtauを超える)場合はalpha=1にクリップされ、
    delta_actualはそのステップ内で指令に完全追従する(異常値の吹き出し防止)。"""
    tau = 0.001  # 極端に短いtau
    car = make_car(tau=tau)
    f, A, B = car.linearize(v_ref=1.0, kappa_ref=0.0, delta_s=1.0)
    assert A[3, 3] == pytest.approx(0.0)
    assert B[3, 1] == pytest.approx(1.0)


def test_alpha_scales_inversely_with_speed_when_unsaturated():
    """同じtauでも速度が低いほど1ステップで進む"時間"(delta_s/v_ref)が長くなり、
    alphaが大きくなる(delta_actualがより追従しやすくなる)。alpha<1(未飽和)を保つため
    delta_sを意図的に小さくしている(現実的なwaypoint間隔0.6-1mでは下のテストの通り
    ほぼ常に飽和するため、この式自体の傾向を確認する目的の単体テスト)。"""
    tau = 0.055
    delta_s = 0.02  # 未飽和領域を作るための小さい値(現実のwaypoint間隔より遥かに小)
    car = make_car(tau=tau)
    _, A_slow, B_slow = car.linearize(v_ref=1.0, kappa_ref=0.0, delta_s=delta_s)
    _, A_fast, B_fast = car.linearize(v_ref=10.0, kappa_ref=0.0, delta_s=delta_s)
    assert B_slow[3, 1] < 1.0 and B_fast[3, 1] < 1.0  # 前提: 両方とも未飽和であること
    assert B_slow[3, 1] > B_fast[3, 1]


def test_realistic_waypoint_spacing_saturates_alpha_to_1_important_caveat():
    """④遡及効果・重要な設計上の注意点(201節続報での発見): 実際のwaypoint間隔
    (0.6-1m程度)とtau=55msの組み合わせでは、通常のレース速度域(5-20km/h)で
    delta_s/v_ref(1ホライズンステップが表す実時間、約0.1-0.4s)が常にtauを上回るため、
    alphaが1.0に飽和し、delta_actualは1ホライズンステップ以内に指令値へ完全収束する。
    つまりこの設計では、tau=55msはe_psi行(a_2[3])に事実上何の影響も与えず、旧3状態
    モデルと数値的に区別できない(tau=0時と同じ結果になる)。このテストは「壊れている」
    のではなく、このモデル化(ホライズンステップ単位の離散化)がtau=55msという短い
    時定数を表現するには粒度が粗すぎるという設計上の限界を、恒久的な回帰として
    記録するためのものである。"""
    tau = 0.055
    for v_kmh, delta_s in [(5, 0.6), (15, 0.6), (20, 0.6), (15, 1.0)]:
        v_ref = v_kmh / 3.6
        car = make_car(tau=tau)
        _, A, B = car.linearize(v_ref=v_ref, kappa_ref=0.0, delta_s=delta_s)
        assert B[3, 1] == pytest.approx(1.0), (
            f"v={v_kmh}km/h delta_s={delta_s}m で飽和しなかった(想定外)")
        assert A[1, 3] == pytest.approx(0.0)


def test_tau_130ms_engages_unsaturated_above_about_18kmh():
    """202節続報: tau=130ms(新既定値)は、waypoint間隔0.6mの場合、v>~18.5km/h
    (delta_s/v_ref<tau)で初めてalpha<1.0(未飽和)となりdelta_actualモデルが
    実際にe_psi行へ寄与する。15km/hは僅かに届かず依然飽和し(alpha≈1.108→1.0)、
    20km/hは未飽和(alpha≈0.83)になる。tau=55msでは全く飽和が解けなかった
    (test_realistic_waypoint_spacing_saturates_alpha_to_1_important_caveat)のと
    対照的に、130msなら少なくとも高速域では実際にモデルとして機能する。"""
    tau = 0.13
    car = make_car(tau=tau)

    _, A_15, B_15 = car.linearize(v_ref=15.0 / 3.6, kappa_ref=0.0, delta_s=0.6)
    assert B_15[3, 1] == pytest.approx(1.0)  # 15km/hは僅かに届かずまだ飽和

    _, A_20, B_20 = car.linearize(v_ref=20.0 / 3.6, kappa_ref=0.0, delta_s=0.6)
    assert B_20[3, 1] < 1.0, "20km/hで飽和してしまった(想定外)"
    assert A_20[1, 3] > 0.0

    # 低速(5km/h)は同じtauでも依然飽和する(dt_per_step=432ms > tau=130ms)
    _, A_slow, B_slow = car.linearize(v_ref=5.0 / 3.6, kappa_ref=0.0, delta_s=0.6)
    assert B_slow[3, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ①非矛盾性: v_ref==0(速度ゼロ)ではdelta_actualを凍結し、ゼロ除算しない
# ---------------------------------------------------------------------------

def test_v_ref_zero_freezes_delta_actual_and_does_not_raise():
    car = make_car(tau=0.055)
    f, A, B = car.linearize(v_ref=0.0, kappa_ref=0.1, delta_s=1.0)
    assert A[3, 3] == pytest.approx(1.0)
    assert B[3, 1] == pytest.approx(0.0)
    # e_psi行もv_ref=0時はdelta_actual項の寄与が消える(alpha=0 -> 係数delta_s)
    assert A[1, 3] == pytest.approx(1.0)  # delta_s * (1 - 0)
    assert B[1, 1] == pytest.approx(0.0)  # delta_s * 0
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(f))


# ---------------------------------------------------------------------------
# ①非矛盾性: get_temporal_derivatives/get_spatial_derivativesの拡張
# ---------------------------------------------------------------------------

def test_temporal_derivatives_uses_delta_actual_not_commanded_delta():
    car = make_car(tau=0.055)
    state = [0.0, 0.0, 0.0, 0.3]  # delta_actual=0.3, コマンドは別の値
    s_dot, psi_dot = car.get_temporal_derivatives(state, input=[4.0, 0.9], kappa=0.0)
    expected_psi_dot = 4.0 / car.length * np.tan(0.3)
    assert psi_dot == pytest.approx(expected_psi_dot)
    # 0.9(指令)を使った場合の値とは一致しないこと(delta_actualが使われている証拠)
    assert psi_dot != pytest.approx(4.0 / car.length * np.tan(0.9))


def test_spatial_derivatives_returns_4_elements_with_lag_term():
    car = make_car(tau=0.055)
    state = [0.0, 0.0, 0.0, 0.0]
    deriv = car.get_spatial_derivatives(state, input=[4.0, 0.1], kappa=0.0)
    assert deriv.shape == (4,)
    s_dot = 4.0  # e_y=0,e_psi=0 -> s_dot=v
    expected_d_delta_actual_d_s = (0.1 - 0.0) / (0.055 * s_dot)
    assert deriv[3] == pytest.approx(expected_d_delta_actual_d_s)


# ---------------------------------------------------------------------------
# ①非矛盾性(核心): t2s()がdelta_actualを幾何情報から独立して前回値から引き継ぐ
# ---------------------------------------------------------------------------

def test_t2s_carries_over_delta_actual_from_current_spatial_state():
    car = make_car(tau=0.055)
    car.spatial_state.delta_actual = 0.123
    new_state = car.t2s(reference_waypoint=car.current_waypoint,
                         reference_state=car.temporal_state)
    assert new_state.delta_actual == pytest.approx(0.123)


def test_t2s_defaults_delta_actual_to_zero_if_absent():
    car = make_car(tau=0.055)
    # 明示的にdelta_actual属性を持たない状態オブジェクトを模擬
    class _Bare:
        e_y, e_psi, t = 0.0, 0.0, 0.0
    car.spatial_state = _Bare()
    new_state = car.t2s(reference_waypoint=car.current_waypoint,
                         reference_state=car.temporal_state)
    assert new_state.delta_actual == pytest.approx(0.0)
