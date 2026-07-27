"""Unit tests for _predicted_time_to_wp (2026-07-15, ユーザー提案: コース形状は
既知なので「どのコーナーで相手に追いつくか」は逆算できるはず).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
method is extracted via AST from the real source file and bound to a minimal
mock `self`, exercising the ACTUAL production code (not a mirror).

Motivation: the engage-distance formula added in 69節 (_engage_dist_dynamic)
assumes the ego closes on the opponent at a constant rate (v_pot - vopp). But
the track's corner-by-corner reference speed profile (ref_vel_configulator) is
known in advance, so for a genuinely stopped/slow opponent (whose position is
effectively fixed), the ACTUAL time for the ego to physically arrive there can
be computed exactly by integrating that known profile along the path — capturing
planned deceleration through corners between the ego and the opponent, which the
constant-speed approximation cannot see.
"""
import ast
import os
import types

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")


def _extract_method(name):
    with open(_SRC_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f"{name} not found in {_SRC_PATH}")


def _kmh_to_m_per_sec(kmh):
    return kmh / 3.6


_NS = {"np": np, "kmh_to_m_per_sec": _kmh_to_m_per_sec}
exec(compile(_extract_method("_predicted_time_to_wp"), "<_predicted_time_to_wp>", "exec"), _NS)


class _RefPath:
    def __init__(self, circular=True, length=None):
        self.circular = circular
        self.length = length if length is not None else 0.0


class _RefVelConfigulator:
    """区間ごとのkm/h速度を返す簡易モック。start_wpごとの一定値を保持。"""

    def __init__(self, per_wp_kmh):
        self._per_wp_kmh = per_wp_kmh

    def get_ref_vel(self, wp):
        return self._per_wp_kmh[wp % len(self._per_wp_kmh)]


class _CfgMpc:
    def __init__(self, v_max_kmh=15.0):
        self.v_max = v_max_kmh


class _Cfg:
    def __init__(self, v_max_kmh=15.0):
        self.mpc = _CfgMpc(v_max_kmh)


def make_self(seg_lengths, per_wp_kmh, circular=True, ref_vel_configulator="default"):
    m = types.SimpleNamespace()
    n = len(seg_lengths)
    m._wp_s_cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))[:n]
    m._reference_path = _RefPath(circular=circular, length=float(np.sum(seg_lengths)))
    m._cfg = _Cfg()
    if ref_vel_configulator == "default":
        m._ref_vel_configulator = _RefVelConfigulator(per_wp_kmh)
    else:
        m._ref_vel_configulator = ref_vel_configulator
    m._predicted_time_to_wp = types.MethodType(_NS["_predicted_time_to_wp"], m)
    return m


def test_constant_speed_profile_matches_simple_division():
    """単純ケース: 全区間が同一速度なら、単純な距離/速度の合計と一致する
    (v_pot一定近似と同じ結果になることの確認)。"""
    seg_lengths = [2.0] * 10  # 1m/wp*2m間隔、計20m
    per_wp_kmh = [15.0] * 10  # 全区間15km/h=4.1667m/s
    m = make_self(seg_lengths, per_wp_kmh)
    t = m._predicted_time_to_wp(0, 5, max_dist=100.0)
    expected = (2.0 * 5) / _kmh_to_m_per_sec(15.0)
    assert t == pytest.approx(expected, rel=1e-6)


def test_corner_deceleration_increases_predicted_time_vs_constant_speed():
    """本修正の中核: 経路中に低速区間(コーナー)があると、その区間の実際の
    計画速度を反映して所要時間が伸びる。v_pot一定(15km/h)を仮定した単純計算
    より確実に長くなることを確認する。"""
    seg_lengths = [2.0] * 10
    # wp3-5だけコーナーで6km/hに減速する計画
    per_wp_kmh = [15.0, 15.0, 15.0, 6.0, 6.0, 6.0, 15.0, 15.0, 15.0, 15.0]
    m = make_self(seg_lengths, per_wp_kmh)
    t_profile = m._predicted_time_to_wp(0, 8, max_dist=100.0)
    t_naive_constant_speed = (2.0 * 8) / _kmh_to_m_per_sec(15.0)
    assert t_profile > t_naive_constant_speed  # コーナー減速を正しく織り込んでいる


def test_ref_vel_configulator_none_returns_none_for_fallback():
    """回帰: ref_vel_configulatorが無い場合はNoneを返し、呼び出し側が
    v_pot近似(_engage_dist_dynamic)へフォールバックできるようにする。"""
    m = make_self([2.0] * 5, [15.0] * 5, ref_vel_configulator=None)
    assert m._predicted_time_to_wp(0, 3, max_dist=100.0) is None


def test_target_beyond_max_dist_returns_none_bounded_cost():
    """回帰: to_wpがmax_dist(既存fwd_max_considerと同じ範囲)より遠い場合は
    Noneを返す(無制限に歩き続けない=計算コストが有界)。"""
    seg_lengths = [2.0] * 20
    m = make_self(seg_lengths, [15.0] * 20)
    t = m._predicted_time_to_wp(0, 15, max_dist=10.0)  # 30m先、上限10mを大幅超過
    assert t is None


def test_circular_wraparound_handles_wp_before_start():
    """回帰: 周回コースで、to_wpが数値上from_wpより小さい(周回して手前に見える)
    場合でも、circular=Trueなら正しく1周分の弧長を辿って計算できる。"""
    seg_lengths = [2.0] * 10  # 全周20m
    per_wp_kmh = [15.0] * 10
    m = make_self(seg_lengths, per_wp_kmh, circular=True)
    t = m._predicted_time_to_wp(8, 2, max_dist=100.0)  # wp8->9->0->1->2、4区間
    expected = (2.0 * 4) / _kmh_to_m_per_sec(15.0)
    assert t == pytest.approx(expected, rel=1e-6)


def test_same_wp_returns_zero_regression():
    """回帰: from_wp == to_wpの場合は所要時間0を返す(既にそこにいる)。"""
    m = make_self([2.0] * 5, [15.0] * 5)
    assert m._predicted_time_to_wp(3, 3, max_dist=100.0) == pytest.approx(0.0)


def test_get_ref_vel_exception_falls_back_to_v_max():
    """回帰: get_ref_velが例外を投げる区間があっても、cfg.mpc.v_max(既存の
    絶対上限)にフォールバックして計算を続行する(クラッシュしない)。"""
    class _Raising:
        def get_ref_vel(self, wp):
            raise ValueError("no data for this section")
    m = make_self([2.0] * 5, [15.0] * 5, ref_vel_configulator=_Raising())
    t = m._predicted_time_to_wp(0, 3, max_dist=100.0)
    expected = (2.0 * 3) / _kmh_to_m_per_sec(15.0)  # cfg.mpc.v_max=15.0のフォールバック
    assert t == pytest.approx(expected, rel=1e-6)
