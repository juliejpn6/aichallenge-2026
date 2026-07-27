"""Unit tests for _switchback_curvature_veto (76節対処案①、2026-07-15).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
method is extracted via AST from the real source file and bound to a minimal
mock `self`, exercising the ACTUAL production code (not a mirror).

Motivation: 0715-06実測でhas_rescued(75節)導入後にswitchback頻度が6→11回に倍増し
(A_rescue分+5)、COLLISION-SUSPECTEDが0→7回に増加する回帰が発生した。根本原因の一つは、
switchback判定(通常branch=A・A_rescue共通)がその瞬間のspace/opp_spaceのみで反転先を
決めており、_plan_passのk_corner先読みveto相当のロジックが無いこと(wp297のA_rescue
×2件、反転5〜6秒後にRfreeが1.0〜1.9mまで縮小してCOLLISION-SUSPECTEDが発火)。
_switchback_curvature_vetoは、_plan_passのk_corner検出と同じ閾値
(_ot_pass_block_kappa)・窓(_fwd_max_consider)を再利用し、反転先を閉じる方向の
強いコーナーが直近にあれば反転を抑制する。
"""
import ast
import os
import types

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


_NS = {}
exec(compile(_extract_method("_switchback_curvature_veto"),
             "<_switchback_curvature_veto>", "exec"), _NS)


class _Wp:
    def __init__(self, kappa):
        self.kappa = kappa


class _RefPath:
    def __init__(self, waypoints, circular=True, length=None):
        self.waypoints = waypoints
        self.circular = circular
        self.length = length if length is not None else 0.0


class _Model:
    def __init__(self, wp_id):
        self.wp_id = wp_id


class _Mpc:
    def __init__(self, wp_id):
        self.model = _Model(wp_id)


def make_self(kappas, wp_id=0, seg_len=1.0, fwd_max_consider=20.0,
              pass_block_kappa=0.10, circular=True):
    m = types.SimpleNamespace()
    wps = [_Wp(k) for k in kappas]
    m._reference_path = _RefPath(wps, circular=circular,
                                  length=seg_len * len(kappas))
    m._wp_s_cum = [seg_len * i for i in range(len(kappas))]
    m._mpc = _Mpc(wp_id)
    m._fwd_max_consider = fwd_max_consider
    m._ot_pass_block_kappa = pass_block_kappa
    m._switchback_curvature_veto = types.MethodType(
        _NS["_switchback_curvature_veto"], m)
    return m


def test_no_corner_in_window_never_vetoes():
    """回帰: 窓内にきついコーナー(|kappa|>=閾値)が無ければ、どちらの側へも反転を
    抑制しない。"""
    m = make_self([0.0] * 30, wp_id=0)
    assert m._switchback_curvature_veto(new_side=1) is False
    assert m._switchback_curvature_veto(new_side=-1) is False


def test_left_corner_vetoes_flip_to_left_side():
    """本修正の中核: kappa>0(左コーナー)が窓内にあれば、その内側=左(+1)への
    反転のみ抑制する。"""
    kappas = [0.0] * 5 + [0.15] + [0.0] * 24  # wp5に左コーナー(0.15>=0.10)
    m = make_self(kappas, wp_id=0)
    assert m._switchback_curvature_veto(new_side=1) is True   # 左へは反転させない
    assert m._switchback_curvature_veto(new_side=-1) is False  # 右は無関係


def test_right_corner_vetoes_flip_to_right_side():
    """左コーナーの鏡像: kappa<0(右コーナー)は右(-1)への反転のみ抑制する。"""
    kappas = [0.0] * 5 + [-0.15] + [0.0] * 24
    m = make_self(kappas, wp_id=0)
    assert m._switchback_curvature_veto(new_side=-1) is True
    assert m._switchback_curvature_veto(new_side=1) is False


def test_corner_below_threshold_does_not_veto():
    """回帰: 閾値未満の緩いカーブでは抑制しない(_plan_passのk_cornerと同じ閾値)。"""
    kappas = [0.0] * 5 + [0.05] + [0.0] * 24  # 0.05 < 0.10
    m = make_self(kappas, wp_id=0)
    assert m._switchback_curvature_veto(new_side=1) is False


def test_corner_beyond_fwd_max_consider_window_not_seen():
    """回帰: _fwd_max_consider窓の外にあるコーナーは無視する(遠すぎて今は
    無関係な地形まで反転を渋らない)。"""
    kappas = [0.0] * 25 + [0.15] + [0.0] * 4  # 25m先(窓20mの外)
    m = make_self(kappas, wp_id=0, seg_len=1.0, fwd_max_consider=20.0)
    assert m._switchback_curvature_veto(new_side=1) is False


def test_only_first_corner_in_window_matters():
    """境界値: 窓内に複数のコーナーがあっても、最初に見つかったものだけで
    判定する(_plan_passのk_corner検出と同じ「最初の1つ」方式)。"""
    kappas = [0.0] * 3 + [-0.15] + [0.0] * 2 + [0.15] + [0.0] * 23
    m = make_self(kappas, wp_id=0)
    # 最初に見つかるのは右コーナー(wp3)なので、右への反転のみ抑制される
    assert m._switchback_curvature_veto(new_side=-1) is True
    assert m._switchback_curvature_veto(new_side=1) is False


def test_wraps_around_circular_course():
    """回帰: 周回コースで、現在wp_idが終端付近でも0番目へラップして正しく
    窓内を走査できる。"""
    n = 30
    kappas = [0.0] * n
    kappas[3] = 0.15  # ラップ後3周期先に左コーナー
    m = make_self(kappas, wp_id=n - 2, seg_len=1.0, fwd_max_consider=20.0)
    assert m._switchback_curvature_veto(new_side=1) is True


def test_reference_path_access_failure_returns_false_safely():
    """回帰: reference_path等へのアクセスが例外を投げても(想定外の初期化順序等)、
    安全側(抑制しない=既存挙動を維持)にフォールバックしクラッシュしない。"""
    m = types.SimpleNamespace()
    m._switchback_curvature_veto = types.MethodType(
        _NS["_switchback_curvature_veto"], m)
    assert m._switchback_curvature_veto(new_side=1) is False
