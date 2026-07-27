"""Unit tests for _switchback_wall_veto (125節, A-1, 2026-07-20).

Background: switchbackの側反転判定(_switchback_curvature_veto)は静的な
per-waypointトラック曲率(kappa)しか見ておらず、対向車のoccupancyや実際の
壁形状を反映しない。ユーザー指摘(「どのみち空いている隙間からしか抜けない
のだから、素直に隙間の有無を見るべきでは」)を受け、静的テーブルではなく
MPC自身が毎周期実際に解いている動的コリドー(壁+占有格子込み、wall_slow
(124節)が消費するdbg_corr_ub0/lb0と同一の_corridor()計算)の配列全体
(dbg_corr_ub_arr/lb_arr、core/MPC.py側で新規計算ゼロで公開)を再利用する。

設計上の制約(ユーザーへ開示済み、テストでも明示的に検証する): この配列は
update_path_constraints()が「複数の空きセグメントがある場合は面積最大の
経路を1本選ぶ」設計であるため、new_side固有の空きではなく「MPCが現在
計画している単一経路」の幅を表す。よって_switchback_wall_vetoはnew_sideを
引数に取らず、その計画経路自体がalong_min_width(カート幅未満の物理下限、
_opponent_room_aheadと同一の既存閾値)を下回るほど狭い区間が先読み内に
あるかを見る、保守的なwall_slowの先読み版として機能する。

mpc_controller.pyはautoware_auto_control_msgs等のROSメッセージ型をモジュール
スコープでimportしており単体テスト環境では直接importできないため、
test_switchback_curvature_veto.pyと同じくASTで実物のメソッドを抽出し、
最小のmock selfへバインドして本番コードそのものを検証する。
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


_NS = {}
exec(compile(_extract_method("_switchback_wall_veto"),
             "<_switchback_wall_veto>", "exec"), _NS)


class _FakeMpc:
    def __init__(self, ub_arr, lb_arr):
        self.dbg_corr_ub_arr = None if ub_arr is None else np.asarray(ub_arr, dtype=float)
        self.dbg_corr_lb_arr = None if lb_arr is None else np.asarray(lb_arr, dtype=float)


def make_self(ub_arr, lb_arr, along_min_width=1.45):
    m = types.SimpleNamespace()
    m._mpc = _FakeMpc(ub_arr, lb_arr)
    m._along_min_width = along_min_width
    m._switchback_wall_veto = types.MethodType(_NS["_switchback_wall_veto"], m)
    return m


def test_no_narrow_point_never_vetoes():
    """回帰: 先読み全区間の幅がalong_min_width以上なら反転を抑制しない。"""
    m = make_self(ub_arr=[3.0] * 20, lb_arr=[-3.0] * 20)
    assert m._switchback_wall_veto() is False


def test_single_narrow_point_vetoes():
    """本修正の中核: 先読み内の1点でも幅(ub-lb)がalong_min_width未満なら
    反転を抑制する。"""
    ub = [3.0] * 20
    lb = [-3.0] * 20
    ub[10] = -1.5  # wp10で幅 = -1.5 - (-3.0) = 1.5m >= 1.45m、まだ抑制しない境界の外側を確認
    m = make_self(ub_arr=ub, lb_arr=lb)
    assert m._switchback_wall_veto() is False
    ub[10] = -1.6  # 幅 = -1.6-(-3.0) = 1.4m < 1.45m
    m2 = make_self(ub_arr=ub, lb_arr=lb)
    assert m2._switchback_wall_veto() is True


def test_boundary_exactly_at_along_min_width_does_not_veto():
    """境界値: 幅がちょうどalong_min_widthと等しい場合は抑制しない(<のみ)。"""
    ub = [1.45 / 2.0]
    lb = [-1.45 / 2.0]
    m = make_self(ub_arr=ub, lb_arr=lb)
    assert m._switchback_wall_veto() is False


def test_opponent_occupied_corridor_narrows_and_vetoes():
    """対向車が窓内でコリドーを狭めているケース(占有格子由来を模擬)。
    壁自体はub=3.0/lb=-3.0で十分広いが、対向車occupancyによりwp5だけ
    lb=1.0まで押し上げられ、幅=3.0-1.0=2.0m。along_min_width=1.45未満に
    はならないため抑制しない一方、対向車がさらに幅寄せしてlb=1.7まで
    来た場合(幅=1.3m<1.45m)は抑制する、という2ケースを確認する。"""
    ub = [3.0] * 20
    lb_wide = [-3.0] * 5 + [1.0] + [-3.0] * 14
    m_wide = make_self(ub_arr=ub, lb_arr=lb_wide)
    assert m_wide._switchback_wall_veto() is False

    lb_narrow = [-3.0] * 5 + [1.7] + [-3.0] * 14
    m_narrow = make_self(ub_arr=ub, lb_arr=lb_narrow)
    assert m_narrow._switchback_wall_veto() is True


def test_uses_configured_along_min_width_no_new_magic_number():
    """②非冗長性: 閾値は呼び出し元のself._along_min_widthをそのまま使い、
    メソッド内部に独自の数値を持たない。"""
    ub = [0.5]
    lb = [-0.5]  # 幅=1.0m
    m_default = make_self(ub_arr=ub, lb_arr=lb, along_min_width=1.45)
    assert m_default._switchback_wall_veto() is True  # 1.0 < 1.45
    m_loose = make_self(ub_arr=ub, lb_arr=lb, along_min_width=0.5)
    assert m_loose._switchback_wall_veto() is False  # 1.0 >= 0.5


def test_none_array_fails_open():
    """回帰: dbg_corr_ub_arr/lb_arrが未初期化(None、MPC.py初回get_control前)の
    場合は安全側(抑制しない=fail-open)にフォールバックする。"""
    m = make_self(ub_arr=None, lb_arr=None)
    assert m._switchback_wall_veto() is False


def test_empty_array_fails_open():
    """回帰: 配列が空(N=0相当)の場合も抑制しない。"""
    m = make_self(ub_arr=[], lb_arr=[])
    assert m._switchback_wall_veto() is False


def test_far_point_beyond_mpc_horizon_still_checked():
    """回帰(設計上の制約の確認): 動的配列はMPCのQP horizon分(約N*resolution)
    しかカバーしないため、配列の末尾(=先読みの最遠点)が狭くても正しく
    検知できることを確認する(20点先読みの最終点)。"""
    ub = [3.0] * 20
    lb = [-3.0] * 19 + [1.6]  # 最後の点だけ幅=1.4m<1.45m
    m = make_self(ub_arr=ub, lb_arr=lb)
    assert m._switchback_wall_veto() is True


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_switchback_wall_veto_reuses_along_min_width_no_new_magic_number():
    idx = _SRC.index("def _switchback_wall_veto")
    snippet = _SRC[idx:idx + 1800]
    assert "self._along_min_width" in snippet
    assert "dbg_corr_ub_arr" in snippet
    assert "dbg_corr_lb_arr" in snippet


def test_new_side_wall_blocked_computed_and_gated_on_ot_side():
    idx = _SRC.index("_new_side_wall_blocked = (")
    snippet = _SRC[idx:idx + 200]
    assert "self._switchback_wall_veto()" in snippet
    assert "if self._ot_side != 0 else False" in snippet


def test_new_side_wall_blocked_passed_to_update_call():
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx_update:idx_update + 900]
    assert "new_side_wall_blocked=_new_side_wall_blocked" in snippet


def test_new_side_wall_blocked_computed_before_update():
    idx_wall = _SRC.index("_new_side_wall_blocked = (")
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_wall < idx_update


def test_wall_reason_added_to_switchback_suppressed_reason_string():
    idx = _SRC.index('_reason = ("cleared_margin"')
    snippet = _SRC[idx:idx + 300]
    assert '"wall" if _lat_dec.switchback_wall_blocked' in snippet


def test_wall_blocked_logged_in_giveup_trigger_line():
    idx = _SRC.index('f"[LAT-TTC-ACT] giveup trigger=')
    snippet = _SRC[idx:idx + 500]
    assert "wall_blocked={_lat_dec.switchback_wall_blocked}" in snippet
