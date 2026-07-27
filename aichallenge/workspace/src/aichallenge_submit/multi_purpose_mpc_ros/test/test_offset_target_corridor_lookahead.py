"""Unit tests for the offset-target corridor lookahead fix (147節、2026-07-21):
switching lateral_target's clamp from a single current-cycle corridor point
(dbg_corr_ub0/lb0) to the minimum over the dynamic corridor array's near-term
horizon (dbg_corr_ub_arr/lb_arr, already exposed since 125節).

背景: 0720-07/08予選ログの深掘りで、コーナー内側でのオーバーテイク中に壁へ
寄り切る事象(wp270→282)を実測した。ENGAGE時点(wp270)ではLfree=2.96m・
Rfree=2.05mと相手車基準の空きは十分だったが、そこから約14m先で動的コリドー
(dbg_corr_lb0)がカーブ形状のみを理由に-3.54m→-0.89mまで一方的に収縮しており、
これは相手車の位置とは無関係な「壁単独の狭まり」だった。従来の
`self._mpc.lateral_target`クランプは`dbg_corr_ub0/lb0`(現在の1点のみ)を見て
おり、収縮が実際に到達するまでオフセット目標を緩め始めない「後追い」設計
だったため、車両側の横方向応答が追いつかないまま壁マージンがゼロまで
悪化していた。

`_switchback_wall_veto()`(125節)が既に使っている動的コリドー配列全体
(dbg_corr_ub_arr/lb_arr、MPC自身が毎周期解くQPコリドーの配列)の最小値を
新設ヘルパー`_corr_bound_ahead(side)`で取得し、単一点の代わりに使うことで、
狭まりが到達する前段階からオフセット目標自体を早めに緩める(新規パラメータ
0個、既存配列の再利用のみ)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー
実装した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

import numpy as np

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _corr_bound_ahead(side, ub_arr, lb_arr, ub0=float('inf'), lb0=-float('inf')):
    """_corr_bound_ahead()のミラー実装。"""
    arr = ub_arr if side > 0 else lb_arr
    if arr is None or len(arr) == 0:
        return ub0 if side > 0 else -lb0
    return float(np.min(arr)) if side > 0 else float(-np.max(arr))


# --- ①非矛盾性: 配列が使える場合は配列全体の最小(最も保守的な値)を返す ---

def test_left_side_uses_minimum_of_ub_array():
    """side>0(左)は、配列内の最小ub(=最も狭い箇所)を返す。実測(0720-07
    wp282直前の収縮)に相当する値: 途中はub0=3.19と広いが、その先2.43まで
    狭まる区間があれば2.43を返さねばならない。"""
    ub_arr = np.array([3.19, 2.80, 2.43, 2.60])
    assert _corr_bound_ahead(1, ub_arr, None) == 2.43


def test_right_side_uses_minimum_magnitude_of_lb_array():
    """side<0(右)は、lb_arr(負値)のうち最もゼロに近い値(=最も狭い箇所)の
    符号反転を返す。実測(wp282、lb0=-0.89)相当。"""
    lb_arr = np.array([-3.54, -3.03, -0.89, -1.50])
    assert _corr_bound_ahead(-1, None, lb_arr) == 0.89


def test_falls_back_to_single_point_when_array_unavailable():
    """②非冗長性/回帰防止: 配列が未取得(None、MPC初回ソルブ前)の場合は、
    従来通り単一点(dbg_corr_ub0/lb0)へ安全にフォールバックする。"""
    assert _corr_bound_ahead(1, None, None, ub0=0.5063966169647436) == 0.5063966169647436
    assert _corr_bound_ahead(-1, None, None, lb0=-3.541903985586557) == 3.541903985586557


def test_falls_back_when_array_empty():
    assert _corr_bound_ahead(1, np.array([]), None, ub0=1.5) == 1.5


# --- ④過去ログへの遡及効果 ---

def test_retroactive_0720_07_wp270_to_wp282_would_clamp_earlier():
    """0720-07実測(wp270→282、side=-1のインサイドオーバーテイク中の壁激突)。
    対処前は各周期の単一点(dbg_corr_lb0)のみを見るため、wp270時点では
    lb0=-3.54(広い)のままクランプが効かず、wp282でlb0=-0.89まで収縮して
    初めて反応していた。対処後、もしwp270の時点でこの後の収縮(wp282の
    -0.89)が既にMPCの先読み配列内に含まれていれば、wp270の時点から
    既に-0.89を検出しオフセット目標を早期に緩められることを確認する。"""
    # wp270時点でMPCが解いた配列に、この先の収縮(-0.89)が含まれていたと仮定
    lb_arr_at_wp270 = np.array([-3.54, -3.30, -2.10, -0.89])
    bound = _corr_bound_ahead(-1, None, lb_arr_at_wp270)
    assert bound == 0.89
    # 対処前(単一点)ならwp270時点ではlb0=-3.54(広い)のままだった
    old_single_point_bound = 3.54
    assert bound < old_single_point_bound


# --- ③配線確認: 実装がヘルパー経由になっていること ---

def test_lateral_target_uses_corr_bound_ahead_helper():
    idx = _SRC.index("self._mpc.lateral_target = float(self._ot_side) * _target_mag")
    # 2026-07-22追加(153節): 発生地点(何m先)の診断フィールド代入2行が間に
    #   挿入されたため、窓を400→600へ拡大(検証対象そのものは無変更)。
    # 2026-07-24追加(168節): 非正転落時に凍結保持する分岐(_ot_last_valid_target_mag)
    #   が間に挿入されたため、窓を600→1600へ再拡大(検証対象そのものは無変更)。
    snippet = _SRC[max(0, idx - 1600):idx]
    assert "self._corr_bound_ahead(self._ot_side)" in snippet


def test_corr_bound_ahead_reuses_existing_arrays_no_new_computation():
    """②非冗長性: 新規のコリドー計算を持ち込まず、125節で公開済みの
    dbg_corr_ub_arr/lb_arrをそのまま再利用することを確認する。"""
    idx = _SRC.index("def _corr_bound_ahead(")
    idx_end = _SRC.index("def _switchback_wall_veto(")
    snippet = _SRC[idx:idx_end]
    assert "self._mpc.dbg_corr_ub_arr" in snippet
    assert "self._mpc.dbg_corr_lb_arr" in snippet


def test_switchback_wall_veto_unaffected_regression():
    """①非矛盾性: 125節の_switchback_wall_vetoは無変更のまま独立して存在する
    ことを確認する(本対処は別ヘルパーの新設であり、既存関数の書き換えでは
    ない)。"""
    idx = _SRC.index("def _switchback_wall_veto(self) -> bool:")
    snippet = _SRC[idx:idx + 1200]
    assert "ub_arr = self._mpc.dbg_corr_ub_arr" in snippet
    assert "lb_arr = self._mpc.dbg_corr_lb_arr" in snippet
    assert "self._along_min_width" in snippet
