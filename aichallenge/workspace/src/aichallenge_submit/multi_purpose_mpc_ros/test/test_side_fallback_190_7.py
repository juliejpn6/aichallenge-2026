"""Unit tests for 190-7節(2026-07-26): _plan_passの障害物分岐(vopp<opp_obstacle_speed)
に反対側フォールバックを追加する対処。

背景: 5日分18ログの機械的横断調査(190節)で、0722-04ログにおいて選んだ側
(argmaxで決まった_side)が2台目の存在(K-check/kveto)によって35秒以上ブロック
され続け、その間ずっと反対側は一度も試されていなかった実例を発見した
(`[K-CHECK] blocked side=-1 ... room=2.12 combined=0.71 need=1.45`——選んだ側
自体は2.12mと十分広いが、2台目を踏まえた実効幅だけが不足していた)。

過去の類似事例(design_docs/stage15_perf_20260707.html 917-922行、0713-03)は
K-check自体の判定式(2台目を踏まえた_combined計算)のバグで、既に2026-07-13に
修正済みだった。今回0722-04で見た値(combined=0.71〜1.37 < need=1.45)は、その
修正後の正しい計算結果であり、K-check自体にバグは無い。真の欠落は
「_side_blocked_by_other_carで選んだ側がブロックされた場合、_plan_passが
反対側を一切試さずに即座にFalseを返す」という制御フロー側にあった。

対処: 選んだ側がnarrow(_room_debounce_ok不成立)またはkveto(K-check)で
失敗した場合のみ、反対側を同じ2つのチェックで即座に試す。反対側の
room_debounce_okは独立counter_key="fallback"を使うため、主系統
(counter_key="primary"、vid/side変化で即リセットする既存の反転抑制設計)には
一切影響しない。

mpc_controller.pyはrclpy依存で直接importできないため、test_plan_pass_kcorner.py
と同じ手法(AST抽出した実メソッドをmock selfへバインド)を用いる。
"""
import ast
import os
import types

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")


def _extract_methods(names):
    with open(_SRC_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    found[item.name] = ast.get_source_segment(src, item)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"methods not found in {_SRC_PATH}: {missing}")
    return found


_METHOD_NAMES = ["_plan_pass", "_side_blocked_by_other_car", "_plan_obs_log",
                 "_room_debounce_ok"]
_METHODS_SRC = _extract_methods(_METHOD_NAMES)
_NS = {"np": np}
for _name, _src in _METHODS_SRC.items():
    exec(compile(_src, f"<{_name}>", "exec"), _NS)

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


class WP:
    def __init__(self, ub, lb, kappa=0.0):
        self.ub = ub
        self.lb = lb
        self.kappa = kappa


class _RefPath:
    def __init__(self, waypoints, circular=True):
        self.waypoints = waypoints
        self.circular = circular
        self.length = 1000.0


class _Model:
    def __init__(self, wp_id):
        self.wp_id = wp_id
        self.safety_margin = 1.626


class _MPCStub:
    def __init__(self, wp_id):
        self.model = _Model(wp_id)
        self.safety_margin_override = None


def make_self(waypoints, wp_id=0, engage_debounce=1):
    m = types.SimpleNamespace()
    m._reference_path = _RefPath(waypoints)
    m._mpc = _MPCStub(wp_id)
    n = len(waypoints)
    m._wp_s_cum = np.arange(n, dtype=float)
    m._opp_obstacle_speed = 1.67
    m._opp_min_closing = 0.7
    m._v_pot = 4.17
    m._ot_block_half = 0.4
    m._ot_pass_clear = 3.0
    m._ot_t_lateral = 3.0
    m._ot_pass_block_kappa = 0.3
    m._along_lane_need = 1.85
    m._along_min_width = 1.45
    m._ot_engage_debounce = engage_debounce
    m._plan_obs_prev_result = None
    m._plan_obs_log_count = 0
    m._plan_moving_log_count = 0
    m._plan_fail_prev_reason = None
    m._plan_fail_log_count = 0
    m._plan_room_ok_count = 0
    m._plan_room_prev_vid = None
    m._plan_room_prev_side = None
    m._plan_room_ok_count_by_key = {}
    m._plan_room_prev_vid_by_key = {}
    m._plan_room_prev_side_by_key = {}
    m._log_calls = []
    m.get_logger = lambda: types.SimpleNamespace(
        info=lambda msg, *a, **k: m._log_calls.append(msg),
        warn=lambda msg, *a, **k: m._log_calls.append(msg))
    m._plan_pass = types.MethodType(_NS["_plan_pass"], m)
    m._side_blocked_by_other_car = types.MethodType(_NS["_side_blocked_by_other_car"], m)
    m._plan_obs_log = types.MethodType(_NS["_plan_obs_log"], m)
    m._room_debounce_ok = types.MethodType(_NS["_room_debounce_ok"], m)
    return m


def make_scan(fwd_ds, fwd_lat, vopp, fwd_vid="d3", fwd_wp=5, cars=None):
    return {"fwd_ds": fwd_ds, "fwd_vopp": vopp, "fwd_vid": fwd_vid, "fwd_wp": fwd_wp,
            "fwd_lat": fwd_lat, "fwd_dlat": abs(fwd_lat), "cars": cars or []}


# ---------------------------------------------------------------------------
# ①非矛盾性: kvetoで主系統側が失敗 → 反対側が空いていれば救済される
# ---------------------------------------------------------------------------

def test_kveto_blocked_side_falls_back_to_open_opposite_side():
    """核心: 選んだ側(左、lf=3.6が広くargmaxで選ばれる)が2台目(c2)でkvetoされ、
    反対側(右、rf=1.6)は2台目の影響を受けず十分空いている場合、反対側へ
    フォールバックしてengageが成立する(0722-04実測の再現)。
    c2はlat=2.3(左側)に配置し、c_room=ub-(c_lat+block_half)=4.0-(2.3+0.4)=1.3<1.45
    となるようにして、combined=min(lf=3.6, c_room=1.3)=1.3<need(1.45)で確実に
    左側をkvetoさせる(c_lat=2.0だとcombined=1.6でblockされないため、既存の
    test_obstacle_branch_plan_pass_uses_along_min_width_for_k_checkとは異なる値を使う)。"""
    wps = [WP(ub=4.0, lb=-2.0) for _ in range(20)]  # lf=3.6(左が広い→argmaxで選ばれる)
    m = make_self(wps, wp_id=0, engage_debounce=1)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3",
                      cars=[(3.0, 2.3, 0.0, 2.3, "c2", 0)])
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == -1  # 右へフォールバック
    assert any("[SIDE-FALLBACK]" in msg for msg in m._log_calls)


def test_both_sides_blocked_still_fails_as_before():
    """回帰: 反対側も物理的に狭ければ、従来通りengageしない(安全性は失われない)。"""
    wps = [WP(ub=1.7, lb=-1.7) for _ in range(20)]  # lf=rf=1.3(両側とも1.45未満)
    m = make_self(wps, wp_id=0, engage_debounce=1)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3")
    ok, side, _req = m._plan_pass(scan)
    assert ok is False
    assert side == 0
    assert m._dbg_plan_reason == "narrow"
    assert not any("[SIDE-FALLBACK]" in msg for msg in m._log_calls)


def test_primary_side_success_does_not_invoke_fallback():
    """①非矛盾性: 主系統側がそもそも成立する通常ケースでは、フォールバックの
    ログすら出力されない(既存の成功パスに影響を与えない)。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0, engage_debounce=1)
    scan = make_scan(fwd_ds=6.0, fwd_lat=-2.0, vopp=0.0)  # 0713-06型、通常通り左が選ばれ成立
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert not any("[SIDE-FALLBACK]" in msg for msg in m._log_calls)


# ---------------------------------------------------------------------------
# ②非冗長性・③検証: フォールバック用カウンタは主系統から完全に独立
# ---------------------------------------------------------------------------

def test_fallback_counter_independent_of_primary_counter():
    """主系統(counter_key="primary")のデバウンス状態を汚さずに、フォールバック
    (counter_key="fallback")が独立して積算されることを確認する。"""
    wps = [WP(ub=4.0, lb=-2.0) for _ in range(20)]
    m = make_self(wps, wp_id=0, engage_debounce=1)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3",
                      cars=[(3.0, 2.3, 0.0, 2.3, "c2", 0)])
    m._plan_pass(scan)
    # 主系統は"左"(argmax勝者)を評価し続けたはずで、フォールバックキーとは別に記録される
    assert "primary" in m._plan_room_prev_side_by_key or m._plan_room_prev_side == 1
    assert m._plan_room_prev_side_by_key.get("fallback") == -1


def test_room_debounce_ok_primary_path_byte_identical_when_key_omitted():
    """④遡及効果: counter_keyを省略した呼び出しは、190-7節導入前と完全に同一の
    挙動(単一スカラーself._plan_room_ok_count)であることを確認する。"""
    m = make_self([WP(ub=4.0, lb=-4.0)], wp_id=0, engage_debounce=2)
    assert m._room_debounce_ok("d3", 1, 2.0, need=1.45) is False  # 1回目
    assert m._plan_room_ok_count == 1
    assert m._room_debounce_ok("d3", 1, 2.0, need=1.45) is True   # 2回目で成立
    assert m._plan_room_ok_count == 2
    # フォールバック専用の辞書は一切触られていない
    assert m._plan_room_ok_count_by_key == {}


# ---------------------------------------------------------------------------
# ソーステキスト検証
# ---------------------------------------------------------------------------

def test_source_fallback_uses_independent_counter_key():
    idx = _SRC.index("def _side_fail_reason(")
    idx_end = _SRC.index("_fail_reason = _side_fail_reason(_side, _room,")
    snippet = _SRC[idx:idx_end]
    assert 'counter_key' in snippet


def test_source_fallback_only_triggers_on_primary_failure():
    idx = _SRC.index("_fail_reason = _side_fail_reason(_side, _room, \"primary\")")
    snippet = _SRC[idx:idx + 500]
    assert "if _fail_reason is not None:" in snippet
    assert '_side_fail_reason(_fb_side, _fb_room, "fallback")' in snippet


def test_source_room_debounce_ok_default_counter_key_is_primary():
    idx = _SRC.index("def _room_debounce_ok(")
    snippet = _SRC[idx:idx + 200]
    assert 'counter_key: str = "primary"' in snippet
