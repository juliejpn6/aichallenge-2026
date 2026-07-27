"""Unit tests for _follow_speed_limit's force_include_vid parameter (2026-07-15,
0715-02実測で確認したswitchback直後のICC見失いバグ対策).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
methods under test are extracted via AST from the real source file and bound
to a minimal mock `self`, exercising the ACTUAL production code.

Bug: right after a side switchback, the ego's own offset has not yet ramped
onto the new side (alpha still low), but the opponent's raw fwd_dlat still
reflects the LARGE separation that existed on the OLD side. If that stale
dlat happens to already exceed near_sep, _follow_speed_limit's near-range
exclusion drops the opponent from consideration entirely (_vlim=None), so the
overall speed candidate stack falls through to eff_v_cap/line_cap (near-full
speed) while the ego is still almost directly behind/beside the opponent.
Confirmed via 0715-02 log replay: t=434.35, dlat=2.31 (>= near_sep=1.8, stale
from the pre-switch side), offset=-0.31 (barely started), d_min=2.99m,
v_safe=4.166 (full) — 0.5s before a [COLLISION-SUSPECTED] event.

Fix: force_include_vid lets the caller (mpc_controller.py's _control(), only
while OVERTAKING with a locked side and _ot_alpha < 1.0-1e-3) exempt exactly
the currently-tracked opponent (fwd_vid) from the near/far exclusion checks,
so ICC keeps braking for it appropriately until the offset commitment
completes. No other car is affected.
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


_METHOD_NAMES = ["_follow_speed_limit", "_g2_speed"]
_METHODS_SRC = _extract_methods(_METHOD_NAMES)
_NS = {"np": np}
for _name, _src in _METHODS_SRC.items():
    exec(compile(_src, f"<{_name}>", "exec"), _NS)


def make_self():
    m = types.SimpleNamespace()
    m._fwd_max_consider = 20.0
    m._fwd_near_range = 6.0
    m._fwd_lateral_halfwidth = 1.5
    m._fwd_a_brake = 1.3
    m._fwd_margin_center = 4.0
    m._opp_map = None
    m._g2_speed = types.MethodType(_NS["_g2_speed"], m)
    m._follow_speed_limit = types.MethodType(_NS["_follow_speed_limit"], m)
    return m


def make_scan(cars):
    return {"cars": cars}


def test_regression_far_dlat_excludes_car_without_force_include():
    """回帰: force_include_vid未指定なら、従来通りdlat>=near_sepの車は除外される
    (_vlim=None、対象なし)。"""
    m = make_self()
    scan = make_scan([(2.99, 2.31, 0.0, 2.31, "d3", 178)])  # ds=2.99<near_range, dlat=2.31>=1.8
    vlim, vtgt = m._follow_speed_limit(scan, path_offset=-0.31, near_sep=1.8)
    assert vlim is None
    assert vtgt is None


def test_retroactive_0715_02_switchback_blind_spot_now_captured():
    """遡及検証(0715-02実測、t=434.35秒、実際に衝突0.5秒前に全開速度が出ていた周期):
    ds=2.99, dlat=2.31(旧側の値のまま), near_sep=1.8(未クリア)という当時の実測値を
    force_include_vid="d3"付きで再生すると、除外されずICCが正しく対象車を捕捉し、
    g2_speedによる制動速度を返すことを確認する(修正前はNone=全開へフォールバック)。"""
    m = make_self()
    scan = make_scan([(2.99, 2.31, 1.5, 2.31, "d3", 178)])
    vlim, vtgt = m._follow_speed_limit(
        scan, path_offset=-0.31, near_sep=1.8, force_include_vid="d3")
    assert vlim is not None
    assert vtgt is not None
    assert vtgt[4] == "d3"
    # 参考: 除外されていれば全開(eff_v_cap等)にフォールバックしていたはずの状況で、
    # 正しくg2_speed(制動計算)による有限の速度が返る
    assert vlim < 4.166


def test_force_include_vid_does_not_affect_unrelated_cars():
    """回帰: force_include_vidに一致しない他の車(例: 2台目)には、従来通りの
    除外判定がそのまま適用される(無関係な車まで巻き込まない)。"""
    m = make_self()
    scan = make_scan([
        (2.99, 2.31, 1.5, 2.31, "d3", 178),   # force_include対象(捕捉されるべき)
        (3.50, 5.00, 2.0, 5.00, "c2", 200),   # 全く無関係、遠方帯の外(捕捉されないべき)
    ])
    vlim, vtgt = m._follow_speed_limit(
        scan, path_offset=-0.31, near_sep=1.8, force_include_vid="d3")
    assert vtgt is not None
    assert vtgt[4] == "d3"  # c2ではなくd3が選ばれる(c2は無関係のまま除外)


def test_force_include_vid_none_is_fully_backward_compatible():
    """回帰: force_include_vid=None(デフォルト、alpha到達時や非OVERTAKING時)は、
    引数を追加する前と完全に同じ挙動になる。"""
    m = make_self()
    scan = make_scan([(2.99, 2.31, 1.5, 2.31, "d3", 178)])
    vlim, vtgt = m._follow_speed_limit(scan, path_offset=-0.31, near_sep=1.8)
    assert vlim is None
    assert vtgt is None


def test_force_include_vid_still_respects_fwd_max_consider():
    """回帰: force_include_vidを指定しても、fwd_max_consider(20m)より遠い車まで
    無条件に捕捉することはない(最外周の安全上限は維持)。"""
    m = make_self()
    scan = make_scan([(25.0, 2.31, 1.5, 2.31, "d3", 500)])  # ds=25m > fwd_max_consider(20m)
    vlim, vtgt = m._follow_speed_limit(
        scan, path_offset=-0.31, near_sep=1.8, force_include_vid="d3")
    assert vlim is None
    assert vtgt is None


def test_boundary_dlat_exactly_at_near_sep_without_force_include_regression():
    """境界値回帰: force_include無しの場合、dlat==near_sepちょうどは`>=`により除外される
    (既存の厳密比較を維持)。"""
    m = make_self()
    scan = make_scan([(3.0, 1.8, 1.0, 1.8, "d3", 100)])
    vlim, vtgt = m._follow_speed_limit(scan, path_offset=0.0, near_sep=1.8)
    assert vlim is None


# ---------------------------------------------------------------------------
# ds_eff = ds + dlat: 既に横に離れている分だけ実効ギャップとして認める (80節, 2026-07-16)
# ---------------------------------------------------------------------------
# 背景: _g2_speed(v_fwd, ds)はdsのみで速度上限を決めており、既にdlat(横間隔)が
# 育っていても「真後ろ」と全く同じ強さで絞り続けていた。0716-01実測(Lap1第3コーナー、
# インからの追い越し初成功)で、wp61〜127の約26秒間、速度が終始5km/h帯(icc_f3クリープ床
# 由来)に留まり、G-2解放(eff_v_cap)にも切り替わらないまま推移する事象を確認した
# (この間dlatは0.3〜2.0mまで断続的に育っていたが、_g2_speedの入力dsには一切反映
# されていなかった)。F3-TAPER(_est_gap=fwd_ds+fwd_dlat、0714-03で導入済み)と全く同じ
# 「横に離れていれば縦距離に関わらず安全」という考え方を、ここでも再利用する。

def test_ds_eff_increases_allowed_speed_as_dlat_grows_regression():
    """本修正の中核: ds(縦距離)が同一でも、dlat(横間隔)が大きいほど_g2_speedへ渡る
    実効距離が伸び、許容速度が単調に増加する。"""
    m = make_self()
    ds = 2.0
    v_low, _ = m._follow_speed_limit(
        make_scan([(ds, 0.1, 3.0, 0.1, "d3", 100)]), path_offset=0.0, near_sep=1.8)
    v_mid, _ = m._follow_speed_limit(
        make_scan([(ds, 0.1, 3.0, 1.0, "d3", 100)]), path_offset=0.0, near_sep=1.8)
    v_high, _ = m._follow_speed_limit(
        make_scan([(ds, 0.1, 3.0, 1.5, "d3", 100)]), path_offset=0.0, near_sep=1.8)
    assert v_low < v_mid < v_high


def test_ds_eff_matches_legacy_ds_only_when_dlat_is_zero_regression():
    """回帰: dlat=0(真後ろ、STOPPING時のicc_stop等を想定)の場合、修正前と全く同じ
    _g2_speed(v_fwd, ds)の結果になる(既存の呼び出しパターンへの影響が無いことを確認)。"""
    m = make_self()
    ds, v_long = 3.0, 2.5
    vlim, _ = m._follow_speed_limit(
        make_scan([(ds, 0.0, v_long, 0.0, "d3", 100)]), path_offset=0.0, near_sep=1.8)
    expected = m._g2_speed(v_long, ds)  # 旧実装と同一の呼び出し(dlat加算なし)
    assert vlim == pytest.approx(expected)


def test_ds_eff_uses_the_dlat_of_the_selected_target_not_others():
    """回帰: 実効距離は選ばれた対象車(最近傍)自身のdlatのみを使う。無関係な
    2台目のdlatが混入しないことを確認する。"""
    m = make_self()
    scan = make_scan([
        (2.0, 0.1, 3.0, 0.1, "d3", 100),   # 最近傍(選ばれる)、dlat小
        (2.0, 5.0, 3.0, 5.0, "c2", 200),   # 同じds、遠方帯の外(選ばれない)、dlat大だが無関係
    ])
    vlim, vtgt = m._follow_speed_limit(scan, path_offset=0.0, near_sep=1.8)
    assert vtgt[4] == "d3"
    expected = m._g2_speed(3.0, 2.0 + 0.1)  # d3自身のdlat(0.1)のみが反映される
    assert vlim == pytest.approx(expected)


def test_retroactive_0716_01_lap1_corner3_ds_eff_relaxation():
    """遡及検証(0716-01実測、Lap1第3コーナーのインからの追い越し成功例):
    wp89(t=48.7秒台)の実測fwd_dlat=1.27m前後という値を用い、dsは同時間帯の
    典型的な実測レンジ(約2.0〜2.5m、ログのd_min欄より)を代表値として設定する
    (_follow_speed_limit内部のds単体は個別にログ出力されておらず、正確な値の
    再現ではなく代表値による定性確認である点を明記する)。旧実装(dlat非反映)相当と
    新実装(ds+dlat)を比較し、後者が明確に緩和されることを確認する。"""
    m = make_self()
    ds_typ, dlat_typ, vopp_typ = 2.2, 1.27, 3.0  # 実測(vopp≈3.05m/s, dlat≈1.27m)に基づく代表値
    v_legacy = m._g2_speed(vopp_typ, ds_typ)                 # 旧実装相当(dlat非反映)
    v_new, _ = m._follow_speed_limit(
        make_scan([(ds_typ, 0.1, vopp_typ, dlat_typ, "d3", 89)]),
        path_offset=0.0, near_sep=1.8)
    assert v_new > v_legacy
    # 5km/h(≈1.39m/s)近辺に張り付いていた実測に対し、修正後は明確な緩和幅が出ることを
    # 定量的に示す(この代表値では約+0.5m/s=約+1.8km/hの緩和)。
    assert (v_new - v_legacy) > 0.3
