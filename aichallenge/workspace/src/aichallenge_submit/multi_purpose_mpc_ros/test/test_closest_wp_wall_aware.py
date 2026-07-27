"""Unit tests for the wall-aware waypoint matching fix (2026-07-14, ユーザー指摘:
「壁の向こう側にいる相手」誤認識対策).

mpc_controller.py imports rclpy/autoware message types at module scope, so
_closest_wp_and_s is extracted via AST from the real source file and bound to a
minimal mock `self`, exercising the ACTUAL production code (not a mirror).

Bug: _closest_wp_and_s used a pure global nearest-(x,y)-neighbor search over ALL
waypoints, with no awareness of arc-length continuity. At a hairpin/tight corner
where two arc-length-distant legs of the track run close together separated only
by a wall, a vehicle genuinely on one leg can have its raw (x, y) sample read as
closer to a waypoint on the OTHER leg (due to sensor noise or the legs being
closer together than a single waypoint's own spacing), producing a wildly wrong
arc-length position (s_obs) and hence a spuriously small ds/dlat for a car that
poses zero real collision risk.

Fix: when a `prev_idx` hint (the vehicle's last actual match) is supplied, restrict
the search to a window around it (± wp_match_radius_m, reusing the existing V2X
`position_jump_threshold` as the physically-plausible per-update movement bound).
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


_NS = {"np": np}
exec(compile(_extract_method("_closest_wp_and_s"), "<_closest_wp_and_s>", "exec"), _NS)


class _RefPath:
    def __init__(self, resolution=1.0):
        self.resolution = resolution


def make_self(waypoint_xy, resolution=1.0, wp_match_radius_m=5.0):
    m = types.SimpleNamespace()
    m._waypoint_xy = np.array(waypoint_xy, dtype=np.float64)
    m._wp_s_cum = np.arange(len(waypoint_xy), dtype=float) * resolution
    m._reference_path = _RefPath(resolution)
    m._wp_match_radius_m = wp_match_radius_m
    m._closest_wp_and_s = types.MethodType(_NS["_closest_wp_and_s"], m)
    return m


def _hairpin_track():
    """leg A(idx 0-9): (i, 0)を直進。leg B(idx 10-19): (9-(i-10), 0.3)で
    折り返し、leg Aとわずか0.3m(壁一枚分)離れて並走する「ヘアピン」を模擬する。"""
    leg_a = [(float(i), 0.0) for i in range(10)]
    leg_b = [(float(9 - (i - 10)), 0.3) for i in range(10, 20)]
    return leg_a + leg_b


def test_no_prev_idx_uses_full_global_search_regression():
    """回帰: prev_idx省略時(初回)は従来通り全waypointから探索する。"""
    track = _hairpin_track()
    m = make_self(track)
    idx, s = m._closest_wp_and_s(4.0, 0.0)  # leg AのwP4ちょうど
    assert idx == 4


def test_windowed_search_avoids_crossing_to_the_other_leg_of_a_hairpin():
    """0714-05実測相当の再現: 相手車は実際にはleg B(戻り側)を走行中で、直前は
    wp15(leg B)にマッチしていた。今回の生座標(4.0, 0.05)は雑音により、
    本来continuousなleg B側waypointよりもleg A側のwp4(4.0, 0.0)に近い値になって
    しまっている(壁一枚分=0.3mの間隔よりノイズが大きいケース)。
    prev_idx=15を渡すと、探索窓がleg B近傍に限定され、壁の向こう側(leg A)の
    wp4へ誤ってジャンプしない。"""
    track = _hairpin_track()
    m = make_self(track, wp_match_radius_m=3.0)  # 窓半径3m(=3waypoint相当)
    query = (4.0, 0.05)
    # 対比: prev_idxなし(グローバル探索)だと壁の向こう(leg A)のwp4へ誤マッチする
    idx_global, _ = m._closest_wp_and_s(*query)
    assert idx_global == 4  # leg A側へ誤ってジャンプすることを確認(バグの再現)

    idx_windowed, _ = m._closest_wp_and_s(*query, prev_idx=15)
    assert idx_windowed != 4          # leg A(壁の向こう側)へは飛ばない
    assert 12 <= idx_windowed <= 18   # prev_idx=15近傍(leg B側)に留まる


def test_windowed_search_still_tracks_genuine_forward_motion_regression():
    """回帰: 対象車が本当に(壁越えではなく)通常通り前進した場合は、探索窓内で
    正しく追従できる(過剰に狭い窓で本来の动きまで妨げない)。"""
    track = _hairpin_track()
    m = make_self(track, wp_match_radius_m=3.0)
    # leg B上をwp15→wp16相当(前進)へ実際に動いた場合
    idx, _ = m._closest_wp_and_s(3.0, 0.3, prev_idx=15)  # wp16の座標ちょうど
    assert idx == 16


def test_search_radius_derived_from_resolution_and_position_jump_threshold():
    """境界値: wp_match_radius_m(既存position_jump_threshold流用)とresolutionから
    導出される探索半径(waypoint数換算)が正しく機能し、半径ちょうど外側の点は
    捕捉されない一方、半径内は捕捉されることを確認する。"""
    # resolution=1.0m/wp, wp_match_radius_m=2.0 → radius_idx=2
    track = [(float(i), 0.0) for i in range(30)]
    m = make_self(track, resolution=1.0, wp_match_radius_m=2.0)
    # prev_idx=10、半径2 → 探索範囲は idx 8..12。idx13の位置(13,0)は範囲外のはずだが
    # 窓内(8..12)で最も近いidx12を選ぶ(idx13そのものは候補にすら入らない)。
    idx, _ = m._closest_wp_and_s(13.0, 0.0, prev_idx=10)
    assert idx == 12  # 窓の外(idx13)は選べず、窓内の最近傍(idx12)に留まる


def test_no_position_change_regression():
    """回帰: 対象車が全く動いていない場合、windowed探索でも同じidxを維持する。"""
    track = _hairpin_track()
    m = make_self(track, wp_match_radius_m=3.0)
    idx, _ = m._closest_wp_and_s(4.0, 0.3, prev_idx=15)  # wp15ちょうどの座標
    assert idx == 15
