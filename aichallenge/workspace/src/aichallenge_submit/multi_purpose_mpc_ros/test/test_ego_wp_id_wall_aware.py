"""Unit tests for the ego-vehicle wp_id wall-aware matching fix (2026-07-14,
同型バグ水平展開: 他車向け _closest_wp_and_s 修正と全く同じ脆弱性が自車自身の
wp_id管理(spatial_bicycle_models.BicycleModel.get_closest_waypoint/update_states)
にも存在すると判明).

spatial_bicycle_models.py はrclpy非依存のため、実モジュールを直接importし、実クラス
BicycleModelに対して実メソッドを呼び出す(モックやAST抽出ではない)。

Bug: get_closest_waypoint(x, y) は従来、全waypointからの単純な(x,y)最近傍探索の
みで、弧長連続性を考慮していなかった。ヘアピン等、コースが壁一枚を挟んで自分自身に
近接する箇所では、自車の生座標が壁の反対側のwaypointへ誤ってマッチし得る
(mpc_controller.py 2708-2710行のコメントは「MPC空間モデルが逐次的にwp_idを管理して
おり壁越え耐性が既にある」という誤った前提を含んでいた)。

Fix: get_closest_waypoint/update_statesにprev_idx/radius_mを追加し、与えられた
場合のみ窓探索(±radius_idx、既存position_jump_threshold流用)に限定する。
prev_idx=None(初回起動時)は従来通り全waypointから探索する。
"""
import types

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.reference_path import Waypoint


class _RefPathStub:
    """ReferencePathの必要最小限(waypoints/resolution/segment_lengths)のみを持つ
    軽量スタブ。実ReferencePathはmap/スムージング等の重い前処理を要するため使わない。"""

    def __init__(self, xy, resolution=1.0):
        self.waypoints = [Waypoint(x, y, 0.0, 0.0) for x, y in xy]
        self.resolution = resolution
        self.segment_lengths = np.full(len(xy), resolution, dtype=float)


def _hairpin_track():
    """leg A(idx 0-9): (i, 0)を直進。leg B(idx 10-19): (9-(i-10), 0.3)で折り返し、
    leg Aとわずか0.3m(壁一枚分)離れて並走する「ヘアピン」を模擬する。"""
    leg_a = [(float(i), 0.0) for i in range(10)]
    leg_b = [(float(9 - (i - 10)), 0.3) for i in range(10, 20)]
    return leg_a + leg_b


def make_car(xy, resolution=1.0, wp_id0=0):
    ref_path = _RefPathStub(xy, resolution=resolution)
    car = BicycleModel(reference_path=ref_path, length=1.0, width=0.5, Ts=0.1)
    car.wp_id = wp_id0
    car.s = car.get_s_at_waypoint(wp_id0)
    return car


def test_no_prev_idx_uses_full_global_search_regression():
    """回帰: prev_idx省略時(初回起動)は従来通り全waypointから探索する。"""
    car = make_car(_hairpin_track())
    idx = car.get_closest_waypoint(4.0, 0.0)
    assert idx == 4


def test_windowed_search_avoids_crossing_to_the_other_leg_of_a_hairpin():
    """本質: 壁越え誤認識バグの再現+修正確認。自車は実際にはleg B(戻り側)を走行中で
    直前はwp15にマッチしていた。今回の生座標(4.0, 0.05)はノイズにより、leg B側の
    どのwaypointよりもleg A側のwp4に幾何学的に近い値になってしまっている
    (壁一枚分=0.3mの間隔よりノイズが大きいケース)。"""
    car = make_car(_hairpin_track())
    query = (4.0, 0.05)

    # 対比: prev_idxなし(旧来のグローバル探索)だと壁の向こう(leg A)のwp4へ誤マッチ
    idx_global = car.get_closest_waypoint(*query)
    assert idx_global == 4  # バグの再現(壁の向こう側へ誤ってジャンプする)

    idx_windowed = car.get_closest_waypoint(*query, prev_idx=15, radius_m=3.0)
    assert idx_windowed != 4          # leg A(壁の向こう側)へは飛ばない
    assert 12 <= idx_windowed <= 18   # prev_idx=15近傍(leg B側)に留まる


def test_update_states_wires_prev_idx_and_radius_end_to_end():
    """update_states経由でも同じ窓探索が効くことを確認する(実際の呼び出し経路)。"""
    car = make_car(_hairpin_track(), wp_id0=15)
    car.update_states(4.0, 0.05, 0.0, prev_idx=15, radius_m=3.0)
    assert car.wp_id != 4
    assert 12 <= car.wp_id <= 18


def test_update_states_without_prev_idx_regression():
    """回帰: prev_idx/radius_mを渡さない既存の呼び出し(初回起動時のrun())は
    シグネチャ変更後も従来通り動作する(デフォルト引数で後方互換)。"""
    car = make_car(_hairpin_track())
    car.update_states(9.0, 0.0, 0.0)
    assert car.wp_id == 9


def test_windowed_search_still_tracks_genuine_forward_motion_regression():
    """回帰: 対象車が本当に(壁越えではなく)通常通り前進した場合は、探索窓内で
    正しく追従できる(過剰に狭い窓で本来の動きまで妨げない)。"""
    car = make_car(_hairpin_track(), wp_id0=15)
    car.update_states(3.0, 0.3, 0.0, prev_idx=15, radius_m=3.0)  # wp16の座標ちょうど
    assert car.wp_id == 16


def test_search_radius_derived_from_resolution_and_radius_m():
    """境界値: radius_m(既存_wp_match_radius_m相当)とresolutionから導出される
    探索半径(waypoint数換算)が正しく機能し、窓の外は捕捉されないことを確認する。"""
    track = [(float(i), 0.0) for i in range(30)]
    car = make_car(track, resolution=1.0)
    # resolution=1.0m/wp, radius_m=2.0 -> radius_idx=2、探索範囲はidx 8..12
    idx = car.get_closest_waypoint(13.0, 0.0, prev_idx=10, radius_m=2.0)
    assert idx == 12  # 窓の外(idx13)は選べず、窓内の最近傍(idx12)に留まる
