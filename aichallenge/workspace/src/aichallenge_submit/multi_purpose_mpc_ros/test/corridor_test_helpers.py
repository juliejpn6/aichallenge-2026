"""コリドー計算(update_path_constraints)等価最適化のためのテスト基盤(254節続報続)。

`core/reference_path.py`・`core/map.py`はrclpy非依存のため直接importできる。
本モジュールは、YAML/画像ファイルを介さずに合成占有格子(Map)・合成waypoint列
(ReferencePath)を直接構築するためのヘルパーを提供する(`__new__`で`__init__`を
バイパスし、update_path_constraints/_compute_free_segmentsが実際に参照する属性
のみを手動設定する)。

規約: map.data は 1=free(走行可能)/0=occupied(障害物)(core/map.py:131参照)。
waypoint配置は全て直線(psi=0、車両はx軸正方向を向く、y軸が左右)とし、
static_border_cellsは十分広く取ることで、各テストケースが用意した占有格子内の
分断パターンをそのままfree segment検出させる。
"""
import numpy as np

from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath, Waypoint


def make_synthetic_map(data, resolution=0.1, origin=(0.0, 0.0)):
    """dataは2D numpy配列(shape=(height,width)、1=free/0=occupied、dtype問わず)。"""
    m = Map.__new__(Map)
    m.data = np.asarray(data, dtype=np.int8)
    m.height, m.width = m.data.shape
    m.resolution = float(resolution)
    m.origin = origin
    m.threshold_occupied = 0.5
    m.obstacles = []
    m.boundaries = []
    m.data_backup = m.data.copy()
    return m


def make_waypoint(x, y, psi=0.0, kappa=0.0, half_width=5.0):
    """半幅half_width[m]でstatic_border_cellsを設定した直線区間用waypoint。
    psi=0の直線を仮定: 左(ub側)は+y方向、右(lb側)は-y方向。"""
    wp = Waypoint(x, y, psi, kappa)
    left = (x - half_width * np.sin(psi), y + half_width * np.cos(psi))
    right = (x + half_width * np.sin(psi), y - half_width * np.cos(psi))
    wp.static_border_cells = (left, right)
    # ub/lb: add_constraint内のnarrow-segmentフォールバックでのみ参照される
    # (本テストケースでは分断幅は常にmin_segment_length以上を狙うため通常は
    # 使われない想定だが、フォールバック時に例外を起こさぬよう有限値を設定する)。
    wp.ub = half_width
    wp.lb = -half_width
    wp.ub_sm = half_width
    wp.lb_sm = -half_width
    wp.dynamic_border_cells = wp.static_border_cells
    return wp


def make_straight_waypoints(n, x0=0.0, y0=0.0, dx=0.5, half_width=5.0):
    """x軸方向へdx間隔で並ぶ直線waypoint列(psi=0固定)をn個生成する。"""
    return [make_waypoint(x0 + i * dx, y0, psi=0.0, half_width=half_width)
            for i in range(n)]


def make_synthetic_reference_path(map_obj, waypoints, circular=False,
                                   corridor_widen_step_m=float('inf')):
    rp = ReferencePath.__new__(ReferencePath)
    rp.map = map_obj
    rp.waypoints = waypoints
    rp.n_waypoints = len(waypoints)
    rp.circular = circular
    rp.corridor_widen_step_m = corridor_widen_step_m
    rp.rect_points = []
    rp.upper_cols = []
    rp.lower_cols = []
    rp.free_segs = []
    rp.select_free_segs = []
    rp.dbg_nseg0 = 0
    rp.dbg_nseg1 = 0
    rp.dbg_nseg2 = 0
    rp.dbg_ncomb_max = 0
    rp.COUNT = 0
    return rp


def blank_grid(height, width):
    """全面free(1)のグリッド。"""
    return np.ones((height, width), dtype=np.int8)


def add_vertical_wall(grid, col, row_start=None, row_end=None):
    """指定列[col]の[row_start,row_end)を occupied(0) にする(縦方向の壁)。"""
    h, w = grid.shape
    if row_start is None:
        row_start = 0
    if row_end is None:
        row_end = h
    grid[row_start:row_end, col] = 0
    return grid


def add_horizontal_wall(grid, row, col_start=None, col_end=None):
    """指定行[row]の[col_start,col_end)を occupied(0) にする(1行分のみ)。"""
    h, w = grid.shape
    if col_start is None:
        col_start = 0
    if col_end is None:
        col_end = w
    grid[row, col_start:col_end] = 0
    return grid


def add_full_width_barrier(grid, col_start, col_end):
    """指定列範囲[col_start,col_end)の全行(=コース全幅)をoccupied(0)にする。
    waypointはx軸方向(=列方向)に並ぶため、この列範囲を横切る衝突判定線分
    (has_collision_in_line)は必ずこの壁に当たる=「コース全幅を横断する遮断壁」。"""
    grid[:, col_start:col_end] = 0
    return grid
