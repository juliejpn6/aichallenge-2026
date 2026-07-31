"""コリドー等価性ファジングの合成グリッド生成ロジック(254節続報続)。

各ケースは`case_idx`から導出した専用の`random.Random`インスタンスで
決定的に再現できるため、グリッド自体をコーパスファイルへ保存する必要が
ない(パラメータのみ保存すれば、いつでも同一のグリッドを再構築できる)。
`gen_fuzz_corpus.py`(オラクル出力の収集)とpytestテスト(新実装との比較)
の両方から共有される。
"""
import random

from corridor_test_helpers import blank_grid, make_straight_waypoints, add_vertical_wall, add_full_width_barrier

RES = 0.1
ORIGIN = (-1.0, -8.0)
H, W = 170, 260


def col_of(x):
    return int(round((x - ORIGIN[0]) / RES))


def make_grid_for_case(case_idx, seed=20260731):
    """case_idxから決定的にグリッド・waypoint列・パラメータを再構築する
    (グリッド自体はコーパスファイルへ保存せず、この関数で毎回再生成する)。"""
    rng = random.Random(seed * 1000003 + case_idx)
    n_wp = rng.randint(6, 9)
    dx = rng.choice([0.3, 0.5, 0.7])
    half_width = rng.choice([3.0, 5.0, 6.0, 8.0])
    wall_prob = rng.choice([0.3, 0.5, 0.7, 0.9])
    N = rng.randint(4, min(6, n_wp))
    model_width = rng.choice([0.3, 0.5])
    safety_margin = rng.choice([0.01, 0.05, 0.1])

    grid = blank_grid(H, W)
    wps = make_straight_waypoints(n_wp + 2, x0=0.0, y0=0.0, dx=dx, half_width=half_width)
    for i in range(n_wp + 2):
        if rng.random() > wall_prob:
            continue
        c = col_of(i * dx)
        n_walls = rng.randint(1, 3)
        for _ in range(n_walls):
            row_start = rng.randint(5, H - 10)
            thickness = rng.randint(1, 3)
            add_vertical_wall(grid, c, row_start=row_start, row_end=min(H, row_start + thickness))
    if rng.random() < 0.15:
        bi = rng.randint(1, n_wp)
        bc = col_of(bi * dx) + rng.randint(1, 3)
        thickness = rng.randint(2, 4)
        add_full_width_barrier(grid, bc, min(W, bc + thickness))

    params = {
        "n_wp": n_wp, "dx": dx, "half_width": half_width, "wall_prob": wall_prob,
        "N": N, "model_length": 0.3, "model_width": model_width,
        "safety_margin": safety_margin, "pose": [0.0, 0.0, 0.0],
    }
    return grid, wps, params
