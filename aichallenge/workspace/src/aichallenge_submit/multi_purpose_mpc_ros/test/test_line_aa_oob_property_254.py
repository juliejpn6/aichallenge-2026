"""line_aa(skimage)の境界外(OOB)出力の常設プロパティテスト
(256節続報、クローズ作業Phase 3)。

背景: コリドー計算(_compute_free_segments)は、waypointのstatic_border_cellsを
Map.w2m()で「地図座標(ピクセル)」へ変換した2点間にline_aa(skimage.draw)で
ラスタ線を引き、その線上のセルを占有格子map.data[y,x]でgatherする。もし
line_aaが範囲外(x<0やx>=width等)のピクセル座標を返すことがあれば、この
gatherがIndexErrorでノードごと落ちる。

Phase 2実験(2026-07-31、5種のマップサイズ・10,000件超のランダム境界ケース)で
実際に検証した結果、Map.w2m()が既にnp.clip()で出力座標をクランプしており、
その結果line_aaへ渡される2端点は常に[0,width-1]x[0,height-1]の範囲内であるため、
line_aa(skimage 0.25.2)が範囲外座標を返すことは一度も無かった。この事実は
Phase 2完了条件として「境界安全修正は不要」という結論の根拠になっている
(design_docs 256節参照)。

本ファイルは、その一度きりの実験を常設の回帰テストへ格上げしたものである。
skimageバージョン更新時にこのテストが自動で再検証する。もし将来失敗した
場合は、reference_path.py `_compute_free_segments`の`_fs_line_cache`構築時
クリップ(256節続報Phase 4-1、既に恒等操作として追加済み)がその場しのぎの
対症療法にしかならないことに注意し、`has_collision_in_line`(reference_path.py)
など他のline_aa呼び出し箇所すべてに同様の実行時クリップを追加することを
検討すること。

CI実行時間を考慮し、常設テストは1,500件に抑える(シード固定、決定的)。
Phase 2で実施した全件版(10,000件超・5マップサイズ)は
scripts/相当のスクリプトとして別途保管可能だが、本テストと同じロジックを
シードだけ変えて多く回せば再現できるため、専用スクリプトは必須ではない。
"""
import random

import numpy as np
from skimage.draw import line_aa

from corridor_test_helpers import make_synthetic_map, blank_grid

_SEED = 20260731
_N_CASES = 1500

# Phase 2実験で使った5種のマップサイズ(小さいもの・非正方形・大きいものを含む)。
_MAP_SIZES = [
    (10, 10),
    (50, 30),
    (120, 200),
    (170, 260),
    (400, 400),
]


def _random_endpoint_world_coords(rng, width, height, resolution, origin):
    """マップ範囲を大きく外れる座標も含めて生成する(w2m()のクランプを
    実際に発火させ、境界に張り付いた入力でline_aaを検証するため)。"""
    margin = max(width, height) * resolution
    x = rng.uniform(origin[0] - margin, origin[0] + width * resolution + margin)
    y = rng.uniform(origin[1] - margin, origin[1] + height * resolution + margin)
    return x, y


def test_line_aa_never_returns_out_of_bounds_pixels_for_w2m_clamped_endpoints():
    rng = random.Random(_SEED)
    n_violations = 0
    n_checked = 0

    for (height, width) in _MAP_SIZES:
        resolution = 0.1
        origin = (-1.0, -float(height) * resolution / 2.0)
        grid = blank_grid(height, width)
        m = make_synthetic_map(grid, resolution=resolution, origin=origin)

        for _ in range(_N_CASES // len(_MAP_SIZES)):
            x0, y0 = _random_endpoint_world_coords(rng, width, height, resolution, origin)
            x1, y1 = _random_endpoint_world_coords(rng, width, height, resolution, origin)
            p0 = m.w2m(x0, y0)
            p1 = m.w2m(x1, y1)

            # w2m()自身が[0,width-1]x[0,height-1]にクランプしていることの前提確認
            # (この前提が崩れたら、そもそもline_aaへ渡る入力自体が既に境界外)。
            assert 0 <= p0[0] < width and 0 <= p0[1] < height
            assert 0 <= p1[0] < width and 0 <= p1[1] < height

            x_list, y_list, _ = line_aa(p0[0], p0[1], p1[0], p1[1])
            x_arr = np.asarray(x_list)
            y_arr = np.asarray(y_list)
            n_checked += 1

            out_of_bounds = (
                np.any(x_arr < 0) or np.any(x_arr >= width)
                or np.any(y_arr < 0) or np.any(y_arr >= height))
            if out_of_bounds:
                n_violations += 1

    assert n_checked >= 1000
    assert n_violations == 0, (
        f"line_aaが範囲外座標を返すケースが{n_violations}/{n_checked}件見つかった。"
        "skimageのバージョン更新等でラスタライズアルゴリズムが変わった可能性がある。"
        "reference_path.pyの_fs_line_cache構築時クリップ(既存、現状は恒等操作)に加え、"
        "has_collision_in_line等の他のline_aa呼び出し箇所にも実行時クリップの追加を検討すること。")


def test_line_aa_oob_property_covers_multiple_map_sizes():
    """設計上5種類のマップサイズ(小・非正方形・大)を横断していることを固定する
    (特定のサイズだけで偶然OOBが起きない、という偏りを防ぐ)。"""
    assert len(_MAP_SIZES) == 5
    assert len(set(_MAP_SIZES)) == 5
