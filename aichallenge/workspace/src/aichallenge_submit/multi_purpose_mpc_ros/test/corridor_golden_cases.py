"""コリドー等価最適化(254節続報続)の手作りゴールデンケース9種の構築ロジック。

各関数は(grid, wps, params)を返す。paramsはupdate_path_constraints呼び出しに
必要な残りの引数(wp_id, N, pose, model_length, model_width, safety_margin)。
期待値(現行実装で実際に計測し確定した値)はcorridor_golden_expected.jsonに
保存されている。

設計意図(各ケースが検証する性質):
  1. 障害物なし: 最も基本的な健全性(全waypointが最大幅)
  2. 1waypointのみ2分断: 孤立した2-segment判定の基本動作
  3. 連続する複数waypointで2~4分断: 本格的な組み合わせ探索(itertools.product)を起動
  4. free_segmentsゼロでall_segmentsフォールバック起動: _compute_free_segments内の
     「min_width未満でも実在する最幅の区間を返す」経路(101節続報)
  5. マップ端付近のwaypoint: static_border_cellsがw2mでクランプされる境界条件
  6. 完全封鎖: 全組み合わせが衝突(-1000000.0)でタイになり、itertools.productの
     辞書式順序で最初の(0,0,0,0,0)が選ばれる(Python max()の同値先勝ち)
  7. 始点封鎖: 6とは異なる理由(探索窓の最初の層への遷移自体が全滅)で同じ
     タイブレークに帰着することを確認する
  8. 一本道生存: 6のバリアに1箇所だけ開口を作り、実際に非自明な経路
     (wp2,wp3でlower segmentを選択)が選ばれることを確認する(6/7/9と異なる結果)
  9. 同点タイ(厳密に等しい幅): 上下2つのsegmentの幅がビット単位で等しい場合、
     若いインデックス(scan順で最初=upper側)が選ばれることを確認する
"""
import os
import random

from corridor_test_helpers import (
    blank_grid, make_straight_waypoints, add_vertical_wall, add_full_width_barrier,
)

RES = 0.1
ORIGIN = (-1.0, -6.0)
H, W = 120, 200


def col_of(x):
    return int((x - ORIGIN[0]) / RES + 0.5)


def _base_default_params(wp_id=0, N=6, pose=(0.0, 0.0, 0.0), model_length=0.3,
                          model_width=0.3, safety_margin=0.1):
    return {"wp_id": wp_id, "N": N, "pose": list(pose), "model_length": model_length,
            "model_width": model_width, "safety_margin": safety_margin}


def case_1_no_obstacle():
    grid = blank_grid(H, W)
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    return grid, wps, _base_default_params()


def case_2_single_wp_2segs():
    grid = blank_grid(H, W)
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    add_vertical_wall(grid, col_of(1.0), row_start=59, row_end=61)
    return grid, wps, _base_default_params()


def case_3_multi_wp_split():
    grid = blank_grid(H, W)
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    for i, ncols in zip(range(1, 5), [2, 3, 4, 3]):
        col = col_of(i * 0.5)
        seg_h = H // (ncols + 1)
        for k in range(1, ncols + 1):
            add_vertical_wall(grid, col, row_start=seg_h * k, row_end=seg_h * k + 2)
    return grid, wps, _base_default_params()


def case_4_all_segments_fallback():
    grid = blank_grid(H, W)
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    c = col_of(0.5)
    add_vertical_wall(grid, c, row_start=0, row_end=59)
    add_vertical_wall(grid, c, row_start=61, row_end=H)
    return grid, wps, _base_default_params(model_width=0.3, safety_margin=0.01)


def case_5_map_edge_clamped():
    grid = blank_grid(H, W)
    # half_width(8.0)がマップの実効半幅(約6.0m)を超え、static_border_cellsが
    # w2mでクランプされる。
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=8.0)
    return grid, wps, _base_default_params()


def _base_grid_and_wps(split_wp_indices=(1, 2, 3, 4, 5), wall_row_start=59, wall_row_end=61,
                        n_wp=10):
    grid = blank_grid(H, W)
    wps = make_straight_waypoints(n_wp, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    for i in split_wp_indices:
        c = col_of(i * 0.5)
        add_vertical_wall(grid, c, row_start=wall_row_start, row_end=wall_row_end)
    return grid, wps


def case_6_complete_blockage():
    grid, wps = _base_grid_and_wps()
    barrier_col_start = col_of(1.0) + 1
    barrier_col_end = barrier_col_start + 3
    add_full_width_barrier(grid, barrier_col_start, barrier_col_end)
    return grid, wps, _base_default_params()


def case_7_start_blockage():
    grid, wps = _base_grid_and_wps()
    barrier_col_start = col_of(0.0) + 1
    barrier_col_end = barrier_col_start + 3
    add_full_width_barrier(grid, barrier_col_start, barrier_col_end)
    return grid, wps, _base_default_params()


def case_8_single_surviving_path():
    grid, wps = _base_grid_and_wps()
    barrier_col_start = col_of(1.0) + 1
    barrier_col_end = barrier_col_start + 3
    add_full_width_barrier(grid, barrier_col_start, barrier_col_end)
    grid[65:110, barrier_col_start:barrier_col_end] = 1  # 下側(lower)開口
    return grid, wps, _base_default_params()


def case_9_symmetric_tie():
    grid, wps = _base_grid_and_wps(wall_row_start=59, wall_row_end=60)
    return grid, wps, _base_default_params()


# 2026-07-31追加(256節続報、クローズ作業Phase 2): 差分ファジングのcase_idx=125
# (corridor_fuzz_gen.make_grid_for_case(125, seed=20260731))を、シード再生成に
# 依存しない独立ケースとして昇格させる。将来コーパスを再生成すると「実績のある
# 地雷原」が静かに消えるため(コーパス自体はcase_idx+期待値のみ保存しグリッドは
# 都度再構築する設計、corridor_fuzz_gen.pyのdocstring参照)、この特定の入力だけは
# wall配置を直接ハードコードして固定する。
#
# 経緯: Phase 3-2で当初実装した後ろ向きDP(suffix DP)は、このケースで実際に
# 等価性を破った。原因は浮動小数点加算の結合則崩れ: 旧実装(itertools.product+
# 前向き=左結合の逐次累積)とDP(後ろ向き=右結合の累積)は、数学的に同じ値でも
# 異なるビットパターンに丸まりうる。具体的には、wp3(horizon内n=3)の2つの
# セグメント幅が単独では4.199999999999999/4.2と区別されるが、wp2までの部分和
# (約21.2)に加算すると両方とも同一の浮動小数点値25.400000000000002へ「吸収」
# された。DPは局所比較(5.6+4.2 > 5.6+4.199999999999999)で厳密な大小として
# 扱ってしまい、旧実装の前向き累積では実質タイになる2つの組み合わせのうち
# 誤った方を選んでしまった(詳細はdesign_docs 256節参照)。
#
# このケースは「itertools.product+前向き左結合累積+max()+.index()のタイブレーク」
# という現行の安全設計が将来も維持され続ける限りPASSし続ける、いわば回帰の
# カナリアである。加算順序を変える最適化(DP化・math.fsum・numpy.sum等への
# 置換)を将来誰かが試みた場合、真っ先にここで失敗するはずである。
#
# 2026-08-01追加(258節続報、マージ後フォローアップPhase 2): このケースが
# 将来役目を終えた場合(コンパイラ最適化・ライブラリ更新等で偶然この特定の
# 入力では吸収現象が起きなくなり、test_case_10_actually_exhibits_floating_
# point_absorptionが失敗した場合)、後任ケースを探索せず決定的に構築できる
# レシピを以下に記す(検証はtest_fp_absorption_case_construction_recipe_
# is_reproducible参照、既に数値レベルで実演済み):
#   1. 接頭辞和S(先行レイヤーの累積、例: 21.2級)と幅wを選ぶ
#   2. w2 = w、w1 = math.nextafter(w, 0.0)(1ULP下の隣接値)とする
#   3. w1 != w2 かつ S + w1 == S + w2 を検証する。不成立ならSを大きくする
#      (Sが大きいほど加算の丸め粒度が粗くなるため、十分大きなSで必ず成立する)
#   4. 検証済み(S, w1, w2)から逆算し、該当レイヤーのセグメント幅がw1/w2、
#      先行レイヤーの累積がSになる境界セル配置を合成する(壁位置の調整で
#      所望の幅を作る手順は、本ファイルの既存ケース構築時の試行錯誤
#      ——case_4のsafety_margin調整、case_9の壁行位置探索など——と同じ要領)
# 補助手段(未実装、方針のみ): corridor_fuzz_gen.pyの生成器を再利用し、
# 隣接する2レイヤーの幅が異なる(w1!=w2)にも関わらず、任意の先行部分和と
# 前向き左結合で加算した結果がビット一致するペアを総当たりで走査する
# 「吸収スキャナ」を書けば、上記の手動レシピより多様なケースを機械的に
# 発見できる。今回は既存レシピで十分間に合っているため実装していない。
_CASE_10_RES = 0.1
_CASE_10_ORIGIN = (-1.0, -8.0)
_CASE_10_H, _CASE_10_W = 170, 260


def case_10_fp_absorption_tiebreak():
    grid = blank_grid(_CASE_10_H, _CASE_10_W)
    wps = make_straight_waypoints(10, x0=0.0, y0=0.0, dx=0.5, half_width=5.0)
    # (col, row_start, row_end) 各壁。列位置はcol_of(wp_index*dx)相当(dx=0.5)。
    walls = [
        (10, 77, 78), (10, 29, 31), (10, 70, 71),    # wp0
        (15, 44, 46), (15, 32, 33), (15, 148, 151),  # wp1
        (20, 146, 147), (20, 95, 97),                # wp2
        (25, 81, 82), (25, 123, 125), (25, 15, 18),  # wp3
        (35, 139, 141),                              # wp5
        (40, 100, 101), (40, 131, 132), (40, 67, 68),  # wp6
        (45, 76, 77),                                # wp7
        (55, 23, 24),                                # wp9
    ]
    for col, row_start, row_end in walls:
        add_vertical_wall(grid, col, row_start=row_start, row_end=row_end)
    add_full_width_barrier(grid, 33, 35)
    params = _base_default_params(
        wp_id=0, N=4, pose=(0.0, 0.0, 0.0),
        model_length=0.3, model_width=0.5, safety_margin=0.05)
    return grid, wps, params


# 一部のケースは既定の共有ジオメトリ(RES/ORIGIN、モジュール冒頭定義)ではなく
# 専用のジオメトリを必要とする(case_10は元のファジングケースを1ビットも
# 変えず再現するため、corridor_fuzz_gen.pyと同一のRES/ORIGIN/H/Wを使う)。
# 未登録のケースは共有ジオメトリ(RES, ORIGIN)を使う。
CASE_GEOMETRY = {
    "10_fp_absorption_tiebreak": (_CASE_10_RES, _CASE_10_ORIGIN),
}


def geometry_for(case_name):
    """テストハーネスから呼ぶ: そのケースが使うべき(resolution, origin)を返す。"""
    return CASE_GEOMETRY.get(case_name, (RES, ORIGIN))


ALL_CASES = {
    "1_no_obstacle": case_1_no_obstacle,
    "2_single_wp_2segs": case_2_single_wp_2segs,
    "3_multi_wp_split": case_3_multi_wp_split,
    "4_all_segments_fallback": case_4_all_segments_fallback,
    "5_map_edge_clamped": case_5_map_edge_clamped,
    "6_complete_blockage": case_6_complete_blockage,
    "7_start_blockage": case_7_start_blockage,
    "8_single_surviving_path": case_8_single_surviving_path,
    "9_symmetric_tie": case_9_symmetric_tie,
    "10_fp_absorption_tiebreak": case_10_fp_absorption_tiebreak,
}
