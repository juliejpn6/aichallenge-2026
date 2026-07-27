"""Regression tests for _compute_free_segments's narrow-gap fallback (101節続報, 2026-07-18).

Background: 0718-01実測で、giveup直後に壁際へ寄りながら直前まで並走していた相手を
見送る局面(wall=0.38〜0.39m)で、reference_path.pyの`_compute_free_segments`が
先読み上の複数waypointで自由区間を1本も返せず、"No feasible free segment found!"
という生printが131回連続発生した(約7〜8秒間、nseg0=3が持続)。

根本原因: `_compute_free_segments`はラスタ上の連続した空きセル区間のうち、幅が
`min_width`(=カート自身の全幅)以上のものだけを候補として返す。壁+相手車に挟まれ
生の空き幅がカート幅よりわずかに狭い場合、候補は0本になり、呼び出し元
(update_path_constraints)は境界をwaypoint自身の座標(幅ゼロの一点)へ一気に潰す
という、粗すぎるフォールバックに入っていた(今回は運良く実害が出なかったが、
幾何が少し違えば本物のQP infeasibility/STUCKに発展しうる)。

対処(ユーザー承認済み設計): min_width以上の区間が1本も無い場合でも、ラスタ上に
何らかの区間(min_width未満)が実在すれば、その中で最も幅の広いものを返すよう
`_compute_free_segments`自体を変更した。安全マージン込みの最終判定は既存の
add_constraint側(segment_length_sm<min_segment_lengthなら既存のwp.ub/wp.lb
静的境界へフォールバックする仕組み)にそのまま委ねる(新規のフォールバック機構は
追加しない)。ラスタ上に区間が1本も無い(完全に塞がれている)場合は、従来通り
呼び出し元の幅ゼロフォールバックへ委ねる(変更なし)。

テスト方針: `_compute_free_segments`は`self.map`(w2m/m2w/data)と`wp.static_border_cells`
にしか依存しないため、重い`ReferencePath`/`Map`の完全なコンストラクタ(実画像ファイル
読み込み等)を経由せず、resolution=1.0・origin=(0,0)の1行だけの軽量な偽Mapを
`Map.__new__(Map)`で組み立て、実物の`w2m`/`m2w`(core/map.py)をそのままバインドして
使う。これにより、モック化せず実際のアルゴリズムコードを直接検証する。
"""
import os

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.map import Map
from multi_purpose_mpc_ros.core.reference_path import ReferencePath


class _FakeWp:
    def __init__(self, static_border_cells):
        self.static_border_cells = static_border_cells


class _FakeSelf:
    """_compute_free_segmentsはself.mapしか参照しないため、これで十分。"""
    def __init__(self, map_obj):
        self.map = map_obj


def _make_fake_map(data_row):
    """resolution=1.0, origin=(0,0)の1行のみの軽量マップ(実物のw2m/m2wを使う)。"""
    fake_map = Map.__new__(Map)
    fake_map.origin = (0.0, 0.0)
    fake_map.resolution = 1.0
    fake_map.height = 1
    fake_map.width = len(data_row)
    fake_map.data = np.array([data_row], dtype=np.uint8)
    return fake_map


def _segments(data_row, min_width):
    fake_map = _make_fake_map(data_row)
    n = len(data_row)
    wp = _FakeWp(static_border_cells=[(0.0, 0.0), (float(n - 1), 0.0)])
    inst = _FakeSelf(fake_map)
    return ReferencePath._compute_free_segments(inst, wp, min_width)


def test_wide_segment_unchanged_when_above_min_width():
    """回帰: min_width以上の区間が実在する既存ケースは、従来通りそのまま1本返る。"""
    segs = _segments([1] * 11, min_width=2.0)
    assert len(segs) == 1


def test_narrow_gap_returns_the_gap_instead_of_empty():
    """核心: min_width未満の区間しか無い場合でも、空リストではなく実在する
    区間(位置情報を持つ)が1本返る(旧: 空リスト→呼び出し元でwaypoint自身の
    座標=幅ゼロの一点へフォールバックしていた)。"""
    # このパターンの唯一の区間は幅3.0(占有セルの境界を含む算出方式のため、
    # 空きセル自体は位置4-5の2つだが、境界を含めた実測幅は3.0になる)。
    # min_width=3.5とすることで、既存フィルタ(3.0<3.5)を通さずフォール
    # バック経路を確実に通す。
    segs = _segments([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], min_width=3.5)
    assert len(segs) == 1
    ub, lb = segs[0]
    # 返る区間はwaypoint自身の座標(0.0, 0.0)ではなく、実際の空き位置を反映する。
    assert ub != (0.0, 0.0)
    assert lb != (0.0, 0.0)


def test_widest_narrow_segment_is_selected_not_the_first_one():
    """回帰: min_width未満の区間が複数ある場合、走査順で最初に見つかったものでは
    なく、最も幅の広いものが選ばれる(順序非依存であることの確認)。"""
    # 位置1-2(境界込み実測幅3.0)と位置6-9(境界込み実測幅5.0)の2つの狭い区間。
    # min_width=6.0とすることでどちらも候補失格になるが、幅5.0の方が広い。
    segs = _segments([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], min_width=6.0)
    assert len(segs) == 1
    ub, lb = segs[0]
    width = ((ub[0] - lb[0]) ** 2 + (ub[1] - lb[1]) ** 2) ** 0.5
    assert width == pytest.approx(5.0, abs=1e-6)


def test_fully_occupied_line_still_returns_empty_regression():
    """回帰防止: ラスタ上に空きセルが1つも無い(完全に塞がれている)場合は、
    従来通り空リストのままとする(呼び出し元の幅ゼロフォールバックはこの
    ケース専用として維持する)。"""
    segs = _segments([0] * 11, min_width=2.0)
    assert segs == []


def test_retroactive_0718_01_narrow_squeeze_now_yields_nonzero_width():
    """遡及検証(0718-01実測、101節): giveup直後・壁際・相手見送り中に相当する、
    生の空きがカート全幅よりわずかに狭い状況を模擬。旧実装なら空リスト→
    呼び出し元でNo feasible free segment found!のprint+幅ゼロの一点フォール
    バックだったが、新実装では実在する(幅は狭いが非ゼロの)区間が返る。"""
    # このパターンの唯一の区間は境界込み実測幅4.0。min_width=4.5とすることで
    # 既存フィルタ(4.0<4.5)を通さずフォールバック経路を確実に通す。
    segs = _segments([0, 0, 0, 1, 1, 1, 0, 0, 0], min_width=4.5)
    assert len(segs) == 1  # 旧実装ならlen==0(呼び出し元がNo feasible...を出力していた)
    ub, lb = segs[0]
    width = ((ub[0] - lb[0]) ** 2 + (ub[1] - lb[1]) ** 2) ** 0.5
    assert 0.0 < width < 4.5


# ---------------------------------------------------------------------------
# 呼び出し元(update_path_constraints)側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "reference_path.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_all_segments_tracked_regardless_of_width():
    idx = _SRC.index("def _compute_free_segments")
    snippet = _SRC[idx:idx + 900]
    assert "all_segments = []" in snippet


def test_widest_fallback_only_applies_when_free_segments_empty_and_all_segments_nonempty():
    idx = _SRC.index("def _compute_free_segments")
    snippet = _SRC[idx:idx + 4500]
    assert "if not free_segments and all_segments:" in snippet
    assert "max(" in snippet


def test_zero_width_fallback_in_caller_is_unchanged_regression():
    """回帰防止: 呼び出し元のNo feasible free segment found!+幅ゼロフォール
    バック自体はコードとして残っている(all_segmentsも空の完全に塞がれた
    ケース専用として維持する、削除していない)ことを確認する。"""
    assert 'print(f"No feasible free segment found! wp_id: {wp_id}, n: {n}")' in _SRC
    assert "ub_ls, lb_ls = (wp.x, wp.y), (wp.x, wp.y)" in _SRC
