"""コリドー計算(update_path_constraints)等価最適化のリグレッションハーネス
(254節続報続、Phase 0)。

Phase 3(_compute_free_segmentsのベクトル化、itertools.productのDP化)を
実装する前に、現行(素朴なループ+itertools.product)実装の出力をロックする。
Phase 3実装後は本ファイルを一切変更せず、そのままPASSすることが「1ビットも
出力が変わっていない」ことの実行可能な証拠になる。

構成:
  1. ゴールデンケース9種(corridor_golden_cases.py) — 個別に設計意図を持つ
     手作りシナリオ。各ケースの意図はcorridor_golden_cases.pyのdocstring参照。
     期待値はcorridor_golden_expected.json(現行実装で実測し確定済み)。
  2. 差分ファジング1500ケース(corridor_fuzz_corpus.json) — case_idxから
     corridor_fuzz_gen.make_grid_for_case()で決定的に再構築したランダム合成
     グリッドに対し、現行実装の出力(オラクル)と一致するかを検証する。
     グリッド自体はコーパスに保存せず、シード固定の乱数から都度再構築する
     (コーパスファイルを191MBではなく169KBに収めるため)。

いずれも「現行実装が自分自身と一致する」健全性チェックとして機能する
(このテストの初回パスは自明)が、Phase 3実装後にこのテストがそのまま
PASSし続けることこそが等価性の実証になる。1件でも不一致が出た場合は
即座に停止し、最小再現(case_idxまたはケース名)を報告すること。
"""
import json
import os

import numpy as np
import pytest

import corridor_golden_cases as golden_cases
import corridor_fuzz_gen
from corridor_test_helpers import make_synthetic_map, make_synthetic_reference_path

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "corridor_golden_expected.json")) as _f:
    _GOLDEN_EXPECTED = json.load(_f)

with open(os.path.join(_HERE, "corridor_fuzz_corpus.json")) as _f:
    _FUZZ_CORPUS = json.load(_f)


def _run(grid, wps, params, resolution, origin, circular=False):
    m = make_synthetic_map(grid, resolution=resolution, origin=origin)
    rp = make_synthetic_reference_path(m, wps, circular=circular)
    ub, lb, border_cells = rp.update_path_constraints(
        params.get("wp_id", 0), params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    return np.round(ub, 6).tolist(), np.round(lb, 6).tolist()


@pytest.mark.parametrize("case_name", sorted(golden_cases.ALL_CASES.keys()))
def test_golden_case_matches_frozen_expected_output(case_name):
    grid, wps, params = golden_cases.ALL_CASES[case_name]()
    res, origin = golden_cases.geometry_for(case_name)
    ub, lb = _run(grid, wps, params, res, origin)
    expected = _GOLDEN_EXPECTED[case_name]
    assert ub == expected["ub"], (
        f"{case_name}: ub mismatch — got {ub}, frozen expected {expected['ub']}")
    assert lb == expected["lb"], (
        f"{case_name}: lb mismatch — got {lb}, frozen expected {expected['lb']}")


def test_golden_expected_file_covers_all_defined_cases():
    assert set(_GOLDEN_EXPECTED.keys()) == set(golden_cases.ALL_CASES.keys())


def test_case_10_actually_exhibits_floating_point_absorption():
    """2026-07-31追加(256節続報、クローズ作業Phase 2): 10_fp_absorption_tiebreak
    (旧差分ファジングcase_idx=125)が、名前の由来である「浮動小数点の吸収現象」を
    実際に踏んでいることを直接検証する。単に出力が凍結値と一致するだけでは、
    将来コードが変わって別の理由でたまたま同じ出力になった場合を検出できない
    ため、このケースの「地雷」そのもの——horizon内n=3の2つのセグメント幅が
    単独では区別されるが、n=0〜2の部分和に加算すると同一の浮動小数点値へ
    丸まる——が今も成立していることを明示的にassertする。

    このassertが失敗する状態(=もう吸収が起きない状態)になった場合、この
    ケースはもはや「DP化の罠」を検出できなくなっているため、別の吸収現象を
    示す新しいケースへ差し替える必要がある。"""
    from multi_purpose_mpc_ros.core.reference_path import dist

    grid, wps, params = golden_cases.ALL_CASES["10_fp_absorption_tiebreak"]()
    res, origin = golden_cases.geometry_for("10_fp_absorption_tiebreak")
    m = make_synthetic_map(grid, resolution=res, origin=origin)
    rp = make_synthetic_reference_path(m, wps, circular=False)

    free_segments_hor = []
    for n in range(params["N"]):
        wp = rp.get_waypoint(params["wp_id"] + n)
        free_segments_hor.append(
            rp._compute_free_segments_scalar(wp, params["model_width"]))

    def width(n, idx):
        ub_fs, lb_fs = free_segments_hor[n][idx]
        return dist(ub_fs[0], ub_fs[1], lb_fs[0], lb_fs[1])

    # n=3(最終レイヤー)の2つの候補セグメント幅は、単独では明確に区別される。
    w3_0 = width(3, 0)
    w3_1 = width(3, 1)
    assert w3_0 == 4.199999999999999
    assert w3_1 == 4.2
    assert w3_0 != w3_1

    # n=0(index2)・n=1(index0)・n=2(index0)を選んだ場合の部分和(旧実装と
    # 同じ前向き=左結合の逐次加算で再現する)。
    partial = 0.0
    partial += width(0, 2)
    partial += width(1, 0)
    partial += width(2, 0)

    # この部分和にw3_0/w3_1をそれぞれ加算すると、真に異なる値(w3_0!=w3_1)
    # にも関わらず、丸めにより同一の浮動小数点値へ「吸収」される。
    total_with_w3_0 = partial + w3_0
    total_with_w3_1 = partial + w3_1
    assert total_with_w3_0 == total_with_w3_1 == 25.400000000000002, (
        f"吸収現象が再現しない(partial={partial!r}, "
        f"partial+w3_0={total_with_w3_0!r}, partial+w3_1={total_with_w3_1!r})。"
        "このケースはもはやDP化の罠を検出できないため、別の吸収現象を示す"
        "新しいケースへ差し替える必要がある。構築レシピはcorridor_golden_cases.py"
        "のcase_10_fp_absorption_tiebreak直前のコメント(2026-08-01追加)を参照——"
        "探索は不要で、(S, w1=nextafter(w,0.0), w2=w)から決定的に構築できる"
        "(test_fp_absorption_case_construction_recipe_is_reproducibleで実演済み)。")


@pytest.mark.parametrize("case", _FUZZ_CORPUS["cases"], ids=lambda c: f"case{c['case_idx']}")
def test_fuzz_case_matches_frozen_expected_output(case):
    grid, wps, params = corridor_fuzz_gen.make_grid_for_case(
        case["case_idx"], seed=_FUZZ_CORPUS["seed"])
    ub, lb = _run(grid, wps, params, corridor_fuzz_gen.RES, corridor_fuzz_gen.ORIGIN)
    assert ub == case["expected_ub"], (
        f"case_idx={case['case_idx']}: ub mismatch — got {ub}, "
        f"frozen expected {case['expected_ub']}")
    assert lb == case["expected_lb"], (
        f"case_idx={case['case_idx']}: lb mismatch — got {lb}, "
        f"frozen expected {case['expected_lb']}")


def test_fuzz_corpus_has_at_least_1000_cases():
    assert len(_FUZZ_CORPUS["cases"]) >= 1000


def test_vectorized_fast_path_never_falls_back_on_golden_or_fuzz_corpus():
    """_compute_free_segments()のベクトル化パス(Phase 3-1)がゴールデン9種+
    ファジング1500件のいずれでもフォールバック(_compute_free_segments_scalar
    への委譲)を発火させていないことを確認する。上の等価性テストは出力の
    一致のみを見るため、フォールバックし続けていても偶然パスしうる
    (素朴実装同士を比較しているだけになる)ことを防ぐための、ベクトル化パス
    自体が実際に使われたことの直接証拠。"""
    for case_name in golden_cases.ALL_CASES:
        grid, wps, params = golden_cases.ALL_CASES[case_name]()
        res, origin = golden_cases.geometry_for(case_name)
        m = make_synthetic_map(grid, resolution=res, origin=origin)
        rp = make_synthetic_reference_path(m, wps, circular=False)
        # 2026-07-31修正(256節続報、クローズ作業): 従来はここで_run()が内部で
        # 別のrpを新規構築しており、この行より上で作ったrpは一度も実行されない
        # まま_fs_fallback_countを検査していた(getattr既定値0で常にパスする、
        # 何も検証していなかったテスト)。同一rpに対して実際に
        # update_path_constraintsを呼ぶよう修正した(下のファジングループと
        # 同じ書き方に統一)。
        rp.update_path_constraints(
            params.get("wp_id", 0), params["pose"], params["N"],
            params["model_length"], params["model_width"], params["safety_margin"])
        assert getattr(rp, "_fs_fallback_count", 0) == 0, case_name

    for case in _FUZZ_CORPUS["cases"]:
        grid, wps, params = corridor_fuzz_gen.make_grid_for_case(
            case["case_idx"], seed=_FUZZ_CORPUS["seed"])
        m = make_synthetic_map(grid, resolution=corridor_fuzz_gen.RES, origin=corridor_fuzz_gen.ORIGIN)
        rp = make_synthetic_reference_path(m, wps, circular=False)
        rp.update_path_constraints(
            params.get("wp_id", 0), params["pose"], params["N"],
            params["model_length"], params["model_width"], params["safety_margin"])
        assert getattr(rp, "_fs_fallback_count", 0) == 0, case["case_idx"]


def test_fs_line_cache_invalidated_and_falls_back_when_map_geometry_changes():
    """2026-07-31追加(256節続報、クローズ作業Phase 1): _fs_line_cacheは
    マップ幾何(origin/resolution/width/height)が起動後不変であることに
    依存している。調査の結果、現行コードにはこの前提を破る経路は存在しない
    (core/map.pyにorigin/resolution/width/heightを書き換えるメソッドが無く、
    reference_path再構築時も同一Mapインスタンスを再利用する)ことを確認済みだが、
    将来この前提が崩れても「古いピクセル座標を黙って使い続ける」ことがないよう、
    軽量ガード(キャッシュ構築時の幾何タプルとの比較)を追加した。このテストは
    そのガードが実際に機能することを、意図的にマップ幾何を書き換えて確認する
    (現行コードでは決して起こらないはずの人工的なシナリオ)。"""
    grid, wps, params = golden_cases.ALL_CASES["2_single_wp_2segs"]()
    m = make_synthetic_map(grid, resolution=golden_cases.RES, origin=golden_cases.ORIGIN)
    rp = make_synthetic_reference_path(m, wps, circular=False)

    ub1, lb1, _ = rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    assert rp._fs_fallback_count == 0
    wp1 = rp.get_waypoint(1)
    assert hasattr(wp1, "_fs_line_cache")
    cached_geom_before = wp1._fs_line_cache[4]

    # マップ幾何を人工的に書き換える(現行コードでは起こらない想定外シナリオ)。
    rp.map.origin = (rp.map.origin[0] + 100.0, rp.map.origin[1])

    ub2, lb2, _ = rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    # フォールバックが発火していること。_compute_free_segmentsはホライズン内の
    # 全waypoint(N個)に対して毎周期呼ばれるため、幾何変更直後の1周期では
    # N個全てのキャッシュが不一致と判定され、N回分のフォールバックが発生する。
    # フォールバック先の_compute_free_segments_scalar()は_fs_line_cacheを
    # 一切構築しないため、この時点ではキャッシュは「削除されたまま」になる
    # (誤った古い座標が残り続けるよりも安全な状態)。
    N = params["N"]
    assert rp._fs_fallback_count == N
    wp1_after_fallback = rp.get_waypoint(1)
    assert not hasattr(wp1_after_fallback, "_fs_line_cache")

    # 3回目の呼び出しでキャッシュが新しい幾何で再構築され、以降は
    # 追加のフォールバックが発生しない(高速パスへ復帰する)。
    rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    assert rp._fs_fallback_count == N


def test_fs_line_cache_detects_data_shape_change_even_when_attributes_unchanged():
    """2026-08-01追加(258節続報、マージ後フォローアップPhase 1): origin/
    resolution/width/heightのタプル照合だけでは、これら属性を変えずに
    map.dataだけ別形状の配列へ差し替えられた場合を検出できない。縮小方向は
    (幸い)IndexErrorで顕在化するが、拡大方向はキャッシュ座標が新しい配列の
    有効範囲内にある別セルを静かに読んでしまう——ガードが本来防ぎたい失敗
    そのものである。map_geomタプルへdata.shapeを加えたことで、この経路も
    検出できることを確認する(現行コードでは起こらないはずの人工的な
    シナリオ)。"""
    grid, wps, params = golden_cases.ALL_CASES["2_single_wp_2segs"]()
    m = make_synthetic_map(grid, resolution=golden_cases.RES, origin=golden_cases.ORIGIN)
    rp = make_synthetic_reference_path(m, wps, circular=False)

    rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    assert rp._fs_fallback_count == 0
    N = params["N"]

    # origin/resolution/width/heightは一切変更せず、map.dataだけをより大きい
    # 形状の配列へ差し替える(全面free)。width/heightは意図的に古いまま
    # (更新しない)ことで、「属性は不変・dataだけ拡大」という想定外シナリオを
    # 再現する。
    old_h, old_w = grid.shape
    bigger = np.ones((old_h + 20, old_w + 20), dtype=np.int8)
    rp.map.data = bigger

    rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    assert rp._fs_fallback_count == N, (
        "map.dataの形状変化(属性は不変)がガードで検出されなかった")
    wp1_after_fallback = rp.get_waypoint(1)
    assert not hasattr(wp1_after_fallback, "_fs_line_cache")

    # 3回目の呼び出しでキャッシュが新しいdata.shapeで再構築され、以降は
    # 追加のフォールバックが発生しない。
    rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])
    assert rp._fs_fallback_count == N
    wp1_rebuilt = rp.get_waypoint(1)
    assert hasattr(wp1_rebuilt, "_fs_line_cache")
    assert wp1_rebuilt._fs_line_cache[4][-1] == bigger.shape


def test_fp_absorption_case_construction_recipe_is_reproducible():
    """2026-08-01追加(258節続報、マージ後フォローアップPhase 2): 10_fp_
    absorption_tiebreakが将来役目を終えた(吸収が起きなくなりテストが失敗
    した)場合に、後任ケースを探索せず決定的に構築できるレシピを、実際に
    そのレシピどおり実行して検証する(グリッド合成までは不要、浮動小数点の
    吸収現象そのものが数値レベルで再現できることの確認で十分)。

    レシピ:
      1. 接頭辞和S(先行レイヤーの累積、例: 21.2級)と幅wを選ぶ
      2. w2 = w、w1 = math.nextafter(w, 0.0)(1ULP下の隣接値)とする
      3. w1 != w2 かつ S + w1 == S + w2 を検証する。不成立ならSを大きくする
         (Sが大きいほど加算の丸め粒度が粗くなるため、十分大きなSで必ず成立する)
      4. 検証済み(S, w1, w2)から逆算し、該当レイヤーのセグメント幅がw1/w2、
         先行レイヤーの累積がSになる境界セル配置を合成する(グリッド構築は
         corridor_golden_cases.pyの既存ケースと同じ要領で行う)

    このテストは手順1〜3(数値レベルの核心部分)を実際に実行し、吸収現象が
    再現することを直接assertする。手順4(グリッド合成)は
    10_fp_absorption_tiebreak自体で既に実演済みのため、ここでは繰り返さない。"""
    import math

    S = 21.2
    w = 4.2
    w2 = w
    w1 = math.nextafter(w, 0.0)

    assert w1 != w2, "隣接値が区別できていない(wの選び方を見直すこと)"
    assert (S + w1) == (S + w2), (
        f"S={S}では吸収が起きなかった(S+w1={S+w1!r}, S+w2={S+w2!r})。"
        "Sをより大きくして再試行すること(丸め粒度はSが大きいほど粗くなるため、"
        "十分大きなSで必ず吸収が起きる)。")

    # 生成された実際の値を記録(将来の後任ケース構築時の参考値として)。
    assert repr(w1) == '4.199999999999999'
    assert repr(w2) == '4.2'
    assert repr(S + w1) == repr(S + w2) == '25.4'


def test_fp_absorption_recipe_succeeds_with_larger_s_when_initial_s_is_insufficient():
    """レシピの手順3(Sを大きくすれば必ず成立する)自体を、意図的に小さすぎる
    Sから始めて確認する回帰テスト(レシピの頑健性そのものの検証)。"""
    import math

    w = 4.2
    w2 = w
    w1 = math.nextafter(w, 0.0)
    assert w1 != w2

    S = 0.0  # 小さすぎるS: この時点ではまだ吸収が起きない可能性が高い
    absorbed_at = None
    for _ in range(200):
        if (S + w1) == (S + w2):
            absorbed_at = S
            break
        S += 1.0
    assert absorbed_at is not None, "Sをどれだけ大きくしても吸収が起きなかった"
