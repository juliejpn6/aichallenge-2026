"""[PERF-CORRIDOR]計測分解のユニットテスト(254節続報続、Phase 1)。

update_path_constraints()に追加した内訳計測(phaseA=_compute_free_segments
のラスタ走査時間、phaseB=itertools.productによる組み合わせ探索時間、および
呼び出し回数/最大セグメント数/組み合わせ数/has_collision_in_line呼び出し
回数/完全封鎖フォールバック回数)が、既知の9種ゴールデンケースに対して
意図通りの値を積算し、_PERFC_REPORT_EVERY周期ごとに[PERF-CORRIDOR]ログを
出力してから窓をリセットすることを検証する。

期待値は本実装を実際に1サイクル実行して実測した値(corridor_golden_cases.py
の各ケースの設計意図と整合することを個別に確認済み)。control_rateの
レートスケーリング機構(mpc_controller.py)とは独立した固定周期(400)の窓で
あることも確認する(「control_rateは別課題」という本タスクの制約の裏付け)。
"""
import io
import re
from contextlib import redirect_stdout

import corridor_golden_cases as golden_cases
from corridor_test_helpers import make_synthetic_map, make_synthetic_reference_path


def _build(case_name):
    grid, wps, params = golden_cases.ALL_CASES[case_name]()
    m = make_synthetic_map(grid, resolution=golden_cases.RES, origin=golden_cases.ORIGIN)
    rp = make_synthetic_reference_path(m, wps, circular=False)
    rp._perfc_init()
    return rp, params


def _run_once(rp, params):
    return rp.update_path_constraints(
        params["wp_id"], params["pose"], params["N"],
        params["model_length"], params["model_width"], params["safety_margin"])


# case_name -> (phaseB_invocations, max_nseg, combination_count,
#               collision_check_count, complete_blockage_count) after 1 cycle
# 2026-07-31追加(Phase 3-2): has_collision_in_lineの呼び出しは隣接レイヤー間の
# ペアごとに一度だけ計算するメモ化に変更されたため、collision_check_countは
# Phase 3-2適用前(itertools.product全組み合わせ×D回、例: 3_multi_wp_splitは
# 932回)より大幅に少ない値になる。combination_count(=列挙した組み合わせ数
# そのもの)はitertools.productの列挙自体を変更していないため不変。
_EXPECTED_COUNTERS = {
    "1_no_obstacle": (0, 1, 0, 0, 0),
    "2_single_wp_2segs": (1, 2, 2, 6, 0),
    "3_multi_wp_split": (1, 5, 240, 59, 0),
    "6_complete_blockage": (1, 2, 32, 18, 1),
    "7_start_blockage": (1, 2, 32, 18, 1),
    "8_single_surviving_path": (1, 2, 32, 18, 0),
    "9_symmetric_tie": (1, 2, 32, 18, 0),
}


def test_counters_match_known_values_per_golden_case():
    for case_name, expected in _EXPECTED_COUNTERS.items():
        rp, params = _build(case_name)
        _run_once(rp, params)
        got = (rp._perfc_phaseB_invocations, rp._perfc_max_nseg,
               rp._perfc_combination_count, rp._perfc_collision_check_count,
               rp._perfc_complete_blockage_count)
        assert got == expected, f"{case_name}: got {got}, expected {expected}"


def test_no_obstacle_case_has_zero_phaseB_activity():
    rp, params = _build("1_no_obstacle")
    _run_once(rp, params)
    assert rp._perfc_phaseB_invocations == 0
    assert rp._perfc_combination_count == 0
    assert rp._perfc_collision_check_count == 0
    assert rp._perfc_complete_blockage_count == 0


def test_complete_blockage_counter_fires_only_for_full_blockage_cases():
    for case_name in ("6_complete_blockage", "7_start_blockage"):
        rp, params = _build(case_name)
        _run_once(rp, params)
        assert rp._perfc_complete_blockage_count == 1, case_name

    for case_name in ("8_single_surviving_path", "9_symmetric_tie",
                      "2_single_wp_2segs", "3_multi_wp_split"):
        rp, params = _build(case_name)
        _run_once(rp, params)
        assert rp._perfc_complete_blockage_count == 0, case_name


def test_phaseA_and_phaseB_times_are_nonnegative_and_phaseA_always_runs():
    rp, params = _build("3_multi_wp_split")
    _run_once(rp, params)
    # サイクル完了直後は_perfc_init()でリセットされる前(まだ400周期未満)なので
    # 直近のphaseA/phaseB時間は_perfc_phaseA_times/_perfc_phaseB_timesの末尾に残る。
    assert len(rp._perfc_phaseA_times) == 1
    assert len(rp._perfc_phaseB_times) == 1
    assert rp._perfc_phaseA_times[0] >= 0.0
    assert rp._perfc_phaseB_times[0] >= 0.0


def test_report_fires_exactly_at_report_every_cycles_and_then_resets():
    rp, params = _build("2_single_wp_2segs")
    every = rp._PERFC_REPORT_EVERY
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(every - 1):
            _run_once(rp, params)
        assert "[PERF-CORRIDOR]" not in buf.getvalue()
        _run_once(rp, params)
    out = buf.getvalue()
    lines = [l for l in out.splitlines() if "[PERF-CORRIDOR]" in l]
    assert len(lines) == 1, out
    assert f"n={every}" in lines[0]
    # レポート後は窓がリセットされ、次サイクルからカウントし直す(control_rateとは無関係の固定周期)。
    assert rp._perfc_cycles == 0
    assert rp._perfc_phaseA_times == []
    assert rp._perfc_phaseB_times == []
    assert rp._perfc_phaseB_invocations == 0
    assert rp._perfc_combination_count == 0
    assert rp._perfc_collision_check_count == 0
    assert rp._perfc_complete_blockage_count == 0
    assert rp._perfc_max_nseg == 0
    assert rp._perfc_cache_build_count == 0


def test_cache_build_count_fires_once_per_waypoint_then_stays_zero():
    """2026-07-31追加(256節続報、クローズ作業Phase 4-2): _fs_line_cacheは
    遅延構築(各waypointが初めて参照された周期にのみ発生)であることを直接
    検証する。同一waypoint集合に対して複数周期回した場合、キャッシュ構築は
    最初の周期(N個のwaypoint全て)でのみ発生し、以降の周期ではゼロが
    継続するはずである(プリウォームは行わないため、初回のみのコストで
    あることの裏付け)。

    2026-08-01追記(258節続報、マージ後フォローアップPhase 4-1): 本テストの
    「巡航中はゼロが継続する」という期待は、**同一のReferencePath/waypoint
    集合を使い続ける**シナリオに限定される。mpc_controller.pyの
    update_by_topic有効構成(100周期ごとに新しいReferencePathを構築し直す
    経路)では、再構築直後の周期にホライズン内N個分のキャッシュ構築バーストが
    **正常として**発生する(新しいWaypointオブジェクトはキャッシュを持たない
    ため)。本テストはこの再構築シナリオを一切exerciseしていないため、
    「巡航中ゼロ」という期待とは矛盾しない(両者は別々のシナリオを検証して
    いる)。再構築バースト自体の挙動は
    test_cache_build_count_bursts_after_reference_path_rebuild で別途確認する。"""
    rp, params = _build("3_multi_wp_split")
    N = params["N"]

    _run_once(rp, params)
    assert rp._perfc_cache_build_count == N  # 1周期目: ホライズン内全waypoint分

    build_count_after_first = rp._perfc_cache_build_count
    for _ in range(5):
        _run_once(rp, params)
    # 2周期目以降は同じwaypointオブジェクトを使い続けるため、追加の
    # キャッシュ構築は発生しない。
    assert rp._perfc_cache_build_count == build_count_after_first


def test_cache_build_count_bursts_after_reference_path_rebuild():
    """2026-08-01追加(258節続報、マージ後フォローアップPhase 4-1):
    update_by_topic有効構成でmpc_controller.pyが100周期ごとに新しい
    ReferencePathを構築し直す経路(_create_reference_path_from_autoware_
    trajectory)を模擬する。同一のMapを再利用しつつ新しいwaypoint集合
    (=新しいWaypointオブジェクト群)を持つReferencePathを構築すると、
    直後の1周期でホライズン内N個分のキャッシュ構築バーストが発生する
    ことを確認する——これは258節Phase 1で調査済みの通り正常な設計であり、
    バグではない。"""
    grid, wps, params = golden_cases.ALL_CASES["3_multi_wp_split"]()
    m = make_synthetic_map(grid, resolution=golden_cases.RES, origin=golden_cases.ORIGIN)
    N = params["N"]

    rp1 = make_synthetic_reference_path(m, wps, circular=False)
    rp1._perfc_init()
    _run_once(rp1, params)
    assert rp1._perfc_cache_build_count == N
    for _ in range(3):
        _run_once(rp1, params)
    assert rp1._perfc_cache_build_count == N  # 巡航中は増えない(既存テストと同じ)

    # 「100周期ごとの再構築」を模擬: 同一のMapオブジェクトmを再利用しつつ、
    # 新しいwaypoint集合(同じgridから作り直した、別インスタンスのWaypoint群)
    # を持つ新しいReferencePathへ切り替える。
    _, wps2, _ = golden_cases.ALL_CASES["3_multi_wp_split"]()
    rp2 = make_synthetic_reference_path(m, wps2, circular=False)
    rp2._perfc_init()
    _run_once(rp2, params)
    # 再構築直後の1周期はN個分のキャッシュ構築バーストが発生する(正常)。
    assert rp2._perfc_cache_build_count == N
    for _ in range(3):
        _run_once(rp2, params)
    # バースト後は再び巡航状態(ゼロ継続)へ戻る。
    assert rp2._perfc_cache_build_count == N


def test_report_line_contains_percentile_and_count_fields():
    rp, params = _build("2_single_wp_2segs")
    every = rp._PERFC_REPORT_EVERY
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(every):
            _run_once(rp, params)
    line = [l for l in buf.getvalue().splitlines() if "[PERF-CORRIDOR]" in l][0]
    for field in ("phaseA(segments)", "phaseB(search)", "avg=", "p50=", "p95=",
                  "p99=", "max=", "invocations=", "max_nseg=", "combos=",
                  "collision_checks=", "complete_blockage=", "cache_builds="):
        assert field in line, f"missing {field!r} in: {line}"


def test_report_cadence_is_independent_of_control_rate_not_wired_to_rate_scaling():
    """_PERFC_REPORT_EVERYはmpc_controller.pyの_rate_scaled_cycles()を一切
    経由しない固定定数であることをソース上でも確認する(control_rateは別課題、
    という本タスクの絶対制約の裏付け)。"""
    import inspect
    from multi_purpose_mpc_ros.core import reference_path as rp_module
    src = inspect.getsource(rp_module.ReferencePath._perfc_init)
    assert "_rate_scaled_cycles" not in src
    assert "control_rate" not in src
    assert rp_module.ReferencePath._PERFC_REPORT_EVERY == 400
