"""263節(2026-08-02、予選環境ギャップ分析の準備Phase 4): analyze_perf_gap.py
の単体テスト。

analyze_perf_gap.pyはscripts/直下の独立ツール(rclpy非依存)であるため、
mpc_controller.py系テストのようなソーステキスト構造検証ではなく、モジュールを
直接importして実データ相当の合成ログで検証する。

合成ログの数値は全て手計算で検証可能な単純な値を選び、期待値をコメントで
併記する。実ログ(output/配下のローカル実験ログ、予選環境ログ
autoware(0801-01).log)でのスモークテストはPhase 4実施時に手動で確認済み
(コミットメッセージ・報告参照)であり、ここでは回帰を防ぐための合成ケースに
専念する。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import analyze_perf_gap as apg  # noqa: E402


# ---------------------------------------------------------------------------
# ANSIエスケープ除去(PartB-4実測で発覚: ROSログのカラー化により
# 'cpu_count=16\x1b[0m'のように値の直後へエスケープシーケンスが混入する)
# ---------------------------------------------------------------------------

def test_analyze_log_strips_ansi_escape_codes(tmp_path):
    p = tmp_path / "ansi.log"
    p.write_text(
        "\x1b[0m[PERF-PLATFORM] governor=performance scaling_max_freq=4000MHz "
        "rapl_power_limit=N/A cores_sampled=4 cpu_affinity=[2, 3, 4, 5] "
        "use_sim_time=True\x1b[0m\n"
        "\x1b[0m[PERF-PLATFORM] cgroup=v2 cpu_quota_cores=16.00 cpuset_cpus=0-15 "
        "memory_max=8.00GiB cpu_model=\"AMD Ryzen 9 6900HS\" cpu_count=16\x1b[0m\n")
    result = apg.analyze_log(p)
    assert result['platform']['cpu_count'] == '16'
    assert result['platform']['use_sim_time'] == 'True'


# ---------------------------------------------------------------------------
# パーサ単体
# ---------------------------------------------------------------------------

def test_parse_perf_dt_lines_extracts_all_fields():
    text = (
        "[PERF-DT] n=400 p50=10.00ms p95=15.00ms p99=20.00ms p999=22.00ms "
        "max=25.00ms eff_rate=39.50Hz over_budget=1.25% max_consec_over=2 "
        "margin=85.000ms")
    windows = apg.parse_perf_dt_lines(text)
    assert len(windows) == 1
    w = windows[0]
    assert w['n'] == 400
    assert w['p50_ms'] == 10.00
    assert w['p99_ms'] == 20.00
    assert w['p999_ms'] == 22.00
    assert w['max_ms'] == 25.00
    assert w['eff_rate_hz'] == 39.50
    assert w['over_budget_pct'] == 1.25
    assert w['max_consec_over'] == 2
    assert w['margin_ms'] == 85.0


def test_parse_perf_dt_lines_multiple_windows():
    text = "\n".join([
        "[PERF-DT] n=400 p50=10.00ms p95=15.00ms p99=20.00ms p999=22.00ms "
        "max=25.00ms eff_rate=40.00Hz over_budget=0.00% max_consec_over=0 "
        "margin=0.000ms",
        "[PERF-DT] n=400 p50=12.00ms p95=18.00ms p99=30.00ms p999=32.00ms "
        "max=40.00ms eff_rate=35.00Hz over_budget=2.00% max_consec_over=1 "
        "margin=0.000ms",
    ])
    windows = apg.parse_perf_dt_lines(text)
    assert len(windows) == 2


def test_parse_perf_lines_extracts_budget_and_overs():
    line = (
        "[PERF] n=400 work avg=6.0ms max=44.5ms >13.9ms=5回 "
        "work_cpu avg=5.0ms max=18.1ms >13.9ms=1回 n_dynobs_max=3 | mpc=6.0")
    windows = apg.parse_perf_lines(line)
    assert len(windows) == 1
    w = windows[0]
    assert w['n'] == 400
    assert w['work_avg_ms'] == 6.0
    assert w['work_max_ms'] == 44.5
    assert w['work_cpu_avg_ms'] == 5.0
    assert w['work_cpu_max_ms'] == 18.1
    assert w['budget_ms'] == 13.9
    assert w['work_over'] == 5
    assert w['work_cpu_over'] == 1


def test_parse_perf_rusage_lines_handles_na_run_delay():
    line = (
        "[PERF-RUSAGE] n=400 cpu_time=2.00s wall_time=4.00s cpu_ratio=0.50 "
        "nivcsw=100 nvcsw=10 freq_avg=N/A freq_min=N/A freq_max=N/A "
        "run_delay_avg=N/A run_delay_max=N/A")
    windows = apg.parse_perf_rusage_lines(line)
    w = windows[0]
    assert w['cpu_time_s'] == 2.0
    assert w['wall_time_s'] == 4.0
    assert w['cpu_ratio'] == 0.50
    assert w['nivcsw'] == 100
    assert w['run_delay_avg_ms'] is None
    assert w['run_delay_max_ms'] is None


def test_parse_perf_rusage_lines_extracts_run_delay_when_present():
    line = (
        "[PERF-RUSAGE] n=400 cpu_time=2.00s wall_time=4.00s cpu_ratio=0.50 "
        "nivcsw=100 nvcsw=10 freq_avg=N/A freq_min=N/A freq_max=N/A "
        "run_delay_avg=3.50ms run_delay_max=9.10ms")
    w = apg.parse_perf_rusage_lines(line)[0]
    assert w['run_delay_avg_ms'] == 3.50
    assert w['run_delay_max_ms'] == 9.10


def test_parse_perf_spike_lines_dominant_excludes_rollup_keys():
    """mpc(親ロールアップ)より子(mpc_setup)が大きい場合、子が
    dominant_componentに選ばれること(mpcは常に子の合計に近く数値的に
    優位になりやすいため、除外対象)。"""
    line = (
        "[PERF-SPIKE] loop=100 loop_mod100=0 work=40.00ms work_cpu=15.00ms "
        "budget=13.90ms cache_builds_diff=0 nivcsw_diff=5 gen2_gc=False "
        "gen2_gc_dur=0.00ms nseg=(0, 20, 0) nseg_changed=False freq_avg=N/A "
        "freq_min=N/A freq_max=N/A run_delay=N/A | mpc=25.00 mpc_setup=20.00 "
        "mpc_corridor=4.00 mpc_solve=1.00 prep=5.00 raster=3.00")
    spikes = apg.parse_perf_spike_lines(line)
    assert len(spikes) == 1
    s = spikes[0]
    assert s['loop'] == 100
    assert s['work_ms'] == 40.00
    assert s['work_cpu_ms'] == 15.00
    assert s['budget_ms'] == 13.90
    assert s['nivcsw_diff'] == 5
    assert s['gen2_gc'] is False
    assert s['run_delay_ms'] is None
    assert s['dominant_component'] == 'mpc_setup'
    assert s['dominant_component_ms'] == 20.00


def test_parse_perf_spike_lines_gen2_gc_true_detected():
    line = (
        "[PERF-SPIKE] loop=1 loop_mod100=1 work=30.00ms work_cpu=10.00ms "
        "budget=13.90ms cache_builds_diff=0 nivcsw_diff=1 gen2_gc=True "
        "gen2_gc_dur=5.00ms nseg=(0, 20, 0) nseg_changed=False freq_avg=N/A "
        "freq_min=N/A freq_max=N/A run_delay=2.50ms | gc=5.00 prep=1.00")
    s = apg.parse_perf_spike_lines(line)[0]
    assert s['gen2_gc'] is True
    assert s['run_delay_ms'] == 2.50


def test_parse_perf_platform_extracts_all_263_fields():
    text = "\n".join([
        '[PERF-PLATFORM] governor=performance scaling_max_freq=4935MHz '
        'rapl_power_limit=N/A cores_sampled=16 cpu_affinity=[2, 3, 4, 5] '
        'use_sim_time=True',
        "[PERF-PLATFORM] colocated_affinity={'rviz2': '0-1,6-15'}",
        '[PERF-PLATFORM] cgroup=v2 cpu_quota_cores=3.00 cpuset_cpus=7-9 '
        'memory_max=12.00GiB cpu_model="AMD Ryzen 9 6900HS with Radeon '
        'Graphics" cpu_count=16',
        '[PERF-PLATFORM] availability: scaling_cur_freq=OK '
        'sched_schedstats=N/A cgroup_cpu_quota=OK cgroup_cpuset=OK '
        'cgroup_memory_max=OK',
    ])
    p = apg.parse_perf_platform(text)
    assert p['governor'] == 'performance'
    assert p['cpu_affinity'] == '[2, 3, 4, 5]'
    assert p['use_sim_time'] == 'True'
    assert p['colocated_affinity'] == "{'rviz2': '0-1,6-15'}"
    assert p['cgroup_version'] == 'v2'
    assert p['cpu_quota_cores'] == '3.00'
    assert p['cpuset_cpus'] == '7-9'
    assert p['memory_max'] == '12.00GiB'
    assert p['cpu_model'] == 'AMD Ryzen 9 6900HS with Radeon Graphics'
    assert p['cpu_count'] == '16'
    assert p['availability_scaling_cur_freq'] == 'OK'
    assert p['availability_sched_schedstats'] == 'N/A'


def test_parse_perf_platform_all_none_when_no_lines_present():
    """263節Phase 1/2より前のログ(PERF-PLATFORM行自体が無い)でも
    クラッシュせず全項目Noneを返すこと。"""
    p = apg.parse_perf_platform("no relevant lines here")
    assert all(v is None for v in p.values())


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------

def test_aggregate_dt_uses_max_across_windows_for_percentiles():
    windows = [
        {'n': 400, 'p99_ms': 20.0, 'eff_rate_hz': 40.0, 'over_budget_pct': 0.0,
         'max_consec_over': 0, 'margin_ms': 0.0, 'p50_ms': 10, 'p95_ms': 15,
         'p999_ms': 22, 'max_ms': 25},
        {'n': 400, 'p99_ms': 30.0, 'eff_rate_hz': 30.0, 'over_budget_pct': 2.0,
         'max_consec_over': 1, 'margin_ms': 0.0, 'p50_ms': 12, 'p95_ms': 18,
         'p999_ms': 32, 'max_ms': 40},
    ]
    agg = apg.aggregate_dt(windows)
    assert agg['p99_ms'] == 30.0  # 2窓のうち大きい方
    assert agg['total_n'] == 800
    assert agg['eff_rate_hz'] == 35.0  # 重み(n)が等しいので単純平均と一致


def test_aggregate_dt_returns_none_when_no_windows():
    assert apg.aggregate_dt([]) is None
    assert apg.aggregate_dt([{'n': None}]) is None


def test_aggregate_perf_computes_over_ratio_from_pooled_counts():
    windows = [
        {'n': 100, 'work_avg_ms': 5.0, 'work_max_ms': 10.0, 'work_over': 1,
         'work_cpu_avg_ms': 4.0, 'work_cpu_max_ms': 8.0, 'work_cpu_over': 0,
         'budget_ms': 13.9},
        {'n': 100, 'work_avg_ms': 7.0, 'work_max_ms': 20.0, 'work_over': 3,
         'work_cpu_avg_ms': 6.0, 'work_cpu_max_ms': 9.0, 'work_cpu_over': 1,
         'budget_ms': 13.9},
    ]
    agg = apg.aggregate_perf(windows)
    assert agg['total_n'] == 200
    assert agg['work_over_pct'] == 2.0  # (1+3)/200*100
    assert agg['work_cpu_over_pct'] == 0.5  # (0+1)/200*100
    assert agg['work_avg_ms'] == 6.0  # 等重み平均
    assert agg['work_max_ms'] == 20.0
    assert agg['budget_ms'] == 13.9


def test_aggregate_rusage_computes_overall_cpu_ratio_from_pooled_seconds():
    windows = [
        {'n': 100, 'cpu_time_s': 1.0, 'wall_time_s': 2.0, 'nivcsw': 50,
         'run_delay_avg_ms': None, 'run_delay_max_ms': None},
        {'n': 100, 'cpu_time_s': 3.0, 'wall_time_s': 4.0, 'nivcsw': 150,
         'run_delay_avg_ms': None, 'run_delay_max_ms': None},
    ]
    agg = apg.aggregate_rusage(windows)
    assert agg['cpu_ratio_overall'] == (1.0 + 3.0) / (2.0 + 4.0)
    assert agg['nivcsw_avg_per_window'] == 100.0
    assert agg['nivcsw_total'] == 200
    assert agg['run_delay_available'] is False


def test_aggregate_rusage_run_delay_available_when_present():
    windows = [
        {'n': 100, 'cpu_time_s': 1.0, 'wall_time_s': 2.0, 'nivcsw': 50,
         'run_delay_avg_ms': 2.0, 'run_delay_max_ms': 5.0},
        {'n': 100, 'cpu_time_s': 1.0, 'wall_time_s': 2.0, 'nivcsw': 50,
         'run_delay_avg_ms': 4.0, 'run_delay_max_ms': 9.0},
    ]
    agg = apg.aggregate_rusage(windows)
    assert agg['run_delay_available'] is True
    assert agg['run_delay_avg_ms'] == 3.0
    assert agg['run_delay_max_ms'] == 9.0


def test_aggregate_spikes_counts_dominant_components():
    spikes = [
        {'dominant_component': 'raster', 'work_ms': 40.0},
        {'dominant_component': 'raster', 'work_ms': 50.0},
        {'dominant_component': 'prep', 'work_ms': 30.0},
    ]
    agg = apg.aggregate_spikes(spikes)
    assert agg['count'] == 3
    assert agg['dominant_components'] == {'raster': 2, 'prep': 1}
    assert agg['work_ms_max'] == 50.0


def test_aggregate_spikes_empty_list():
    agg = apg.aggregate_spikes([])
    assert agg == {'count': 0, 'dominant_components': {}}


# ---------------------------------------------------------------------------
# end-to-end: analyze_log + レポート整形 + 比較/推奨ロジック
# ---------------------------------------------------------------------------

def _write_synthetic_log(tmp_path, name, budget_ms, p99_ms, work_cpu_avg_ms,
                          work_cpu_max_ms, nivcsw, run_delay_avg_ms=None,
                          cpu_affinity="[2, 3, 4, 5]"):
    over_pct = 0.0
    lines = [
        f"[PERF-DT] n=400 p50={p99_ms * 0.5:.2f}ms p95={p99_ms * 0.8:.2f}ms "
        f"p99={p99_ms:.2f}ms p999={p99_ms * 1.1:.2f}ms max={p99_ms * 1.3:.2f}ms "
        f"eff_rate=40.00Hz over_budget={over_pct:.2f}% max_consec_over=0 "
        f"margin=0.000ms",
        f"[PERF] n=400 work avg={work_cpu_avg_ms + 1.0:.1f}ms "
        f"max={work_cpu_max_ms + 5.0:.1f}ms >{budget_ms:.1f}ms=1回 "
        f"work_cpu avg={work_cpu_avg_ms:.1f}ms max={work_cpu_max_ms:.1f}ms "
        f">{budget_ms:.1f}ms=0回 n_dynobs_max=2 | mpc={work_cpu_avg_ms:.2f}",
        f"[PERF-RUSAGE] n=400 cpu_time=2.00s wall_time=4.00s cpu_ratio=0.50 "
        f"nivcsw={nivcsw} nvcsw=5 freq_avg=N/A freq_min=N/A freq_max=N/A "
        + (f"run_delay_avg={run_delay_avg_ms:.2f}ms run_delay_max="
           f"{run_delay_avg_ms * 2:.2f}ms" if run_delay_avg_ms is not None
           else "run_delay_avg=N/A run_delay_max=N/A"),
        f'[PERF-PLATFORM] governor=performance scaling_max_freq=4000MHz '
        f'rapl_power_limit=N/A cores_sampled=4 cpu_affinity={cpu_affinity} '
        f'use_sim_time=False',
    ]
    p = tmp_path / name
    p.write_text("\n".join(lines))
    return p


def test_analyze_log_computes_j_from_dt_p99_and_perf_budget(tmp_path):
    p = _write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=30.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1000)
    result = apg.analyze_log(p)
    assert result['budget_ms'] == 13.9
    assert abs(result['j_ms'] - (30.0 - 13.9)) < 1e-9


def test_analyze_log_handles_missing_perf_dt_gracefully(tmp_path):
    p = tmp_path / "no_dt.log"
    p.write_text("[PERF-RUSAGE] n=400 cpu_time=1.00s wall_time=2.00s "
                 "cpu_ratio=0.50 nivcsw=10 nvcsw=1 freq_avg=N/A freq_min=N/A "
                 "freq_max=N/A run_delay_avg=N/A run_delay_max=N/A")
    result = apg.analyze_log(p)
    assert result['dt'] is None
    assert result['j_ms'] is None
    # クラッシュせずレポート整形もできること
    report = apg.format_single_report(result)
    assert '[PERF-DT]行が見つからない' in report


def test_format_single_report_smoke(tmp_path):
    p = _write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500)
    result = apg.analyze_log(p)
    report = apg.format_single_report(result)
    assert 'J=' in report
    assert 'work_cpu avg=' in report
    assert '[PERF-PLATFORM]' in report


def test_recommend_go_when_qualifying_work_cpu_well_within_new_budget(tmp_path):
    local = apg.analyze_log(_write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=8.0, nivcsw=1500,
        run_delay_avg_ms=1.0))
    qual = apg.analyze_log(_write_synthetic_log(
        tmp_path, "qual.log", budget_ms=25.0, p99_ms=40.0,
        work_cpu_avg_ms=6.0, work_cpu_max_ms=9.0, nivcsw=2000,
        run_delay_avg_ms=1.5, cpu_affinity="[7, 8, 9]"))
    verdict, reasons = apg._recommend(local, qual, target_hz=72.0)
    assert verdict == 'GO'
    assert any('新予算内' in r for r in reasons)


def test_recommend_flags_when_qualifying_work_cpu_max_exceeds_new_budget(tmp_path):
    qual = apg.analyze_log(_write_synthetic_log(
        tmp_path, "qual.log", budget_ms=25.0, p99_ms=40.0,
        work_cpu_avg_ms=6.0, work_cpu_max_ms=34.6, nivcsw=6130,
        cpu_affinity="[7, 8, 9]"))
    local = apg.analyze_log(_write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=18.1, nivcsw=1706))
    verdict, reasons = apg._recommend(local, qual, target_hz=72.0)
    assert verdict == '追加情報必要'
    assert any('新予算' in r and 'max' in r for r in reasons)


def test_recommend_handles_missing_j_without_crash(tmp_path):
    qual_no_dt = tmp_path / "qual_no_dt.log"
    qual_no_dt.write_text("[PERF] n=400 work avg=5.0ms max=10.0ms >25.0ms=0回 "
                          "work_cpu avg=4.0ms max=8.0ms >25.0ms=0回 "
                          "n_dynobs_max=1 | mpc=4.0")
    qual = apg.analyze_log(qual_no_dt)
    local = apg.analyze_log(_write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500))
    verdict, reasons = apg._recommend(local, qual, target_hz=72.0)
    assert verdict == '追加情報必要'
    assert any('J_予選が算出できず' in r for r in reasons)


def test_format_comparison_report_smoke(tmp_path):
    local = apg.analyze_log(_write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500))
    qual = apg.analyze_log(_write_synthetic_log(
        tmp_path, "qual.log", budget_ms=25.0, p99_ms=40.0,
        work_cpu_avg_ms=6.0, work_cpu_max_ms=12.0, nivcsw=3000,
        cpu_affinity="[7, 8, 9]"))
    report = apg.format_comparison_report(local, qual, target_hz=72.0)
    assert '72Hz成立予測' in report
    assert 'GO' in report or '追加情報必要' in report


def test_main_single_mode_exit_zero(tmp_path, capsys):
    p = _write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500)
    rc = apg.main([str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'ジッタ' in out


def test_main_comparison_mode_exit_zero(tmp_path, capsys):
    local = _write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500)
    qual = _write_synthetic_log(
        tmp_path, "qual.log", budget_ms=25.0, p99_ms=40.0,
        work_cpu_avg_ms=6.0, work_cpu_max_ms=12.0, nivcsw=3000,
        cpu_affinity="[7, 8, 9]")
    rc = apg.main([str(local), str(qual)])
    assert rc == 0
    out = capsys.readouterr().out
    assert '72Hz成立予測' in out


def test_main_json_output_is_valid_json(tmp_path, capsys):
    import json
    p = _write_synthetic_log(
        tmp_path, "local.log", budget_ms=13.9, p99_ms=25.0,
        work_cpu_avg_ms=5.0, work_cpu_max_ms=10.0, nivcsw=1500)
    rc = apg.main([str(p), '--json'])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed['budget_ms'] == 13.9
