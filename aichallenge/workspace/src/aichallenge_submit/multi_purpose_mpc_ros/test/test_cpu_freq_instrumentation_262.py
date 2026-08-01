"""262節続報(2026-08-01、プラットフォーム状態計装Phase 1): CPU周波数
(scaling_cur_freq)を[PERF-SPIKE]/[PERF-RUSAGE]計装へ追加。

背景: 72Hz再実測(出力制限解除後)は大幅改善したが、2/4基準が依然未達
だった。[PERF-SPIKE]の帰属分析で仮説(a)reference_path再構築・(c)GC世代2・
(d)OSQP次元変化は完全に排除され、残る容疑は(b)系統(nivcswに現れる
「CPU競合」)のみだが、(b)には「競合」だけでなく「周波数変動(DVFS/電源制限、
nivcswには現れない)」も含まれうる。本計装はCPU周波数を直接観測し、
出力制限の有無・スパイクとの同時性を切り分ける材料を残す。

mpc_controller.pyはrclpy依存のため直接importできない。ロジック自体は
純粋な算術のためミラー関数で検証し、mpc_controller.py側の実装(呼び出し
配線・N/A分岐・失敗時の安全側フォールバック)はソーステキスト構造検証で
確認する(既存のtest_perf_spike_forensics_261.pyと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def mirror_freq_stats_mhz(freq_khz):
    """_pf_sample_cpu_freq()の全コア平均・最小・最大算出部分のミラー。"""
    avg_mhz = sum(freq_khz) / len(freq_khz) / 1000.0
    min_mhz = min(freq_khz) / 1000.0
    max_mhz = max(freq_khz) / 1000.0
    return avg_mhz, min_mhz, max_mhz


def mirror_window_update(win_sum, win_min, win_max, win_samples, avg_mhz):
    """窓集計(min/avg/max)の更新部分のミラー。"""
    win_sum += avg_mhz
    win_samples += 1
    win_min = avg_mhz if win_min is None else min(win_min, avg_mhz)
    win_max = avg_mhz if win_max is None else max(win_max, avg_mhz)
    return win_sum, win_min, win_max, win_samples


# ---------------------------------------------------------------------------
# ①非矛盾性: 周波数統計・窓集計そのものの正しさ
# ---------------------------------------------------------------------------

def test_freq_stats_known_values():
    avg_mhz, min_mhz, max_mhz = mirror_freq_stats_mhz([2400000, 1200000, 3600000])
    assert avg_mhz == 2400.0
    assert min_mhz == 1200.0
    assert max_mhz == 3600.0


def test_freq_stats_single_core():
    avg_mhz, min_mhz, max_mhz = mirror_freq_stats_mhz([1800000])
    assert avg_mhz == min_mhz == max_mhz == 1800.0


def test_window_update_accumulates_across_samples():
    win_sum, win_min, win_max, n = 0.0, None, None, 0
    for avg_mhz in (2000.0, 3000.0, 1500.0):
        win_sum, win_min, win_max, n = mirror_window_update(
            win_sum, win_min, win_max, n, avg_mhz)
    assert n == 3
    assert win_sum == 6500.0
    assert win_min == 1500.0
    assert win_max == 3000.0


def test_window_update_first_sample_sets_min_and_max():
    win_sum, win_min, win_max, n = mirror_window_update(0.0, None, None, 0, 2500.0)
    assert win_min == win_max == 2500.0
    assert n == 1


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_pf_read_cpu_freqs_khz_returns_none_on_any_failure():
    """1コアでも読み取りに失敗したら、部分的な値ではなくNoneを返す
    (帰属分析を歪めないための全部か無かの規律)。"""
    idx = _SRC.index("def _pf_read_cpu_freqs_khz(self, paths=None):")
    idx_end = _SRC.index("\n    def _pf_sample_cpu_freq", idx)
    snippet = _SRC[idx:idx_end]
    assert "except (OSError, ValueError):" in snippet
    assert "return None" in snippet


def test_pf_init_disables_freq_when_startup_probe_fails():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_freq_available = len(self._pf_freq_paths) > 0" in snippet
    assert "self._pf_read_cpu_freqs_khz() is None" in snippet
    assert "self._pf_freq_available = False" in snippet


def test_pf_init_warns_once_when_freq_unavailable():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "self.get_logger().warn(" in snippet
    assert "N/A" in snippet


def test_pf_cycle_end_samples_cpu_freq():
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_sample_cpu_freq()" in snippet
    # スパイクダンプ判定より前(=毎サイクル無条件)にサンプルされること。
    assert snippet.index("self._pf_sample_cpu_freq()") < snippet.index(
        "self._pf_dump_spike_if_needed(work, work_cpu)")


def test_perf_spike_log_line_contains_freq_fields():
    """freq_avg/min/maxはfreq_str変数として組み立てられ%sで埋め込まれるため、
    print呼び出し直前のfreq_str構築を含む範囲で検証する。"""
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "freq_str" in snippet
    for field in ("freq_avg=", "freq_min=", "freq_max="):
        assert field in snippet, f"missing {field!r} in [PERF-SPIKE] log line"


def test_perf_rusage_log_line_contains_freq_fields():
    idx = _SRC.index("_cpu_ratio = _cpu_time / _wall_time")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "_freq_str" in snippet
    for field in ("freq_avg=", "freq_min=", "freq_max="):
        assert field in snippet, f"missing {field!r} in [PERF-RUSAGE] log line"


def test_perf_rusage_resets_freq_window_after_report():
    idx = _SRC.index("'[PERF-RUSAGE]")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_freq_win_sum = 0.0" in snippet
    assert "self._pf_freq_win_min = None" in snippet
    assert "self._pf_freq_win_max = None" in snippet
    assert "self._pf_freq_win_samples = 0" in snippet


def test_freq_fields_report_na_when_unavailable():
    """freq_available=Falseまたはサンプル無しの場合、[PERF-SPIKE]/
    [PERF-RUSAGE]ともN/Aへフォールバックし、クラッシュしないこと。"""
    spike_idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    spike_end = _SRC.index("\n    # 2026-07-31追加(255節続報", spike_idx)
    spike_snippet = _SRC[spike_idx:spike_end]
    assert "freq_avg=N/A freq_min=N/A freq_max=N/A" in spike_snippet

    rusage_idx = _SRC.index("_cpu_ratio = _cpu_time / _wall_time")
    rusage_end = _SRC.index("flush=True)", rusage_idx)
    rusage_snippet = _SRC[rusage_idx:rusage_end]
    assert "freq_avg=N/A freq_min=N/A freq_max=N/A" in rusage_snippet


def test_platform_checklist_logged_once_at_pf_init():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_log_platform_checklist()" in snippet


def test_platform_checklist_log_line_contains_required_fields():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("governor=", "scaling_max_freq=", "rapl_power_limit=",
                  "cores_sampled="):
        assert field in snippet, f"missing {field!r} in [PERF-PLATFORM] log line"


def test_platform_checklist_does_not_crash_on_missing_sysfs_files():
    """governor/max_freq/raplいずれも個別にtry/exceptで守られ、1項目の
    欠落が他項目やノード起動全体を止めないことを確認する。"""
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    assert snippet.count("except OSError:") + snippet.count(
        "except (OSError, ValueError):") >= 3


def test_glob_import_present():
    assert "import glob as _glob" in _SRC


def test_existing_perf_spike_and_rusage_tags_unchanged():
    """既存の[PERF-SPIKE]/[PERF-RUSAGE]タグ自体は維持されていることの回帰確認
    (周波数フィールドは追加のみ)。"""
    assert "'[PERF-SPIKE] loop=%d" in _SRC
    assert "'[PERF-RUSAGE] n=%d" in _SRC


# ---------------------------------------------------------------------------
# 262節続報(判定基準改訂+コア限定化Phase 1): 周波数記録のコア限定化
# ---------------------------------------------------------------------------

def test_pf_affinity_freq_paths_uses_sched_getaffinity():
    idx = _SRC.index("def _pf_affinity_freq_paths(self):")
    idx_end = _SRC.index("\n    def _pf_sample_cpu_freq", idx)
    snippet = _SRC[idx:idx_end]
    assert "os.sched_getaffinity(0)" in snippet


def test_pf_affinity_freq_paths_falls_back_to_all_cores_on_failure():
    """アフィニティが取得できない・既知コアと一致しない場合は、N/A化せず
    全コアへフォールバックする(限定できないなら従来通り安全側に倒す)。"""
    idx = _SRC.index("def _pf_affinity_freq_paths(self):")
    idx_end = _SRC.index("\n    def _pf_sample_cpu_freq", idx)
    snippet = _SRC[idx:idx_end]
    assert "except (AttributeError, OSError):" in snippet
    assert "return self._pf_freq_paths" in snippet
    assert "paths if paths else self._pf_freq_paths" in snippet


def test_pf_sample_cpu_freq_uses_affinity_scoped_paths():
    idx = _SRC.index("def _pf_sample_cpu_freq(self):")
    idx_end = _SRC.index("\n    def ", idx + 40)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_affinity_freq_paths()" in snippet


def test_pf_init_builds_freq_path_by_core_map():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_freq_path_by_core = {}" in snippet
    assert "self._pf_freq_path_by_core[core_id] = p" in snippet


def test_core_id_parsing_known_value():
    """core_idパース(os.path.basename(os.path.dirname(os.path.dirname(p)))[3:])の
    ミラー検証: /sys/.../cpu7/cpufreq/scaling_cur_freq -> 7。"""
    p = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq"
    core_id = int(os.path.basename(os.path.dirname(os.path.dirname(p)))[3:])
    assert core_id == 7


# ---------------------------------------------------------------------------
# 262節続報(判定基準改訂+work_cpu計装Phase 4): work_cpu計装・cpu_affinity
# ---------------------------------------------------------------------------

def test_work_cpu_captured_via_thread_time():
    idx = _SRC.index("def _control(self):")
    idx_end = _SRC.index("if self._loop % 100 == 0:", idx)
    snippet = _SRC[idx:idx_end]
    assert "_pf_work0 = _time.perf_counter()" in snippet
    assert "_pf_work_cpu0 = _time.thread_time()" in snippet


def test_work_cpu_passed_to_pf_cycle_end():
    idx = _SRC.index("self._pf_cycle_end(_time.perf_counter() - _pf_work0,")
    idx_end = _SRC.index("\n\n    def run", idx)
    snippet = _SRC[idx:idx_end]
    assert "_time.thread_time() - _pf_work_cpu0" in snippet


def test_pf_cycle_end_tracks_work_cpu_window_stats():
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_work_cpu_sum += work_cpu" in snippet
    assert "self._pf_work_cpu_max" in snippet
    assert "work_cpu avg=" in snippet or "'work_cpu avg=" in _SRC


def test_perf_spike_log_line_contains_work_cpu_field():
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "work_cpu=%.2fms" in snippet


def mirror_work_cpu_stats(work, work_cpu):
    """wall≫cpuならスケジューラ起因、wall≈cpuなら真の計算量増、という帰属判定の
    ミラー(閾値は解釈上の目安であり本体には実装しない——テストは差分算出のみ検証)。"""
    return work - work_cpu


def test_work_cpu_gap_positive_when_scheduler_delay_dominates():
    assert mirror_work_cpu_stats(work=0.032, work_cpu=0.006) > 0.02


def test_work_cpu_gap_near_zero_when_compute_dominates():
    assert abs(mirror_work_cpu_stats(work=0.032, work_cpu=0.030)) < 0.005


def test_cpu_affinity_default_empty_and_exempt_from_rate_scaling():
    idx = _SRC.index('getattr(self._cfg.mpc, "cpu_affinity"')
    line_end = _SRC.index("\n", idx)
    line = _SRC[max(0, idx - 10):line_end]
    assert ', [])' in line or ',[])' in line


def test_cpu_affinity_noop_when_empty():
    idx = _SRC.index('cpu_affinity = getattr(self._cfg.mpc, "cpu_affinity"')
    idx_end = _SRC.index("\n    def _pf_log_platform_checklist", idx)
    snippet = _SRC[idx:idx_end]
    assert "if cpu_affinity:" in snippet


def test_sched_setaffinity_wrapped_in_try_except():
    idx = _SRC.index('cpu_affinity = getattr(self._cfg.mpc, "cpu_affinity"')
    idx_end = _SRC.index("\n    def _pf_log_platform_checklist", idx)
    snippet = _SRC[idx:idx_end]
    assert "os.sched_setaffinity(0, set(" in snippet
    assert "except (AttributeError, OSError, ValueError)" in snippet


def test_platform_checklist_logs_effective_affinity():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    assert "os.sched_getaffinity(0)" in snippet
    assert "cpu_affinity={effective_affinity}" in snippet


def test_cpu_affinity_registered_in_config_default():
    """config.yamlのデフォルトが空リスト(=無効、現状と同一挙動)であることの確認。"""
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_src = f.read()
    assert "cpu_affinity: []" in cfg_src
