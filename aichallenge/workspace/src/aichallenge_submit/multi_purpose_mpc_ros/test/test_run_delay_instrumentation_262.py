"""262節続報(2026-08-02、判定基準改訂+work_cpu計装Part B): run_delay計装。

背景: C4実測(72Hz)でwall(work)がwork_cpuを大きく上回るサイクルが確認され、
work_cpu計装(先行導入)により「計算量増加ではない」ことは示せたが、原因が
本当に「スケジューラのランキュー待ち」であるかは間接証拠(nivcsw・work-work_cpu
の差)に留まっていた。/proc/<pid>/task/<tid>/schedstatの第2フィールド
(run_delay、ランキュー待ち累積ns)を直接計測することで、wall-cpuギャップの
原因をランキュー待ちそのものと数字で結びつける。

カーネルのkernel.sched_schedstats設定が無効(既定)だと全フィールドが常に0に
なり見かけ上「読めてしまう」ため、可用性はsysctl自体の値で判定する
(schedstatファイルの読み取り成功では判定しない)。

mpc_controller.pyはrclpy依存のため直接importできない。ロジック自体は
純粋な算術のためミラー関数で検証し、mpc_controller.py側の実装(呼び出し
配線・可用性判定・N/A分岐)はソーステキスト構造検証で確認する
(既存のtest_cpu_freq_instrumentation_262.pyと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def mirror_run_delay_diff(prev_ns, cur_ns):
    """_pf_sample_run_delay()の差分算出部分のミラー。負値(schedstatが何らかの
    理由で減少した場合)は0にクランプする。"""
    return max(0, cur_ns - prev_ns)


def mirror_window_avg_max(win_sum, win_max, diff_ns):
    win_sum += diff_ns
    win_max = max(win_max, diff_ns)
    return win_sum, win_max


# ---------------------------------------------------------------------------
# ①非矛盾性: 差分・窓集計そのものの正しさ
# ---------------------------------------------------------------------------

def test_run_delay_diff_known_value():
    assert mirror_run_delay_diff(1000, 1500) == 500


def test_run_delay_diff_clamped_to_zero_when_decreasing():
    assert mirror_run_delay_diff(2000, 1500) == 0


def test_window_avg_max_accumulates():
    win_sum, win_max = 0, 0
    for d in (100, 500, 200):
        win_sum, win_max = mirror_window_avg_max(win_sum, win_max, d)
    assert win_sum == 800
    assert win_max == 500


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_availability_determined_by_sysctl_not_read_success():
    """schedstatファイルは無効時でも読み取り自体は成功する(常に0を返す)ため、
    可用性判定はsysctl(/proc/sys/kernel/sched_schedstats)の値で行うこと。"""
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "/proc/sys/kernel/sched_schedstats" in snippet
    assert "self._pf_run_delay_available = f.read().strip() == '1'" in snippet


def test_run_delay_tid_uses_os_gettid_with_fallback():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_freqs_khz", idx)
    snippet = _SRC[idx:idx_end]
    assert "os.gettid()" in snippet
    assert "except AttributeError:" in snippet


def test_read_run_delay_prefers_per_thread_schedstat():
    idx = _SRC.index("def _pf_read_run_delay_ns(self):")
    idx_end = _SRC.index("\n    def _pf_sample_run_delay", idx)
    snippet = _SRC[idx:idx_end]
    assert "/proc/self/task/{self._pf_run_delay_tid}/schedstat" in snippet
    assert "/proc/self/schedstat" in snippet
    assert "fields[1]" in snippet


def test_sample_run_delay_noop_when_unavailable():
    idx = _SRC.index("def _pf_sample_run_delay(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    assert "if not self._pf_run_delay_available:" in snippet
    assert "return" in snippet


def test_pf_cycle_end_samples_run_delay_every_cycle():
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_sample_run_delay()" in snippet
    assert snippet.index("self._pf_sample_run_delay()") < snippet.index(
        "self._pf_dump_spike_if_needed(work, work_cpu)")


def test_perf_spike_log_line_contains_run_delay_field():
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "run_delay_str" in snippet
    assert "run_delay=" in snippet
    assert "run_delay=N/A" in snippet


def test_perf_rusage_log_line_contains_run_delay_window_fields():
    idx = _SRC.index("_cpu_ratio = _cpu_time / _wall_time")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "run_delay_avg=" in snippet
    assert "run_delay_max=" in snippet
    assert "run_delay_avg=N/A run_delay_max=N/A" in snippet


def test_perf_rusage_resets_run_delay_window_after_report():
    idx = _SRC.index("'[PERF-RUSAGE]")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_run_delay_win_sum = 0.0" in snippet
    assert "self._pf_run_delay_win_max = 0.0" in snippet


def test_run_delay_diff_clamped_non_negative_in_source():
    idx = _SRC.index("def _pf_sample_run_delay(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    assert "max(0, cur - self._pf_prev_run_delay_ns)" in snippet


# ---------------------------------------------------------------------------
# dt p99.9 参考記録(判定基準には使わない)
# ---------------------------------------------------------------------------

def test_perf_dt_line_contains_p999_reference_field():
    idx = _SRC.index("'[PERF-DT]")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "p999=" in snippet


def test_perf_dt_existing_fields_unchanged():
    """p999追加が既存フィールド(判定基準に使うp99・over_budget等)を壊していない
    ことの回帰確認。"""
    idx = _SRC.index("'[PERF-DT]")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("p50=", "p95=", "p99=", "max=", "eff_rate=", "over_budget=",
                  "max_consec_over=", "margin="):
        assert field in snippet, f"missing {field!r} in [PERF-DT] log line"
