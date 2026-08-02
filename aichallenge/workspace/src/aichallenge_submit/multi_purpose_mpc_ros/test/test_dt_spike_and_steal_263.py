"""263節続報(2026-08-02、蛇行/性能ギャップ分析Part A-1): [PERF-DT-SPIKE]計装
(dtそのものが予算×factorを超えた周期のダンプ)+steal時間(ハイパーバイザ横取り)
計装の単体テスト。

背景: 予選ログ(2026-08-02)で約1秒間、全ノード(mpc_controller・imu_gnss_poser等)
が同時に完全停止するストールを発見した。[PERF-DT]のp999/max=764.37msとして
現れたが、対応する[PERF-SPIKE](work基準)は全件work最大62msどまりで、この
ストールに対応するダンプが存在しなかった——work計装は「処理時間」しか見て
おらず、「呼び出し間隔(dt)そのものが伸びる」現象(プロセスがスケジューラに
実行権を奪われていた)を原理的に捕捉できないという盲点があった。本計装は
この盲点を埋める。あわせて、AWS等クラウド環境特有の「ハイパーバイザに
CPUを横取りされた時間」(/proc/statのsteal)を直接観測し、764ms級ストールの
原因候補(noisy neighbor/VM steal time)を検証可能にする。

mpc_controller.pyはrclpy依存のため直接importできない。ロジック自体は純粋な
算術のためミラー関数で検証し、mpc_controller.py側の実装(呼び出し配線・
発火条件・N/A分岐)はソーステキスト構造検証で確認する(既存の
test_run_delay_instrumentation_262.py等と同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: 発火条件・steal差分算出のミラー検証
# ---------------------------------------------------------------------------

def mirror_dtspike_fires(dt, budget_s, factor):
    return dt > budget_s * factor


def mirror_steal_diff_ms(prev_jiffies, cur_jiffies, clk_tck):
    diff = max(0, cur_jiffies - prev_jiffies)
    return diff * 1000.0 / clk_tck


def test_dtspike_fires_above_threshold():
    assert mirror_dtspike_fires(dt=0.130, budget_s=0.025, factor=5.0) is True


def test_dtspike_does_not_fire_at_threshold_boundary():
    # 764ms級を確実に捕捉しつつ通常ジッタで誤発火しない、という設計意図の
    # 境界確認(0.025*5.0=0.125ちょうどは非発火)。
    assert mirror_dtspike_fires(dt=0.125, budget_s=0.025, factor=5.0) is False


def test_dtspike_does_not_fire_for_normal_jitter():
    assert mirror_dtspike_fires(dt=0.030, budget_s=0.025, factor=5.0) is False


def test_dtspike_fires_for_observed_764ms_stall():
    """実際に発見した764msストール(25ms予算、factor既定5.0)は確実に発火する。"""
    assert mirror_dtspike_fires(dt=0.764, budget_s=0.025, factor=5.0) is True


def test_steal_diff_ms_known_value():
    # 100Hz(USER_HZ標準)で50 jiffies差分 = 500ms。
    assert mirror_steal_diff_ms(1000, 1050, clk_tck=100) == 500.0


def test_steal_diff_ms_clamped_to_zero_when_decreasing():
    assert mirror_steal_diff_ms(2000, 1500, clk_tck=100) == 0.0


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_pf_read_cpu_steal_jiffies_reads_proc_stat_field_8():
    idx = _SRC.index("def _pf_read_cpu_steal_jiffies(self):")
    idx_end = _SRC.index("\n    def _pf_log_colocated_affinity", idx)
    snippet = _SRC[idx:idx_end]
    assert "/proc/stat" in snippet
    assert "fields[8]" in snippet
    assert "except (OSError, IndexError, ValueError):" in snippet
    assert "return None" in snippet


def test_pf_init_checks_steal_availability_before_platform_checklist():
    idx = _SRC.index("def _pf_init(self):")
    idx_checklist = _SRC.index("self._pf_log_platform_checklist()", idx)
    snippet = _SRC[idx:idx_checklist]
    assert "self._pf_steal_available = self._pf_read_cpu_steal_jiffies() is not None" in snippet


def test_platform_availability_line_includes_steal_time():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_item_root", idx)
    snippet = _SRC[idx:idx_end]
    assert "steal_time=" in snippet
    assert "self._pf_steal_available" in snippet


def test_pf_cycle_end_tracks_last_work_for_dtspike():
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_last_work_ms = work * 1000.0" in snippet
    assert "self._pf_last_work_cpu_ms = work_cpu * 1000.0" in snippet


def test_dtperf_init_sets_up_dtspike_state():
    idx = _SRC.index("def _dtperf_init(self):")
    idx_end = _SRC.index("\n    def _dtperf_reset_window", idx)
    snippet = _SRC[idx:idx_end]
    assert 'getattr(self._cfg.mpc, "perf_dt_spike_factor", 5.0)' in snippet
    assert "self._dtperf_dtspike_count = 0" in snippet
    assert "self._dtperf_prev_nivcsw = _resource.getrusage" in snippet
    assert "self._dtperf_prev_steal_jiffies = (" in snippet
    assert "os.sysconf('SC_CLK_TCK')" in snippet


def test_dtperf_reset_window_resets_dtspike_count():
    idx = _SRC.index("def _dtperf_reset_window(self):")
    idx_end = _SRC.index("\n    def _dtperf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._dtperf_dtspike_count = 0" in snippet


def test_dtperf_record_calls_dump_spike_every_cycle_not_only_at_window_boundary():
    """既存[PERF-SPIKE]と同じ理由: 窓境界だけで判定するとストール自体を
    取りこぼすため、毎サイクル(窓境界の判定より前)で呼ぶこと。"""
    idx = _SRC.index("def _dtperf_record(self, dt):")
    idx_end = _SRC.index("\n    def _g2_release_ready", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._dtperf_dump_spike_if_needed(dt)" in snippet
    assert snippet.index("self._dtperf_dump_spike_if_needed(dt)") < snippet.index(
        "if self._pf_report_every and self._dtperf_cycles >= self._pf_report_every:")


def test_dtperf_dump_spike_early_return_when_below_threshold():
    idx = _SRC.index("def _dtperf_dump_spike_if_needed(self, dt):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    assert "if dt <= self._pf_over_budget_s * self._dtperf_spike_factor:" in snippet
    assert "return" in snippet


def test_dtperf_dump_spike_updates_nivcsw_and_steal_unconditionally():
    """発火条件チェックより前に差分追跡(nivcsw・steal)を行うこと
    (早期returnしても次回の差分が正しく取れるように)。"""
    idx = _SRC.index("def _dtperf_dump_spike_if_needed(self, dt):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    nivcsw_idx = snippet.index("self._dtperf_prev_nivcsw = nivcsw")
    return_idx = snippet.index("if dt <= self._pf_over_budget_s * self._dtperf_spike_factor:")
    assert nivcsw_idx < return_idx


def test_dtperf_dump_spike_log_line_contains_required_fields():
    idx = _SRC.index("def _dtperf_dump_spike_if_needed(self, dt):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("loop=", "dt=", "budget=", "factor=", "threshold=",
                  "prev_work=", "prev_work_cpu=", "run_delay_prev=",
                  "nivcsw_diff=", "steal_diff="):
        assert field in snippet, f"missing {field!r} in [PERF-DT-SPIKE] log line"
    assert "'[PERF-DT-SPIKE] loop=%d" in snippet


def test_dtperf_dump_spike_reports_loop_plus_one():
    """_dtperf_recordはself._loop += 1より前に呼ばれるため、work基準の
    [PERF-SPIKE](インクリメント後のself._loopを使う)と同一サイクルで
    loop番号を突き合わせられるよう、+1して報告すること。"""
    idx = _SRC.index("def _dtperf_dump_spike_if_needed(self, dt):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    assert "% (self._loop + 1, dt * 1000" in snippet


def test_dtperf_dump_spike_na_fallbacks_present():
    idx = _SRC.index("def _dtperf_dump_spike_if_needed(self, dt):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    assert "'N/A'" in snippet
    # prev_work/prev_work_cpu/run_delay_prev/stealそれぞれにN/Aフォールバックがある。
    assert snippet.count("'N/A'") >= 4


def test_perf_dt_window_line_includes_dtspike_count():
    idx = _SRC.index("print('[PERF-DT] n=%d")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    assert "dtspike=" in snippet
    assert "self._dtperf_dtspike_count" in snippet


def test_config_default_registered():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_src = f.read()
    assert "perf_dt_spike_factor: 5.0" in cfg_src


def test_existing_perf_dt_tags_and_fields_unchanged():
    """既存の[PERF-DT]タグ自体・既存フィールド(p50/p95/p99/p999/max/eff_rate/
    over_budget/max_consec_over/margin)は維持されていることの回帰確認
    (dtspikeフィールドは追加のみ)。"""
    idx = _SRC.index("print('[PERF-DT] n=%d")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("p50=", "p95=", "p99=", "p999=", "max=", "eff_rate=",
                  "over_budget=", "max_consec_over=", "margin="):
        assert field in snippet, f"missing {field!r} (regression)"
