"""261節続報(2026-08-01、72Hzスパイク調査Phase 3): work>予算×factorの周期を
1行でダンプする[PERF-SPIKE]計装。

背景: 72Hz実測(単一車両・約9.7分)が不合格になった原因は、平均処理時間
(work avg=9.17ms、予算13.9msの約66%で健全)ではなく、work maxのスパイク
(平均45.41ms・最悪65.30ms、予算の3〜4.7倍)だった。既存の[PERF]は窓ごとの
maxしか出さないため、「どの区間が同一周期で同時に膨らんだか」が判別できない。
容疑仮説は4つ: (a)100周期ごとのreference_path再構築バースト、(b)AWSIM同居に
よるCPU競合、(c)Python GC世代2回収、(d)OSQPフルセットアップ再実行。
[PERF-SPIKE]は、スパイク発生周期そのものについて、区間別実測値・
cache_builds差分(a)・nivcsw差分(b)・GC世代2フラグ(c)・コリドーセグメント数
変化(d)を1行にまとめてダンプすることで、これらの仮説を切り分ける材料を残す。

mpc_controller.pyはrclpy依存(かつ本環境では未インストールのメッセージ型に
依存)のため直接importできない。ロジック自体は純粋な算術+文字列整形のため
ミラー関数で検証し、mpc_controller.py側の実装(呼び出し配線・差分の毎サイクル
更新・オーバーヘッド規律)はソーステキスト構造検証で確認する(既存の
test_rate_scaling_254.py/test_perf_dt_instrumentation_255.pyと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def mirror_spike_diffs(prev_cache_builds, cache_builds, prev_nivcsw, nivcsw,
                        prev_nseg, nseg):
    """_pf_dump_spike_if_needed()の差分計算部分のミラー(発火有無に関わらず
    毎サイクル計算される部分)。"""
    cache_builds_diff = cache_builds - prev_cache_builds
    nivcsw_diff = nivcsw - prev_nivcsw
    nseg_changed = (prev_nseg is not None and nseg != prev_nseg)
    return cache_builds_diff, nivcsw_diff, nseg_changed


def mirror_spike_fires(work, budget_s, factor):
    """発火条件のミラー。"""
    return work > budget_s * factor


# ---------------------------------------------------------------------------
# ①非矛盾性: 差分計算・発火条件そのものの正しさ
# ---------------------------------------------------------------------------

def test_cache_builds_diff_known_value():
    diff, _, _ = mirror_spike_diffs(10, 16, 0, 0, None, (1, 2, 3))
    assert diff == 6


def test_nivcsw_diff_known_value():
    _, diff, _ = mirror_spike_diffs(0, 0, 1000, 1042, None, (1, 2, 3))
    assert diff == 42


def test_nseg_changed_true_when_tuple_differs():
    _, _, changed = mirror_spike_diffs(0, 0, 0, 0, (1, 2, 3), (1, 2, 4))
    assert changed is True


def test_nseg_changed_false_when_tuple_identical():
    _, _, changed = mirror_spike_diffs(0, 0, 0, 0, (1, 2, 3), (1, 2, 3))
    assert changed is False


def test_nseg_changed_false_on_first_cycle_when_prev_is_none():
    """初回サイクル(prev_nsegがまだ無い)ではnseg_changedをFalseとする
    (「変化した」と誤検出しない、比較対象が無いだけの状態)。"""
    _, _, changed = mirror_spike_diffs(0, 0, 0, 0, None, (1, 2, 3))
    assert changed is False


def test_spike_fires_when_work_exceeds_budget_times_factor():
    assert mirror_spike_fires(work=0.030, budget_s=0.0139, factor=2.0) is True


def test_spike_does_not_fire_at_healthy_work():
    assert mirror_spike_fires(work=0.010, budget_s=0.0139, factor=2.0) is False


def test_spike_fires_at_72hz_budget_with_default_factor():
    """72Hz(予算13.9ms)・既定factor=2.0での実際の発火閾値(27.8ms)を固定する。"""
    budget_s = 1.0 / 72.0
    threshold_ms = budget_s * 2.0 * 1000
    assert round(threshold_ms, 1) == 27.8
    assert mirror_spike_fires(work=0.030, budget_s=budget_s, factor=2.0) is True
    assert mirror_spike_fires(work=0.020, budget_s=budget_s, factor=2.0) is False


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_pf_spike_dump_factor_computed_once_from_config_with_default_2():
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_gc_cb", idx)
    snippet = _SRC[idx:idx_end]
    assert 'getattr(self._cfg.mpc, "perf_spike_dump_factor", 2.0)' in snippet
    assert "self._pf_spike_dump_factor" in snippet


def test_pf_cycle_end_calls_dump_spike_if_needed():
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    def _pf_dump_spike_if_needed", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_dump_spike_if_needed(work, work_cpu)" in snippet


def test_diff_trackers_update_unconditionally_not_only_on_spike():
    """核心: cache_builds/nivcsw/nsegの差分追跡(および前回値の更新)は、
    work>予算×factorのif文より前(=毎サイクル無条件)に行われることを確認する。
    そうでないと、あるサイクルがスパイクしなかった場合に次のスパイク検出時の
    差分が「直前サイクルからの差分」ではなく「前回スパイクからの差分」に
    ずれてしまう。"""
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    # 2026-07-31追加(255節続報", idx)
    snippet = _SRC[idx:idx_end]

    idx_if = snippet.index("if work > self._pf_over_budget_s * self._pf_spike_dump_factor:")
    before_if = snippet[:idx_if]

    # 前回値の更新(=次サイクルの差分計算の基準点)がif文より前で行われている。
    assert "self._pf_prev_cache_builds = cache_builds" in before_if
    assert "self._pf_prev_nivcsw = nivcsw" in before_if
    assert "self._pf_prev_nseg = nseg" in before_if


def test_string_formatting_and_print_only_happen_inside_the_if_block():
    """性能規律: parts整形・print呼び出しはif文の内側(発火時のみ)に限定され、
    差分計算そのもの(前段、毎サイクル実行)には文字列整形コストが混ざって
    いないことを確認する。"""
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    # 2026-07-31追加(255節続報", idx)
    snippet = _SRC[idx:idx_end]

    idx_if = snippet.index("if work > self._pf_over_budget_s * self._pf_spike_dump_factor:")
    before_if = snippet[:idx_if]
    after_if = snippet[idx_if:]

    assert "parts = " not in before_if
    assert "print(" not in before_if
    assert "parts = " in after_if
    assert "print(" in after_if


def test_pf_add_records_into_last_cycle_dict():
    idx = _SRC.index("def _pf_add(self, name, dt):")
    idx_end = _SRC.index("\n    def _pf_mark", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_last_cycle[name] = dt" in snippet


def test_control_resets_last_cycle_state_at_cycle_start():
    idx = _SRC.index("def _control(self):")
    idx_end = _SRC.index("now = self.get_clock().now()", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_last_cycle = {}" in snippet
    assert "self._pf_cycle_gen2_flag = False" in snippet
    assert "self._pf_cycle_gen2_duration = 0.0" in snippet


def test_gc_callback_tracks_generation_2_specifically():
    idx = _SRC.index("def _pf_gc_cb(self, phase, info):")
    idx_end = _SRC.index("\n    def _pf_add", idx)
    snippet = _SRC[idx:idx_end]
    assert "info.get('generation')" in snippet
    assert "== 2" in snippet
    assert "self._pf_cycle_gen2_flag = True" in snippet
    assert "self._pf_cycle_gen2_duration += dt" in snippet


def test_perf_spike_log_line_contains_all_required_fields():
    idx = _SRC.index("'[PERF-SPIKE]")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("loop=", "loop_mod100=", "work=", "budget=", "cache_builds_diff=",
                  "nivcsw_diff=", "gen2_gc=", "gen2_gc_dur=", "nseg=", "nseg_changed="):
        assert field in snippet, f"missing {field!r} in [PERF-SPIKE] log line"


def test_perf_spike_uses_reference_path_via_getattr_defensively():
    """self._reference_pathが(想定外に)未設定でもAttributeErrorで
    ノードごと落ちないよう、getattrで安全に参照していることを確認する
    (計装自体が新たな不具合の原因にならないための防御)。"""
    idx = _SRC.index("def _pf_dump_spike_if_needed(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    # 2026-07-31追加(255節続報", idx)
    snippet = _SRC[idx:idx_end]
    assert "getattr(self, '_reference_path', None)" in snippet


def test_existing_perf_and_perfdt_tags_unchanged():
    """既存の[PERF]/[PERF-DT]の出力・意味は変更していない(追加のみ)ことの
    回帰確認。[PERF-CORRIDOR]はreference_path.py側(258/261節、別ファイル)の
    ため、ここではmpc_controller.py内の2タグのみ確認する。"""
    assert "'[PERF] n=%d" in _SRC
    assert "'[PERF-DT] n=%d" in _SRC
