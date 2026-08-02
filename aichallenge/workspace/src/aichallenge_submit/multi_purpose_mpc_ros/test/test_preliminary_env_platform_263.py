"""263節(2026-08-02、予選環境ギャップ分析の準備Phase 1・2): [PERF-DT]の
クロックソース確認+[PERF-PLATFORM]の予選環境向け補強。

背景: ローカル側の特性化(J=17.5〜20ms構成依存、work_cpu avg≈6ms、スケジューラ
競合の帰属手法)は確立済みだが、予選環境側は「Autowareに約3vCPU/12GiB」という
2026-07-24時点の推定値のみ判明している。次の予選環境走行(control_rate=40.0の
まま)で1本ログを回収すれば、J_予選・実リソース制限・計算余裕・競合実態が
一度に埋まる——本作業はそのための計装補強である。

Phase 1: [PERF-DT]の計測入力が既にperf_counterベースの独立dt
(self._dtperf_record(_wall_dt)、mpc_controller.py:4924-4929)へ分離済み
であることをソーステキストで確認する(制御ロジックが使う既存dt(ROSクロック
由来)には一切触れていないことも併せて確認)。あわせて[PERF-PLATFORM]起動時
ログにuse_sim_timeの実効値を追加する(回収ログの解釈——sim time/非等速再生の
可能性——に必須)。

Phase 2: [PERF-PLATFORM]へcgroup v2/v1のCPUクォータ・cpuset・メモリ上限、
CPUモデル名・論理コア数、可用性マップ(項目別OK/N-A)を追加する。「約3vCPU/
12GiB」推定値を実測で確定させるための計装であり、制御には一切影響しない。

mpc_controller.pyはrclpy依存のため直接importできない。ロジック自体は純粋な
文字列/算術処理のためミラー関数で検証し、mpc_controller.py側の実装(呼び出し
配線・N/A分岐・失敗時の安全側フォールバック)はソーステキスト構造検証で確認
する(既存のtest_cpu_freq_instrumentation_262.py等と同じ方針)。

cgroup v2ルート解決(_pf_find_cgroup_v2_root)は実際の開発ホストで実地検証
済み: 葉cgroup(/proc/self/cgroupが示すパス)にcpu.maxが無く、祖先の
systemd user.sliceレベルに初めて見つかるケースを確認した(cgroup v2は
cgroup.subtree_controlで委譲された階層にしかコントローラファイルを置かない
仕様のため)。ミラー関数のテストはこの実地知見を反映した合成ケースを含む。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: cgroup解析・v2ルート探索の純粋ロジック検証
# ---------------------------------------------------------------------------

def mirror_cpu_quota_str(quota_str, period_str):
    """cpu.max(v2)/cfs_quota_us・cfs_period_us(v1、周期は既にus単位)の
    quota/period文字列からcpu_quota_strを算出する部分のミラー。"""
    if quota_str == 'max':
        return 'unlimited'
    return '%.2f' % (int(quota_str) / int(period_str))


def mirror_memory_max_str_v2(mem_str):
    if mem_str == 'max':
        return 'unlimited'
    return '%.2fGiB' % (int(mem_str) / (1024 ** 3))


def mirror_memory_max_str_v1(mem_bytes):
    if mem_bytes >= (1 << 62):
        return 'unlimited'
    return '%.2fGiB' % (mem_bytes / (1024 ** 3))


def mirror_find_v2_root(cgroup_line, existing_cpu_max_dirs):
    """_pf_find_cgroup_v2_root()の候補生成+探索部分のミラー。ファイル
    システムへは触れず、existing_cpu_max_dirs(cpu.maxが存在すると仮定する
    ディレクトリの集合)との照合だけで判定する。"""
    candidates = []
    parts = cgroup_line.strip().split(':', 2)
    if len(parts) == 3 and parts[0] == '0' and parts[2]:
        p = '/sys/fs/cgroup' + parts[2]
        while p.startswith('/sys/fs/cgroup'):
            candidates.append(p)
            if p == '/sys/fs/cgroup':
                break
            p = os.path.dirname(p)
    if '/sys/fs/cgroup' not in candidates:
        candidates.append('/sys/fs/cgroup')
    for cand in candidates:
        if cand in existing_cpu_max_dirs:
            return cand
    return None


def test_cpu_quota_str_capped_known_value():
    assert mirror_cpu_quota_str('300000', '100000') == '3.00'


def test_cpu_quota_str_unlimited():
    assert mirror_cpu_quota_str('max', '100000') == 'unlimited'


def test_memory_max_str_v2_capped():
    assert mirror_memory_max_str_v2(str(12 * 1024 ** 3)) == '12.00GiB'


def test_memory_max_str_v2_unlimited():
    assert mirror_memory_max_str_v2('max') == 'unlimited'


def test_memory_max_str_v1_capped():
    assert mirror_memory_max_str_v1(12 * 1024 ** 3) == '12.00GiB'


def test_memory_max_str_v1_unlimited_sentinel():
    """v1の「無制限」は極端に大きな値(例: 9223372036854771712)で表現される。"""
    assert mirror_memory_max_str_v1(9223372036854771712) == 'unlimited'


def test_find_v2_root_at_leaf():
    cgroup_line = "0::/docker/abc123"
    root = mirror_find_v2_root(cgroup_line, {'/sys/fs/cgroup/docker/abc123'})
    assert root == '/sys/fs/cgroup/docker/abc123'


def test_find_v2_root_falls_back_to_ancestor():
    """実地検証(開発ホスト)で確認したケース: 葉にはcpu.maxが無く、祖先の
    systemd user.sliceレベルに初めて見つかる。"""
    cgroup_line = (
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "app-org.gnome.Terminal.slice/vte-spawn-xxx.scope")
    root = mirror_find_v2_root(cgroup_line, {'/sys/fs/cgroup/user.slice'})
    assert root == '/sys/fs/cgroup/user.slice'


def test_find_v2_root_falls_back_to_bare_root():
    cgroup_line = "0::/some/deep/path"
    root = mirror_find_v2_root(cgroup_line, {'/sys/fs/cgroup'})
    assert root == '/sys/fs/cgroup'


def test_find_v2_root_none_when_nowhere_found():
    cgroup_line = "0::/some/deep/path"
    root = mirror_find_v2_root(cgroup_line, set())
    assert root is None


def test_find_v2_root_malformed_line_falls_back_to_bare_root():
    """/proc/self/cgroupの形式が想定外(':'区切りが2要素以下等)でも
    クラッシュせず、素の/sys/fs/cgroupへフォールバックする。"""
    root = mirror_find_v2_root("garbage", {'/sys/fs/cgroup'})
    assert root == '/sys/fs/cgroup'


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_dtperf_uses_independent_perf_counter_clock():
    """[PERF-DT]計測入力(self._dtperf_record)がROSクロックではなく
    time.perf_counter()由来の独立dtを受け取ること(既存の分離実装の回帰確認、
    263節で新規導入したものではない)。"""
    idx = _SRC.index("def _control(self):")
    idx_end = _SRC.index("_pf_work0 = _time.perf_counter()", idx)
    snippet = _SRC[idx:idx_end]
    assert "_now_wall = _time.perf_counter()" in snippet
    assert "self._dtperf_record(_wall_dt)" in snippet


def test_control_logic_dt_untouched_by_perf_counter_separation():
    """制御ロジックが使う既存dt(self.get_clock()由来)には[PERF-DT]専用の
    perf_counter分離が一切影響していないことの回帰確認。"""
    idx = _SRC.index("def _control(self):")
    idx_end = _SRC.index("self._dtperf_record(_wall_dt)", idx)
    snippet = _SRC[idx:idx_end]
    assert "dt = (now - self._last_t).nanoseconds / 1e9" in snippet


def test_platform_checklist_logs_use_sim_time():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_root", idx)
    snippet = _SRC[idx:idx_end]
    assert "use_sim_time={self.use_sim_time}" in snippet


def test_platform_checklist_logs_cgroup_and_cpu_model_fields():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_root", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("cgroup=", "cpu_quota_cores=", "cpuset_cpus=", "memory_max=",
                  "cpu_model=", "cpu_count="):
        assert field in snippet, f"missing {field!r} in [PERF-PLATFORM] log line"


def test_platform_checklist_logs_availability_summary():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_root", idx)
    snippet = _SRC[idx:idx_end]
    assert "[PERF-PLATFORM] availability:" in snippet
    for field in ("scaling_cur_freq=", "sched_schedstats=", "cgroup_cpu_quota=",
                  "cgroup_cpuset=", "cgroup_memory_max="):
        assert field in snippet, f"missing {field!r} in availability summary"


def test_platform_checklist_calls_cgroup_and_cpu_model_readers():
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_root", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_read_cgroup_limits()" in snippet
    assert "self._pf_read_cpu_model()" in snippet


def test_pf_find_cgroup_v2_root_walks_ancestors():
    idx = _SRC.index("def _pf_find_cgroup_v2_root(self):")
    idx_end = _SRC.index("\n    def _pf_read_cgroup_limits", idx)
    snippet = _SRC[idx:idx_end]
    assert "os.path.dirname(p)" in snippet
    assert "os.path.exists(os.path.join(cand, 'cpu.max'))" in snippet


def test_pf_find_cgroup_v2_root_handles_missing_proc_self_cgroup():
    idx = _SRC.index("def _pf_find_cgroup_v2_root(self):")
    idx_end = _SRC.index("\n    def _pf_read_cgroup_limits", idx)
    snippet = _SRC[idx:idx_end]
    assert "except OSError:" in snippet


def test_pf_read_cgroup_limits_v2_branch_reads_three_files():
    idx = _SRC.index("def _pf_read_cgroup_limits(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_model", idx)
    snippet = _SRC[idx:idx_end]
    assert "'cpu.max'" in snippet
    assert "cpuset.cpus.effective" in snippet
    assert "'memory.max'" in snippet
    assert "result['version'] = 'v2'" in snippet


def test_pf_read_cgroup_limits_v1_fallback_only_when_v2_root_none():
    idx = _SRC.index("def _pf_read_cgroup_limits(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_model", idx)
    snippet = _SRC[idx:idx_end]
    assert "v2_root = self._pf_find_cgroup_v2_root()" in snippet
    assert "if v2_root is not None:" in snippet
    assert "cpu.cfs_quota_us" in snippet
    assert "result['version'] = 'v1'" in snippet


def test_pf_read_cgroup_limits_distinguishes_unlimited_from_na():
    """「unlimited」(読めたが上限未設定)と「N/A」(読めない)を区別する
    仕様であることの確認。"""
    idx = _SRC.index("def _pf_read_cgroup_limits(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_model", idx)
    snippet = _SRC[idx:idx_end]
    assert "'unlimited'" in snippet
    assert "'N/A'" in snippet


def test_pf_read_cgroup_limits_does_not_crash_on_missing_files():
    idx = _SRC.index("def _pf_read_cgroup_limits(self):")
    idx_end = _SRC.index("\n    def _pf_read_cpu_model", idx)
    snippet = _SRC[idx:idx_end]
    assert snippet.count("except (OSError, ValueError):") >= 4
    assert snippet.count("except OSError:") >= 1


def test_pf_read_cpu_model_reads_proc_cpuinfo():
    idx = _SRC.index("def _pf_read_cpu_model(self):")
    idx_end = _SRC.index("\n    def _pf_log_colocated_affinity", idx)
    snippet = _SRC[idx:idx_end]
    assert "/proc/cpuinfo" in snippet
    assert "model name" in snippet
    assert "return 'N/A'" in snippet


def test_platform_checklist_calls_itself_at_pf_init_unchanged():
    """既存配線(_pf_init内で_pf_log_platform_checklist()を1回だけ呼ぶ)の
    回帰確認。"""
    idx = _SRC.index("def _pf_init(self):")
    idx_end = _SRC.index("\n    def _pf_log_platform_checklist", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_log_platform_checklist()" in snippet


def test_existing_perf_platform_governor_fields_unchanged():
    """263節の追加(cgroup・cpu_model・use_sim_time・可用性マップ)が既存の
    governor/scaling_max_freq/rapl_power_limit/cores_sampled/cpu_affinity
    フィールドを壊していないことの回帰確認。"""
    idx = _SRC.index("def _pf_log_platform_checklist(self):")
    idx_end = _SRC.index("\n    def _pf_find_cgroup_v2_root", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("governor=", "scaling_max_freq=", "rapl_power_limit=",
                  "cores_sampled=", "cpu_affinity="):
        assert field in snippet, f"missing {field!r} (regression)"
