"""Unit tests for the process-rusage instrumentation (179節続報, 2026-07-25).

背景: dev3ローカル実測ではdocker statsで「autowareコンテナがcpus=3上限(300%)に
張り付いている」ことを外部から直接確認できたが、予選環境ではdocker host側の計測
手段が無い。同じ診断能力をプロセス内部から得るため、resource.getrusage(RUSAGE_SELF)
の実CPU時間(ru_utime+ru_stime)と不随意コンテキストスイッチ回数(ru_nivcsw、CPUを
横取りされた=競合で追い出された回数の直接証拠)を、既存の[PERF]と同じ~10秒窓で
[PERF-RUSAGE]として報告する。cpu_ratio(実CPU時間/壁時計時間)が1を大きく下回れば
「計算コストではなくスケジューリング待ちで遅れている」ことを示す。

mpc_controller.pyはautoware_auto_control_msgs等をモジュールスコープでimportしており
単体テスト環境では直接importできないため、他の巨大メソッド関連テストと同じく実物の
ソーステキストに対する構造的検証を行う。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_resource_module_imported():
    assert "import resource as _resource" in _SRC


def test_rusage_baseline_captured_in_pf_init():
    idx = _SRC.index("    def _pf_init(self):")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_rusage_prev = _resource.getrusage(_resource.RUSAGE_SELF)" in snippet


def test_pf_cycle_end_reports_rusage_delta_alongside_perf():
    idx_perf_print = _SRC.index("print('[PERF] n=%d work avg=")
    idx_rusage_print = _SRC.index("print('[PERF-RUSAGE] n=%d cpu_time=", idx_perf_print)
    idx_reset = _SRC.index("self._pf_acc = {}", idx_rusage_print)
    # [PERF-RUSAGE]は既存[PERF]ログの直後・窓リセットの前に出力されること
    assert idx_perf_print < idx_rusage_print < idx_reset


def test_rusage_delta_uses_both_utime_and_stime():
    idx = _SRC.index("_cpu_time = ((_ru.ru_utime + _ru.ru_stime)")
    idx_end = idx + 200
    snippet = _SRC[idx:idx_end]
    assert "_prev.ru_utime + _prev.ru_stime" in snippet


def test_involuntary_context_switches_tracked_as_the_decisive_signal():
    """④遡及効果: docker statsで確認したCPU競合(cpus=3上限への張り付き)と同じ現象を
    予選環境でも直接検証できるよう、不随意コンテキストスイッチ(ru_nivcsw)を追跡する。
    これはCPUを横取りされた回数そのものであり、cpu_ratioより曖昧さの無い決定的な指標。"""
    assert "_nivcsw = _ru.ru_nivcsw - _prev.ru_nivcsw" in _SRC
    idx = _SRC.index("print('[PERF-RUSAGE]")
    idx_end = idx + 400
    snippet = _SRC[idx:idx_end]
    assert "nivcsw=%d" in snippet


def test_rusage_baseline_reset_after_each_report():
    """①非矛盾性: 次の窓の差分が前の窓の値を含まないよう、報告直後に基準値を
    更新することを確認する(既存の_pf_acc等のリセットと同じパターン)。"""
    idx = _SRC.index("print('[PERF-RUSAGE] n=%d cpu_time=")
    idx_end = _SRC.index("self._pf_acc = {}", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_rusage_prev = _ru" in snippet
