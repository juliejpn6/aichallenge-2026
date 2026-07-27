"""Unit tests for the setup/solve split instrumentation (178節続報, 2026-07-25).

背景: 「mpc」区間(平均14-20ms、処理落ちの最大要因)が障害物数(n_dynobs_max)と
無相関(r=-0.07)と実測で判明したため、Python側の行列組み立て(_init_problem、
以下"setup")とOSQPソルバー本体(.solve()、以下"solve")のどちらが支配的かを
切り分ける計装を追加した。MPC.get_control()側でリトライ(最大2回の再solve)分も
含めて累積計測し、mpc_controller.py側は[PERF]へ転記するだけ。

MPC.pyはautoware依存が薄く直接importできるため、実物のクラスに対して構造的検証
(ソース文字列)を行う。数値的な計測結果自体は次回の実走行ログで確認する。
"""
import os

_MPC_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "MPC.py")
_CTRL_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_MPC_SRC_PATH) as _f:
    _MPC_SRC = _f.read()
with open(_CTRL_SRC_PATH) as _f:
    _CTRL_SRC = _f.read()


def test_time_module_imported():
    assert "import time as _time" in _MPC_SRC


def test_timing_state_declared_in_init():
    idx = _MPC_SRC.index("    def __init__(")
    idx_end = _MPC_SRC.index("\n    def ", idx + 10)
    snippet = _MPC_SRC[idx:idx_end]
    assert "self.last_setup_time = 0.0" in snippet
    assert "self.last_solve_time = 0.0" in snippet
    assert "self.last_retry_count = 0" in snippet


def test_get_control_resets_accumulators_before_first_solve():
    idx = _MPC_SRC.index("    def get_control(self)")
    idx_first_solve = _MPC_SRC.index("dec = self._active.solve()", idx)
    snippet = _MPC_SRC[idx:idx_first_solve]
    assert "self.last_setup_time = 0.0" in snippet
    assert "self.last_solve_time = 0.0" in snippet
    assert "self.last_retry_count = 0" in snippet


def test_first_init_problem_and_solve_are_both_timed():
    idx = _MPC_SRC.index("    def get_control(self)")
    idx_retry_loop = _MPC_SRC.index("for i, relaxed in enumerate", idx)
    snippet = _MPC_SRC[idx:idx_retry_loop]
    assert "self.last_setup_time += _time.perf_counter() - _t0" in snippet
    assert "self.last_solve_time += _time.perf_counter() - _t0" in snippet


def test_retry_loop_accumulates_setup_and_solve_and_counts_retries():
    idx = _MPC_SRC.index("for i, relaxed in enumerate")
    idx_end = _MPC_SRC.index("solved = dec.info.status_val in _OSQP_OK", idx + 10)
    snippet = _MPC_SRC[idx:idx_end]
    assert "self.last_retry_count += 1" in snippet
    assert snippet.count("self.last_setup_time +=") == 1
    assert snippet.count("self.last_solve_time +=") == 1


def test_setup_and_solve_times_are_cumulative_not_overwritten():
    """①非矛盾性: 初回+リトライ2回分を合算する(上書きではなく+=)ことを確認する。
    リトライを含めた「mpc区間の実コスト全体」を漏れなく捉えるため。"""
    assert _MPC_SRC.count("self.last_setup_time += _time.perf_counter()") == 2
    assert _MPC_SRC.count("self.last_solve_time += _time.perf_counter()") == 2


def test_controller_reads_setup_and_solve_into_perf_buckets():
    idx = _CTRL_SRC.index("self._pf_mark('mpc')")
    idx_end = idx + 700
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._pf_add('mpc_setup', getattr(self._mpc, 'last_setup_time', 0.0))" in snippet
    assert "self._pf_add('mpc_solve', getattr(self._mpc, 'last_solve_time', 0.0))" in snippet


def test_retry_count_not_fed_into_pf_add_since_it_is_not_a_duration():
    """②非冗長性: _pf_addは秒単位の時間を前提に×1000でms表示するため、回数
    (last_retry_count)をそのまま渡すと誤解を招くログになる。既存の
    "Relaxed safety margin"ログで追跡する方針を維持し、pf_addには渡さない。"""
    idx = _CTRL_SRC.index("self._pf_mark('mpc')")
    idx_end = idx + 700
    snippet = _CTRL_SRC[idx:idx_end]
    assert "last_retry_count" not in snippet
