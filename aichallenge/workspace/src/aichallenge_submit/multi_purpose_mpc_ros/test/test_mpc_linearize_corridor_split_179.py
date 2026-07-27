"""Unit tests for the linearize/corridor split instrumentation (179節続報, 2026-07-25).

背景: [[dev3-perf-root-cause-179]]で「mpc」区間の内訳がsetup(Python側行列組み立て)
支配的(平均15.9ms、solveの1.86倍)と判明した。setup内部はさらに①線形化ループ
(_stage_data内、N回のmodel.linearize()呼び出し)と②コリドー計算(_corridor→
update_path_constraints、占有格子への光線走査)の2ブロックに分かれるため、
どちらが支配的かを切り分ける計装を追加した。

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


def test_timing_state_declared_in_init():
    idx = _MPC_SRC.index("    def __init__(")
    idx_end = _MPC_SRC.index("\n    def ", idx + 10)
    snippet = _MPC_SRC[idx:idx_end]
    assert "self.last_linearize_time = 0.0" in snippet
    assert "self.last_corridor_time = 0.0" in snippet


def test_get_control_resets_new_accumulators_alongside_existing_ones():
    idx = _MPC_SRC.index("    def get_control(self)")
    idx_first_solve = _MPC_SRC.index("dec = self._active.solve()", idx)
    snippet = _MPC_SRC[idx:idx_first_solve]
    assert "self.last_linearize_time = 0.0" in snippet
    assert "self.last_corridor_time = 0.0" in snippet


def test_linearize_loop_is_timed_around_the_for_n_in_range_block():
    idx_start = _MPC_SRC.index("for n in range(N):")
    idx_add = _MPC_SRC.index("self.last_linearize_time += _time.perf_counter() - _t0")
    idx_corridor_call = _MPC_SRC.index("ub, lb = self._corridor(N, safety_margin)")
    # 計測終了(pf_add相当のaccumulate)がループの後、コリドー呼び出しの前にあること
    assert idx_start < idx_add < idx_corridor_call


def test_corridor_call_is_timed_separately_from_linearize():
    idx_corridor_call = _MPC_SRC.index("ub, lb = self._corridor(N, safety_margin)")
    idx_add = _MPC_SRC.index(
        "self.last_corridor_time += _time.perf_counter() - _t0", idx_corridor_call)
    idx_d_ub0 = _MPC_SRC.index("d['ub0'] = ub", idx_corridor_call)
    assert idx_corridor_call < idx_add < idx_d_ub0


def test_controller_reads_linearize_and_corridor_into_perf_buckets():
    idx = _CTRL_SRC.index("self._pf_add('mpc_setup'")
    idx_end = idx + 500
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._pf_add('mpc_linearize', getattr(self._mpc, 'last_linearize_time', 0.0))" in snippet
    assert "self._pf_add('mpc_corridor', getattr(self._mpc, 'last_corridor_time', 0.0))" in snippet


def test_linearize_and_corridor_together_are_subset_of_setup_not_double_counted():
    """①非矛盾性: linearize+corridorはsetup(_stage_data全体+_init_problemの残り)の
    内訳であり、setup自体の計測式(初回+リトライ2回分、177節続報で確立済み)を
    変更していないことを確認する。linearize/corridorはreuse時(リトライ)は再計算
    されない(_stage_dataのreuse is None分岐内でのみ実行)ため1回のみでよい。"""
    assert _MPC_SRC.count("self.last_setup_time += _time.perf_counter() - _t0") == 2
    assert _MPC_SRC.count("self.last_linearize_time += _time.perf_counter() - _t0") == 1
    assert _MPC_SRC.count("self.last_corridor_time += _time.perf_counter() - _t0") == 1
