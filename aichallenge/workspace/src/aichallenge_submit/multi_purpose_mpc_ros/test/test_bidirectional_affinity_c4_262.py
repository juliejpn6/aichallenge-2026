"""262節続報(2026-08-01、判定基準改訂+work_cpu計装Phase 4): C4実験「双方向
アフィニティ」。

背景: C3(mpc_controllerプロセス単体をos.sched_setaffinityで論理2-3へ固定)は
外部競合(nivcsw)の低減には成功したが、同居する他ROS2ノード(rviz2・
ekf_localizer・autostart_orchestrator等)は依然mpc専有コアへ自由に侵入できる
「一方通行」の隔離だった。C4はこれを解消するため、他ノードのソースコードには
一切手を加えず、`ros2 launch`親プロセス自体(aichallenge/run_autoware.bash)を
taskset(AUTOWARE_OTHER_NODES_CPU_AFFINITY、既定未設定=従来通り無制限)で
起動する。taskset無しで起動された子プロセスは通常CPUアフィニティを親から
継承するため、mpc_controllerだけが自分自身のアフィニティを後から上書きし、
他の全ノードは親のtaskset範囲(mpc専有コアの補集合)に留まる、という双方向の
隔離が実現する。

run_autoware.bashは直接importできないシェルスクリプトのため、ソーステキスト
構造検証で確認する(既存のtest_blas_thread_limit_262.pyと同じ方針)。
"""
import os

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "run_autoware.bash")
with open(_SCRIPT_PATH) as _f:
    _SCRIPT_SRC = _f.read()


def test_taskset_wraps_ros2_launch_when_env_var_set():
    idx = _SCRIPT_SRC.index('if [ -n "${AUTOWARE_OTHER_NODES_CPU_AFFINITY:-}" ]; then')
    idx_end = _SCRIPT_SRC.index("fi\n", idx)
    snippet = _SCRIPT_SRC[idx:idx_end]
    assert 'taskset -c "${AUTOWARE_OTHER_NODES_CPU_AFFINITY}"' in snippet
    assert "ros2 launch aichallenge_system_launch aichallenge_system.launch.xml" in snippet


def test_ros2_launch_unwrapped_when_env_var_unset():
    """既定(環境変数未設定)ではtasksetを一切経由しない従来通りの起動経路が
    else節に存在すること(既定挙動を変えない安全側の設計)。"""
    idx = _SCRIPT_SRC.index('if [ -n "${AUTOWARE_OTHER_NODES_CPU_AFFINITY:-}" ]; then')
    idx_else = _SCRIPT_SRC.index("else", idx)
    idx_fi = _SCRIPT_SRC.index("fi\n", idx_else)
    else_snippet = _SCRIPT_SRC[idx_else:idx_fi]
    assert "taskset" not in else_snippet
    assert "ros2 launch aichallenge_system_launch aichallenge_system.launch.xml" in else_snippet


def test_taskset_check_uses_dash_n_not_equality():
    """空文字列と未設定変数の両方を「無効」として扱う-nチェックであることを
    確認する(:-によるデフォルト展開と組み合わせた安全な判定)。"""
    assert '[ -n "${AUTOWARE_OTHER_NODES_CPU_AFFINITY:-}" ]' in _SCRIPT_SRC


def mirror_should_use_taskset(env_value):
    """シェルの[ -n "${VAR:-}" ]構文のミラー(未設定/空文字ならFalse)。"""
    return bool(env_value)


def test_mirror_no_taskset_when_unset_or_empty():
    assert mirror_should_use_taskset(None) is False
    assert mirror_should_use_taskset("") is False


def test_mirror_taskset_when_set():
    assert mirror_should_use_taskset("0,1,6-15") is True


def test_pf_log_colocated_affinity_called_after_own_affinity_set():
    """mpc_controller自身のsched_setaffinity実行後に同居プロセスのアフィニティ
    走査を呼ぶ順序であること(自分のアフィニティ確定後の状態を記録するため)。"""
    py_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
    with open(py_path) as f:
        py_src = f.read()
    idx_affinity_set = py_src.index('cpu_affinity = getattr(self._cfg.mpc, "cpu_affinity"')
    idx_checklist_call = py_src.index("self._pf_log_platform_checklist()", idx_affinity_set)
    assert idx_affinity_set < idx_checklist_call


def test_pf_log_colocated_affinity_scans_expected_targets():
    py_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
    with open(py_path) as f:
        py_src = f.read()
    idx = py_src.index("def _pf_log_colocated_affinity(self):")
    idx_end = py_src.index("\n    def ", idx + 30)
    snippet = py_src[idx:idx_end]
    for target in ("rviz2", "component_container", "autostart_orchestrator",
                    "v2x_marker_publisher"):
        assert f"'{target}'" in snippet, f"missing target {target!r}"
    assert "Cpus_allowed_list:" in snippet
    assert "except OSError:" in snippet


def test_c4_experiment_compose_file_sets_complement_affinity():
    compose_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
        "docker-compose.c4-experiment.yml")
    with open(compose_path) as f:
        compose_src = f.read()
    assert "AUTOWARE_OTHER_NODES_CPU_AFFINITY=0,1,6-15" in compose_src
