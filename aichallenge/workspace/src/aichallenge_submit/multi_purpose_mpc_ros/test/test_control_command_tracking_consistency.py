"""Unit tests for issue④②③: 制御コマンド追跡変数の陳腐化バグ群(2026-07-22)。

背景: issue④(fallback_forwardがwall_slowをバイパスする問題)の根本原因を
切り分ける過程で、同じ「特殊経路の間、レート制限/平滑化の基準となる
“直前に実際に発行したコマンド” を追跡する変数が更新されない」という
バグが3箇所に存在することが判明した。

- issue④①(元の発見): mpc_controller.pyのfallback_forward分岐自体。
  core/MPC.py get_control()がQP infeasibleの間、前回成功時の計画軌道を
  1歩ずつ先送りし、それも尽きると u=[0.0, 0.0](操舵も強制ゼロ)を返す。
  速度はv_safe(wall_slow等)で正しく再キャップされるが、操舵は
  コリドー・壁・相手車の状況と一切無関係な値になる。
- issue④②(本ファイルで検証): core/MPC.py の self.previous_steering は
  QPが解けた場合(tryの628行目相当)のみ更新され、infeasibleの間の
  except節では更新されないため、再解決成功時のレートクランプが
  凍結中に実際に出力していた操舵とは無関係な古い基準を使ってしまう。
- issue④③(本ファイルで検証): mpc_controller.py の self._last_u/
  self._last_acc は通常の_control()フロー内でのみ更新されていた。
  STUCK復帰(_handle_stuck_recovery)は_publish_control_commandを
  直接呼ぶだけでこの更新をバイパスするため、STUCK復帰完了直後の
  最初の周期、下流の低域通過フィルタがSTUCK突入前の古い値を基準に
  平滑化してしまう不整合があった。

対処方針(ユーザー要望「まとめてで良い、一貫性のある、矛盾しないシンプルな
処理」): 「実際に発行するコマンドを追跡する変数は、それを発行する経路が
どれであっても必ず同じ場所で更新される」という単一の原則で3箇所とも統一。
- previous_steeringはexcept節でも同じ意味(直近に実際に出力した操舵)で
  更新する(try節と対称、新規状態0個)。
- _last_u/_last_accの更新を_publish_control_command内(全呼び出し元が
  経由する唯一の場所)へ一本化し、呼び出し元(通常フロー/STUCK復帰)に
  よらず実発行値と常に一致することを構造的に保証する(新規状態0個、
  既存の2箇所の代入を1箇所へ集約しただけ)。

mpc_controller.py/core/MPC.pyはrclpy依存のため直接importできないため、
test_footprint_taper.pyと同じ方針(純Pythonミラー関数+ソーステキストに
よる構造的検証)を用いる。
"""
import os

import pytest

_MPC_CONTROLLER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_MPC_CONTROLLER_PATH) as _f:
    _CTRL_SRC = _f.read()

_CORE_MPC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "MPC.py")
with open(_CORE_MPC_PATH) as _f:
    _MPC_SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: previous_steeringのミラー実装(try/except両方で同じ意味を持つか)
# ---------------------------------------------------------------------------

class _FakeGetControl:
    """core/MPC.pyのget_control()のprevious_steering更新部分のみを抽出した
    複製ミラー。solve_ok=Falseのとき例外相当の経路(except節)を通り、
    Trueのとき解けた経路(try節)を通る。"""

    def __init__(self):
        self.previous_steering = 0.0
        self.infeasibility_counter = 0

    def step(self, solve_ok: bool, solved_delta: float, coasted_or_zero_delta: float):
        if solve_ok:
            self.previous_steering = solved_delta
            self.infeasibility_counter = 0
        else:
            self.previous_steering = coasted_or_zero_delta
            self.infeasibility_counter += 1
        return self.previous_steering


def test_previous_steering_updated_on_infeasible_cycle_not_frozen():
    """②の回帰防止: infeasibleが連続しても、previous_steeringは毎周期
    実際にexcept節が返す操舵(先送り値、または枯渇後の0.0)に追従し、
    infeasible突入前の値のまま凍結されない。"""
    mpc = _FakeGetControl()
    mpc.step(solve_ok=True, solved_delta=0.3, coasted_or_zero_delta=0.0)
    assert mpc.previous_steering == pytest.approx(0.3)
    # infeasible突入、先送り値が徐々に変化
    mpc.step(solve_ok=False, solved_delta=0.0, coasted_or_zero_delta=0.25)
    assert mpc.previous_steering == pytest.approx(0.25)  # 0.3の凍結ではない
    mpc.step(solve_ok=False, solved_delta=0.0, coasted_or_zero_delta=0.0)  # 先送りバッファ枯渇
    assert mpc.previous_steering == pytest.approx(0.0)


def test_previous_steering_resume_clamp_anchored_to_actual_last_value():
    """①非矛盾性の核心: infeasible解消後に新しい解が求まった際のレートクランプは、
    「直近に実際に出力していた値」(この例ではexcept節が返した0.0)を基準にする
    べきで、infeasible突入前の値(0.3)を基準にしてはならない。"""
    mpc = _FakeGetControl()
    mpc.step(solve_ok=True, solved_delta=0.3, coasted_or_zero_delta=0.0)
    mpc.step(solve_ok=False, solved_delta=0.0, coasted_or_zero_delta=0.0)  # 実際は0.0を出力していた
    max_delta_change = 0.05
    anchor = mpc.previous_steering  # 修正後は0.0を基準にすべき
    assert anchor == pytest.approx(0.0)
    # 新しい解0.2はこの基準からmax_delta_change以内にクランプされる
    clamped = max(anchor - max_delta_change, min(anchor + max_delta_change, 0.2))
    assert clamped == pytest.approx(0.05)  # 0.3基準だと0.25にクランプされ実態と乖離していた


# ---------------------------------------------------------------------------
# ③非矛盾性: _last_u/_last_accのミラー実装(通常フロー/STUCK復帰で同じか)
# ---------------------------------------------------------------------------

class _FakeController:
    """mpc_controller.pyの_last_u/_last_acc更新を_publish_control_command内へ
    一本化した後の挙動の複製ミラー。"""

    def __init__(self):
        self.last_u = [0.0, 0.0]
        self.last_acc = 0.0

    def publish_control_command(self, u, acc):
        self.last_u[0] = float(u[0])
        self.last_u[1] = float(u[1])
        self.last_acc = float(acc)

    def normal_control_cycle(self, u, acc):
        """通常の_control()フロー: 低域通過フィルタ計算後にpublish_control_commandを呼ぶ。"""
        filtered_steer = self.last_u[1] + (u[1] - self.last_u[1]) * 0.5
        self.publish_control_command([u[0], filtered_steer], acc)

    def stuck_recovery_cycle(self, u, acc):
        """STUCK復帰(_handle_stuck_recovery)経路: フィルタを経由せず直接publish_control_commandを呼ぶ。"""
        self.publish_control_command(u, acc)


def test_last_u_updated_during_stuck_recovery_path():
    """③の回帰防止: STUCK復帰経路でpublishしたコマンドも_last_uへ反映される
    (従来はここが更新されず凍結していた)。"""
    ctrl = _FakeController()
    ctrl.normal_control_cycle([2.0, 0.1], acc=0.5)
    assert ctrl.last_u == [2.0, pytest.approx(0.05)]
    ctrl.stuck_recovery_cycle([-1.0, 0.0], acc=-0.8)  # BACKUP相当のコマンド
    assert ctrl.last_u[0] == pytest.approx(-1.0)
    assert ctrl.last_u[1] == pytest.approx(0.0)
    assert ctrl.last_acc == pytest.approx(-0.8)


def test_low_pass_filter_anchored_to_actual_last_published_value_after_stuck():
    """①非矛盾性の核心: STUCK復帰完了直後、通常フローに戻った最初の周期の
    低域通過フィルタは、STUCK復帰中に実際に発行していた最後の操舵
    (この例では0.0、BACKUP相当)を基準にする必要がある。旧実装では
    STUCK突入前の古い値(0.1)のまま凍結されていたため、フィルタが誤った
    基準から出発していた。"""
    ctrl = _FakeController()
    ctrl.normal_control_cycle([2.0, 0.1], acc=0.5)
    ctrl.stuck_recovery_cycle([-1.0, 0.0], acc=-0.8)  # BACKUP: 操舵0.0を複数周期発行
    ctrl.stuck_recovery_cycle([-1.0, 0.0], acc=-0.8)
    # STUCK復帰完了、通常フローが再開。新しい解が0.2を要求。
    ctrl.normal_control_cycle([1.5, 0.2], acc=0.3)
    # 基準はSTUCK中に実際に発行していた0.0であるべき(0.1ではない)
    expected_steer = 0.0 + (0.2 - 0.0) * 0.5
    assert ctrl.last_u[1] == pytest.approx(expected_steer)


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_source_mpc_previous_steering_updated_in_except_branch():
    """②: core/MPC.pyのexcept節にprevious_steering更新が追加されていることを
    確認する。"""
    idx = _MPC_SRC.index("except (TypeError, ValueError):")
    snippet = _MPC_SRC[idx:idx + 1400]
    assert "self.previous_steering = float(u[1])" in snippet


def test_source_mpc_previous_steering_still_updated_in_try_branch():
    """回帰: try節側(628行目相当)の既存の意味は変更していない。"""
    idx = _MPC_SRC.index("self.previous_steering = delta")
    idx_try = _MPC_SRC.index("dec = self._active.solve()")
    idx_except = _MPC_SRC.index("except (TypeError, ValueError):")
    assert idx_try < idx < idx_except  # try節側(except節より前)にあることを確認


def test_source_last_u_and_last_acc_consolidated_in_publish_control_command():
    """③: _last_u/_last_accの更新が_publish_control_command内へ一本化され、
    通常フロー側(旧位置)には代入が残っていないことを確認する。"""
    idx = _CTRL_SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    snippet = _CTRL_SRC[idx:idx + 900]
    assert "self._last_u[0] = float(u[0])" in snippet
    assert "self._last_u[1] = float(u[1])" in snippet
    assert "self._last_acc = float(acc)" in snippet


def test_source_last_u_last_acc_not_duplicated_elsewhere():
    """②非冗長性: _last_u[0]=/_last_u[1]=/_last_acc=(読み取りではなく代入)が
    _publish_control_command以外に存在しない(通常フロー側の旧代入は削除済み、
    唯一の更新箇所という不変条件が壊れていない)ことを確認する。"""
    assert _CTRL_SRC.count("self._last_u[0] = ") == 1
    assert _CTRL_SRC.count("self._last_u[1] = ") == 1
    assert _CTRL_SRC.count("self._last_acc = ") == 2  # 更新1箇所 + __init__相当の初期化1箇所


def test_source_handle_stuck_recovery_calls_publish_control_command():
    """③の前提確認: STUCK復帰経路(_handle_stuck_recovery)が_publish_control_command
    を経由してコマンドを発行しており、_last_u/_last_accの一本化が実際に
    STUCK復帰にも適用されることを確認する。"""
    idx = _CTRL_SRC.index("def _handle_stuck_recovery(self, now, pose) -> None:")
    idx_end = _CTRL_SRC.index("def ", idx + 10)
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._publish_control_command(now, u, acc, False)" in snippet


def test_source_normal_flow_calls_same_publish_control_command():
    """③: 通常フローも同じ_publish_control_commandを経由することを確認する
    (2箇所の呼び出し元が同一の更新ロジックを共有している)。"""
    assert _CTRL_SRC.count("self._publish_control_command(") == 2
