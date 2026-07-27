"""Unit tests for the corridor-boundary ratchet deadlock fix (80節, 2026-07-16).

Root cause (0716-01実測, Lap3第3コーナー): ReferencePath.update_path_constraints()
(core/reference_path.py) clamps each cycle's freshly-computed ub_sm/lb_sm to be no
WIDER than the previously-stored wp.ub_sm/wp.lb_sm — a one-way ratchet that only
narrows, never widens, unless reset_dynamic_constraints() is called. That reset is
gated on mpc_controller.py's `_rebuild_deadband` check, which only fires when some
tracked obstacle has moved >0.3m since the last rasterization. A genuinely
STATIONARY opponent (v=0.00) never triggers this on its own, so while the ego is
stalled at the same waypoint attempting an F3-creep pass, any cycle-to-cycle noise
in the freshly-computed bound gets ratcheted into a PERMANENT narrowing — squeezing
the achievable offset back toward center, shrinking fwd_dlat, shrinking the F3-taper
floor, driving commanded speed toward 0, and triggering STUCK-detection. The
BACKUP+PUSH recovery repositions the ego by ~0m net but does nothing to the ratchet
state, so the identical deadlock recurs (observed at ~18-19s intervals, matching
gaps between a moving second opponent occasionally tripping the rebuild deadband).

Fix: call reference_path.reset_dynamic_constraints() once the stuck-recovery
maneuver completes (both the PUSH-based path and the plain WAIT_DRIVE path),
clearing the ratchet memory so the very next cycle's update_path_constraints()
call recomputes ub_sm/lb_sm from the current (accurate) map rather than the
stale, needlessly-narrowed one that caused the stall.

core/reference_path.py has no rclpy dependency, so ReferencePath.reset_dynamic_
constraints() is tested by real import/execution below (not a mirror). The
mpc_controller.py call-site wiring is verified structurally (source-text check,
same technique as test_switchback_token_wiring.py) since mpc_controller.py
imports rclpy/autoware message types at module scope.
"""
import os
import types

import pytest

from multi_purpose_mpc_ros.core.reference_path import ReferencePath, Waypoint


def make_waypoint(ub, lb, ub_sm, lb_sm):
    wp = Waypoint(x=0.0, y=0.0, psi=0.0, kappa=0.0)
    wp.ub = ub
    wp.lb = lb
    wp.ub_sm = ub_sm
    wp.lb_sm = lb_sm
    wp.static_border_cells = (("ub_cell",), ("lb_cell",))
    wp.dynamic_border_cells = None
    return wp


def test_reset_restores_ratcheted_bounds_to_static_baseline():
    """本修正の中核: ratchetで狭められたub_sm/lb_smが、reset_dynamic_constraints()
    呼び出し1回で静的なub/lb(本来の全幅)へ完全に戻ることを、実際のReferencePath
    メソッドを実行して確認する。"""
    wp = make_waypoint(ub=3.0, lb=-3.0, ub_sm=0.4, lb_sm=-0.3)  # ratchetで極端に狭まった状態を再現
    mock_self = types.SimpleNamespace(waypoints=[wp])
    ReferencePath.reset_dynamic_constraints(mock_self)
    assert wp.ub_sm == pytest.approx(3.0)
    assert wp.lb_sm == pytest.approx(-3.0)


def test_reset_restores_multiple_waypoints_independently():
    """回帰: horizon内の複数waypointがそれぞれ独立してratchetされていても、
    reset_dynamic_constraints()は全waypointを個別に正しく復元する。"""
    wp1 = make_waypoint(ub=2.5, lb=-2.5, ub_sm=0.1, lb_sm=-0.1)
    wp2 = make_waypoint(ub=3.5, lb=-1.5, ub_sm=0.5, lb_sm=-0.2)
    mock_self = types.SimpleNamespace(waypoints=[wp1, wp2])
    ReferencePath.reset_dynamic_constraints(mock_self)
    assert (wp1.ub_sm, wp1.lb_sm) == pytest.approx((2.5, -2.5))
    assert (wp2.ub_sm, wp2.lb_sm) == pytest.approx((3.5, -1.5))


def test_reset_also_restores_dynamic_border_cells_from_static():
    """回帰: dynamic_border_cellsもstatic_border_cellsのdeepcopyへ戻ることを確認する
    (ratchet解除がub_sm/lb_smの数値だけでなく描画用セルも含め完全にリセットされる)。"""
    wp = make_waypoint(ub=3.0, lb=-3.0, ub_sm=0.4, lb_sm=-0.3)
    mock_self = types.SimpleNamespace(waypoints=[wp])
    ReferencePath.reset_dynamic_constraints(mock_self)
    assert wp.dynamic_border_cells == wp.static_border_cells


def test_retroactive_0716_01_wp40_ratchet_scenario():
    """遡及検証(0716-01実測、Lap3 wp40のSTUCKデッドロック): 実測されたoffset推移
    (-3.0 -> -2.469 -> ... -> -2.436、fwd_dlatが0.16m相当まで収縮)に対応する
    lb_sm(左側境界)のratchet状態を再現し、reset_dynamic_constraints()適用前後で
    利用可能な幅がどれだけ回復するかを定量的に示す。"""
    # 実測: 静的な全幅相当(narrow前、t=144.6時点のub0=1.576 lb0=-2.530相当)に対し、
    # ratchetにより実測lb_smが-2.436まで収縮していた(狭まり幅 -2.530 -> -2.436 = 0.094m
    # だが、この収縮がSTUCK発生まで毎周期繰り返し蓄積し続けたことが問題の本質)。
    wp = make_waypoint(ub=1.576, lb=-2.530, ub_sm=1.596, lb_sm=-2.436)
    mock_self = types.SimpleNamespace(waypoints=[wp])
    _lb_before_reset = wp.lb_sm
    ReferencePath.reset_dynamic_constraints(mock_self)
    _recovered_width = wp.lb_sm - _lb_before_reset
    assert wp.lb_sm == pytest.approx(-2.530)
    assert _recovered_width < -0.05  # 実測で失われていた左側マージンが回復することを確認
    assert wp.ub_sm == pytest.approx(1.576)


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線(構造的な回帰防止、AST/文字列検証)
# ---------------------------------------------------------------------------
_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_reset_dynamic_constraints_called_at_push_completion():
    """回帰防止: STUCK-PUSH完了(NORMAL復帰)の分岐でreset_dynamic_constraints()が
    呼ばれていることを確認する(deadband頼みだけでなく、stuck-recovery完了時にも
    ratchetを解除する新しい経路)。
    2026-07-21修正(148節②): 4箇所の完了処理が_stuck_recovery_complete()へ
    統合されたため、直接呼び出しではなく reset_corridor=True での委譲を確認する。
    2026-07-24更新(171節続報): 経路1/2専用だったWAIT_DRIVE(ステア0固定の直進
    復帰、旧test_reset_dynamic_constraints_called_at_wait_drive_completion)が
    PUSHへ統合され到達不能になったため削除された。経路1/2/3は全てこのPUSH完了
    分岐を通るため、本テストが実質的に全経路をカバーする。
    2026-07-26更新(184節): _stuck_recovery_complete()にシャッフルエピソード
    記録用のnow/pose引数が追加され、呼び出しが複数行にまたがるようになった
    ため、検索対象の文字列とウィンドウ幅を実際のフォーマットに合わせて更新した
    (reset_backup_state/reset_corridorの値自体は無変更)。"""
    idx = _SRC.index('[STUCK] PUSH終了')
    snippet = _SRC[idx:idx + 500]
    assert "self._stuck_recovery_complete(reset_backup_state=False, reset_corridor=True," in snippet
    assert "now=now, pose=pose)" in snippet


def test_stuck_recovery_complete_calls_reset_dynamic_constraints_when_requested():
    """148節②で新設した_stuck_recovery_complete()自体が、reset_corridor=Trueの
    時のみreset_dynamic_constraints()を呼ぶことを確認する(PUSH/WAIT_DRIVE完了は
    True、BACKUP系の断念経路はFalseという従来の使い分けを、共通ヘルパー内の
    条件分岐として保存している)。"""
    idx = _SRC.index("def _stuck_recovery_complete(")
    idx_end = _SRC.index("def _handle_stuck_recovery(")
    snippet = _SRC[idx:idx_end]
    assert "if reset_corridor:" in snippet
    assert "self._reference_path.reset_dynamic_constraints()" in snippet


def test_corridor_reset_log_tag_present_for_future_verification():
    """検証ロギング: [CORRIDOR-RESET]タグが今後のログでratchet解除が実際に
    発火したかどうかを追跡できることを確認する。"""
    assert _SRC.count("[CORRIDOR-RESET]") >= 1
    assert _SRC.count("self._reference_path.reset_dynamic_constraints()") >= 1
