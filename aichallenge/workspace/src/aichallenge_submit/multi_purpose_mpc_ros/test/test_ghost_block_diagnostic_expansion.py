"""Unit tests for the expanded [GHOST-BLOCK] diagnostic (139節続報、
2026-07-20).

Background: 0720-03実測(wp187、72秒間の完全停止)で、fwd=0・n_dynobs=0
(追跡上は近くに誰もいない)・EKF横偏差も小さい(0.16〜0.25m)にもかかわらず、
前進も後退も実速度がほぼ0のまま72秒続く事例を発見した。当初「対戦車に
挟まれている」と推測したが、opp座標を参照経路の弧長へ逆算した結果egoから
約73m離れておりこの仮説は否定された(訂正、design_docs 140節参照)。

真因(AWSIM上の物理的な引っ掛かり/アクチュエータ飽和/未知の固着)を次回
ログで特定するため、[GHOST-BLOCK]ログへ以下を追加した(新規購読・新規
スキャン処理0個、すべて既存の値の再利用): ①ギア状態(self._gear_report)、
②直近の操舵指令(self._last_u[1])、③MPC不可解カウンタ
(self._mpc.infeasibility_counter)、④動的コリドー(dbg_corr_ub0/lb0)、
⑤egoの現在位置における占有格子の値(self._map.data)、⑥生のpose。

mpc_controller.pyはrclpy依存でモジュールスコープの型を直接importできない
ため、ソーステキストの構造的検証で配線を確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _ghost_block_snippet():
    idx = _SRC.index('f"[GHOST-BLOCK]')
    idx_end = _SRC.index('opp[{self._opp_snapshot_str()}]")')
    return _SRC[max(0, idx - 900):idx_end]


def test_ghost_block_includes_gear_state_reusing_existing_field():
    """②非冗長性: 新規購読を追加せず、既存self._gear_report
    (STUCK-BACKUPが既に使う_gear_label)を再利用する。"""
    snippet = _ghost_block_snippet()
    assert "self._gear_label(self._gear_report.report)" in snippet


def test_ghost_block_includes_last_steering_command():
    """②非冗長性: 新規状態を追加せず、既存self._last_u[1]
    (低域通過フィルタ後の直近操舵指令)を再利用する。"""
    snippet = _ghost_block_snippet()
    assert "self._last_u[1]" in snippet


def test_ghost_block_includes_infeasibility_counter():
    snippet = _ghost_block_snippet()
    assert "self._mpc.infeasibility_counter" in snippet


def test_ghost_block_includes_dynamic_corridor_same_source_as_ot_log():
    """②非冗長性: [OT]ログと同じgetattr(self._mpc, "dbg_corr_ub0"/"dbg_corr_lb0", ...)
    取得方法を再利用する(新規のコリドー取得経路を作らない)。"""
    snippet = _ghost_block_snippet()
    assert 'getattr(self._mpc, \'dbg_corr_ub0\'' in snippet
    assert 'getattr(self._mpc, \'dbg_corr_lb0\'' in snippet


def test_ghost_block_includes_occupancy_grid_value_at_ego_pose():
    """新規追加: egoの現在位置における占有格子の値(self._map.data)を確認する
    (「モデルが壁の中にいると認識しているか」の直接診断)。既存self._map.w2m
    (座標変換、他所で既に使用中)を再利用する。"""
    snippet = _ghost_block_snippet()
    assert "self._map.w2m(pose.x, pose.y)" in snippet
    assert "self._map.data[_py, _px]" in snippet
    assert "occ_at_pose=" in snippet


def test_ghost_block_occupancy_lookup_is_exception_safe():
    """回帰防止: 占有格子ルックアップが座標変換異常時(範囲外等)に
    GHOST-BLOCKログ自体をクラッシュさせないよう、例外を捕捉しNoneに
    フォールバックすることを確認する(既存の他所のtry/exceptパターンと同じ)。"""
    idx = _SRC.index("_occ_val = None")
    idx_end = _SRC.index('f"[GHOST-BLOCK]')
    snippet = _SRC[idx:idx_end]
    assert "try:" in snippet
    assert "except Exception:" in snippet
    assert "pass" in snippet


def test_ghost_block_includes_raw_pose_for_mcap_cross_reference():
    """mcapの実位置と突き合わせるため、生のpose(x/y/theta)を出力することを
    確認する。"""
    snippet = _ghost_block_snippet()
    assert "pose_x={pose.x:.2f}" in snippet
    assert "pose_y={pose.y:.2f}" in snippet
    assert "pose_theta={pose.theta:.3f}" in snippet


def test_ghost_block_still_fires_once_per_episode_no_regression():
    """回帰防止: エッジトリガー方式(_ghost_block_logged)自体は変更していない
    ことを確認する(診断項目の追加のみ、発火頻度ロジックは無変更)。"""
    idx = _SRC.index("if (self._stuck_count >= self._ghost_block_hold_cycles")
    snippet = _SRC[idx:idx + 200]
    assert "not self._ghost_block_logged" in snippet
    assert "self._ghost_block_logged = True" in snippet


def test_ghost_block_existing_fields_unchanged_regression():
    """回帰防止: 既存フィールド(u0_last/v/count/wp/ot_state/opp)が
    引き続き全て出力されることを確認する。"""
    snippet = _ghost_block_snippet()
    for field in ("u0_last={self._stuck_u0_last:.2f}",
                  "v={_v_odom_now:.2f}",
                  "count={self._stuck_count}",
                  "wp={self._mpc.model.wp_id}",
                  "ot_state={self._ot_state}"):
        assert field in snippet
    assert "opp[{self._opp_snapshot_str()}]" in _SRC[
        _SRC.index('f"[GHOST-BLOCK]'):_SRC.index('f"[GHOST-BLOCK]') + 1200]
