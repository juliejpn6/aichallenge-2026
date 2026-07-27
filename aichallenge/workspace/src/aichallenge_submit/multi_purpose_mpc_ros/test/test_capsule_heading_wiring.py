"""Structural tests for the capsule-obstacle wiring in mpc_controller.py
(131-6節②、寸法モデルの一元化, 2026-07-20).

Background: 円形近似(vehicle_radius=0.8)による相手車の全長方向過小表現に対処する
ため、現在位置(t=0)の円を進行方向へ前後分割するpredictions_to_obstacles_capsule
を実装した(v2x_vehicle_tracker.py側の単体テストはtest_v2x_vehicle_tracker.py)。
本ファイルは、mpc_controller.py側の配線(見出し計算・関数呼び出し・既存パラメータ
の再利用・エッジトリガーログ)が正しいことをソーステキスト検証で確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_dynamic_obstacles_uses_capsule_function_not_the_old_one():
    """実際に使われている関数がpredictions_to_obstacles_capsuleであることを確認する
    (旧predictions_to_obstaclesへの呼び出しがコールサイトに残っていないこと)。"""
    idx = _SRC.index("self._dynamic_obstacles = predictions_to_obstacles_capsule(")
    assert idx > 0
    assert "self._dynamic_obstacles = predictions_to_obstacles(\n" not in _SRC


def test_capsule_import_present():
    assert "predictions_to_obstacles_capsule," in _SRC or \
        "predictions_to_obstacles_capsule\n" in _SRC


def test_half_length_derived_from_along_min_length_no_new_parameter():
    """②非冗長性: half_lengthが既存along_min_length/2から導出されており、
    新規パラメータを追加していないことを確認する。"""
    idx = _SRC.index("self._dynamic_obstacles = predictions_to_obstacles_capsule(")
    snippet = _SRC[idx:idx + 300]
    assert "self._along_min_length / 2.0" in snippet


def test_heading_threshold_reuses_opp_obstacle_speed_no_new_parameter():
    """②非冗長性: 速度ベース見出し採用の閾値が、既存opp_obstacle_speed
    (障害物クラス判定と同一の閾値)を再利用していることを確認する。"""
    idx = _SRC.index('np.hypot(_vx, _vy) >= self._opp_obstacle_speed')
    assert idx > 0


def test_heading_falls_back_to_track_tangent_when_slow():
    """低速/停止車は速度ベクトルの代わりに参照経路接線(wp.psi)へ
    フォールバックすることを確認する。"""
    idx = _SRC.index('np.hypot(_vx, _vy) >= self._opp_obstacle_speed')
    snippet = _SRC[idx:idx + 300]
    assert "_wpo_h.psi" in snippet


def test_heading_computed_independent_of_on_pit_gate():
    """見出し計算(headings[vid]の確定)が、既存のon_pit学習ガード
    (if not self._on_pit:)より前に行われている(ピット中も安全側の
    capsule分割が効くようにする)ことを確認する。"""
    idx_heading = _SRC.index("headings[vid] = _heading")
    idx_on_pit_gate = _SRC.index("if not self._on_pit:")
    assert idx_heading < idx_on_pit_gate


def test_capsule_heading_log_is_edge_triggered():
    """[CAPSULE-HEADING]ログが、見出しソース(velocity/track_tangent)が
    前回から変化した周期のみ発火するエッジトリガー方式であることを確認する。"""
    idx = _SRC.index('f"[CAPSULE-HEADING]')
    snippet = _SRC[max(0, idx - 300):idx]
    assert "_prev_src != _heading_src" in snippet


def test_headings_dict_initialized_before_loop():
    """headings辞書がループの外(try節の前)で初期化されており、
    例外発生時もpredictions_to_obstacles_capsuleへ未定義変数を渡さないことを確認する。"""
    idx_init = _SRC.index("headings = {}")
    idx_call = _SRC.index("self._dynamic_obstacles = predictions_to_obstacles_capsule(")
    assert idx_init < idx_call
