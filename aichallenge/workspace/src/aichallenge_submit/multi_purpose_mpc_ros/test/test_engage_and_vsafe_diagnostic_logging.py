"""Structural regression guard for the 91節 diagnostic logging additions (2026-07-17).

Background: 08節のオープン項目のうち、③(エンゲージ判定の再設計: engage_max_dist効果
検証・engage_cooldown固定値見直し・stage2追いつき予測)と④(v_safe候補の相互作用:
wall_slow×switchback頻度・v_safe_srcのチャーン)は、どちらも次回ログでの実測データが
無いと設計を進められない項目だった。ユーザー指示により、対処(設計)はまだ行わず、
次回ログ収集に向けた検証ロギングのみを追加する。

- [ENGAGE]: _can_engage=Trueになった瞬間(この分岐はstate!=OVERTAKINGの間のみ
  評価されるため、常に新規エンゲージそのもの)に、fwd_ds/engage_dist_dynamic/
  t_reach_profileを記録する(③の実測データ収集用)。
- [V-SAFE-SRC-CHANGE]: OVERTAKING中、v_safe_srcが前周期と変化した瞬間を
  (既存[OT]ログの1Hz間引きとは独立に)間引かずに記録する(④のチャーン頻度・
  遷移パターンの実測データ収集用)。

mpc_controller.py(rclpy依存のため直接importできない)に対する構造的なソーステキスト
検証で、両ログが正しい位置・条件で追加されていることを確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_engage_log_fires_inside_can_engage_branch():
    """[ENGAGE]ログが_can_engage=Trueブロックの中にあり、new_engageを判定する
    prevフラグ等を必要としない(このブロック自体がstate!=OVERTAKINGの間のみ
    評価されるため、_can_engage=Trueは常に新規エンゲージ)ことを確認する。"""
    idx = _SRC.index("if _eval.can_engage:")
    # 2026-07-20追加(132節、Gap①Phase0)でENGAGEログにdlat_v_ema/dlat_shrink_run
    # (診断専用)が追加され、以降のstate遷移行までの距離が伸びたため窓を拡大した。
    snippet = _SRC[idx:idx + 1900]
    assert '"[ENGAGE] side={_eval.plan_side}' in snippet
    assert "self._ot_state = \"OVERTAKING\"" in snippet


def test_engage_log_includes_both_engage_distance_paths():
    """[ENGAGE]ログが、追いつき予測(profile経路)と旧v_pot近似(dynamic経路)の
    両方の実測値を記録し、どちらの経路でエンゲージしたか(path=)を区別できる
    ことを確認する(③のengage_max_dist効果検証に必要)。"""
    idx = _SRC.index('"[ENGAGE] side={_eval.plan_side}')
    # 2026-07-20追加(132節、Gap①Phase0)のdlat_v_ema/dlat_shrink_runフィールド分、窓を拡大。
    snippet = _SRC[max(0, idx - 50):idx + 900]
    assert "engage_dist_dynamic={_eval.engage_dist_dynamic:.2f}" in snippet
    assert "t_reach_profile={_eval.t_reach_profile}" in snippet
    assert "path={'profile' if _eval.t_reach_profile is not None else 'dynamic'}" in snippet


def test_v_safe_src_change_log_gated_to_overtaking_and_edge_triggered():
    """[V-SAFE-SRC-CHANGE]ログが、OVERTAKING中のみ・かつ前周期からの変化時のみ
    (既存[OT]ログの1Hz間引きに関わらず)発火することを確認する。"""
    idx = _SRC.index('"[V-SAFE-SRC-CHANGE]')
    snippet = _SRC[max(0, idx - 300):idx + 100]
    assert 'self._ot_state == "OVERTAKING"' in snippet
    assert "_v_safe_src != self._v_safe_src_prev" in snippet


def test_v_safe_src_prev_initialized_and_updated():
    """回帰: _v_safe_src_prevが__init__で初期化され、ログ判定の直後に必ず
    更新される(更新漏れがあると変化のたびに発火し続け、事実上1Hzログと
    変わらなくなる)ことを確認する。"""
    assert "self._v_safe_src_prev = None" in _SRC
    idx = _SRC.index('"[V-SAFE-SRC-CHANGE]')
    snippet = _SRC[idx:idx + 400]
    assert "self._v_safe_src_prev = _v_safe_src" in snippet
