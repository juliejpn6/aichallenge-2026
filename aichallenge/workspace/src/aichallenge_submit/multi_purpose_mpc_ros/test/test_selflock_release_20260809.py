"""Unit tests for the footprint_risk self-lock release (2026-08-09, design_docs
opp_lat_pred_overlap_guard_design_20260806.md §45, task#300).

背景: footprint_risk由来のis_closing_trend(常時True)が、footprint_riskを
解消する側(=_plan_passが物理的に検証済みの空き側)へのENGAGEまでブロックし
続ける自己ロックを実測127件で確認した(footprint_risk giveup後30秒以内STUCK
190件中67%)。外部AIレビュー(Gemini・別Claude)の指摘を受け、以下の3層で
対処した:
  (1) ENGAGEゲート(_dlat_ttc_veto)の解除条件: configゲート・実トレンド非該当・
      V2Xクランプ非該当かつ鮮度化済み・予測post-offset dlatが閾値超過、の
      4条件AND
  (2) OVERTAKING遷移直後のforce_giveup(footprint_risk由来のみ)チャタリング対策:
      同一対象車・同一側・猶予周期内・dlat悪化なし、の条件が崩れたら即座に
      通常のforce_giveupへ復帰する有限のエスケープ猶予(Geminiレビュー指摘の
      「40Hzチャタリング」への対処)
  (3) 3箇所のOVERTAKING離脱経路すべてでエスケープ状態を確実にクリア

mpc_controller.pyはautoware_auto_control_msgs等をモジュールスコープでimport
しており単体テスト環境では直接importできないため、他の巨大メソッド関連
テストと同じくソーステキストに対する構造的検証を行う。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
_TRACKER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "v2x_vehicle_tracker.py")
_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()
with open(_TRACKER_PATH) as _f:
    _TRACKER_SRC = _f.read()
with open(_YAML_PATH) as _f:
    _YAML_SRC = _f.read()


# ---------------------------------------------------------------------------
# configゲート(must-fix 1): 既定false、退行防止
# ---------------------------------------------------------------------------

def test_config_gate_declared_enabled():
    # 2026-08-10有効化(タスク#300続報): 予選ログ0809-02のwp333で本事象そのものを
    # 実測(footprint_risk giveup 6回連続・約115秒ロック、ラップ5が183.5秒に伸びた
    # 直接原因)、design_docs opp_lat_pred_overlap_guard_design_20260806.md §49参照。
    # Phase3実地検証(dev3)は保留のまま先行してtrue化。
    assert "selflock_release_enabled: true" in _YAML_SRC


def test_state_vars_declared_and_default_safe():
    idx = _SRC.index("self._ot_selflock_release_enabled = bool(")
    snippet = _SRC[idx:idx + 900]
    for tok in ["_ot_selflock_escape_active = False",
                "_ot_selflock_escape_vid = None",
                "_ot_selflock_escape_side = 0",
                "_ot_selflock_escape_best_dlat = None",
                "_ot_selflock_escape_cycle = 0",
                "_ot_selflock_escape_cap_cycles = 0"]:
        assert tok in snippet, f"{tok} が宣言されていない"


def test_escape_cap_reuses_existing_t_lateral_no_new_magic_number():
    """新規マジックナンバー禁止(レビュー指摘): 猶予周期数は既存のt_lateral
    (横移動フェーズの想定所要時間)から算出し、独自の秒数定数を導入していない。"""
    idx = _SRC.index("self._ot_selflock_escape_cap_cycles = self._rate_scaled_cycles(")
    snippet = _SRC[idx:idx + 200]
    assert "self._ot_t_lateral * self._RATE_SCALE_REFERENCE_HZ" in snippet


# ---------------------------------------------------------------------------
# ENGAGEゲート解除条件(4条件AND): must-fix 1〜3を全て反映していること
# ---------------------------------------------------------------------------

def test_release_gated_by_config_flag():
    idx = _SRC.index("if (self._ot_selflock_release_enabled and _dlat_ttc_veto")
    assert idx > 0


def test_release_requires_real_trend_veto_false():
    idx = _SRC.index("if (self._ot_selflock_release_enabled and _dlat_ttc_veto")
    snippet = _SRC[idx:idx + 900]
    assert "_real_trend_veto = self._dlat_closing_trend(" in snippet
    assert "footprint_risk=False)" in snippet


def test_must_fix_2_v2x_clamp_and_settled_guard():
    """相手が停止扱いになるための条件に、V2Xクランプ非該当・鮮度化済みが
    必須条件として入っていることを確認する(V2X異常でfwd_vopp=0クランプされた
    走行中の相手を「停止」と誤認する穴への防御)。"""
    idx = _SRC.index("_v2x_trustworthy = True")
    snippet = _SRC[idx:idx + 500]
    assert "not self._v2x_tracker.is_speed_clamped(opp_sit.fwd_vid)" in snippet
    assert "self._v2x_tracker.is_settled(opp_sit.fwd_vid)" in snippet
    idx_stopped = _SRC.index("_opponent_stopped = (opp_sit.fwd_vopp is not None")
    snippet2 = _SRC[idx_stopped:idx_stopped + 200]
    assert "and _v2x_trustworthy)" in snippet2


def test_must_fix_3_predicted_dlat_reuses_room_to_wall_no_new_formula():
    """新規計算式禁止(レビュー指摘): 予測post-offset dlatは_plan_passと同一の
    _room_to_wallヘルパーを、相手の実測lat(scan["fwd_lat"])に対して再利用する
    だけで、独自の幾何計算を追加していない。閾値もalong_min_width+
    overlap_margin_m(いずれも既存定数)のみで新規定数は導入していない。"""
    idx = _SRC.index("_predicted_dlat = None")
    snippet = _SRC[idx:idx + 700]
    assert "self._room_to_wall(" in snippet
    assert "want_left=(_plan_side > 0)" in snippet
    assert ("self._along_min_width + self._ot_overlap_margin_m" in snippet)


def test_release_requires_plan_pass_valid_side():
    idx = _SRC.index("if (self._ot_selflock_release_enabled and _dlat_ttc_veto")
    snippet = _SRC[idx:idx + 200]
    assert "_plan_ok and _plan_side != 0" in snippet


def test_release_sets_escape_state_and_logs():
    idx = _SRC.index("if not _real_trend_veto and _opponent_stopped and _predicted_dlat_ok:")
    snippet = _SRC[idx:idx + 900]
    assert "_dlat_ttc_veto = False" in snippet
    assert "self._ot_selflock_escape_active = True" in snippet
    assert "self._ot_selflock_escape_vid = opp_sit.fwd_vid" in snippet
    assert "self._ot_selflock_escape_side = _plan_side" in snippet
    assert "self._ot_selflock_escape_cycle = 0" in snippet
    assert '"[DLAT-TTC-VETO-SELFLOCK-RELEASE]' in snippet


def test_g2_release_and_force_include_vid_untouched():
    """G2-RELEASE(_g2_release_ready)・force_include_vid(ICC近接除外)は
    このメソッド外のローカル計算のため無変更であることを確認する
    (共有元_dlat_closing_trend自体にも一切手を入れていない)。"""
    idx_def = _SRC.index("def _dlat_closing_trend(self, fwd_dlat")
    idx_def_end = _SRC.index("\n    def ", idx_def + 10)
    body = _SRC[idx_def:idx_def_end]
    assert "selflock" not in body.lower()
    assert "_ot_selflock" not in body


# ---------------------------------------------------------------------------
# OVERTAKING遷移直後のチャタリング対策(force_giveupエスケープ猶予)
# ---------------------------------------------------------------------------

def test_side_blocked_override_gated_by_all_five_conditions():
    idx = _SRC.index("_escape_ok = (")
    snippet = _SRC[idx:idx + 500]
    assert "_lat_dec.footprint_risk_triggered" in snippet
    assert "_opp_sit.fwd_vid == self._ot_selflock_escape_vid" in snippet
    assert "_locked == self._ot_selflock_escape_side" in snippet
    assert ("self._ot_selflock_escape_cycle\n"
            "                                < self._ot_selflock_escape_cap_cycles") in snippet
    assert "_lat_dec.dlat_v_ema >= 0.0" in snippet


def test_escape_abort_falls_back_to_normal_force_giveup_fail_closed():
    """条件が1つでも崩れたらフェイルクローズ(即座に通常のforce_giveupへ復帰、
    無期限の抑制はしない)ことを確認する。"""
    idx = _SRC.index("else:\n                            self.get_logger().info(\n"
                      "                                f\"[SELFLOCK-ESCAPE-ABORT]")
    snippet = _SRC[idx:idx + 700]
    assert "self._ot_selflock_escape_active = False" in snippet


def test_side_blocked_uses_override_only_when_true():
    idx = _SRC.index("_side_blocked = ((_lat_dec.force_giveup and not _selflock_escape_override)")
    snippet = _SRC[idx:idx + 150]
    assert "or _room_exhausted)" in snippet


def test_only_footprint_risk_triggered_giveups_are_overridable():
    """force_giveupの他の緊急経路(C2/C2_cleared等、footprint_risk_triggeredを
    伴わない)はこの機構の対象外であり、従来どおり即座に処理されることを
    footprint_risk_triggeredが必須条件であることから確認する(上記
    test_side_blocked_override_gated_by_all_five_conditionsと同じ箇所を
    別の観点[退行防止]で再確認)。"""
    idx = _SRC.index("_escape_ok = (")
    snippet = _SRC[idx:idx + 200]
    assert snippet.strip().startswith("_escape_ok = (\n") or "_escape_ok = (" in snippet
    assert "_lat_dec.footprint_risk_triggered" in _SRC[idx:idx + 300]


def test_escape_state_cleared_at_all_three_overtaking_exit_paths():
    """OVERTAKING離脱の3経路(giveup合流・通過完了[NORMAL復帰]・infeasible
    恒久失敗)すべてでエスケープ状態がクリアされ、次の無関係なENGAGEへ
    持ち越さないことを確認する。"""
    assert _SRC.count(
        "self._ot_selflock_escape_active = False  # 2026-08-09追加(§45.3)") == 2
    # giveup合流経路は専用コメント文言
    idx_giveup = _SRC.index(
        'self._log_ot_outcome(\n                            "giveup", self._ot_side,')
    giveup_snippet = _SRC[idx_giveup:idx_giveup + 700]
    assert "self._ot_selflock_escape_active = False" in giveup_snippet


# ---------------------------------------------------------------------------
# v2x_vehicle_tracker.py: クランプ検出フラグ(must-fix 2の基盤)
# ---------------------------------------------------------------------------

def test_tracker_clamped_state_declared():
    assert "self._clamped: Dict[str, bool] = {}" in _TRACKER_SRC


def test_tracker_sets_clamped_true_on_clamp_branch():
    idx = _TRACKER_SRC.index("if math.hypot(vx, vy) > self._v_max_safety:")
    snippet = _TRACKER_SRC[idx:idx + 300]
    assert "self._clamped[vid] = True" in snippet


def test_tracker_sets_clamped_false_on_valid_branch():
    idx = _TRACKER_SRC.index("self._last_valid_velocity[vid] = (vx, vy)")
    snippet = _TRACKER_SRC[idx:idx + 200]
    assert "self._clamped[vid] = False" in snippet


def test_is_speed_clamped_query_method_exists_and_defaults_false():
    idx = _TRACKER_SRC.index("def is_speed_clamped(self, vehicle_id: str) -> bool:")
    snippet = _TRACKER_SRC[idx:idx + 300]
    assert "self._clamped.get(vehicle_id, False)" in snippet


def test_is_speed_clamped_works_regardless_of_clamp_hold_enabled():
    """clamp_hold_enabledの設定に関わらずクランプ検出ができること(clamp_hold
    無効[既定]でも、その周期の速度がクランプされたという事実自体は判定できる)。"""
    idx_flag = _TRACKER_SRC.index("self._clamped[vid] = True")
    idx_hold_check = _TRACKER_SRC.index("if not self._clamp_hold_enabled")
    # クランプ検出フラグのセット自体はclamp_hold_enabledの分岐より前(=無条件)
    assert idx_flag < idx_hold_check
