"""Fix C(並走中の離脱保留、2026-08-07)。

背景: design_docs/opp_lat_pred_overlap_guard_design_20260806.md §3。
giveup条件が成立した瞬間、`_ot_side`を即座に0へスナップして
`_reset_ot_offset_state()`で`lateral_target=0.0`まで即時ゼロ化する現行
実装は、並走中に発火すると相手へ向けて横に引き戻す形になり衝突を招く
(18節の事例)。並走中の非緊急giveupは、縦方向の車間が空くまで離脱を
保留する。

footprint_risk(緊急反応系トリガー)はここで除外し、現行どおり即座に
処理する(82/83節の教訓、CLAUDE.md §1.3の慎重領域=cleared判定周りへの
安易なガード追加は厳禁、安全反応系の遅延は厳禁)。

安全弁(必須): `_ot_pending_disengage_count`が
`_ot_pending_disengage_max_cycles`(既定80周期≈2秒、既存giveup_cyclesの
2倍)に達したら、並走が解消していなくても強制的に通常のgiveup処理へ
合流する(無期限保留を禁止)。

configゲート`overtake.pending_disengage_enabled`(既定false)でON/OFF
切替、OFF時は`_giveup_now`の値をそのまま使う=現行動作とビット等価
(段階導入、Fix A/Bとは独立ゲート)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと
同じ「ソーステキスト構造検証」+「ロジックのミラー実装による数値検証」の
方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG_SRC = _f.read()


def _giveup_block():
    """2026-08-09改訂(§45.3): 開始アンカーが`_side_blocked = _lat_dec.force_giveup
    or _room_exhausted`から`_side_blocked = ((_lat_dec.force_giveup and not
    _selflock_escape_override) or _room_exhausted)`へ変わった(自己ロック解除
    エスケープ、configゲート既定OFF時はビット等価)。"""
    idx = _SRC.index("_side_blocked = ((_lat_dec.force_giveup and not _selflock_escape_override)")
    idx_end = _SRC.index("else:\n                        self._ot_side = _locked", idx)
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①config/状態変数
# ---------------------------------------------------------------------------

def test_config_yaml_has_pending_disengage_keys():
    assert "pending_disengage_enabled:" in _CFG_SRC
    assert "pending_disengage_max_cycles:" in _CFG_SRC


def test_state_vars_declared_with_safe_defaults():
    assert 'self._ot_pending_disengage_enabled = bool(\n' in _SRC
    assert '_otget("pending_disengage_enabled", False))' in _SRC
    assert "self._ot_pending_disengage_count = 0" in _SRC


def test_max_cycles_derived_from_giveup_cycles_default_no_new_scale():
    """新規config既定値(80)がgiveup_cycles(40)の2倍という既存スケール感覚を
    踏襲していることを確認する(design_docs §3.3)。"""
    assert "pending_disengage_max_cycles: 80" in _CFG_SRC
    assert "giveup_cycles: 40" in _CFG_SRC


# ---------------------------------------------------------------------------
# ②_giveup_now変数化: 既存のif条件がそのまま_giveup_nowへ代入されていること
# ---------------------------------------------------------------------------

def test_giveup_now_computed_from_original_condition_unchanged():
    snippet = _giveup_block()
    assert "_giveup_now = (self._ot_giveup_count >= self._ot_giveup_cycles" in snippet
    assert "or _locked == 0 or _side_blocked)" in snippet


def test_giveup_block_entry_guarded_by_giveup_now():
    snippet = _giveup_block()
    assert "if _giveup_now:" in snippet
    assert "self._ot_pending_disengage_count = 0" in snippet


# ---------------------------------------------------------------------------
# ③ゲート構造: enabled かつ giveup_now かつ footprint_risk以外の時のみ
#   Fix Cロジックが介入する
# ---------------------------------------------------------------------------

def test_fix_c_gated_by_enabled_giveup_now_and_not_footprint_risk():
    snippet = _giveup_block()
    idx = snippet.index("if (self._ot_pending_disengage_enabled and _giveup_now")
    idx_end = snippet.index("):", idx)
    cond = snippet[idx:idx_end]
    assert "self._ot_pending_disengage_enabled" in cond
    assert "_giveup_now" in cond
    assert "not _lat_dec.footprint_risk_triggered" in cond


def test_footprint_risk_excluded_emergency_path_untouched():
    """82/83節の教訓: footprint_risk等の緊急系は保留ロジックの対象外とし、
    現行どおり即座に処理されることを確認する。"""
    snippet = _giveup_block()
    idx = snippet.index("if (self._ot_pending_disengage_enabled and _giveup_now")
    idx_end = snippet.index("):", idx)
    assert "footprint_risk_triggered" in snippet[idx:idx_end]


def test_force_giveup_excluded_emergency_path_untouched():
    """2026-08-07修正(統合整合性レビュー、外部AI[別Claude]指摘): force_giveup
    (lateral_ttc_monitor.pyのLAT-TTC C2/C2_cleared分岐、「最終防波堤」)も
    footprint_risk_triggeredと同様に緊急系であり、保留ロジックの対象外と
    すること。設計書§3.2は当初から「緊急系=footprint_risk・force_giveup
    由来」と明記していたが、実装がfootprint_risk_triggeredのみを見ていた
    欠陥の再発防止テスト。C2/C2_cleared分岐はforce_giveup=Trueを返す際
    footprint_risk_triggeredを伴わない(lateral_ttc_monitor.py:861-862、
    デフォルトのFalseのまま)ため、force_giveupの除外がなければ緊急giveup
    が誤って保留されうる。"""
    snippet = _giveup_block()
    idx = snippet.index("if (self._ot_pending_disengage_enabled and _giveup_now")
    idx_end = snippet.index("):", idx)
    cond = snippet[idx:idx_end]
    assert "not _lat_dec.force_giveup" in cond


def test_overlap_check_reuses_update_overlap_state_and_fwd_ds():
    snippet = _giveup_block()
    assert "self._update_overlap_state(_opp_sit.fwd_ds)" in snippet


# ---------------------------------------------------------------------------
# ④must-fix 2(外部AIレビュー): giveup条件自体が不成立の周期は
#   保留カウントを必ず0へ戻す(以前のエピソードの残存カウントを
#   次の無関係なgiveupへ引き継がない)
# ---------------------------------------------------------------------------

def test_must_fix_2_resets_count_when_giveup_not_triggered_from_start():
    snippet = _giveup_block()
    idx_if = snippet.index("if (self._ot_pending_disengage_enabled and _giveup_now")
    idx_else = snippet.index("\n                    else:\n", idx_if)
    idx_giveup_now_check = snippet.index("if _giveup_now:", idx_else)
    else_body = snippet[idx_else:idx_giveup_now_check]
    assert "self._ot_pending_disengage_count = 0" in else_body


# ---------------------------------------------------------------------------
# ⑤安全弁(必須): 上限到達で強制合流、無期限保留の禁止
# ---------------------------------------------------------------------------

def test_safety_valve_forces_fallback_at_max_cycles():
    snippet = _giveup_block()
    assert (
        "if (self._ot_pending_disengage_count\n"
        "                                    < self._ot_pending_disengage_max_cycles):"
        in snippet)
    assert "_giveup_now = False" in snippet


def test_forced_fallback_logs_distinct_reason():
    snippet = _giveup_block()
    assert '"[PENDING-DISENGAGE] resolved reason=forced_fallback ' in snippet


# ---------------------------------------------------------------------------
# ⑥診断ログ: 保留開始・自然解消の2イベント
# ---------------------------------------------------------------------------

def test_start_log_present():
    assert '"[PENDING-DISENGAGE] start side=' in _SRC


def test_natural_clear_log_present():
    assert '"[PENDING-DISENGAGE] resolved reason=natural_overlap_clear ' in _SRC


def test_start_log_edge_triggered_only_first_cycle():
    """[PENDING-DISENGAGE] startはエピソード内で1回だけ(count==1の周期のみ)
    発火することを確認する(毎周期のログ氾濫を避ける、既存パターン踏襲)。"""
    snippet = _giveup_block()
    idx = snippet.index('"[PENDING-DISENGAGE] start side=')
    preceding = snippet[:idx]
    assert "if self._ot_pending_disengage_count == 1:" in preceding[-160:]


# ---------------------------------------------------------------------------
# ⑦状態機械への影響確認(design_docs §3.4)
# ---------------------------------------------------------------------------

def test_ot_state_unchanged_new_state_value_not_introduced():
    """_ot_stateへ新規state値("PENDING"等)を追加していないことを確認する
    (既存3値NORMAL/STOPPING/OVERTAKINGのみ維持)。"""
    snippet = _giveup_block()
    assert 'self._ot_state = "PENDING"' not in snippet
    assert 'self._ot_state = "HOLD"' not in snippet


def test_stuck_entry_resets_episode_tracking_state():
    """STUCK突入時(_stuck_enter_wait_reverse)にも
    _reset_ot_episode_tracking_state()を呼び、保留カウント・並走状態・床を
    持ち越さないことを確認する(外部AIレビュー推奨4)。"""
    idx = _SRC.index("def _stuck_enter_wait_reverse(self")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._reset_ot_episode_tracking_state()" in snippet
    idx_reset = snippet.index("self._reset_ot_episode_tracking_state()")
    idx_recovery = snippet.index("self._handle_stuck_recovery(now, pose)")
    assert idx_reset < idx_recovery  # _handle_stuck_recovery呼び出しより前


def test_no_fwd_vid_switch_path_without_reengage():
    """OVERTAKING継続のままfwd_vid(対象車ID)だけが切り替わる経路が
    存在しないことを確認する(外部AIレビュー推奨7)。_ot_target_vidの
    代入箇所が__init__+新規エンゲージ時の2箇所のみであれば、そのような
    経路は存在しない(新規エンゲージは既に_reset_ot_episode_tracking_state()
    を呼んでいるためFix Cの状態も正しくリセットされる)。"""
    n = _SRC.count("self._ot_target_vid = ")
    assert n == 2, (
        f"_ot_target_vidの代入箇所が2箇所から変わっている(現在{n}箇所)。"
        "新しい代入経路が追加された場合、そこにも_reset_ot_episode_tracking_state()"
        "の追加を検討する必要がある。")


# ---------------------------------------------------------------------------
# ⑧ミラー数値検証: 保留ロジックのコア計算
# ---------------------------------------------------------------------------

def _pending_disengage_mirror(giveup_now, enabled, footprint_risk, overlapping,
                               count, max_cycles=80, force_giveup=False):
    """giveup分岐冒頭に挿入したFix Cロジックの1:1ミラー(数値検証用)。
    2026-08-07改訂(統合整合性レビュー、外部AI[別Claude]指摘): force_giveup
    (LAT-TTC C2/C2_cleared「最終防波堤」)もfootprint_riskと同様に緊急系
    として除外する。戻り値: (giveup_now_after, count_after)"""
    if enabled and giveup_now and not footprint_risk and not force_giveup:
        if overlapping:
            count += 1
            if count < max_cycles:
                giveup_now = False
        else:
            count = 0
    else:
        count = 0
    if giveup_now:
        count = 0
    return giveup_now, count


def test_mirror_gate_off_bit_equivalent():
    """ゲートOFF時はgiveup_nowの値をそのまま使う(現行動作とビット等価)。"""
    for giveup_now in (True, False):
        g_after, _ = _pending_disengage_mirror(
            giveup_now, enabled=False, footprint_risk=False, overlapping=True, count=5)
        assert g_after == giveup_now


def test_mirror_holds_while_overlapping_and_under_limit():
    g_after, count_after = _pending_disengage_mirror(
        True, enabled=True, footprint_risk=False, overlapping=True, count=10, max_cycles=80)
    assert g_after is False  # 保留(OVERTAKING継続)
    assert count_after == 11


def test_mirror_footprint_risk_bypasses_hold_immediate_giveup():
    """footprint_risk起因のgiveupは保留の対象外、即座に処理される
    (giveup_nowがTrueのまま)。"""
    g_after, count_after = _pending_disengage_mirror(
        True, enabled=True, footprint_risk=True, overlapping=True, count=0, max_cycles=80)
    assert g_after is True
    assert count_after == 0  # if _giveup_now: の最終リセットが効く


def test_mirror_force_giveup_bypasses_hold_immediate_giveup():
    """2026-08-07修正(統合整合性レビュー、外部AI[別Claude]指摘): force_giveup
    (LAT-TTC C2/C2_cleared「最終防波堤」、footprint_risk_triggeredを伴わない)
    起因のgiveupも保留の対象外、即座に処理されることを確認する。"""
    g_after, count_after = _pending_disengage_mirror(
        True, enabled=True, footprint_risk=False, force_giveup=True,
        overlapping=True, count=0, max_cycles=80)
    assert g_after is True
    assert count_after == 0


def test_mirror_natural_clear_resets_count_and_proceeds_with_giveup():
    """並走が解消(overlapping=False)すれば保留カウントは0へ戻り、
    giveup_nowはTrueのまま(通常のgiveup処理が進む)。"""
    g_after, count_after = _pending_disengage_mirror(
        True, enabled=True, footprint_risk=False, overlapping=False, count=15, max_cycles=80)
    assert g_after is True
    assert count_after == 0


def test_mirror_forced_fallback_at_max_cycles():
    """並走中止まないまま保留カウントがmax_cyclesへ到達すると、強制的に
    giveup_nowがTrueへ戻る(無期限保留の禁止)。"""
    g_after, count_after = _pending_disengage_mirror(
        True, enabled=True, footprint_risk=False, overlapping=True, count=79, max_cycles=80)
    assert g_after is True  # count=80に到達、80<80が偽なのでholdしない
    assert count_after == 0


def test_mirror_giveup_not_triggered_from_start_resets_count():
    """must-fix 2: giveup条件自体が不成立(giveup_now=False)の周期は、
    残存カウントを必ず0へ戻す。"""
    g_after, count_after = _pending_disengage_mirror(
        False, enabled=True, footprint_risk=False, overlapping=True, count=5, max_cycles=80)
    assert g_after is False
    assert count_after == 0
