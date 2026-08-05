"""engage_cooldown早期解除(①相手が速すぎる・③room_exhausted)の実装
(2026-08-05)。

背景: ユーザー指摘「4秒何もしないのはロスになるときがある」。148節②で確立済みの
footprint_risk専用「実測解消」パターン(固定タイマーではなくfootprint_risk条件
自体が解消したかで判定)を、他のgiveup理由へ横展開した。外部AI(Gemini・別Claude
インスタンス)への相談を経て以下の設計で確定:

- 値ヒステリシス必須(同一閾値の往復はデバウンスだけでは防げない、コーナー/直線で
  接近速度が数秒周期で振動する構造的リスクを指摘): ①解除条件は
  opp_giveup_closingの2倍(対称的マージン)、③はalong_min_width(既存の幅マージン
  定数、二重管理を避け再利用)。
- デバウンスは「対称性の原則」: giveup自体がself._ot_giveup_cycles(40周期≈1秒)
  悪い状態の継続で発火するなら、解除も同じ周期数の良い状態の継続を要求する
  (新規パラメータ0個)。
- ①には既知のV2X速度クランプ異常(fwd_vopp=0誤認、
  v2x_anomaly_defense_gap_review_20260803.md)への防御ガードが必須。
  fwd_vopp<=0.0の周期は実測解消判定をスキップする。
- ③はgiveup時点で記録済みの永続変数self._ot_prev_side(既存)を使う(_lockedは
  giveup発生周期のローカル変数のため以降の周期では参照できない)。
- ②(ロック外れ)は実測解消経路を設けず固定タイマーのみ(両AI一致)。
- force_giveup起因(room_exhausted以外のlat_ttc branch)も③の対象外(corr_bound_
  aheadの回復が必ずしも解消を意味しない、安全側)。
- cooldown_per_side(#265)とは独立のconfig gate(1変更1検証、両方同時ONの
  相互作用は未検証、design_docsに明記)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①状態変数・config: 新規変数が既定値付きで宣言されていること
# ---------------------------------------------------------------------------

def test_new_state_variables_declared_with_safe_defaults():
    assert "self._ot_early_release_enable = bool(\n" \
           "                _otget(\"early_release_enable\", False))" in _SRC
    assert "self._ot_speed_gated = False" in _SRC
    assert "self._ot_speed_recover_count = 0" in _SRC
    assert "self._ot_room_gated = False" in _SRC
    assert "self._ot_room_recover_count = 0" in _SRC


def test_config_yaml_has_early_release_enable_default_false():
    _cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(_cfg_path) as f:
        cfg = f.read()
    assert "early_release_enable: false" in cfg


# ---------------------------------------------------------------------------
# ②毎周期の回復判定: gate ON時のみ計算、ヒステリシス・ガードを含む
# ---------------------------------------------------------------------------

def _recover_block():
    idx = _SRC.index("if self._ot_early_release_enable:")
    idx_end = _SRC.index("# 攻めの価値判定", idx)
    return _SRC[idx:idx_end]


def test_speed_recovery_uses_hysteresis_margin():
    """①の解除条件がopp_giveup_closingそのものではなく2倍(ヒステリシス)で
    あることを確認する(同一閾値往復によるチャタリング防止)。"""
    snippet = _recover_block()
    assert "self._opp_giveup_closing * 2.0" in snippet


def test_speed_recovery_guards_against_vopp_clamp_anomaly():
    """既知のV2X速度クランプ異常(fwd_vopp=0誤認)への防御ガードが
    含まれていることを確認する。"""
    snippet = _recover_block()
    assert "_fwd_vopp is not None and _fwd_vopp > 0.0" in snippet


def test_room_recovery_uses_along_min_width_margin():
    """③の解除条件が単純な>0.0ではなく、既存の幅マージン定数
    (along_min_width)を再利用していることを確認する。"""
    snippet = _recover_block()
    assert "_room_bound >= self._along_min_width" in snippet


def test_room_recovery_uses_persisted_prev_side_not_local_locked():
    """③はgiveup時点で記録済みの永続変数self._ot_prev_sideを使うこと
    (_lockedはgiveup発生周期限りのローカル変数のため使えない)。"""
    snippet = _recover_block()
    assert "self._corr_bound_ahead(self._ot_prev_side)" in snippet
    assert "self._ot_prev_side != 0" in snippet


def test_recover_counts_only_updated_when_respective_gate_flag_set():
    snippet = _recover_block()
    assert "if self._ot_speed_gated:" in snippet
    assert "if self._ot_room_gated:" in snippet


# ---------------------------------------------------------------------------
# ③giveup時のgatedフラグ設定: 優先順位(footprint_risk > room_exhausted > 速度)
# ---------------------------------------------------------------------------

def test_gated_flags_set_with_correct_priority_on_giveup():
    idx = _SRC.index("self._ot_speed_gated = (")
    idx_end = _SRC.index("self._ot_cleared = False", idx)
    snippet = _SRC[idx:idx_end]
    assert "not _lat_dec.footprint_risk_triggered" in snippet
    assert "not _room_exhausted" in snippet
    assert "self._ot_giveup_count >= self._ot_giveup_cycles" in snippet
    assert "self._ot_room_gated = (" in snippet
    assert "self._ot_speed_recover_count = 0" in snippet
    assert "self._ot_room_recover_count = 0" in snippet


# ---------------------------------------------------------------------------
# ④_cd_clear計算: footprint_riskと同型のOR構造、末端に固定タイマーフォールバック
# ---------------------------------------------------------------------------

def test_cd_clear_includes_speed_and_room_early_release_branches():
    idx = _SRC.index("_cd_clear = (")
    snippet = _SRC[idx:idx + 900]
    assert "self._ot_speed_recover_count >= self._ot_giveup_cycles" in snippet
    assert "if self._ot_speed_gated" in snippet
    assert "self._ot_room_recover_count >= self._ot_giveup_cycles" in snippet
    assert "if self._ot_room_gated" in snippet
    # 末端は必ず固定タイマーへ収束すること(gate OFF時の退行防止)
    assert snippet.count("self._ot_engage_cooldown == 0") >= 3
