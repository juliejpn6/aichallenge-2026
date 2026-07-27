"""Unit tests for footprint_risk-scaled engage_cooldown (138-5節②、2026-07-20).

Background: 0720-04実測(wp240-243)で、完全停止した相手車の狭所にegoが
3回以上ENGAGEを試み、いずれもfootprint_riskで0.5〜1秒以内に断念する往復を
約9秒間繰り返していた。_plan_passの静的room計算(デバウンス込み)は
「わずかに間に合う」と判定するが、footprint_risk(実測ベース)は毎回すぐに
危険と判定しており、両者の認識がズレたまま即座に再試行していたことが原因。

対処: footprint_risk起因のgiveupの場合のみ、既存engage_cooldown_cyclesを
2倍にする(92節①で確立済みの「min_trend_cycles*2」という既存の倍化パターンを
踏襲、新規パラメータ0個)。相手が速すぎる等の他のgiveup理由は従来通りの
長さのまま。

このロジックは複雑度が高くモック実行が難しいため、既存の同種テストと同じく
ミラー実装+ソーステキスト構造検証で確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _cooldown_after_giveup(footprint_risk_triggered, base_cooldown_cycles=80):
    """mpc_controller.pyのcooldown代入式のミラー実装。"""
    return base_cooldown_cycles * 2 if footprint_risk_triggered else base_cooldown_cycles


def test_footprint_risk_giveup_doubles_cooldown():
    assert _cooldown_after_giveup(footprint_risk_triggered=True, base_cooldown_cycles=80) == 160


def test_other_giveup_reasons_keep_default_cooldown_no_regression():
    """回帰防止: 相手が速すぎる等、footprint_risk以外の理由によるgiveupは
    従来通りのcooldownのまま(不要に待たせない)。"""
    assert _cooldown_after_giveup(footprint_risk_triggered=False, base_cooldown_cycles=80) == 80


def test_scaling_respects_configured_base_value():
    """config.yamlでengage_cooldownが変更された場合も、倍化は相対的に
    追従することを確認する(ハードコード値ではない)。"""
    assert _cooldown_after_giveup(footprint_risk_triggered=True, base_cooldown_cycles=40) == 80
    assert _cooldown_after_giveup(footprint_risk_triggered=False, base_cooldown_cycles=40) == 40


# --- ソーステキスト構造検証 ---

def test_cooldown_scaling_reuses_footprint_risk_triggered_flag():
    """②非冗長性: 新規の判定フラグを追加せず、既存_lat_dec.footprint_risk_triggered
    (128節で導入済み)をそのまま再利用していることを確認する。"""
    idx = _SRC.index("self._ot_engage_cooldown = (")
    snippet = _SRC[idx:idx + 250]
    assert "self._ot_engage_cooldown_cycles * 2" in snippet
    assert "_lat_dec.footprint_risk_triggered" in snippet
    assert "else self._ot_engage_cooldown_cycles" in snippet


def test_cooldown_scaling_uses_established_doubling_pattern():
    """②非冗長性: 92節①(lateral_ttc_monitor.pyのmin_trend_cycles*2)で確立
    済みの「既存値 * 2」という倍化パターンを踏襲しており、新規のスケール
    係数を発明していないことを確認する。"""
    _lat_ttc_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "lateral_ttc_monitor.py")
    with open(_lat_ttc_path) as _f:
        _lat_ttc_src = _f.read()
    assert "min_trend_cycles * 2" in _lat_ttc_src  # 92節①の既存パターン
    assert "self._ot_engage_cooldown_cycles * 2" in _SRC


def test_cooldown_assignment_is_single_site_no_duplication():
    """①非矛盾性: cooldown設定箇所が1箇所のみであり(131節の第1案のように
    複数箇所に同じロジックを重複実装していない)ことを確認する。"""
    assert _SRC.count("self._ot_engage_cooldown = (") == 1


def test_cooldown_scaling_applied_at_the_only_giveup_to_stopping_transition():
    """cooldown代入が、既存のgiveup→STOPPING遷移ブロック(_ot_state="STOPPING"
    への代入と同じ箇所)にあることを確認する。"""
    idx_state = _SRC.index('self._ot_state = "STOPPING"')
    idx_cooldown = _SRC.index("self._ot_engage_cooldown = (")
    idx_cleared = _SRC.index("self._ot_cleared = False", idx_cooldown)
    # cooldown代入がstate="STOPPING"より後、かつ同じ遷移ブロックの
    # 締めくくり(_ot_cleared=False)より前にあることを確認する。
    assert idx_state < idx_cooldown < idx_cleared
