"""Unit tests for issue⑤③: dlat_ttc_veto(ENGAGEゲート)の縦方向盲点への対処
(2026-07-22)。

背景: 0722-4/5(4台走行ログ)で、ENGAGEから0.02〜0.04秒でfootprint_risk
giveupに陥る事例を確認した(0722-4で44件中6件、0722-5のd3では13件中9件=
69%)。原因は_dlat_closing_trend()(is_closing_trendの計算式)がTTC
(fwd_dlat/|dlat_v_ema|)のみを見ており、その時点でfwd_ds(縦方向)が既に
footprint_risk相当の物理的接触リスク域(fwd_dlat<along_min_widthかつ
fwd_ds<along_min_length、127/163節で既に状態非依存・毎周期算出済み)に
入っているかどうかを一切見ていなかったこと。実測値を逆算すると、問題の
2件(fwd_ds=0.998m/1.957m)はいずれもfwd_dlatが0.65〜0.93m程度と算出でき、
along_min_width(1.45m)未満、すなわちENGAGEの瞬間footprint_riskは既に
Trueだった。

対処: 既存のfootprint_riskを_dlat_closing_trend()へ短絡条件として渡し、
is_closing_trend(ENGAGEゲート・G2-RELEASE・force_include_vidの3箇所が
共有)を「TTCトレンドが危険 “または” 既に物理的接触リスク域」という
単一の意味へ拡張した。新規パラメータ0個(既存footprint_riskを再利用)。
ENGAGEゲート側(_dlat_ttc_veto = opp_sit.is_closing_trend)は無変更。

一貫性検証(ユーザー指摘の深掘りを経て確定した設計):
①非矛盾性: 当初案は「OpponentSituationに新フィールドfootprint_riskを
追加しENGAGEゲート側だけでor」だったが、これだと同じ「相手が危険に近い」
という概念をis_closing_trendとfootprint_riskの2つの変数に分けたまま
持ち回ることになり、G2-RELEASE/force_include_vidは古い(狭い)定義の
ままになる非対称設計だった。footprint_risk=True時にG2-RELEASE解放を
控える・force_include_vidが対象に含めるのはいずれも安全側の変更である
ため、共有元の_dlat_closing_trend()自体を拡張する方が一貫性が高いと
判断した。
②非冗長性: footprint_riskは呼び出し元が_lat_dec確定前に既に算出済みの
値をそのまま渡すのみで、新規の距離判定式は追加していない。
③検証ロギング: [DLAT-TTC-VETO]ログにfootprint_risk=を追加し、TTCトレンド
起因かfootprint_risk起因かを次回ログから判別できるようにした。
④デッドロック不発生の確認: 161節のproactive_bias_side(STOPPING中の能動的
オフセット)はopp_sit.is_closing_trend/_dlat_ttc_vetoより前に計算される
_eval.plan_ok/plan_sideのみに依存するため、is_closing_trendが拡張されても
オフセットで間合いを広げる既存の脱出弁は動き続ける(163-5節の「cheap_ok
へ直接組み込む」デッドロック懸念とはスコープが異なる)。

mpc_controller.pyはrclpy依存のため直接importできないため、既存テストと
同じ方針(純Pythonミラー関数+ソーステキストによる構造的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

TTC_CRITICAL_S = 0.8
MIN_TREND_CYCLES = 3


def dlat_closing_trend(fwd_dlat, dlat_v_ema, dlat_shrink_run, footprint_risk=False,
                        min_trend_cycles=MIN_TREND_CYCLES, ttc_critical_s=TTC_CRITICAL_S):
    """_dlat_closing_trend()のミラー実装(issue⑤③、footprint_risk短絡追加)。"""
    if footprint_risk:
        return True
    if dlat_shrink_run < min_trend_cycles:
        return False
    if dlat_v_ema >= 0.0:
        return False
    if fwd_dlat is None:
        return False
    ttc = fwd_dlat / max(abs(dlat_v_ema), 1e-6)
    return ttc <= ttc_critical_s


# --- ①遡及効果: 0722-4/5実測値で修正前は見逃していたケースを検知できること ---

def test_retroactive_0722_04_d4_case_ttc_0_857_but_already_in_footprint_zone():
    """0722-4実測: fwd_ds=0.998m(along_min_length=2.00m未満)、
    dlat_v_ema=-1.090、fwd_dlat≈0.934m(TTC≈0.857秒、旧式では閾値0.8秒を
    わずかに上回り見逃していた)。footprint_risk=Trueを渡せば検知できる。"""
    ttc = 0.934 / 1.090
    assert ttc > 0.8  # 旧式は通過してしまっていたことの確認
    assert dlat_closing_trend(fwd_dlat=0.934, dlat_v_ema=-1.090, dlat_shrink_run=111,
                               footprint_risk=False) is False
    assert dlat_closing_trend(fwd_dlat=0.934, dlat_v_ema=-1.090, dlat_shrink_run=111,
                               footprint_risk=True) is True


def test_retroactive_0722_04_d4_case_ttc_0_887_but_already_in_footprint_zone():
    """0722-4実測: fwd_ds=1.957m、dlat_v_ema=-0.743、fwd_dlat≈0.659m
    (TTC≈0.887秒、旧式では見逃していた)。"""
    ttc = 0.659 / 0.743
    assert ttc > 0.8
    assert dlat_closing_trend(fwd_dlat=0.659, dlat_v_ema=-0.743, dlat_shrink_run=17,
                               footprint_risk=False) is False
    assert dlat_closing_trend(fwd_dlat=0.659, dlat_v_ema=-0.743, dlat_shrink_run=17,
                               footprint_risk=True) is True


def test_case_ttc_1_44_still_not_vetoed_even_with_footprint_risk_false():
    """0722-4実測(fwd_ds=1.99m、TTC≈1.44秒): footprint_risk自体がFalseの
    通常ケースでは従来通りvetoされないことを確認する(過検知していないか)。"""
    assert dlat_closing_trend(fwd_dlat=1.365, dlat_v_ema=-0.948, dlat_shrink_run=130,
                               footprint_risk=False) is False


# --- ②非冗長性: footprint_riskは既存の式を再利用するだけで新規距離判定を追加しない ---

def test_footprint_risk_short_circuits_regardless_of_trend_state():
    """footprint_risk=Trueなら、shrink_run不足・dlat_v_ema>=0(縮小トレンド
    不成立)であっても常にTrueを返すことを確認する(瞬時の物理的接触リスクは
    トレンド成立を待たない、という設計意図の確認)。"""
    assert dlat_closing_trend(fwd_dlat=0.5, dlat_v_ema=0.2, dlat_shrink_run=0,
                               footprint_risk=True) is True


def test_formula_unchanged_when_footprint_risk_false():
    """③非矛盾性: footprint_risk=False(既定値)の場合、旧141節の式と完全に
    同一の結果を返すことを確認する(退行検証)。"""
    assert dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=-0.5, dlat_shrink_run=5) is True
    assert dlat_closing_trend(fwd_dlat=3.0, dlat_v_ema=-0.5, dlat_shrink_run=5) is False
    assert dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=0.5, dlat_shrink_run=5) is False
    assert dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=-0.5, dlat_shrink_run=1) is False


# --- ③配線確認: ソーステキストによる構造検証 ---

def test_dlat_closing_trend_short_circuits_on_footprint_risk_first():
    idx = _SRC.index("def _dlat_closing_trend(")
    snippet = _SRC[idx:idx + 2000]
    idx_if = snippet.index("if footprint_risk:")
    idx_return_true = snippet.index("return True", idx_if)
    idx_trend_formula = snippet.index("dlat_shrink_run >= self._lat_ttc.min_trend_cycles")
    assert idx_if < idx_return_true < idx_trend_formula


def test_build_opponent_situation_passes_through_footprint_risk_no_new_computation():
    """②非冗長性: _build_opponent_situationはfootprint_riskをそのまま
    _dlat_closing_trendへ渡すだけで、新規の距離判定式(along_min_length等)を
    自前で計算しないことを確認する。"""
    idx = _SRC.index("def _build_opponent_situation(")
    idx_end = _SRC.index("def _evaluate_engage_readiness(")
    snippet = _SRC[idx:idx_end]
    assert "footprint_risk" in snippet
    assert "along_min_length" not in snippet
    assert "along_min_width" not in snippet


def test_call_site_passes_existing_footprint_risk_variable_not_new_one():
    """呼び出し元(_control())が、127/163節で既に算出済みの_footprint_riskを
    そのまま渡していることを確認する(新規の計算箇所を作らない)。"""
    assert "_opp_sit = self._build_opponent_situation(_scan, _lat_dec, _footprint_risk)" in _SRC


def test_engage_gate_formula_itself_unchanged():
    """①非矛盾性: ENGAGEゲート側(_dlat_ttc_veto)は今回無変更であることを
    確認する(共有元1箇所の拡張のみで3消費先が自動的に恩恵を受ける設計)。"""
    assert _SRC.count("_dlat_ttc_veto = opp_sit.is_closing_trend") == 1


def test_veto_log_distinguishes_footprint_risk_reason():
    """③検証ロギング: [DLAT-TTC-VETO]ログにfootprint_risk=を追加し、
    TTCトレンド起因かfootprint_risk起因かを次回ログから判別できることを
    確認する。"""
    idx = _SRC.index('f"[DLAT-TTC-VETO]')
    snippet = _SRC[idx:idx + 500]
    assert 'footprint_risk={int(footprint_risk)}' in snippet


def test_evaluate_engage_readiness_receives_footprint_risk_for_logging_only():
    """footprint_riskは_evaluate_engage_readinessの引数として渡されるが、
    _dlat_ttc_veto自体の計算式には使われず診断ログ専用であることを確認する。"""
    idx_def = _SRC.index("def _evaluate_engage_readiness(")
    sig_snippet = _SRC[idx_def:idx_def + 300]
    assert "footprint_risk" in sig_snippet
    idx_veto = _SRC.index("_dlat_ttc_veto = opp_sit.is_closing_trend")
    veto_line = _SRC[idx_veto:idx_veto + 60]
    assert "footprint_risk" not in veto_line


def test_call_site_of_evaluate_engage_readiness_passes_footprint_risk():
    assert ("_being_overtaken, _lat_dec, _opp_sit, now, _footprint_risk)"
            in _SRC)


# --- ④デッドロック不発生の確認(163-5節の懸念とスコープが異なることの構造検証) ---

def test_plan_ok_computed_before_dlat_ttc_veto_not_gated_by_it():
    """④遡及効果/デッドロック不発生: _plan_ok/_plan_side(161節proactive_bias_
    sideが依拠する)の計算が、_dlat_ttc_veto(footprint_riskを織り込んだ
    is_closing_trend)の計算より前に行われており、依存していないことを
    確認する。footprint_riskでENGAGEをvetoしても、オフセットで間合いを
    広げる既存の脱出弁(161節)は動き続けるため、163-5節が懸念した
    「cheap_okへ直接組み込む」場合の永久デッドロックとは異なる。"""
    idx_plan = _SRC.index("_plan_ok, _plan_side, _plan_req = self._plan_pass(")
    idx_veto = _SRC.index("_dlat_ttc_veto = opp_sit.is_closing_trend")
    assert idx_plan < idx_veto
