"""Unit tests for Phase2 (143節続報, 2026-07-20): wiring OpponentSituation's
is_closing_trend into the ICC near_sep exclusion (force_include_vid) and into
G2-RELEASE's release decision — the actual fix for P0①(第二コーナー衝突).

背景: 0720-05実測(wp139-141)で、ICCのnear_sep(1.8m)静的ゲートが対象車d3を
除外し_vlim=None→eff_v_cap(前車なし、無制限速度)へ抜けた瞬間、同じd3を
[LAT-TTC]は横方向closingトレンド(dlat_v_ema=-0.87〜-1.54)として継続追跡して
おり、単一サイクルでv=4.18→2.02m/sの実質衝突が発生した。

143節続報のスリム化点検で、G2-RELEASE(側方確保済みでの解放判定)も同じ盲点
(静的な瞬時値のみでtrend非考慮)を持つことが判明したため、フェーズ2は
①force_include_vid(93節の既存機構)へclosingトレンド対象車を追加
②_g2_release_readyの解放判定にもis_closing_trendのANDガードを追加、の2箇所を
同時に塞ぐ設計とした。両方とも新規パラメータ0個、既存のOpponentSituation
(141-143節フェーズ1)・force_include_vid(93節)を再利用するのみ。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _force_include_vid_snippet():
    """2026-07-26更新(190-5節): is_closing_trend起因の条件式自体は
    `_force_include_via_trend`という中間変数へ抽出された(診断ロギング追加の
    ための変更、条件式の意味・評価順序は無変更)。このヘルパーはその抽出元
    から`_force_include_vid`本体の代入までを1つの区間としてまとめて返す。"""
    idx = _SRC.index("_force_include_via_trend = (")
    idx_end = _SRC.index("_vlim, _vtgt = self._follow_speed_limit(")
    return _SRC[idx:idx_end]


# --- ①force_include_vidの拡張 ---

def test_force_include_vid_extended_with_closing_trend_or_condition():
    """①非矛盾性: switchback-alpha救済(93節)とclosingトレンド救済(143節続報)が
    OR条件で共存し、既存経路を壊さずに新条件が追加されていることを確認する。"""
    snippet = _force_include_vid_snippet()
    assert "self._ot_side != 0 and self._ot_alpha < 1.0 - 1e-3" in snippet
    assert "_opp_sit.is_closing_trend" in snippet
    assert " or " in snippet


def test_switchback_alpha_rescue_stays_scoped_to_overtaking_only():
    """①非矛盾性: switchback-alpha救済(_ot_side/_ot_alpha、OVERTAKING固有の概念)は
    145節続報(STOPPING拡張)後もOVERTAKING限定のままであることを確認する
    (STOPPING中はside/alphaという概念自体が存在しないため)。"""
    snippet = _force_include_vid_snippet()
    idx_side = snippet.index("self._ot_side != 0 and self._ot_alpha < 1.0 - 1e-3")
    idx_state = snippet.rindex('self._ot_state == "OVERTAKING"', 0, idx_side)
    assert idx_state < idx_side
    assert 'self._ot_state == "OVERTAKING"' in snippet[:idx_side]


def test_force_include_vid_uses_same_source_vid_as_switchback_case():
    """②非冗長性: closingトレンド救済も既存のswitchback救済と同じ
    scan.get("fwd_vid")をそのまま使い、別の対象車IDソースを持ち込まないことを
    確認する(_opp_sit.fwd_vidとscan["fwd_vid"]は同一サイクルの同一値)。"""
    snippet = _force_include_vid_snippet()
    assert snippet.count('_scan.get("fwd_vid")') == 1


# ---------------------------------------------------------------------------
# 145節続報(フェーズ3①): STOPPING側の同型盲点の対処
# ---------------------------------------------------------------------------
# 背景: 144節のスリム化点検で、STOPPING側のicc_stop(_vlimを直接使用)もOVERTAKING
# と全く同じnear_sep静的ゲートを共有する「統一ICC」であるにもかかわらず、
# is_closing_trend救済がOVERTAKING状態限定のままだったことが判明した。
# icc_stop_fallback/STOPPING-NO-VSAFEブリッジという段階的な保険はあるが、
# それぞれnear_sep(1.8m)より更に狭い窓(engage_lat_max=2.0m/along_min_length=2.0m
# の縦距離条件込み)でしか対象車を捕捉できず、trend自体は見ていない。

def test_is_closing_trend_rescue_extended_to_stopping_state():
    """①非矛盾性の核心: is_closing_trend救済がOVERTAKINGだけでなくSTOPPINGでも
    適用されることを確認する(_vlimはOVERTAKING/STOPPING共通の「統一ICC」で
    あるため、同じ救済を同じ理由で適用するのが一貫している)。"""
    snippet = _force_include_vid_snippet()
    assert 'self._ot_state in ("OVERTAKING", "STOPPING")' in snippet
    idx_in = snippet.index('self._ot_state in ("OVERTAKING", "STOPPING")')
    idx_trend = snippet.index("_opp_sit.is_closing_trend")
    assert idx_in < idx_trend


def test_is_closing_trend_rescue_does_not_extend_to_normal_state():
    """②非冗長性: NORMAL状態はicc_stop/eff_v_cap等いずれもvlimを参照しない
    (04節の候補選択で「候補なし(全開)」のみ)ため、is_closing_trend救済を
    NORMALへ拡張する意味が無い。in ("OVERTAKING", "STOPPING")の2値のみで
    あることを確認する。"""
    snippet = _force_include_vid_snippet()
    idx = snippet.index('self._ot_state in (')
    line_end = snippet.index(")", idx)
    assert "NORMAL" not in snippet[idx:line_end + 1]


# --- ②_g2_release_readyへのトレンドガード ---

def test_g2_release_ready_accepts_is_closing_trend_param():
    idx = _SRC.index("def _g2_release_ready(self, scan, fwd_vopp, vtgt, left_free, right_free,")
    snippet = _SRC[idx:idx + 200]
    assert "is_closing_trend: bool = False" in snippet


def test_g2_release_final_decision_blocked_by_closing_trend():
    """④過去ログへの遡及効果に相当: is_closing_trend=Trueの間、デバウンスが
    既に完了していても(_g2_release_debounced=True)最終的な解放は起きない
    ことを確認する(P0①のシナリオでは、この2番目の分岐も無制限速度へ抜ける
    経路になり得たため、①だけでなくここも塞ぐ必要があった)。"""
    idx = _SRC.index(
        "_side_clear = self._g2_release_debounced and not is_closing_trend")
    assert idx > 0


def test_debounce_counter_unaffected_by_closing_trend():
    """①非矛盾性: is_closing_trendはデバウンスカウンタの積算自体には影響せず
    (_side_clear_rawの計算にis_closing_trendは含まれない)、最終判定にのみ
    ANDで効くことを確認する(トレンドが収まった瞬間に再デバウンス待ちが
    発生しない設計、既存の「解放は緩やか・制限は即時」という非対称設計を踏襲)。"""
    idx = _SRC.index("_side_clear_raw = (")
    idx_end = _SRC.index("if _side_clear_raw:")
    snippet = _SRC[idx:idx_end]
    assert "is_closing_trend" not in snippet


def test_call_site_passes_opponent_situation_closing_trend():
    idx = _SRC.index("elif self._g2_release_ready(")
    snippet = _SRC[idx:idx + 400]
    assert "is_closing_trend=_opp_sit.is_closing_trend" in snippet


def test_g2_release_log_includes_closing_trend_for_diagnosability():
    idx = _SRC.index('f"[G2-RELEASE]')
    snippet = _SRC[idx:idx + 900]
    assert "is_closing_trend={is_closing_trend}" in snippet


# --- ④過去ログへの遡及検証(ミラー実装) ---

def _g2_release_final(side_clear_debounced, is_closing_trend):
    return side_clear_debounced and not is_closing_trend


def test_retroactive_p0_1_scenario_blocked():
    """0720-05 wp139相当: side_room等の静的条件は満たされ得るが(デバウンス完了と
    仮定)、is_closing_trend=Trueなら解放されないことを確認する。"""
    assert _g2_release_final(side_clear_debounced=True, is_closing_trend=True) is False


def test_normal_release_unaffected_when_not_closing():
    assert _g2_release_final(side_clear_debounced=True, is_closing_trend=False) is True


def test_still_blocked_when_debounce_not_ready_regardless_of_trend():
    assert _g2_release_final(side_clear_debounced=False, is_closing_trend=False) is False
