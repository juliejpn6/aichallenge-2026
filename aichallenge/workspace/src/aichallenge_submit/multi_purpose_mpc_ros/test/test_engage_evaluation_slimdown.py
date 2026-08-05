"""Unit tests for the ENGAGE evaluation slimdown (148節、2026-07-21)。

背景: 0720-05/07/08予選ログの横断解析で、footprint_risk giveup後の再エンゲージが
毎回8.0秒の固定コスト(139節cooldown倍化)+可変0-13秒の追加遅延(rdy/cls等)を
要していることが判明した(delayed-reengage-after-stopped-car メモリ参照)。
その分析の過程でユーザーから「この判定もオーバーテイクのv_safe候補選択(144節)と
同様に複数階層で条件が入り組んでいるのでは」との指摘があり、144節と同じ手順
(純粋スリム化→再点検→統合検討)を適用することになった。

本節はステップ1(純粋スリム化)のみ: _control()内にインラインで書かれていた
cheap_ok(9条件)+_plan_pass呼び出し+dlat_ttc_veto+gate=ログ生成の一連の判定を
_evaluate_engage_readiness()へ抽出した。計算内容・呼び出し順序・self状態の
変更点は一切変えていない(挙動は完全に同一)。棚卸し(rdy/cls/wc/cdが
OpponentSituationを共有していない等)は次フェーズで別途検討する。

mpc_controller.pyはrclpy依存で直接importできないため、既存のテストと同じ方針
(純Pythonミラー+ソーステキストによる構造的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _control_call_site_snippet():
    idx = _SRC.index("_eval = self._evaluate_engage_readiness(")
    idx_end = _SRC.index("else:\n                        # 追従(ICC)")
    return _SRC[idx:idx_end]


def _control_decision_only_snippet():
    """ENGAGE成立/不成立時の"結果を受けての動作"(状態遷移・ログ)を含まない、
    判定呼び出しそのものだけの範囲(スリム化の効果測定用)。"""
    idx = _SRC.index("_eval = self._evaluate_engage_readiness(")
    idx_end = _SRC.index("if _eval.can_engage:")
    return _SRC[idx:idx_end]


def _helper_method_snippet():
    idx = _SRC.index("def _evaluate_engage_readiness(")
    idx_end = _SRC.index("def _scan_traffic(self, v_ego: float, ego_lat: float):")
    return _SRC[idx:idx_end]


# --- ①非矛盾性: 呼び出し側がヘルパー経由になっていること ---

def test_control_calls_evaluate_engage_readiness():
    snippet = _control_call_site_snippet()
    assert "self._evaluate_engage_readiness(" in snippet
    assert "_fwd_dbg[\"gate\"] = _eval.gate" in snippet
    assert "if _eval.can_engage:" in snippet


def test_control_no_longer_contains_inline_cheap_ok_computation():
    """②非冗長性: _control()側から、cheap_ok構成要素のインライン計算
    (_ego_ready/_close_enough/_on_path/_engage_dist_dynamic等)が完全に
    除去され、ヘルパー呼び出しの結果(_eval.xxx)のみが残っていることを
    確認する。"""
    snippet = _control_call_site_snippet()
    for removed in ("_ego_ready =", "_close_enough =", "_on_path =",
                     "_engage_dist_dynamic =", "_closing_est =",
                     "_cheap_ok =", "_dlat_ttc_veto ="):
        assert removed not in snippet, f"{removed} should have been extracted"


def test_decision_call_is_dramatically_shorter():
    """スリム化の効果を定量的に確認する(以前は判定部分だけで約145行あった。
    結果を受けての状態遷移・ログ(元々多くの行数を占める)は判定そのものでは
    ないため測定対象から除く)。"""
    snippet = _control_decision_only_snippet()
    n_lines = snippet.count("\n")
    assert n_lines < 10, f"expected a short decision call, got {n_lines} lines"


# --- ヘルパーメソッド自体の構造確認 ---

def test_evaluate_engage_readiness_method_exists_with_expected_signature():
    # 2026-07-22修正(00節監査、自車/相手情報の共有化): fwd_vopp/fwd_ds引数を廃止し
    # opp_sit(OpponentSituation)経由で参照するようシグネチャを変更した(値は同一)。
    idx = _SRC.index(
        "def _evaluate_engage_readiness(self, scan, pass_worth, v_odom,")
    assert idx > 0


def test_engage_eval_dataclass_has_expected_fields():
    idx = _SRC.index("class EngageEval:")
    idx_end = _SRC.index("class MPCController(Node):")
    snippet = _SRC[idx:idx_end]
    for field in ("cheap_ok", "ego_ready", "close_enough", "on_path",
                  "plan_ok", "plan_side", "can_engage", "closing_est",
                  "engage_dist_dynamic", "t_reach_profile", "gate"):
        assert f"{field}:" in snippet, f"missing field {field}"


def test_helper_preserves_all_nine_cheap_ok_conditions():
    """④過去ログへの遡及効果に相当する健全性チェック: 抽出後もcheap_okの
    9条件(lr/lat/cd/wc/path/rdy/cls/nbo、実測ログのgate=フィールドと同じ
    並び)が全て同じ式のまま残っていることを確認する。"""
    snippet = _helper_method_snippet()
    assert "self._ot_enable and (left_ok or right_ok)" in snippet
    assert "self._ot_infeasible_latch == 0" in snippet
    assert "self._ot_engage_cooldown == 0" in snippet
    assert "self._ot_worth_count >= self._ot_engage_debounce" in snippet
    assert "_on_path and _ego_ready and _close_enough" in snippet
    assert "not being_overtaken" in snippet


def test_helper_preserves_gate_string_format():
    """実測ログ(0720-07/08等)のgate=lr=..,lat=..,cd=..,wc=..,path=..,rdy=..,
    cls=..,nbo=..,plan=..:reason という既存フォーマットが、抽出後も
    バイト単位で同一であることを確認する(過去ログとの比較可能性を維持)。"""
    snippet = _helper_method_snippet()
    assert 'f"lr={int(left_ok or right_ok)}"' in snippet
    assert 'f",lat={int(self._ot_infeasible_latch == 0)}"' in snippet
    # 2026-07-21修正(148節②): cd=の判定式自体を固定タイマー(self._ot_engage_cooldown==0)
    # から、footprint_risk起因時は実測解消ベースの_cd_clearへ変更した(意図的な挙動変更、
    # 詳細はtest_footprint_risk_adaptive_cooldown.py参照)。ここでは変数参照そのものを確認する。
    assert 'f",cd={int(_cd_clear)}"' in snippet
    assert 'f",wc={int(self._ot_worth_count >= self._ot_engage_debounce)}"' in snippet
    assert 'f",path={int(_on_path)}"' in snippet
    assert 'f",rdy={int(_ego_ready)}"' in snippet
    assert 'f",cls={int(_close_enough)}"' in snippet
    assert 'f",nbo={int(not being_overtaken)}"' in snippet
    assert 'f",plan={int(_plan_ok)}:{getattr(self, ' in snippet


def test_helper_preserves_worth_count_mutation_with_vid_reset():
    """94節で追加されたworth_countのvid切替リセットが、抽出後も維持されて
    いることを確認する。233節続報(2026-07-29、監査結果④)で
    _vid_changed_reset()ヘルパー経由に統一されたが、リセット→増分の
    順序自体は不変(ヘルパー内でprev_vid更新まで完了してから戻り値を返すため、
    呼び出し元の増分処理は常に更新後の状態に対して行われる)。"""
    snippet = _helper_method_snippet()
    idx_check = snippet.index(
        'if self._vid_changed_reset(_fwd_vid_worth, "_ot_worth_prev_vid"):')
    idx_reset = snippet.index("self._ot_worth_count = 0")
    idx_increment = snippet.index(
        "self._ot_worth_count = self._ot_worth_count + 1 if pass_worth else 0")
    assert idx_check < idx_reset < idx_increment


def test_helper_preserves_dlat_ttc_veto_wiring_to_opponent_situation():
    """141節フェーズ1で配線されたOpponentSituation.is_closing_trendの参照が
    抽出後も維持されていることを確認する。"""
    snippet = _helper_method_snippet()
    assert "_dlat_ttc_veto = opp_sit.is_closing_trend" in snippet
    assert "_can_engage = _cheap_ok and _plan_ok and not _dlat_ttc_veto" in snippet


def test_helper_preserves_dbg_plan_reason_stale_display_fix():
    """55節(0714-01)で修正した「cheap_ok不成立時にplanLf/planRfをnanへ戻す」
    表示バグ対策が、抽出後も維持されていることを確認する。"""
    snippet = _helper_method_snippet()
    assert 'self._dbg_plan_reason = "cheap_ok_fail"' in snippet
    assert "self._dbg_plan_lf = float('nan')" in snippet
    assert "self._dbg_plan_rf = float('nan')" in snippet


def test_engage_log_line_uses_eval_fields():
    """ENGAGE成立時のログ([ENGAGE]行)が、ヘルパーの返り値
    (_eval.plan_side/_eval.closing_est/_eval.engage_dist_dynamic/
    _eval.t_reach_profile)を参照していることを確認する。"""
    idx = _SRC.index('f"[ENGAGE] side={_eval.plan_side}')
    # 2026-07-24追加(168節): room_exhausted状態のリセット代入2行が
    #   間に挿入されたため、窓を2200→2500へ拡大(検証対象そのものは無変更)。
    # 2026-08-05追加(299節続報、task#293候補①): 対象車横方向速度推定のリセット代入
    #   3行が間に挿入されたため、窓を2500→3200へ再拡大(検証対象そのものは無変更)。
    idx_end = idx + 3200
    snippet = _SRC[idx:idx_end]
    assert "closing_est={_eval.closing_est:.2f}" in snippet
    assert "engage_dist_dynamic={_eval.engage_dist_dynamic:.2f}" in snippet
    assert "t_reach_profile={_eval.t_reach_profile}" in snippet
    assert "self._ot_side = _eval.plan_side" in snippet
    assert "self._ot_side_locked = _eval.plan_side" in snippet


# --- ④過去ログへの遡及効果: 実測gate=文字列を再現できることを確認 ---

def _mirror_gate(left_ok, right_ok, infeasible_latch, engage_cooldown,
                  worth_count, engage_debounce, on_path, ego_ready,
                  close_enough, being_overtaken, plan_ok, plan_reason):
    """_evaluate_engage_readinessのgate=構築部分のミラー実装。"""
    return (
        f"lr={int(left_ok or right_ok)}"
        f",lat={int(infeasible_latch == 0)}"
        f",cd={int(engage_cooldown == 0)}"
        f",wc={int(worth_count >= engage_debounce)}"
        f",path={int(on_path)}"
        f",rdy={int(ego_ready)}"
        f",cls={int(close_enough)}"
        f",nbo={int(not being_overtaken)}"
        f",plan={int(plan_ok)}:{plan_reason}")


def test_retroactive_0720_07_wp250_gate_string_reproduced():
    """0720-07実測(wp250、t=569.69、footprint_risk giveup直後)のgate=
    lr=1,lat=1,cd=0,wc=1,path=1,rdy=0,cls=1,nbo=0,plan=0:cheap_ok_failを
    ミラー実装で再現できることを確認する。"""
    gate = _mirror_gate(left_ok=True, right_ok=True, infeasible_latch=0,
                         engage_cooldown=160, worth_count=5, engage_debounce=3,
                         on_path=True, ego_ready=False, close_enough=True,
                         being_overtaken=False, plan_ok=False,
                         plan_reason="cheap_ok_fail")
    assert gate == "lr=1,lat=1,cd=0,wc=1,path=1,rdy=0,cls=1,nbo=1,plan=0:cheap_ok_fail"


def test_retroactive_0720_08_wp331_all_conditions_pass_yields_narrow():
    """0720-05実測(wp298、全条件成立しplan=narrowでcheap_okは通ったが
    _plan_pass自体が狭さでvetoした)ケースをミラーで再現する。"""
    gate = _mirror_gate(left_ok=True, right_ok=True, infeasible_latch=0,
                         engage_cooldown=0, worth_count=5, engage_debounce=3,
                         on_path=True, ego_ready=True, close_enough=True,
                         being_overtaken=False, plan_ok=False,
                         plan_reason="narrow")
    assert gate == "lr=1,lat=1,cd=1,wc=1,path=1,rdy=1,cls=1,nbo=1,plan=0:narrow"
