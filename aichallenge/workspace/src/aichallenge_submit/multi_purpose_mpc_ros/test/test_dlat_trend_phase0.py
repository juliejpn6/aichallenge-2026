"""Unit tests for the fwd_dlat trend tracker (132節, Gap①Phase0, 2026-07-20).

Background: 131節の効率性レビューで、ENGAGEから5秒以内に断念するケースが
19回中8回(42%)発生していることを確認した。0720-02実測ログのwp284
(giveup直後、同一対象車vid=d3を4.2秒後に再エンゲージし、0.55秒後に
FOOTPRINT_RISKで強制giveup)を深掘りしたところ、LateralTTCMonitor.update()の
`if side == 0 or space is None:` 早期return(未エンゲージ中は毎周期通る経路)が、
本来「1エンゲージにつき反転1回」制限用のhas_switched等と一緒に、fwd_dlatの
縮小トレンドまで巻き添えでリセットしていたことが根本原因と判明した(132節)。

本Phase0は診断専用で、ENGAGE可否判定には一切影響しない。次回予選ログで
dlat_v_ema/dlat_shrink_runの実測値を集め、数値検証してからPhase1(_plan_pass
への配線)を検討する前段。
"""
import os

import pytest

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor, TTCDecision

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _MPC_SRC = _f.read()

_LAT_TTC_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "lateral_ttc_monitor.py")
with open(_LAT_TTC_SRC_PATH) as _f:
    _LAT_TTC_SRC = _f.read()


def make_monitor(**overrides):
    """決定的に検証するため、既定でbeta=1.0・space_ema_alpha=1.0とする
    (既存test_lateral_ttc_monitor.pyのmake_monitorと同じ考え方)。"""
    kwargs = dict(
        beta=1.0, space_ema_alpha=1.0,
        ttc_danger_s=2.0, ttc_critical_s=0.8,
        giveup_space_m=1.85, switchback_space_m=2.35,
        side_by_side_dlat_m=1.6, side_by_side_ds_m=1.0,
        caution_speed_margin_kmh=2.0, min_trend_cycles=3,
        cleared_space_m=1.45, v_inst_max=5.0,
    )
    kwargs.update(overrides)
    return LateralTTCMonitor(**kwargs)


def test_dlat_v_ema_becomes_negative_when_fwd_dlat_shrinks():
    """fwd_dlatが縮み続けると、dlat_v_emaが負(縮小方向)になることを確認する。"""
    mon = make_monitor()
    dec = None
    for dlat in (3.0, 2.5, 2.0, 1.5, 1.0):
        dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                          fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    assert dec.dlat_v_ema < 0.0


def test_dlat_shrink_run_counts_consecutive_shrinking_cycles():
    """連続して縮小した周期数がdlat_shrink_runへ蓄積されることを確認する。"""
    mon = make_monitor()
    dec = None
    for dlat in (3.0, 2.5, 2.0, 1.5):
        dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                          fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    # 1周期目=基準値、2周期目=warmup(_prev_dlat_ema初回セット)なので、
    # 実際に微分が始まるのは3周期目から。4周期分投入したので2回はshrinkを検知する。
    assert dec.dlat_shrink_run >= 1


def test_dlat_shrink_run_resets_when_dlat_stops_shrinking():
    """縮小が止まる(横ばい/拡大)とdlat_shrink_runが0へ戻ることを確認する。"""
    mon = make_monitor()
    for dlat in (3.0, 2.5, 2.0, 1.5):
        mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                   fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=1.5,
                      fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    assert dec.dlat_shrink_run == 0


def test_trend_survives_across_side_zero_cycles_core_fix_claim():
    """本Phase0の核心: side==0(未エンゲージ)の間もfwd_dlatトレンドが消えずに
    持続することを確認する。0720-02実測wp264→wp273 giveup→STOPPING(side=0)
    →wp284再エンゲージのシーケンスを模した、同一vid・縮小継続のシナリオ。"""
    mon = make_monitor()
    # side=1でエンゲージ中に縮小が始まる(wp264のエピソード相当)。
    for dlat in (2.0, 1.8, 1.6):
        mon.update(side=1, space=3.0, opp_space=3.0, fwd_dlat=dlat,
                   fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    # giveup→STOPPING、side=0が続く間も同一対象車(d3)を引き続き縮小しながら
    # 追跡する(4.2秒相当、dt=1.0で4周期)。
    dec = None
    for dlat in (1.4, 1.2, 1.0, 0.8):
        dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                          fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    # side==0の間もdlat_shrink_runが蓄積し続けている(=消えていない)ことを確認する。
    assert dec.dlat_shrink_run >= 3
    assert dec.dlat_v_ema < 0.0


def test_reset_episode_does_not_clear_dlat_trend_state():
    """reset_episode()(_ot_side==0の間、呼び出し元が毎周期呼ぶ)がdlat系の
    トレンド状態を巻き添えでリセットしないことを直接確認する(132節の根本原因の
    再発防止)。

    2026-07-26追記(186節続報): dlat_shrink_run/dlat_v_emaは本Phase0時点では
    診断専用だったが、186節続報でbranch=A_dlat(fwd_dlat起点の早期switchback
    トリガー)へ実際に配線された。本テストの元の意図(トレンド状態そのものが
    reset_episode()で消えないこと)を汚さないよう、opp_space(1.0)を
    switchback_space_m未満に据えてA_dlatの発火条件(_dlat_switchback_eligible)
    を意図的に不成立にしている(発火するとその副作用でdlat_shrink_run/
    v_dlat_ema自体が0へリセットされてしまい、reset_episode()単体の効果を
    切り分けられなくなるため)。"""
    mon = make_monitor()
    for dlat in (2.0, 1.6, 1.2, 0.8):
        mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=dlat,
                   fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    v_before = mon._v_dlat_ema
    run_before = mon._dlat_shrink_run
    assert run_before > 0
    mon.reset_episode()
    assert mon._v_dlat_ema == v_before
    assert mon._dlat_shrink_run == run_before


def test_trend_resets_on_vid_change():
    """対象車が切り替わった周期は、既存の_vid_changed処理(space系)と同じ考え方で
    dlat系トレンドも静かに再スタートすることを確認する(別の車の間隔が急に
    飛び込んでも縮小として誤検知しない)。"""
    mon = make_monitor()
    for dlat in (2.0, 1.6, 1.2, 0.8):
        mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                   fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    assert mon._dlat_shrink_run > 0
    # 別の車(d5)へ切替。dlat自体は縮小方向の値が来ても、切替直後は静かに再基準化する。
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=0.5,
                      fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d5")
    assert dec.dlat_shrink_run == 0
    assert dec.dlat_v_ema == 0.0


def test_trend_resets_when_fwd_dlat_is_none():
    """対象車ロスト(fwd_dlat=None)の周期でトレンドが破棄されることを確認する。"""
    mon = make_monitor()
    for dlat in (2.0, 1.6, 1.2, 0.8):
        mon.update(side=0, space=None, opp_space=None, fwd_dlat=dlat,
                   fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    assert mon._dlat_shrink_run > 0
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=None,
                      fwd_ds=None, vopp=None, dt=1.0, fwd_vid=None)
    assert dec.dlat_shrink_run == 0
    assert dec.dlat_v_ema == 0.0


def test_decision_always_carries_dlat_fields_regardless_of_branch():
    """branch="none"(side==0)・branch="FOOTPRINT_RISK"等、どの分岐でも
    TTCDecisionにdlat_v_ema/dlat_shrink_runが付与されることを、_decision()
    ラッパー経由の一元化により確認する。"""
    mon = make_monitor()
    dec_none = mon.update(side=0, space=None, opp_space=None, fwd_dlat=2.0,
                           fwd_ds=3.0, vopp=2.0, dt=1.0, fwd_vid="d3")
    assert dec_none.branch == "none"
    assert hasattr(dec_none, "dlat_v_ema")
    assert hasattr(dec_none, "dlat_shrink_run")

    dec_fr = mon.update(side=1, space=3.0, opp_space=3.0, fwd_dlat=0.5,
                         fwd_ds=1.0, vopp=2.0, dt=1.0, fwd_vid="d3",
                         footprint_risk=True)
    assert dec_fr.branch == "FOOTPRINT_RISK"
    assert hasattr(dec_fr, "dlat_v_ema")
    assert hasattr(dec_fr, "dlat_shrink_run")


def test_ttc_decision_default_dlat_fields_are_zero():
    """回帰: TTCDecisionの新規フィールドの既定値が0(既存呼び出し元・既存テストの
    暗黙のTTCDecision(...)生成に影響しない)ことを確認する。"""
    dec = TTCDecision()
    assert dec.dlat_v_ema == 0.0
    assert dec.dlat_shrink_run == 0


# --- 構造テスト: ②非冗長性・実装配線の確認 ---

def test_update_calls_dlat_trend_tracker_before_side_zero_early_return():
    """_update_dlat_trend()の呼び出しが、side==0の早期returnより前(=side==0でも
    毎周期実行される位置)にあることをソース上で確認する。"""
    idx_call = _LAT_TTC_SRC.index("self._update_dlat_trend(fwd_dlat, fwd_vid, dt)")
    idx_early_return = _LAT_TTC_SRC.index("if side == 0 or space is None:")
    assert idx_call < idx_early_return


def test_dlat_trend_tracker_reuses_existing_ema_and_clamp_params_no_new_tuning_values():
    """②非冗長性: _update_dlat_trendがspace系と同じ既存パラメータ
    (space_ema_alpha/beta/v_inst_max)を再利用し、新規チューニング値を
    増やしていないことを確認する。"""
    idx = _LAT_TTC_SRC.index("def _update_dlat_trend(")
    idx_end = _LAT_TTC_SRC.index("def _decision(")
    snippet = _LAT_TTC_SRC[idx:idx_end]
    assert "self.space_ema_alpha" in snippet
    assert "self.beta" in snippet
    assert "self.v_inst_max" in snippet


def test_reset_episode_docstring_does_not_reset_dlat_fields_in_source():
    """reset_episode()本体(docstringの解説文ではなく実際の代入文)に
    _dlat_ema等への代入が無い(=巻き添えリセットしない設計が実際に
    ソース上も維持されている)ことを確認する。
    2026-07-30(247節): reset_episode()の直後にforce_rescue_switch()
    (room_exhausted救済専用、dlatトレンドを意図的にリセットする別メソッド)が
    追加されたため、終端マーカーを次の"def "(直後のメソッド定義)へ変更した
    (reset_episode()自身の範囲だけを正しく切り出すため)。"""
    idx = _LAT_TTC_SRC.index("def reset_episode(self)")
    idx_end = _LAT_TTC_SRC.index("\n    def ", idx + 10)
    snippet = _LAT_TTC_SRC[idx:idx_end]
    assert "self._dlat_ema =" not in snippet
    assert "self._prev_dlat_ema =" not in snippet
    assert "self._v_dlat_ema =" not in snippet
    assert "self._dlat_shrink_run =" not in snippet


def test_engage_log_includes_dlat_trend_fields_for_retroactive_validation():
    """[ENGAGE]ログにdlat_v_ema/dlat_shrink_runが追加され、次回ログでの
    遡及検証(wp264型の正常engageとwp284型の危険engageの比較)に使えることを
    確認する。"""
    idx = _MPC_SRC.index('"[ENGAGE] side={_eval.plan_side}')
    snippet = _MPC_SRC[idx:idx + 900]
    assert "dlat_v_ema={_lat_dec.dlat_v_ema:.3f}" in snippet
    assert "dlat_shrink_run={_lat_dec.dlat_shrink_run}" in snippet


def test_dlat_trend_alert_log_is_edge_triggered_not_every_cycle():
    """[DLAT-TREND]ログが、状態がFalse→Trueへ変化した周期のみ発火する
    エッジトリガー方式であることをソース上で確認する(標準運用手法③検証ロギング)。"""
    idx = _MPC_SRC.index('f"[DLAT-TREND]')
    snippet = _MPC_SRC[max(0, idx - 700):idx]
    assert "_dlat_trend_alert and not self._dlat_trend_alert_active" in snippet


def test_dlat_trend_alert_requires_side_zero_and_min_trend_cycles():
    """[DLAT-TREND]アラートが「未エンゲージ中(side==0)」かつ「既存の
    min_trend_cyclesデバウンス」を満たした場合のみ立つことを確認する
    (新規パラメータ0個、既存min_trend_cyclesの再利用)。"""
    idx = _MPC_SRC.index("_dlat_trend_alert = (")
    snippet = _MPC_SRC[idx:idx + 300]
    assert "self._ot_side == 0" in snippet
    assert "self._lat_ttc.min_trend_cycles" in snippet


def test_phase0_does_not_add_any_new_engage_veto_return_in_plan_pass():
    """回帰(最重要): Phase0は診断専用であり、_plan_passのENGAGE可否判定
    (return文)を一切変更していないことを確認する。131節1回目の実装
    (fwd_dlatの絶対値でengageを拒否する案)が既存の正常ケース6件を壊して
    revertされた反省を踏まえ、Phase0では判定ロジックに一切触れていないことを
    ソースレベルで保証する。"""
    assert "fwd_dlat_near" not in _MPC_SRC
    assert 'scan["fwd_dlat"] < self._along_lane_need' not in _MPC_SRC
    assert 'scan.get("fwd_dlat") is not None and scan["fwd_dlat"] <' not in _MPC_SRC
