"""Unit tests for 191節(2026-07-26): switchback/A_rescue/A_rescue_relaxed/A_dlatの
縦方向盲点対処(AXIS03、理想のレーシングアルゴリズムvs現状ドキュメント参照)。

背景: 予選ログ(wp185-198)実測で、A_rescueがopp_space=3.22・space=2.62(横空間の
条件は満たす)を根拠に側反転したが、その0.7秒後にfwd_ds=0.99mでfootprint_risk
giveup、さらに0.24秒後にCOLLISION-SUSPECTEDが発生した事例を確認した。
_switchback_eligible/_rescue_eligible/_rescue_relaxed_eligible/_dlat_switchback_
eligibleはいずれも横方向の空き(space/opp_space)のみを見ており、相手との縦距離
(fwd_ds)を一切見ていなかった——ENGAGEゲート(165節)で既に対処した「横空間は
見るが縦距離は見ない」盲点が、switchback層に独立して存在していた。

対処: 呼び出し元(mpc_controller.py)がabs(fwd_ds) < along_min_length
(footprint_risk本体と同一の物理下限)から算出したfwd_ds_overlap_riskを、4つの
反転経路(A/A_lookahead, A_rescue, A_rescue_relaxed, A_dlat)すべての成立条件へ
AND追加する。new_side_wall_blocked等の既存の物理ベースveto群と同じ位置・同じ
スコープに追加するのみで、新規閾値は導入しない。

LateralTTCMonitorはrclpy非依存の純Pythonクラスのため、実クラスを直接importして
end-to-endでテストする(test_lateral_ttc_monitor.pyと同じ方針)。mpc_controller.py
側の配線(fwd_ds_overlap_risk算出・呼び出し・reasonマッピング)はソーステキスト
検証で確認する。
"""
import os

import pytest

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _CTRL_SRC = _f.read()

_LTM_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "lateral_ttc_monitor.py")
with open(_LTM_SRC_PATH) as _f:
    _LTM_SRC = _f.read()


def make_monitor(**overrides):
    kwargs = dict(
        beta=1.0, space_ema_alpha=1.0,
        ttc_danger_s=2.0, ttc_critical_s=0.8,
        giveup_space_m=1.85, switchback_space_m=2.35,
        side_by_side_dlat_m=1.6, side_by_side_ds_m=1.0,
        caution_speed_margin_kmh=2.0, min_trend_cycles=1,
        cleared_space_m=1.45, v_inst_max=5.0,
    )
    kwargs.update(overrides)
    return LateralTTCMonitor(**kwargs)


# ---------------------------------------------------------------------------
# ①非矛盾性: fwd_ds_overlap_risk=Trueが4つの反転経路すべてを塞ぐ
# ---------------------------------------------------------------------------

def test_branch_a_blocked_by_fwd_ds_overlap_risk():
    """回帰の核心①: 通常switchback(branch=A)が本来成立する設定
    (test_switchback_branch_a_fires_on_wide_opposite_sideと同一条件)でも、
    fwd_ds_overlap_risk=Trueなら反転せず抑制される。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=1.5,
                      vopp=3.0, dt=1.0, fwd_vid="car1", fwd_ds_overlap_risk=True)
    assert dec.branch != "A"
    assert dec.side_override is None
    assert dec.switchback_ds_blocked is True
    assert mon.has_switched is False


def test_branch_a_unaffected_when_fwd_ds_overlap_risk_false():
    """回帰: fwd_ds_overlap_risk=False(既定)なら従来通りbranch=Aが成立する
    (既存test_switchback_branch_a_fires_on_wide_opposite_sideと同一条件・同一結果)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "A"
    assert dec.side_override == -1


def test_a_rescue_blocked_by_fwd_ds_overlap_risk():
    """回帰の核心②: 0722-04予選ログの実例そのもの
    (test_rescue_fires_when_has_switched_already_consumed_and_ttc_criticalと
    同一条件、has_switched=True・critical局面)でも、fwd_ds_overlap_risk=Trueなら
    A_rescueは発火せず、安全側のC2(強制giveup)へ落ちる。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=0.99,
                      vopp=3.0, dt=1.0, fwd_vid="car1", fwd_ds_overlap_risk=True)
    assert dec.branch == "C2"
    assert dec.force_giveup is True
    assert dec.side_override is None
    assert dec.switchback_ds_blocked is True
    assert mon.has_rescued is False


def test_a_rescue_unaffected_when_fwd_ds_overlap_risk_false():
    """回帰: 既存test_rescue_fires_when_has_switched_already_consumed_and_ttc_
    criticalと完全に同一の結果(fwd_ds_overlap_risk省略時、挙動は無変更)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "A_rescue"
    assert dec.side_override == -1
    assert mon.has_rescued is True


def test_a_rescue_relaxed_blocked_by_fwd_ds_overlap_risk():
    """回帰の核心③: A_rescue_relaxed(既存test_relaxed_rescue_fires_when_opp_
    space_between_cleared_and_switchbackと同一条件)も、fwd_ds_overlap_risk=True
    なら発火せず抑制される。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=0.8,
                      vopp=3.0, dt=1.0, fwd_vid="car1", fwd_ds_overlap_risk=True)
    assert dec.branch != "A_rescue_relaxed"
    assert dec.side_override is None
    assert mon.has_rescued is False


# ---------------------------------------------------------------------------
# ③ハンチング防止: fwd_ds_overlap_riskはトークン(has_switched/has_rescued)を
# 消費しない(不成立として扱われ、次に真に安全になった時点で再挑戦できる)
# ---------------------------------------------------------------------------

def test_fwd_ds_overlap_risk_does_not_consume_switchback_token():
    """fwd_ds_overlap_riskで抑制された周期はhas_switchedを消費しないため、
    後続の周期でfwd_ds_overlap_riskがFalseに戻れば通常通りbranch=Aが成立する
    (space自体は継続して縮小させ、縮小トレンド自体は途切れさせない)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_blocked = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5,
                              fwd_ds=1.5, vopp=3.0, dt=1.0, fwd_vid="car1",
                              fwd_ds_overlap_risk=True)
    assert dec_blocked.branch != "A"
    assert mon.has_switched is False
    dec_recovered = mon.update(side=1, space=2.0, opp_space=3.0, fwd_dlat=2.5,
                                fwd_ds=3.0, vopp=3.0, dt=1.0, fwd_vid="car1",
                                fwd_ds_overlap_risk=False)
    assert dec_recovered.branch == "A"
    assert mon.has_switched is True


# ---------------------------------------------------------------------------
# ソーステキスト検証: mpc_controller.py側の配線
# ---------------------------------------------------------------------------

def test_source_controller_computes_fwd_ds_overlap_risk_from_along_min_length():
    idx = _CTRL_SRC.index("_fwd_ds_overlap_risk = (")
    snippet = _CTRL_SRC[idx:idx + 200]
    assert "abs(_fwd_ds) < self._along_min_length" in snippet


def test_source_controller_passes_fwd_ds_overlap_risk_to_update():
    idx = _CTRL_SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _CTRL_SRC[idx:idx + 1200]
    assert "fwd_ds_overlap_risk=_fwd_ds_overlap_risk" in snippet


def test_source_controller_reason_mapping_includes_ds():
    idx = _CTRL_SRC.index('_reason = ("cleared_margin" if _lat_dec.switchback_cleared_margin_blocked')
    snippet = _CTRL_SRC[idx:idx + 1200]
    assert '"ds" if _lat_dec.switchback_ds_blocked' in snippet
    # dsはmargin(既定値)の直前、offsetの直後に位置する(優先順位: 他の物理veto群と同格)
    assert snippet.index('"ds"') < snippet.index('else "margin")')


def test_source_no_new_config_parameter_introduced():
    """②非冗長性: 191節はalong_min_length(既存)を再利用するのみで、新規
    config.yamlパラメータを導入していない。"""
    idx = _CTRL_SRC.index("_fwd_ds_overlap_risk = (")
    snippet = _CTRL_SRC[idx:idx + 200]
    assert "_otget(" not in snippet
