"""Unit tests for dlat_trend_reset_reason diagnostic field (149節続報③、2026-07-21)。

背景: 148節③(footprint_risk過検知見直し)の診断のため、0720-05/07/08/0721-01の
footprint_risk発火64件全てで既存ログの"v_ema=0.0 shrink_run=0"を確認したが、
これは実は_lat_dec.v_corridor_ema/shrink_run(壁ベースの別トレンド、footprint_risk
自身が発火時に明示的に0リセットする値)であり、本当に見るべきdlat_v_ema/
dlat_shrink_run(自車〜相手の実測間隔トレンド)ではなかったと判明した(誤った
フィールドを見ていたことによる誤診断、正直に訂正)。

dlat_v_ema/dlat_shrink_runは_update_dlat_trend()内で計算されるが、これまで
giveup発生時のログ([LAT-TTC-ACT] giveup trigger=...)には一度も出力されて
いなかった。本節では、なぜ0.0/未蓄積のままになるケースがあるのかを診断できる
よう、_update_dlat_trend()内のどのリセット経路(dlat_none/vid_changed/warmup/
正常update=none)を通ったかを示すdlat_trend_reset_reasonフィールドを追加した。

lateral_ttc_monitor.pyはrclpy非依存のため、実際のLateralTTCMonitorを直接
importして検証する(test_lateral_ttc_monitor.pyと同じ方針)。
"""
import os

import pytest

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor, TTCDecision


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


# --- ①非矛盾性: デフォルト値・フィールド存在確認 ---

def test_ttc_decision_has_dlat_trend_reset_reason_field_default_none():
    dec = TTCDecision()
    assert dec.dlat_trend_reset_reason == "none"


# --- ②各リセット経路が正しく記録されること ---

def test_reason_is_dlat_none_when_target_lost():
    mon = make_monitor()
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=None, fwd_ds=None,
                      vopp=None, dt=1.0, fwd_vid=None)
    assert dec.dlat_trend_reset_reason == "dlat_none"


def test_reason_is_warmup_on_first_valid_call():
    mon = make_monitor()
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=2.0, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.dlat_trend_reset_reason == "warmup"
    assert dec.dlat_v_ema == 0.0
    assert dec.dlat_shrink_run == 0


def test_reason_is_vid_changed_when_target_switches():
    mon = make_monitor()
    mon.update(side=0, space=None, opp_space=None, fwd_dlat=2.0, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    mon.update(side=0, space=None, opp_space=None, fwd_dlat=1.8, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=1.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car2")
    assert dec.dlat_trend_reset_reason == "vid_changed"
    assert dec.dlat_v_ema == 0.0
    assert dec.dlat_shrink_run == 0


def test_reason_is_none_after_two_valid_consecutive_cycles_same_vid():
    """同一対象車で2周期分の実測が揃えば、reset_reasonは"none"(正常計算)へ
    遷移し、dlat_v_emaが実際の縮小方向(負値)を示すことを確認する。"""
    mon = make_monitor()
    mon.update(side=0, space=None, opp_space=None, fwd_dlat=2.0, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=1.0, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.dlat_trend_reset_reason == "none"
    assert dec.dlat_v_ema < 0.0  # 2.0m -> 1.0mへ縮小しているので負値のはず
    assert dec.dlat_shrink_run == 1


# --- ③非干渉性: footprint_risk自身の明示的リセット(_shrink_run等)と
#     dlat系フィールドが独立していることを確認する ---

def test_footprint_risk_branch_does_not_reset_dlat_trend_fields():
    """footprint_risk分岐は_prev_space/_space_ema/_shrink_run/_critical_curvature_run
    (壁ベースのトレンド)を明示的に0リセットするが、dlat_v_ema/dlat_shrink_run/
    dlat_trend_reset_reason(自車〜相手の実測間隔トレンド)には触れないことを確認する
    (今回の誤診断の直接の原因箇所——両者が独立したフィールドであることの裏付け)。"""
    mon = make_monitor()
    # 2周期分の実測でdlat_v_emaを負値(縮小中)に育てておく。
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.0, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    # footprint_risk=Trueで発火させる。
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=0.9, fwd_ds=1.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", footprint_risk=True)
    assert dec.branch == "FOOTPRINT_RISK"
    assert dec.footprint_risk_triggered is True
    # 壁ベースのshrink_runは明示的に0リセットされる(既存挙動)。
    assert dec.shrink_run == 0
    # しかしdlat系は前周期の値(既に蓄積済みの負のトレンド)をそのまま引き継ぐはず。
    assert dec.dlat_trend_reset_reason == "none"
    assert dec.dlat_v_ema < 0.0


# --- ④過去ログへの遡及効果: 既存ログの誤診断を訂正する裏付け ---

def test_retroactive_v_corridor_ema_is_separate_field_from_dlat_v_ema():
    """0720-05/07/08/0721-01ログで観測された"v_ema=0.0 shrink_run=0"が
    実際にはv_corridor_ema/shrink_run(壁ベース)であり、dlat_v_ema/
    dlat_shrink_run(実測間隔ベース)とは別フィールドであることを、
    TTCDecisionの実際のフィールド名で確認する。"""
    fields = TTCDecision.__dataclass_fields__
    assert "v_corridor_ema" in fields or "shrink_run" in fields
    assert "dlat_v_ema" in fields
    assert "dlat_shrink_run" in fields
    assert "dlat_trend_reset_reason" in fields


# --- ③配線確認: mpc_controller.pyのgiveupログに新フィールドが出力されること ---

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_giveup_log_includes_dlat_trend_diagnostic_fields():
    """168節でtriggerラベルが_giveup_trigger変数(room_exhausted/lat_ttc_*の分岐)へ
    変わったが、ログ本体のdlat診断フィールド自体は無変更のまま。"""
    idx = _SRC.index('f"[LAT-TTC-ACT] giveup trigger={_giveup_trigger}')
    idx_end = idx + 1600
    snippet = _SRC[idx:idx_end]
    assert "dlat_v_ema={_lat_dec.dlat_v_ema:.3f}" in snippet
    assert "dlat_shrink_run={_lat_dec.dlat_shrink_run}" in snippet
    assert "dlat_trend_reset_reason={_lat_dec.dlat_trend_reset_reason}" in snippet
