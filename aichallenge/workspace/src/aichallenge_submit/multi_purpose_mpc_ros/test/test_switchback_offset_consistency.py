"""Unit tests for the switchback decision-layer/execution-layer consistency fix
(159節、2026-07-22)。

背景: A_rescue/通常switchbackの可否判定(space/opp_space・new_side_wall_blocked
=コリドー全体幅)は、実際にオフセット目標を動かす層(_corr_bound_ahead=反転先
方向への実測先読み最小値)とは異なる指標を使っていた。0722-03予選ログの実測
(video_t≈34.3秒)で、A_rescueが成立し_ot_sideが+1→-1へ反転したにも関わらず、
直後のcorr_bound_ahead(新側)が負値(-0.587)となりオフセット目標が実質ゼロへ
クランプされ、「側だけ反転し車両は動かない」という内部矛盾状態が発生した。
この状態が、対象車切替(_scan_trafficの選択が側に依存)→新対象車への
footprint_risk即発火、という連鎖を招いた(157/158節)。

ユーザー指摘「厳密にはそれぞれで計算するのではなくその周期で計算した同じ値を
使用すれば矛盾は生じない」を受け、mpc_controller.py側の呼び出し順序
(_new_side_corr_bound計算→_lat_ttc.update()→_corr_bound_ahead(オフセット目標
計算)→get_control())を確認した。dbg_corr_ub_arr/lb_arrはget_control()が
呼ばれるまでこの周期内で凍結されているため、両呼び出しは決定論的に同一値を
返すことが構造上保証されており、明示的なキャッシュは不要と判断した。

lateral_ttc_monitor.pyはROS非依存のため直接importして検証する
(test_switchback_curvature_override.pyと同じ方針)。
"""
import os

import pytest

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor


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


def two_step(mon, s_prev, s_now, cleared=False, fwd_dlat=2.5, fwd_ds=3.0,
             opp_space=3.0, vopp=3.0, dt=1.0, fwd_vid="car1",
             new_side_offset_blocked=False, **kw):
    mon.update(side=1, space=s_prev, opp_space=opp_space, fwd_dlat=fwd_dlat,
               fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
               new_side_offset_blocked=new_side_offset_blocked, **kw)
    return mon.update(side=1, space=s_now, opp_space=opp_space, fwd_dlat=fwd_dlat,
                       fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
                       new_side_offset_blocked=new_side_offset_blocked, **kw)


# --- 通常switchback(branch=A)経路 ---

def test_regression_default_offset_blocked_false_still_switches():
    """回帰: new_side_offset_blockedを渡さない(デフォルトFalse)場合、
    従来通り反転が成立することを確認する。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0)
    assert dec.branch == "A"
    assert dec.side_override == -1


def test_offset_blocked_suppresses_switch_even_when_otherwise_eligible():
    """本修正の中核: 他の条件(margin>=0・opp_space>=switchback_space_m・
    wall/room/curvature全てFalse)が全て満たされていても、
    new_side_offset_blocked=Trueなら反転が抑制される。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                   new_side_offset_blocked=True)
    assert dec.branch != "A"
    assert dec.switchback_suppressed is True
    assert dec.switchback_offset_blocked is True


def test_offset_blocked_diagnostic_priority_matches_code_order():
    """診断の優先順位確認: wall_blockedとoffset_blockedが同時にTrueの場合、
    コード上のelif順序(wall優先)通りswitchback_wall_blockedが報告される。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                   new_side_wall_blocked=True, new_side_offset_blocked=True)
    assert dec.switchback_wall_blocked is True
    assert dec.switchback_offset_blocked is False


# --- A_rescue経路 ---

def test_rescue_branch_blocked_when_offset_blocked():
    """A_rescue経路でもnew_side_offset_blocked=Trueなら抑制される。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_a = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec_a.branch == "A"
    mon.update(side=-1, space=2.0, opp_space=0.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_r = mon.update(side=-1, space=0.1, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1", new_side_offset_blocked=True)
    assert dec_r.branch != "A_rescue"
    assert dec_r.switchback_offset_blocked is True


def test_rescue_branch_allowed_when_offset_not_blocked_regression():
    """回帰: new_side_offset_blocked=False(実行層でも到達可能)なら
    従来通りA_rescueが成立する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_a = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec_a.branch == "A"
    mon.update(side=-1, space=2.0, opp_space=0.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_r = mon.update(side=-1, space=0.1, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1", new_side_offset_blocked=False)
    assert dec_r.branch == "A_rescue"
    assert dec_r.switchback_offset_blocked is False


# --- ④過去ログへの遡及効果: 0722-03実測(video_t≈34.3秒) ---

def test_retroactive_0722_03_a_rescue_would_have_been_suppressed():
    """遡及検証: 0722-03実測(opp_space=2.25、switchback_space_m=2.35未満だが
    ここではmargin>=0成立を想定した近似シナリオで、new_side_offset_blocked=True
    相当のcorr_bound_ahead負値を模して反転が抑制されることを確認する。"""
    dec = two_step(make_monitor(), s_prev=2.5, s_now=2.01, opp_space=2.25,
                   new_side_offset_blocked=True)
    assert dec.branch != "A"
    assert dec.switchback_offset_blocked is True


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証(159節)
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_source_new_side_corr_bound_reuses_existing_function_and_threshold():
    """②非冗長性: 判定層の新チェックが既存の_corr_bound_ahead()・
    既存のself._along_min_widthを再利用しており、新規計算式を持たないことを
    確認する。"""
    idx = _SRC.index("_new_side_corr_bound = (self._corr_bound_ahead(-self._ot_side)")
    snippet = _SRC[idx:idx + 300]
    assert "self._corr_bound_ahead(-self._ot_side)" in snippet
    assert "_new_side_offset_blocked = _new_side_corr_bound < self._along_min_width" in snippet


def test_source_new_side_corr_bound_computed_before_update_call():
    """①非矛盾性: _new_side_corr_boundがupdate()呼び出しより前に計算されている
    (出現順で確認)。"""
    idx_compute = _SRC.index("_new_side_corr_bound = (self._corr_bound_ahead(-self._ot_side)")
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_compute < idx_update


def test_source_update_call_passes_new_side_offset_blocked():
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx:idx + 900]
    assert "new_side_offset_blocked=_new_side_offset_blocked," in snippet


def test_source_reason_string_includes_offset():
    """③検証ロギング: switchback_suppressedのreason文字列にofsetが
    追加されていることを確認する。"""
    idx = _SRC.index('_reason = ("cleared_margin"')
    snippet = _SRC[idx:idx + 700]
    assert '"offset" if _lat_dec.switchback_offset_blocked' in snippet


def test_source_offset_bound_computed_before_get_control_call():
    """①非矛盾性の裏付け: _new_side_corr_boundの計算・オフセット目標計算
    (_corr_bound_ahead(self._ot_side)、168節でroom_exhausted判定との共有値
    _room_ahead_lockedの再利用を追加)のいずれも、この周期のget_control()
    (dbg_corr_ub_arr/lb_arrを更新する箇所)より前に位置している(=同一の
    凍結された配列を参照することが保証される)ことを出現順で確認する。"""
    idx_new_side = _SRC.index("_new_side_corr_bound = (self._corr_bound_ahead(-self._ot_side)")
    idx_offset_target = _SRC.index(
        "_corr_bound = (_room_ahead_locked if _room_ahead_locked is not None")
    idx_get_control = _SRC.index("u, max_delta = self._mpc.get_control()")
    assert idx_new_side < idx_offset_target < idx_get_control
