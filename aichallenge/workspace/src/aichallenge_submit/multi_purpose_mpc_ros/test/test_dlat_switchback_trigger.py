"""Unit tests for branch="A_dlat"(186節続報、クロスライン対策)。

第3コーナー追突(0726-01実測、wp161 ENGAGE→wp170 footprint_risk giveup)の分析で、
壁コリドー(_v_corridor_ema)は終始「安定」だったにも関わらずfwd_dlat(相手との実測
横間隔)だけが11周期連続で縮小し続けていたことが判明した。既存のswitchback判定
(branch=A)は壁コリドーが縮小しない限り評価ブロックへ到達できない構造だったため、
fwd_dlat起点で独立に評価される早期トリガー(branch=A_dlat)を追加した。

可否条件(_switchback_eligible相当)・veto群(new_side_blocked/wall_blocked/
room_blocked/offset_blocked)・has_switchedラッチ・発火時の状態リセットは既存の
branch=Aと完全に同一のものを再利用しており、新規の安全弁・新規パラメータは
一切追加していない。本ファイルはA_dlat固有の発火条件と、既存branch=A/wall系
テストへの非干渉(fwd_dlatが変化しない限りA_dlatブロックは常にno-opであること)
を検証する。
"""
from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor


def make_monitor(**overrides):
    """test_lateral_ttc_monitor.pyのmake_monitorと同一の決定的設定。"""
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


def two_step_dlat(mon, dlat_prev, dlat_now, space=2.6, opp_space=3.0,
                   fwd_ds=3.0, vopp=3.0, dt=1.0, fwd_vid="car1", cleared=False,
                   new_side_blocked=False, new_side_curvature_override=False,
                   new_side_wall_blocked=False, new_side_room_blocked=False,
                   new_side_offset_blocked=False, lookahead_favor_switch=False):
    """space/opp_spaceは既定で一定(壁コリドーは常に"安定")のまま、fwd_dlatのみを
    dlat_prev→dlat_nowへ変化させ、A_dlatトリガーがfwd_dlat単独で評価されることを
    検証する。"""
    kwargs = dict(side=1, space=space, opp_space=opp_space, fwd_ds=fwd_ds,
                   vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
                   new_side_blocked=new_side_blocked,
                   new_side_curvature_override=new_side_curvature_override,
                   new_side_wall_blocked=new_side_wall_blocked,
                   new_side_room_blocked=new_side_room_blocked,
                   new_side_offset_blocked=new_side_offset_blocked,
                   lookahead_favor_switch=lookahead_favor_switch)
    mon.update(fwd_dlat=dlat_prev, **kwargs)
    return mon.update(fwd_dlat=dlat_now, **kwargs)


# ---------------------------------------------------------------------------
# 発火条件(壁コリドーが安定でも、fwd_dlat単独の縮小で発火する)
# ---------------------------------------------------------------------------

def test_a_dlat_fires_when_wall_corridor_stable_but_dlat_closing():
    """186節続報の核心: space/opp_spaceが不変(壁は"安定")でも、fwd_dlatが
    cleared_space_m未満まで縮み続ければbranch=A_dlatが発火する。"""
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5)
    assert dec.branch == "A_dlat"
    assert dec.side_override == -1
    assert mon.has_switched is True


def test_a_dlat_does_not_fire_when_dlat_opening():
    """fwd_dlatが拡大方向なら(壁も不変のため)stableのまま、A_dlatは発火しない。"""
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=0.5, dlat_now=3.0)
    assert dec.branch != "A_dlat"
    assert dec.side_override is None


def test_wall_branch_a_still_fires_when_only_wall_closes_and_dlat_flat():
    """回帰確認: fwd_dlatが不変(v_dlat_ema=0)の場合、A_dlatブロックは完全に
    no-opとなり、既存の壁ベースbranch=Aがそのまま発火する
    (test_lateral_ttc_monitor.py::test_switchback_branch_a_fires_on_wide_opposite_side
    と同一条件)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "A"
    assert dec.side_override == -1


# ---------------------------------------------------------------------------
# 既存veto群の再利用確認(branch=Aと完全に同一の可否条件を共有する)
# ---------------------------------------------------------------------------

def test_a_dlat_suppressed_by_new_side_wall_blocked():
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5, new_side_wall_blocked=True)
    assert dec.branch != "A_dlat"
    assert dec.side_override is None
    assert dec.switchback_suppressed is True
    assert dec.switchback_wall_blocked is True
    assert mon.has_switched is False


def test_a_dlat_blocked_by_new_side_blocked_without_curvature_override():
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5, new_side_blocked=True)
    assert dec.branch != "A_dlat"
    assert dec.switchback_curvature_blocked is True
    assert mon.has_switched is False


def test_a_dlat_curvature_override_allows_fire_despite_new_side_blocked():
    """157節のcurvature override(実測opp_spaceがswitchback_space_mを満たせば
    静的曲率懸念を上書き)がA_dlatにもそのまま適用されることを確認する。"""
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5,
                         new_side_blocked=True, new_side_curvature_override=True)
    assert dec.branch == "A_dlat"
    assert dec.side_override == -1


def test_a_dlat_requires_nonnegative_margin():
    """84節①のmargin>=0ガード(反対側の方が狭い反転は許可しない)を共有する。"""
    mon = make_monitor()
    # opp_space(2.4)>=switchback_space_m(2.35)は満たすが、space(2.6)の方が広い
    # (margin=-0.2)ため発火しない。
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5, opp_space=2.4, space=2.6)
    assert dec.branch != "A_dlat"
    assert dec.side_override is None


def test_a_dlat_blocked_when_already_side_by_side():
    """既にis_side_by_side(真横)ならswitchback自体の対象外
    (branch=Aと同じ`not self.is_side_by_side`ガード)。"""
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=1.5, dlat_now=0.5, fwd_ds=0.5)
    assert dec.branch != "A_dlat"
    assert dec.side_override is None


# ---------------------------------------------------------------------------
# ハンチング防止(has_switchedラッチの共有)
# ---------------------------------------------------------------------------

def test_a_dlat_respects_has_switched_latch_no_double_fire():
    """A_dlatが一度発火した後、同一エピソード内でfwd_dlatが再度縮小しても、
    既存branch=Aと共有するhas_switchedラッチにより再発火しない
    (1エンゲージ1回のみという既存制限をそのまま継承)。"""
    mon = make_monitor()
    dec1 = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5)
    assert dec1.branch == "A_dlat"
    assert mon.has_switched is True

    dec2 = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=0.2,
                       fwd_ds=3.0, vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec2.branch != "A_dlat"
    assert dec2.side_override is None
    assert mon.has_switched is True


def test_a_dlat_fire_resets_own_dlat_trend():
    """発火時に自身のdlatトレンド(_v_dlat_ema/_dlat_shrink_run)もリセットされる
    (反転先で基準が変わるため、古い縮小方向を残さない設計)。"""
    mon = make_monitor()
    two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5)
    assert mon._v_dlat_ema == 0.0
    assert mon._dlat_shrink_run == 0


# ---------------------------------------------------------------------------
# lookahead_favor_switchバイパスの共有(branch=Aと同じスコープのみ)
# ---------------------------------------------------------------------------

def test_a_dlat_lookahead_favor_switch_does_not_bypass_negative_margin():
    """107節案A'の回帰確認をA_dlatでも踏襲する: lookahead_favor_switch=Trueでも
    margin(opp_space-space)自体が負なら発火しない。opp_space(2.4)は
    switchback_space_m(2.35)を満たすが、space(3.0)の方が広い(margin=-0.6)。"""
    mon = make_monitor()
    dec = two_step_dlat(mon, dlat_prev=3.0, dlat_now=0.5, opp_space=2.4, space=3.0,
                         lookahead_favor_switch=True)
    assert dec.branch != "A_dlat"
    assert dec.side_override is None
    assert mon.has_switched is False
