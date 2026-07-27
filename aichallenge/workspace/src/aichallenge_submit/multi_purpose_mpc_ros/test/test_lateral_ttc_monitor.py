"""Unit tests for LateralTTCMonitor (pure Python, no rclpy).

Covers three fixes landed 2026-07-13〜07-14:
  - cleared緩和(B_cleared/C2_cleared): 45節
  - switchback margin>=0ガード: 46節
  - v_inst物理妥当性クランプ: 50節
Retained here (not in /tmp scratchpad) so the suite survives across sessions and
protects against regression permanently, per user request 2026-07-14.
"""
import pytest

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor, TTCDecision


def make_monitor(**overrides):
    """決定的に検証するため、既定でbeta=1.0・space_ema_alpha=1.0・
    min_trend_cycles=1とし、1回のshrinkで即座に閾値判定へ到達できるようにする
    (既存のデバウンス機能自体は本ファイルの対象外)。"""
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
             opp_space=1.0, vopp=3.0, dt=1.0, fwd_vid="car1",
             lookahead_favor_switch=False):
    """2周期分updateし、2周期目の結果を返す(1周期目=warmup)。"""
    mon.update(side=1, space=s_prev, opp_space=opp_space, fwd_dlat=fwd_dlat,
               fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
               lookahead_favor_switch=lookahead_favor_switch)
    return mon.update(side=1, space=s_now, opp_space=opp_space, fwd_dlat=fwd_dlat,
                       fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
                       lookahead_favor_switch=lookahead_favor_switch)


# ---------------------------------------------------------------------------
# 既存分岐の回帰(none/warmup/A/B/C1)
# ---------------------------------------------------------------------------

def test_side_zero_returns_none_branch():
    mon = make_monitor()
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=None, fwd_ds=None,
                      vopp=None, dt=1.0, fwd_vid=None)
    assert dec.branch == "none"


def test_first_call_returns_warmup():
    mon = make_monitor()
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", cleared=False)
    assert dec.branch == "warmup"


def test_switchback_branch_a_fires_on_wide_opposite_side():
    mon = make_monitor()
    # opp_space(3.0) >= switchback_space_m(2.35) かつ margin>=0(3.0>=2.6)
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0)
    assert dec.branch == "A"
    assert dec.side_override == -1


def test_c1_speed_cap_zone_when_not_cleared():
    mon = make_monitor()
    # residual=3.6-1.85=1.75, v_ema=1.0 -> ttc=1.75 (critical<1.75<=danger)
    dec = two_step(mon, s_prev=4.6, s_now=3.6, cleared=False)
    assert dec.branch == "C1"
    assert dec.v_safe_cap is not None


def test_is_side_by_side_takes_priority_over_giveup():
    mon = make_monitor()
    dec = two_step(mon, s_prev=2.3, s_now=1.3, cleared=True, fwd_dlat=1.0, fwd_ds=0.5)
    assert dec.branch == "B"
    assert dec.force_giveup is False


# ---------------------------------------------------------------------------
# cleared緩和 (45節): B_cleared/C2_cleared
# ---------------------------------------------------------------------------

def test_cleared_false_reproduces_original_bug_giveup_despite_separation():
    """fwd_dlat=2.5m(離れている)でもcleared=Falseだとgiveupする(修正前の挙動)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, cleared=False)
    assert dec.branch == "C2"
    assert dec.force_giveup is True


def test_cleared_true_bypasses_c1_speed_cap():
    """同じ縮小トレンドでも、cleared=Trueなら介入しない(修正の核心)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, cleared=True)
    assert dec.branch == "B_cleared"
    assert dec.force_giveup is False
    assert dec.v_safe_cap is None
    assert dec.cleared is True


def test_cleared_true_still_gives_up_below_physical_minimum():
    """cleared=Trueでも物理下限(cleared_space_m)を割れば最終防波堤としてgiveupする。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=2.3, s_now=1.3, cleared=True)
    assert dec.branch == "C2_cleared"
    assert dec.force_giveup is True


def test_cleared_omitted_defaults_to_false():
    """cleared省略時はcleared=Falseと同じ挙動(後方互換)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "C2"


def test_ttc_decision_default_cleared_is_false():
    assert TTCDecision().cleared is False


@pytest.mark.parametrize("cleared,expected_branch", [
    # 同一の縮小トレンド(v_ema=-1.0)でも、cleared=Falseは閾値1.85基準でttc=1.75s
    # (critical<1.75<=danger→C1で速度キャップ)、cleared=Trueは閾値1.45基準で
    # ttc=2.15s(>danger→stableで完全に介入なし)と、緩和の効果が段階的に出ることを
    # 確認する。
    (False, "C1"),
    (True, "stable"),
])
def test_cleared_widens_stable_zone_at_same_space(cleared, expected_branch):
    """同一条件でcleared=Trueの方が閾値が緩む(空きの解釈がより寛容になる)ことを確認。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=4.6, s_now=3.6, cleared=cleared)
    assert dec.branch == expected_branch


# ---------------------------------------------------------------------------
# switchback marginガード (46節)
# ---------------------------------------------------------------------------

def test_switchback_suppressed_when_opposite_side_narrower():
    """margin<0(反対側が現在側より狭い)場合は反転を抑制する。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=2.4)  # 2.4 < 2.6
    assert dec.branch != "A"
    assert dec.side_override is None
    assert dec.switchback_suppressed is True


def test_switchback_not_suppressed_when_opposite_side_wider():
    """margin>=0(回帰): 従来通り反転する。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=2.9)  # 2.9 > 2.6
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.switchback_suppressed is False


def test_switchback_margin_exact_zero_boundary_allows_switch():
    """margin=0ちょうどの境界値: >=なので反転を許可する。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=2.6)  # ==
    assert dec.branch == "A"


def test_switchback_below_absolute_threshold_unaffected_by_margin():
    """opp_spaceが絶対閾値(switchback_space_m)未満なら、margin以前にそもそも対象外(回帰)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=2.0)  # 2.0 < 2.35
    assert dec.branch != "A"
    assert dec.switchback_suppressed is False


# ---------------------------------------------------------------------------
# v_inst物理妥当性クランプ (50節)
# ---------------------------------------------------------------------------

def test_v_inst_clamp_engages_on_anomalous_tiny_dt():
    """dt異常等でv_instが物理的にあり得ない値になっても、v_inst_maxでクランプされる。"""
    mon = make_monitor()
    mon.update(side=1, space=2.70, opp_space=1.0, fwd_dlat=3.0, fwd_ds=4.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)
    dec = mon.update(side=1, space=2.68, opp_space=1.0, fwd_dlat=3.0, fwd_ds=4.0,
                      vopp=3.0, dt=0.001, fwd_vid="car1", cleared=True)
    assert dec.v_corridor_ema == pytest.approx(-5.0)
    assert dec.v_inst_clamped is True


def test_v_inst_clamp_symmetric_for_widening_direction():
    """拡大方向(正)の外れ値も同様にクランプされる。"""
    mon = make_monitor()
    mon.update(side=1, space=2.68, opp_space=1.0, fwd_dlat=3.0, fwd_ds=4.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)
    dec = mon.update(side=1, space=2.70, opp_space=1.0, fwd_dlat=3.0, fwd_ds=4.0,
                      vopp=3.0, dt=0.001, fwd_vid="car1", cleared=True)
    assert dec.v_corridor_ema == pytest.approx(5.0)
    assert dec.v_inst_clamped is True


def test_v_inst_clamp_does_not_affect_normal_dt():
    """正常なdtでの通常の縮小トレンドは、クランプの影響を受けない(回帰)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, cleared=False)
    assert dec.v_corridor_ema == pytest.approx(-1.0)
    assert dec.v_inst_clamped is False
    assert dec.branch == "C2"


def test_v_inst_clamp_boundary_exactly_at_max_not_flagged():
    """v_inst_maxちょうどの値はクランプ発動とみなさない(境界値、abs()>との厳密比較)。"""
    mon = make_monitor(v_inst_max=1.0)
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    # d_space = 2.0-3.0=-1.0, dt=1.0 -> v_inst_raw=-1.0 == v_inst_max(1.0)ちょうど
    dec = mon.update(side=1, space=2.0, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.v_inst_clamped is False
    assert dec.v_corridor_ema == pytest.approx(-1.0)


def test_v_inst_clamp_retroactive_0713_06_wp243():
    """0713-06 wp243-246実測値での遡及再計算(50-6節の手計算をコードで再現)。

    実測: v_ema(前周期)=-0.6552、当該周期でv_ema=-3.9094(異常値混入、giveup発火、
    ttc=0.325秒)。本番既定のbeta=0.15を使い、クランプ適用後の期待値
    v_ema≈-1.3069(手計算通り)、ttc_lat≈0.972秒>critical(0.8)でgiveup不発火となる
    ことを確認する。
    """
    mon = make_monitor(beta=0.15, cleared_space_m=1.45)  # betaのみ本番既定値に戻す
    mon._v_corridor_ema = -0.6552   # 実測の前周期値を直接注入
    mon._shrink_run = 79             # 実測ログのshrink_run
    mon._prev_vid = "d3"
    mon._space_ema = 2.74             # d_space≈-0.02(実測相当の縮小)を作るための直前値
    mon._prev_space = 2.74
    mon.is_side_by_side = False
    mon.has_switched = True           # 実測ではhas_switched=True(switchback対象外)
    dec = mon.update(side=1, space=2.720285965039487, opp_space=2.62,
                      fwd_dlat=2.997224205733176, fwd_ds=3.0193915201983543,
                      vopp=3.0, dt=0.001, fwd_vid="d3", cleared=True)
    assert dec.v_inst_clamped is True
    assert dec.v_corridor_ema == pytest.approx(-1.3069, abs=1e-3)
    assert dec.branch == "B_cleared"
    assert dec.force_giveup is False


# ---------------------------------------------------------------------------
# is_side_by_sideヒステリシス(水平展開、事象C対策、2026-07-14)
# ---------------------------------------------------------------------------
# 単一閾値(side_by_side_dlat_m)だと境界付近の測位ノイズでis_side_by_sideが周期ごとに
# 反転し、branch B(速度介入なし)⇔C1/C2(速度キャップ/強制giveup)がチャーンしうる
# (mpc_controller.py側の_ot_clearedが既に持つ二段階閾値と同じ発想で解消)。

def test_is_side_by_side_enters_at_narrow_threshold():
    """境界値: dlat<=side_by_side_dlat_m(1.6)ちょうどで入る(従来通り、回帰)。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.6, fwd_ds=0.5,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is True


def test_is_side_by_side_does_not_enter_without_prior_state_regression():
    """回帰: 一度も side_by_side に入っていなければ、enter閾値(1.6)は従来通り厳格
    (1.6を超えるdlatでは即Falseのまま、ヒステリシスは離脱側にのみ働く)。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.9, fwd_ds=0.5,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is False


def test_is_side_by_side_stays_true_inside_hysteresis_band():
    """一度Trueになった後、dlatがenter(1.6)とrelease(2.1)の間で変動しても
    is_side_by_sideはTrueを維持する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=0.5,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    for d in (1.9, 1.7, 2.0, 1.8):
        dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=d, fwd_ds=0.5,
                          vopp=3.0, dt=1.0, fwd_vid="car1")
        assert dec.is_side_by_side is True


def test_is_side_by_side_exits_only_past_release_threshold():
    """境界値: release閾値(2.1)ちょうどで解放する(厳密<比較なので2.1自体は解放側)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=0.5,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.1, fwd_ds=0.5,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is False


def test_is_side_by_side_ds_condition_still_required_regression():
    """回帰: dlatが狭くてもds(縦方向)条件を満たさなければside_by_sideにならない。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=5.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is False


def test_hysteresis_prevents_branch_churn_under_oscillating_dlat():
    """事象C再現+修正効果の定量確認: 縮小トレンド中にdlatがヒステリシス帯
    (1.6〜2.1)内で振動しても、is_side_by_side=Trueを維持しbranch=B/stableで
    安定する(C1/C2への抜き差しが起きない)。ヒステリシスを無効化(release_m=
    dlat_mに設定)すると同じ入力列でC1→C2まで抜け出てしまうことを対比で示す。"""
    mon = make_monitor()
    mon.update(side=1, space=10.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=0.5,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    spaces = [8.0, 6.0, 4.0, 2.0]
    dlats = [1.9, 1.7, 2.0, 1.8]   # 測位ノイズ相当、全てヒステリシス帯(1.6〜2.1)内で振動
    branches = []
    for s, d in zip(spaces, dlats):
        dec = mon.update(side=1, space=s, opp_space=1.0, fwd_dlat=d, fwd_ds=0.5,
                          vopp=3.0, dt=1.0, fwd_vid="car1")
        assert dec.is_side_by_side is True
        branches.append(dec.branch)
    assert set(branches) <= {"stable", "B"}   # C1/C2への抜けは一切発生しない

    # 対比: ヒステリシス無効化(release_m == dlat_m)だと同じ入力列でC1/C2まで抜ける
    mon_no_hyst = make_monitor(side_by_side_dlat_release_m=1.6)
    mon_no_hyst.update(side=1, space=10.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=0.5,
                        vopp=3.0, dt=1.0, fwd_vid="car1")
    branches_no_hyst = []
    for s, d in zip(spaces, dlats):
        dec = mon_no_hyst.update(side=1, space=s, opp_space=1.0, fwd_dlat=d, fwd_ds=0.5,
                                  vopp=3.0, dt=1.0, fwd_vid="car1")
        branches_no_hyst.append(dec.branch)
    assert "C1" in branches_no_hyst or "C2" in branches_no_hyst


# ---------------------------------------------------------------------------
# is_side_by_side: fwd_ds再チェック撤廃(0714-04実測、2026-07-14)
# ---------------------------------------------------------------------------
# 一度is_side_by_side=Trueに達した後は、fwd_dlatのヒステリシスのみで離脱判定する
# (fwd_dsはエントリー判定にのみ必要で、離脱まで毎周期要求すると真横到達直後の
# fwd_dsの1周期だけの跨ぎでis_side_by_sideが反転し、蓄積済みのshrink_runに基づく
# C2が露出して危険水準ではないspaceで強制giveupが発火しうる)。

def test_0714_04_ds_single_cycle_blip_does_not_exit_side_by_side():
    """0714-04実測再現(wp141付近): dlat=1.04, ds=0.9986でis_side_by_side=Trueに到達後、
    次周期でdsが1.0をわずかに超えても(dlatが小さいままなら)is_side_by_sideを維持する。
    修正前はdsの単発跨ぎで即Falseへ反転し、直後のgiveupを誘発していた。"""
    mon = make_monitor()
    dec1 = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.04, fwd_ds=0.9986,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec1.is_side_by_side is True
    dec2 = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.10, fwd_ds=1.05,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec2.is_side_by_side is True  # dsが1.0を超えたが、dlatは小さいまま→維持
    assert dec2.branch not in ("C1", "C2")  # C1/C2へ露出しない


def test_entry_still_requires_both_dlat_and_ds_conditions_regression():
    """回帰: 初めてis_side_by_sideへ入る際は、従来通りdlat・ds両方の条件を要求する
    (2026-07-11修正の「縦に離れているのに横位置だけ近い」誤判定防止は維持)。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is False  # dlatは小さいがdsが遠い→エントリーしない


def test_ds_no_longer_gates_exit_once_side_by_side_regression():
    """回帰: 一度side_by_sideに入った後、dsが大きく開いても(dlatが小さいままなら)
    離脱しない(dlatのヒステリシスのみで離脱判定する設計を明示的に確認)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=0.5,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=1.0, fwd_ds=10.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.is_side_by_side is True


# ---------------------------------------------------------------------------
# giveup_space_m: along_lane_need(1.85m)→along_min_widthへの統一(フローチャート
# ギャップ②、2026-07-14)
# ---------------------------------------------------------------------------

def test_retroactive_0714_05_wp182_giveup_no_longer_fires_with_relaxed_threshold():
    """0714-05実測wp182の遡及再計算: space=2.881894182376756、v_corridor_ema=
    -1.3631452405285913(実測クランプ後値)、cleared=False、is_sbs=False。
    旧giveup_space_m(1.85m)ではttc_lat≈0.757秒<=critical(0.8秒)でC2(強制giveup)が
    発火していた(実測と一致)。新giveup_space_m(along_min_width=1.45m)では
    ttc_lat≈1.050秒>0.8秒となりC2は発火せず、C1(速度キャップのみ)に留まる。

    2026-07-15追記(has_rescued導入に伴う再検証): このepisodeの実測opp_space
    (3.2308770654948007)はswitchback_space_m(2.35)以上かつspace(2.881894182376756)
    以上——つまりこのepisodeの実際の計測値自体が、0715-03で発見したものと全く同じ
    「反対側が現在側より広い」パターンを満たしていた。そのためmon_old(旧閾値1.85m
    を再現したもの)は、has_rescued機構が無かった当時はC2(強制giveup)に終わって
    いたが、本機構を持つ現行コードでこの当時の値を再生するとA_rescueが発火し
    giveupを回避する。つまりhas_rescued修正は0715-03だけでなく、この0714-05の
    episodeも遡及的に救えていたことになる(④過去バグへの遡及効果がもう1件増える)。
    本番のgiveup_space_m(=1.45m、mon_new側)ではそもそもcritical閾値に到達しない
    ためhas_rescuedは無関係のまま(dec_newの結果は従来通りC1)。"""
    space_now = 2.881894182376756
    v_ema_target = -1.3631452405285913
    space_prev = space_now - v_ema_target * 1.0  # dt=1.0, beta=1.0(make_monitor既定)で
                                                  # d_space/dt = v_ema_target になるよう逆算

    mon_old = make_monitor(giveup_space_m=1.85)
    mon_old.update(side=1, space=space_prev, opp_space=3.2308770654948007,
                   fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
                   vopp=7.3, dt=1.0, fwd_vid="d3")  # warmup
    mon_old.has_switched = True  # 実測: このepisodeでは既にswitchback済み(分岐A対象外)
    dec_old = mon_old.update(side=1, space=space_now, opp_space=3.2308770654948007,
                              fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
                              vopp=7.3, dt=1.0, fwd_vid="d3")
    assert dec_old.v_corridor_ema == pytest.approx(v_ema_target, abs=1e-6)
    assert dec_old.ttc_lat == pytest.approx(0.7571, abs=1e-3)
    # has_rescued導入後: opp_space(3.23)>=switchback_space_m(2.35)かつ>=space(2.88)を
    # 満たすため、critical到達時にA_rescueが介入しC2までは到達しない(下記の
    # has_rescued専用テスト群でこの分岐自体は個別に検証済み)。
    assert dec_old.branch == "A_rescue"
    assert dec_old.force_giveup is False
    assert dec_old.side_override == -1

    # giveup_space_mの効果のみを純粋に切り出す比較用: has_rescuedの救済枠を
    # 使い切っている(=このepisode内で既に一度救済済み)場合は、本当にrescueの
    # 選択肢が無くなり、旧閾値ではC2まで到達することを確認する(純粋比較の保持)。
    mon_old_no_rescue = make_monitor(giveup_space_m=1.85)
    mon_old_no_rescue.update(side=1, space=space_prev, opp_space=3.2308770654948007,
                              fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
                              vopp=7.3, dt=1.0, fwd_vid="d3")  # warmup
    mon_old_no_rescue.has_switched = True
    mon_old_no_rescue.has_rescued = True  # 救済枠を使い切った状態を再現
    dec_old_no_rescue = mon_old_no_rescue.update(
        side=1, space=space_now, opp_space=3.2308770654948007,
        fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
        vopp=7.3, dt=1.0, fwd_vid="d3")
    assert dec_old_no_rescue.branch == "C2"
    assert dec_old_no_rescue.force_giveup is True

    mon_new = make_monitor(giveup_space_m=1.45)  # along_min_width
    mon_new.update(side=1, space=space_prev, opp_space=3.2308770654948007,
                   fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
                   vopp=7.3, dt=1.0, fwd_vid="d3")
    mon_new.has_switched = True
    dec_new = mon_new.update(side=1, space=space_now, opp_space=3.2308770654948007,
                              fwd_dlat=0.5635323917091817, fwd_ds=4.0460318615392055,
                              vopp=7.3, dt=1.0, fwd_vid="d3")
    assert dec_new.ttc_lat == pytest.approx(1.0504, abs=1e-3)
    assert dec_new.branch == "C1"          # 強制giveupではなく速度キャップのみに緩和
    assert dec_new.force_giveup is False


# ---------------------------------------------------------------------------
# has_rescued: giveup直前の最終救済スイッチバック (2026-07-15)
# ---------------------------------------------------------------------------
# 背景: 通常のswitchback(branch=A)は1エピソード1回のみ(has_switched)だが、実測
# (0715-03, t=16.90〜23.42)で「最初のswitchbackを既に消費した後、断念(giveup)に
# 至る過程で反対側が現在側より明確に広がる」逆転が発生し、既存のswitchback判定式
# (opp_space>=switchback_space_m AND opp_space>=space)を満たすにも関わらず
# has_switched消費済みのため反転できず、そのまま強制giveup(急ブレーキ)に至った。
# has_rescuedは「もはや断念寸前(ttc_lat<=ttc_critical_s)」という最終局面に限り、
# 既存のswitchback判定式をそのまま再利用して1エピソード1回だけ側変更を許可する。

def test_normal_switchback_branch_a_unaffected_by_has_rescued_field():
    """回帰: 通常のswitchback(has_switched=False時のbranch=A)はhas_rescued追加後も
    従来通り発火し、has_rescuedフィールド自体はFalseのまま(消費されない)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.has_rescued is False
    assert mon.has_rescued is False


def test_rescue_does_not_fire_before_ttc_reaches_critical():
    """回帰: ttc_latがdanger〜critical間(C1域)に留まっている間は、opp_space>=space
    かつhas_switched=Trueであってもrescueは発火しない(断念寸前の局面に厳密に限定
    されるべきで、一般的な2回目switchback許可ではないことを確認)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=3.6, opp_space=3.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec = mon.update(side=1, space=3.0, opp_space=3.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    # residual=3.0-1.85=1.15, v_ema=-0.6 -> ttc=1.9167s (critical<ttc<=danger)
    assert dec.ttc_lat == pytest.approx(1.9167, abs=1e-3)
    assert dec.branch == "C1"
    assert dec.side_override is None
    assert mon.has_rescued is False


def test_rescue_fires_when_has_switched_already_consumed_and_ttc_critical():
    """本修正の核心: has_switched=True(最初のswitchbackを既に消費)かつ
    ttc_lat<=ttc_critical_sに達した局面で、opp_space>=switchback_space_mかつ
    opp_space>=spaceを満たせば、強制giveup(C2)の代わりにbranch=A_rescueで
    側変更を許可する。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    # residual=2.2-1.85=0.35, v_ema=-0.5 -> ttc=0.7s(<=critical 0.8s)
    assert dec.ttc_lat == pytest.approx(0.7, abs=1e-3)
    assert dec.branch == "A_rescue"
    assert dec.side_override == -1
    assert dec.force_giveup is False
    assert mon.has_rescued is True


def test_rescue_does_not_fire_when_opposite_side_narrower_at_critical_moment():
    """回帰: 同じcritical局面でも、opp_space<space(反対側の方が狭い)であれば
    rescueは発火せず、従来通りC2(強制giveup)に落ちる(既存のmarginガードと
    同じ安全側の性質を保持する)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=4.6, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec = mon.update(side=1, space=3.0, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    # residual=3.0-1.85=1.15, v_ema=-1.6 -> ttc=0.71875s(<=critical)
    # opp_space(2.5)>=switchback_space_m(2.35)だが、opp_space(2.5)<space(3.0)
    assert dec.ttc_lat <= 0.8
    assert dec.branch == "C2"
    assert dec.force_giveup is True
    assert dec.side_override is None
    assert mon.has_rescued is False


def test_rescue_fires_at_most_once_per_episode():
    """境界値: rescueは1エピソード1回のみ(has_rescued消費)。1回目のrescue発火後、
    同じ条件(opp_space>=space)がもう一度critical局面で成立しても、2回目は
    通常のC2(強制giveup)まで落ちる(反転の使い過ぎで際限なくスイッチし続ける
    ことを防ぐ)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec1 = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec1.branch == "A_rescue"
    assert mon.has_rescued is True

    # rescue発火時にprev_space/space_ema/v_corridor_ema/shrink_runがリセットされる
    # ため、再度warmup->critical到達の2周期を踏ませる。
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec2 = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec2.branch == "C2"          # has_rescued消費済みのため今度は本当にgiveup
    assert dec2.force_giveup is True
    assert dec2.side_override is None


def test_retroactive_0715_03_t2342_rescue_prevents_giveup():
    """遡及検証(0715-03実測、t=23.42秒相当): t=16.90に最初のswitchback(branch=A)を
    既に消費した後、t=21.5〜22.9にかけてopp_space(1.86→3.14m)がspace(3.14→2.54m)を
    追い越して逆転し、t=23.42時点でspace≈2.58m/opp_space≈3.10m/ttc_lat≈0.71秒
    (<=critical)に到達した。giveup_space_mは本番配線通りalong_min_width(1.45m)を
    使用する。

    旧コード(has_rescued機構が存在しない状態、ここではhas_rescued=Trueを注入して
    強制的に無効化することでシミュレート): opp_space>=switchback_space_m(2.35)かつ
    opp_space>=space(3.10>=2.58)を満たしていたにも関わらず、has_switched消費済みの
    ため側変更ができず、そのままC2(強制giveup=急ブレーキ、実測と一致)に至る。

    新コード(has_rescued): 全く同じ数値でA_rescueが発火し、giveupを回避して
    インサイドへスイッチバックする。"""
    space_now = 2.58
    v_ema_target = -1.6  # 残差1.13m基準でttc_lat≈0.706秒(実測≈0.71秒に近似)
    space_prev = space_now - v_ema_target * 1.0

    mon_legacy = make_monitor(giveup_space_m=1.45)  # along_min_width(本番配線値)
    mon_legacy.update(side=1, space=space_prev, opp_space=3.10, fwd_dlat=2.5,
                       fwd_ds=3.0, vopp=3.0, dt=1.0, fwd_vid="car1")
    mon_legacy.has_switched = True
    mon_legacy.has_rescued = True  # has_rescued機構が無かった当時の状態を模擬
    dec_legacy = mon_legacy.update(side=1, space=space_now, opp_space=3.10,
                                    fwd_dlat=2.5, fwd_ds=3.0, vopp=3.0, dt=1.0,
                                    fwd_vid="car1")
    assert dec_legacy.ttc_lat <= 0.8
    assert dec_legacy.branch == "C2"
    assert dec_legacy.force_giveup is True

    mon_fixed = make_monitor(giveup_space_m=1.45)
    mon_fixed.update(side=1, space=space_prev, opp_space=3.10, fwd_dlat=2.5,
                      fwd_ds=3.0, vopp=3.0, dt=1.0, fwd_vid="car1")
    mon_fixed.has_switched = True
    dec_fixed = mon_fixed.update(side=1, space=space_now, opp_space=3.10,
                                  fwd_dlat=2.5, fwd_ds=3.0, vopp=3.0, dt=1.0,
                                  fwd_vid="car1")
    assert dec_fixed.branch == "A_rescue"
    assert dec_fixed.force_giveup is False
    assert dec_fixed.side_override == -1


# ---------------------------------------------------------------------------
# new_side_blocked: switchbackトークン浪費バグの根治 (79節, 2026-07-16)
# ---------------------------------------------------------------------------
# 背景: 77節のcurvature veto(_switchback_curvature_veto)は、mpc_controller.py側で
# update()の戻り値(side_override)を受け取った後に別途反転を止めていたが、
# has_switched/has_rescuedは本メソッド内で既にTrueへ更新済みだったため、veto発生
# 時にこのエピソードの反転トークンが両方とも浪費されていた(0715-08実測: wp61で
# switchback成功→wp73でA_rescueがcurvature vetoされる→わずか1.3秒後のwp75、
# ttc=0.09秒という選択肢皆無の緊急giveupに至った)。
# 本修正はnew_side_blockedを呼び出し元がupdate()の入力として渡すことで、
# _switchback_eligible/_rescue_eligible自体を不成立にし、has_switched/has_rescuedを
# 消費させない。

def test_new_side_blocked_prevents_normal_switchback_and_preserves_has_switched():
    """本修正の中核(通常A): new_side_blocked=Trueの場合、opp_space>=spaceでも
    branch=Aは発火せず、has_switchedは消費されない(Falseのまま)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0)
    # two_step()はnew_side_blockedを渡さないヘルパーのため、直接呼び出しで確認する。
    mon2 = make_monitor()
    mon2.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                vopp=3.0, dt=1.0, fwd_vid="car1")
    dec2 = mon2.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=True)
    assert dec2.branch != "A"
    assert dec2.side_override is None
    assert dec2.switchback_suppressed is True
    assert dec2.switchback_curvature_blocked is True
    assert mon2.has_switched is False


def test_new_side_blocked_false_regression_normal_switchback_still_fires():
    """回帰: new_side_blocked省略時(既定False)は従来通りbranch=Aが発火する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=False)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.switchback_curvature_blocked is False
    assert mon.has_switched is True


def test_new_side_blocked_prevents_rescue_and_preserves_has_rescued():
    """本修正の中核(A_rescue): has_switched=True済みの状態でrescue条件を満たしても、
    new_side_blocked=Trueならbranch=A_rescueは発火せず、has_rescuedは消費されない。
    代わりに即座にforce_giveup(C2)が発火する(断念以外の選択肢が無いことを
    正しく認識し、その場で安全に断念する)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=True)
    assert dec.ttc_lat == pytest.approx(0.7, abs=1e-3)
    assert dec.branch == "C2"
    assert dec.force_giveup is True
    assert dec.switchback_curvature_blocked is True
    assert mon.has_rescued is False  # 消費されない(旧バグではここでTrueになっていた)


def test_new_side_blocked_false_regression_rescue_still_fires():
    """回帰: new_side_blocked=Falseなら75節で実装した通りbranch=A_rescueが発火する。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=False)
    assert dec.branch == "A_rescue"
    assert dec.force_giveup is False
    assert mon.has_rescued is True


def test_token_preserved_allows_rescue_once_curvature_clears():
    """デッドロック解消の実証: 1回目のcritical局面でcurvatureにブロックされても、
    has_rescuedは温存され、直後にcurvatureが晴れた(new_side_blocked=False)瞬間に
    正しくA_rescueが発火できる。旧バグではここでhas_rescued=True(浪費済み)のため
    2回目も必ずforce_giveupしていた。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec1 = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=True)
    assert dec1.branch == "C2"
    assert mon.has_rescued is False
    dec2 = mon.update(side=1, space=1.9, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=False)
    assert dec2.ttc_lat <= 0.8
    assert dec2.branch == "A_rescue"
    assert dec2.side_override == -1
    assert mon.has_rescued is True


def test_retroactive_0715_08_wp73_curvature_blocked_forces_earlier_safer_giveup():
    """遡及検証(0715-08実測、wp73/t=1784123798.67): 実測ログの厳密な値
    (space=2.539849473743266, opp_space=3.1395490293245225,
    ttc_lat=0.5673648914278576, has_switched=True, cleared=False, thr=1.85)を
    実際のLateralTTCMonitor.update()へ投入し、旧コード/新コードの分岐差を検証する。

    旧コード(new_side_blockedを知らない): 実際のログ通りbranch=A_rescueを返し、
    has_rescuedをTrueへ消費する。mpc_controller.py側の後付けvetoがこの反転の
    「実行」だけを止めるため、この時点(ttc=0.567秒、まだ余裕がある)ではgiveupが
    発生せず、1.3秒後のwp75(ttc=0.0897秒、実測値、ほぼ選択肢皆無)まで危機的状況が
    先送りされていた。

    新コード(new_side_blocked=Trueをupdate()の入力として渡す): rescueが不成立と
    正しく判断され、即座にforce_giveup(branch=C2)が発火する。これはwp75の緊急
    giveup(ttc=0.0897秒)より遥かに安全な余裕(ttc=0.567秒、6倍以上)での断念であり、
    「断念を先送りしてぎりぎりで慌てる」から「断念が必要な瞬間に安全マージンを
    持って断念する」への改善を定量的に示す。"""
    space_now = 2.539849473743266
    opp_space = 3.1395490293245225
    v_ema_target = -1.2158832599020322  # 実測ログのv_ema値と一致
    space_prev = space_now - v_ema_target * 1.0

    mon_old = make_monitor()
    mon_old.has_switched = True
    mon_old.update(side=1, space=space_prev, opp_space=opp_space, fwd_dlat=1.83,
                   fwd_ds=5.91, vopp=1.88, dt=1.0, fwd_vid="d3")
    dec_old = mon_old.update(side=1, space=space_now, opp_space=opp_space,
                              fwd_dlat=1.83, fwd_ds=5.91, vopp=1.88, dt=1.0,
                              fwd_vid="d3")  # new_side_blocked省略=False(旧コード相当)
    assert dec_old.ttc_lat == pytest.approx(0.5673648914278576, abs=1e-6)
    assert dec_old.branch == "A_rescue"
    assert mon_old.has_rescued is True  # 旧コード: 実行されずとも消費される

    mon_new = make_monitor()
    mon_new.has_switched = True
    mon_new.update(side=1, space=space_prev, opp_space=opp_space, fwd_dlat=1.83,
                   fwd_ds=5.91, vopp=1.88, dt=1.0, fwd_vid="d3")
    dec_new = mon_new.update(side=1, space=space_now, opp_space=opp_space,
                              fwd_dlat=1.83, fwd_ds=5.91, vopp=1.88, dt=1.0,
                              fwd_vid="d3", new_side_blocked=True)
    assert dec_new.ttc_lat == pytest.approx(0.5673648914278576, abs=1e-6)
    assert dec_new.branch == "C2"
    assert dec_new.force_giveup is True
    assert mon_new.has_rescued is False

    _wp75_real_ttc = 0.08966128185265791  # 実測: 旧コードが実際に緊急giveupした時点
    assert dec_new.ttc_lat > _wp75_real_ttc * 6  # 新コードは遥かに安全な余裕で断念する


# ---------------------------------------------------------------------------
# 通常switchback(branch A)のcleared中抑制 → 83節でrevert (2026-07-16)
# ---------------------------------------------------------------------------
# 経緯: 81節で発見したep5型の問題(0716-02、cleared到達後にswitchbackが確保済みの
# 横間隔を無駄に破棄する)に対し、82節で`_switchback_eligible`へ`not cleared`を
# 追加した。しかし0716-03実走行で、clearedは実際には長時間(数秒〜十数秒)成立
# したままの局面が大半であり、その間もコリドーは変動し続けるにも関わらず通常
# switchbackが一律に封じられた結果、反対側が明確に(時に2倍以上)広くても安全に
# 反転できず、コリドー限界(offset=±3.00)に長時間張り付いたままCOLLISION-SUSPECTED
# が多発した(Lap1 wp217-240・wp196-205、Lap2 wp202-232で計5件)。82節の前提
# (cleared中の反転は得るものがない)は実測で反証されたため、ユーザー承認のもと
# clearedによるガードを削除しrevertした。当時導入した診断用フィールド
# switchback_cleared_blocked(常にFalse・実質未使用)は2026-07-17に削除した。

def test_switchback_branch_a_fires_even_when_cleared_true_reverted():
    """本revertの核心: 82節で追加したclearedガードを削除したため、通常switchback
    成立条件(margin>=0・is_side_by_side=False・has_switched=False)を満たせば
    cleared=Trueであっても82節導入前と同様に発火する(0716-03実測で確認した、
    反対側が明確に広いのに反転できず衝突する回帰の再発防止)。
    opp_space=3.2はmargin=0.6となり、84節のcleared_marginガード(必要margin=0.5)も
    上回るため、本テストは84節実装後も引き続き成立する。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.2, cleared=True)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert mon.has_switched is True


def test_switchback_branch_a_still_fires_when_not_cleared_regression():
    """回帰: cleared=Falseでも従来通りbranch=Aが発火する(cleared引数の有無で
    結果が変わらないことの確認)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0, cleared=False)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert mon.has_switched is True


def test_rescue_still_fires_when_cleared_true_safety_valve_unaffected_by_revert():
    """回帰: cleared=True中の断念直前最終救済(A_rescue)は、82節導入前後・
    revert後を通じて常に安全弁として機能する(本revertの影響を受けない)。"""
    mon = make_monitor()
    mon.has_switched = True  # 通常switchbackは既に消費済みという前提
    mon.update(side=1, space=2.0, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)  # warmup
    dec = mon.update(side=1, space=1.6, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)
    assert dec.ttc_lat == pytest.approx(0.375, abs=1e-3)
    assert dec.branch == "A_rescue"
    assert dec.side_override == -1
    assert dec.force_giveup is False
    assert mon.has_rescued is True


def test_retroactive_0716_03_wp202_collision_cluster_now_escapes_via_normal_switchback():
    """遡及検証(0716-03実測、Lap2 wp202-204、t=497.8〜499.1): 実測space=2.60〜2.82・
    opp_space=4.09〜4.29(反対側が現在側より最大65%広い)という状況で、82節時点の
    コードは`switchback_suppressed reason=cleared`を連発し、offset=-3.00
    (コリドー限界)に7秒以上張り付いたままt=498.42にCOLLISION-SUSPECTED
    (v: 3.63→1.77 m/s)が発生していた。

    82節時点相当(cleared=Trueでガードされる)ではswitchbackが不成立のまま
    留まり続ける一方、revert後(本コード)ではcleared=Trueであっても通常
    switchbackが発火し、反対側への安全な反転が選択されることを確認する。"""
    space_now = 2.60
    opp_space = 4.29
    v_ema_target = -0.8  # ttc_danger_s(2.0s)以内に収まる収縮速度
    space_prev = space_now - v_ema_target * 1.0

    mon = make_monitor()
    mon.update(side=-1, space=space_prev, opp_space=opp_space, fwd_dlat=2.0,
               fwd_ds=5.0, vopp=3.0, dt=1.0, fwd_vid="d2", cleared=True)
    dec = mon.update(side=-1, space=space_now, opp_space=opp_space,
                      fwd_dlat=2.0, fwd_ds=5.0, vopp=3.0, dt=1.0,
                      fwd_vid="d2", cleared=True)
    assert dec.branch == "A"


# ---------------------------------------------------------------------------
# 84節①: cleared中のmargin 0.5m追加要求 (2026-07-16、ユーザー承認済み設計)
# ---------------------------------------------------------------------------
# 背景: 82節(cleared中は通常switchbackを一律禁止)は0716-03実測で重大な回帰を
# 引き起こし83節でrevertした。しかし81節で見つけたep5型の問題(0716-02、
# margin=0.05というほぼ得るものが無い反転でclearedを無駄に破棄する)自体は
# 未対処のまま残っていた。83節revert後のコード(margin>=0のみ要求)に対し、
# 「cleared中のみ、margin(opp_space-space)が既存のswitchback_space_m-
# giveup_space_m(0.5m、新規パラメータ0個)以上であること」を追加要求する。
# 0716-02実測ep5(margin=0.05)は狙い撃ちで抑制される一方、0716-03実測の
# 衝突クラスタ(margin=1.6〜1.8)は従来通り反転できる
# (test_retroactive_0716_03_wp202_collision_cluster_now_escapes_via_normal_switchbackで
# margin=1.69のケースが既に確認済み)。

def test_switchback_blocked_by_cleared_margin_when_margin_too_thin_regression():
    """本修正の核心: cleared=True かつ margin(opp_space-space)=0.4(<0.5)という、
    0716-02実測ep5(margin=0.05)と同種の「ほぼ得るものが無い反転」を狙い撃ちで
    抑制する。margin>=0のみを見ていた83節revert直後のコードならbranch=Aが
    発火していたはずの場面。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0, cleared=True)
    assert dec.branch != "A"
    assert dec.side_override is None
    assert mon.has_switched is False  # 消費されない(不成立として扱われる)


def test_switchback_cleared_margin_diagnostic_flag_set():
    """検証ロギング用: cleared_margin理由でswitchbackが抑制された周期は
    switchback_suppressed=True かつ switchback_cleared_margin_blocked=True となり、
    margin(cleared=False時)やk_cornerとはログ上区別できる。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0, cleared=True)
    assert dec.switchback_suppressed is True
    assert dec.switchback_cleared_margin_blocked is True
    assert dec.switchback_curvature_blocked is False


def test_switchback_fires_at_exactly_the_cleared_margin_threshold():
    """境界値: margin(opp_space-space)がちょうど0.5(switchback_space_m-
    giveup_space_m)であれば、`>=`により発火する(既存の境界値規約に合わせる)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.1, cleared=True)  # margin=0.5ちょうど
    assert dec.branch == "A"
    assert dec.side_override == -1


def test_cleared_margin_gate_does_not_apply_when_not_cleared_regression():
    """回帰: cleared=Falseであれば、margin=0.4(<0.5)でも従来通りmargin>=0のみで
    発火する(cleared_marginガードはcleared=True中のみに限定)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0, cleared=False)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert mon.has_switched is True


def test_retroactive_0716_02_ep5_margin_0_05_now_correctly_suppressed():
    """遡及検証(0716-02実測、Lap3 ep5、wp318→wp320): 実測space=2.65・
    opp_space=2.70(margin=0.05)・cleared=Trueという、81節で発見した「確保済みの
    横間隔を無駄に破棄する」反転が、84節①の実装により正しく抑制されることを
    確認する。83節revert直後のコード(margin>=0のみ)ではbranch=Aが発火していた
    (test_retroactive_0716_02_ep5_wp320_switchback_would_have_discarded_clearanceの
    dec_old相当)。"""
    space_now = 2.65
    opp_space = 2.70
    v_ema_target = -0.6
    space_prev = space_now - v_ema_target * 1.0

    mon = make_monitor()
    mon.update(side=1, space=space_prev, opp_space=opp_space, fwd_dlat=1.94,
               fwd_ds=4.0, vopp=3.0, dt=1.0, fwd_vid="d3", cleared=True)
    dec = mon.update(side=1, space=space_now, opp_space=opp_space,
                      fwd_dlat=1.94, fwd_ds=4.0, vopp=3.0, dt=1.0,
                      fwd_vid="d3", cleared=True)
    assert dec.branch != "A"
    assert dec.side_override is None
    assert dec.switchback_cleared_margin_blocked is True


def test_rescue_unaffected_by_cleared_margin_gate_safety_valve_preserved():
    """回帰(重要): A_rescue(断念直前の最終救済)はcleared_marginガードの対象外で
    あり、margin<0.5でもcleared中に発火する(既存コメント「cleared中でも
    最終防波堤として残す」を維持)。opp_space=2.5はswitchback_space_m(2.35)以上
    (_rescue_eligibleの絶対下限)を満たしつつ、margin(2.5-1.6=0.9)は0.5を上回るが、
    これはcleared_marginガードが最初からA_rescueには適用されない
    (別コードパスである)ことの確認であり、margin自体の大小は本質ではない。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.0, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)  # warmup
    dec = mon.update(side=1, space=1.6, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", cleared=True)
    assert dec.branch == "A_rescue"
    assert dec.side_override == -1
    assert mon.has_rescued is True


# ---------------------------------------------------------------------------
# 84節②: カーブ先回り切り替え (2026-07-16、ユーザー承認済み設計)
# ---------------------------------------------------------------------------
# 背景: ユーザー要望「コーナーでイン/アウトが大きくずれる場面では、通常の
# margin判定を待たずに早めに切り替えてほしい。ただし直線走行中の無駄な反転は
# 不要」に対応する。呼び出し元(mpc_controller.py)が既存の
# _switchback_curvature_veto()を現在側/反対側の両方に適用して算出した
# lookahead_favor_switch(現在側が前方カーブで閉じ、反対側は閉じない)をTTC
# Monitorへ渡すと、通常のmargin/cleared_margin判定を待たずに早期反転
# (branch=A_lookahead)を許可する。

def test_lookahead_favor_switch_no_longer_bypasses_negative_margin_regression():
    """107節案A'で修正: 旧実装はlookahead_favor_switch=Trueならmargin(opp_space-space)
    が負(反対側の方が狭い)でも無条件に早期反転していたが、これは84節①が過去ログ
    61件の定量検証で確立した「opp_space<space(反対側の方が狭い)への反転は
    21件中21件(100%)が15秒以内にgiveupに終わる」というmargin>=0ガードを
    素通ししてしまうバグだった(0718-05実測T=701.15で実際に発火し、直後に
    wall_slow・COLLISION-SUSPECTEDが発生)。修正後はmargin>=0が常時必須となり、
    このケース(margin=-0.6)ではlookahead_favor_switch=Trueでも反転しない。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=3.0, opp_space=2.4, cleared=False,
                   lookahead_favor_switch=True)  # margin=-0.6(反対側の方が狭い)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"
    assert dec.side_override is None
    assert mon.has_switched is False


def test_lookahead_favor_switch_still_bypasses_cleared_margin_when_absolute_margin_ok():
    """107節案A'の回帰確認: lookahead_favor_switchが引き続きバイパスできるのは
    _cleared_margin_ok(clearedの間だけ要求される追加0.5mバッファ)のみであり、
    絶対的なmargin>=0自体は満たしている場面では84節②本来の「早めに反転する」
    設計意図が維持されることを確認する。"""
    mon = make_monitor()
    # margin = opp_space(2.4) - space(2.0) = 0.4 (>=0だがcleared_margin_required
    # である switchback_space_m(2.35) - giveup_space_m(1.85) = 0.5 には届かない)。
    # s_prevからの縮小量を大きくし、ttc_lat<=ttc_danger_sへ到達させてswitchback
    # 判定ブロックまで進ませる(cleared=Trueのためstable判定はcleared_space_m基準)。
    dec = two_step(mon, s_prev=3.6, s_now=2.0, opp_space=2.4, cleared=True,
                   lookahead_favor_switch=True)
    assert dec.branch == "A_lookahead"
    assert dec.side_override == -1
    assert mon.has_switched is True


def test_retroactive_0718_05_wp110_negative_margin_lookahead_no_longer_fires():
    """遡及検証(107節、0718-05実測T=701.15, wp=110): config.yaml実値
    (switchback_space_m=1.95, giveup_space_m=1.45)のもとで、opp_space=2.07・
    space=2.19(margin=-0.12)・lookahead_favor_switch=Trueという実測条件を再現する。
    旧実装ではbranch=A_lookaheadが発火し(実測ログ通り)、直後にwall_slow・
    COLLISION-SUSPECTEDが発生した。修正後はmargin<0のため発火しない。"""
    mon = make_monitor(switchback_space_m=1.95, giveup_space_m=1.45)
    dec = two_step(mon, s_prev=2.3, s_now=2.19, opp_space=2.07, cleared=True,
                   lookahead_favor_switch=True)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"
    assert dec.side_override is None


def test_lookahead_still_requires_absolute_switchback_space_m_floor_regression():
    """回帰: lookahead_favor_switch=Trueでも、反対側の絶対的な空き幅が
    switchback_space_m(2.35m)を割っていれば発火しない(物理下限は維持)。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=3.0, opp_space=2.0, cleared=False,
                   lookahead_favor_switch=True)  # opp_space(2.0)<switchback_space_m(2.35)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"
    assert dec.side_override is None


def test_lookahead_false_behaves_identically_to_before_regression():
    """回帰: lookahead_favor_switch=False(既定値)では、84節②導入前と全く同じ
    margin判定のみで結果が決まる。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=3.0, opp_space=2.4, cleared=False,
                   lookahead_favor_switch=False)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"  # margin=-0.6のため通常判定でも不成立(margin抑制)


def test_branch_is_plain_a_not_lookahead_when_reactive_condition_also_satisfied():
    """回帰: lookahead_favor_switch=Trueであっても、通常のmargin判定
    (cleared_margin込み)も同時に満たしていれば、branchは"A"のまま
    ("A_lookahead"にはならない)。診断ログ上、本当に先回りが必要だった周期
    (通常判定では不成立だった周期)だけをA_lookaheadとして区別するため。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0, cleared=False,
                   lookahead_favor_switch=True)  # margin=0.4>=0、cleared=Falseなので通常判定も成立
    assert dec.branch == "A"


def test_lookahead_respects_has_switched_token_regression():
    """回帰: lookahead_favor_switch=Trueでも、既にhas_switched=True
    (このepisodeで通常switchbackを消費済み)であれば発火しない
    (1エピソード1回のトークン制限は先回り切り替えにも適用される、無駄な
    多重反転を防ぐ設計)。"""
    mon = make_monitor()
    mon.has_switched = True
    dec = two_step(mon, s_prev=3.6, s_now=3.0, opp_space=2.4, cleared=False,
                   lookahead_favor_switch=True)
    assert dec.branch != "A_lookahead"
    assert dec.side_override is None


def test_lookahead_favor_switch_field_echoed_in_diagnostics():
    """検証ロギング用: lookahead_favor_switchはbranchに関わらず(発火有無を
    問わず)_diag経由で毎周期エコーされ、TTCDecision.lookahead_favor_switch
    としてログで確認できる。"""
    mon = make_monitor()
    dec = two_step(mon, s_prev=3.6, s_now=3.0, opp_space=2.4, cleared=False,
                   lookahead_favor_switch=True)
    assert dec.lookahead_favor_switch is True


# ---------------------------------------------------------------------------
# 92節①②: カーブ起因のTTC猶予(C1_deferred) + 最終救済の適格緩和(A_rescue_relaxed)
# (2026-07-17、ユーザー承認済み設計)
# ---------------------------------------------------------------------------
# 背景: A_rescue(既存)は反転先(-side)がカーブで閉じるかどうか(new_side_blocked)は
# 見ているが、現在側(side)がカーブで閉じつつあるだけの一時的な収縮を、相手に
# 実際に迫られている場合と区別せずにC2(強制giveup)していた。①は「現在側が
# カーブ起因で閉じる」と分かっている間だけTTC猶予を与え、②は猶予が対象外/
# 使い切った最終局面に限り、反対側の必要幅を物理下限まで緩和して再挑戦する。

def test_curvature_deferred_c1_when_current_side_closing_ahead():
    """①の核心: current_side_closing_ahead=Trueの間、TTC-criticalに達しても
    即座にforce_giveupせず、branch=C1_deferredで縦速度キャップのみ課す。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec.ttc_lat == pytest.approx(0.75, abs=1e-3)  # ttc_critical_s(0.8)を下回る
    assert dec.branch == "C1_deferred"
    assert dec.force_giveup is False
    assert dec.side_override is None
    assert dec.critical_curvature_run == 1
    assert dec.v_safe_cap is not None  # 完全ノーガードにはしない(C1相当のキャップ)


def test_curvature_deferred_exhausts_after_min_trend_cycles_times_two():
    """①の境界: 猶予はmin_trend_cyclesの2倍(make_monitorはmin_trend_cycles=1
    のため2周期)まで。使い切った後は通常通りforce_giveupする
    (opp_spaceが物理下限未満のため②も不成立の場面)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec1 = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec1.branch == "C1_deferred"
    assert dec1.critical_curvature_run == 1
    dec2 = mon.update(side=1, space=1.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec2.branch == "C2"
    assert dec2.force_giveup is True
    assert dec2.critical_curvature_run == 2


def test_curvature_run_resets_when_corridor_recovers_to_stable():
    """回帰: 猶予中にコーナーを抜けてv_corridor_emaが回復(>=0)すると、
    shrink_runと同様にcritical_curvature_runも0へリセットされる
    (次に別のカーブで猶予が必要になった際、前回の消費分を持ち越さない)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec1 = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec1.critical_curvature_run == 1
    dec2 = mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec2.branch == "stable"
    assert dec2.critical_curvature_run == 0


# ---------------------------------------------------------------------------
# 144節続報: C1_deferred猶予中、逃げ道(switchback/rescue)が封鎖されている場合の
# キャップ強化(2026-07-20、0720-05実測d1 wp320-326の衝突分析)
# ---------------------------------------------------------------------------
# 背景: 0720-05実測(第二コーナー、t=857.9〜858.5)で、C1_deferred(92節①、当時
# 設計通り発火・v_cap=2.14相当)が猶予を与えていたにもかかわらず、猶予中ずっと
# switchback_suppressed reason=wall(new_side_wall_blocked=True、逃げ道が壁で
# 封鎖)だったため、猶予を使い切った直後(curvature_run=6でC2_clearedへ
# フォールバック)に実質衝突した。voppベースの緩いキャップ(相手と同程度の速度を
# 許容)では、逃げ道が無い状況での安全マージンとして不十分だった。
# 対処: new_side_wall_blocked/new_side_room_blockedがTrueの間は、他の緊急層
# (wall_slow/footprint_risk)と共通のwall_slow_speed(より保守的)をmin()で
# 重ねる。switchback/rescue自体の成立条件は一切変更しない(82/83節の教訓:
# 逃げ道の成立条件を広く制限すると重大な回帰を招いた実測がある)。

def test_c1_deferred_cap_strengthened_when_escape_wall_blocked():
    """①非矛盾性の核心: 通常はvoppベースのキャップ(vopp - caution_margin)だが、
    new_side_wall_blocked=Trueの間はwall_slow_speed(デフォルト2.0)がmin()で
    優先される(vopp=3.0なら通常2.44、wall_slow_speed=2.0の方が保守的)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
               new_side_wall_blocked=True)
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
                      new_side_wall_blocked=True)
    assert dec.branch == "C1_deferred"
    assert dec.v_safe_cap == pytest.approx(2.0, abs=1e-6)
    assert "逃げ道封鎖" in dec.v_safe_cap_label


def test_c1_deferred_cap_strengthened_when_escape_room_blocked():
    """new_side_room_blocked(相手位置先読み)でも同様にキャップが強化される。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
               new_side_room_blocked=True)
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
                      new_side_room_blocked=True)
    assert dec.branch == "C1_deferred"
    assert dec.v_safe_cap == pytest.approx(2.0, abs=1e-6)


def test_c1_deferred_cap_unaffected_when_escape_available_regression():
    """回帰防止: 逃げ道が封鎖されていない通常のC1_deferred(既存92節①の挙動)は
    無変更のまま、voppベースの緩いキャップを使う。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec.branch == "C1_deferred"
    assert dec.v_safe_cap == pytest.approx(max(0.0, 3.0 - 2.0 / 3.6), abs=1e-6)
    assert "逃げ道封鎖" not in dec.v_safe_cap_label


def test_c1_deferred_cap_takes_stricter_of_vopp_and_wall_slow():
    """②非冗長性: 単純な置き換えではなくmin()であることを確認する
    (相手がwall_slow_speedより遅い場合は、より保守的なvoppベースの値を維持)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=1.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
               new_side_wall_blocked=True)
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=1.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
                      new_side_wall_blocked=True)
    _expected_vopp_cap = max(0.0, 1.0 - 2.0 / 3.6)  # ≈0.444、wall_slow_speed(2.0)より保守的
    assert dec.v_safe_cap == pytest.approx(_expected_vopp_cap, abs=1e-6)


def test_switchback_rescue_eligibility_unchanged_by_wall_slow_speed_param():
    """①非矛盾性(82/83節の教訓の直接確認): wall_slow_speedパラメータの新設が
    switchback(branch=A)自体の成立条件に一切影響しないことを回帰確認する。"""
    mon = make_monitor(wall_slow_speed=0.5)  # 極端な値でも通常switchbackは無影響
    dec = two_step(mon, s_prev=3.6, s_now=2.6, opp_space=3.0)
    assert dec.branch == "A"
    assert dec.side_override == -1


def test_wall_slow_speed_defaults_to_existing_constant():
    """②非冗長性: 新規パラメータのデフォルト値が既存config.yamlのwall_slow_speed
    (2.0)と一致することを確認する(呼び出し元が省略しても既存の想定値になる)。"""
    mon = LateralTTCMonitor()
    assert mon.wall_slow_speed == pytest.approx(2.0)


def test_retroactive_0720_05_d1_wp320_scenario():
    """④過去ログへの遡及効果: 0720-05実測d1 wp320-326(t=858.229、直前のOTログで
    space系がcurvature_run=1に至った際の実測値、vopp≈2.7)相当の値を投入し、
    対処後はwall_slow_speedベースのキャップ(2.0)が効き、対処前のvoppベースの
    値(≈2.14、より緩い)より保守的になることを確認する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.75, fwd_ds=3.0,
               vopp=2.7, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True,
               new_side_wall_blocked=True)
    dec_blocked = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.75,
                              fwd_ds=3.0, vopp=2.7, dt=1.0, fwd_vid="car1",
                              current_side_closing_ahead=True, new_side_wall_blocked=True)
    _old_vopp_cap = max(0.0, 2.7 - 2.0 / 3.6)  # ≈2.144(対処前の実際の値)
    assert dec_blocked.v_safe_cap == pytest.approx(2.0, abs=1e-6)
    assert dec_blocked.v_safe_cap < _old_vopp_cap


def test_relaxed_rescue_fires_when_opp_space_between_cleared_and_switchback():
    """②の核心: current_side_closing_ahead=False(カーブ起因ではない=相手に
    実際に迫られている)場合、①の猶予は適用されず、通常のA_rescue(閾値
    switchback_space_m=2.35)は不成立でも、物理下限(cleared_space_m=1.45)を
    上回っていればA_rescue_relaxedで側反転する。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "A_rescue_relaxed"
    assert dec.side_override == -1
    assert mon.has_rescued is True


# ---------------------------------------------------------------------------
# 102節続報(検討の記録、2026-07-18): 案2(current_side_closing_ahead=True時に
# 閾値をswitchback_space_mへ引き上げる)は撤回済み
# ---------------------------------------------------------------------------
# 背景: 0718-03実測(同一地点wp≈333〜335)で、「猶予切れ直後にcleared_space_m基準の
# 緩い判定で反転→約0.8秒後に反転先も同じカーブで閉じ強制giveup」というパターンが
# 2回再現した(102節)。案2として、current_side_closing_ahead=True時のみ閾値を
# switchback_space_mへ引き上げる変更を試みたが、これは上のA_rescue(_rescue_eligible、
# 常時switchback_space_m基準で本ブロック冒頭で先に評価される)と完全に同一の式に
# なり、A_rescueが不成立だった時点で必ずこちらも不成立になる到達不能コードだと
# 実装検証で判明したため撤回した(②非冗長性の観点で不可)。根本対処は103節で
# OpponentSpeedMap.lat_mean(対象車両IDごとの学習済み走行ライン)を使った先読み
# room計算として設計中(Phase 0はmpc_controller.py側の診断ロギングのみ)。
# 以下は「current_side_closing_aheadの値に関わらず閾値はcleared_space_mのまま」
# という撤回後(=92節②のオリジナル)の挙動を確認する回帰テスト。

def test_relaxed_rescue_threshold_unaffected_by_current_side_closing_ahead_regression():
    """回帰防止(102節続報の撤回): current_side_closing_ahead=Trueであっても、
    A_rescue_relaxedの閾値は従来通りcleared_space_m(物理下限)のままであり、
    switchback_space_mへ引き上げられていないことを確認する。opp_space=1.7は
    cleared_space_m(1.45)以上・switchback_space_m(2.35)未満のため、閾値が
    switchback_space_mに引き上げられていれば不成立になるはずだが、撤回後は
    成立してA_rescue_relaxedが発火する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec1 = mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec1.branch == "C1_deferred"  # 猶予中(1周期目)
    dec2 = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec2.branch == "A_rescue_relaxed"  # 猶予切れ(2周期目)、cleared_space_m基準で成立
    assert dec2.side_override == -1


def test_relaxed_rescue_still_blocked_below_physical_floor():
    """回帰(安全下限の維持): 反対側が物理下限(cleared_space_m)未満の場合は
    ②でも救済されず、従来通りforce_giveupする。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec.branch == "C2"
    assert dec.force_giveup is True


# ---------------------------------------------------------------------------
# 107節案C(103節Phase 1): A_rescue_relaxedへ相手位置認識(new_side_room_blocked)
# を統合。呼び出し元(mpc_controller.py)が_opponent_room_ahead()(OpponentSpeedMap.
# lat_meanベースの先読みroom)をupdate()呼び出し前に計算し渡す。0718-03実測
# (wp≈333〜335、A_rescue_relaxed発火の約0.8秒後に反転先も同じカーブで閉じ
# 強制giveup)を踏まえ、反転先が近い将来閉じると予測される場合はA_rescue_relaxed
# 自体を不成立にし、force_giveup(C2/C2_cleared)へ直接倒す。
# ---------------------------------------------------------------------------

def test_new_side_room_blocked_prevents_relaxed_rescue_and_falls_to_giveup():
    """②の核心: new_side_room_blocked=True(反転先が学習済み走行ライン基準の
    先読みで物理下限未満になると予測)の場合、opp_space自体はcleared_space_mを
    上回っていてもA_rescue_relaxedは不成立となり、force_giveupへ直接倒れる。
    has_rescuedは消費されない(無駄な反転トークン浪費を避ける)。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=True)
    assert dec.branch != "A_rescue_relaxed"
    assert dec.branch == "C2"
    assert dec.force_giveup is True
    assert dec.side_override is None
    assert mon.has_rescued is False


def test_new_side_room_blocked_false_regression_relaxed_rescue_still_fires():
    """回帰: new_side_room_blocked=False(既定、または学習データ未取得時の
    fail-open)では、従来通りA_rescue_relaxedが発火する(102節続報撤回後の
    挙動が維持される)。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=False)
    assert dec.branch == "A_rescue_relaxed"
    assert dec.side_override == -1
    assert mon.has_rescued is True


def test_new_side_room_blocked_omitted_defaults_to_false_regression():
    """回帰: new_side_room_blockedを省略した既存の全呼び出しパターン
    (学習データが無いvid/waypoint、あるいは呼び出し元が未対応の場合)は
    引き続き既定False(素通し)として扱われ、既存テストの挙動を変えない。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")  # new_side_room_blocked省略
    assert dec.branch == "A_rescue_relaxed"


def test_new_side_room_blocked_does_not_affect_strict_rescue_regression():
    """回帰(スコープ確認): new_side_room_blockedは_rescue_relaxed_eligible
    のみに影響し、通常のA_rescue(厳格閾値、_rescue_eligible)には一切
    影響しない(107節案Cのスコープ通り)。"""
    mon = make_monitor()
    mon.has_switched = True  # 通常switchback消費済み、A_rescue経路を評価させる
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=True)
    assert dec.branch == "A_rescue"
    assert dec.side_override == -1


def test_critical_curvature_run_resets_on_side_switch_regression():
    """回帰(トークン整合性監査、2026-07-17): critical_curvature_runは、
    既存のshrink_run/v_corridor_ema/prev_space/space_emaと同じ「側が変わったら
    トレンド追跡を仕切り直す」原則に従い、A_rescue発火(側反転)時にも0へ
    リセットされる。リセット漏れがあると、反転後の新しい側で①のカーブ猶予
    (min_trend_cycles*2周期分)が前の側の消費分だけ短く扱われてしまう。"""
    mon = make_monitor()
    mon.has_switched = True  # 通常switchbackは消費済みという前提
    # 旧側でカーブ猶予を1周期分消費してからA_rescueが発火する場面を作る。
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    dec1 = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec1.branch == "C1_deferred"
    assert mon._critical_curvature_run == 1
    # 同じ危機の中でA_rescueが発火(反対側が十分広がった)。
    dec2 = mon.update(side=1, space=1.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1", current_side_closing_ahead=True)
    assert dec2.branch == "A_rescue"
    assert mon._critical_curvature_run == 0  # 消費分を持ち越さない


def test_relaxed_rescue_shares_has_rescued_token_with_strict_rescue():
    """回帰: ②は既存のhas_rescuedトークンを共有し、1エピソード1回のみ。
    通常のA_rescue(厳格な閾値)が既に発火済みなら、②の緩和条件を満たす
    別の危機が来ても再度は発火しない。"""
    mon = make_monitor()
    mon.has_switched = True  # 通常switchback(branch A)は既に消費済みという前提
    # 1回目の危機: 厳格な閾値(switchback_space_m=2.35)を満たしA_rescueで消費。
    mon.update(side=1, space=2.5, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec1 = mon.update(side=1, space=1.5, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec1.branch == "A_rescue"
    assert mon.has_rescued is True
    # 2回目の危機(反転後の新しい側で再び縮小): 緩和条件(opp_space=1.7)を
    # 満たしていても、has_rescued消費済みのため②は発火しない。
    mon.update(side=-1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec2 = mon.update(side=-1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec2.branch == "C2"
    assert dec2.force_giveup is True


# ---------------------------------------------------------------------------
# 100節(Tier1裁定の外出し、2026-07-18、ユーザー承認済み設計): 92節続報の
# C1_obstacle_yield分岐はlateral_ttc_monitor.pyから削除した。障害物クラス
# 判定(vopp<opp_obstacle_speed)はコリドー物理量と無関係な外部裁定であり、
# 「C1のv_capを候補として使うかどうか」の決定はmpc_controller.py側(Tier1)へ
# 移設した(構造的検証はtest_switchback_token_wiring.py参照)。本モジュールは
# 障害物クラスに関わらず常にC1のv_capを計算して返す、という単純な契約に戻る。
# ---------------------------------------------------------------------------

def test_fwd_is_obstacle_class_no_longer_a_valid_kwarg_regression():
    """回帰防止: fwd_is_obstacle_classパラメータが復活していないことを、
    渡した場合にTypeErrorとなることで直接確認する。"""
    mon = make_monitor()
    with pytest.raises(TypeError):
        mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                   vopp=0.5, dt=1.0, fwd_vid="car1", fwd_is_obstacle_class=True)


def test_c1_always_applies_velocity_cap_regardless_of_opponent_speed_range():
    """回帰(100節): 相手速度が障害物クラスの範囲(低速)であっても、LAT-TTC自体は
    従来通りC1のvopp基準キャップを計算して返す(候補を使うかどうかの裁定は
    もう本モジュールの責務ではない)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=0.5, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=3.0, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=0.5, dt=1.0, fwd_vid="car1")
    assert dec.branch == "C1"
    assert dec.v_safe_cap == pytest.approx(max(0.0, 0.5 - 2.0 / 3.6), abs=1e-3)


def test_c1_reports_cap_for_obstacle_class_opponent_after_tier1_extraction():
    """回帰(100節): 0717-02実測(t=199.6〜201.1秒付近)と同じ相手速度域
    (vopp=0.3、障害物クラス相当)でコリドーがC1のTTC窓(critical_s<ttc_lat<=danger_s)
    まで縮小した場合、LAT-TTC自体は変わらずC1を発火しv_capを計算する(旧
    C1_obstacle_yieldはここでは発生しない)。この値を実際にv_safe候補として
    使うかどうかの裁定はmpc_controller.py側で行われる
    (test_switchback_token_wiring.pyのTIER1-C1-YIELD関連テスト参照)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.50, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=0.3, dt=1.0, fwd_vid="d2")
    dec = mon.update(side=1, space=2.90, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=0.3, dt=1.0, fwd_vid="d2")
    assert dec.branch == "C1"
    assert dec.v_safe_cap == pytest.approx(max(0.0, 0.3 - 2.0 / 3.6), abs=1e-3)


# ---------------------------------------------------------------------------
# new_side_room_blocked を通常switchback(branch=A/A_lookahead)へ適用(2026-07-19、
# 103/106/107節が確立・部分対処した非対称性の解消、120節続報)。
# 107節はnew_side_room_blockedをA_rescue_relaxedの適格判定にのみ統合し、通常の
# switchback(branch=A/A_lookahead)には意図的に未適用のままだった。0719-03実測
# (lap2 wp189-193)で、この未対処範囲(branch=A)が原因の追突→壁挟み込みを確認した
# ため、new_side_blocked(静的曲率veto)と全く同じ位置・同じ考え方で追加する。
# ---------------------------------------------------------------------------

def test_new_side_room_blocked_prevents_normal_switchback_and_preserves_has_switched():
    """本修正の中核(通常A): new_side_room_blocked=Trueの場合、margin>=0・
    curvature veto無しでもbranch=Aは発火せず、has_switchedは消費されない。
    shrinkはttc_lat(=1.917s)がcritical(0.8s)とdanger(2.0s)の間に収まるよう
    選定し、厳格A_rescue(_rescue_eligibleはnew_side_room_blockedを見ない
    設計、107節でA_rescue_relaxedのみへ意図的にスコープ)への意図しない
    フォールスルーを避けて、branch=A抑止のみを純粋に検証する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=3.0, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=True)
    assert dec.branch != "A", dec.branch
    assert dec.side_override is None
    assert dec.switchback_suppressed is True
    assert dec.switchback_room_blocked is True
    assert dec.switchback_curvature_blocked is False  # 曲率vetoとは独立した理由であることの確認
    assert mon.has_switched is False


def test_new_side_room_blocked_false_regression_normal_switchback_still_fires():
    """回帰: new_side_room_blocked省略時(既定False)は従来通りbranch=Aが発火する
    (103/107節の設計方針=データ未学習時はfail-open、を損なわないことの確認)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=False)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.switchback_room_blocked is False
    assert mon.has_switched is True


def test_new_side_room_blocked_prevents_lookahead_switchback():
    """new_side_room_blocked=Trueは、lookahead_favor_switch=True(84節②の
    早期切替経路)でも反転をブロックする(margin>=0を満たす通常のlookahead成立
    条件下でも、roomガードは独立に効く)。config.yaml実値
    (switchback_space_m=1.95)を使用。shrink(3.0→2.3)はttc_lat(=1.214s)が
    critical(0.8s)とdanger(2.0s)の間に収まるよう選定し、厳格A_rescueへの
    フォールスルーを避ける。"""
    mon = make_monitor(switchback_space_m=1.95, giveup_space_m=1.45)
    mon.update(side=1, space=3.0, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.3, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1",
                      lookahead_favor_switch=True, new_side_room_blocked=True)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"
    assert dec.switchback_room_blocked is True
    assert mon.has_switched is False


def test_new_side_room_blocked_does_not_affect_rescue_relaxed_path_regression():
    """回帰(非干渉性): new_side_room_blockedは既に103/107節でA_rescue_relaxedの
    適格判定(呼び出し元mpc_controller.py側で別途_rescue_relaxed_eligibleへ
    組み込み済み)に統合されている。本モジュール内のbranch=A/A_lookaheadへの
    今回の追加が、A_rescue系列(has_switched=True後の最終救済)の挙動には
    一切影響しないことを確認する(has_switched=True済みなら_switchback_eligible
    自体がFalseのため、new_side_room_blockedの新しいガードは経由しない)。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_room_blocked=True)
    # branch=A_rescue自体はhas_switched=True後の別ブロック(_rescue_eligible)で
    # 判定され、本モジュール内ではnew_side_room_blockedを参照しないため無変更。
    assert dec.branch == "A_rescue"
    assert mon.has_rescued is True


def test_retroactive_0719_03_lap2_wp189_squeeze_now_blocked():
    """遡及検証(120節続報、0719-03実測lap2 wp189-193): 速い対戦車d3との間合いが
    fwd_dlat=0.017mまで潰れる直前、wp189でbranch=A(space=2.52, opp_space=3.32,
    margin=+0.80、実測値そのまま)が発火し、その後ttc=0.0の緊急giveup→
    Rfree≈0.19mで約3秒の壁挟み込みに至った。room先読みがこの反転先の物理的な
    崩壊を検知できていた(new_side_room_blocked=True)と仮定した場合、
    margin>=0で他条件を満たしていても反転が抑制されることを確認する。
    config.yaml実値(switchback_space_m=1.95/giveup_space_m=1.45)を使用し、
    space側はwp189実測値(2.52)へ収束するshrink(3.22→2.52)とすることで、
    ttc_lat(=1.529s)がcritical(0.8s)とdanger(2.0s)の間に収まり、厳格
    A_rescueへのフォールスルーを避けてbranch=A抑止のみを検証できるようにする。"""
    mon = make_monitor(switchback_space_m=1.95, giveup_space_m=1.45)
    mon.update(side=-1, space=3.22, opp_space=3.32, fwd_dlat=3.27, fwd_ds=9.03,
               vopp=3.14, dt=1.0, fwd_vid="d3")
    dec = mon.update(side=-1, space=2.52, opp_space=3.32, fwd_dlat=3.27, fwd_ds=9.03,
                      vopp=3.14, dt=1.0, fwd_vid="d3", new_side_room_blocked=True)
    assert dec.branch != "A"
    assert dec.switchback_room_blocked is True
    assert mon.has_switched is False


# ---------------------------------------------------------------------------
# 125節(A-1): new_side_wall_blocked(動的コリドー、壁+占有格子込みの先読み
# veto)。new_side_blocked(静的曲率)・switchback_curvature_blockedと全く同じ
# スコープ(通常switchback branch=A/A_lookahead、厳格A_rescue、緩和
# A_rescue_relaxedの3箇所すべて)へ適用する点が、A_rescue_relaxedにのみ
# スコープされたnew_side_room_blockedとの設計上の違い(ユーザー承認済み)。
# ---------------------------------------------------------------------------

def test_new_side_wall_blocked_prevents_normal_switchback_and_preserves_has_switched():
    """本修正の中核(通常A): new_side_wall_blocked=Trueの場合、margin>=0・
    curvature/room veto無しでもbranch=Aは発火せず、has_switchedは消費されない。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=3.0, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_wall_blocked=True)
    assert dec.branch != "A", dec.branch
    assert dec.side_override is None
    assert dec.switchback_suppressed is True
    assert dec.switchback_wall_blocked is True
    assert dec.switchback_curvature_blocked is False  # 曲率vetoとは独立した理由
    assert mon.has_switched is False


def test_new_side_wall_blocked_false_regression_normal_switchback_still_fires():
    """回帰: new_side_wall_blocked省略時(既定False)は従来通りbranch=Aが発火する。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_wall_blocked=False)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.switchback_wall_blocked is False
    assert mon.has_switched is True


def test_new_side_wall_blocked_prevents_lookahead_switchback():
    """new_side_wall_blocked=Trueは、lookahead_favor_switch=True(84節②の
    早期切替経路)でも反転をブロックする。"""
    mon = make_monitor(switchback_space_m=1.95, giveup_space_m=1.45)
    mon.update(side=1, space=3.0, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.3, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1",
                      lookahead_favor_switch=True, new_side_wall_blocked=True)
    assert dec.branch != "A_lookahead"
    assert dec.branch != "A"
    assert dec.switchback_wall_blocked is True
    assert mon.has_switched is False


def test_new_side_wall_blocked_also_prevents_strict_rescue():
    """本修正の中核(厳格A_rescue): switchback_room_blockedと異なり、
    new_side_wall_blockedは_rescue_eligible(厳格閾値)にも効く
    (curvature_blockedと同じスコープ、ユーザー承認済み設計)。"""
    mon = make_monitor()
    mon.has_switched = True  # 通常switchback消費済み、A_rescue経路を評価させる
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")  # warmup
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_wall_blocked=True)
    assert dec.branch != "A_rescue"
    assert dec.side_override is None
    assert dec.switchback_wall_blocked is True
    assert mon.has_rescued is False


def test_new_side_wall_blocked_false_regression_strict_rescue_still_fires():
    """回帰: new_side_wall_blocked=False(既定)では厳格A_rescueは従来通り発火する。"""
    mon = make_monitor()
    mon.has_switched = True
    mon.update(side=1, space=2.7, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=2.2, opp_space=2.5, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_wall_blocked=False)
    assert dec.branch == "A_rescue"
    assert mon.has_rescued is True


def test_new_side_wall_blocked_prevents_relaxed_rescue_and_falls_to_giveup():
    """本修正の中核(A_rescue_relaxed): new_side_wall_blocked=Trueの場合、
    opp_space自体はcleared_space_mを上回っていてもA_rescue_relaxedは不成立と
    なり、force_giveupへ直接倒れる。has_rescuedは消費されない。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", new_side_wall_blocked=True)
    assert dec.branch != "A_rescue_relaxed"
    assert dec.branch == "C2"
    assert dec.force_giveup is True
    assert dec.side_override is None
    assert mon.has_rescued is False


def test_new_side_wall_blocked_omitted_defaults_to_false_regression():
    """回帰: new_side_wall_blockedを省略した既存の全呼び出しパターンは
    引き続き既定False(素通し)として扱われ、既存テストの挙動を変えない。"""
    mon = make_monitor()
    mon.update(side=1, space=2.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec = mon.update(side=1, space=1.5, opp_space=1.7, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")  # new_side_wall_blocked省略
    assert dec.branch == "A_rescue_relaxed"


# ---------------------------------------------------------------------------
# 127節続報(2026-07-20): footprint_risk(fwd_dlat<along_min_widthかつ
# fwd_ds<along_min_length、呼び出し元mpc_controller.pyが算出)。space/opp_space
# (壁〜相手の隙間の広さ、自車の現在位置を含まない式)が「安全」を報告していても、
# 実測(0720-01予選ログwp173、space=3.12mなのにfwd_dlat=0.198m)のような矛盾を
# トレンド判定を待たず即座に検知するための最優先オーバーライド。
# ---------------------------------------------------------------------------

def test_footprint_risk_forces_immediate_giveup_even_with_large_safe_looking_space():
    """本修正の中核: footprint_risk=Trueの場合、space/opp_spaceが十分広い
    (=trend判定なら安全とみなされる)値であっても、初回呼び出しから即座に
    branch=FOOTPRINT_RISK・force_giveup=Trueとなる(warmup等を経由しない)。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=5.0, opp_space=5.0, fwd_dlat=0.2, fwd_ds=0.5,
                      vopp=3.0, dt=1.0, fwd_vid="car1", footprint_risk=True)
    assert dec.branch == "FOOTPRINT_RISK"
    assert dec.force_giveup is True
    assert dec.footprint_risk_triggered is True
    assert dec.side_override is None  # 反転は行わない(保守的設計)


def test_footprint_risk_false_regression_normal_warmup_still_applies():
    """回帰: footprint_risk=False(既定)では、既存のwarmup(初回呼び出し)挙動が
    変わらない。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=5.0, opp_space=5.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1", footprint_risk=False)
    assert dec.branch == "warmup"


def test_footprint_risk_omitted_defaults_to_false_regression():
    """回帰: footprint_riskを省略した既存の全呼び出しパターンは引き続き
    既定False(素通し)として扱われる。"""
    mon = make_monitor()
    dec = mon.update(side=1, space=5.0, opp_space=5.0, fwd_dlat=2.5, fwd_ds=3.0,
                      vopp=3.0, dt=1.0, fwd_vid="car1")  # footprint_risk省略
    assert dec.branch == "warmup"


def test_footprint_risk_takes_priority_over_ongoing_trend_state():
    """footprint_risk=Trueは、既にshrink_run等のトレンドが蓄積している最中でも
    即座にオーバーライドし、トレンド状態をリセットする(次回footprint_risk=Falseの
    周期でトレンドが汚染されないようにするため)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    assert mon._shrink_run > 0  # トレンド蓄積済みであることの前提確認
    dec = mon.update(side=1, space=2.6, opp_space=1.0, fwd_dlat=0.1, fwd_ds=0.3,
                      vopp=3.0, dt=1.0, fwd_vid="car1", footprint_risk=True)
    assert dec.branch == "FOOTPRINT_RISK"
    assert mon._shrink_run == 0
    assert mon._critical_curvature_run == 0


def test_footprint_risk_does_not_fire_when_side_is_zero_regression():
    """回帰: side==0(まだOVERTAKING側が確定していない)の場合、footprint_risk=True
    でもside==0チェックが先に評価され、branch=noneのまま(既存の最優先ガードを
    footprint_riskが迂回しないことの確認)。"""
    mon = make_monitor()
    dec = mon.update(side=0, space=None, opp_space=None, fwd_dlat=0.1, fwd_ds=0.2,
                      vopp=None, dt=1.0, fwd_vid=None, footprint_risk=True)
    assert dec.branch == "none"
