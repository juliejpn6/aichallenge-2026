"""Unit tests for the k_corner switchback veto override (157節、2026-07-22)。

背景: 0721-02/0722-03の実測で、_switchback_curvature_veto(静的トラック曲率のみ)
がC2緊急giveupの約50〜56%で、反対側が実時間的に明らかに有利(opp_space>space)な
最中でも反転を一律に抑制していたことを確認した。82/83節の教訓(switchback適格性の
広範な制限緩和は4.3倍の衝突増・完走率半減という重大な回帰を招いた)を踏まえ、
適格性判定式自体(_switchback_eligible/_rescue_eligible)には手を入れず、通常の
switchback自体が既に要求している実測ベース閾値(switchback_space_m)を反転先の
opp_spaceが満たす場合に限り、静的曲率の懸念を上書きする狭い変更とした。

ユーザー指摘で発見した上流-下流の整合性問題: new_side_blocked(生の静的曲率判定)
はupdate()内部の可否判定と診断フィールド(switchback_curvature_blocked)の両方に
使われている。overrideをnew_side_blocked自体に混ぜ込むと、overrideが成立した
ケースでcurvature_blocked=Falseとログされ「そもそも曲率の懸念が無かった」ように
見えてしまい、将来の遡及検証(76節のwp297事例のように反転後に本当に閉じたか
追跡する)ができなくなる。この対処ではnew_side_blocked自体は無変更のまま渡し、
新規引数new_side_curvature_overrideと新規診断フィールド
switchback_curvature_overriddenを追加することで、「曲率は懸念ありだったが
実測で上書きして通した」ケースを診断上も区別できるようにした。

_lookahead_favor_switch(84節、早回り切り替え)も同じnew_side_blockedから
作られており、override非対応のままだと通常経路とlookahead経路で挙動が食い違う
ため、同じoverride条件を両方の消費先で共有するよう修正した(mpc_controller.py側、
構造検証テストで確認)。

lateral_ttc_monitor.pyはROS非依存のため直接importして検証する
(test_lateral_ttc_monitor.pyと同じ方針)。
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
             lookahead_favor_switch=False, new_side_blocked=False,
             new_side_curvature_override=False):
    mon.update(side=1, space=s_prev, opp_space=opp_space, fwd_dlat=fwd_dlat,
               fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
               lookahead_favor_switch=lookahead_favor_switch,
               new_side_blocked=new_side_blocked,
               new_side_curvature_override=new_side_curvature_override)
    return mon.update(side=1, space=s_now, opp_space=opp_space, fwd_dlat=fwd_dlat,
                       fwd_ds=fwd_ds, vopp=vopp, dt=dt, fwd_vid=fwd_vid, cleared=cleared,
                       lookahead_favor_switch=lookahead_favor_switch,
                       new_side_blocked=new_side_blocked,
                       new_side_curvature_override=new_side_curvature_override)


# --- 通常switchback(branch=A)経路 ---

def test_regression_curvature_block_without_override_still_suppresses():
    """回帰: overrideを渡さない(=デフォルトFalse)場合、従来通りcurvatureで
    反転が抑制されることを確認する(79節の元の挙動を維持)。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                    new_side_blocked=True)
    assert dec.branch != "A"
    assert dec.switchback_suppressed is True
    assert dec.switchback_curvature_blocked is True
    assert dec.switchback_curvature_overridden is False


def test_override_allows_switch_despite_curvature_block():
    """本修正の中核: new_side_blocked=Trueでもnew_side_curvature_override=True
    (反転先opp_spaceがswitchback_space_m以上)なら反転が成立する。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                    new_side_blocked=True, new_side_curvature_override=True)
    assert dec.branch == "A"
    assert dec.side_override == -1
    assert dec.switchback_curvature_overridden is True


def test_override_flag_has_no_effect_when_not_blocked():
    """回帰: new_side_blocked=Falseの場合、overrideフラグの値に関わらず通常通り
    反転する(overrideが余計な副作用を持たないことの確認)。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                    new_side_blocked=False, new_side_curvature_override=True)
    assert dec.branch == "A"
    assert dec.switchback_curvature_overridden is False  # not new_side_blocked なのでFalse


def test_curvature_blocked_diagnostic_stays_pure_when_overridden():
    """①非矛盾性の核心: overrideが成立してもswitchback_curvature_blockedは
    「今回抑制されなかった」ことを正しく反映する(=False、抑制ログの対象外)。
    switchback_curvature_overriddenのみが「曲率は懸念ありだった」事実を保持する。"""
    dec = two_step(make_monitor(), s_prev=3.6, s_now=2.6, opp_space=3.0,
                    new_side_blocked=True, new_side_curvature_override=True)
    assert dec.switchback_curvature_blocked is False
    assert dec.switchback_curvature_overridden is True


def test_wall_veto_still_blocks_even_with_curvature_override():
    """①非矛盾性: overrideはcurvature vetoのみに作用し、wall veto
    (new_side_wall_blocked、125節)には一切影響しないことを確認する。
    new_side_blocked自体はFalse(curvature側は懸念なし)にして、wall_blockedの
    単独効果だけを切り分ける(elifの優先順位でcurvature側が先に報告されるのを
    避けるため)。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1",
               new_side_blocked=False, new_side_curvature_override=True,
               new_side_wall_blocked=True)
    dec2 = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                       vopp=3.0, dt=1.0, fwd_vid="car1",
                       new_side_blocked=False, new_side_curvature_override=True,
                       new_side_wall_blocked=True)
    assert dec2.branch != "A"
    assert dec2.switchback_wall_blocked is True


# --- A_rescue経路 ---

def test_rescue_branch_blocked_by_curvature_without_override():
    """回帰: A_rescue経路でもoverride無しならcurvatureで抑制される。"""
    mon = make_monitor()
    # 通常switchbackを使い切る(has_switched=True)ためのAブランチ発火
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_a = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec_a.branch == "A"
    # 側が反転(-1)した状態で、断念寸前(ttc_critical以下)のA_rescueを試みる。
    mon.update(side=-1, space=2.0, opp_space=0.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_r = mon.update(side=-1, space=0.1, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1", new_side_blocked=True)
    assert dec_r.branch != "A_rescue"


def test_rescue_branch_allowed_with_override():
    """本修正: A_rescue経路もoverride成立時はcurvature blockを上書きできる。"""
    mon = make_monitor()
    mon.update(side=1, space=3.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_a = mon.update(side=1, space=2.6, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1")
    assert dec_a.branch == "A"
    mon.update(side=-1, space=2.0, opp_space=0.5, fwd_dlat=2.5, fwd_ds=3.0,
               vopp=3.0, dt=1.0, fwd_vid="car1")
    dec_r = mon.update(side=-1, space=0.1, opp_space=3.0, fwd_dlat=2.5, fwd_ds=3.0,
                        vopp=3.0, dt=1.0, fwd_vid="car1",
                        new_side_blocked=True, new_side_curvature_override=True)
    assert dec_r.branch == "A_rescue"
    assert dec_r.switchback_curvature_overridden is True


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証(157節)
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_source_override_reuses_existing_switchback_space_m_no_new_threshold():
    """②非冗長性: overrideの判定基準が既存のself._lat_ttc.switchback_space_mを
    再利用しており、新規の閾値を持たないことを確認する。"""
    idx = _SRC.index("_new_side_curvature_override = (")
    snippet = _SRC[idx:idx + 200]
    assert "self._lat_ttc.switchback_space_m" in snippet
    assert "_lat_opp_space" in snippet


def test_source_new_side_blocked_itself_unchanged():
    """①非矛盾性: _new_side_blocked自体の計算式(生の静的曲率判定)は
    無変更のまま_switchback_curvature_vetoの結果をそのまま使うことを確認する。"""
    idx = _SRC.index("_new_side_blocked = (self._switchback_curvature_veto(-self._ot_side)")
    snippet = _SRC[idx:idx + 160]
    assert "if self._ot_side != 0 else False" in snippet


def test_source_lookahead_favor_switch_uses_override_too():
    """①非矛盾性: _lookahead_favor_switch(84節、早回り切り替え経路)も
    _new_side_curvature_overrideを考慮するよう更新されており、通常経路と
    挙動が食い違わないことを確認する。"""
    idx = _SRC.index("_lookahead_favor_switch = _current_side_closing_ahead and (")
    snippet = _SRC[idx:idx + 150]
    assert "_new_side_curvature_override" in snippet


def test_source_update_call_passes_override_parameter():
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx:idx + 700]
    assert "new_side_blocked=_new_side_blocked," in snippet
    assert "new_side_curvature_override=_new_side_curvature_override," in snippet


def test_source_verification_log_includes_overridden_field():
    """③検証ロギング: switchback成功ログにcurvature_overridden=フィールドが
    含まれ、次回ログでoverride起因の反転を直接追跡できることを確認する。"""
    idx = _SRC.index('f"[LAT-TTC-ACT] switchback branch=')
    snippet = _SRC[idx:idx + 1500]
    assert "curvature_overridden={_lat_dec.switchback_curvature_overridden}" in snippet
