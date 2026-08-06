"""Unit tests for issue⑤①: STOPPING中の能動的空き確保バイアス
(160節続報、2026-07-22)。

背景: 157/158節で分析した第1〜2コーナー間の事故連鎖で、根本課題として
「検知を待たず最初から十分な間隔を確保しにいく」設計(issue⑤、能動的空き選択)
の欠如が160節で正式に整理された。従来は、追い越しを本エンゲージするまで
STOPPING中の追従(icc_stop)は横方向に一切間隔を作りにいかず、footprint_risk
(fwd_dlat<along_min_widthかつfwd_ds<along_min_length)という反応的な検知が
発火して初めて速度側でしか介入していなかった(154/155節)。

対処方針:
- 側の判定は新設せず、_evaluate_engage_readiness()が非OVERTAKING中に毎周期
  既に計算しているplan_side/plan_ok(_plan_passの結果)をそのまま使う。
  本エンゲージ時にself._ot_sideへ採用される値と完全に同一のため、
  switchbackで判明した判定層/実行層の乖離(159節)は生じない。
- オフセット量は新設のself._ot_proactive_bias_max(0.3m、_ot_d_offより
  十分小さい)を上限とし、_corr_bound_ahead()(147節、動的コリドー配列の
  先読み最小値)でクランプする。これはOVERTAKING本番と全く同じ式・同じ関数。
- 停止/低速の相手(vopp<opp_obstacle_speed、155節と同一閾値)にのみ適用。
- 発見された上流-下流不一致: use_obstacle_avoidance=Falseのままだと
  _corr_bound_ahead()が読むdbg_corr_ub_arr/lb_arrは静的テーブル(相手車の
  存在を無視)のままになり、クランプが実質機能しない「安全なふりをした
  無防備」状態になる。バイアス発動条件と同一周期でuse_obstacle_avoidance
  をTrueにし、コリドー急変によるinfeasible化を防ぐためOVERTAKING開始時と
  同じlateral_funnel_stepsも流用する。

mpc_controller.pyはrclpy依存のため直接importできないため、
test_footprint_taper.pyと同じ方針(純Pythonミラー関数+ソーステキストに
よる構造的検証)を用いる。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

OPP_OBSTACLE_SPEED = 6.0 / 3.6  # [m/s] 6km/h(既存閾値)
D_OFF = 3.0
PROACTIVE_BIAS_MAX = 0.3


def stopped_opp(fwd_vopp, opp_obstacle_speed=OPP_OBSTACLE_SPEED):
    """mpc_controller.py内の_stopped_opp計算式の複製ミラー(155/160節で共有)。"""
    return fwd_vopp is not None and fwd_vopp < opp_obstacle_speed


def lateral_target_and_side(state, ot_side, plan_ok, plan_side, fwd_vopp,
                             corr_bound_overtaking, corr_bound_proactive,
                             d_off=D_OFF, bias_max=PROACTIVE_BIAS_MAX,
                             opp_obstacle_speed=OPP_OBSTACLE_SPEED):
    """mpc_controller.py 4145-4246行目付近(if OVERTAKING / elif STOPPING+plan_ok /
    else)ブロックの複製ミラー。戻り値: (lateral_target, a_target, lat_active_side)。
    """
    _stopped = stopped_opp(fwd_vopp, opp_obstacle_speed)
    if state == "OVERTAKING" and ot_side != 0:
        target_mag = d_off
        if corr_bound_overtaking is not None:
            target_mag = min(target_mag, max(0.0, corr_bound_overtaking))
        return float(ot_side) * target_mag, 1.0, ot_side
    if state == "STOPPING" and plan_ok and plan_side != 0 and _stopped:
        target_mag = bias_max
        if corr_bound_proactive is not None:
            target_mag = min(target_mag, max(0.0, corr_bound_proactive))
        return float(plan_side) * target_mag, 1.0, plan_side
    return 0.0, 0.0, 0


def use_obstacle_avoidance_stopping(being_overtaken, lat_active_side):
    """mpc_controller.py STOPPING分岐のuse_obstacle_avoidance式の複製ミラー。"""
    return bool(being_overtaken) or lat_active_side != 0


def funnel_steps_stopping(ot_funnel_steps, lat_active_side):
    """mpc_controller.py STOPPING分岐のlateral_funnel_steps式の複製ミラー。"""
    return ot_funnel_steps if lat_active_side != 0 else 0


# --- ①非矛盾性: 側の値が本エンゲージと完全に同一のソースであること ---

def test_bias_side_equals_plan_side_which_becomes_ot_side_at_engage():
    """バイアス発動中の側は、本エンゲージが成立した場合にself._ot_sideへ
    採用される値(plan_side)と完全に同一(159節の教訓: 判定層/実行層に
    別々の指標を使わない)。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=-1,
        fwd_vopp=1.0, corr_bound_overtaking=None, corr_bound_proactive=2.0)
    assert side == -1  # 本エンゲージ時にot_sideへ採用される値と同じ


def test_no_side_flip_between_stopping_bias_and_overtaking_commit():
    """状態遷移の一貫性: STOPPING中のバイアス側とOVERTAKING確定直後の側が
    同一のplan_side由来であるため、遷移時に反転が起きない。"""
    plan_side = 1
    _, _, stop_side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=plan_side,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=1.0)
    # 本エンゲージ成立の瞬間、self._ot_side = _eval.plan_side (4091行目相当)
    ot_side_after_engage = plan_side
    _, _, overtaking_side = lateral_target_and_side(
        "OVERTAKING", ot_side=ot_side_after_engage, plan_ok=False, plan_side=0,
        fwd_vopp=0.5, corr_bound_overtaking=2.5, corr_bound_proactive=None)
    assert stop_side == overtaking_side == plan_side


def test_bias_and_overtaking_branches_mutually_exclusive():
    """①非矛盾性: OVERTAKING分岐とSTOPPING分岐(elif)は排他的で、
    同一周期に二重発火しない。"""
    tgt, a, side = lateral_target_and_side(
        "OVERTAKING", ot_side=1, plan_ok=True, plan_side=-1,  # plan_side不一致でも
        fwd_vopp=0.5, corr_bound_overtaking=2.0, corr_bound_proactive=0.1)
    assert side == 1  # OVERTAKING優先、STOPPING側の値は一切参照されない


# --- バイアス本体の挙動 ---

def test_bias_magnitude_capped_by_proactive_bias_max_not_full_d_off():
    """量は_ot_d_off(3.0m)ではなく、小さい_ot_proactive_bias_max(0.3m)が
    上限であることを確認する(本追い越しと混同しない)。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=1,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=10.0)
    assert tgt == pytest.approx(PROACTIVE_BIAS_MAX)  # corr_boundが十分広くてもcap止まり


def test_bias_magnitude_clamped_by_corr_bound_when_narrower_than_cap():
    """_corr_bound_ahead()の実測値がbias_maxより狭ければ、それが優先される
    (147/159節と同じクランプ規約)。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=1,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=0.1)
    assert tgt == pytest.approx(0.1)


def test_bias_clamped_to_zero_when_corr_bound_negative():
    """実行不可能な方向(corr_bound<0)へは絶対に押し出さない(max(0.0, ...))。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=1,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=-0.5)
    assert tgt == pytest.approx(0.0)


def test_no_bias_when_plan_not_ok():
    """_plan_passが地形的に成立していない(plan_ok=False)間はバイアスしない
    (cooldown中を含む、cheap_ok経由でplan_okがFalseになる152節の設計と両立)。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=False, plan_side=1,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=2.0)
    assert tgt == 0.0 and side == 0


def test_no_bias_when_opponent_not_stopped_or_slow():
    """走行中(速い)相手にはバイアスしない(155節のRAMP-BYPASSと同じ
    stopped_oppゲートを共有、対象は停止/低速のみ)。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=1,
        fwd_vopp=10.0, corr_bound_overtaking=None, corr_bound_proactive=2.0)
    assert tgt == 0.0 and side == 0


def test_no_bias_when_plan_side_zero():
    """plan_side=0(側自由=terrain判定が両側とも不可、または未評価)の間は
    バイアス方向が定まらないため作動しない。"""
    tgt, a, side = lateral_target_and_side(
        "STOPPING", ot_side=0, plan_ok=True, plan_side=0,
        fwd_vopp=0.5, corr_bound_overtaking=None, corr_bound_proactive=2.0)
    assert tgt == 0.0 and side == 0


# --- use_obstacle_avoidance / funnel_stepsの一貫性(上流-下流不一致の修正) ---

def test_use_obstacle_avoidance_true_when_bias_active_even_if_not_being_overtaken():
    """発見された不一致の回帰防止: バイアス作動中は被追い越し中でなくても
    use_obstacle_avoidance=True(_corr_bound_ahead()が動的コリドーを読める
    ようにする)。"""
    assert use_obstacle_avoidance_stopping(being_overtaken=False, lat_active_side=1) is True


def test_use_obstacle_avoidance_still_false_when_no_bias_and_not_being_overtaken():
    """回帰: バイアス非作動かつ被追い越しでもない通常STOPPINGは、従来通り
    use_obstacle_avoidance=False(静的コリドー、CPU負荷を増やさない)。"""
    assert use_obstacle_avoidance_stopping(being_overtaken=False, lat_active_side=0) is False


def test_use_obstacle_avoidance_true_when_being_overtaken_regardless_of_bias():
    """回帰: 被追い越し中は従来通りTrue(バイアスの有無と独立)。"""
    assert use_obstacle_avoidance_stopping(being_overtaken=True, lat_active_side=0) is True


def test_funnel_steps_reused_from_overtaking_when_bias_active():
    """コリドー急変によるinfeasible化防止(OVERTAKING開始時と同じ理由)のため、
    バイアス作動中は既存のself._ot_funnel_stepsをそのまま流用する
    (新規パラメータではない)。"""
    assert funnel_steps_stopping(ot_funnel_steps=12, lat_active_side=1) == 12


def test_funnel_steps_zero_when_no_bias():
    """回帰: バイアス非作動時のfunnel_stepsは従来通り0のまま。"""
    assert funnel_steps_stopping(ot_funnel_steps=12, lat_active_side=0) == 0


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_source_eval_initialized_to_none_before_n_fwd_branch():
    """_evaluate_engage_readiness()はOVERTAKING継続中・前方クリア確定時は
    呼ばれず_evalが未定義のままになりうるため、_n_fwd>0分岐へ入る前に
    Noneで初期化されており、古い周期の値を使い回さない安全側デフォルトに
    なっていることを確認する。"""
    idx_init = _SRC.index("_eval = None")
    idx_branch = _SRC.index("if _n_fwd > 0:")
    assert idx_init < idx_branch
    assert idx_branch - idx_init < 500  # すぐ手前で初期化(離れた場所での代入取り違え防止)


def test_source_proactive_branch_reuses_eval_plan_side_and_plan_ok():
    """②非冗長性: バイアス発動条件が新規判定式ではなく、既存の
    _eval.plan_ok/_eval.plan_side(_evaluate_engage_readinessが毎周期
    計算済み)をそのまま参照していることを確認する。"""
    idx = _SRC.index('elif (self._ot_state == "STOPPING" and _eval is not None')
    snippet = _SRC[idx:idx + 300]
    assert "_eval.plan_ok" in snippet
    assert "_eval.plan_side != 0" in snippet
    assert "_stopped_opp" in snippet


def test_source_proactive_branch_uses_corr_bound_ahead_with_plan_side():
    """判定層/実行層の乖離を防ぐため、_corr_bound_ahead()を_eval.plan_side
    (本エンゲージ時と同一の値)で呼んでいることを確認する。"""
    idx = _SRC.index('elif (self._ot_state == "STOPPING" and _eval is not None')
    snippet = _SRC[idx:idx + 800]
    assert "self._corr_bound_ahead(_eval.plan_side)" in snippet


def test_source_proactive_branch_uses_proactive_bias_max_not_d_off():
    """②非冗長性: 本追い越し用の_ot_d_offを流用せず、専用の
    _ot_proactive_bias_maxを上限として使っていることを確認する。"""
    idx = _SRC.index('elif (self._ot_state == "STOPPING" and _eval is not None')
    snippet = _SRC[idx:idx + 800]
    assert "self._ot_proactive_bias_max" in snippet
    assert "self._ot_d_off" not in snippet


def test_source_stopped_opp_computed_once_and_shared():
    """②非冗長性: _stopped_oppがif/elif連鎖より前に1回だけ計算され、
    OVERTAKING分岐のRAMP-BYPASS判定とSTOPPING分岐のゲート条件の両方で
    共有されている(重複計算・値の食い違いが起きない)ことを確認する。"""
    assert _SRC.count("_stopped_opp = (") == 1
    idx_compute = _SRC.index("_stopped_opp = (")
    idx_overtaking_if = _SRC.index('if self._ot_state == "OVERTAKING" and self._ot_side != 0:')
    idx_stopping_elif = _SRC.index('elif (self._ot_state == "STOPPING" and _eval is not None')
    assert idx_compute < idx_overtaking_if < idx_stopping_elif


def test_source_use_obstacle_avoidance_stopping_includes_lat_active_side():
    """発見された上流-下流不一致の修正が実コードに反映されていることを
    確認する: STOPPING分岐のuse_obstacle_avoidanceが_lat_active_side!=0を
    ORで含む。"""
    idx = _SRC.index('elif self._ot_state == "STOPPING":')
    snippet = _SRC[idx:idx + 700]
    assert 'self._mpc.use_obstacle_avoidance = bool(_being_overtaken) or _lat_active_side != 0' in snippet


def test_source_funnel_steps_stopping_reuses_ot_funnel_steps_when_active():
    """STOPPING分岐のlateral_funnel_stepsが、バイアス作動中は既存の
    self._ot_funnel_steps(新規パラメータではない)を流用していることを
    確認する。"""
    idx = _SRC.index('elif self._ot_state == "STOPPING":')
    snippet = _SRC[idx:idx + 1000]
    assert "self._ot_funnel_steps if _lat_active_side != 0 else 0" in snippet


def test_source_config_param_proactive_bias_max_declared():
    """新規パラメータproactive_bias_maxが既存の_otget経由で宣言されている
    ことを確認する(config.yamlとの一貫性)。"""
    assert '_otget("proactive_bias_max"' in _SRC


def test_source_ot_log_includes_proactive_bias_side_field():
    """③検証ロギング: [OT]ログへproactive_bias_side=フィールドが含まれ、
    次回ログでバイアスの発火状況を直接確認できることを確認する。"""
    idx = _SRC.index('_fwd_dbg["proactive_bias_side"]')
    assert idx > 0
    idx_log = _SRC.index('f"[OT] state=')
    log_snippet = _SRC[idx_log:idx_log + 2900]  # 2026-08-06(Fix A'診断lat_vel_src追加): 2500->2900再拡大(検証対象は無変更)
    assert "proactive_bias_side={_fwd_dbg.get('proactive_bias_side')}" in log_snippet


def test_source_proactive_bias_side_zero_when_not_stopping():
    """診断フィールドproactive_bias_sideは、STOPPING以外(OVERTAKING等)では
    常に0固定であり、本追い越しのofferset=フィールドと混同されないことを
    確認する。"""
    idx = _SRC.index('_fwd_dbg["proactive_bias_side"]')
    snippet = _SRC[idx:idx + 200]
    assert '_lat_active_side if self._ot_state == "STOPPING" else 0' in snippet
