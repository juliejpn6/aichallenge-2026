"""Unit tests for 2026-07-26(徹底解析): アクチュエータ遅延の速度依存性に対する動的
wp_id_offset補償。

背景: AWSIM操舵アクチュエータ遅延(≈200ms、2026-07-03実測r=0.992)は速度に依らない
固定の"時間"だが、既存のwp_id_offset(内巻き対策、config.yaml wp_id_offset=1、
2026-07-05決定実験)は固定の"距離"(1点≈1m、waypoint間隔がほぼ完全に一定0.999mと
実測済み)だった。そのため実効補償"時間"(距離÷速度)は速度に反比例して目減りし、
15km/h≈240ms(過剰気味)→20km/h≈180ms(不足)という非対称を生む。これは186/187節で
実測した速度依存の蛇行悪化(15→20km/hでstd約1.9倍、速度比1.33倍を上回る超線形の
悪化)と定性的に整合する仮説である。

対処: 既存wp_id_offset(inside-cut用に別途検証済みの固定値)自体は変更せず、現在速度で
T_delay(既定200ms、2026-07-03実測値)をカバーするのに必要な波数を毎周期計算し、
既存値を上回る場合のみ底上げする(max()、新規の安全弁緩和ではなく追加候補のみ)。
ceil()で切り上げる(round()だと約27km/hまで変化が出ず、既に悪化を実測済みの20km/hに
間に合わない)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装した上で
ソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import math
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

WP_SPACING_M = 0.999  # 実測(traj_mincurv.csv、標準偏差5e-8で実質完全一定)


def mirror_wp_id_offset_target(v_mps, t_delay_s, base, wp_spacing=WP_SPACING_M):
    """update前のwp_id_offset決定ロジックのミラー(mpc_controller.py、_control()内)。"""
    needed = math.ceil(v_mps * t_delay_s / wp_spacing)
    return max(base, needed)


def kmh_to_mps(v_kmh):
    return v_kmh / 3.6


# ---------------------------------------------------------------------------
# ①非矛盾性: 15km/h基準では従来と完全に同じ挙動を保つ(regression-safety)
# ---------------------------------------------------------------------------

def test_15kmh_baseline_unchanged_from_existing_tuned_value():
    """15km/h(既存wp_id_offset=1が検証済みの基準速度)では、動的計算を導入しても
    結果は従来と完全に同じ(1)であること。"""
    target = mirror_wp_id_offset_target(kmh_to_mps(15.0), t_delay_s=0.2, base=1)
    assert target == 1


def test_zero_speed_never_drops_below_base():
    """速度0でも、既存のinside-cut用固定値(base)を下回らない(max()の下限保証)。"""
    target = mirror_wp_id_offset_target(0.0, t_delay_s=0.2, base=1)
    assert target == 1


def test_base_above_needed_always_wins():
    """必要量より既存baseの方が大きい場合は常にbaseが採用される
    (inside-cut用の値を勝手に下げることはない)。"""
    target = mirror_wp_id_offset_target(kmh_to_mps(5.0), t_delay_s=0.2, base=3)
    assert target == 3


# ---------------------------------------------------------------------------
# ①非矛盾性: 20km/h以上で実際に底上げが発生する(186/187節の悪化速度域に対応)
# ---------------------------------------------------------------------------

def test_20kmh_triggers_increase_to_2():
    """186/187節で蛇行悪化を実測した20km/hでは、ceil()により1から2へ底上げされる
    (round()だと発生しない、下のtest_round_would_miss_20kmh参照)。"""
    target = mirror_wp_id_offset_target(kmh_to_mps(20.0), t_delay_s=0.2, base=1)
    assert target == 2


def test_35kmh_upper_target_range():
    target = mirror_wp_id_offset_target(kmh_to_mps(35.0), t_delay_s=0.2, base=1)
    assert target == 2


def test_monotonic_non_decreasing_with_speed():
    speeds_kmh = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    targets = [mirror_wp_id_offset_target(kmh_to_mps(v), 0.2, base=1) for v in speeds_kmh]
    assert targets == sorted(targets)


def test_round_would_miss_20kmh_regression_rationale():
    """設計判断の裏付け: round()を使った場合、20km/hではまだ1のまま(切り上げが
    必要な理由の直接確認)。ceil()を選んだ根拠を数値で残す。"""
    v = kmh_to_mps(20.0)
    needed_round = round(v * 0.2 / WP_SPACING_M)
    assert needed_round == 1  # round()だと20km/hで変化なし
    needed_ceil = math.ceil(v * 0.2 / WP_SPACING_M)
    assert needed_ceil == 2   # ceil()なら20km/hで底上げされる


# ---------------------------------------------------------------------------
# ②非冗長性・③検証ロギング: ソーステキストで配線・量子化ゲート・ログを確認
# ---------------------------------------------------------------------------

def test_dynamic_update_gated_by_change_and_calls_existing_api():
    idx = _SRC.index('_wp_id_offset_target = max(self._wp_id_offset_base, _wp_id_offset_needed)')
    idx_end = idx + 700
    snippet = _SRC[idx:idx_end]
    assert 'if _wp_id_offset_target != self._wp_id_offset_applied:' in snippet
    assert 'self._mpc.update_wp_id_offset(_wp_id_offset_target)' in snippet
    assert '[WP-OFFSET-DELAY]' in snippet
    assert 'self._wp_id_offset_applied = _wp_id_offset_target' in snippet


def test_uses_ceil_not_round():
    idx = _SRC.index('_wp_id_offset_needed = int(np.ceil(')
    assert idx > 0


def test_base_derived_from_existing_config_value_not_new_constant():
    """②非冗長性: 下限(_wp_id_offset_base)は既存config.yamlのwp_id_offset値そのものを
    再利用しており、新しい固定値を作っていない。"""
    idx = _SRC.index('self._wp_id_offset_base = int(self._cfg.mpc.wp_id_offset)')
    assert idx > 0


def test_delay_update_runs_before_get_control_call():
    """毎周期、MPCソルブ(get_control)より前にwp_id_offsetの底上げが反映される
    (MPC.py側のwp_id_offset参照タイミングと整合)。"""
    idx_delay = _SRC.index('[WP-OFFSET-DELAY]')
    idx_solve = _SRC.index('u, max_delta = self._mpc.get_control()')
    assert idx_delay < idx_solve


def test_only_one_new_config_parameter_introduced():
    """②非冗長性: 新規パラメータはT_delay(delay_t_delay_s)の1個のみ。"""
    assert _SRC.count('self._delay_t_delay_s = float(') == 1


def test_default_t_delay_matches_user_specified_starting_value():
    idx = _SRC.index('getattr(self._cfg.mpc, "delay_t_delay_s", 0.2)')
    assert idx > 0


# ---------------------------------------------------------------------------
# 2026-07-27追加(199節)→撤去(200節): debug_extra_actuator_delay_sをT_delayへ
#   上乗せする拡張を一時追加したが、実ログ照合(wp_spacingがレース速度域で早期に
#   閾値飽和し追加分を拾えない)により実効性がないと判明し撤去した。以後は
#   debug_extra_actuator_delay_s=0.055を注入した状態を標準チューニング条件とした
#   Q/R再チューニングでこの差分を吸収する方針に転換した(200節)。
#   このテストクラスに対応する実装は撤去済みのため、テストも撤去済み(199節と同じ
#   「実装を消したらテストも消す」方針、198節delta_actual撤去時と同一パターン)。
# ---------------------------------------------------------------------------


def test_source_does_not_reference_debug_extra_actuator_delay_s_in_offset_calc():
    """④遡及効果: wp_id_offset計算(_wp_id_offset_needed)がdebug_extra_actuator_delay_s
    を参照していないこと(200節撤去の確認)。_publish_control_command側の注入機構
    自体は別用途(196節)のため撤去していない。"""
    idx = _SRC.index('_wp_id_offset_needed = int(np.ceil(')
    snippet = _SRC[idx:idx + 120]
    assert 'self._delay_t_delay_s /' in snippet
    assert 'debug_extra_actuator_delay_s' not in snippet
    assert '_t_delay_effective' not in snippet
