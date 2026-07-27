"""Unit tests for the footprint_risk approach taper (153節、2026-07-22)。

背景: 第3コーナー(0721-03予選ログ wp172-176)で、footprint_risk(fwd_dlat<
along_min_widthかつfwd_ds<along_min_length)による強制停止が3〜4回連続で
繰り返される事象を実座標データ(/localization/kinematic_state)で検証した。
footprint_risk発火直前の1.76秒間、オフセット目標(_ot_alpha*lateral_target)は
約1m動いていたが、実測したego実位置の変位はほぼ100%前進成分で横方向の実移動は
ゼロに近かった。これは112節(2026-07-19)で既に定量化されていた現象(オフセット
目標-3.0mに対し実位置_cur_eyの収束に約9秒を要した)と一致し、第3コーナーが
オーバーテイク以前から操舵余力の乏しい難所であることとも整合する(112-1節)。

つまりfwd_dsは1〜2秒で危険域に達する一方、実オフセットは数秒〜9秒規模でしか
育たず、間に合わないままfootprint_riskの二値急停止に何度も陥っていた。124節で
wall_slowに適用した「二値→線形テーパー」と同じ設計を、footprint_risk本体が
発火する手前(まだdlatが物理下限未満のまま接近している間)にも適用し、実測
fwd_ds/fwd_dlatに対して閉ループで反応する滑らかな減速を追加した。新規パラメータ
は0個(along_min_width/along_min_length/_ot_pass_clear/wall_slow_speedを再利用)。

mpc_controller.pyはrclpy依存のため直接importできないため、
test_wall_slow_universal.pyと同じ方針(純Pythonミラー関数+ソーステキストに
よる構造的検証)を用いる。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

ALONG_MIN_WIDTH = 1.45
ALONG_MIN_LENGTH = 2.00
OT_PASS_CLEAR = 3.00
WALL_SLOW_SPEED = 2.0
V_MAX = 4.1667


def footprint_v_safe_check(fwd_dlat, fwd_ds, v_safe_pre,
                           along_min_width=ALONG_MIN_WIDTH,
                           along_min_length=ALONG_MIN_LENGTH,
                           ot_pass_clear=OT_PASS_CLEAR,
                           wall_slow_speed=WALL_SLOW_SPEED, v_max=V_MAX):
    """mpc_controller.pyのfootprint_risk本体+テーパー(153節)ブロックの複製ミラー。

    footprint_risk自体(fwd_dlat<along_min_widthかつabs(fwd_ds)<along_min_length)
    は既存通り無変更。elif節として、footprint_riskが不発火の間、dlatがまだ狭い
    (<along_min_width)まま fwd_ds が [along_min_length, ot_pass_clear) の範囲に
    ある場合のみ、wall_slow_speed〜v_maxの線形テーパーを適用する。
    戻り値: (v_safe_pre, branch) branchは"footprint_risk"/"footprint_taper"/None。
    """
    footprint_risk = (fwd_dlat is not None and fwd_ds is not None
                       and fwd_dlat < along_min_width
                       and abs(fwd_ds) < along_min_length)
    if footprint_risk:
        v_safe_pre = (wall_slow_speed if v_safe_pre is None
                      else min(v_safe_pre, wall_slow_speed))
        return v_safe_pre, "footprint_risk"
    if (fwd_dlat is not None and fwd_dlat < along_min_width
            and fwd_ds is not None
            and along_min_length <= abs(fwd_ds) < ot_pass_clear):
        frac = (abs(fwd_ds) - along_min_length) / (ot_pass_clear - along_min_length)
        cap = wall_slow_speed + frac * (v_max - wall_slow_speed)
        v_safe_pre = cap if v_safe_pre is None else min(v_safe_pre, cap)
        return v_safe_pre, "footprint_taper"
    return v_safe_pre, None


# --- ①非矛盾性: footprint_risk本体との排他性・境界の連続性 ---

def test_footprint_risk_hard_cap_unchanged():
    """回帰: footprint_risk本体(ds<along_min_length)は従来通りwall_slow_speedへの
    完全キャップのまま(テーパーの影響を受けない)。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.3, fwd_ds=1.5, v_safe_pre=None)
    assert branch == "footprint_risk"
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_boundary_at_along_min_length_is_continuous_no_gap():
    """境界値: fwd_ds==along_min_lengthちょうどでは、footprint_risk本体は不発火
    (厳密<のため)だがテーパー側が引き継ぎ、frac=0すなわちwall_slow_speedと
    完全に同じ値になる(153節で追加した<=により隙間を解消)。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=ALONG_MIN_LENGTH, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_taper_and_hard_cap_mutually_exclusive():
    """①非矛盾性: 同一周期でfootprint_risk本体とテーパーが二重に適用されることは
    ない(elif構造により排他)。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.3, fwd_ds=1.9, v_safe_pre=None)
    assert branch == "footprint_risk"  # ds<along_min_lengthなので本体側のみ


# --- テーパー本体の挙動 ---

def test_no_taper_when_dlat_already_safe():
    """dlatが既にalong_min_width以上(十分離れている)なら、fwd_dsが近くても
    テーパーは作用しない(既に安全な並走中に不要な減速をしない)。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=1.6, fwd_ds=2.2, v_safe_pre=None)
    assert branch is None
    assert v_safe_pre is None


def test_no_taper_beyond_pass_clear_distance():
    """fwd_dsがot_pass_clear以上ならまだ十分距離があるためテーパー対象外。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.3, fwd_ds=3.5, v_safe_pre=None)
    assert branch is None
    assert v_safe_pre is None


def test_taper_edge_near_pass_clear_approaches_v_max():
    """テーパー開始点(ot_pass_clear)近くでは、キャップはV_MAXにほぼ等しい
    (急激な段差ではなく滑らかに全開速度へ収束する、124節と同じ設計)。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.3, fwd_ds=2.99, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre > 4.0
    assert v_safe_pre < V_MAX


def test_taper_midpoint_gives_speed_between_hard_and_v_max():
    """テーパー中間点(fwd_ds=2.5、hard-soft間のちょうど中央)では、
    wall_slow_speedとV_MAXのちょうど中間程度の速度になる。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.3, fwd_ds=2.5, v_safe_pre=None)
    assert branch == "footprint_taper"
    expected = WALL_SLOW_SPEED + 0.5 * (V_MAX - WALL_SLOW_SPEED)
    assert v_safe_pre == pytest.approx(expected, abs=1e-3)


def test_coexists_with_other_v_safe_candidates_taking_the_minimum():
    """回帰: 他のv_safe候補(icc_f3等)と共存する場合、より厳しい方(min)が採用される。"""
    other_candidate = 1.0  # テーパー値より厳しい
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=2.5, v_safe_pre=other_candidate)
    assert branch == "footprint_taper"
    assert v_safe_pre == pytest.approx(1.0)  # min(1.0, テーパー値)=1.0


# --- ④過去ログへの遡及効果: 0721-03実測(wp172-176) ---

def test_retroactive_0721_03_wp172_engage_moment_barely_tapered():
    """遡及検証: ENGAGE直後(t=198.924、fwd_ds=2.917)はot_pass_clear(3.0)に
    近く、テーパーはまだ僅か(ほぼV_MAX相当)にしか効かない。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.263, fwd_ds=2.917, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre > 3.9  # ほぼV_MAXのまま(frac≈0.917)


def test_retroactive_0721_03_wp172_approaching_footprint_risk_now_tapers_down():
    """遡及検証: footprint_risk発火直前(fwd_ds=1.954、実測はこの値でfootprint_risk
    自体が発火した瞬間)をテーパー式に通すと、本体側(footprint_risk)に該当し
    従来通りwall_slow_speedとなることを確認する(退行なし)。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.257, fwd_ds=1.954246815712537,
                                                 v_safe_pre=None)
    assert branch == "footprint_risk"
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_retroactive_0721_03_midway_ds_would_have_been_tapered_before_hard_stop():
    """遡及検証の核心: 実測のENGAGE(ds=2.917)からfootprint_risk発火(ds=1.954)までの
    中間(例: ds=2.4、dlatはこの間ずっと0.26付近で不変だったと実測済み)では、
    本節の対処導入前は速度候補が無く(何も介入せず)全開に近い速度で接近し続けて
    いたが、導入後はテーパーにより既に減速が始まっていることを確認する。"""
    v_safe_pre, branch = footprint_v_safe_check(fwd_dlat=0.26, fwd_ds=2.4, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre < V_MAX  # 導入前は候補無し(None)だったが、導入後は必ず減速側に働く


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def _taper_block_snippet():
    idx = _SRC.index('_v_safe_cand.append(("footprint_risk(車体重なりリスク)"')
    idx_end = _SRC.index("並走ねばり(2026-07-03)", idx)
    return _SRC[idx:idx_end]


def test_source_taper_is_elif_of_footprint_risk_not_independent_if():
    """①非矛盾性: テーパーがfootprint_risk本体のelifとして実装されており、
    二重発火しない構造になっていることを確認する。2026-07-22修正(issue⑤②):
    条件式は_fp_near_zone(footprint_risk本体と同じ場所で1回だけ計算済みの
    危険域全体)を再利用する形になった。"""
    idx_if = _SRC.index("if _footprint_risk:")
    snippet = _SRC[idx_if:idx_if + 2000]
    assert "elif _fp_near_zone:" in snippet


def test_source_fp_near_zone_gates_on_along_min_width():
    """②非冗長性: _fp_near_zone(footprint_risk本体・154節taper・152節cooldown解除の
    3箇所が共有する危険域全体)がalong_min_widthでdlatをゲートしていることを確認する
    (2026-07-22修正、issue⑤②で1箇所に集約)。"""
    idx = _SRC.index("_fp_near_zone = (")
    snippet = _SRC[idx:idx + 300]
    assert "self._along_min_width" in snippet
    assert "self._ot_pass_clear" in snippet


def test_source_taper_reuses_existing_constants_no_new_parameters():
    """②非冗長性: 新規パラメータを使わず、既存のalong_min_length/_ot_pass_clear/
    wall_slow_speed/input_constraints["umax"]を再利用していることをソーステキストで
    確認する(along_min_widthは_fp_near_zone側で既にゲート済みのため、テーパー本体の
    式には現れない。上のtest_source_fp_near_zone_gates_on_along_min_widthで別途確認)。"""
    snippet = _taper_block_snippet()
    assert "self._along_min_length" in snippet
    assert "self._ot_pass_clear" in snippet
    assert "self._wall_slow_speed" in snippet
    assert 'self._mpc.input_constraints["umax"][0]' in snippet


def test_source_taper_boundary_has_no_gap_via_elif_exclusivity():
    """境界値の実装確認: 2026-07-22修正(issue⑤②)により、along_min_lengthとの
    境界連続性は明示的な<=比較ではなく、_footprint_risk(厳密<along_min_length)と
    elif _fp_near_zone(その否定)の排他性によって構造的に保証されるようになった。
    _footprint_riskが厳密不等号を使っていることを確認する(elifの補集合が
    ds>=along_min_lengthになり隙間が生じないことの前提)。"""
    idx = _SRC.index("_footprint_risk = _fp_near_zone and")
    snippet = _SRC[idx:idx + 100]
    assert "abs(_fwd_ds) < self._along_min_length" in snippet


def test_source_taper_logs_footprint_taper_debug_field():
    """③検証ロギング: [OT]ログへfp_taper=フィールドを追加し、次回ログでテーパーの
    発火状況(fwd_dsの値)を直接確認できることを確認する。"""
    snippet = _taper_block_snippet()
    assert '_fwd_dbg["footprint_taper"]' in snippet
    idx = _SRC.index('f"[OT] state=')
    idx_end = idx + 2500
    ot_log_snippet = _SRC[idx:idx_end]
    assert "fp_taper={_fwd_dbg.get('footprint_taper')}" in ot_log_snippet
