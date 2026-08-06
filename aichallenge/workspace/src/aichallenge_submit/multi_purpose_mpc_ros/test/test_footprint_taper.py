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

2026-07-29修正(230節): 上記の距離のみの線形補間(closing rate非考慮)は、
直近6ログ分析で発見した追突4件のうち0729-03 wp171の実測(fwd_ds=2.95m時点で
cap≈v_max=4.06m/s、実際の指令速度と完全一致)により、テーパー帯域の大半で
事実上機能していなかったことが式レベルで確認された。相手速度(vopp)を一切
見ないため、相手が遅い/停止しているほど閉じる速度が速いにもかかわらず対応が
遅れる構造的欠陥があった(design_docs 157-3/161-1節で「先回りして間隔を確保する
縦方向の仕組みが存在しない」と既に自己診断されていたが未着手だった課題)。
icc_stop等が既に使うG2式(_g2_speed、制動距離ベースのキネマティック安全速度、
v=sqrt(max(0, v_fwd²+2a(ds-margin))))と同じ考え方を採用し、fwd_ds=
along_min_lengthで相手速度に一致(それ以上詰めない)・fwd_dsが大きいほど
緩和される、相手速度(closing rate)を反映したキャップへ置き換えた。新規
パラメータは0個(既存の_fwd_a_brake/along_min_lengthを再利用)。

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
FWD_A_BRAKE = 1.3


def footprint_v_safe_check(fwd_dlat, fwd_ds, fwd_vopp, v_safe_pre,
                           along_min_width=ALONG_MIN_WIDTH,
                           along_min_length=ALONG_MIN_LENGTH,
                           ot_pass_clear=OT_PASS_CLEAR,
                           wall_slow_speed=WALL_SLOW_SPEED,
                           a_brake=FWD_A_BRAKE):
    """mpc_controller.pyのfootprint_risk本体+テーパー(153節、230節でキネマティック化)
    ブロックの複製ミラー。

    footprint_risk自体(fwd_dlat<along_min_widthかつabs(fwd_ds)<along_min_length)
    は既存通り無変更。elif節として、footprint_riskが不発火の間、dlatがまだ狭い
    (<along_min_width)まま fwd_ds が [along_min_length, ot_pass_clear) の範囲に
    ある場合のみ、G2式と同型のキネマティック制動距離キャップ(相手速度fwd_vopp・
    制動能力a_brake・物理下限along_min_lengthを基準とする)を適用する。
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
        rad = fwd_vopp * fwd_vopp + 2.0 * a_brake * (abs(fwd_ds) - along_min_length)
        cap = max(0.0, rad) ** 0.5
        v_safe_pre = cap if v_safe_pre is None else min(v_safe_pre, cap)
        return v_safe_pre, "footprint_taper"
    return v_safe_pre, None


# --- ①非矛盾性: footprint_risk本体との排他性・境界の連続性 ---

def test_footprint_risk_hard_cap_unchanged():
    """回帰: footprint_risk本体(ds<along_min_length)は従来通りwall_slow_speedへの
    完全キャップのまま(テーパーの影響を受けない、キネマティック化の対象外)。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=1.5, fwd_vopp=3.0, v_safe_pre=None)
    assert branch == "footprint_risk"
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_boundary_at_along_min_length_matches_opponent_speed():
    """境界値: fwd_ds==along_min_lengthちょうどでは、footprint_risk本体は不発火
    (厳密<のため)だがテーパー側が引き継ぎ、キネマティック式のrad=vopp²となり
    cap=vopp(相手速度に一致、それ以上は詰めない)になる。旧式(固定wall_slow_speed)
    と異なり、相手速度に応じて境界値そのものが変わる点が230節の変更の核心。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=ALONG_MIN_LENGTH, fwd_vopp=3.2, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre == pytest.approx(3.2)


def test_taper_and_hard_cap_mutually_exclusive():
    """①非矛盾性: 同一周期でfootprint_risk本体とテーパーが二重に適用されることは
    ない(elif構造により排他)。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=1.9, fwd_vopp=3.0, v_safe_pre=None)
    assert branch == "footprint_risk"  # ds<along_min_lengthなので本体側のみ


# --- テーパー本体の挙動(ゾーン判定はキネマティック化前と不変) ---

def test_no_taper_when_dlat_already_safe():
    """dlatが既にalong_min_width以上(十分離れている)なら、fwd_dsが近くても
    テーパーは作用しない(既に安全な並走中に不要な減速をしない)。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=1.6, fwd_ds=2.2, fwd_vopp=3.0, v_safe_pre=None)
    assert branch is None
    assert v_safe_pre is None


def test_no_taper_beyond_pass_clear_distance():
    """fwd_dsがot_pass_clear以上ならまだ十分距離があるためテーパー対象外。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=3.5, fwd_vopp=3.0, v_safe_pre=None)
    assert branch is None
    assert v_safe_pre is None


# --- 230節の核心: 相手速度(closing rate)への反応 ---

def test_slow_opponent_gets_strongly_capped_even_far_in_taper_zone():
    """230節の核心検証: 相手がほぼ停止(vopp=0)の場合、テーパー帯域の遠端
    (ot_pass_clear近く、fwd_ds=2.95)でも旧式(cap≈v_max)と異なり大きく減速する。
    実際の追突事例(0729-03 wp171)はこの「遅い相手に対する早期減速」が
    働いていなかったことが根本原因だった。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=2.95, fwd_vopp=0.0, v_safe_pre=None)
    assert branch == "footprint_taper"
    expected = (2.0 * FWD_A_BRAKE * (2.95 - ALONG_MIN_LENGTH)) ** 0.5
    assert v_safe_pre == pytest.approx(expected, abs=1e-3)
    assert v_safe_pre < 1.6  # 旧式ならcap≈4.03(ほぼ無制限)だった地点で大幅減速


def test_fast_opponent_barely_capped_near_taper_far_edge():
    """相手がほぼ全開速度(v_max付近)の場合、テーパー帯域の遠端では実質無制限に
    近い(不要な減速をしない、既に安全な等速追従を妨げない)。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=2.95, fwd_vopp=V_MAX, v_safe_pre=None)
    assert branch == "footprint_taper"
    assert v_safe_pre > V_MAX  # 後段のv_max全体クリップで最終的に丸められる


def test_cap_monotonically_increases_with_distance_for_fixed_vopp():
    """②非冗長性/一貫性: 相手速度を固定した場合、fwd_dsが大きいほどキャップは
    単調に緩和される(急激な段差がない滑らかなテーパーという設計意図を維持)。"""
    vopp = 2.0
    caps = []
    for ds in [2.0, 2.25, 2.5, 2.75, 2.99]:
        v_safe_pre, branch = footprint_v_safe_check(
            fwd_dlat=0.3, fwd_ds=ds, fwd_vopp=vopp, v_safe_pre=None)
        assert branch == "footprint_taper"
        caps.append(v_safe_pre)
    assert caps == sorted(caps)
    assert caps[0] < caps[-1]


def test_cap_monotonically_increases_with_vopp_for_fixed_distance():
    """230節の核心検証その2: 距離を固定した場合、相手速度が速いほどキャップは
    緩和される(closing rateへの反応そのもの)。旧式ではこの依存性が存在しなかった。"""
    ds = 2.5
    caps = []
    for vopp in [0.0, 1.0, 2.0, 3.0, 4.0]:
        v_safe_pre, branch = footprint_v_safe_check(
            fwd_dlat=0.3, fwd_ds=ds, fwd_vopp=vopp, v_safe_pre=None)
        assert branch == "footprint_taper"
        caps.append(v_safe_pre)
    assert caps == sorted(caps)
    assert caps[0] < caps[-1]


def test_coexists_with_other_v_safe_candidates_taking_the_minimum():
    """回帰: 他のv_safe候補(icc_f3等)と共存する場合、より厳しい方(min)が採用される。"""
    other_candidate = 1.0  # テーパー値より厳しい
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.3, fwd_ds=2.5, fwd_vopp=3.0, v_safe_pre=other_candidate)
    assert branch == "footprint_taper"
    assert v_safe_pre == pytest.approx(1.0)  # min(1.0, テーパー値)=1.0


# --- ④過去ログへの遡及効果: 0729-03 wp171実測(追突事例、230節のきっかけ) ---

def test_retroactive_0729_03_wp171_collision_scenario_now_brakes_earlier():
    """遡及検証の核心: 追突が発生した0729-03 wp171の実測値(fwd_ds=2.95、
    fwd_vopp=3.5、fwd_dlat=0.92)を新式に通すと、旧式のcap(≈4.06、実測の
    指令速度と一致=事実上無制限)より明確に低いキャップとなり、この場面で
    導入前より早期に減速が働くことを確認する。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.92, fwd_ds=2.95, fwd_vopp=3.5, v_safe_pre=None)
    assert branch == "footprint_taper"
    old_cap = WALL_SLOW_SPEED + ((2.95 - ALONG_MIN_LENGTH) / (OT_PASS_CLEAR - ALONG_MIN_LENGTH)) * (V_MAX - WALL_SLOW_SPEED)
    assert old_cap == pytest.approx(4.06, abs=0.01)  # 旧式は事実上無制限だったことの再確認
    assert v_safe_pre < old_cap
    assert v_safe_pre == pytest.approx(3.837, abs=0.01)


def test_retroactive_footprint_risk_hard_boundary_still_reached_when_very_close():
    """遡及検証: footprint_risk発火直前相当(fwd_ds=1.954、本体側の閾値未満)を
    テーパー式に通すと、従来通りfootprint_risk本体(wall_slow_speed)に該当し
    退行がないことを確認する。"""
    v_safe_pre, branch = footprint_v_safe_check(
        fwd_dlat=0.257, fwd_ds=1.954246815712537, fwd_vopp=3.0, v_safe_pre=None)
    assert branch == "footprint_risk"
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


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
    危険域全体)を再利用する形になった(230節のキネマティック化でも維持)。"""
    idx_if = _SRC.index("if _footprint_risk:")
    snippet = _SRC[idx_if:idx_if + 2000]
    assert "elif _fp_near_zone:" in snippet


def test_source_fp_near_zone_gates_on_along_min_width():
    """②非冗長性: _fp_near_zone(footprint_risk本体・154節taper・152節cooldown解除の
    3箇所が共有する危険域全体)がalong_min_widthでdlatをゲートしていることを確認する
    (2026-07-22修正、issue⑤②で1箇所に集約。230節のキネマティック化後も不変)。"""
    idx = _SRC.index("_fp_near_zone = (")
    snippet = _SRC[idx:idx + 300]
    assert "self._along_min_width" in snippet
    assert "self._ot_pass_clear" in snippet


def test_source_taper_reuses_g2_style_constants_no_new_parameters():
    """②非冗長性: 2026-07-29(230節)のキネマティック化後、新規パラメータを使わず、
    既存のalong_min_length・_fwd_a_brake(G2式と共有)・_fwd_vopp(既存スキャン結果)を
    再利用していることをソーステキストで確認する。"""
    snippet = _taper_block_snippet()
    assert "self._along_min_length" in snippet
    assert "self._fwd_a_brake" in snippet
    assert "_fwd_vopp" in snippet


def test_source_taper_boundary_has_no_gap_via_elif_exclusivity():
    """境界値の実装確認: 2026-07-22修正(issue⑤②)により、along_min_lengthとの
    境界連続性は明示的な<=比較ではなく、_footprint_risk(厳密<along_min_length)と
    elif _fp_near_zone(その否定)の排他性によって構造的に保証されるようになった。
    _footprint_riskが厳密不等号を使っていることを確認する(elifの補集合が
    ds>=along_min_lengthになり隙間が生じないことの前提。230節のキネマティック化後も
    この境界の排他構造自体は不変)。"""
    idx = _SRC.index("_footprint_risk = _fp_near_zone and")
    snippet = _SRC[idx:idx + 100]
    assert "abs(_fwd_ds) < self._along_min_length" in snippet


def test_source_taper_logs_footprint_taper_debug_field():
    """③検証ロギング: [OT]ログへfp_taper=フィールドを追加し、次回ログでテーパーの
    発火状況(fwd_dsの値)を直接確認できることを確認する(230節のキネマティック化後も
    ロギング自体は維持)。"""
    snippet = _taper_block_snippet()
    assert '_fwd_dbg["footprint_taper"]' in snippet
    idx = _SRC.index('f"[OT] state=')
    idx_end = idx + 2700  # 2026-08-06(Fix A'診断lat_vel_src追加): 2500->2700再拡大(検証対象は無変更)
    ot_log_snippet = _SRC[idx:idx_end]
    assert "fp_taper={_fwd_dbg.get('footprint_taper')}" in ot_log_snippet


def test_source_taper_uses_sqrt_kinematic_formula():
    """230節の実装確認: キャップ計算がG2式と同型のsqrt(max(0, v²+2a(ds-margin)))
    構造になっていることをソーステキストで確認する(np.sqrtとmax(0.0, ...)の
    両方が使われている、G2式のstd np.sqrt(max(0.0, rad))パターンとの一貫性)。"""
    snippet = _taper_block_snippet()
    assert "np.sqrt(max(0.0, _fp_rad))" in snippet
    assert "_fwd_vopp * _fwd_vopp" in snippet
