"""Unit tests for the state-independent wall-proximity speed cap (49節, 2026-07-14).

Mirrors the exact formula shipped in mpc_controller.py's "C守り" block (the
`if _being_overtaken:` gate was removed so this now runs every cycle) plus the
downstream u0 = min(u0, v_safe) clip applied after get_control().

2026-07-19追加(122節、Sランク根本原因の修正): 旧実装は起動時1回計算の静的
wp.ub/lb(safety_margin未控除)を使っていたが、MPC自身が実際にQPで拘束・追従する
corridor(dbg_corr_ub0/lb0、毎周期動的計算・safety_margin控除済み)とは独立した
別ソースだったため、wall_slowが「余裕あり」と判定していてもMPCの実際のcorridorは
既に極小、ということが起こり得た(0719-04実測wp330-336、ユーザー目視で右壁接触)。
本節でwall_slowの判定式をdbg_corr_ub0/lb0ベースへ置き換えた。dbg_corr_ub0/lb0は
既にsafety_margin込みのため、hw(半幅)の追加減算は二重マージン化になり行わない。

2026-07-19追加(123節、閾値の再較正): 122節の切替後、wall_slow_marginを旧方式
向けの0.5のまま据え置いていたため、既にsafety_margin込みの値へさらに0.5mを
要求する二重マージンとなり、0719-05実測で対戦車の全くいない安全なコーナーの
過半数(NORMAL状態256サンプル中64サンプル)で誤発動し、Lap1が240秒まで悪化した。
実測(wp330-336)で本来検知すべき超過はmargin=0.046m相当だった一方、安全な
NORMALコーナーのp10は0.43mだったため、両者を分離できる0.15へ再較正した。

2026-07-19追加(124節、二値応答のテーパー化): 123節の再較正後もmargin=+0.01〜
+0.15(実際には未逸脱)の27箇所全てで一律にwall_slow_speed(2.0m/s)へ急減速し、
1周あたり約4秒の不要な減速(83秒→87秒)を招いていた(122節以前の静的テーブル
方式では同じ27箇所とも発火していなかったことを遡及検証済み)。既存のicc_f3
テーパー(hard_stop_gap〜f3_taper_gap)と同じ考え方で、wall_slow_margin(テーパー
開始点)〜wall_slow_margin_hard(この値以下で完全にwall_slow_speedへ)の間を
線形補間する。
"""
import math
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()
_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CONFIG_PATH) as _f:
    _CONFIG_SRC = _f.read()

WALL_SLOW_MARGIN = 0.15
WALL_SLOW_MARGIN_HARD = 0.0
WALL_SLOW_SPEED = 2.0
V_MAX = 4.1667


def wall_margin_check(corr_ub0, corr_lb0, cur_ey, v_safe_pre):
    """mpc_controller.pyの"C守り"ブロック(124節修正後)の複製ミラー。

    corr_ub0/corr_lb0はself._mpc.dbg_corr_ub0/dbg_corr_lb0(既にsafety_margin
    控除済み)を模す。非有限(NaN、MPC未初期化時の既定値)ならfail-open(介入なし)。
    wmargin<WALL_SLOW_MARGINで発火するが、実際のキャップ値はWALL_SLOW_MARGIN_HARD
    (完全にWALL_SLOW_SPEED)〜WALL_SLOW_MARGIN(V_MAXへ収束)の間を線形補間する。
    """
    if not (math.isfinite(corr_ub0) and math.isfinite(corr_lb0)):
        return v_safe_pre, False, float('nan')
    m_left = corr_ub0 - cur_ey
    m_right = cur_ey - corr_lb0
    wmargin = min(m_left, m_right)
    fired = wmargin < WALL_SLOW_MARGIN
    if fired:
        if wmargin <= WALL_SLOW_MARGIN_HARD:
            wall_cap = WALL_SLOW_SPEED
        else:
            frac = (wmargin - WALL_SLOW_MARGIN_HARD) / (WALL_SLOW_MARGIN - WALL_SLOW_MARGIN_HARD)
            wall_cap = WALL_SLOW_SPEED + frac * (V_MAX - WALL_SLOW_SPEED)
        v_safe_pre = wall_cap if v_safe_pre is None else min(v_safe_pre, wall_cap)
    return v_safe_pre, fired, wmargin


def clip_u0(u0, v_safe_pre, v_max_clip=V_MAX):
    if v_safe_pre is None:
        return u0
    return min(u0, min(v_safe_pre, v_max_clip))


def test_solo_no_traffic_section_now_gets_wall_check():
    """0713-05 wp270-280再現: fwd=0/n_dynobs=0の完全ソロ区間(従来はbeing_overtaken=Falseで
    ノーチェックだった)でも、コリドーが狭ければ壁際減速が発火しv_max全開からクリップされる。
    margin=-0.05(hard以下)としてテーパーの影響を受けない完全キャップのケースを使う。"""
    v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=-0.05, corr_lb0=-3.5, cur_ey=0.0, v_safe_pre=None)
    u0 = clip_u0(V_MAX, v_safe_pre)
    assert fired is True
    assert u0 == pytest.approx(WALL_SLOW_SPEED)


def test_wide_corridor_no_intervention_regression():
    """回帰: 壁マージンが十分広い場合は介入しない。"""
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=3.0, corr_lb0=-3.0, cur_ey=0.0, v_safe_pre=None)
    u0 = clip_u0(V_MAX, v_safe_pre)
    assert fired is False
    assert u0 == pytest.approx(V_MAX)


def test_overtaking_offset_pinned_to_corridor_boundary_also_detected():
    """0713-06 wp136相当: OVERTAKING中にオフセット目標が壁境界(corr_bound)まで
    伸びきった場合も、実位置ベースのチェックが検知する。"""
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=2.903, corr_lb0=-1.66, cur_ey=2.90, v_safe_pre=None)
    assert fired is True


@pytest.mark.parametrize("corr_ub0_input,expected_fired", [
    (WALL_SLOW_MARGIN + 0.001, False),  # margin=0.151: 僅かに広い -> 介入しない
    (WALL_SLOW_MARGIN, False),          # margin=0.15ちょうど: `<`厳密比較で介入しない
    (WALL_SLOW_MARGIN - 0.001, True),   # margin=0.149: 僅かに狭い -> 介入する
])
def test_wall_slow_margin_boundary(corr_ub0_input, expected_fired):
    """境界値: wall_slow_margin(0.15m、123節で0.5から再較正)ちょうどでの扱いを確認する。"""
    _v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=corr_ub0_input, corr_lb0=-10.0, cur_ey=0.0, v_safe_pre=None)
    assert fired == expected_fired


def test_coexists_with_other_v_safe_candidates_taking_the_minimum():
    """回帰: 他の速度候補(icc_f3等)と共存する場合、より厳しい方(min)が採用される。
    margin=-0.05(hard以下)としてテーパーの影響を受けない完全キャップのケースを使う。"""
    other_candidate = 3.0  # wall_slow_speed(2.0)より緩い候補
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=-0.05, corr_lb0=-3.0, cur_ey=0.0, v_safe_pre=other_candidate)
    assert fired is True
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)  # min(3.0, 2.0)=2.0


def test_tighter_other_candidate_still_wins_over_wall_slow():
    """他候補がwall_slow_speedよりさらに厳しい場合は、そちらが維持される(二重min)。
    margin=-0.05(hard以下)としてテーパーの影響を受けない完全キャップのケースを使う。"""
    other_candidate = 1.0  # wall_slow_speed(2.0)より厳しい
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=-0.05, corr_lb0=-3.0, cur_ey=0.0, v_safe_pre=other_candidate)
    assert fired is True
    assert v_safe_pre == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 79節(2026-07-16): 77節で一時追加した先読み(_along_lookahead窓内のub/lb走査)を
# revertしたことの回帰防止。先読みは自車自身の現在の横偏差(cur_ey)を固定したまま
# 曲がっていく先のwaypointと比較する設計自体が誤りで、実走行(0715-07/08)で
# wall=が50〜65%の頻度で負値(物理的にあり得ない、最悪-2.41m)になり、障害物の
# 無いクリアな直線でも誤発動しCOLLISION-SUSPECTEDまで発火した。
# ---------------------------------------------------------------------------

def test_current_point_only_signature_takes_single_waypoint_not_a_window():
    """回帰防止: wall_margin_check()は単一時刻のcorr_ub0/lb0のみを引数に取る
    (窓・リストを取らない)。将来再び先読みが混入していないかの構造的な確認。"""
    import inspect
    params = list(inspect.signature(wall_margin_check).parameters)
    assert params == ["corr_ub0", "corr_lb0", "cur_ey", "v_safe_pre"]


def test_current_point_formula_cannot_go_pathologically_negative_for_safe_position():
    """遡及検証(0715-07/08実測の再発防止): 先読み(77節)導入後は障害物の無い
    クリアな直線でもwall=が-0.02〜-2.41まで振れる誤発動を50〜65%の頻度で記録した
    (0715-04/06の修正前は0%)。現在時刻のcorr_ub0/lb0のみを使う計算式(122節修正後も
    この性質は不変)では、実際にカートが壁から離れた安全な位置(cur_ey=0、十分広い
    corridor)にいる限り、マージンが物理的にあり得ない大きな負値になることはない。"""
    _, fired, wmargin = wall_margin_check(
        corr_ub0=2.5, corr_lb0=-2.5, cur_ey=0.0, v_safe_pre=None)
    assert fired is False
    assert wmargin > 0.0  # 先読みバグのような物理的にあり得ない負値にはならない


def test_offset_construction_still_correctly_detected_without_lookahead():
    """revert後も、OVERTAKING中に実際に壁へ寄せている(cur_eyが壁境界近くまで
    伸びている)ケース自体は現在時刻の実測のみで正しく検知できることを確認する
    (先読みが無くても本来の目的=実位置ベースの検知は損なわれない)。"""
    _, fired, wmargin = wall_margin_check(
        corr_ub0=3.0, corr_lb0=-3.0, cur_ey=2.9, v_safe_pre=None)
    assert fired is True
    assert wmargin < WALL_SLOW_MARGIN


# ---------------------------------------------------------------------------
# 122節(2026-07-19): wall_slowを静的wp.ub/lbから動的dbg_corr_ub0/lb0へ切替
# (Sランク根本原因、0719-04実測wp330-336の右壁接触)。
# ---------------------------------------------------------------------------

def test_nan_corridor_fails_open_no_intervention():
    """回帰(fail-open): dbg_corr_ub0/lb0がまだ計算されていない(MPC初期化直後の
    既定値NaN、core/MPC.py:89-90)場合、例外を投げず介入なしとして扱う
    (本モジュール内の既存fail-open方針、103/107節の新側room判定と同じ考え方)。"""
    v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=float('nan'), corr_lb0=float('nan'), cur_ey=0.0, v_safe_pre=None)
    assert fired is False
    assert v_safe_pre is None
    assert math.isnan(wmargin)


def test_retroactive_0719_04_wp332_corridor_now_triggers():
    """遡及検証(122節、0719-04実測lap1 wp330-336): 1周目最終コーナー(kappa最大0.147)で
    dbg_corr_ub0=0.484m(wp332実測値そのまま)まで狭まっていたところ、直後にekf_ey=0.51m
    (wp333実測値)まで外側へ膨らんでいた。修正後の計算式(dbg_corr_ub0直接比較)では、
    この実測値の組み合わせで確実に介入(wall_slow_speedへの減速)が発火することを確認する。
    旧実装(静的wp.ub、safety_margin未控除)では、同じ場面でwall=None(介入なし)が
    実際にログへ記録されていた。"""
    v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=0.484, corr_lb0=-3.5, cur_ey=0.51, v_safe_pre=None)
    assert fired is True
    assert wmargin < 0.0  # 既に境界を超えて外側にいたことも確認できる
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_retroactive_before_after_comparison_at_wp332():
    """遡及検証(122節、比較): wp332実測に基づく再構成値で、旧式(静的wp.ub相当・
    safety_margin未控除・hw減算あり・閾値0.5)は介入せず、新式(dbg_corr_ub0直接・
    閾値0.15)は介入することを対比する。静的wp.ubの正確な実測値はログに残っていない
    ため、dbg_corr_ub0(0.484)+safety_margin(NORMAL時1.626、122-4節でsource確認済み
    のwidth/√2)から概算再構成した値(2.11)を用いる。旧閾値0.5はここでは123節で
    変更される前の値としてリテラルで固定する(WALL_SLOW_MARGIN定数は123節時点の
    新値0.15を指すため、ここで参照すると比較の意味が変わってしまう)。"""
    OLD_WALL_SLOW_MARGIN = 0.5  # 123節で0.15へ再較正される前の値(リテラル固定)
    reconstructed_static_ub = 0.484 + 1.626  # ≈ 2.11、safety_margin控除前の概算
    cur_ey = 0.438  # wp332実測ekf_ey
    hw = 0.725

    old_m_left = reconstructed_static_ub - cur_ey - hw
    old_fired = old_m_left < OLD_WALL_SLOW_MARGIN
    assert old_fired is False  # 旧式は「余裕あり」と誤判定していた

    _v_safe_pre, new_fired, _wmargin = wall_margin_check(
        corr_ub0=0.484, corr_lb0=-3.5, cur_ey=cur_ey, v_safe_pre=None)
    assert new_fired is True  # 新式はMPCが実際に追従するcorridorの狭さを正しく検知する


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証(122節)。mpc_controller.pyはrclpy依存のため
# 直接importできず、上のミラー関数と実装の一致は目視レビュー + 以下の構造確認で担保する。
# ---------------------------------------------------------------------------

def test_source_wall_slow_uses_dynamic_corridor_not_static_table():
    """wall_slowブロックが静的wp.ub/lb(_wpc)ではなく、self._mpc.dbg_corr_ub0/lb0を
    参照していることをソーステキストで確認する。"""
    idx = _SRC.index('# C 守り: 壁近接減速')
    idx_end = _SRC.index("並走ねばり(2026-07-03)", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._mpc.dbg_corr_ub0" in snippet
    assert "self._mpc.dbg_corr_lb0" in snippet
    assert "_wpc" not in snippet
    assert "_hw = 0.725" not in snippet


def test_source_wall_slow_guards_against_non_finite_corridor():
    """非有限(NaN)のdbg_corr_ub0/lb0に対してnp.isfiniteでガードしていることを確認する
    (MPC未初期化直後にNaNのまま誤って発火しないためのfail-open実装)。"""
    idx = _SRC.index("_corr_ub0 = self._mpc.dbg_corr_ub0")
    snippet = _SRC[idx:idx + 300]
    assert "np.isfinite(_corr_ub0)" in snippet
    assert "np.isfinite(_corr_lb0)" in snippet


# ---------------------------------------------------------------------------
# 123節(2026-07-19): wall_slow_marginを二重マージン(0.5)から0.15へ再較正
# (0719-05実測、対戦車の全くいない安全なコーナーの過半数で誤発動しLap1が
# 240秒まで悪化。ユーザー指摘:「すべてのコーナーは15km/h以上で曲がれる」
# 「減速しすぎ」「カーブでの減速が多い」)。
# ---------------------------------------------------------------------------

def test_retroactive_0719_05_normal_corner_p10_no_longer_over_triggers():
    """遡及検証(123節、0719-05実測NORMAL状態256サンプルの分布): 対戦車のいない
    通常コーナーでのコリドー半幅の下位10%点(p10=0.43m)は、旧閾値0.5では誤発動して
    いたが、再較正後の閾値0.15では介入しないことを確認する(=このp10相当のコーナーは
    ユーザーの言う「15km/h以上で曲がれる」安全な区間であり、不要な減速をしない)。"""
    OLD_WALL_SLOW_MARGIN = 0.5  # 123節で再較正される前の値(リテラル固定)
    p10_corridor_half_width = 0.43  # 0719-05実測、NORMAL・n_dynobs=0、256サンプルのp10

    old_fired = p10_corridor_half_width < OLD_WALL_SLOW_MARGIN
    assert old_fired is True  # 旧閾値は誤って介入していた(過剰減速の直接原因)

    _v_safe_pre, new_fired, _wmargin = wall_margin_check(
        corr_ub0=p10_corridor_half_width, corr_lb0=-10.0, cur_ey=0.0, v_safe_pre=None)
    assert new_fired is False  # 新閾値は安全なコーナーに介入しない


def test_retroactive_0719_05_wp330_still_caught_after_recalibration():
    """遡及検証(123節): 122節が捕捉対象とした本物の危険域(wp330-336、margin=0.046m
    相当)は、閾値を0.5→0.15へ下げた後も引き続き確実に捕捉されることを確認する
    (再較正が「緩めすぎ」になっていないことの確認)。"""
    _v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=0.484, corr_lb0=-3.5, cur_ey=0.438, v_safe_pre=None)
    assert fired is True
    assert wmargin == pytest.approx(0.046, abs=1e-3)


def test_config_and_source_default_agree_on_recalibrated_value():
    """config.yamlのwall_slow_marginデフォルト値と、mpc_controller.py側の_otget
    フォールバックデフォルトが、再較正後の0.15で一致していることを構造的に確認する
    (config省略時にpre-123節の0.5へ意図せず戻らないようにするための整合性チェック)。"""
    assert "wall_slow_margin: 0.15" in _CONFIG_SRC
    idx = _SRC.index('self._wall_slow_margin = float(_otget("wall_slow_margin"')
    snippet = _SRC[idx:idx + 100]
    assert '"wall_slow_margin", 0.15' in snippet


# ---------------------------------------------------------------------------
# 124節(2026-07-19): wall_slowを二値応答(介入なし/wall_slow_speed一律)から
# icc_f3と同じ線形テーパー(wall_slow_margin_hard〜wall_slow_margin)へ変更。
# 0719-05以降の実測(run_perffix_20260719_231648等)で、margin=+0.01〜+0.15の
# 27箇所全てが122節以前(静的テーブル・閾値0.5)では発火しておらず、これが
# 83秒→87秒(約4秒/周)の不要な減速の直接原因と判明。ユーザー要望:「15km/hで
# まともに走行できるようにし、残りの課題は速度を上げたことに起因するものだけに
# したい」。
# ---------------------------------------------------------------------------

def test_taper_at_hard_threshold_gives_full_wall_slow_speed():
    """境界値: margin<=wall_slow_margin_hard(0.0)では、テーパー無しの完全な
    wall_slow_speed(2.0)が適用される(実際に境界へ到達/超過した最終防衛ライン)。"""
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=0.0, corr_lb0=-3.5, cur_ey=0.0, v_safe_pre=None)
    assert fired is True
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_taper_at_soft_threshold_edge_approaches_v_max():
    """境界値: margin→wall_slow_margin(0.15)に近づくほど、キャップはV_MAXに近づく
    (テーパー上限、急激な段差ではなく滑らかに全開速度へ収束する)。"""
    v_safe_pre, fired, _wmargin = wall_margin_check(
        corr_ub0=0.149, corr_lb0=-3.5, cur_ey=0.0, v_safe_pre=None)
    assert fired is True
    assert v_safe_pre > 4.0  # V_MAX(4.1667)にほぼ等しい、wall_slow_speed(2.0)ではない
    assert v_safe_pre < V_MAX


def test_taper_midpoint_gives_speed_between_hard_and_v_max():
    """テーパー中間点(margin=0.075、hard-soft間のちょうど中央)では、
    wall_slow_speedとV_MAXのちょうど中間程度の速度になる(線形補間の確認)。"""
    v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=0.075, corr_lb0=-3.5, cur_ey=0.0, v_safe_pre=None)
    assert fired is True
    assert wmargin == pytest.approx(0.075, abs=1e-6)
    expected = WALL_SLOW_SPEED + 0.5 * (V_MAX - WALL_SLOW_SPEED)
    assert v_safe_pre == pytest.approx(expected, abs=1e-3)


def test_retroactive_run_perffix_231648_gentle_corners_now_get_taper_not_hard_cap():
    """遡及検証(124節、run_perffix_20260719_231648実測): 27箇所のうち大半を占める
    margin=+0.01〜+0.15の代表例(wp334: margin=0.08、wp121: margin=0.02)は、
    テーパー適用後は完全な2.0m/sキャップではなく、それぞれの余裕に応じた
    中間速度になることを確認する(不要な急減速の解消)。"""
    v_safe_pre_1, fired_1, _ = wall_margin_check(
        corr_ub0=0.670, corr_lb0=-1.593, cur_ey=0.556, v_safe_pre=None)  # wp334実測相当(margin≈0.08)
    assert fired_1 is True
    assert v_safe_pre_1 > WALL_SLOW_SPEED + 0.5  # 2.0べったりではなく明確に緩和

    v_safe_pre_2, fired_2, _ = wall_margin_check(
        corr_ub0=0.499, corr_lb0=-2.509, cur_ey=0.433, v_safe_pre=None)  # wp121実測相当(margin≈0.02)
    assert fired_2 is True
    assert v_safe_pre_2 < v_safe_pre_1  # marginがより小さい(より危険)方がより厳しく減速


def test_retroactive_wp330_336_incident_still_gets_near_full_cap():
    """遡及検証(124節、非退行): wp330-336の実際の事故(margin=-0.026、既に境界
    超過)は、テーパー導入後も引き続きwall_slow_speedそのもの(境界超過はhard以下
    のため線形補間の対象外)が適用されることを確認する。テーパー化が安全側の
    後退になっていないことの確認。"""
    v_safe_pre, fired, wmargin = wall_margin_check(
        corr_ub0=0.484, corr_lb0=-3.5, cur_ey=0.51, v_safe_pre=None)
    assert fired is True
    assert wmargin < 0.0
    assert v_safe_pre == pytest.approx(WALL_SLOW_SPEED)


def test_source_wall_slow_implements_linear_taper_reusing_umax():
    """wall_slowブロックがwall_slow_margin_hardとの線形補間を実装しており、
    テーパー上限にself._mpc.input_constraints["umax"][0](既存のv_max参照、
    将来の速度引き上げにも自動追従)を再利用していることをソーステキストで確認する。"""
    idx = _SRC.index("_corr_ub0 = self._mpc.dbg_corr_ub0")
    idx_end = _SRC.index("並走ねばり(2026-07-03)", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._wall_slow_margin_hard" in snippet
    assert 'self._mpc.input_constraints["umax"][0]' in snippet
    assert "_frac" in snippet


def test_config_and_source_default_agree_on_hard_threshold():
    """config.yamlのwall_slow_margin_hardデフォルト値と、mpc_controller.py側の
    _otgetフォールバックデフォルトが一致していることを構造的に確認する。"""
    assert "wall_slow_margin_hard: 0.0" in _CONFIG_SRC
    idx = _SRC.index('self._wall_slow_margin_hard = float(_otget("wall_slow_margin_hard"')
    snippet = _SRC[idx:idx + 100]
    assert '"wall_slow_margin_hard", 0.0' in snippet
