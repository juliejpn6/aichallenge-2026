"""Unit tests for issue⑤②: footprint_riskのcooldown解除条件を危険域全体
(footprint_taper込み)へ拡張する対処(2026-07-22)。

背景: 実測(0722-2ログ、d1)で、footprint_risk起因のcooldownがENGAGEから
1秒未満で再度footprint_riskへ陥る事例が25件中大半を占めていた。原因は
_ot_footprint_risk_clear_count(152節の適応的cooldown)が_footprint_risk本体
(fwd_ds<along_min_length)の不成立のみで解除されており、154節のtaper域
(along_min_length<=fwd_ds<ot_pass_clear)にまだ留まっていても解除されて
いたため、間合いが回復する前に即座にENGAGEし即座に再失敗する無意味な
ループが生じていた。

対処: footprint_risk本体・154節taper・152節cooldown解除の3箇所が同一の
危険域判定(_fp_near_zone = fwd_dlat<along_min_width AND fwd_ds<ot_pass_clear)
を1回だけ計算して共有するよう統一した(159節と同じ「同じ周期の同じ値を
使う」原則)。新規パラメータ0個。

一貫性検証(ユーザー指摘): 当初案はtaper側の条件式を独立に再定義した
コピーを使う予定だったが、それでは「taper側の閾値が将来変わった時に
cooldown解除側だけ古いままになる」という上流-下流の不整合リスクが
あったため、単一の共有変数へ統一した。

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

ALONG_MIN_WIDTH = 1.45
ALONG_MIN_LENGTH = 2.00
OT_PASS_CLEAR = 3.00


def fp_near_zone(fwd_dlat, fwd_ds, along_min_width=ALONG_MIN_WIDTH, ot_pass_clear=OT_PASS_CLEAR):
    """mpc_controller.pyの_fp_near_zone計算式の複製ミラー。"""
    return (fwd_dlat is not None and fwd_ds is not None
            and fwd_dlat < along_min_width and abs(fwd_ds) < ot_pass_clear)


def footprint_risk(fwd_dlat, fwd_ds, along_min_length=ALONG_MIN_LENGTH, **kwargs):
    """mpc_controller.pyの_footprint_risk計算式(_fp_near_zoneベース)の複製ミラー。"""
    return fp_near_zone(fwd_dlat, fwd_ds, **kwargs) and abs(fwd_ds) < along_min_length


def cooldown_clear_count_increment(fwd_dlat, fwd_ds, prev_count, gated, debounce=8):
    """mpc_controller.pyのcooldown解除カウンタ更新式(_fp_near_zoneベース)の
    複製ミラー。戻り値: (new_count, cleared)。"""
    if not gated:
        return prev_count, None
    zone = fp_near_zone(fwd_dlat, fwd_ds)
    new_count = 0 if zone else prev_count + 1
    return new_count, new_count >= debounce


# --- ①非矛盾性: 3箇所(footprint_risk本体/taper/cooldown解除)の関係 ---

def test_footprint_risk_is_subset_of_fp_near_zone():
    """footprint_risk本体はfp_near_zoneの部分集合(ds<along_min_lengthの範囲)。"""
    assert footprint_risk(fwd_dlat=0.3, fwd_ds=1.5) is True
    assert fp_near_zone(fwd_dlat=0.3, fwd_ds=1.5) is True


def test_taper_zone_is_fp_near_zone_minus_footprint_risk():
    """テーパー域(154節)は、fp_near_zoneが真かつfootprint_risk本体が偽の範囲
    (ds>=along_min_lengthかつds<ot_pass_clear)と一致する。"""
    assert fp_near_zone(fwd_dlat=0.3, fwd_ds=2.5) is True
    assert footprint_risk(fwd_dlat=0.3, fwd_ds=2.5) is False  # elif側(taper)に該当


def test_no_gap_at_along_min_length_boundary():
    """境界値: ds=along_min_lengthちょうどではfootprint_risk本体は不成立(厳密<)
    だが、fp_near_zoneはTrueのままなのでcooldown解除はブロックされ続ける
    (footprint_risk⇔taperの間に隙間が生じない、既存の153節保証を維持)。"""
    assert footprint_risk(fwd_dlat=0.3, fwd_ds=ALONG_MIN_LENGTH) is False
    assert fp_near_zone(fwd_dlat=0.3, fwd_ds=ALONG_MIN_LENGTH) is True


# --- cooldown解除カウンタの挙動(本節の核心) ---

def test_cooldown_not_cleared_while_still_in_taper_zone():
    """回帰防止の核心: footprint_risk本体は不成立(ds>=along_min_length)でも
    taper域内(ds<ot_pass_clear)にいる間はcooldownカウンタがリセットされ
    続け、解除されない。"""
    count = 0
    for _ in range(20):
        count, cleared = cooldown_clear_count_increment(
            fwd_dlat=0.85, fwd_ds=2.99, prev_count=count, gated=True)
    assert cleared is False
    assert count == 0


def test_cooldown_clears_once_fully_outside_danger_zone_via_ds():
    """危険域を完全に抜けて8周期連続で維持されればcooldownは解除される
    (dsがot_pass_clear以上へ回復したケース)。"""
    count = 0
    cleared = None
    for _ in range(8):
        count, cleared = cooldown_clear_count_increment(
            fwd_dlat=0.85, fwd_ds=3.5, prev_count=count, gated=True)
    assert cleared is True


def test_cooldown_clears_once_fully_outside_danger_zone_via_dlat():
    """危険域を完全に抜けて8周期連続で維持されればcooldownは解除される
    (dlatがalong_min_width以上へ回復したケース、ds自体は近いまま)。"""
    count = 0
    cleared = None
    for _ in range(8):
        count, cleared = cooldown_clear_count_increment(
            fwd_dlat=1.6, fwd_ds=2.5, prev_count=count, gated=True)
    assert cleared is True


def test_retroactive_0722_2_log_wp141_scenario_now_blocks_premature_reengage():
    """遡及検証: 実測(0722-2ログ、d1 wp141)のENGAGE時点の値(fwd_dlat=0.847、
    fwd_ds=2.99)は旧実装ではfootprint_risk本体が不成立のためcooldown解除
    条件を満たしていたが、新実装ではfp_near_zone=Trueのままなので
    解除されないことを確認する。"""
    assert footprint_risk(fwd_dlat=0.847, fwd_ds=2.99) is False  # 旧実装: 解除されていた
    assert fp_near_zone(fwd_dlat=0.847, fwd_ds=2.99) is True     # 新実装: 解除されない


def test_not_gated_cooldown_path_unaffected():
    """回帰: footprint_risk起因でないcooldown(gated=False)は本対処の対象外
    (149節の固定タイマー方式のまま、挙動不変)。"""
    count, cleared = cooldown_clear_count_increment(
        fwd_dlat=0.85, fwd_ds=2.99, prev_count=0, gated=False)
    assert cleared is None  # 本カウンタの対象外(固定タイマー側で別途処理)


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_source_fp_near_zone_computed_once_before_footprint_risk():
    """②非冗長性・①非矛盾性: _fp_near_zoneが_footprint_risk本体の直前で
    1回だけ計算され、_footprint_risk自体がそれを再利用する形で定義されて
    いることを確認する。"""
    idx_zone = _SRC.index("_fp_near_zone = (")
    idx_risk = _SRC.index("_footprint_risk = _fp_near_zone and")
    assert idx_zone < idx_risk
    assert idx_risk - idx_zone < 400


def test_source_cooldown_clear_count_uses_fp_near_zone_not_footprint_risk():
    """本対処の核心: cooldown解除カウンタの更新式が_footprint_risk単体ではなく
    _fp_near_zone(危険域全体)を参照していることを確認する。"""
    idx = _SRC.index("self._ot_footprint_risk_clear_count = (\n                    0 if")
    snippet = _SRC[idx:idx + 150]
    assert "_fp_near_zone" in snippet
    assert "0 if _footprint_risk else" not in snippet


def test_source_taper_condition_reuses_fp_near_zone():
    """②非冗長性: 154節のtaper条件が独立した再定義ではなく_fp_near_zoneを
    再利用していることを確認する。"""
    idx = _SRC.index("if _footprint_risk:")
    snippet = _SRC[idx:idx + 2000]
    assert "elif _fp_near_zone:" in snippet


def test_source_lat_ttc_update_still_receives_footprint_risk_unchanged():
    """回帰: LateralTTCMonitor.update()へ渡す値は引き続き_footprint_risk本体
    (危険域全体ではなく物理下限本体)のままであることを確認する
    (footprint_risk=_fp_near_zoneに誤って置き換わっていないか)。"""
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx:idx + 900]
    assert "footprint_risk=_footprint_risk" in snippet
    assert "footprint_risk=_fp_near_zone" not in snippet


def test_source_v_safe_hard_cap_still_gated_by_footprint_risk_not_near_zone():
    """回帰: v_safe候補スタックの本体側キャップ(wall_slow_speedへの完全キャップ)
    は引き続き_footprint_risk本体でゲートされており、_fp_near_zoneに誤って
    広げられていないことを確認する(誤って広げるとteaper域でも急停止して
    しまい153/154節の改善が無効化される)。"""
    idx = _SRC.index('_v_safe_cand.append(("footprint_risk(車体重なりリスク)"')
    snippet = _SRC[max(0, idx - 400):idx]
    assert "if _footprint_risk:" in snippet
