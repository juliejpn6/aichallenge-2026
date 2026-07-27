"""Unit tests for the F3-taper est_gap formula fix (2026-07-14, 0714-03 事象②).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
formula is mirrored here (verbatim transcription of the shipped lines):

    _est_gap = float(_vtgt[0]) + float(_vtgt[3])   # ds + dlat (2026-07-14, was hypot)
    if _est_gap >= f3_taper_gap: floor = v_creep
    elif _est_gap <= hard_stop_gap: floor = 0.0
    else: floor = v_creep * (_est_gap - hard_stop_gap) / (f3_taper_gap - hard_stop_gap)

Bug (0714-03実測, wp171-174): hypot(ds, dlat) under-credits lateral progress when ds
dominates (Euclidean combination discounts the smaller leg). With ds≈1.93m stuck near
hard_stop_gap while dlat grew from 0.39→1.06m over ~6s, the floor stayed pinned at
~0.2-0.3 m/s the whole time (hypot barely exceeds ds alone), starving the vehicle of
the forward speed needed to develop any meaningful lateral offset via steering — a
self-sustaining low-speed trap. Per the stated domain rule ("approaching the
opponent's side incurs no penalty, only rear-ending/complete stop does"), lateral
clearance should count fully toward the escape gap, not be discounted geometrically.
"""
import pytest

HARD_STOP_GAP = 1.8
F3_TAPER_GAP = 3.0
V_CREEP = 1.5


def est_gap_new(ds, dlat):
    return float(ds) + float(dlat)


def floor_from_gap(est_gap, hard_stop_gap=HARD_STOP_GAP, f3_taper_gap=F3_TAPER_GAP,
                    v_creep=V_CREEP):
    if est_gap >= f3_taper_gap:
        return v_creep
    if est_gap <= hard_stop_gap:
        return 0.0
    frac = (est_gap - hard_stop_gap) / (f3_taper_gap - hard_stop_gap)
    return v_creep * frac


def test_retroactive_0714_03_wp171_floor_roughly_triples():
    """0714-03実測wp171(ds=1.933557, dlat=0.391613)での遡及比較。
    旧hypot: floor≈0.216(実測v_safe=0.21602067771758715と一致)。
    新ds+dlat: floorは約3倍(0.65超)へ改善する。"""
    ds, dlat = 1.933557415591764, 0.39161311224951456
    import numpy as np
    old_gap = float(np.hypot(ds, dlat))
    old_floor = floor_from_gap(old_gap)
    assert old_floor == pytest.approx(0.216, abs=0.01)  # 実測値の再現(回帰確認用)

    new_gap = est_gap_new(ds, dlat)
    new_floor = floor_from_gap(new_gap)
    assert new_floor > old_floor * 2.5   # 大幅な改善(2.5倍超)


def test_dlat_zero_is_backward_compatible():
    """回帰: dlat=0の場合、新旧の式は完全に一致する(後方互換)。"""
    ds = 2.2
    assert est_gap_new(ds, 0.0) == pytest.approx(ds)


def test_boundary_exactly_at_hard_stop_gap_still_zero_floor():
    """境界値: est_gapがhard_stop_gap(1.8)ちょうどならfloor=0(完全停止、変更なし)。"""
    assert floor_from_gap(1.8) == 0.0


def test_boundary_exactly_at_f3_taper_gap_gives_full_creep():
    """境界値: est_gapがf3_taper_gap(3.0)ちょうどならfloor=v_creep(通常クリープ全開)。"""
    assert floor_from_gap(3.0) == pytest.approx(V_CREEP)


def test_lateral_progress_alone_can_release_floor_even_with_close_ds():
    """ドメイン仕様確認: dsが依然hard_stop_gap未満でも、dlatの成長だけでfloorが
    v_creepまで解放されうる(相手サイドへの接近自体にペナルティが無いため)。"""
    gap = est_gap_new(ds=1.0, dlat=2.0)  # ds単独ならhard_stop_gap未満で本来floor=0
    assert gap == pytest.approx(3.0)
    assert floor_from_gap(gap) == pytest.approx(V_CREEP)


def test_no_overshoot_beyond_v_creep():
    """回帰: est_gapがf3_taper_gapを超えてもfloorはv_creepを超えない(二値の上限維持)。"""
    assert floor_from_gap(est_gap_new(10.0, 10.0)) == pytest.approx(V_CREEP)
