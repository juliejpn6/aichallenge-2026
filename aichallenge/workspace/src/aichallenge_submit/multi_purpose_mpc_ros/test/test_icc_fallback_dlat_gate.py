"""Unit tests for the ICC-fallback dlat gate fix (2026-07-14, 0714-04 事象①最重要).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
decision logic is mirrored here (verbatim transcription of the shipped lines,
matching the style of test_ot_offset_ramp.py):

    _icc_fallback_candidate = (_vlim is None and _fwd_ds is not None
                                and _fwd_vopp is not None
                                and _fwd_ds <= self._fwd_near_range)
    _icc_fallback_on_path = (_fwd_dlat_val is None
                             or _fwd_dlat_val <= self._ot_engage_lat_max)
    _icc_fallback_on = _icc_fallback_candidate and _icc_fallback_on_path

Bug (0714-04実測, wp160-178, recurring across all 3 laps): the ICC-fallback
(_icc_fallback_on, added 2026-07-12) had no lateral(dlat) term at all -- unlike
the properly H2-unified (2026-07-09) engage gate `_on_path`, which requires
fwd_dlat <= engage_lat_max. A stopped opponent sitting clearly to the side
(fwd_dlat=2.15-2.45m, Rfree=5.44m of clear room) still got braked to a dead
stop (v_safe=0.0) by the fallback's pure-longitudinal G2 formula, while the
engage gate refused to even consider it a pass target (dlat > engage_lat_max=
2.0m) -- reproducing the exact "ICC forces v=0 forever, engage never fires"
deadlock class that the 2026-07-09 H2 fix had already closed once, through a
fallback path added 3 days later that never inherited the fix. In the actual
log this specific instance only ever resolved via the unrelated STUCK-BACKUP
recovery state machine, never via the overtake/ICC logic itself.
"""
import pytest

FWD_NEAR_RANGE = 6.0
ENGAGE_LAT_MAX = 2.0
A_BRAKE = 1.3
MARGIN_CENTER = 4.0


def g2_speed(v_fwd, ds, a_brake=A_BRAKE, margin_center=MARGIN_CENTER):
    import math
    rad = v_fwd * v_fwd + 2.0 * a_brake * (ds - margin_center)
    return math.sqrt(max(0.0, rad))


def icc_fallback_decision(vlim, fwd_ds, fwd_vopp, fwd_dlat,
                           near_range=FWD_NEAR_RANGE, engage_lat_max=ENGAGE_LAT_MAX):
    """mpc_controller.py の該当ブロックの逐語ミラー。戻り値: (on, skip)。"""
    candidate = (vlim is None and fwd_ds is not None and fwd_vopp is not None
                 and fwd_ds <= near_range)
    on_path = (fwd_dlat is None or fwd_dlat <= engage_lat_max)
    on = candidate and on_path
    skip = candidate and not on_path
    return on, skip


def test_retroactive_0714_04_wp176_no_longer_forces_dead_stop():
    """0714-04実測(wp176, fwd_ds=3.00, fwd_dlat=2.25, vopp=0.0)の遡及検証。
    修正前はfallbackが発動しv_safe=g2_speed(0.0, 3.00)=0.0(完全停止、実測と一致)。
    修正後はfwd_dlat(2.25)がengage_lat_max(2.0)を超えるためfallback非発動となり、
    このフォールバック起因の停止は解消される。"""
    on_old, _ = icc_fallback_decision(vlim=None, fwd_ds=3.00, fwd_vopp=0.0,
                                       fwd_dlat=2.25, engage_lat_max=float("inf"))
    assert on_old is True
    assert g2_speed(0.0, 3.00) == pytest.approx(0.0)  # 実測v_safe=0.0と一致

    on_new, skip_new = icc_fallback_decision(vlim=None, fwd_ds=3.00, fwd_vopp=0.0,
                                              fwd_dlat=2.25)
    assert on_new is False
    assert skip_new is True


def test_retroactive_0714_04_wp178_frozen_gap_no_longer_forces_dead_stop():
    """0714-04実測(wp178, fwd_ds=0.9994, fwd_dlat=2.39, 15秒以上固着していた地点)の
    遡及検証。修正後はこの地点でもfallbackが介入しなくなる。"""
    on_new, skip_new = icc_fallback_decision(vlim=None, fwd_ds=0.9994364943018184,
                                              fwd_vopp=0.0, fwd_dlat=2.39)
    assert on_new is False
    assert skip_new is True


def test_dlat_none_still_falls_back_regression():
    """回帰: dlatが取得できない場合は従来通り安全側(fallback発動)を維持する。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=3.0, fwd_vopp=0.0, fwd_dlat=None)
    assert on is True
    assert skip is False


def test_original_0712_03_rescue_case_still_works_regression():
    """回帰: 0712-03の本来の救済対象(dlatがnear_sepをわずかに超えただけの残存オフセット、
    例えばdlat=1.9m)は、engage_lat_max(2.0m)以内のためfallbackが引き続き発動する。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=3.0, fwd_vopp=0.0, fwd_dlat=1.9)
    assert on is True
    assert skip is False


def test_boundary_exactly_at_engage_lat_max_still_activates():
    """境界値: fwd_dlat=engage_lat_max(2.0)ちょうどは`<=`のためfallback対象のまま。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=3.0, fwd_vopp=0.0, fwd_dlat=2.0)
    assert on is True
    assert skip is False


def test_boundary_just_above_engage_lat_max_skips():
    """境界値: engage_lat_maxをわずかに上回るとfallbackは介入しない。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=3.0, fwd_vopp=0.0, fwd_dlat=2.001)
    assert on is False
    assert skip is True


def test_vlim_present_never_activates_fallback_regression():
    """回帰: 通常ICC(_vlim)が既に対象車を捕捉している場合はfallback自体が候補にならない
    (dlatに関わらずcandidate=False)。"""
    on, skip = icc_fallback_decision(vlim=2.0, fwd_ds=3.0, fwd_vopp=0.0, fwd_dlat=0.5)
    assert on is False
    assert skip is False


def test_ds_beyond_near_range_never_activates_fallback_regression():
    """回帰: fwd_dsがfwd_near_range(6.0m)を超える遠方の相手にはfallbackが介入しない
    (dlatに関わらずcandidate=False)。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=6.01, fwd_vopp=0.0, fwd_dlat=0.5)
    assert on is False
    assert skip is False


def test_boundary_exactly_at_fwd_near_range_still_a_candidate_regression():
    """境界値: fwd_ds=fwd_near_range(6.0)ちょうどは`<=`のため引き続きcandidate扱い
    (dlatがengage_lat_max以内なら発動、超えていればskip)。"""
    on, skip = icc_fallback_decision(vlim=None, fwd_ds=6.0, fwd_vopp=0.0, fwd_dlat=0.5)
    assert on is True
    assert skip is False
    on2, skip2 = icc_fallback_decision(vlim=None, fwd_ds=6.0, fwd_vopp=0.0, fwd_dlat=2.5)
    assert on2 is False
    assert skip2 is True
