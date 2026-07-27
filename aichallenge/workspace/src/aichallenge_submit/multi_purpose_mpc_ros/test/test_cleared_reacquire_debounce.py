"""Unit tests for the _ot_cleared reacquire-debounce fix (2026-07-14, 0714-03 事象③追補).

mpc_controller.py imports rclpy/autoware message types at module scope, so the
Fix-2 cleared-recompute block is mirrored here (verbatim transcription):

    if fd < clear_lat_reacquire:
        reacquire_count += 1
        if reacquire_count >= engage_debounce:
            cleared = False
    else:
        reacquire_count = 0
        if fd >= clear_lat_release or (fd >= fwd_min_lat_sep and fs <= clear_ds_beside):
            if alpha >= 1.0 - 1e-3:
                cleared = True

Bug (0714-03実測, wp204-209/wp232 pattern): once cleared=True was reached (dlat grew
past clear_lat_release=2.1 through a corner), track-curvature-driven dlat oscillation
briefly dipped fwd_dlat back below clear_lat_reacquire=1.6 for a single cycle (a
transient side-effect of the reference line's curvature, not genuine opponent
re-approach), instantly resetting cleared to False. This re-armed LAT-TTC's strict
giveup_space_m(1.85) threshold right as the corridor was also naturally narrowing from
track shape, producing a spurious C2 giveup and a repeating
engage→almost-clear→reacquire→giveup cycle that never let the pass complete.
"""
import pytest

CLEAR_LAT_REACQUIRE = 1.6
CLEAR_LAT_RELEASE = 2.1
FWD_MIN_LAT_SEP = 1.8
CLEAR_DS_BESIDE = 1.0
ENGAGE_DEBOUNCE = 8


def cleared_step(fd, fs, alpha, cleared, reacquire_count,
                  reacquire=CLEAR_LAT_REACQUIRE, release=CLEAR_LAT_RELEASE,
                  min_lat_sep=FWD_MIN_LAT_SEP, ds_beside=CLEAR_DS_BESIDE,
                  debounce=ENGAGE_DEBOUNCE):
    if fd < reacquire:
        reacquire_count += 1
        if reacquire_count >= debounce:
            cleared = False
    else:
        reacquire_count = 0
        if fd >= release or (fd >= min_lat_sep and fs is not None and fs <= ds_beside):
            if alpha >= 1.0 - 1e-3:
                cleared = True
    return cleared, reacquire_count


def test_single_cycle_dip_below_reacquire_does_not_clear_immediately():
    """0714-03再現: 一度cleared=Trueに達した後、dlatが1周期だけreacquire閾値を
    下回っても(コーナー形状由来の一時的なもの)、即座にはclearedを解除しない。"""
    cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=True, reacquire_count=0)
    assert cleared is True
    assert count == 1


def test_sustained_reacquire_over_debounce_cycles_eventually_clears():
    """真に相手が再接近する場合(dlatが継続的に閾値未満)は、engage_debounce周期
    以内に確実にcleared=Falseへ戻る(安全性は犠牲にしない)。"""
    cleared, count = True, 0
    for _ in range(ENGAGE_DEBOUNCE - 1):
        cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=cleared,
                                       reacquire_count=count)
        assert cleared is True  # debounce未満の間は維持
    cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=cleared,
                                   reacquire_count=count)
    assert cleared is False  # engage_debounce周期目でようやく解除


def test_recovering_above_reacquire_before_debounce_resets_count():
    """回帰: デバウンス完了前にdlatがreacquire閾値以上へ回復すれば、カウントは
    リセットされ、clearedは解除されないまま維持される(単発ディップの吸収)。"""
    cleared, count = True, 0
    cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=cleared, reacquire_count=count)
    assert count == 1
    cleared, count = cleared_step(fd=2.0, fs=5.0, alpha=1.0, cleared=cleared, reacquire_count=count)
    assert count == 0
    assert cleared is True


def test_boundary_debounce_count_exactly_at_threshold_clears():
    """境界値: reacquire_countがengage_debounceちょうどで解除する。"""
    cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=True,
                                   reacquire_count=ENGAGE_DEBOUNCE - 1)
    assert count == ENGAGE_DEBOUNCE
    assert cleared is False


def test_boundary_debounce_count_just_below_threshold_does_not_clear():
    """境界値: reacquire_countがengage_debounceを1下回る場合は維持する。"""
    cleared, count = cleared_step(fd=1.4, fs=5.0, alpha=1.0, cleared=True,
                                   reacquire_count=ENGAGE_DEBOUNCE - 2)
    assert count == ENGAGE_DEBOUNCE - 1
    assert cleared is True


def test_release_path_unaffected_by_reacquire_debounce_regression():
    """回帰: 解放側(dlat>=release、alpha>=1)は従来通り即座にclearedへ昇格する
    (今回の変更は再取得側のみに影響)。"""
    cleared, count = cleared_step(fd=2.5, fs=5.0, alpha=1.0, cleared=False, reacquire_count=3)
    assert cleared is True
    assert count == 0  # 解放ゾーンに入ったのでカウントもリセットされる
