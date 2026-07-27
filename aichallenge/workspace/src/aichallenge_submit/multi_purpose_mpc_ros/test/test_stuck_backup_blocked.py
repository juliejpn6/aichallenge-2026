"""Unit tests for the BACKUP-blocked detector (48節, 2026-07-14).

`_handle_stuck_recovery` depends on GearCommand/GearReport (ROS message types),
so the decision logic is mirrored verbatim here (see test_ot_offset_ramp.py for
the same rationale). Mirrors the exact branch order shipped in mpc_controller.py:
dist>=backup_dist_eff -> success; zero_v_elapsed>=blocked_confirm_s -> blocked
(shuffle continues towards PUSH, or gives up once shuffle_max_cycles is
exhausted); backup_elapsed>=timeout_s -> retry; else continue.

2026-07-26更新(184節、ユーザー提案「縦列駐車から抜け出すように、短い後退→
隙間への微調整前進を繰り返す」): 壁リカバリー(AWSIM組み込みの90°自動修正)を
off化した(183節)ため、後方がほぼ塞がっていて僅かしか後退できない状況は
異常ではなく想定内のケースとして扱うことにした。従来「blocked」は常に
即座にNORMALへ断念していたが、シャッフル上限(shuffle_max_cycles)に達する
までは断念せずWAIT_DRIVE_PUSHへ進む("blocked_shuffle")よう変更し、上限到達後
のみ従来通り断念する("blocked_giveup")。またBACKUP距離の閾値も固定の
BACKUP_DISTではなく、後方の相手車までの安全マージンから決まる実効値
(backup_dist_eff、_rear_clearance_mで計算)を使うようになった。
"""
import math

import pytest

BACKUP_DIST = 2.0
BACKUP_TIMEOUT_S = 5.0
BLOCKED_V_THR = 0.05
BLOCKED_CONFIRM_S = 1.5
REAR_CLEARANCE_MARGIN_M = 0.5
SHUFFLE_MAX_CYCLES = 6
SHUFFLE_EPISODE_GAP_S = 6.0
SHUFFLE_EPISODE_RADIUS_M = 3.0


class BackupSim:
    """mpc_controller.py `_handle_stuck_recovery`のBACKUP分岐のミラー。"""

    def __init__(self, backup_dist_eff=BACKUP_DIST):
        self.zero_v_since = None
        self.t = 0.0
        self.backup_dist_eff = backup_dist_eff

    def step(self, dist, v_now, backup_elapsed, shuffle_cycle=0,
              shuffle_max_cycles=SHUFFLE_MAX_CYCLES):
        if abs(v_now) < BLOCKED_V_THR:
            if self.zero_v_since is None:
                self.zero_v_since = self.t
            zero_v_elapsed = self.t - self.zero_v_since
        else:
            self.zero_v_since = None
            zero_v_elapsed = 0.0

        if dist >= self.backup_dist_eff:
            self.zero_v_since = None
            return "success"
        elif zero_v_elapsed >= BLOCKED_CONFIRM_S:
            self.zero_v_since = None
            if shuffle_cycle < shuffle_max_cycles:
                return "blocked_shuffle"
            return "blocked_giveup"
        elif backup_elapsed >= BACKUP_TIMEOUT_S:
            return "retry"
        else:
            return "continue"


def mirror_rear_clearance(backup_dist, ds_behind_list, margin=REAR_CLEARANCE_MARGIN_M,
                           scan_max_dist=3.0):
    """`_rear_clearance_m`のミラー。ds<0(後方)の相手車のみを対象に、最も近い
    相手車までの距離からmarginを引いた値を安全な後退距離の上限とする。"""
    limit = backup_dist
    for ds in ds_behind_list:
        if ds >= 0.0 or ds < -scan_max_dist:
            continue
        room = max(0.0, -ds - margin)
        limit = min(limit, room)
    return max(0.0, limit)


def mirror_shuffle_update(prev_cycle, last_end_time, last_pose, now, pose,
                           gap_s=SHUFFLE_EPISODE_GAP_S, radius_m=SHUFFLE_EPISODE_RADIUS_M):
    """`_stuck_update_shuffle_cycle`のミラー。"""
    if last_end_time is not None and last_pose is not None:
        dt = now - last_end_time
        d = math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
        if dt <= gap_s and d <= radius_m:
            return prev_cycle + 1
    return 0


def run_until_result(sim, dist_fn, v_fn, dt=0.25, max_cycles=40, shuffle_cycle=0):
    result = "continue"
    for i in range(max_cycles):
        sim.t = (i + 1) * dt
        result = sim.step(dist=dist_fn(sim.t), v_now=v_fn(sim.t), backup_elapsed=sim.t,
                           shuffle_cycle=shuffle_cycle)
        if result != "continue":
            break
    return result, sim.t


def test_wedged_against_wall_detected_within_1_5s_not_5s():
    """0713-05実測(v=-0.00が44回連続)相当: 完全に動けない場合、5秒/600秒の
    タイムアウトを待たず1.5秒以内に確定する。
    2026-07-26更新(184節): シャッフル上限未満(既定shuffle_cycle=0)では即断念
    (旧"blocked")ではなく、PUSHへ進む"blocked_shuffle"になる。"""
    sim = BackupSim()
    result, t = run_until_result(sim, dist_fn=lambda t: 0.0, v_fn=lambda t: -0.00)
    assert result == "blocked_shuffle"
    assert t <= 2.0  # 5.0s(既存timeout)より十分早い


def test_normal_backup_reaches_success_without_false_blocked():
    """回帰: 正常に後退できている場合はblockedを誤検知せずsuccessに到達する。"""
    sim = BackupSim()
    result, _t = run_until_result(sim, dist_fn=lambda t: min(2.0, t * 1.0),
                                   v_fn=lambda t: -1.0)
    assert result == "success"


def test_slow_but_moving_backup_falls_through_to_existing_timeout_retry():
    """回帰: 動いてはいるが2m未到達のまま5秒経過した場合、blockedにはならず
    従来通りretry(既存のdist/timeout判定)になる。"""
    sim = BackupSim()
    result, _t = run_until_result(sim, dist_fn=lambda t: 0.3, v_fn=lambda t: -0.2)
    assert result == "retry"


def test_brief_stall_under_1_5s_recovers_without_blocked():
    """境界値: 1.5秒未満の一時停止から回復した場合はblockedと確定しない。"""
    sim = BackupSim()
    # 5周期(1.25秒、1.5秒未満)だけ静止し、その後動き出す
    result = "continue"
    for i in range(6):
        sim.t = (i + 1) * 0.25
        v = -0.04 if i < 5 else -1.0
        result = sim.step(dist=0.0, v_now=v, backup_elapsed=sim.t)
    assert result == "continue"


@pytest.mark.parametrize("v_now,expected_zero_v", [
    (-0.049, True),   # 閾値未満(僅かに)
    (-0.05, False),   # 閾値ちょうど(厳密な `<` 比較なので「動いている」扱い)
    (-0.051, False),  # 閾値をわずかに超える
])
def test_blocked_v_threshold_boundary(v_now, expected_zero_v):
    """境界値: BLOCKED_V_THR(0.05)ちょうどの扱いを確認する。"""
    sim = BackupSim()
    sim.t = 0.25
    sim.step(dist=0.0, v_now=v_now, backup_elapsed=0.25)
    assert (sim.zero_v_since is not None) == expected_zero_v


def test_dist_reaching_target_takes_priority_even_if_currently_near_zero_v():
    """回帰: 既に2m進んだ後にたまたま瞬間速度が0近くでも、成功判定が優先される。"""
    sim = BackupSim()
    sim.t = 0.25
    result = sim.step(dist=2.05, v_now=-0.01, backup_elapsed=0.25)
    assert result == "success"


# --- 184節追加: シャッフル(縦列駐車脱出)関連 ---

def test_blocked_proceeds_to_shuffle_push_below_max_cycles():
    """新規: シャッフル上限未満なら、後退不能でも断念せずPUSHへ進む
    ("blocked_shuffle")。"""
    sim = BackupSim()
    result, _t = run_until_result(sim, dist_fn=lambda t: 0.0, v_fn=lambda t: -0.0,
                                   shuffle_cycle=SHUFFLE_MAX_CYCLES - 1)
    assert result == "blocked_shuffle"


def test_blocked_gives_up_at_max_cycles():
    """新規: シャッフル上限に達していたら、後退不能時は従来通り断念する
    ("blocked_giveup")。無限リトライを避ける安全側バックストップ。"""
    sim = BackupSim()
    result, _t = run_until_result(sim, dist_fn=lambda t: 0.0, v_fn=lambda t: -0.0,
                                   shuffle_cycle=SHUFFLE_MAX_CYCLES)
    assert result == "blocked_giveup"


def test_rear_clearance_caps_backup_distance_when_opponent_close_behind():
    """新規: 後方0.8mに相手車がいる場合、margin(0.5m)を引いた0.3mが
    実効後退距離の上限になる(2.0mへは到達しない)。"""
    room = mirror_rear_clearance(BACKUP_DIST, ds_behind_list=[-0.8])
    assert room == pytest.approx(0.3)


def test_rear_clearance_uses_nearest_of_multiple_opponents():
    """新規: 後方に複数の相手車がいる場合、最も近い(=最も制限が厳しい)車が
    採用される。"""
    room = mirror_rear_clearance(BACKUP_DIST, ds_behind_list=[-2.5, -0.6, -1.5])
    assert room == pytest.approx(0.1)


def test_rear_clearance_ignores_forward_and_far_vehicles():
    """新規: 前方(ds>=0)の車、探索範囲外(3.0mより遠い)の車は無視され、
    後退距離はbackup_distのまま(壁激突直後は前方に相手車がいてもおかしくない
    ため、後方チェックに前方車を混同しない)。"""
    room = mirror_rear_clearance(BACKUP_DIST, ds_behind_list=[0.5, -5.0])
    assert room == pytest.approx(BACKUP_DIST)


def test_rear_clearance_never_goes_negative_when_opponent_touching():
    """境界値: 相手車が既にmargin以内まで接近している場合でも、後退距離は
    0未満にはならない(0でBACKUPをほぼ即完了扱いにし、直ちにPUSHへ進む)。"""
    room = mirror_rear_clearance(BACKUP_DIST, ds_behind_list=[-0.1])
    assert room == 0.0


def test_shuffle_cycle_increments_when_restuck_soon_and_nearby():
    """新規: 直前のPUSH完了から短時間・近距離で再スタックした場合、
    同一エピソードの継続とみなしシャッフルサイクルをインクリメントする。"""
    cycle = mirror_shuffle_update(prev_cycle=2, last_end_time=10.0, last_pose=(5.0, 5.0),
                                   now=13.0, pose=(6.0, 5.0))
    assert cycle == 3


def test_shuffle_cycle_resets_when_gap_too_long():
    """新規: 前回の復帰完了から時間が経ちすぎている場合は別エピソードとみなし
    0へ戻す。"""
    cycle = mirror_shuffle_update(prev_cycle=3, last_end_time=10.0, last_pose=(5.0, 5.0),
                                   now=20.0, pose=(5.0, 5.0))
    assert cycle == 0


def test_shuffle_cycle_resets_when_too_far_away():
    """新規: 前回の復帰完了地点から離れすぎている場合は別エピソードとみなし
    0へ戻す(別のコーナーでの新規STUCKを、前回のシャッフル続きとして誤集計
    しない)。"""
    cycle = mirror_shuffle_update(prev_cycle=3, last_end_time=10.0, last_pose=(5.0, 5.0),
                                   now=11.0, pose=(50.0, 50.0))
    assert cycle == 0


def test_shuffle_cycle_starts_at_zero_on_first_episode():
    """回帰: エピソード記録が無い(初回のSTUCK)場合はシャッフルサイクル0から
    始まる。"""
    cycle = mirror_shuffle_update(prev_cycle=0, last_end_time=None, last_pose=None,
                                   now=1.0, pose=(0.0, 0.0))
    assert cycle == 0
