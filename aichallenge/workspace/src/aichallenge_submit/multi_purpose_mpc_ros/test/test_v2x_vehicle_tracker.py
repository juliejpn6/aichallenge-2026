"""Unit tests for V2XVehicleTracker (pure Python, no rclpy)."""

from dataclasses import dataclass
from typing import List

import pytest

from multi_purpose_mpc_ros.v2x_vehicle_tracker import V2XVehicleTracker


# Lightweight stand-ins for v2x_msgs / std_msgs / geometry_msgs so tests
# do not require the ROS message DLLs to be importable.
@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp


@dataclass
class _Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _V2XVehiclePosition:
    header: _Header
    vehicle_id: str
    position: _Point


@dataclass
class _V2XVehiclePositionArray:
    header: _Header
    vehicles: List[_V2XVehiclePosition]


def _msg(stamp_sec: float, vehicles):
    """Build a fake V2XVehiclePositionArray with the given (vehicle_id, x, y)."""
    sec = int(stamp_sec)
    nanosec = int((stamp_sec - sec) * 1e9)
    array_header = _Header(_Stamp(sec, nanosec))
    out = []
    for vid, x, y in vehicles:
        out.append(_V2XVehiclePosition(
            header=_Header(_Stamp(sec, nanosec)),
            vehicle_id=vid,
            position=_Point(x=x, y=y),
        ))
    return _V2XVehiclePositionArray(header=array_header, vehicles=out)


def test_two_samples_constant_velocity_yields_finite_difference():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)

    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 2.5)]))

    vx, vy = tracker.velocity("d2")
    assert vx == pytest.approx(10.0)
    assert vy == pytest.approx(5.0)


def test_single_sample_yields_zero_velocity():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)

    tracker.update(_msg(0.0, [("d2", 1.0, 2.0)]))

    assert tracker.velocity("d2") == (0.0, 0.0)


def test_unknown_vehicle_velocity_is_zero():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    assert tracker.velocity("d9") == (0.0, 0.0)


def test_predict_positions_constant_velocity():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 2.5)]))  # vx=10, vy=5, latest (5,2.5)

    points = tracker.predict_positions("d2", [0.0, 0.5, 1.0])

    assert points[0] == pytest.approx((5.0, 2.5))
    assert points[1] == pytest.approx((10.0, 5.0))
    assert points[2] == pytest.approx((15.0, 7.5))


def test_position_jump_resets_velocity_and_drops_old_sample():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))  # 100 m jump > 5 m

    assert tracker.velocity("d2") == (0.0, 0.0)
    # Predictions should anchor at the *new* position with zero velocity.
    points = tracker.predict_positions("d2", [0.0, 0.5])
    assert points[0] == pytest.approx((100.0, 0.0))
    assert points[1] == pytest.approx((100.0, 0.0))


def test_velocity_above_safety_cap_is_zeroed():
    # 50 m / 0.05 s = 1000 m/s, well above v_max_safety=30
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=200.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.05, [("d2", 50.0, 0.0)]))

    assert tracker.velocity("d2") == (0.0, 0.0)


def test_retroactive_0715_01_implausible_vopp_now_clamped_with_config_value():
    """遡及検証(2026-07-15、v_max_safetyの是正): 0715-01実測ログでは、全カート
    15km/h(4.17m/s)キャップの本競技において、vopp(相手速度推定)が6〜16m/sという
    物理的に不可能な値を頻発していた。旧config値(v_max_safety=30.0)ではこの
    サニティクランプが一度も発動していなかったことを確認したうえで、是正後の値
    (config.yaml、6.0)で同じ入力を再生するとクランプが正しく発動することを確認する。
    実測相当: 12m離れた2点間を0.8秒で移動(=15m/s相当)。"""
    old_tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=200.0)
    old_tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    old_tracker.update(_msg(0.8, [("d2", 12.0, 0.0)]))
    assert old_tracker.velocity("d2") == pytest.approx((15.0, 0.0))  # 旧値では素通し(バグ実測を再現)

    new_tracker = V2XVehicleTracker(v_max_safety=6.0, position_jump_threshold=200.0)
    new_tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    new_tracker.update(_msg(0.8, [("d2", 12.0, 0.0)]))
    assert new_tracker.velocity("d2") == (0.0, 0.0)  # 是正後は正しくゼロへクランプ


def test_genuine_top_speed_kart_not_clamped_regression():
    """回帰: 本競技の最高速度(15km/h=4.1667m/s)相当の正当な速度は、是正後の
    v_max_safety(6.0)でもクランプされない(誤検知しない余裕を持たせている)。"""
    tracker = V2XVehicleTracker(v_max_safety=6.0, position_jump_threshold=200.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(1.0, [("d2", 4.1667, 0.0)]))
    assert tracker.velocity("d2") == pytest.approx((4.1667, 0.0))


def test_two_vehicles_tracked_independently():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0), ("d3", 10.0, 12.5)]))

    assert tracker.velocity("d2") == pytest.approx((10.0, 0.0))
    assert tracker.velocity("d3") == pytest.approx((0.0, 5.0))


def test_active_ids_reflect_latest_message_only():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0)]))  # d3 dropped this tick

    assert tracker.active_vehicle_ids() == ["d2"]
    # d3 is still in the internal state but not reported as active.
    assert tracker.velocity("d3") == pytest.approx((0.0, 0.0))


def test_predict_all_returns_only_active_vehicles():
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=20.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0), ("d3", 10.0, 10.0)]))
    tracker.update(_msg(0.5, [("d2", 5.0, 0.0)]))  # d3 dropped

    out = tracker.predict_all([0.0, 1.0])
    assert set(out.keys()) == {"d2"}
    assert out["d2"][0] == pytest.approx((5.0, 0.0))
    assert out["d2"][1] == pytest.approx((15.0, 0.0))


@dataclass
class _StubObstacle:
    cx: float
    cy: float
    radius: float


def test_predictions_to_obstacles_flattens_with_radius():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles

    predictions = {
        "d2": [(1.0, 2.0), (3.0, 4.0)],
        "d3": [(5.0, 6.0)],
    }
    obstacles = predictions_to_obstacles(
        predictions, vehicle_radius=0.5, obstacle_cls=_StubObstacle)

    centers = sorted((o.cx, o.cy, o.radius) for o in obstacles)
    assert centers == sorted([
        (1.0, 2.0, 0.5),
        (3.0, 4.0, 0.5),
        (5.0, 6.0, 0.5),
    ])


def test_predictions_to_obstacles_empty_input():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles
    assert predictions_to_obstacles(
        {}, vehicle_radius=0.5, obstacle_cls=_StubObstacle) == []


# --- predictions_to_obstacles_capsule (131-6節②、寸法モデルの一元化, 2026-07-20) ---
# 背景: 相手車1台=円1個(半径vehicle_radius)を将来位置ごとにスタンプする方式では、
# 停止/低速車(将来サンプルが現在位置とほぼ重なる)の全長方向が一切表現されず、また
# 予測は未来方向のみのため「相手の後端」側(egoが追い越し中に最も接近する側)が
# 速度に関わらず一度も表現されない盲点があった。現在位置(t=0)のみ進行方向へ前後
# 分割する。


def test_capsule_splits_t0_into_two_circles_along_heading():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(0.0, 0.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={"d3": 0.0},
        half_length=1.0, obstacle_cls=_StubObstacle)

    centers = sorted((round(o.cx, 6), round(o.cy, 6), o.radius) for o in obstacles)
    assert centers == sorted([
        (0.2, 0.0, 0.8),
        (-0.2, 0.0, 0.8),
    ])


def test_capsule_offset_uses_official_vehicle_spec_derived_values():
    """遡及検証: 実際のconfig値(along_min_length/2=1.00, vehicle_radius=0.8)を
    使った場合、オフセットが0.20mになることを確認する(公式車両仕様: 全長200cm由来)。"""
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(10.0, 5.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={"d3": 0.0},
        half_length=1.0, obstacle_cls=_StubObstacle)

    xs = sorted(round(o.cx, 6) for o in obstacles)
    assert xs == [9.8, 10.2]


def test_capsule_future_samples_stay_single_circle_no_cpu_doubling():
    """[[mpc-cost-doubles-with-obstacles]]対策: t>0(将来予測)は従来通り1個のまま、
    t=0のみ2個になることを確認する(近傍車1台あたり+1個に抑える設計)。"""
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={"d3": 0.0},
        half_length=1.0, obstacle_cls=_StubObstacle)

    assert len(obstacles) == 4  # t=0→2個 + t=1,2→各1個
    future = sorted((round(o.cx, 6), o.radius) for o in obstacles if o.cx in (1.0, 2.0))
    assert future == [(1.0, 0.8), (2.0, 0.8)]


def test_capsule_offset_clamped_to_zero_when_half_length_not_larger_than_radius():
    """half_length<=vehicle_radius(円だけで全長を覆えている場合)は分割せず
    従来通り1個のままにする(オフセット負値による退化を防ぐ)。"""
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(0.0, 0.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={"d3": 0.0},
        half_length=0.5, obstacle_cls=_StubObstacle)

    assert len(obstacles) == 1
    assert (obstacles[0].cx, obstacles[0].cy, obstacles[0].radius) == (0.0, 0.0, 0.8)


def test_capsule_missing_heading_defaults_to_zero():
    """headingsに未登録のvid(異常系)はheading=0.0(+x方向)へfail-openすることを確認する。"""
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(0.0, 0.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={},
        half_length=1.0, obstacle_cls=_StubObstacle)

    ys = [round(o.cy, 6) for o in obstacles]
    assert ys == [0.0, 0.0]


def test_capsule_heading_perpendicular_offsets_along_y():
    import math
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    predictions = {"d3": [(0.0, 0.0)]}
    obstacles = predictions_to_obstacles_capsule(
        predictions, vehicle_radius=0.8, headings={"d3": math.pi / 2},
        half_length=1.0, obstacle_cls=_StubObstacle)

    centers = sorted((round(o.cx, 6), round(o.cy, 6)) for o in obstacles)
    assert centers == [(0.0, -0.2), (0.0, 0.2)]


def test_capsule_empty_points_skipped():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    obstacles = predictions_to_obstacles_capsule(
        {"d3": []}, vehicle_radius=0.8, headings={"d3": 0.0},
        half_length=1.0, obstacle_cls=_StubObstacle)
    assert obstacles == []


def test_capsule_empty_input():
    from multi_purpose_mpc_ros.v2x_vehicle_tracker import predictions_to_obstacles_capsule

    assert predictions_to_obstacles_capsule(
        {}, vehicle_radius=0.8, headings={}, half_length=1.0,
        obstacle_cls=_StubObstacle) == []


def test_position_jump_invokes_warn_callback():
    msgs = []
    tracker = V2XVehicleTracker(
        v_max_safety=30.0,
        position_jump_threshold=5.0,
        warn_callback=msgs.append,
    )
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))

    assert any("position jump" in m for m in msgs)
    assert any("d2" in m for m in msgs)


def test_velocity_cap_invokes_warn_callback():
    msgs = []
    tracker = V2XVehicleTracker(
        v_max_safety=30.0,
        position_jump_threshold=200.0,
        warn_callback=msgs.append,
    )
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.05, [("d2", 50.0, 0.0)]))

    assert any("velocity" in m for m in msgs)
    assert any("d2" in m for m in msgs)


# --- クランプ時の直前値保持(2026-08-03、Part2/PartB-1) ---
# 0803-02実測: V2X速度クランプが対戦車'd2'固有で継続発生(前回比4.7-6.2倍)し、
# fwd_vopp=0への強制が「相手完全停止」誤認識→closing_est過大評価という危険な穴に
# なりうることが判明した。既定(clamp_hold_enabled=False)では現行の0クランプを
# 完全に維持し、有効化時のみ直前値保持→鮮度切れ後は保守的フォールバックへ倒す。


def test_clamp_hold_disabled_by_default_matches_legacy_zero_behavior():
    """既定(clamp_hold_enabled省略=False)は、直前に有効な速度があってもクランプ時は
    従来通り(0,0)になる(回帰なし)。"""
    tracker = V2XVehicleTracker(v_max_safety=6.0, position_jump_threshold=200.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(1.0, [("d2", 4.0, 0.0)]))  # 4.0 m/s、正常
    assert tracker.velocity("d2") == pytest.approx((4.0, 0.0))
    tracker.update(_msg(1.1, [("d2", 100.0, 0.0)]))  # 960 m/s相当、クランプ発生
    assert tracker.velocity("d2") == (0.0, 0.0)


def test_clamp_hold_enabled_returns_last_valid_velocity_within_freshness():
    """有効化時、クランプ直後(鮮度期限内)は直前の有効速度をそのまま返す。"""
    tracker = V2XVehicleTracker(
        v_max_safety=6.0, position_jump_threshold=200.0,
        clamp_hold_enabled=True, clamp_hold_freshness_s=0.5)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(1.0, [("d2", 4.0, 0.0)]))  # 4.0 m/s、正常値として記録
    assert tracker.velocity("d2") == pytest.approx((4.0, 0.0))
    tracker.update(_msg(1.1, [("d2", 100.0, 0.0)]))  # クランプ発生、0.1s後
    assert tracker.velocity("d2") == pytest.approx((4.0, 0.0))  # 直前値を保持


def test_clamp_hold_falls_back_to_conservative_speed_after_freshness_expires():
    """鮮度切れ後は0ではなく、直前の進行方向を維持しつつ大きさをclamp_fallback_mps
    へ差し替える(相手を実際より遅く見積もる=安全側)。"""
    tracker = V2XVehicleTracker(
        v_max_safety=6.0, position_jump_threshold=200.0,
        clamp_hold_enabled=True, clamp_hold_freshness_s=0.5,
        clamp_fallback_mps=5.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(1.0, [("d2", 4.0, 0.0)]))  # 方向+x、4.0 m/s
    tracker.update(_msg(1.1, [("d2", 100.0, 0.0)]))  # クランプ開始(t=1.1)
    tracker.update(_msg(1.8, [("d2", 200.0, 0.0)]))  # 鮮度切れ(0.7s > 0.5s)後もクランプ継続
    vx, vy = tracker.velocity("d2")
    assert vx == pytest.approx(5.0)  # 方向は+xのまま、大きさがfallbackへ
    assert vy == pytest.approx(0.0)


def test_clamp_fallback_defaults_to_v_max_safety_when_unspecified():
    """clamp_fallback_mps未指定時は、保守的な想定値としてv_max_safety自体を使う。"""
    tracker = V2XVehicleTracker(
        v_max_safety=6.0, position_jump_threshold=200.0,
        clamp_hold_enabled=True, clamp_hold_freshness_s=0.1)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(1.0, [("d2", 4.0, 0.0)]))
    tracker.update(_msg(1.05, [("d2", 100.0, 0.0)]))  # クランプ開始
    tracker.update(_msg(1.3, [("d2", 200.0, 0.0)]))  # 鮮度切れ(0.25s > 0.1s)
    vx, _ = tracker.velocity("d2")
    assert vx == pytest.approx(6.0)  # v_max_safetyへフォールバック


def test_clamp_hold_no_prior_valid_velocity_stays_zero():
    """直前の有効速度が一度も無い車両(起動直後からクランプされ続ける)は、
    有効化時でも従来通り(0,0)のまま(フォールバックしようがない、安全側・退行なし)。"""
    tracker = V2XVehicleTracker(
        v_max_safety=6.0, position_jump_threshold=200.0,
        clamp_hold_enabled=True, clamp_hold_freshness_s=0.5)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.05, [("d2", 100.0, 0.0)]))  # 初回からクランプ
    assert tracker.velocity("d2") == (0.0, 0.0)


def test_warn_callback_optional_default_is_silent():
    # Construct without a callback; clamp fires must not raise.
    tracker = V2XVehicleTracker(v_max_safety=30.0, position_jump_threshold=5.0)
    tracker.update(_msg(0.0, [("d2", 0.0, 0.0)]))
    tracker.update(_msg(0.1, [("d2", 100.0, 0.0)]))  # would warn if a callback existed

    assert tracker.velocity("d2") == (0.0, 0.0)  # clamp still fires
