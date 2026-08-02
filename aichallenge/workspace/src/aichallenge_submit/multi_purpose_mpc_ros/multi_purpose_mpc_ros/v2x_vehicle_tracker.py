"""Per-vehicle finite-difference velocity tracker for V2X positions.

This module is intentionally pure Python with no rclpy dependency: it
operates on duck-typed messages whose attributes match
``v2x_msgs/V2XVehiclePositionArray``. That keeps it cheap to unit-test
and reusable from non-ROS contexts (e.g. offline replay of rosbag CSVs).
"""

import math
from collections import deque
from typing import Deque, Dict, List, Tuple


def _stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class V2XVehicleTracker:
    """Tracks the latest two samples per ``vehicle_id`` and exposes
    constant-velocity predictions over a caller-provided time grid."""

    def __init__(self, v_max_safety: float, position_jump_threshold: float,
                 warn_callback=None, speed_window: int = 6,
                 clamp_hold_enabled: bool = False,
                 clamp_hold_freshness_s: float = 0.5,
                 clamp_fallback_mps: float = None):
        self._v_max_safety = float(v_max_safety)
        self._jump_thresh = float(position_jump_threshold)
        self._warn = warn_callback if warn_callback is not None else (lambda _msg: None)
        # 速度平滑窓: 2サンプル差分は 27km/h級のスパイクを出す(v2x ~13Hz)。窓端点差で平滑化。
        self._speed_window = max(2, int(speed_window))
        self._samples: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self._velocities: Dict[str, Tuple[float, float]] = {}
        self._active: List[str] = []
        # 264節続報Task1と対になる防御(2026-08-03、Part2/PartB-1): V2X速度クランプが
        #   継続する対戦車(0803-02実測、d2固有・数千件規模)に対し、下流のfwd_vopp計算
        #   (mpc_controller.py)がクランプ=速度0=「完全停止」と誤認しclosing_est(追い越し
        #   判断)を過大評価する穴への対処。既定は無効(現行の0クランプ挙動を完全維持)。
        #   有効時:
        #   (a) クランプ発生から`clamp_hold_freshness_s`以内は直前の有効速度をそのまま返す。
        #   (b) 鮮度切れ後は0(相手停止=closing過大評価という穴そのもの)には戻さず、直前の
        #       進行方向を維持しつつ大きさだけ`clamp_fallback_mps`(未指定ならv_max_safety
        #       自体、=物理的に想定しうる最大速度という保守的な値)へ差し替える。これにより
        #       closing_estが常に相手を実際より遅く見積もる側(安全マージンを確保する側)へ
        #       倒れる。直前の有効値が一度も無い車両(起動直後等)はフォールバックしようが
        #       ないため、既存どおり0のまま(安全側、退行なし)。
        self._clamp_hold_enabled = bool(clamp_hold_enabled)
        self._clamp_hold_freshness_s = float(clamp_hold_freshness_s)
        self._clamp_fallback_mps = (
            float(clamp_fallback_mps) if clamp_fallback_mps is not None
            else self._v_max_safety)
        self._last_valid_velocity: Dict[str, Tuple[float, float]] = {}
        self._last_valid_time: Dict[str, float] = {}

    def update(self, msg) -> None:
        active: List[str] = []
        for v in msg.vehicles:
            vid = v.vehicle_id
            t = _stamp_to_seconds(v.header.stamp)
            x = float(v.position.x)
            y = float(v.position.y)
            buf = self._samples.setdefault(vid, deque(maxlen=self._speed_window))

            # Detect a position jump against the previous sample (if any).
            jumped = False
            if buf:
                _t_prev, x_prev, y_prev = buf[-1]
                if math.hypot(x - x_prev, y - y_prev) > self._jump_thresh:
                    buf.clear()
                    jumped = True
                    self._warn(
                        f"V2X: position jump for vehicle '{vid}' "
                        f"(>{self._jump_thresh} m) — velocity reset")

            buf.append((t, x, y))

            if jumped or len(buf) < 2:
                self._velocities[vid] = (0.0, 0.0)
            else:
                # 窓の端点(最古〜最新)で差分 → スパイク耐性のある平滑速度
                t0, x0, y0 = buf[0]
                t1, x1, y1 = buf[-1]
                dt = t1 - t0
                if dt > 0.0:
                    vx = (x1 - x0) / dt
                    vy = (y1 - y0) / dt
                    if math.hypot(vx, vy) > self._v_max_safety:
                        self._velocities[vid] = self._clamp_fallback_velocity(vid, t)
                        self._warn(
                            f"V2X: velocity for vehicle '{vid}' exceeds "
                            f"{self._v_max_safety} m/s — clamped to zero")
                    else:
                        self._velocities[vid] = (vx, vy)
                        self._last_valid_velocity[vid] = (vx, vy)
                        self._last_valid_time[vid] = t
                else:
                    self._velocities[vid] = (0.0, 0.0)
            active.append(vid)
        self._active = active

    def _clamp_fallback_velocity(self, vid: str, t: float) -> Tuple[float, float]:
        """クランプ発生時の速度を決める(既定=常に(0,0)、clamp_hold_enabled時のみ
        直前値保持→鮮度切れ後は保守的フォールバックへ切り替える)。"""
        if not self._clamp_hold_enabled or vid not in self._last_valid_time:
            return (0.0, 0.0)
        age = t - self._last_valid_time[vid]
        lvx, lvy = self._last_valid_velocity[vid]
        if age <= self._clamp_hold_freshness_s:
            return (lvx, lvy)
        lspeed = math.hypot(lvx, lvy)
        if lspeed <= 1e-6:
            return (0.0, 0.0)
        scale = self._clamp_fallback_mps / lspeed
        return (lvx * scale, lvy * scale)

    def velocity(self, vehicle_id: str) -> Tuple[float, float]:
        return self._velocities.get(vehicle_id, (0.0, 0.0))

    def is_settled(self, vehicle_id: str) -> bool:
        """速度推定が落ち着いているか(窓が満杯=平滑が効いている)。速度マップの採否に使う。"""
        buf = self._samples.get(vehicle_id)
        return buf is not None and len(buf) >= self._speed_window

    def predict_positions(
        self, vehicle_id: str, t_samples
    ) -> List[Tuple[float, float]]:
        buf = self._samples.get(vehicle_id)
        if not buf:
            return []
        _t_last, x_last, y_last = buf[-1]
        vx, vy = self._velocities.get(vehicle_id, (0.0, 0.0))
        return [(x_last + vx * t, y_last + vy * t) for t in t_samples]

    def active_vehicle_ids(self) -> List[str]:
        return list(self._active)

    def predict_all(self, t_samples) -> Dict[str, List[Tuple[float, float]]]:
        return {vid: self.predict_positions(vid, t_samples) for vid in self._active}


def predictions_to_obstacles(predictions, vehicle_radius: float, obstacle_cls=None):
    """Flatten a ``{vehicle_id: [(x, y), ...]}`` mapping into a list of
    circular obstacles consumable by ``multi_purpose_mpc_ros.core.map``.

    ``obstacle_cls`` is injectable for testability; production callers
    leave it as ``None`` to use ``core.map.Obstacle``. The deferred
    import keeps this module's load time fast and lets the unit tests
    on hosts without ``scikit-image`` exercise the helper with a stub
    dataclass.
    """
    if obstacle_cls is None:
        from multi_purpose_mpc_ros.core.map import Obstacle as obstacle_cls
    out = []
    for _vid, points in predictions.items():
        for x, y in points:
            out.append(obstacle_cls(cx=x, cy=y, radius=vehicle_radius))
    return out


def predictions_to_obstacles_capsule(predictions, vehicle_radius: float,
                                      headings, half_length: float,
                                      obstacle_cls=None):
    """``predictions_to_obstacles`` の全長考慮版(2026-07-20、131-6節②
    「寸法モデルの一元化」対処)。

    背景: 相手車1台につき円1個(半径vehicle_radius、実質半幅ベース)を
    等速外挿した将来位置ごとにスタンプする方式では、(a) 停止/低速車
    (vopp≈0)は将来サンプルが現在位置とほぼ重なり全長方向の広がりが
    一切表現されない、(b) 予測は未来方向のみで、egoが追い越し中に
    最接近する「相手の後端」側は速度に関わらず一度も表現されない、
    という2つの盲点があった(128節footprint_riskが実測で確認した
    「space=3.12mなのにfwd_dlat=0.198m」という矛盾の一因)。

    現在位置(t=0、points[0])のみ、進行方向(headings[vid]、呼び出し元が
    速度ベクトルまたは走行不能時は参照経路接線からフォールバック算出)に
    沿って前後 ``half_length - vehicle_radius`` だけオフセットした2個の円へ
    分割する。half_lengthは既存along_min_length/2(両車半長合計の片側分)を
    再利用し、新規パラメータは導入しない。

    将来サンプル(t>0)は従来通り1個のまま変更しない
    ([[mpc-cost-doubles-with-obstacles]]、過去に障害物点数を2倍化した際に
    MPC発散・22Hzまでの制御レート低下を実測した経緯があるため、点数増加を
    近傍車1台あたり+1個(t=0のみ1→2個)に抑える設計とした)。
    """
    if obstacle_cls is None:
        from multi_purpose_mpc_ros.core.map import Obstacle as obstacle_cls
    offset = max(0.0, half_length - vehicle_radius)
    out = []
    for vid, points in predictions.items():
        if not points:
            continue
        x0, y0 = points[0]
        if offset > 0.0:
            heading = headings.get(vid, 0.0)
            dx = offset * math.cos(heading)
            dy = offset * math.sin(heading)
            out.append(obstacle_cls(cx=x0 + dx, cy=y0 + dy, radius=vehicle_radius))
            out.append(obstacle_cls(cx=x0 - dx, cy=y0 - dy, radius=vehicle_radius))
        else:
            out.append(obstacle_cls(cx=x0, cy=y0, radius=vehicle_radius))
        for x, y in points[1:]:
            out.append(obstacle_cls(cx=x, cy=y, radius=vehicle_radius))
    return out
