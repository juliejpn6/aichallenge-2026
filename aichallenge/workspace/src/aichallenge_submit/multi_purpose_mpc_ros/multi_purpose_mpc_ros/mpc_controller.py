#!/usr/bin/env python3

import yaml
import gc as _gc
import resource as _resource
import time as _time
from collections import deque
from typing import List, Tuple, Optional, NamedTuple, Dict
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import numpy as np
import copy
import os
import shutil
from datetime import datetime

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty, Bool, Float32MultiArray, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D, Point, Vector3, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA

from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import GearCommand, GearReport, SteeringReport
from autoware_auto_planning_msgs.msg import Trajectory
from v2x_msgs.msg import V2XVehiclePositionArray
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    predictions_to_obstacles,
    predictions_to_obstacles_capsule,
)

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.obstacle_manager import ObstacleManager
from multi_purpose_mpc_ros.opponent_speed_map import OpponentSpeedMap
from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor
from multi_purpose_mpc_ros.exexution_stats import ExecutionStats
from multi_purpose_mpc_ros_msgs.msg import AckermannControlBoostCommand, PathConstraints, BorderCells
from multi_purpose_mpc_ros.tools.reference_velocity_configulator import ReferenceVelocityConfigulator


RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
CYAN = ColorRGBA(r=0.0, g=156.0 / 255.0, b=209.0 / 255.0, a=1.0)

def array_to_ackermann_control_command(stamp, u: np.ndarray, acc: float) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    msg.stamp = stamp
    msg.lateral.stamp = stamp
    msg.lateral.steering_tire_angle = u[1]
    msg.lateral.steering_tire_rotation_rate = 2.0
    msg.longitudinal.stamp = stamp
    msg.longitudinal.speed = u[0]
    msg.longitudinal.acceleration = acc
    return msg

def yaw_from_quaternion(q: Quaternion):
    sqx = q.x * q.x
    sqy = q.y * q.y
    sqz = q.z * q.z
    sqw = q.w * q.w

    # Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    sarg = -2 * (q.x*q.z - q.w*q.y) / (sqx + sqy + sqz + sqw) # normalization added from urdfom_headers

    if sarg <= -0.99999:
        yaw = -2. * np.arctan2(q.y, q.x)
    elif sarg >= 0.99999:
        yaw = 2. * np.arctan2(q.y, q.x)
    else:
        yaw = np.arctan2(2. * (q.x*q.y + q.w*q.z), sqw + sqx - sqy - sqz)

    return yaw

def odom_to_pose_2d(odom: Odometry) -> Pose2D:
    pose = Pose2D()
    pose.x = odom.pose.pose.position.x
    pose.y = odom.pose.pose.position.y
    pose.theta = yaw_from_quaternion(odom.pose.pose.orientation)

    return pose

@dataclasses.dataclass
class MPCConfig:
    N: int
    Q: dia_matrix
    R: dia_matrix
    QN: dia_matrix
    v_max: float
    a_min: float
    a_max: float
    ay_max: float
    delta_max: float
    steer_rate_max: float
    control_rate: float
    steering_tire_angle_gain_var: float
    accel_low_pass_gain: float
    steer_low_pass_gain: float
    wp_id_offset: int
    use_max_kappa_pred: bool
    debug_extra_actuator_delay_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class OpponentSituation:
    """2026-07-20追加(141節、フェーズ1: 共有状況スナップショット)。
    ユーザー指摘(「自車位置・コース状況・相手位置・両者の未来位置を推測して
    オーバーテイク/追従/停止を判断しているのだから、どの層も同じ情報に
    アクセスして判断すべき」)を受けた設計。128節(LAT-TTCのspace式が自車位置を
    含まず矛盾した安全判定をしていた)・P0#1(ICCのnear_sepとLAT-TTCの
    dlat_v_emaが同じ相手について食い違っていた)の根本原因はいずれも
    「同じ相手について、層ごとに別々の式・別々のタイミングで再計算していた」
    ことだった。既存の計算結果(_scan_traffic/_lat_ttc.update)を1箇所へ
    集約するのみで新規計算は行わない(既存挙動を変えないフェーズ1のスコープ)。
    第一弾はENGAGEゲートのみこの構造へ移行し、ICC等の他レイヤーは後続ラウンドで
    対応する(段階移行、一度に全て変更しない)。"""
    fwd_vid: Optional[str]
    fwd_ds: Optional[float]
    fwd_dlat: Optional[float]
    fwd_vopp: Optional[float]
    dlat_v_ema: float
    dlat_shrink_run: int
    is_closing_trend: bool


@dataclasses.dataclass(frozen=True)
class EngageEval:
    """2026-07-21追加(148節、ENGAGE判定の純粋スリム化フェーズ1)。
    ユーザー指摘(「相手に追従して停止してしまう問題も、オーバーテイクと
    同様に複数階層で条件が入り組んでいるのでは」)を受け、144節でOVERTAKING側の
    v_safe候補選択(_g2_release_ready/_f3_taper_speed)を抽出した手順と同じ形で、
    cheap_ok(9条件)+_plan_pass+dlat_ttc_vetoの一連の判定を1箇所へ抽出する。
    挙動は_control()から移設前と完全に同一(純粋リファクタリング)。rdy/cls/wc/cdが
    OpponentSituationを共有していない等の棚卸しは次フェーズで別途検討する。"""
    cheap_ok: bool
    ego_ready: bool
    close_enough: bool
    on_path: bool
    plan_ok: bool
    plan_side: int
    can_engage: bool
    closing_est: float
    engage_dist_dynamic: float
    t_reach_profile: Optional[float]
    gate: str


class MPCController(Node):

    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"
    # MAX_LAPS = 6
    MAX_LAPS = 10000
    BUG_VEL = 40.0 # km/h
    BUG_ACC = 400.0

    SHOW_PLOT_ANIMATION = False
    PLOT_RESULTS = False
    ANIMATION_INTERVAL = 20

    KP = 100.0

    def __init__(self, config_path: str, ref_vel_config_path: Optional[str]) -> None:
        super().__init__("mpc_controller") # type: ignore

        # declare parameters
        self.declare_parameter("use_boost_acceleration", False)
        self.declare_parameter("use_obstacle_avoidance", False)
        self.declare_parameter("use_stats", False)

        # get parameters
        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.USE_BUG_ACC = self.get_parameter("use_boost_acceleration").get_parameter_value().bool_value
        self.USE_OBSTACLE_AVOIDANCE = self.get_parameter("use_obstacle_avoidance").get_parameter_value().bool_value
        self.use_stats = self.get_parameter("use_stats").get_parameter_value().bool_value

        self._config_path = config_path
        self._ref_vel_config_path: Optional[str] = ref_vel_config_path
        self._cfg = self._load_config()
        self._odom: Optional[Odometry] = None
        self._gnss_pose: Optional[PoseWithCovarianceStamped] = None  # ピット内自己位置補正用(GNSS実測)。
        # 2026-07-19追加: [LOC-XCHECK]診断(EKFベースe_yとの独立比較)でも同じ購読値を再利用する。
        self._gnss_hist: List[Tuple[float, float]] = []              # GNSS位置履歴(ピット内の進行方位算出用)
        # センシング切り分け計装D(2026-07-19、118節続報): [GNSS-EKF-XCORR]用の
        #   時刻付き位置履歴(上記_gnss_histはピット用途で時刻を持たないため別バッファ)。
        self._xcorr_ekf_hist: List[Tuple[float, float, float]] = []   # (t, x, y) EKF(kinematic_state)
        self._xcorr_gnss_hist: List[Tuple[float, float, float]] = []  # (t, x, y) 生GNSS
        # 2026-07-27追加(192節続報): アクチュエータ遅延特性(純粋遅延/一次遅れ/FOPDT)の
        #   実測診断用。予選環境レコーダー(aichallenge/utils/record_rosbag.bash)への
        #   /vehicle/status/steering_status追加はパブリックリポジトリ側の変更でpush権限が
        #   無いため反映できないと判明(deployment gapの詳細はdesign_docs 196節)。
        #   自分たちの提出物内(mpc_controller.py)からGNSS-EKF-XCORRと同じ相互相関パターンで
        #   直接ロギングすることで、bagレコーダーを経由せず既存のautoware.logテキストログへ
        #   実測遅延を記録する。
        self._xcorr_steercmd_hist: List[Tuple[float, float]] = []   # (t, 実発行操舵角[rad]、gain適用後)
        self._xcorr_steeract_hist: List[Tuple[float, float]] = []   # (t, 実測操舵角[rad]、steering_status)
        # デバッグ専用(196節続報): debug_extra_actuator_delay_s>0の時のみ使う発行待ちキュー。
        #   既定0.0では_publish_control_command内のifへ一切入らず常に空のまま(挙動不変)。
        self._delayed_cmd_queue: List[Tuple[float, object, object]] = []  # (due_t, raw_cmd_msg, gained_cmd_msg)
        # 2026-07-27追加(208節続報、AXIS06過操舵ホットスポット監視): 過渡応答リンギングが
        #   顕著だったwaypoint(ローカル実測・ユーザー目視で特定)通過時、実測舵角のピークと
        #   パス自体が要求する理論舵角(kappa_ref由来)との乖離を記録する。予選環境で同じ
        #   地点が同様に現れるか比較するための診断のみで、制御には一切影響しない。
        self._HOTSPOT_WPS = (178, 189, 258, 289, 334)
        self._hotspot_monitor: Optional[Dict[str, float]] = None  # {'wp':, 'end_t':, 'theo_deg':, 'peak_dev_deg':}
        self._enable_control = True
        self._initialize()
        self._setup_parameters_callback()
        self._setup_pub_sub()

        if self.use_sim_time:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("use_sim_time is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_BUG_ACC:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_BUG_ACC is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_OBSTACLE_AVOIDANCE:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_OBSTACLE_AVOIDANCE is enabled!")
            self.get_logger().warn("------------------------------------")

    def _load_config(self) -> NamedTuple:

        # logging content
        with open(self._config_path, "r") as f:
            config_content = f.read()
            self.get_logger().info(
                "\n" +
                "----- config.yaml -----\n"+
                config_content + "\n" +
                "-----------------------")

        if self._ref_vel_config_path is not None:
            with open(self._ref_vel_config_path, "r") as f:
                ref_vel_config_content = f.read()
                self.get_logger().info(
                    "\n" +
                    "----- ref_vel.yaml -----\n"+
                    ref_vel_config_content + "\n" +
                    "-----------------------")

        with open(self._config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f)) # type: ignore

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path] # type: ignore
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _create_reference_path_from_autoware_trajectory(self, trajectory: Trajectory) -> Optional[ReferencePath]:
        wp_x = [0] * len(trajectory.points)
        wp_y = [0] * len(trajectory.points)
        for i, p in enumerate(trajectory.points):
            wp_x[i] = p.pose.position.x
            wp_y[i] = p.pose.position.y

        cfg_ref_path = self._cfg.reference_path # type: ignore
        reference_path = ReferencePath(
            self._map,
            wp_x,
            wp_y,
            cfg_ref_path.resolution,
            cfg_ref_path.smoothing_distance,
            cfg_ref_path.max_width,
            cfg_ref_path.circular)

        mpc_config = self._mpc_cfg
        speed_profile_constraints = {
            "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
            "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}

        if not reference_path.compute_speed_profile(speed_profile_constraints):
            return None

        return reference_path

    def _setup_parameters_callback(self) -> None:
        def declatre_parameters():
            cfg_mpc = self._cfg.mpc
            self.declare_parameter("v_max", cfg_mpc.v_max)
            self.declare_parameter("steering_tire_angle_gain_var", cfg_mpc.steering_tire_angle_gain_var)
            self.declare_parameter("Q0", cfg_mpc.Q[0])
            self.declare_parameter("Q1", cfg_mpc.Q[1])
            self.declare_parameter("Q2", cfg_mpc.Q[2])
            self.declare_parameter("R0", cfg_mpc.R[0])
            self.declare_parameter("R1", cfg_mpc.R[1])
            self.declare_parameter("QN0", cfg_mpc.QN[0])
            self.declare_parameter("QN1", cfg_mpc.QN[1])
            self.declare_parameter("QN2", cfg_mpc.QN[2])

            mpc_cfg = self._mpc_cfg
            self.declare_parameter("ay_max", mpc_cfg.ay_max)
            self.declare_parameter("accel_low_pass_gain", mpc_cfg.accel_low_pass_gain)
            self.declare_parameter("steer_low_pass_gain", mpc_cfg.steer_low_pass_gain)
            self.declare_parameter("wp_id_offset", mpc_cfg.wp_id_offset)
            # 対策①: レースライン追従ブレンド重み（既存 lateral_blend/lateral_target を再利用）
            self._line_follow_w = float(getattr(cfg_mpc, "line_follow_w", 0.0))
            self.declare_parameter("line_follow_w", self._line_follow_w)
            # デバッグ専用(196節続報): 予選環境との遅延差再現実験。既定0.0=無効。
            self.declare_parameter("debug_extra_actuator_delay_s", mpc_cfg.debug_extra_actuator_delay_s)

        def param_cb(parameters):
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_cfg = self._mpc_cfg

            def update_Q(index: int, value: float):
                cfg_mpc.Q[index] = value
                mpc_cfg.Q = sparse.diags(cfg_mpc.Q)
                self._mpc.update_Q(mpc_cfg.Q)
                self.get_logger().warn(f"Q[{index}] was updated to '{value}'")

            def update_R(index: int, value: float):
                cfg_mpc.R[index] = value
                mpc_cfg.R = sparse.diags(cfg_mpc.R)
                self._mpc.update_R(mpc_cfg.R)
                self.get_logger().warn(f"R[{index}] was updated to '{value}'")

            def update_QN(index: int, value: float):
                cfg_mpc.QN[index] = value
                mpc_cfg.QN = sparse.diags(cfg_mpc.QN)
                self._mpc.update_QN(mpc_cfg.QN)
                self.get_logger().warn(f"QN[{index}] was updated to '{value}'")

            for param in parameters:
                if param.name == "v_max" and param.type_ == Parameter.Type.DOUBLE:
                    # 単位バグ修正(2026-07-04): mpc_cfg.v_max は m/s フィールド。km/h生値を
                    #   入れると毎周期の min(区間速度[m/s], v_max) が壊れ、グローバル上限が
                    #   実質無効化(区間値35km/hが上限になる)していた。
                    mpc_cfg.v_max = kmh_to_m_per_sec(param.value)
                    self._mpc.update_v_max(kmh_to_m_per_sec(param.value))
                    # 旧: 全wp一律のフラット充填 → 曲率減速が消えるため廃止。初期化と同じ
                    #   制約でプロファイル再計算(ref_vel併用時は毎周期のmin側で反映される)。
                    self._reference_path.compute_speed_profile({
                        "a_min": mpc_cfg.a_min, "a_max": mpc_cfg.a_max,
                        "v_min": 0.0, "v_max": mpc_cfg.v_max, "ay_max": mpc_cfg.ay_max})
                    # 追い越しworth判定が使う ego 潜在速度 _v_pot も更新する。
                    #   未更新だと worth=(_v_pot − vopp) が __init__時の旧v_maxで凍結し、
                    #   v_maxを上げても追い越しに入れないバグになる(2026-07-04 修正)。
                    if hasattr(self, "_v_pot"):
                        self._v_pot = kmh_to_m_per_sec(param.value)

                    self.get_logger().warn(f"v_max was updated to '{param.value}' [km/h]")

                elif param.name == "steering_tire_angle_gain_var" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steering_tire_angle_gain_var = param.value
                    self.get_logger().warn(f"steering_tire_angle_gain_var was updated to '{param.value}'")

                elif param.name == "Q0" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(0, param.value)
                elif param.name == "Q1" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(1, param.value)
                elif param.name == "Q2" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(2, param.value)


                elif param.name == "R0" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(0, param.value)
                elif param.name == "R1" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(1, param.value)

                elif param.name == "QN0" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(0, param.value)
                elif param.name == "QN1" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(1, param.value)
                elif param.name == "QN2" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(2, param.value)

                elif param.name == "ay_max" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.ay_max = param.value
                    self._mpc.update_ay_max(param.value)
                    self.get_logger().warn(f"ay_max was updated to '{param.value}'")

                elif param.name == "accel_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.accel_low_pass_gain = param.value
                    self.get_logger().warn(f"accel_low_pass_gain was updated to '{param.value}'")

                elif param.name == "steer_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steer_low_pass_gain = param.value
                    self.get_logger().warn(f"steer_low_pass_gain was updated to '{param.value}'")

                elif param.name == "wp_id_offset" and param.type_ == Parameter.Type.INTEGER:
                    mpc_cfg.wp_id_offset = param.value
                    self._mpc.update_wp_id_offset(param.value)
                    self.get_logger().warn(f"wp_id_offset was updated to '{param.value}'")

                elif param.name == "line_follow_w" and param.type_ == Parameter.Type.DOUBLE:
                    self._line_follow_w = float(np.clip(param.value, 0.0, 1.0))
                    self.get_logger().warn(f"line_follow_w was updated to '{self._line_follow_w}'")

                elif param.name == "debug_extra_actuator_delay_s" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.debug_extra_actuator_delay_s = float(max(0.0, param.value))
                    self.get_logger().warn(
                        f"debug_extra_actuator_delay_s was updated to '{mpc_cfg.debug_extra_actuator_delay_s}'")


            return SetParametersResult(successful=True)

        declatre_parameters()
        self.add_on_set_parameters_callback(param_cb)

    def _initialize(self) -> None:
        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path # type: ignore

            is_ref_path_given = cfg_ref_path.csv_path != "" # type: ignore
            if is_ref_path_given:
                print("Using given reference path")
                wp_x, wp_y, _, _ = load_ref_path(self.in_pkg_share(self._cfg.reference_path.csv_path)) # type: ignore
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

            else:
                print("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore

                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)


        def create_obstacles() -> List[Obstacle]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                self._obstacle_manager = ObstacleManager(self._map, obstacles)
                return obstacles
            else:
                return []

        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model # type: ignore
            return BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                1.0 / self._cfg.mpc.control_rate) # type: ignore

        def create_mpc(car: BicycleModel) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore

            mpc_cfg = MPCConfig(
                cfg_mpc.N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(cfg_mpc.R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(self.BUG_VEL if self.USE_BUG_ACC else cfg_mpc.v_max),
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                np.deg2rad(cfg_mpc.delta_max_deg),
                cfg_mpc.steer_rate_max,
                cfg_mpc.control_rate,
                cfg_mpc.steering_tire_angle_gain_var,
                cfg_mpc.accel_low_pass_gain,
                cfg_mpc.steer_low_pass_gain,
                cfg_mpc.wp_id_offset,
                cfg_mpc.use_max_kappa_pred,
                float(getattr(cfg_mpc, "debug_extra_actuator_delay_s", 0.0)))

            state_constraints = {
                "xmin": np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                "xmax": np.array([np.inf, np.inf, np.inf, np.inf])}
            input_constraints = {
                "umin": np.array([0.0, -np.tan(mpc_cfg.delta_max) / car.length]),
                "umax": np.array([mpc_cfg.v_max, np.tan(mpc_cfg.delta_max) / car.length])}

            # mpcからのsteer指令出力は、gainを掛けて出力され、その状態で車体のsteer rate limit が適用されるため、
            # mpcの制御計算におけるsteer_rate_maxは、実際のsteer_rate_maxをgainで除した値で設定する
            scaled_steer_rate_max = mpc_cfg.steer_rate_max / mpc_cfg.steering_tire_angle_gain_var

            mpc = MPC(
                car,
                mpc_cfg.N,
                mpc_cfg.Q,
                mpc_cfg.R,
                mpc_cfg.QN,
                state_constraints,
                input_constraints,
                mpc_cfg.ay_max,
                scaled_steer_rate_max,
                mpc_cfg.wp_id_offset,
                self.USE_OBSTACLE_AVOIDANCE,
                self._cfg.reference_path.use_path_constraints_topic,
                mpc_cfg.use_max_kappa_pred,
                r_drate=float(getattr(cfg_mpc, "r_drate", 0.0)),
                use_osqp_update=bool(getattr(cfg_mpc, "use_osqp_update", True)),
                shadow_cycles=int(getattr(cfg_mpc, "osqp_shadow_cycles", 50)))

            return mpc_cfg, mpc

        def compute_speed_profile(car: BicycleModel, mpc_config: MPCConfig) -> None:
            speed_profile_constraints = {
                "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
                "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}
            car.reference_path.compute_speed_profile(speed_profile_constraints)

        def create_ref_vel_configulator() -> Optional[ReferenceVelocityConfigulator]:
            if self._ref_vel_config_path is None:
                return None
            return ReferenceVelocityConfigulator(self, self._config_path, self._ref_vel_config_path)

        # 2026-07-26追加(190-3節): コリドー境界ラチェットの拡大方向レートリミット。
        #   __init__の他の初期化より前にここで確定させ、以降の_set_active_path()での
        #   経路差し替え時にも同じ値を適用する(190-3節のテストがソース走査で確認)。
        self._corridor_widen_step_m = (
            float(getattr(self._cfg.v2x_obstacle_avoidance, "corridor_widen_rate_mps", 1.0))  # type: ignore
            / float(self._cfg.mpc.control_rate))  # type: ignore
        self._map = create_map()
        self._reference_path = create_ref_path(self._map)
        self._reference_path.corridor_widen_step_m = self._corridor_widen_step_m
        self._car = create_car(self._reference_path)
        self._mpc_cfg, self._mpc = create_mpc(self._car)
        # 曲率エンベロープ(2026-07-04): ヘアピン(s≈72 R=6m / s≈169 R=8m)で毎周壁接触した対策。
        #   ref_vel適用(毎周期)が v_ref を区間スカラでフラット上書きするため曲率減速が皆無だった。
        #   ay_profile は「現状の操舵系(遅れ200ms+チャタ)で追従できる横G予算」。グリップ限界
        #   (実測~20m/s²)ではない。チャタ/レート修正後に引き上げ余地あり。
        #   既存 compute_speed_profile を v_max=∞ 相当で流用し、per-wp包絡線として保存
        #   → 毎周期のref_velブロックで要素毎minに使う(直線・緩コーナーは不変、ヘアピンのみ減速)。
        _ay_prof = float(getattr(self._cfg.mpc, "ay_profile", 3.0))  # type: ignore
        self._ay_profile = _ay_prof   # オフセットライン減速(_offset_line_speed_cap)でも使用
        self._reference_path.compute_speed_profile({
            "a_min": self._mpc_cfg.a_min, "a_max": self._mpc_cfg.a_max,
            "v_min": 0.0, "v_max": 99.0, "ay_max": _ay_prof})
        self._v_envelope = np.array(
            [float(wp.v_ref) for wp in self._reference_path.waypoints])
        compute_speed_profile(self._car, self._mpc_cfg)  # 通常プロファイルへ復元

        # 176節続報(曲率スイング検知→R[delta]動的引き上げ、2026-07-24): 176節で「蛇行の
        #   大きさは曲率の絶対値(r=0.346)より、前後17m窓内での曲率の変動幅=swing(r=0.807)
        #   と強く相関する」と判明した(=急コーナーではなくシケイン=短距離での符号反転区間
        #   が揺れの主因)。局所化層(EKF/GNSS)のノイズは全区間で一定と確認済みで無罪。
        #   Q[e_y]ではなくR[delta](舵角変化ペナルティ)を触るのは、v1-v5のQ[e_y]曲率
        #   スケジュールと同じ「目標を動かす」失敗パターンを避け、「動き自体の滑らかさ」
        #   に直接効くパラメータで対処するため。OT/pit状態とは独立(シケインの地形はどの
        #   状態でも変わらないため、状態分岐の外・毎周期無条件で計算する)。
        #   R[delta]の「常時・一律」な引き上げは500→1000実験(2026-07-23)で既に撤回済み
        #   (直線・wp78・wp257いずれも悪化)。今回はシケイン区間のみへの限定的な引き上げ
        #   である点が異なる。計算本体・量子化ゲート・検証ログはmain loop側(get_control
        #   呼び出し直前、状態分岐の外)を参照。
        self._r_delta_swing_boost = float(getattr(self._cfg.mpc, "r_delta_swing_boost", 400.0))  # type: ignore
        self._r_delta_swing_kappa_lo = float(getattr(self._cfg.mpc, "r_delta_swing_kappa_lo", 0.12))  # type: ignore
        self._r_delta_swing_kappa_hi = float(getattr(self._cfg.mpc, "r_delta_swing_kappa_hi", 0.30))  # type: ignore
        self._r_delta_swing_lookahead_wp = int(getattr(self._cfg.mpc, "r_delta_swing_lookahead_wp", 16))  # type: ignore
        self._r_delta_swing_ema_beta = float(getattr(self._cfg.mpc, "r_delta_swing_ema_beta", 0.15))  # type: ignore
        self._r_delta_swing_ema = None  # 現在のEMA値(初回呼び出しで生値から初期化)
        self._r_delta_applied_value = None  # 直近update_Rに実際に反映したR[delta]値(未適用ならNone)
        self._r_delta_swing_update_count = 0  # [R-DELTA-SWING]検証: 実際にupdate_Rを呼んだ回数
        self._r_delta_swing_dbg_loop = 0  # [R-DELTA-SWING]ログの1Hzスロットル用カウンタ

        # 2026-07-26追加(徹底解析: アクチュエータ遅延の速度依存性): 実測(2026-07-03、
        #   r=0.992)確認済みのAWSIM操舵アクチュエータ遅延(≈200ms)は速度によらない
        #   固定の"時間"だが、既存wp_id_offset(内巻き対策、config.yaml参照)は固定の
        #   "距離"(1点≈1m)であるため、速度が上がるほど実効補償時間(距離÷速度)が
        #   目減りする(15km/h≈240ms>200ms→適正、20km/h≈180ms<200ms→不足)。
        #   186/187節の速度依存の蛇行悪化(15→20km/hでstd約1.9倍)と整合する仮説。
        #   既存wp_id_offset(inside-cut用に別途検証済みの固定値)自体は変更せず、
        #   現在速度で必要な分がそれを上回る周期のみ底上げする(max()で下限維持、
        #   新規の安全弁緩和ではなく追加候補のみ)。
        self._delay_t_delay_s = float(getattr(self._cfg.mpc, "delay_t_delay_s", 0.2))  # type: ignore
        self._wp_id_offset_base = int(self._cfg.mpc.wp_id_offset)  # inside-cut用の既存値(下限)
        self._wp_id_offset_applied = self._wp_id_offset_base  # 直近update_wp_id_offsetへ反映済みの値

        # 発進地点(車庫/ピット)判別とピット経路（gate1/2/3/eval 共通）
        self._race_ref_path = self._reference_path   # レースライン(周回)
        self._pit_ref_path = None
        self._on_pit = False
        self._pit_enable = False
        _pit = getattr(self._cfg, "pit_lane", None)
        if _pit is not None and bool(getattr(_pit, "enable", False)):
            try:
                # 生のピット経路点(densify済)を保持。実際の ReferencePath は run() で
                # 発進地点近傍から構築する。
                self._pit_raw_x, self._pit_raw_y = self._load_densified_pit_points()
                self._pit_enter_dist = float(getattr(_pit, "enter_dist", 5.0))
                self._pit_course_in_dist = float(getattr(_pit, "course_in_dist", 3.0))
                self._pit_course_in_count = int(getattr(_pit, "course_in_count", 5))
                self._pit_v_max = kmh_to_m_per_sec(float(getattr(_pit, "v_max_pit_kmh", 5.0)))
                self._pit_q_ey = float(getattr(_pit, "q_ey_pit", 5000000.0))
                self._pit_safety_margin = float(getattr(_pit, "safety_margin_pit", 0.9))
                self._pit_heading_track_dist = float(getattr(_pit, "heading_track_dist", 0.4))
                self._pit_course_in_acc = 0
                # lanelet2 ピット境界(壁の真値)を読み、コリドー(wp.ub/lb)に使う。
                # 失敗しても従来(占有格子)動作にフォールバック。
                self._pit_bound_A = None
                self._pit_bound_B = None
                try:
                    self._pit_bound_A, self._pit_bound_B = self._load_pit_lane_bounds()
                except Exception as e:
                    self.get_logger().warn(f"pit lanelet bounds load failed (fallback to occupancy): {e}")
                # レースライン距離判定用キャッシュ（経路を切り替えても保持）
                self._race_xy = np.asarray(
                    [(wp.x, wp.y) for wp in self._race_ref_path.waypoints], dtype=np.float64)
                self._pit_enable = True
                self.get_logger().info("pit_lane enabled (start-point auto detection)")
            except Exception as e:
                self.get_logger().warn(f"pit_lane init failed, disabled: {e}")
                self._pit_enable = False

        self._ref_vel_configulator: Optional[ReferenceVelocityConfigulator] = create_ref_vel_configulator()

        self._trajectory: Optional[Trajectory] = None
        self._path_constraints = None

        # Obstacles
        if self.USE_OBSTACLE_AVOIDANCE:
            self._static_obstacles: List[Obstacle] = create_obstacles()
            self._dynamic_obstacles: List[Obstacle] = []
            self._obstacles_updated = bool(self._static_obstacles)
            v2x_cfg = self._cfg.v2x_obstacle_avoidance  # type: ignore
            self._v2x_tracker = V2XVehicleTracker(
                v_max_safety=float(v2x_cfg.v_max_safety),
                position_jump_threshold=float(v2x_cfg.position_jump_threshold),
                warn_callback=self.get_logger().warn,
                speed_window=int(getattr(v2x_cfg, "speed_window", 6)),
            )
            self._v2x_vehicle_radius = float(v2x_cfg.vehicle_radius)
            # 2026-07-14追加(ユーザー指摘: 「壁の向こう側にいる相手」誤認識対策):
            #   _closest_wp_and_sは全waypointからの単純な(x,y)最近傍探索のため、
            #   ヘアピン等でコースが自分自身に壁一枚を挟んで近接する箇所では、
            #   弧長的に無関係な(=壁の反対側の)相手車をこちら側のwaypointへ
            #   誤ってマッチさせうる。既存position_jump_threshold(V2X生値の
            #   異常ジャンプ検知に使う「1周期あたりの物理的に妥当な最大移動量」)
            #   をそのまま探索半径として再利用し、前回マッチしたwp_id近傍だけを
            #   探索するようにする(新規パラメータ0個)。
            self._wp_match_radius_m = float(v2x_cfg.position_jump_threshold)
            self._wp_match_prev: dict = {}   # vid -> 前回マッチしたwp_id(他車、call跨ぎで共有)
            mpc_N = int(self._cfg.mpc.N)  # type: ignore
            t_horizon = mpc_N / float(self._cfg.mpc.control_rate)  # type: ignore
            # B1: 他車予測軌道の円の数。旧実装は mpc_N(=20)点固定で obs=台数×20(過剰・重複)。
            #   n_pred_samples で間引き、同じ 0.5s horizon を少数点で被覆して制御レートを回復する。
            n_samp = int(getattr(v2x_cfg, "n_pred_samples", mpc_N))
            n_samp = max(2, min(n_samp, mpc_N))
            self._v2x_t_samples = [
                k * t_horizon / max(n_samp - 1, 1) for k in range(n_samp)
            ]
            # H1: 後方車のマップ除外(ヒステリシス)。enter/exitは後方距離[m](正値)。
            self._rear_map_enter = float(getattr(v2x_cfg, "rear_map_enter", 0.5))
            self._rear_map_exit = float(getattr(v2x_cfg, "rear_map_exit", 1.5))
            self._fwd_map_enter = float(getattr(v2x_cfg, "fwd_map_enter", 12.0))
            self._fwd_map_exit = float(getattr(v2x_cfg, "fwd_map_exit", 13.5))
            self._map_included = {}   # vid → 前回マップ投入したか(境界フラッピング防止)
            # 2026-07-20追加(131-6節②、寸法モデルの一元化): [CAPSULE-HEADING]
            #   エッジトリガーログ用。vid → 前回の進行方向ソース(velocity/track_tangent)。
            self._capsule_heading_src = {}
            # コリドー外の V2X 障害物で MPC のコリドー狭窄/反転が起きないよう、
            # ref-path 近傍のみに絞り込む。閾値 = max_width/2 + vehicle_radius + 余白。
            ref_max_width = float(self._cfg.reference_path.max_width)  # type: ignore
            self._v2x_corridor_threshold_sq = (
                ref_max_width / 2.0 + self._v2x_vehicle_radius + 0.5
            ) ** 2
            wps = self._reference_path.waypoints
            self._waypoint_xy = np.asarray(
                [(wp.x, wp.y) for wp in wps], dtype=np.float64)
            # --- forward obstacle longitudinal ACC(ICC相当) params (arc-length based) ---
            self._fwd_a_brake = 1.3     # 安全減速度 [m/s^2]（Fix-3: 1.0→1.3=指令上限1.37に整合。想定が弱く
                                        #   相手のコーナー強減速に追突した(0703_02で4件)。20m検知で~23km/hまで停止可）
            self._fwd_margin_center = 4.0      # 追従車間(中心間)[m]（Fix-3: 3.0→4.0。相手の強ブレーキ+速度検知遅れ0.46sの吸収代）
            self._fwd_lateral_halfwidth = 1.5  # 進路帯の半幅 [m]（遠方のみ使用。近距離はF1の実横間隔判定）
            self._fwd_max_consider = 20.0      # この弧長距離より遠い前方他車は無視 [m]（40→20。停車車も~21km/hまで対応）
            # F1(2026-07-03 接触4件の対策): 近距離は「参照帯」でなく「自車との実横間隔」で減速対象を判定。
            #   手動/ライン外の低速車が帯(±1.5m)から外れ ICC が沈黙→全速追突した(d1.0でv_safe=None u4.17)。
            #   近距離では横位置に関係なく、実際にぶつかる横関係(実横間隔<カート幅1.45+余裕)なら減速対象とする。
            self._fwd_near_range = 6.0         # [m] この弧長距離以内は実横間隔で判定
            self._fwd_min_lat_sep = 1.8        # [m] 実横間隔がこれ未満なら減速対象(=横をすり抜けない)
            self._fwd_clear_count = 0          # 前方クリア連続カウンタ(NORMAL復帰判定)
            # 攻めの追いつき判定(C1)は「自分が出せる速度」基準(現在速度だと追従減速でclosing=0になり永久に抜けない)
            self._v_pot = float(self._cfg.mpc.v_max) / 3.6  # type: ignore  # [m/s]
            # --- end forward obstacle params ---

            # --- gate2 overtake params (方式A: 回避ONのまま空き側を自動選択) ---
            # config.yaml の overtake: セクションから読む（無ければ安全側デフォルト）。
            ot = getattr(self._cfg, "overtake", None)
            def _otget(name, default):
                try:
                    return getattr(ot, name) if ot is not None else default
                except AttributeError:
                    return default
            self._ot_enable = bool(_otget("enable", True))
            self._ot_min_gap = float(_otget("min_gap", 2.5))        # [m] 片側の空き幅しきい値
            self._ot_block_half = float(_otget("block_half", 0.9))  # [m] 他車1台の横半幅
            self._ot_max_consider = float(_otget("max_consider", 40.0))  # [m] 前方弧長の上限
            self._ot_v_cap = float(_otget("v_cap", 6.0))            # [m/s] 追い越し中の速度上限
            self._ot_exit_clear = int(_otget("exit_clear", 3))
            self._ot_debug = bool(_otget("debug", False))
            # --- 修正3: ヒステリシス＋安全フォールバック ---
            self._ot_gap_hys = float(_otget("gap_hys", 0.5))        # [m] OVERTAKING維持の緩和量(min_gap-hys で離脱)
            # --- Stage2 攻め: 相手速度に応じた判断（同等/速い相手を追わない=蛇行防止）---
            self._opp_obstacle_speed = float(_otget("opp_obstacle_speed", 6.0)) / 3.6  # [m/s] 未満=障害物として抜く
            self._opp_min_closing = float(_otget("opp_min_closing", 0.7))   # [m/s] エンゲージ閾値(ヒステリシス上側)
            # Fix-1: OVERTAKINGコミット(入りにくく・出にくく)
            self._opp_giveup_closing = float(_otget("opp_giveup_closing", 0.2))  # [m/s] これ未満連続で断念
            self._ot_giveup_cycles = int(_otget("giveup_cycles", 40))       # [周期] 断念に要する連続数(≈1s)
            self._ot_engage_debounce = int(_otget("engage_debounce", 8))    # [周期] エンゲージに要するworth連続数(≈0.2s)
            self._ot_engage_lat_max = float(_otget("engage_lat_max", 2.0))  # [m] H2: 進路上(|lat|≤)の相手のみエンゲージ
            self._engage_ego_margin = float(_otget("engage_ego_margin", 0.3)) # [m/s] 段階1: 自分が相手より遅すぎない事
            self._ot_engage_max_dist = float(_otget("engage_max_dist", 6.0))   # [m] 近接ゲート: 前車がこの距離以内でのみエンゲージ
            # 2026-07-10簡素化: 瞬時値+周期カウンタの二重平滑化(side_block_cycles)を廃止し、
            #   EMA(指数移動平均)一本化。時定数は約1秒(N=40@40Hz→alpha=2/41)。当時はgiveup判定
            #   自体にもこのEMAを使っていたが、2026-07-12にLAT-TTCのC2分岐へ一本化し
            #   giveup用のEMA(_ot_side_block_ema)は2026-07-17に削除した。alongレーン幅の
            #   平滑化(_along_lane_ema)は今も同じ時定数を使うため、本パラメータ自体は残す。
            self._ot_ema_alpha = float(_otget("ema_alpha", 0.05))  # EMA時定数(≈1秒@40Hz)
            self._ot_engage_cooldown_cycles = int(_otget("engage_cooldown", 80))  # [周期] 失敗離脱後の再エンゲージ抑制
            self._ot_engage_cooldown = 0
            # 2026-07-21追加(148節②、実測に基づく再設計): 0721-01以降のローカルログ実測
            # (giveup後のfwd_dlat推移をwaypoint単位で追跡)で、footprint_risk起因のgiveup
            # 8件中3件は間隔がわずか1.3〜5.0秒で回復していたにもかかわらず、固定8秒
            # cooldown(139節)によりその後3〜7秒を無駄に待っていたことが判明した。逆に
            # 残り5件は8秒経過時点でもまだ間隔が狭いままで、固定8秒が実際に必要だった。
            # 「固定秒数」ではなく「footprint_risk条件自体が実際に解消したか」で解除する
            # ようにすれば、早く回復すれば早く再開でき、まだ危険なら8秒を超えても正しく
            # 待ち続けられる。footprint_risk起因のgiveupの場合のみ、既存engage_cooldownの
            # 固定タイマーの代わりにこの解除方式を使う(相手が速すぎる等の他のgiveup理由は
            # 従来通りの固定cooldownのまま、139節の元の設計意図を維持)。
            self._ot_footprint_risk_gated = False    # 今回のcooldownがfootprint_risk起因か
            self._ot_footprint_risk_clear_count = 0  # footprint_risk条件が連続で不成立の周期数
            self._ot_fp_clear_logged = False  # [FP-COOLDOWN-CLEAR]の多重ログ防止(エッジ検知用)
            self._ot_t_lateral = float(_otget("t_lateral", 3.0))                  # [s] 横移動フェーズ時間(engage→cl実測3s。この間closing≈0)
            self._ot_pass_block_kappa = float(_otget("pass_block_kappa", 0.10))   # [1/m] 内側可否判定のきついコーナー閾値(R≤10m)
            self._ot_pass_clear = float(_otget("pass_clear", 3.0))                # [m] 「抜き切り」= 相手の前にこれだけ出る
            self._ot_pass_t_max = float(_otget("pass_t_max", 8.0))                # [s] パス所要時間の予算(超過=追従の方が速い)
            self._def_enter_cycles = int(_otget("def_enter_cycles", 5))           # [周期] 被追い越しON確定(≈0.12s連続)
            self._def_exit_cycles = int(_otget("def_exit_cycles", 15))            # [周期] 被追い越しOFF確定(≈0.38s連続)
            self._def_on_count = 0
            self._def_off_count = 0
            self._def_active = False
            self._unlock_after = int(_otget("unlock_inf_cycles", 80))             # [周期] H4-lite発動: inf連続がこれ以上かつ停止
            self._unlock_hold = int(_otget("unlock_hold_cycles", 60))             # [周期] H4-lite保持時間(≈1.5s)
            self._unlock_left = 0
            self._ot_giveup_count = 0
            self._ot_worth_count = 0
            # 2026-07-24追加(168節、wp161スタック再発対策): _corr_bound_ahead(委託側)が
            #   非正転落(=先読み内に正の隙間が皆無、物理的に不可能)した連続周期を数え、
            #   既存のgiveup合流点(_side_blocked)へフィードバックするための状態。
            #   従来はlat_ttc系のforce_giveup(TTC/closing/footprint_risk)のみが
            #   _side_blockedを駆動し、オフセット目標を実際にクランプするcorr_bound_ahead()
            #   の非正転落は一切フィードバックされていなかった(_ot_side="継続中"のまま
            #   実際の指令=lateral_targetだけ0(直進)へ収束する非矛盾性違反、0724-01
            #   実測wp160-163で確認)。
            self._ot_room_exhausted_count = 0
            self._ot_room_exhausted_prev_side = 0
            # 非正転落から実際にgiveupが合流するまでの間、オフセット目標をmax(0,...)で
            #   即座に0(直進)へ落とさず、直近の有効(正マージン)時の値を凍結保持するための値。
            self._ot_last_valid_target_mag = None
            # 2026-07-17追加(94節、トークン整合性監査): scan_traffic の fwd_vid は
            #   毎周期「その時点で最も近い車」を選び直す実装であり、ロック中の対象車に
            #   固定されない。93節で修正したLAT-TTCのcritical_curvature_runと同じ理由で、
            #   このカウンタも対象車IDが変わった周期は仕切り直す(_room_debounce_okと
            #   同一の考え方)。
            self._ot_worth_prev_vid = None
            self._ot_giveup_prev_vid = None
            # Fix-2: ICC解放ヒステリシス(パス中のクリア判定)
            self._clear_lat_release = float(_otget("clear_lat_release", 2.1))    # [m] 解放閾値
            self._clear_ds_beside = float(_otget("clear_ds_beside", 1.0))        # [m] 真横判定
            self._clear_lat_reacquire = float(_otget("clear_lat_reacquire", 1.6)) # [m] 再取得閾値
            self._ot_cleared = False   # パス対象を横にクリアしたか(エピソード内ラッチ)
            # 2026-07-14追加(0714-03実測、事象③追補): 再取得(cleared→False)デバウンス用。
            # コーナー形状によるdlatの一時的な振動(0.2秒未満の単発ディップ)で
            # clearedが不必要に解除され、C1/C2の厳しい閾値へ再度晒される
            # 「engage→ほぼクリア→再取得→giveup」の反復ループを防ぐ(既存engage_debounceを
            # 再利用、新規パラメータなし)。
            self._ot_reacquire_count = 0
            self._g2_release_prev = False  # 2026-07-12追加: G-2解放の遷移検知用(ログ間引き)
            # 2026-07-18追加(104節): _side_clearの非対称デバウンス用状態
            #   (ON方向のみ連続確認、OFF方向は即時反映)。
            self._g2_clear_on_count = 0
            self._g2_release_debounced = False
            self._v_safe_src_prev = None  # 2026-07-17追加(91節): v_safe_src遷移検知用(ログ間引き)
            # 2026-07-18追加(100節、Tier1裁定の外出し): 旧C1_obstacle_yield分岐
            #   (lateral_ttc_monitor.py内、92節続報)をここへ移設した際の遷移検知用。
            self._lat_ttc_c1_yield_prev = False
            self._ot_offset_return_prev = False  # 2026-07-14追加: オフセット復帰(cleared後のa_target=0)の遷移検知用
            self._icc_fallback_prev = False  # 2026-07-12追加: ICC見失いフォールバックの遷移検知用
            self._icc_fallback_skip_prev = False  # 2026-07-14追加: [ICC-FALLBACK-SKIP]遷移検知用
            self._stopping_no_vsafe_prev = False  # 2026-07-18追加(109節続報): [STOPPING-NO-VSAFE]遷移検知用
            self._plan_fail_prev_reason = None  # 2026-07-13追加: [PLAN-FAIL]間引き用
            self._plan_fail_log_count = 0
            self._plan_obs_prev_result = None   # 2026-07-13追加: [PLAN-OBS]間引き用
            self._plan_obs_log_count = 0
            self._plan_moving_log_count = 0     # 2026-07-13追加: [PLAN-MOVING-ENTER]間引き用
            # 2026-07-14追加(事象C対策): min-width veto(along_lane_need境界)のチャーン防止用
            #   デバウンス状態。案B(_ot_prev_side/_ot_prev_side_vid)と同じ考え方。
            self._plan_room_ok_count = 0
            self._plan_room_prev_vid = None
            self._plan_room_prev_side = None
            # 190-7節(2026-07-26追加): 反対側フォールバック(_try_opposite_side_fallback)
            #   専用の独立デバウンス状態。上記の主系統(counter_key="primary"扱い)とは
            #   完全に分離し、主系統の挙動(vid/side変化で即リセット)には一切影響しない。
            self._plan_room_ok_count_by_key = {}
            self._plan_room_prev_vid_by_key = {}
            self._plan_room_prev_side_by_key = {}
            # --- C 守り: 被追い越し検出＋壁近接減速 ---
            self._def_alongside_dist = float(_otget("def_alongside_dist", 3.0))  # [m] 横並び判定(縦)
            self._def_alongside_lat = float(_otget("def_alongside_lat", 1.0))   # [m] 横並び判定(横オフセット下限)
            self._def_rear_dist = float(_otget("def_rear_dist", 8.0))       # [m] 後方接近判定距離
            self._def_rear_faster = float(_otget("def_rear_faster", 1.0))   # [m/s] 後方車が速い判定
            # 2026-07-19修正(123節): dbg_corr_ub0/lb0はsafety_margin込みのため、デフォルト値も
            #   config.yamlと同じく0.15へ変更(0.5のままだと二重マージン化で過剰発火する)。
            # 2026-07-23追加(166節続報): wp257(コース最急コーナー直後)の揺れの原因切り分け実験用。
            #   wall_slowは現在wp1点のみ評価(79節で確定済みの設計、先読み無し)のため、コーナー
            #   頂点でmarginが一瞬0.01m級まで落ち、1周期だけ介入→次周期に解除、という
            #   オンオフの速度指令(実測: 加速指令+1.37/-1.37往復)が起きていた。これがwp257の
            #   揺れの原因かを切り分けるため、config一つでON/OFFできるようにする(既定true=従来通り)。
            self._wall_slow_enable = bool(_otget("wall_slow_enable", True))
            self._wall_slow_margin = float(_otget("wall_slow_margin", 0.15)) # [m] dbg_corr_ub0/lb0境界までの追加余裕(テーパー開始点)
            self._wall_slow_speed = float(_otget("wall_slow_speed", 2.0))   # [m/s] 壁近接時の速度上限(テーパー下限)
            # 2026-07-19追加(124節): wall_slow_margin(テーパー開始点)〜wall_slow_margin_hard
            #   (この値以下で完全にwall_slow_speedへ)の間を線形補間する下限。0.0=実際に
            #   コリドー境界へ到達した時点。
            self._wall_slow_margin_hard = float(_otget("wall_slow_margin_hard", 0.0))  # [m]
            # 並走ねばり: レーンが確保できる限り並走継続。消えたら後退イールド。
            self._along_lane_need = float(_otget("alongside_lane_need", 1.85))  # [m] 並走継続に必要なレーン幅
            self._along_lookahead = float(_otget("alongside_lookahead", 8.0))   # [m] レーン先読み弧長
            # 2026-07-10: 従来コード中のマジックナンバー(1.45=カート幅未満で物理的に通れない)を
            #   named定数化。side_block緩和(cl=1後)にも同じ値を再利用し、新規定数を増やさない。
            self._along_min_width = float(_otget("alongside_min_width", 1.45))  # [m] カート幅未満の物理下限
            # 2026-07-20追加(127節続報): along_min_widthの縦方向版。公式車両仕様(全長200cm)
            #   ベースの「両車の半長合計=1台分の全長」。fwd_dlat<along_min_widthとの
            #   AND判定で「実際に車体が重なるリスクがあるか」を表す(_footprint_risk参照)。
            self._along_min_length = float(_otget("along_min_length", 2.00))  # [m] カート全長未満の物理下限
            self._along_lane_ema = None  # along_lat用の_lane_min EMA(状態: 非OVERTAKING中のみ有効)
            # 2026-07-17追加(97節): line_cap(_offset_line_speed_cap)自体の平滑化用EMA。
            # 既存のalong_lane_ema/v_corridor_emaと同じ考え方・同じ時定数(_ot_ema_alpha)を
            # 再利用する。OVERTAKING中のみ有効(エンゲージ時にリセットする、下記参照)。
            self._line_cap_ema = None
            # 2026-07-17追加(94節、トークン整合性監査): along車の選択も毎周期
            # 「最も近いdlatの車」を選び直す実装で対象車IDに固定されないため、
            # critical_curvature_run/_ot_worth_count/_ot_giveup_countと同じ理由で
            # 対象車IDが変わった周期はEMAを仕切り直す。
            self._along_lane_prev_vid = None
            # 横方向TTC監視(2026-07-11設計、2026-07-12実挙動統合・案3で旧EMA判定を置換):
            #   0711-02ログ3件の実測分析に基づく対処。LAT-TTCのC2分岐(TTC≤ttc_critical_s)に
            #   giveup判定を一本化した。導入直後はconfig.yaml一行で旧EMA判定へ即時ロールバック
            #   できるフラグ(lat_ttc_enabled)を残していたが、7/12以降LAT-TTCは一度も無効化
            #   されておらず、旧EMA経路自体は2026-07-17に削除したため、このフラグも役目を終え
            #   同時に削除した(config.yamlのlat_ttc.enabledは常時有効という前提で読まない)。
            lat_ttc_cfg = getattr(self._cfg, "lat_ttc", None)
            def _lget(name, default):
                try:
                    return getattr(lat_ttc_cfg, name) if lat_ttc_cfg is not None else default
                except AttributeError:
                    return default
            self._lat_ttc = LateralTTCMonitor(
                beta=float(_lget("beta", 0.15)),
                # 2026-07-12追加: 生spaceの事前平滑化。既存のema_alpha(既定0.05、
                #   along_lane_emaと共通)を再利用し、新規チューニング値を増やさない。
                space_ema_alpha=float(_lget("space_ema_alpha", self._ot_ema_alpha)),
                ttc_danger_s=float(_lget("ttc_danger_s", 2.0)),
                ttc_critical_s=float(_lget("ttc_critical_s", 0.8)),
                # 2026-07-14再修正(フローチャートで洗い出したギャップ②): 従来はalong_lane_need
                #   (1.85m、高速すれ違い用の並走継続余裕)だったが、59節でengage側(_plan_pass)
                #   は既にalong_min_width(1.45m、物理下限)へ緩和済みのため、1.45〜1.84mの
                #   区間でengageに成功しても、cleared前はこの閾値でC2が即座に発火し
                #   「一瞬engageして即giveup」というループになっていた。engage側と同じ
                #   along_min_widthへ揃える。安全性は失われない: 真に危険な速さで空きが
                #   縮んでいる場合はttc_critical_s(0.8秒)側が独立に検知するため、この
                #   閾値の変更は「space自体は物理的に十分だが縮小トレンドの計算上わずかに
                #   際どく見える」境界ケースにのみ影響する。
                giveup_space_m=float(_lget("giveup_space_m", self._along_min_width)),
                switchback_space_m=float(_lget("switchback_space_m", self._along_lane_need + 0.5)),
                side_by_side_dlat_m=float(_lget("side_by_side_dlat_m", self._clear_lat_reacquire)),
                # 2026-07-14追加(水平展開、事象C対策): is_side_by_sideの離脱側ヒステリシス。
                # 既存clear_lat_release(_ot_clearedの解放閾値と同一値)をそのまま再利用する。
                side_by_side_dlat_release_m=float(
                    _lget("side_by_side_dlat_release_m", self._clear_lat_release)),
                side_by_side_ds_m=float(_lget("side_by_side_ds_m", self._clear_ds_beside)),
                caution_speed_margin_kmh=float(_lget("caution_speed_margin_kmh", 2.0)),
                min_trend_cycles=int(_lget("min_trend_cycles", 3)),
                # 2026-07-13追加: 真横到達(_ot_cleared)後の緩和閾値。既存along_min_width
                # (カート幅未満の物理下限)を再利用し、新規チューニング値を増やさない。
                cleared_space_m=float(_lget("cleared_space_m", self._along_min_width)),
                # 2026-07-14追加: v_inst(コリドー縮小瞬時微分)の物理妥当性クランプ。
                # 実測で-22〜-27m/s級の外れ値がfwd_dlat=3m超でも誤giveupを誘発していた
                # (0713-05 wp16-21, 0713-06 wp243-246)。
                v_inst_max=float(_lget("v_inst_max", 5.0)),
                # 2026-07-20追加(144節続報): C1_deferred猶予中に逃げ道が封鎖されている
                # 場合のキャップ強化用。既存self._wall_slow_speedをそのまま再利用する。
                wall_slow_speed=self._wall_slow_speed)
            self._lat_ttc_log_count = 0
            self._lat_ttc_prev_branch = "none"  # 2026-07-11: branch遷移検知用(単発イベントの取りこぼし防止)
            # 2026-07-20追加(132節、Gap①Phase0): [DLAT-TREND]エッジトリガーログ用。
            # side==0(未エンゲージ)の間にfwd_dlat縮小トレンドが確立したかどうかの
            # 立ち上がりのみを記録する(診断専用、ENGAGE判定への影響なし)。
            self._dlat_trend_alert_active = False
            # 2026-07-20追加(131-6節①、Gap①Phase1): [DLAT-TTC-VETO]エッジトリガー
            #   ログ用。ENGAGEがdlat縮小トレンドのTTCにより見送られた瞬間のみ記録する。
            self._dlat_ttc_veto_active = False
            self._dlat_ttc_veto_active_cycles = 0  # 190-5節: veto継続周期数(OFF遷移時に秒へ変換してログ)
            # 190-5節(2026-07-26追加): is_closing_trend(141節で3消費先=ENGAGEゲート/
            #   G2-RELEASE/force_include_vidへ一元共有)がどれだけ連続してTrueだったか、
            #   footprint_risk起因かトレンド起因かを1箇所で計測する。5日分18ログ横断調査
            #   (190節)で「相手停止中に完全停止し再発進不可」症状の一因として
            #   dlat_ttc系ゲートが複数ログで確認されたが、3消費先のどれが実際に長時間
            #   ブロックしていたか個別に切り分けるログが無かったための追加(診断専用、
            #   判定ロジックへの影響なし)。
            self._dlat_trend_true_cycles = 0
            self._dlat_trend_true_via_fp = False
            # 190-5節: force_include_vidがis_closing_trend起因で発火した瞬間のみを
            #   記録するエッジトリガー用(従来ログ皆無だった箇所)。
            self._force_include_vid_trend_active = False
            # 190-6節(2026-07-26追加、診断専用): _on_path(engage_lat_max=2.0m)が
            #   Falseで居続ける継続時間の計測。5日分18ログ横断調査(190節)で、
            #   両者静止中にfwd_dlatが単調増加し続けENGAGE不発火が長時間続く事例
            #   (0726-05)が見つかったが、_scan_traffic()のdlat計算(相手側・自車側で
            #   別々のwaypoint基準の局所座標系を差し引いている、直線では無視できる
            #   誤差が曲率区間では系統的に蓄積しうる)を直接改修するには影響範囲が
            #   広すぎるため、まず実地頻度・大きさを計測してから対処を設計する。
            self._on_path_false_cycles = 0
            self._on_path_false_start_dlat = None
            # (v_min は統一ICC化でクリープ(v_creep)+ハード停止に置換済み・不読)
            self._ot_infeasible_stop = int(_otget("infeasible_stop", 5))   # MPCがこの回数連続infeasibleで安全STOPへ
            self._ot_infeasible_latch_cycles = int(_otget("infeasible_latch", 40))  # 安全STOP後、再挑戦を抑制する周期数
            self._ot_infeasible_latch = 0  # >0 の間は OVERTAKING を禁止（オシレーション防止）

            # --- スタック検知バック(2026-07-09) ---
            #   既存_ot_stateとは独立の状態機械。指令速度>実速度の乖離が続く(=壁接触等で
            #   物理的に進めていない)場合のみ発動する。ICCが意図的に低速追従している場面
            #   (指令速度自体が低い)は該当しないため誤発動しない。
            #   ギア/後退値はAutoware公式サンプル(autoware_practice_course/backward.cpp)を
            #   そのまま採用(ユーザー承認済み)。
            stk = getattr(self._cfg, "stuck_recovery", None)
            def _stkget(name, default):
                try:
                    return getattr(stk, name) if stk is not None else default
                except AttributeError:
                    return default
            self._stuck_startup_grace_s = float(_stkget("startup_grace_s", 10.0))  # [s] 起動直後は判定しない
            self._stuck_u0_thr = float(_stkget("u0_thr", 2.0))         # [m/s] 指令速度がこれ以上
            self._stuck_v_thr = float(_stkget("v_thr", 0.3))           # [m/s] 実速度がこれ未満
            self._stuck_hold_cycles = int(_stkget("hold_cycles", 120))  # [周期] 継続でスタック判定(≈3s@40Hz)
            self._ghost_block_hold_cycles = int(_stkget("ghost_block_hold_cycles", 40))  # [周期] GHOST-BLOCK早期ログ(≈1s@40Hz、経路1本発動より早い)
            self._ghost_block_logged = False  # 現在のepisode内で[GHOST-BLOCK]を出力済みか(_stuck_count==0でリセット)
            self._stuck_infeas_thr = int(_stkget("infeas_thr", 300))  # [周期] H4-liteに猶予を与えた上でこれ以上連続infeasibleなら即発動(≈7.5s@40Hz)
            self._stuck_gear_settle_cycles = int(_stkget("gear_settle_cycles", 20))  # [周期] ギア確定待ち(≈0.5s)
            self._stuck_backup_dist = float(_stkget("backup_dist", 2.0))  # [m] 後退距離
            self._stuck_backup_speed = float(_stkget("backup_speed", -3.0))  # [m/s] サンプル準拠
            self._stuck_backup_accel = float(_stkget("backup_accel", 1.5))   # [m/s^2] サンプル準拠
            self._stuck_hold_accel = float(_stkget("hold_accel", -2.5))      # [m/s^2] ギア切替待ち保持
            # 経路3(2026-07-10追加): 相手車に塞がれ続け、MPCが正しく安全停止を選び続ける
            #   (u0=0・infeas=0のまま)デッドロック用。u0/infeasibilityを問わず、実速度のみで判定。
            #   実測: 予選0710-02で363秒間完全停止・0周(相手2台による封鎖、K対策の窓内他車
            #   チェックが両側とも正しく拒否し続けた結果、既存の経路1/2はどちらも構造上発動しない)。
            self._stuck_stall_v_thr = float(_stkget("stall_v_thr", 0.1))    # [m/s] 「まったく動かない」判定
            self._stuck_stall_hold_cycles = int(_stkget("stall_hold_cycles", 400))  # [周期] ≈10s@40Hz
            self._stuck_push_dist = float(_stkget("push_dist", 2.0))        # [m] 前進突入の目標距離
            self._stuck_push_speed = float(_stkget("push_speed", 3.0))      # [m/s] 前進突入の速度指令
            self._stuck_push_accel = float(_stkget("push_accel", 1.5))      # [m/s^2] backup_accel流用
            # 2026-07-21追加(148節②、ユーザー提案「上流から下流まで同じ計算式・同じ値を使う」):
            #   従来PUSHは操舵0固定の完全直進で、_ot_state側が既に持つ状況認識(_plan_passの
            #   側選択、動的コリドー)を一切参照していなかった(0721-01実測、wp332で約193秒間
            #   完全スタック、fwd_dlatが1.35m付近で頭打ちし脱出できなかった根本原因)。
            #   PUSH開始時に既存の_plan_pass(ENGAGE判定と全く同じ関数・同じ判断基準)を
            #   一度だけ呼び、側(plan_side)を決める。
            #   MPCの最終行程解(dbg_corr_ub_arr/lb_arr)がBACKUP後は位置ずれで陳腐化している
            #   可能性があるため、その値は「これ以上は超えない安全上限」としてのみ使う
            #   (陳腐化していても危険側には作用しない、常により保守的な方向にのみ効く)。
            #   PUSH自体はMPCが手詰まりになった時の開ループ脱出ルートという役割を守るため、
            #   実行(操舵のかけ方)自体はMPCに依存しない単純な固定値のままとする。
            # 2026-07-24再設計(171節続報、ユーザー指示): 経路3専用だった小角(6°)の
            #   「押し出し」から、経路1/2/3全てで発動する「低速+最大舵角での障害物回避」へ
            #   拡張する(BACKUP後は必ず眼前に障害物または壁があるため)。舵角上限は独自の
            #   定数を持たず、MPC自体のハード制約(delta_max_deg)をそのまま上限として使う
            #   (「最大限」の定義を1箇所に一元化、新規の独立した最大値を増やさない)。
            self._stuck_push_steer_max_deg = float(
                _stkget("push_steer_max_deg", self._cfg.mpc.delta_max_deg))  # [deg] PUSH中の操舵角上限(既定=MPCのdelta_max_deg)
            self._stuck_push_steer_room_ref = float(_stkget("push_steer_room_ref", 1.0))  # [m] この空きで上限角に到達(線形スケール)
            self._stuck_push_side = 0  # PUSH開始時に決めた側(+1/-1/0=不明)。再クリア判定に再利用
            # push_timeout/retry_budgetは周期数ではなく実時間で判定(高負荷時でも一定の実時間で
            #   確実に打ち切るため。stall_hold_cyclesのみ既存のu0_thr等と表記を揃え周期数のまま)。
            self._stuck_push_timeout_s = float(_stkget("push_timeout_s", 5.0))  # [s] 距離未到達でも打切り
            self._stuck_stall_retry_budget_s = float(_stkget("stall_retry_budget_s", 360.0))  # [s] 経路3の合計リトライ許容時間(無限リトライ回避)
            # BACKUPウォッチドッグ(2026-07-13追加): 0713-03実測で、REVERSEギア確認済みでも
            #   実速度がv≈0のまま500秒以上(ログの残り全体)固まり続けた事例を確認。PUSH/経路3には
            #   既に実時間タイムアウトがあるのに、BACKUP自体には無く無限固着の直接原因だった。
            #   同じ「実時間で打切り+合計リトライ予算」パターンを踏襲(ユーザー承認: 予算は10分)。
            self._stuck_backup_timeout_s = float(_stkget("backup_timeout_s", 5.0))  # [s] 距離未到達でも打切り(push_timeout_s同値)
            self._stuck_backup_retry_budget_s = float(_stkget("backup_retry_budget_s", 600.0))  # [s] 合計リトライ許容時間(10分)
            # 後退不能検知(2026-07-14追加): 0713-05実測で、後退方向に障害物があり物理的に
            #   全く動けない(実速度がv≈0のまま)状態でも、既存のdist/timeout判定(5秒経過で
            #   同じ後退を再試行)を42回以上繰り返し、進捗距離が1.42m→0.00mへ単調悪化する
            #   だけで一度も回復せず、リトライ予算(600秒)を消費し尽くすまで無理に押し
            #   続けていた。実速度は既に取得済み(self._odom)のため、新規センシングは不要。
            #   実速度が閾値未満のまま一定時間続いたら「後退方向に障害物があり不可能」と
            #   直接確定し、5秒/600秒を待たずに即座に無理な後退を止める(ユーザー指示)。
            self._stuck_backup_blocked_v_thr = float(_stkget("backup_blocked_v_thr", 0.05))  # [m/s] これ未満は「動いていない」
            self._stuck_backup_blocked_confirm_s = float(_stkget("backup_blocked_confirm_s", 1.5))  # [s] 継続でブロック確定
            # 184節追加(2026-07-26): 隙間狙い操舵+動的後退距離+シャッフル(縦列駐車脱出)。
            self._stuck_gap_lookahead_n = int(_stkget("gap_lookahead_n", 3))
            self._stuck_ey_kp = float(_stkget("ey_kp", 0.8))
            self._stuck_psi_kp = float(_stkget("psi_kp", 1.0))
            self._stuck_rear_clearance_margin_m = float(_stkget("rear_clearance_margin_m", 0.5))
            self._stuck_rear_scan_max_dist_m = float(_stkget("rear_scan_max_dist_m", 3.0))
            self._stuck_push_heading_tol_rad = np.deg2rad(
                float(_stkget("push_heading_tol_deg", 15.0)))
            self._stuck_shuffle_max_cycles = int(_stkget("shuffle_max_cycles", 6))
            # 2026-07-26追加(190-2節): path=3(完全停止検知、count=400≒10.0秒@40Hz)の
            #   検知自体が最低10.0秒かかるため、旧既定6.0秒では同一地点での再STUCKが
            #   常に「別エピソード」扱いになり、187節の反転リトライが機能しなかった。
            #   config.yaml参照。
            self._stuck_shuffle_episode_gap_s = float(_stkget("shuffle_episode_gap_s", 15.0))
            self._stuck_shuffle_episode_radius_m = float(_stkget("shuffle_episode_radius_m", 3.0))
            # 2026-07-26追加(186節続報、0726-01local試験で実測: シャッフル上限到達後の
            #   「復帰断念→NORMAL委譲」が、車が全く動けていないにも関わらず約3秒周期
            #   (STUCK再検知の周期)で永久に繰り返され、一度も回復しないまま95秒以上
            #   固着し続けるケースを確認した。同一地点判定(shuffle_episode_gap_s/
            #   radius_m)の性質上、車が動かない限りこの周期は6.0s未満に収まり続け
            #   「同一エピソード継続」から永久に抜けられない構造的な穴だった。
            #   shuffle_max_cyclesとは別に「シャッフル上限そのものに到達した回数」を
            #   数え、上限内であれば操舵方向を反転して(挟まれ方が非対称なら逆方向で
            #   抜けられる可能性があるため)もう一巡だけシャッフルを再試行する。
            #   新規の安全弁緩和は行わない(既存のBACKUP物理妥当性判定・PUSH完了判定は
            #   無変更、操舵方向の候補を1つ増やすだけ)。
            self._stuck_max_giveup_streak = int(_stkget("max_giveup_streak", 3))
            # 2026-07-26追加(186節続報): PUSHの「reason=cleared」判定が、コリドー幅+
            #   向き一致のみを見ており実際の移動量を見ていなかったため、PUSH開始直後
            #   (dist=0.00m)でも成立し得た。これにより同一地点で全く前進しないまま
            #   「回避成功」扱いでNORMALへ復帰→再STUCK検知、を繰り返しシャッフル回数
            #   だけを浪費するケースを実測(0726-01local、cycle=4/5でdist=0.00-0.01m
            #   のままreason=clearedが成立)した。最小移動量を追加要求することで、
            #   実際に動けていない「見かけ上の回避成功」を弾く。
            self._stuck_push_min_dist_for_cleared = float(
                _stkget("push_min_dist_for_cleared", 0.15))  # [m]
            self._stuck_backup_dist_eff = self._stuck_backup_dist  # 実効後退距離(BACKUP開始時に再計算)
            self._stuck_shuffle_cycle = 0          # 同一エピソード内のBACKUP→PUSH反復回数
            self._stuck_giveup_streak = 0          # 2026-07-26追加(186節続報): シャッフル上限に
            #   到達した回数(同一地点判定が続く限り持続、新規エピソードで0へ戻る)
            self._stuck_push_side_flip = False     # 2026-07-26追加(186節続報): giveup_streak
            #   が進むたびに反転し、PUSHの操舵方向候補を毎回変える
            self._stuck_episode_last_end_time = None  # 直近のPUSH/断念完了時刻(シャッフル判定用)
            self._stuck_episode_last_pose = None      # 同上、完了時の位置
            self._stuck_state = "NORMAL"   # NORMAL/WAIT_REVERSE/BACKUP/WAIT_DRIVE_PUSH/PUSH
            # (2026-07-24, 171節続報: WAIT_DRIVEは経路1/2専用の直進復帰だったが、経路を
            #   問わずWAIT_DRIVE_PUSH→PUSHへ統一したため到達不能になり削除した)
            self._stuck_count = 0          # 指令/実速度乖離の連続周期数(経路1)
            self._stuck_stall_count = 0    # 実速度ほぼ0の連続周期数(経路3)
            self._stuck_u0_last = 0.0      # 前周期の最終指令速度(次周期のスタック判定に使用)
            self._stuck_gear_wait_count = 0
            self._stuck_backup_log_count = 0    # [STUCK-BACKUP]ログの間引き用(2026-07-11)
            self._stuck_backup_start = None  # (x, y) 後退開始位置
            self._stuck_backup_start_time = None  # BACKUP開始時刻(ウォッチドッグ判定用、2026-07-13)
            self._stuck_backup_first_timeout_time = None  # BACKUPウォッチドッグ初回発火時刻(リトライ予算の起点)
            self._stuck_backup_budget_exhausted_logged = False
            self._stuck_backup_zero_v_since = None  # 実速度が閾値未満になった開始時刻(後退不能検知用、2026-07-14追加)
            self._stuck_trigger_path = None  # 1=u0/v, 2=infeasibility, 3=完全停止(→PUSH分岐)
            self._stuck_stall_first_trigger_time = None  # 経路3の初回発火時刻(リトライ予算の起点)
            self._stuck_stall_budget_exhausted_logged = False
            self._stuck_push_start = None       # (x, y) PUSH開始位置
            self._stuck_push_start_time = None  # PUSH開始時刻(タイムアウト判定用)
            self._stuck_push_log_count = 0      # [STUCK-PUSH]ログの間引き用
            self._stuck_push_steer = 0.0        # PUSH開始時に1回だけ決める操舵角[rad](148節②)
            # 自動衝突検知(2026-07-10追加): 予選環境のbagにcondition系トピックがなく、
            #   0710-03の壁衝突も既存の/aichallenge/pitstop/condition経由の検知
            #   (ピット専用の別目的ヒューリスティック)では捕捉できなかった(手動でbagの
            #   生速度を解析して発見)。実速度の1周期あたりの急落を自車のみで直接監視する。
            #   閾値0.8m/s/周期は、正当な制動(a_min=-1.37m/s²)による上限(≈0.03-0.14m/s/周期、
            #   高負荷時のサイクル延びを見込んでも)の何倍もの余裕を持たせた値。
            self._collision_suspect_dv = float(_stkget("collision_suspect_dv", 0.8))  # [m/s] 1周期での急落閾値
            self._collision_check_v_prev = None
            # 累積版(2026-07-11追加): 低速域での接触が複数周期に分散するケースを拾う。
            self._collision_suspect_cum_dv = float(_stkget("collision_suspect_cum_dv", 1.0))  # [m/s] 窓内最大値からの下落閾値
            self._collision_cum_window_cycles = int(_stkget("collision_cum_window_cycles", 5))  # [周期] 監視窓幅
            self._collision_v_window = deque(maxlen=self._collision_cum_window_cycles)
            self._gear_report = GearReport()  # 初期値(report=0)
            # --- 方式B: 空き側へ e_y 目標をオフセット（発進時ランプ付き）---
            self._ot_d_off = float(_otget("d_off", 1.8))           # [m] 参照ラインからの寄せ量
            # 2026-07-22追加(160節続報、issue⑤①: STOPPING中の能動的空き確保): footprint_risk
            #   等の反応的検知を待たず、ENGAGE試行と同じ_plan_pass判定が地形的に成立している間、
            #   停止/低速の相手に対してのみ小さく先行して寄せておく上限値。本追い越し(_ot_d_off)
            #   より十分小さくし、_corr_bound_ahead()による動的コリドークランプ後の実効値は
            #   通常これよりさらに小さくなる。
            self._ot_proactive_bias_max = float(_otget("proactive_bias_max", 0.3))  # [m] STOPPING中の先行寄せ上限
            self._ot_ramp_time = float(_otget("ramp_time", 0.5))   # [s] 0→full への漸増時間
            self._ot_alpha = 0.0       # 現在のブレンド係数 0..1（ランプで漸増漸減）
            # B-lite: ヘディング参照バイアスの上限[rad]（開き側へ車体を傾けて横移動させる）
            self._ot_psi_max = np.deg2rad(float(_otget("psi_bias_max_deg", 20.0)))
            # 「徐々に右」funnel: コリドーを現在e_yからこのwp数かけて回避形状へ遷移（feasible維持）
            self._ot_funnel_steps = int(_otget("funnel_steps", 12))
            # アグレッシブ追い越し: 実寸ベースの安全マージン＋crab前倒し＋安全網
            self._ot_safety_margin = float(_otget("safety_margin_overtake", 0.8))  # [m] 追い越し中マージン
            # 2026-07-14追加: safety_margin_overrideの遷移平滑化(0714-01 事象A対策)。
            #   従来はOVERTAKING(0.8m)⇔STOPPING/NORMAL(None=self._mpc.model.safety_margin、
            #   width/√2から算出される実測既定値)の切替が同一周期内で即座に(1周期=dt秒で)
            #   起きていた。giveup直後の状態遷移(OVERTAKING→STOPPING)がまだ壁際に寄って
            #   いる実車両位置と同時に発生すると、その周期だけ幾何学的にinfeasibleになり
            #   得る(0714-01実測: giveup後0.6秒でinfeas急増・速度3.43→1.98m/s急減)。
            #   新規パラメータは増やさず、オフセットランプと全く同じ_ot_ramp_time(既存の
            #   時定数)でmargin自体も滑らかに遷移させ、「横方向へ戻る速さ」と「マージンが
            #   緩む速さ」を同じ時間スケールに揃える。
            self._ot_margin_full = float(self._mpc.model.safety_margin)  # [m] 通常時(override無し)相当値
            self._ot_margin_cur = self._ot_margin_full   # 現在の実効マージン(ランプ済み)。初期値はNORMAL相当
            self._ot_margin_ramping_prev = False  # [MARGIN-RAMP]ログのエッジ検知用
            self._ot_q_ey = float(_otget("q_ey_overtake", 5.0e6))                  # 追い越し中 Q[e_y]
            # 2026-07-24再導入(170節続報): 169節でQ[e_y]ベース値を3M→5Mへ戻した際、この
            #   Q曲率スケジュール(v1〜v5+量子化ゲート)がONのままだったため、予選ログ
            #   (0724-01→0724-02)の蛇行悪化が実際にはQ[e_y]ベース値の変更(交絡)由来であり、
            #   スケジュール自体の影響ではなかった可能性が高いと判明した(170節でスケジュールを
            #   撤去したがローカルではwp85-89等の悪化が残り、Q[e_y]を3Mへ戻すとそれらが解消した
            #   ため)。ベース値をQ[e_y]=3M(純正中速値)へ確定させた上で171節で再導入した。
            # 2026-07-24再撤去(174節続報、A/Bテスト): 予選ログの「全体的な蛇行」の原因切り分け
            #   のため、Q[e_y]ベース値(3M、確定済み)は維持したまま、スケジュール自体の有無だけを
            #   単独変数として一時的にOFFにする。167〜173節の処理落ち対策・STUCK対策は無関係の
            #   ため無変更のまま。171節以前と同じ撤去手順(mpc_controller.py 5箇所+config.yaml)。
            # (safe_clear/safe_brake は統一ICCの実横間隔判定(min_lat_sep=1.8, 常時適用)に置換済み・不読)
            # F3クリープ: 停止車の後ろで完全停止すると横移動を作れずデッドロック→クリープ前進で斜めに抜け出す
            self._ot_v_creep = float(_otget("v_creep", 1.5))          # [m/s] 未クリア時のクリープ速度
            self._ot_hard_stop_gap = float(_otget("hard_stop_gap", 1.8))  # [m] 実距離これ未満は完全停止
            # F3フロア比例化(2026-07-13追加): hard_stop_gap(完全停止)〜f3_taper_gap(通常クリープ
            #   復帰)の間を線形補間し、floorを滑らかに変化させる。0713-03実測(wp169-178)で
            #   確認したバンバン振動(floorがv_creep⇔0を二値で行き来し、相手が徐行だと停止・
            #   再接近を繰り返す)を構造的に無くすための対処。値はpass_clear/def_alongside_dist
            #   と同じ3.0m級のスケール感で設定(新規チューニング値だが既存の桁感に合わせた)。
            self._ot_f3_taper_gap = float(_otget("f3_taper_gap", 3.0))  # [m] この距離以上で通常クリープに復帰
            self._f3_taper_zone_prev = None  # [F3-TAPER]ログの遷移検知用(stop/taper/creep)
            self._ot_q_applied = None   # Qモード: None/"normal"/"overtake"
            # 追い越し後の「戻りfunnel」: ライン外から徐々にレースラインへ復帰させる
            self._ot_return_done = float(_otget("return_done_ey", 0.7))  # [m] |e_y|<これで復帰完了
            self._ot_returning = False  # 追い越し後の復帰中フラグ
            self._ot_state = "NORMAL"   # NORMAL / OVERTAKING / STOPPING
            self._ot_side = 0           # +1=左, -1=右, 0=未選択（e_y規約: +左/-右）
            self._ot_side_locked = 0    # A: コミット中の側（一時STOPPINGを跨いで保持。通過完了/恒久失敗で解除）
            # 2026-07-20追加(131-6節④、対象車の一意性): エンゲージ時に_plan_passが
            #   実際に計画対象とした相手車ID。オフセット復帰判定が「前方40m以内の
            #   任意の1台」ではなく、この特定車両の縦方向クリアを見るために使う。
            self._ot_target_vid = None
            # 案B(2026-07-11): 側消失(_side_blocked)によるSTOPPING離脱→短時間再エンゲージでの
            #   側反転を抑止するヒステリシス。同一対象車×時間窓内なら前回側を優先する
            #   (0710-06実測: wp176-178でside+1→-1反転直後に実速度が0.18m/sへ8秒張り付いた事象への対策)。
            #   giveup(相手が速すぎる)・infeasible(恒久失敗)による離脱はこの対象外(側の再選択が正当なため)。
            self._ot_side_flip_hyst_s = float(_otget("side_flip_hysteresis_s", 3.0))  # [s]
            self._ot_prev_side = 0
            self._ot_prev_side_vid = None
            self._ot_prev_side_time = None
            # 診断用(2026-07-09): _plan_passが計算した「窓内実測最小幅」をログへ出力するため保持。
            #   scanの単点Lfree/Rfree(Fix A)と別に見えるようにし、次回ログのみで
            #   「単点値と窓内判断値の乖離」を判定できるようにする(bagデコード不要化)。
            self._dbg_plan_lf = float('nan')
            self._dbg_plan_rf = float('nan')
            # 診断用(2026-07-22、153節: オフセット目標が_plan_passのroom見積もりより
            #   大幅に小さい事象の切り分け用): _corr_bound_ahead()が採用した最小値が
            #   現在位置から何m先の地点で発生したかを記録する(相手車両起因かコーナー
            #   形状起因かを次回ログで直接判別するため。既存の返り値・計算式は無変更)。
            self._dbg_corr_bound_at_m = float('nan')
            # 診断用(2026-07-19、wp176-178ウェッジ再調査): 障害物分岐の窓内先読みが
            #   計算するlf_i/rf_iはこれまで最終的な最小値(lf_min/rf_min)しかログに
            #   残らず、「fwd_latを固定したままカーブする壁境界に当てはめる」計算過程
            #   自体がwaypointごとにどう推移したかを検証できなかった。ENGAGE時のみ
            #   [ENGAGE]ログへ窓内トレースを追加出力する(判定ロジックは無変更)。
            self._dbg_plan_trace = []
            self._dbg_n_dynobs = 0
            self._ot_max_width = float(self._cfg.reference_path.max_width)     # コース最大幅
            self._ot_half_w = 0.5 * self._ot_max_width                          # （診断/フィルタ用）半幅
            self._ot_dbg_loop = 0
            # 弧長の累積をキャッシュ（get_s_at_waypoint の毎回 cumsum を回避）。
            # _waypoint_xy（上で生成済み）と同一インデックス規約。
            self._wp_s_cum = np.cumsum(self._reference_path.segment_lengths)
            # --- 相手速度マップ(位置別速度包絡線をオンライン学習。索引=wp_id=使用中traj) ---
            self._opp_map = OpponentSpeedMap(
                n_wp=len(self._reference_path.waypoints), s_cum=self._wp_s_cum)
            self._opp_map_pub = self.create_publisher(
                Float32MultiArray, "/mpc/opponent_speed_map", 1)         # 検証: bag録画・復号用
            self._opp_map_marker_pub = self.create_publisher(
                MarkerArray, "/mpc/opponent_speed_map_markers", 1)       # 検証: RViz色プロット
            self._opp_map_pub_loop = 0
            # --- end gate2 overtake params ---

        # Laps
        self._current_laps = 1
        self._last_lap_time = 0.0
        self._lap_times = [None] * (self.MAX_LAPS + 1) # +1 means include lap 0

        # condition
        self._last_condition = None
        self._last_colliding_time = None

        # stats
        self._stats = ExecutionStats(self.get_logger(), window_size=50, record_count_threshold=1000)

        # save config
        if self._cfg.common.save_config:
            self._save_config()

    def _save_config(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self.PKG_PATH + f"log/{now}"
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(self._config_path, os.path.join(dst_dir, "config.yaml"))

    def _load_densified_pit_points(self):
        """from_garage(約3m間隔)を読み、弧長3次スプラインで平滑・高密度化した (x,y) 配列を返す。
        狭い車庫出口の曲がりが直線近似でカクつくのを防ぐ（追従誤差・オーバーシュート低減）。"""
        import pandas as pd
        from ament_index_python.packages import get_package_share_directory
        from scipy.interpolate import CubicSpline
        pit = self._cfg.pit_lane  # type: ignore
        csv = get_package_share_directory(pit.csv_pkg) + "/" + pit.csv_rel
        df = pd.read_csv(csv)
        wx = np.asarray(df["x"], dtype=float)
        wy = np.asarray(df["y"], dtype=float)
        dens = float(getattr(pit, "densify_resolution", 0.3))
        s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(wx), np.diff(wy)))))
        ss = np.arange(0.0, float(s[-1]), dens)
        sx = CubicSpline(s, wx)(ss)
        sy = CubicSpline(s, wy)(ss)
        self.get_logger().info(f"pit raw path loaded from {csv} (densified {len(wx)}->{len(ss)} @~{dens}m)")
        return sx, sy

    def _build_pit_ref_path(self, spawn_x: float, spawn_y: float) -> ReferencePath:
        """spawn 近傍から始まる densified from_garage の ReferencePath を構築（横シフトなし）。
        車は経路の右側にスポーンするため、横シフトせず本来のfrom_garageを参照に与え、
        MPCが最短で経路へ収束→追従できるようにする（GNSS位置補正と併用）。"""
        x = np.asarray(self._pit_raw_x, dtype=float)
        y = np.asarray(self._pit_raw_y, dtype=float)
        i0 = int(np.argmin((x - spawn_x) ** 2 + (y - spawn_y) ** 2))
        xs = x[i0:]
        ys = y[i0:]
        cfg_ref = self._cfg.reference_path  # type: ignore
        dens = float(getattr(self._cfg.pit_lane, "densify_resolution", 0.3))  # type: ignore
        path = ReferencePath(
            self._map, xs.tolist(), ys.tolist(), dens, cfg_ref.smoothing_distance,
            cfg_ref.max_width, False)  # circular=False（車庫→コースの一本道）
        path.compute_speed_profile({
            "a_min": self._mpc_cfg.a_min, "a_max": self._mpc_cfg.a_max,
            "v_min": 0.0, "v_max": self._mpc_cfg.v_max, "ay_max": self._mpc_cfg.ay_max})
        # lanelet2境界を壁の真値として各wpのコリドー(ub/lb)に上書き（占有格子より正確）
        self._apply_lanelet_corridor(path)
        self.get_logger().info(
            f"pit ref path built from spawn-nearest point ({len(path.waypoints)} wp, no lateral shift)")
        return path

    def _load_pit_lane_bounds(self):
        """lanelet2地図から、from_garage(ピット経路)に最も多く沿うlanelet を選び、その左右境界
        2本の点列(densified numpy配列)を返す。占有格子(ピットで不正確)の代わりに壁の真値として使う。
        左右の役割はlaneletのleft/role表記に依存せず、後段で各wpの法線符号から判定する。"""
        import xml.etree.ElementTree as ET
        from ament_index_python.packages import get_package_share_directory
        _pit = self._cfg.pit_lane  # type: ignore
        pkg = getattr(_pit, "lanelet_pkg", "aichallenge_submit_launch")
        rel = getattr(_pit, "lanelet_rel", "map/lanelet2_map.osm")
        osm_path = get_package_share_directory(pkg) + "/" + rel
        root = ET.parse(osm_path).getroot()
        nodes = {}
        for n in root.findall("node"):
            x = y = None
            for tg in n.findall("tag"):
                if tg.get("k") == "local_x": x = float(tg.get("v"))
                elif tg.get("k") == "local_y": y = float(tg.get("v"))
            if x is not None:
                nodes[n.get("id")] = (x, y)
        ways = {w.get("id"): [nodes[r.get("ref")] for r in w.findall("nd") if r.get("ref") in nodes]
                for w in root.findall("way")}
        # ピット判定用の経路点：from_garage は1周分あるため、発進直後(弧長≤30m)の
        # 「ピット出口区間」だけを使う。全周で判定するとトラック側laneletが誤選択される。
        prx = np.asarray(self._pit_raw_x, dtype=float)
        pry = np.asarray(self._pit_raw_y, dtype=float)
        s_raw = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(prx), np.diff(pry)))))
        m = s_raw <= 30.0
        rp = np.c_[prx[m], pry[m]]
        # 選択: ピット区間のカバー点数最大 → 同点は speed_limit 小(ピット専用レーン) → 平均距離小
        best = None  # (key, wayA, wayB, cover)
        for rl in root.findall("relation"):
            tags = {tg.get("k"): tg.get("v") for tg in rl.findall("tag")}
            if tags.get("type") != "lanelet":
                continue
            mw = [m.get("ref") for m in rl.findall("member")
                  if m.get("type") == "way" and m.get("role") in ("left", "right")]
            if len(mw) < 2:
                continue
            bpts = []
            for wid in mw[:2]:
                bpts += ways.get(wid, [])
            if len(bpts) < 2:
                continue
            B = np.asarray(bpts)
            dmin = np.array([np.min(np.hypot(B[:, 0] - p[0], B[:, 1] - p[1])) for p in rp])
            cover = int(np.sum(dmin < 2.5))
            if cover == 0:
                continue
            avgd = float(np.mean(dmin[dmin < 2.5]))
            try:
                spd = float(tags.get("speed_limit", "999"))
            except ValueError:
                spd = 999.0
            key = (cover, -spd, -avgd)  # cover最大 → speed小 → 距離小
            if best is None or key > best[0]:
                best = (key, mw[0], mw[1], cover)
        if best is None or best[3] == 0:
            raise RuntimeError("no lanelet covers the pit path")
        def densify(pts, res=0.05):
            p = np.asarray(pts, dtype=float)
            seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
            s = np.concatenate(([0.0], np.cumsum(seg)))
            ss = np.arange(0.0, float(s[-1]), res)
            return np.c_[np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])]
        A = densify(ways[best[1]]); Bd = densify(ways[best[2]])
        self.get_logger().info(
            f"pit lanelet bounds loaded from {osm_path} (ways {best[1]}/{best[2]}, cover={best[3]} pts)")
        return A, Bd

    def _apply_lanelet_corridor(self, path: ReferencePath) -> None:
        """ピット経路の各wpについて、lanelet2左右境界への垂直符号付距離から wp.ub(左,+)/wp.lb(右,-)
        を上書きする。境界が及ばない区間(コース側)や wp が境界外の場合は従来値(占有格子)のまま。"""
        A = getattr(self, "_pit_bound_A", None)
        B = getattr(self, "_pit_bound_B", None)
        if A is None or B is None:
            return
        cover = 0
        for wp in path.waypoints:
            nx, ny = -np.sin(wp.psi), np.cos(wp.psi)  # 左法線(+左)
            ia = int(np.argmin((A[:, 0] - wp.x) ** 2 + (A[:, 1] - wp.y) ** 2))
            ib = int(np.argmin((B[:, 0] - wp.x) ** 2 + (B[:, 1] - wp.y) ** 2))
            da = float(np.hypot(A[ia, 0] - wp.x, A[ia, 1] - wp.y))
            db = float(np.hypot(B[ib, 0] - wp.x, B[ib, 1] - wp.y))
            if da > 3.0 or db > 3.0:        # 境界が及ばない（コース側）→ 従来値
                continue
            sa = (A[ia, 0] - wp.x) * nx + (A[ia, 1] - wp.y) * ny  # 符号付横位置(+左)
            sb = (B[ib, 0] - wp.x) * nx + (B[ib, 1] - wp.y) * ny
            ub = max(sa, sb)               # 左壁(+)
            lb = min(sa, sb)               # 右壁(-)
            if ub <= 0.0 or lb >= 0.0:     # wp が両境界の同側(=レーン外/端) → 従来値で安全側
                continue
            wp.ub = ub
            wp.lb = lb
            cover += 1
        self.get_logger().info(
            f"pit lanelet corridor applied to {cover}/{len(path.waypoints)} wp")

    def _race_line_min_dist(self, x: float, y: float) -> float:
        """現在位置からレースライン(traj_mincurv)までの最近傍距離[m]。発進判別・コースイン判定に使用。"""
        d = self._race_xy - np.array([x, y], dtype=np.float64)
        return float(np.sqrt(np.min(np.einsum('ij,ij->i', d, d))))

    def _set_active_path(self, path: ReferencePath, on_pit: bool) -> None:
        """参照経路を切り替える（wp_id再ローカライズ＋追い越し用キャッシュ・静的コリドー再構築）。"""
        self._reference_path = path
        # 190-3節: 経路差し替え後も同じコリドー拡大レートリミットを適用する(新規ReferencePath
        #   インスタンスはcorridor_widen_step_m既定値がinfのため、ここで設定しないと消える)。
        self._reference_path.corridor_widen_step_m = self._corridor_widen_step_m
        self._car.update_reference_path(path)  # model.reference_path 更新＋wp_id/s再ローカライズ
        self._on_pit = on_pit
        self._static_corridor_ready = False     # 新経路で静的コリドーを作り直す
        if self.USE_OBSTACLE_AVOIDANCE:
            wps = path.waypoints
            self._waypoint_xy = np.asarray([(wp.x, wp.y) for wp in wps], dtype=np.float64)
            self._wp_s_cum = np.cumsum(path.segment_lengths)
        try:
            if self._ref_vel_configulator is None:
                self._publish_ref_path_marker(path)
        except Exception:
            pass

    def _setup_pub_sub(self) -> None:
        # Publishers
        if self.USE_BUG_ACC:
          self._command_pub = self.create_publisher(
            AckermannControlBoostCommand, "/boost_commander/command", 1)
        else:
          self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
          self._command_raw_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd_raw", 1)
          print("use normal ackermann control command")

        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._mpc_pred_pub = self.create_publisher(
            MarkerArray, "/mpc/prediction", 1)
        self._mpc_pred_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/motion_planning/obstacle_stop_planner/virtual_wall", 1)

        # gate2: 追い越し診断トピック（rosbag解析用）。
        # data = [state(0:NORMAL/1:OVERTAKING/2:STOPPING), side(+1左/-1右/0), n_fwd,
        #         d_min, left_free, right_free, v_cmd]
        self._overtake_status_pub = self.create_publisher(
            Float32MultiArray, "/mpc/overtake_status", 1)

        # スタック検知バック(2026-07-09): ギア切替専用トピック。Autoware公式サンプル
        #   (autoware_practice_course/backward.cpp)と同一のトピック名・型・パターンを踏襲。
        self._gear_cmd_pub = self.create_publisher(
            GearCommand, "/control/command/gear_cmd", 1)

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)
        self._ref_path_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/debug/bound", latching_qos)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)
        # ピット低速時はEKF位置が壁側へ約0.45mズレるため、GNSS実測位置で補正する（位置のみ使用）
        self._gnss_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/sensing/gnss/pose_with_covariance", self._gnss_pose_callback, 1)
        self._control_mode_request_sub = self.create_subscription(
            Bool, "control/control_mode_request_topic", self._control_mode_request_callback, 1)
        # simple_trajectory_generator publishes with BEST_EFFORT/KEEP_LAST(1) — match it
        # so the subscription is QoS-compatible (rclpy default is RELIABLE).
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._trajectory_sub = self.create_subscription(
            Trajectory, "planning/scenario_planning/trajectory", self._trajectory_callback, trajectory_qos)
        self._stop_request_sub = self.create_subscription(
            Empty, "/control/mpc/stop_request", self._stop_request_callback, 1)
        self._gear_status_sub = self.create_subscription(
            GearReport, "/vehicle/status/gear_status", self._gear_status_callback, 1)
        # 2026-07-27追加(192節続報): アクチュエータ遅延特性の実測診断用。
        self._steering_status_sub = self.create_subscription(
            SteeringReport, "/vehicle/status/steering_status", self._steering_status_callback, 1)

        if self.use_sim_time:
            self._awsim_status_sub = self.create_subscription(
                Float32MultiArray, "/awsim/status", self._awsim_status_callback, 1)
            self._condition_sub = self.create_subscription(
                Int32, "/aichallenge/pitstop/condition", self._condition_callback, 1)

        if self.USE_OBSTACLE_AVOIDANCE:
            if self._cfg.reference_path.use_path_constraints_topic: # type: ignore
                self._path_constraints_sub = self.create_subscription(
                    PathConstraints, "/path_constraints_provider/path_constraints", self._path_constraints_callback, 1)

            if self._cfg.reference_path.use_border_cells_topic: # type: ignore
                self._border_cells_sub = self.create_subscription(
                    BorderCells, "/path_constraints_provider/border_cells", self._border_cells_callback, 1)

            self._v2x_sub = self.create_subscription(
                V2XVehiclePositionArray,
                "/v2x/vehicle_positions",
                self._v2x_callback,
                1)

    def _create_ackerman_control_command(self, stamp, u, acc, bug_acc_enabled):
        v_cmd = u[0]
        steer_cmd = u[1]

        ackerman_cmd = array_to_ackermann_control_command(stamp.to_msg(), [v_cmd, steer_cmd], acc)

        if not self.USE_BUG_ACC:
            return ackerman_cmd

        ackerman_boost_cmd = AckermannControlBoostCommand()
        ackerman_boost_cmd.command = ackerman_cmd
        ackerman_boost_cmd.boost_mode = bug_acc_enabled
        return ackerman_boost_cmd

    def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):
        # 2026-07-22追加(issue④③、_last_u/_last_accの陳腐化対策): 従来はこの2値の
        #   更新が通常の_control()フロー内(旧5070-5071/5069行目)にしかなく、
        #   STUCK復帰(_handle_stuck_recovery)は本メソッドを直接呼ぶだけでこの更新を
        #   バイパスしていた。STUCK復帰完了直後の最初の周期、下流の低域通過フィルタ
        #   (u[1] = _last_u[1] + (u[1]-_last_u[1])*gain)がSTUCK突入前の古い値を基準に
        #   平滑化してしまう不整合があった。「実際に今publishするコマンド」を記録する
        #   唯一の場所をここに一本化し、呼び出し元(通常フロー/STUCK復帰)によらず
        #   _last_u/_last_accが常に最新の実発行値と一致することを構造的に保証する。
        self._last_u[0] = float(u[0])
        self._last_u[1] = float(u[1])
        self._last_acc = float(acc)
        cmd = self._create_ackerman_control_command(stamp, u, acc, bug_acc_enabled)

        # compensate steering angle for the real vehicle
        # AWSIMにおいても後段のactuation_cmd_converter でgainを考慮した指令を生成するため、実機/sim問わず
        # gain を掛ける
        cmd_gained = self._create_ackerman_control_command(stamp, u, acc, bug_acc_enabled)
        cmd_gained.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var

        # 2026-07-27追加(192節続報): 実発行操舵角(gain適用後、実際にアクチュエータへ渡る値)の
        #   時刻付き履歴(STEER-XCORR用)。「この周期でコントローラが意図した値」を記録するため、
        #   下記の実発行タイミング(即時/遅延デバッグ注入)に関わらずここで一度だけ記録する。
        #   stampはrclpy.time.Time型(self.get_clock().now()、_create_ackerman_control_command内の
        #   stamp.to_msg()と同じ型)であり、builtin_interfaces/Timeメッセージとは異なり
        #   .sec/.nanosec属性を持たないため、.nanosecondsプロパティ(ROS epochからのナノ秒、int)を使う
        #   (2026-07-27緊急修正: 初回投入時に.secアクセスでAttributeErrorが発生し
        #   ノードが即死・車両が発進しない致命的な回帰を引き起こしていた)。
        _t_cmd = stamp.nanoseconds * 1e-9
        self._xcorr_steercmd_hist.append((_t_cmd, float(cmd_gained.lateral.steering_tire_angle)))
        _cut = _t_cmd - self._XCORR_WINDOW_S
        while self._xcorr_steercmd_hist and self._xcorr_steercmd_hist[0][0] < _cut:
            self._xcorr_steercmd_hist.pop(0)

        # 2026-07-27追加(デバッグ専用、196節続報): 予選環境で実測した追加遅延(ローカル比+50-60ms)
        #   をローカルで再現し、蛇行が再現するかを検証する実験用フック。既定0.0では以下のifへ
        #   一切入らず、即時publishという従来と完全に同一の経路のみ通る。
        _extra_delay_s = float(getattr(self._mpc_cfg, "debug_extra_actuator_delay_s", 0.0))
        if _extra_delay_s <= 0.0:
            self._command_raw_pub.publish(cmd)
            self._command_pub.publish(cmd_gained)
        else:
            _due = _t_cmd + _extra_delay_s
            self._delayed_cmd_queue.append((_due, cmd, cmd_gained))
            _now_s = self.get_clock().now().nanoseconds * 1e-9
            while self._delayed_cmd_queue and self._delayed_cmd_queue[0][0] <= _now_s:
                _, _raw_due, _gained_due = self._delayed_cmd_queue.pop(0)
                self._command_raw_pub.publish(_raw_due)
                self._command_pub.publish(_gained_due)


    _XCORR_WINDOW_S = 3.5  # [s] センシング切り分け計装D: 相互相関に使う直近窓(±0.3sのラグ探索余裕込み)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        self._xcorr_ekf_hist.append((t, p.x, p.y))
        _cut = t - self._XCORR_WINDOW_S
        while self._xcorr_ekf_hist and self._xcorr_ekf_hist[0][0] < _cut:
            self._xcorr_ekf_hist.pop(0)

    def _gear_status_callback(self, msg: GearReport) -> None:
        self._gear_report = msg

    def _steering_status_callback(self, msg: SteeringReport) -> None:
        """2026-07-27追加(192節続報): 実測操舵角の時刻付き履歴(STEER-XCORR用)。
        _xcorr_ekf_hist等と同じ「毎周期append+古い要素をpop(0)」パターン。"""
        t = msg.stamp.sec + msg.stamp.nanosec * 1e-9
        self._xcorr_steeract_hist.append((t, float(msg.steering_tire_angle)))
        _cut = t - self._XCORR_WINDOW_S
        while self._xcorr_steeract_hist and self._xcorr_steeract_hist[0][0] < _cut:
            self._xcorr_steeract_hist.pop(0)

    _GEAR_LABELS = {0: "NONE", 1: "NEUTRAL", 2: "DRIVE", 20: "REVERSE",
                     21: "REVERSE_2", 22: "PARK", 23: "LOW", 24: "LOW_2"}

    def _gear_label(self, report: int) -> str:
        """GearReport.reportの生値を人間可読なラベルへ変換(2026-07-11、スタック復帰時の
        ギア状態診断用)。DRIVE_2〜18等の中間値は未登録のため raw 表記のみ返す。"""
        return self._GEAR_LABELS.get(int(report), f"raw={report}")

    def _opp_snapshot_str(self) -> str:
        """_v2x_trackerから全相手車の現在位置・速度を1行の文字列へ整形する
        (2026-07-11, [STUCK-PUSH]用に実装したものを[GHOST-BLOCK]と共用するため切り出し)。
        サブスクリプションコールバックは_control()バイパス中も継続更新されるため、
        _handle_stuck_recovery内から呼んでも参照可能。"""
        _tracker = getattr(self, "_v2x_tracker", None)
        if _tracker is None:
            return "none"
        _ids = _tracker.active_vehicle_ids()
        if not _ids:
            return "none"
        _parts = []
        for _vid in _ids:
            _op = _tracker.predict_positions(_vid, [0.0])
            if not _op:
                continue
            _ox, _oy = _op[0]
            _ovx, _ovy = _tracker.velocity(_vid)
            _parts.append(f"{_vid}:x={_ox:.2f},y={_oy:.2f},v={np.hypot(_ovx, _ovy):.2f}")
        return " ".join(_parts) if _parts else "none"

    def _reset_ot_side_for_fresh_replan(self) -> None:
        """側選択のコミットを解除し、次周期のENGAGE判定(_cheap_ok→_plan_pass)が
        壁・相手車の「現在」位置を使って側を自由に再検討できるようにする。
        2026-07-19追加(123節、ユーザー指摘): STUCK復帰(BACKUP/PUSH)は直進のみで
        側の再検討を一切行わないため、復帰後に self._ot_side がSTUCK発生時のまま
        引き継がれ、同じ側へ再突入して同一地点で繰り返しSTUCKする事象を確認した
        (0719-05実測、wp332-333で4回連続のBACKUP+PUSHサイクル)。「前方クリア連続」
        「infeasible」による通常のNORMAL復帰(3548-3554行目等)と全く同じリセット
        セットを再利用する(新規状態変数0個、判定ロジックは_plan_pass側を含め無変更)。"""
        self._ot_state = "NORMAL"
        self._ot_side = 0
        self._ot_side_locked = 0
        self._ot_worth_count = 0
        self._ot_giveup_count = 0
        self._ot_cleared = False
        # 2026-07-24追加(168節): 側を仕切り直す以上、旧側で積み上がっていた
        #   room_exhausted計数・凍結オフセットも持ち越さない(次の側選択と無関係な値)。
        self._ot_room_exhausted_count = 0
        self._ot_last_valid_target_mag = None

    def _stuck_update_shuffle_cycle(self, now, pose) -> None:
        """184節追加(2026-07-26): 新規STUCK検知(WAIT_REVERSE突入)が、直前の
        PUSH完了/復帰断念からshuffle_episode_gap_s以内・shuffle_episode_radius_m
        以内で発生した場合、「同一エピソードの続き(縦列駐車脱出の次の一手)」と
        みなしてself._stuck_shuffle_cycleをインクリメントする。それ以外(十分
        離れている/時間が経っている/初回)はシャッフルサイクルを0へ戻す。
        WAIT_REVERSE突入の全呼び出し元(経路1/2/3)で、状態遷移の直前に呼ぶ。"""
        if (self._stuck_episode_last_end_time is not None
                and self._stuck_episode_last_pose is not None):
            _gap_s = (now - self._stuck_episode_last_end_time).nanoseconds / 1e9
            _dist = float(np.hypot(pose.x - self._stuck_episode_last_pose[0],
                                    pose.y - self._stuck_episode_last_pose[1]))
            if (_gap_s <= self._stuck_shuffle_episode_gap_s
                    and _dist <= self._stuck_shuffle_episode_radius_m):
                self._stuck_shuffle_cycle += 1
                self.get_logger().info(
                    f"[STUCK-SHUFFLE] 直前の復帰完了から{_gap_s:.1f}s/{_dist:.2f}mのため"
                    f"同一エピソード継続とみなす -> cycle={self._stuck_shuffle_cycle}")
                return
        self._stuck_shuffle_cycle = 0
        # 2026-07-26追加(186節続報): 同一地点とみなされない(=離れた/十分時間が経った)
        #   新規エピソードなので、シャッフル上限への到達回数・操舵反転状態も仕切り直す。
        self._stuck_giveup_streak = 0
        self._stuck_push_side_flip = False

    def _stuck_recovery_complete(self, reset_backup_state: bool, reset_corridor: bool,
                                  now=None, pose=None) -> None:
        """スタック復帰完了時の共通後始末(148節②、純粋スリム化)。呼び出し元3箇所
        (BACKUP-BLOCKED断念/BACKUP-TIMEOUT予算超過/PUSH完了)でほぼ同一の後始末
        (NORMAL復帰・側再検討・infeasibility_counterリセット)が個別に書かれていた
        ものを1箇所へ集約した。reset_backup_state/reset_corridorの有無は各呼び出し元の
        従来の挙動をそのまま再現しているだけで、今回は統一・修正していない(挙動不変の
        純粋リファクタ)。
        2026-07-24更新(171節続報): 従来あった4箇所目(WAIT_DRIVE完了、経路1/2専用の
        ステア0固定直進復帰)はPUSHへ統合され到達不能になったため削除し、3箇所になった。
        2026-07-26追加(184節): now/poseが渡された場合、完了時刻・完了位置を記録する。
        次にSTUCKが同じ場所付近・短時間内(shuffle_episode_gap_s/radius_m)で再発した
        場合、_stuck_shuffle_cycleをインクリメントして「同一エピソードの続き(縦列駐車
        脱出の次の一手)」として扱うために使う(STUCKトリガー箇所側で参照)。"""
        self._stuck_state = "NORMAL"
        self._stuck_trigger_path = None
        self._ot_returning = True
        self._reset_ot_side_for_fresh_replan()  # 2026-07-19追加(123節)
        if now is not None and pose is not None:
            self._stuck_episode_last_end_time = now
            self._stuck_episode_last_pose = (pose.x, pose.y)
        if reset_backup_state:
            self._stuck_backup_zero_v_since = None
            self._stuck_backup_first_timeout_time = None
            self._stuck_backup_budget_exhausted_logged = False
        if reset_corridor:
            # 2026-07-16追加(80節): コリドー境界のラチェット解除。
            self._reference_path.reset_dynamic_constraints()
            self.get_logger().info(
                "[CORRIDOR-RESET] stuck-recovery完了によりコリドー境界ラチェットを解除")
        # 2026-07-17追加(90節): 復帰処理中はget_control()が一切呼ばれないため、
        #   infeasibility_counterが復帰開始前の値(しばしば既に閾値300超)のまま
        #   凍結される。リセットしないと、NORMAL復帰直後の最初の周期でMPCが
        #   一度も再実行される前に経路2(infeasibility)が凍結値のまま即座に
        #   再発火する(0717-01実測: 復帰完了の14ms後に再検知、207回連鎖・
        #   332秒間MPCが一度も再稼働しないまま終了)。復帰完了の瞬間に
        #   0へ戻し、既存の300周期(≈7.5秒)閾値の意味(復帰後どれだけ実際に
        #   解けなかったか)を守る。
        self._mpc.infeasibility_counter = 0
        self.get_logger().info(
            "[STUCK-COUNTER-RESET] 復帰完了によりinfeasibility_counterを0へリセット")

    def _fresh_gap_target(self, x: float, y: float, psi: float, prev_idx: int):
        """184節追加(2026-07-26): 壁+相手車(occupancy格子経由で既に統合済み)を
        踏まえた「空き区間の中心」を、STUCK中でも新鮮な自己位置から直接計算する。

        MPC.get_control()はSTUCK中(_stuck_state != "NORMAL")は一切呼ばれないため、
        PUSH開始時に参照していたdbg_corr_ub_arr/lb_arr(147/168節)はBACKUP開始前の
        位置で固まった陳腐化データのままだった。本関数はreference_path.
        update_path_constraints(MPC._corridor()が内部で呼ぶのと同一関数、QP非依存)を
        新鮮なwp_id・自己姿勢で単発呼び出しし、壁+相手車統合済みの空き区間候補から
        最大幅のものを直接得る(新規の空き区間探索アルゴリズムは増やさない)。

        戻り値: (wp_id, target_e_y, wp_psi, width[=ub-lb])。全区間ふさがっている
        (ub<=lb)、または計算不能な場合はNone(呼び出し側は既存のフォールバックへ)。"""
        try:
            wp_id, _ = self._closest_wp_and_s(x, y, prev_idx=prev_idx)
            sm = (self._mpc.safety_margin_override
                  if self._mpc.safety_margin_override is not None
                  else self._mpc.model.safety_margin)
            ub, lb, _ = self._reference_path.update_path_constraints(
                wp_id, [x, y, psi], self._stuck_gap_lookahead_n,
                self._mpc.model.length, self._mpc.model.width, sm)
            if ub is None or lb is None or len(ub) == 0 or len(lb) == 0:
                return None
            u0, l0 = float(ub[0]), float(lb[0])
            if u0 <= l0:
                return None
            wp = self._reference_path.waypoints[wp_id]
            return wp_id, (u0 + l0) / 2.0, float(wp.psi), (u0 - l0)
        except Exception:
            return None

    def _stuck_target_steer(self, target_e_y: float, wp_psi: float, cur_e_y: float,
                             cur_psi: float, reverse: bool) -> float:
        """184節追加: 隙間中心(target_e_y)・経路接線(wp_psi)への単純な比例操舵則。

        後退中(reverse=True)の符号反転根拠: kinematic bicycle modelの
        psi_dot = v/L*tan(delta)(spatial_bicycle_models.py既存の式)には後退用の
        特別扱いが無く、v<0でもそのまま成り立つ。よって後退中は同じdelta符号でも
        ヨーの変化方向が前進時と逆になるため、前進基準で計算した操舵角の符号を
        そのまま反転して適用する(実車のバック操作と同じ直感、mpc_controller.py側の
        array_to_ackermann_control_commandもsteering_tire_angleを符号反転なしで
        そのまま送っており、ギアに応じた変換は行っていないことを確認済み)。"""
        e_y_err = target_e_y - cur_e_y
        psi_err = float(np.arctan2(np.sin(wp_psi - cur_psi), np.cos(wp_psi - cur_psi)))
        delta = self._stuck_ey_kp * e_y_err + self._stuck_psi_kp * psi_err
        delta_max = np.deg2rad(self._stuck_push_steer_max_deg)
        delta = float(np.clip(delta, -delta_max, delta_max))
        return -delta if reverse else delta

    def _rear_clearance_m(self, x: float, y: float, wp_id: int) -> float:
        """184節追加: 後退開始前に、後方の相手車との距離から安全な後退距離の
        上限を求める(ユーザー指摘「後退時に後ろの壁への衝突を確認しているか」への
        対処)。_scan_traffic()のcarsリストは前方偏重(-along_min_length<dsのみ、
        すなわち後方はカート全長未満しか含まない)で後退距離(既定2.0m)の
        安全確認には転用できないため、V2Xトラッカーを直接走査する。壁自体の
        湾曲はbackup_dist程度の距離では無視できるとみなし、対象は他車のみとする
        (壁形状は_fresh_gap_targetが隙間中心を狙う際に既に考慮している)。

        戻り値: 安全な後退距離[m](self._stuck_backup_distを上限とする)。"""
        limit = self._stuck_backup_dist
        tracker = getattr(self, "_v2x_tracker", None)
        if tracker is None:
            return limit
        try:
            total = self._reference_path.length
            s_self = float(self._wp_s_cum[wp_id])
        except Exception:
            return limit
        lat_band = self._ot_max_width / 2.0 + self._ot_block_half
        for vid in tracker.active_vehicle_ids():
            try:
                pos = tracker.predict_positions(vid, [0.0])
                if not pos:
                    continue
                cx, cy = pos[0]
                wp_i, s_obs = self._closest_wp_and_s(
                    cx, cy, prev_idx=self._wp_match_prev.get(vid))
                self._wp_match_prev[vid] = wp_i
            except Exception:
                continue
            wp = self._reference_path.waypoints[wp_i]
            lat = float(np.cos(wp.psi) * (cy - wp.y) - np.sin(wp.psi) * (cx - wp.x))
            ds = (s_obs - s_self + total / 2.0) % total - total / 2.0
            if ds >= 0.0 or ds < -self._stuck_rear_scan_max_dist_m:
                continue
            if abs(lat) > lat_band:
                continue
            room = max(0.0, -ds - self._stuck_rear_clearance_margin_m)
            limit = min(limit, room)
        return max(0.0, limit)

    def _compute_stuck_push_steer(self, pose) -> float:
        """PUSH開始時の操舵角[rad]を1回だけ決める(148節②)。ユーザー提案(「上流から下流まで
        同じ計算式・同じ値を使うべき」)を受け、_ot_state側が既にENGAGE判定で使っている
        _plan_pass(scan, prefer_side)をそのまま呼び、側(plan_side)を決定に使う(新規の
        側選択式は作らない)。動的コリドー先読み(_corr_bound_ahead、147節で新設済み)は
        「これ以上は超えない安全上限」としてのみ使う——BACKUP後は自車位置がMPC最終ソルブ時
        から数m動いているため、この値自体が陳腐化している可能性があるが、上限としてのみ
        使う限り陳腐化はより保守的(操舵量が小さくなる)方向にしか作用しない。
        plan_okがFalse/plan_side==0(相手不明・側判断不能)の場合は、壁マージン比較の
        フォールバックへ移る(下記参照)。
        PUSHの実行自体(速度・タイムアウト・距離)は引き続きMPC非依存の単純な開ループのまま
        変更しない(PUSHはMPCが手詰まりになった時の脱出ルートという役割を守るため)。
        2026-07-24追加(171節続報): 決定した側をself._stuck_push_sideへ保存し、PUSH中の
        「実際に回避できたか」の再判定(_corr_bound_ahead再評価)で使い回せるようにする
        (側選択の式・呼び出しをここ1箇所に留め、PUSH側は保存された結果を読むだけにする)。
        2026-07-24追加(172節続報、ユーザー実測): 0724-03予選ログで、PUSH中の側決定が
        41/41回全てplan_ok=False(側不明)→steer=0.0(直進)のままだったと判明した。原因は
        _plan_pass()が相手車を避けるためのgate2ロジックであり、STUCKの原因が相手車でなく
        「壁に向いている」ことそのものだった場合(実測: 最寄りの相手車でも19〜27m先、
        前方判定圏外)は正しくplan_ok=Falseを返すため。相手車ベースの判定が失敗した場合に
        限り、既存の_corr_bound_ahead()(147/168節で既出、新規の空き幅計算式は増やさない)
        で左右の壁マージンを直接比較し、広い側へ操舵するフォールバックを追加する。

        2026-07-26追加(184節、ユーザー提案「隙間の中心へ向ける」): 上記のいずれよりも
        優先して、_fresh_gap_target()(壁+相手車を統合した新鮮なコリドー中心)を試す。
        取得できた場合はそれを最優先で使い、以下の_plan_pass/壁マージン比較は
        _fresh_gap_targetが失敗した場合(全区間ふさがり等)の既存フォールバックとして残す。"""
        try:
            _v_odom = abs(self._odom.twist.twist.linear.x)
        except Exception:
            _v_odom = 0.0
        _idx, _ = self._closest_wp_and_s(pose.x, pose.y, prev_idx=int(self._mpc.model.wp_id))
        _wp = self._reference_path.waypoints[_idx]
        _cur_ey = float(np.cos(_wp.psi) * (pose.y - _wp.y)
                        - np.sin(_wp.psi) * (pose.x - _wp.x))
        _gap = self._fresh_gap_target(pose.x, pose.y, pose.theta, _idx)
        if _gap is not None:
            _gap_wp_id, _target_ey, _gap_wp_psi, _gap_width = _gap
            _steer = self._stuck_target_steer(_target_ey, _gap_wp_psi, _cur_ey,
                                               pose.theta, reverse=False)
            _diff = _target_ey - _cur_ey
            self._stuck_push_side = 1 if _diff > 1e-3 else (-1 if _diff < -1e-3 else 0)
            self.get_logger().info(
                f"[STUCK-PUSH-GAP] target_ey={_target_ey:.2f} cur_ey={_cur_ey:.2f} "
                f"width={_gap_width:.2f} steer={np.rad2deg(_steer):.1f}deg")
            return _steer
        _scan = self._scan_traffic(_v_odom, _cur_ey)
        _plan_ok, _plan_side, _plan_req = self._plan_pass(_scan, self._ot_side)
        if not _plan_ok or _plan_side == 0:
            # 壁マージンフォールバック(172節続報): 相手車が無い/遠い場合でも、
            #   左右どちらの壁側により空きがあるかだけは既存のcorr_bound_ahead()から
            #   分かる。両側の差が僅少(_along_min_widthの1/10未満)な場合は方向を誤判定
            #   するリスクの方が大きいため、従来通り直進(0.0)のままにする。
            _room_left = self._corr_bound_ahead(1)
            _room_right = self._corr_bound_ahead(-1)
            if (np.isfinite(_room_left) and np.isfinite(_room_right)
                    and abs(_room_left - _room_right) > self._along_min_width * 0.1):
                _wall_side = 1 if _room_left > _room_right else -1
                self._stuck_push_side = _wall_side
                self.get_logger().info(
                    f"[STUCK-PUSH-WALL-FALLBACK] 相手車ベースの側判定不能のため壁マージン比較"
                    f"(left={_room_left:.2f} right={_room_right:.2f})でside={_wall_side}を採用")
                _room = max(0.0, max(_room_left, _room_right))
                _scale = min(1.0, _room / self._stuck_push_steer_room_ref)
                _mag = np.deg2rad(self._stuck_push_steer_max_deg) * _scale
                return float(_mag if _wall_side > 0 else -_mag)
            self._stuck_push_side = 0
            return 0.0
        self._stuck_push_side = _plan_side
        _room = max(0.0, self._corr_bound_ahead(_plan_side))
        _scale = min(1.0, _room / self._stuck_push_steer_room_ref)
        _mag = np.deg2rad(self._stuck_push_steer_max_deg) * _scale
        return float(_mag if _plan_side > 0 else -_mag)

    def _handle_stuck_recovery(self, now, pose) -> None:
        """スタック復帰の状態機械(WAIT_REVERSE/BACKUP/WAIT_DRIVE/WAIT_DRIVE_PUSH/PUSH)。
        2026-07-09追加、2026-07-10改修(起動猶予/infeasibility経路/ギア確認/経路3+PUSH)。
        後退値はAutoware公式サンプル(autoware_practice_course/backward.cpp)準拠(ユーザー承認済み)。
        2026-07-10: 実測で/vehicle/status/gear_statusがこの環境では配信されず、GearReport確認への
        ハード依存(未確認ならタイムアウトで中断)だと永久にBACKUPへ進めなかった(かつ中断時にDへ
        戻すコマンドも送っていなかったため、誤発動後ギアが不整合のまま残った)。
        「GearReportで確認できれば即進む、確認できなくても固定周期(gear_settle_cycles)経過で進める」
        方式に変更し、gear_status配信の有無に関わらず必ず状態が進行するようにする。
        2026-07-24再設計(171節続報、ユーザー指示): 従来は経路3(完全停止デッドロック)の
        時だけBACKUP後にWAIT_DRIVE_PUSH→PUSHへ分岐し、それ以外(経路1/2)はステア0固定の
        まま直進復帰していた(168節で見つけたSTUCK再発バグの温床)。「BACKUPで下がった直後は
        経路によらず必ず眼前に障害物または壁がある」というユーザー指摘を受け、経路によらず
        必ずWAIT_DRIVE_PUSH→PUSHを経由するよう統一する。PUSH自体も低速・最大舵角
        (delta_max_deg上限、_compute_stuck_push_steer参照)で回避走行を続け、実際に前方が
        クリアになったこと(_corr_bound_ahead再評価)を検知したらNORMALへ復帰する
        (固定距離/タイムアウトは実際に避けられなかった場合の安全側バックストップとして残す)。"""
        gear_cmd = GearCommand()
        gear_cmd.stamp = now.to_msg()

        if self._stuck_state == "WAIT_REVERSE":
            gear_cmd.command = GearCommand.REVERSE
            self._gear_cmd_pub.publish(gear_cmd)
            self._stuck_gear_wait_count += 1
            _confirmed = (self._gear_report.report == GearReport.REVERSE)
            if _confirmed or self._stuck_gear_wait_count >= self._stuck_gear_settle_cycles:
                # 2026-07-11診断追加: 「未確認のまま進む」場合、実際のギア値を記録する。
                #   追突後にPレンジのまま発進しなくなる事象の原因切り分け用
                #   (0.5秒タイムアウトでBACKUPへ進むが、実車が本当にREVERSEへ入ったかは
                #   この時点では未検証のため)。
                self.get_logger().warn(
                    f"[STUCK] gear=REVERSE {'confirmed' if _confirmed else '未確認だが規定周期経過'} "
                    f"(実際のgear_report={self._gear_label(self._gear_report.report)}) -> BACKUP")
                self._stuck_state = "BACKUP"
                self._stuck_backup_start = (pose.x, pose.y)
                self._stuck_backup_start_time = now  # 2026-07-13追加: ウォッチドッグ用
                self._stuck_gear_wait_count = 0
                self._stuck_backup_log_count = 0
                self._stuck_backup_zero_v_since = None  # 2026-07-14追加: 後退不能検知を仕切り直す
                # 184節追加(2026-07-26): 後退距離を、後方の相手車から見た安全マージンで
                #   キャップする(壁リカバリーoff化(183節)により、単純な直進バックでは
                #   同じ壁へ再突入しやすくなったための対処。壁自体の湾曲はbackup_dist
                #   程度の距離では無視できるとみなし、対象は他車のみ)。
                _idx0, _ = self._closest_wp_and_s(
                    pose.x, pose.y, prev_idx=int(self._mpc.model.wp_id))
                self._stuck_backup_dist_eff = self._rear_clearance_m(pose.x, pose.y, _idx0)
                if self._stuck_backup_dist_eff < self._stuck_backup_dist:
                    self.get_logger().info(
                        f"[STUCK-BACKUP-REAR-LIMIT] 後方の相手車を考慮し後退距離を"
                        f"{self._stuck_backup_dist:.2f}m->{self._stuck_backup_dist_eff:.2f}mへ制限")
                u = [self._stuck_backup_speed, 0.0]
                acc = self._stuck_backup_accel
            else:
                u = [0.0, 0.0]
                acc = self._stuck_hold_accel

        elif self._stuck_state == "BACKUP":
            gear_cmd.command = GearCommand.REVERSE
            self._gear_cmd_pub.publish(gear_cmd)
            dist = float(np.hypot(pose.x - self._stuck_backup_start[0],
                                   pose.y - self._stuck_backup_start[1]))
            # 2026-07-11診断追加: BACKUP中、実際にREVERSEへ入っているか・実車速はどうかを
            #   間引きログする。「Pレンジのまま発進しない」事象が再発した場合、この行で
            #   実ギア値がREVERSE以外に固定されているかを確認できる。
            if self._stuck_backup_log_count % 10 == 0:
                self.get_logger().info(
                    f"[STUCK-BACKUP] gear_report={self._gear_label(self._gear_report.report)} "
                    f"v={self._odom.twist.twist.linear.x:.2f} dist={dist:.2f}")
            self._stuck_backup_log_count += 1
            _backup_elapsed = (now - self._stuck_backup_start_time).nanoseconds / 1e9
            _v_now = self._odom.twist.twist.linear.x
            # 184節追加(2026-07-26): 隙間の中心へ先頭を向けるための後退中クローズド
            #   ループ操舵。毎周期、新鮮な自己位置から隙間中心(_fresh_gap_target)を
            #   計算し直し、比例操舵則(_stuck_target_steer、reverse=True=後退時の
            #   符号反転込み)で目標操舵角を求める。隙間が計算できない場合は従来通り
            #   直進(0.0)にフォールバックする(安全側、挙動後退なし)。
            _idx_now, _ = self._closest_wp_and_s(
                pose.x, pose.y, prev_idx=int(self._mpc.model.wp_id))
            _wp_now = self._reference_path.waypoints[_idx_now]
            _cur_ey_now = float(np.cos(_wp_now.psi) * (pose.y - _wp_now.y)
                                 - np.sin(_wp_now.psi) * (pose.x - _wp_now.x))
            _gap_now = self._fresh_gap_target(pose.x, pose.y, pose.theta, _idx_now)
            if _gap_now is not None:
                _, _target_ey_now, _gap_wp_psi_now, _ = _gap_now
                _backup_steer = self._stuck_target_steer(
                    _target_ey_now, _gap_wp_psi_now, _cur_ey_now, pose.theta, reverse=True)
            else:
                _backup_steer = 0.0
            # 後退不能検知(2026-07-14追加): 実速度が閾値未満のまま一定時間続くのは、
            #   後退方向に障害物がありそもそも物理的に動けないという直接的な証拠。
            #   0713-05実測で、この状態のまま既存のdist/timeout判定(5秒毎の再試行)を
            #   42回以上繰り返し、進捗距離が1.42m→0.00mへ単調悪化するだけで一度も
            #   回復しなかった(リトライ予算600秒を消費し尽くすまで無理に押し続けていた)。
            #   同じ後退を再試行しても結果は変わらないため、確定次第すぐに無理な後退を
            #   やめてNORMALへ委譲する(ユーザー指示: 障害物がある場合は無理にバックせず停止)。
            if abs(_v_now) < self._stuck_backup_blocked_v_thr:
                if self._stuck_backup_zero_v_since is None:
                    self._stuck_backup_zero_v_since = now
                _zero_v_elapsed = (now - self._stuck_backup_zero_v_since).nanoseconds / 1e9
            else:
                self._stuck_backup_zero_v_since = None
                _zero_v_elapsed = 0.0
            if dist >= self._stuck_backup_dist_eff:
                # 2026-07-24変更(171節続報): 経路によらず必ずPUSH(低速+最大舵角の回避走行)を
                #   経由する(旧: 経路3のみWAIT_DRIVE_PUSH、経路1/2はWAIT_DRIVEでステア0固定復帰)。
                # 2026-07-26変更(184節): 閾値をbackup_dist_eff(後方の相手車を考慮した
                #   実効後退距離、BACKUP開始時に_rear_clearance_mで決定済み)へ変更。
                _next = "WAIT_DRIVE_PUSH"
                self.get_logger().warn(
                    f"[STUCK] backup {dist:.2f}m/{self._stuck_backup_dist_eff:.2f}m done -> {_next}")
                self._stuck_state = _next
                self._stuck_gear_wait_count = 0
                # 2026-07-13追加: BACKUP成功=このepisodeは解消されたとみなし、リトライ予算の
                #   起点をリセットする(既存の経路3 stall_first_trigger_timeと同じ考え方)。
                self._stuck_backup_first_timeout_time = None
                self._stuck_backup_budget_exhausted_logged = False
                self._stuck_backup_zero_v_since = None
                u = [0.0, 0.0]
                acc = self._stuck_hold_accel
            elif _zero_v_elapsed >= self._stuck_backup_blocked_confirm_s:
                # 2026-07-26変更(184節、ユーザー提案「縦列駐車から抜け出すように、
                #   短い後退→隙間への微調整前進を繰り返す」): 後方がほぼ塞がっていて
                #   ごく僅かしか後退できない状況は、壁リカバリーoff化(183節)後は
                #   むしろ通常のケースとして想定する。シャッフル上限に達するまでは
                #   即断念せず、今の到達分でPUSHへ進み、隙間の中心を狙って微調整
                #   前進する(このBACKUP↔PUSHの反復回数は_stuck_shuffle_cycleで
                #   カウントし、STUCK再検知箇所(_handle_stuck_recovery呼び出し元)で
                #   インクリメントする)。上限到達後は従来通りNORMALへ委譲する
                #   (無限ループ回避の安全側バックストップ)。
                if self._stuck_shuffle_cycle < self._stuck_shuffle_max_cycles:
                    _next = "WAIT_DRIVE_PUSH"
                    self.get_logger().warn(
                        f"[STUCK-BACKUP-BLOCKED-SHUFFLE] 後退方向がほぼ塞がっている"
                        f"(実速度{_v_now:.3f}m/sが{_zero_v_elapsed:.1f}s継続、dist={dist:.2f}m) "
                        f"cycle={self._stuck_shuffle_cycle}/{self._stuck_shuffle_max_cycles} -> {_next}")
                    self._stuck_state = _next
                    self._stuck_gear_wait_count = 0
                    self._stuck_backup_first_timeout_time = None
                    self._stuck_backup_budget_exhausted_logged = False
                    self._stuck_backup_zero_v_since = None
                    u = [0.0, 0.0]
                    acc = self._stuck_hold_accel
                else:
                    # 2026-07-26追加(186節続報): シャッフル上限に到達しても、まだ
                    #   反転リトライの余地(max_giveup_streak)があれば即座には完全
                    #   断念しない。挟まれ方が非対称な場合、PUSHの操舵方向を反転
                    #   させるだけで抜けられる可能性があるため、シャッフルカウンタを
                    #   仕切り直してもう一巡だけ試す。既存の物理妥当性判定
                    #   (backup_blocked_v_thr/confirm_s)は無変更、候補方向を
                    #   1つ増やすだけで安全弁の緩和は行わない。
                    self._stuck_giveup_streak += 1
                    if self._stuck_giveup_streak <= self._stuck_max_giveup_streak:
                        self._stuck_push_side_flip = not self._stuck_push_side_flip
                        self._stuck_shuffle_cycle = 0
                        _next = "WAIT_DRIVE_PUSH"
                        self.get_logger().warn(
                            f"[STUCK-BACKUP-BLOCKED] 後退方向に障害物(実速度{_v_now:.3f}m/sが"
                            f"{_zero_v_elapsed:.1f}s継続、閾値{self._stuck_backup_blocked_v_thr:.2f}m/s未満) "
                            f"シャッフル上限({self._stuck_shuffle_max_cycles}回)到達 "
                            f"-> 操舵方向を反転しシャッフルを再試行 "
                            f"(giveup_streak={self._stuck_giveup_streak}/{self._stuck_max_giveup_streak}) -> {_next}")
                        self._stuck_state = _next
                        self._stuck_gear_wait_count = 0
                        self._stuck_backup_first_timeout_time = None
                        self._stuck_backup_budget_exhausted_logged = False
                        self._stuck_backup_zero_v_since = None
                        u = [0.0, 0.0]
                        acc = self._stuck_hold_accel
                    else:
                        self.get_logger().warn(
                            f"[STUCK-BACKUP-BLOCKED] 後退方向に障害物(実速度{_v_now:.3f}m/sが"
                            f"{_zero_v_elapsed:.1f}s継続、閾値{self._stuck_backup_blocked_v_thr:.2f}m/s未満) "
                            f"シャッフル上限({self._stuck_shuffle_max_cycles}回)×"
                            f"反転リトライ上限({self._stuck_max_giveup_streak}回)双方に到達 "
                            f"-> 無理に後退せず停止し復帰断念、NORMAL(通常のMPC/ICC)へ委譲")
                        self._stuck_recovery_complete(reset_backup_state=True, reset_corridor=False,
                                                       now=now, pose=pose)
                        u = [0.0, 0.0]
                        acc = 0.0
            elif _backup_elapsed >= self._stuck_backup_timeout_s:
                # ウォッチドッグ(2026-07-13追加): 0713-03実測で、REVERSEギア確認済みにも
                #   関わらず実速度がv≈0のまま500秒以上(1500周期以上)固まり続け、ログの
                #   終わりまで一度も回復しなかった事例を確認した。PUSH(push_timeout_s)や
                #   経路3(stall_retry_budget_s)には既に実時間ウォッチドッグがあるのに、
                #   BACKUP自体には無く、この穴が無限固着の直接原因だった。同じ設計パターン
                #   (実時間で打ち切り+合計リトライ予算)をBACKUPにも追加する。
                if self._stuck_backup_first_timeout_time is None:
                    self._stuck_backup_first_timeout_time = now
                _budget_elapsed = (now - self._stuck_backup_first_timeout_time).nanoseconds / 1e9
                if _budget_elapsed <= self._stuck_backup_retry_budget_s:
                    self.get_logger().warn(
                        f"[STUCK-BACKUP-TIMEOUT] {_backup_elapsed:.1f}s経過も未到達"
                        f"(dist={dist:.2f}m/{self._stuck_backup_dist:.2f}m) "
                        f"予算消費={_budget_elapsed:.1f}s/{self._stuck_backup_retry_budget_s:.0f}s "
                        f"-> WAIT_REVERSE再試行")
                    self._stuck_state = "WAIT_REVERSE"
                    self._stuck_gear_wait_count = 0
                    u = [0.0, 0.0]
                    acc = self._stuck_hold_accel
                else:
                    if not self._stuck_backup_budget_exhausted_logged:
                        self._stuck_backup_budget_exhausted_logged = True
                        self.get_logger().warn(
                            f"[STUCK-BACKUP-TIMEOUT] リトライ予算(backup_retry_budget_s="
                            f"{self._stuck_backup_retry_budget_s:.0f}s)を超過。"
                            f"復帰断念 -> NORMAL(通常のMPC/ICCへ委譲)")
                    self._stuck_recovery_complete(reset_backup_state=False, reset_corridor=False,
                                                   now=now, pose=pose)
                    u = [0.0, 0.0]
                    acc = 0.0
            else:
                u = [self._stuck_backup_speed, _backup_steer]
                acc = self._stuck_backup_accel

        elif self._stuck_state == "WAIT_DRIVE_PUSH":
            gear_cmd.command = GearCommand.DRIVE
            self._gear_cmd_pub.publish(gear_cmd)
            self._stuck_gear_wait_count += 1
            _confirmed = (self._gear_report.report == GearReport.DRIVE)
            if _confirmed or self._stuck_gear_wait_count >= self._stuck_gear_settle_cycles:
                self.get_logger().warn(
                    f"[STUCK] gear=DRIVE {'confirmed' if _confirmed else '未確認だが規定周期経過'} -> PUSH")
                self._stuck_state = "PUSH"
                self._stuck_gear_wait_count = 0
                self._stuck_push_start = (pose.x, pose.y)
                self._stuck_push_start_time = now
                self._stuck_push_log_count = 0
                # 2026-07-21追加(148節②): PUSH開始時に1回だけ操舵角を決める(_ot_side/
                #   _plan_passと同じ判断基準を参照、詳細は_compute_stuck_push_steer参照)。
                self._stuck_push_steer = self._compute_stuck_push_steer(pose)
                # 2026-07-26追加(186節続報): シャッフル上限到達後の反転リトライ中
                #   (_stuck_push_side_flip=True)は、_compute_stuck_push_steer自体の
                #   計算式は変えず、出力(側候補)だけを反転する。挟まれ方が非対称な
                #   場合、逆方向の方が抜けられる可能性があるための追加候補。
                if self._stuck_push_side_flip:
                    self._stuck_push_steer = -self._stuck_push_steer
                    self._stuck_push_side = -self._stuck_push_side
                self.get_logger().info(
                    f"[STUCK-PUSH-STEER] steer={np.rad2deg(self._stuck_push_steer):.1f}deg "
                    f"(0=side不明/空きなしで従来通り直進)"
                    f"{' [反転リトライ中]' if self._stuck_push_side_flip else ''}")
                u = [self._stuck_push_speed, self._stuck_push_steer]
                acc = self._stuck_push_accel
            else:
                u = [0.0, 0.0]
                acc = self._stuck_hold_accel

        elif self._stuck_state == "PUSH":
            gear_cmd.command = GearCommand.DRIVE
            self._gear_cmd_pub.publish(gear_cmd)
            dist = float(np.hypot(pose.x - self._stuck_push_start[0],
                                   pose.y - self._stuck_push_start[1]))
            elapsed = (now - self._stuck_push_start_time).nanoseconds / 1e9
            v_now = float(self._odom.twist.twist.linear.x)
            # 予選環境での事後検証用(相手車との接触結果を確認できるように)。
            # 2026-07-10訂正: 予選環境のbag録画トピックはcontrol_cmd/clock/acceleration/
            #   kinematic_stateの4つに固定(autostart_orchestrator.param.yaml)で、V2X/相手車
            #   位置トピックは録画されない。よってbagへの依存をやめ、_v2x_trackerから直接
            #   相手車の現在位置・速度を取得しこのログ自体に埋め込む(サブスクリプション
            #   コールバックは_control()バイパス中も継続更新されるため参照可能)。
            _opp_str = self._opp_snapshot_str()
            if self._stuck_push_log_count % 10 == 0:
                self.get_logger().info(
                    f"[STUCK-PUSH] x={pose.x:.2f} y={pose.y:.2f} v={v_now:.2f} "
                    f"dist={dist:.2f} elapsed={elapsed:.1f} opp[{_opp_str}]")
            self._stuck_push_log_count += 1
            # 2026-07-24追加(171節続報、ユーザー指示「回避できたら通常処理に復帰」):
            #   固定距離/タイムアウトだけでなく、実際に選択side方向の先読みコリドー
            #   (_corr_bound_ahead、147/168節で既出)がカート幅超の実マージンまで
            #   回復したかを毎周期再評価する。側不明(push_side==0)の場合はこの判定を
            #   スキップし、従来通り距離/タイムアウトのみに委ねる(安全側)。
            # 2026-07-26変更(184節): corr_bound_ahead()はMPC.get_control()経由でしか
            #   更新されずSTUCK中は陳腐化するため、_fresh_gap_target()による新鮮な
            #   幅+ヨー角一致(経路接線との差がpush_heading_tol_rad未満)の両方を
            #   満たすかを優先判定に使う(向きが直っていないまま終了する既知バグへの
            #   対処)。新鮮な値が得られない場合のみ、従来のcorr_bound_ahead()判定
            #   (陳腐化していてもより保守的な方向にのみ作用する安全網)へ戻る。
            # 2026-07-26追加(186節続報): 下記のコリドー幅+向き一致だけでは、PUSH
            #   開始直後(dist=0.00m、車が全く動いていない)でも成立し得ることを
            #   0726-01local試験で実測した(cycle=4/5でreason=cleared・dist=0.00-
            #   0.01mのまま「回避成功」扱いになりシャッフル回数だけを浪費していた)。
            #   実際に動けたことの直接証拠として、最小移動量も必須条件へ加える。
            _dist_ok = dist >= self._stuck_push_min_dist_for_cleared
            _gap_chk = self._fresh_gap_target(pose.x, pose.y, pose.theta,
                                               int(self._mpc.model.wp_id))
            if _gap_chk is not None:
                _, _, _gap_wp_psi_chk, _gap_width_chk = _gap_chk
                _psi_err_chk = abs(float(np.arctan2(
                    np.sin(_gap_wp_psi_chk - pose.theta), np.cos(_gap_wp_psi_chk - pose.theta))))
                _cleared = (_dist_ok and _gap_width_chk > self._along_min_width
                            and _psi_err_chk < self._stuck_push_heading_tol_rad)
            else:
                _cleared = (_dist_ok and self._stuck_push_side != 0
                            and self._corr_bound_ahead(self._stuck_push_side) > self._along_min_width)
            if dist >= self._stuck_push_dist or elapsed >= self._stuck_push_timeout_s or _cleared:
                _reason = ("cleared" if _cleared
                           else "dist" if dist >= self._stuck_push_dist else "timeout")
                self.get_logger().warn(
                    f"[STUCK] PUSH終了(reason={_reason} dist={dist:.2f}m elapsed={elapsed:.1f}s) "
                    f"-> NORMAL再開")
                self._stuck_recovery_complete(reset_backup_state=False, reset_corridor=True,
                                               now=now, pose=pose)
                u = [0.0, 0.0]
                acc = 0.0
            else:
                u = [self._stuck_push_speed, self._stuck_push_steer]
                acc = self._stuck_push_accel

        self._publish_control_command(now, u, acc, False)

    def _gnss_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._gnss_pose = msg
        p = msg.pose.pose.position
        # 進行方位算出用に位置履歴を保持（直近~3mぶん）
        self._gnss_hist.append((p.x, p.y))
        if len(self._gnss_hist) > 80:
            self._gnss_hist = self._gnss_hist[-80:]
        # センシング切り分け計装D(2026-07-19、118節続報): [GNSS-EKF-XCORR]用の時刻付き履歴。
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._xcorr_gnss_hist.append((t, p.x, p.y))
        _cut = t - self._XCORR_WINDOW_S
        while self._xcorr_gnss_hist and self._xcorr_gnss_hist[0][0] < _cut:
            self._xcorr_gnss_hist.pop(0)

    def _maybe_log_gnss_ekf_xcorr(self) -> None:
        """センシング切り分け計装D(2026-07-19、118節続報): EKF位置とGNSS生位置の
        時系列を相互相関させ、EKFがGNSSに対して何ms先行/遅延しているかを推定する。
        符号規約: lag_ms>0 は EKF(t) ≈ GNSS(t-lag) を意味し、EKFがGNSSより遅延して
        いることを示す(lag_ms<0 はEKFが先行、2026-06-29の手動解析an13.pyと同じ規約)。
        既存の_xcorr_ekf_hist/_xcorr_gnss_histのみを使い、新規購読は行わない。
        2026-07-26追加(ローカル3台走行run_dev3_20260726_171301実測でd1がクラッシュ、
        バグ修正): self._xcorr_ekf_hist/_xcorr_gnss_histは購読コールバック側で
        append/pop(0)により毎周期変化する可変リストであり、それを指すエイリアス
        (ekf_h/gnss_h)から複数のリスト内包表記(ekf_t/ekf_x/ekf_y等)を個別に
        構築していたため、内包表記の間にコールバックが要素を追加/削除すると
        配列長が食い違い、`ekf_x_m = ekf_x[mask]`でIndexError
        (「size of axis is 167 but size of corresponding boolean axis is 168」)
        が発生しノード全体がクラッシュしていた。list()で1回だけ独立コピーを取り、
        以降の全ての内包表記がこの不変スナップショットのみを参照するよう修正する
        (CPythonのlist(list)はC実装でGILを跨がず一括コピーされるため安全)。"""
        ekf_h = list(self._xcorr_ekf_hist)
        gnss_h = list(self._xcorr_gnss_hist)
        if len(ekf_h) < 20 or len(gnss_h) < 10:
            return
        ekf_t = np.array([r[0] for r in ekf_h])
        ekf_x = np.array([r[1] for r in ekf_h])
        ekf_y = np.array([r[2] for r in ekf_h])
        gnss_t = np.array([r[0] for r in gnss_h])
        gnss_x = np.array([r[1] for r in gnss_h])
        gnss_y = np.array([r[2] for r in gnss_h])

        _lag_max = 0.3  # [s] 探索するラグの上下限(既存_XCORR_WINDOW_Sの余裕分と対応)
        _lag_step = 0.02  # [s]
        t_lo = gnss_t[0] + _lag_max
        t_hi = gnss_t[-1] - _lag_max
        mask = (ekf_t >= t_lo) & (ekf_t <= t_hi)
        if int(mask.sum()) < 10:
            return
        ekf_t_m = ekf_t[mask]
        ekf_x_m = ekf_x[mask]
        ekf_y_m = ekf_y[mask]

        lags = np.arange(-_lag_max, _lag_max + 1e-9, _lag_step)
        best_lag = 0.0
        best_resid = None
        resid_at_zero = None
        for lag in lags:
            gx_i = np.interp(ekf_t_m - lag, gnss_t, gnss_x)
            gy_i = np.interp(ekf_t_m - lag, gnss_t, gnss_y)
            resid = float(np.sqrt(np.mean((ekf_x_m - gx_i) ** 2 + (ekf_y_m - gy_i) ** 2)))
            if abs(lag) < 1e-9:
                resid_at_zero = resid
            if best_resid is None or resid < best_resid:
                best_resid = resid
                best_lag = float(lag)

        self.get_logger().info(
            f"[GNSS-EKF-XCORR] lag_ms={best_lag * 1000.0:.0f} "
            f"resid_at_best={best_resid:.3f} resid_at_zero={resid_at_zero:.3f} "
            f"n_ekf={int(mask.sum())} n_gnss={len(gnss_h)}")

    def _maybe_log_steer_xcorr(self) -> None:
        """2026-07-27追加(192節続報、AXIS06アクチュエータ遅延診断): 実発行操舵角(指令)と
        実測操舵角(steering_status)の時系列を相互相関させ、実アクチュエータの遅延[ms]を
        推定する。_maybe_log_gnss_ekf_xcorr()と全く同じ手法(スライド窓+RMS残差最小化)を
        操舵角1軸へ適用したもの。符号規約: lag_ms>0 は actual(t) ≈ cmd(t-lag) を意味し、
        実測操舵角が指令より遅延していることを示す(実アクチュエータが指令に追従する
        向きなので、通常は正の値が出るはず)。既存の_xcorr_steercmd_hist/_xcorr_steeract_hist
        のみを使い、新規購読は行わない。list()で1回だけ独立スナップショットを取ってから
        全ての内包表記を組み立てる(190節xcorr競合状態クラッシュと同じ回避策を最初から適用)。"""
        cmd_h = list(self._xcorr_steercmd_hist)
        act_h = list(self._xcorr_steeract_hist)
        if len(cmd_h) < 20 or len(act_h) < 20:
            return
        cmd_t = np.array([r[0] for r in cmd_h])
        cmd_v = np.degrees(np.array([r[1] for r in cmd_h]))
        act_t = np.array([r[0] for r in act_h])
        act_v = np.degrees(np.array([r[1] for r in act_h]))

        _lag_lo, _lag_hi = -0.05, 0.4  # [s] 実測遅延(約200ms)を包含する探索範囲
        _lag_step = 0.01  # [s]
        t_lo = cmd_t[0] + _lag_hi
        t_hi = cmd_t[-1] - max(_lag_lo, 0.0)
        mask = (act_t >= t_lo) & (act_t <= t_hi)
        if int(mask.sum()) < 10:
            return
        act_t_m = act_t[mask]
        act_v_m = act_v[mask]

        lags = np.arange(_lag_lo, _lag_hi + 1e-9, _lag_step)
        best_lag = 0.0
        best_resid = None
        resid_at_zero = None
        for lag in lags:
            cmd_i = np.interp(act_t_m - lag, cmd_t, cmd_v)
            resid = float(np.sqrt(np.mean((act_v_m - cmd_i) ** 2)))
            if abs(lag) < 1e-9:
                resid_at_zero = resid
            if best_resid is None or resid < best_resid:
                best_resid = resid
                best_lag = float(lag)

        self.get_logger().info(
            f"[STEER-XCORR] lag_ms={best_lag * 1000.0:.0f} "
            f"resid_at_best_deg={best_resid:.2f} resid_at_zero_deg={resid_at_zero:.2f} "
            f"n_cmd={len(cmd_h)} n_act={int(mask.sum())}")

    def _maybe_log_hotspot_deviation(self) -> None:
        """2026-07-27追加(208節続報、AXIS06過操舵ホットスポット監視、Gemini相談):
        ローカル実測+ユーザー目視で過渡応答リンギングが顕著だったwaypoint
        (_HOTSPOT_WPS)通過時、実測舵角のピークとパス自体が要求する理論舵角
        (delta_expected=arctan(kappa_ref×車両長))との最大乖離を計測しログ出力する。
        予選環境で同じ地点が同様の乖離を示すか比較するための診断専用で、制御には
        一切影響しない(Q/R/QP計算を一切変更しない、既存のself._xcorr_steeract_hist
        を読むだけの純粋な観測)。1周回るごとに対象wpを再訪するたび再発火する。"""
        try:
            wp_id = int(self._mpc.model.wp_id)
        except Exception:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9

        if self._hotspot_monitor is not None:
            m = self._hotspot_monitor
            if now_s >= m['end_t']:
                self.get_logger().info(
                    f"[HOTSPOT-DEVIATION] wp={int(m['wp'])} kappa_ref={m['kappa_ref']:.3f} "
                    f"delta_expected={m['theo_deg']:.1f} delta_act_peak={m['peak_act_deg']:.1f} "
                    f"max_dev={m['peak_dev_deg']:.1f}")
                self._hotspot_monitor = None
            elif self._xcorr_steeract_hist:
                act_deg = float(np.degrees(self._xcorr_steeract_hist[-1][1]))
                dev = abs(act_deg - m['theo_deg'])
                if dev > m['peak_dev_deg']:
                    m['peak_dev_deg'] = dev
                    m['peak_act_deg'] = act_deg

        if self._hotspot_monitor is None:
            for hwp in self._HOTSPOT_WPS:
                if abs(wp_id - hwp) <= 1:
                    kappa_ref = float(self._reference_path.waypoints[hwp].kappa)
                    theo_deg = float(np.degrees(np.arctan(kappa_ref * self._mpc.model.length)))
                    self._hotspot_monitor = {
                        'wp': float(hwp), 'end_t': now_s + 2.0,
                        'kappa_ref': kappa_ref, 'theo_deg': theo_deg,
                        'peak_act_deg': 0.0, 'peak_dev_deg': 0.0,
                    }
                    break

    def _gnss_track_heading(self, cx: float, cy: float):
        """直近のGNSS位置履歴から、現在地から track_dist[m] 手前までの進行方位[rad]を返す。
        移動量が小さい(発進直後・停止)場合は None（EKF方位を使う）。
        EKF方位は低速で実方位から約15-19°ずれるため、ピットではこの実進行方位を使う。"""
        h = self._gnss_hist
        if len(h) < 2:
            return None
        need = self._pit_heading_track_dist
        for i in range(len(h) - 1, -1, -1):
            dx = cx - h[i][0]
            dy = cy - h[i][1]
            if (dx * dx + dy * dy) >= need * need:
                return float(np.arctan2(dy, dx))
        return None

    def _apply_pit_localization(self, pose: Pose2D) -> Pose2D:
        """ピット低速走行(on_pit)中は、EKFの位置(壁側へ約0.45m)・方位(約17°)が低速で
        ずれるため、GNSS実測で上書きする：x,y はGNSS位置、theta はGNSS軌跡の進行方位。
        速度はEKFのまま。on_pit=False(gate1/2/レース)では一切作用しない。"""
        if self._pit_enable and self._on_pit and self._gnss_pose is not None:
            p = self._gnss_pose.pose.pose.position
            pose.x = p.x
            pose.y = p.y
            th = self._gnss_track_heading(p.x, p.y)
            if th is not None:
                pose.theta = th
        return pose

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _v2x_callback(self, msg: V2XVehiclePositionArray) -> None:
        # Stage1.5計装: spinスレッド実行だがGIL経由で_controlを止め得るため実測対象
        _t0 = _time.perf_counter()
        try:
            self._v2x_callback_impl(msg)
        finally:
            if hasattr(self, '_pf_acc'):
                self._pf_add('v2x_cb', _time.perf_counter() - _t0)

    def _v2x_callback_impl(self, msg: V2XVehiclePositionArray) -> None:
        self._v2x_tracker.update(msg)
        predictions = self._v2x_tracker.predict_all(self._v2x_t_samples)
        # H1: 後方車の予測円は占有マップに入れない(後方から迫る車の予測円がコリドーを削り
        #   egoを横へ押し出す=過剰回避→壁 の根治)。守り原則: 後方車には自ライン維持。
        #   検知系(_scan_traffic: ICC/守り/ねばり)は独立に後方車を見続ける。
        #   境界フラッピング防止: 投入 ds>-enter / 除外 ds<-exit、間は前回状態を保持。
        headings = {}
        try:
            rp = self._reference_path
            total = rp.length
            s_self = float(self._wp_s_cum[int(self._mpc.model.wp_id)])
            for vid in list(predictions.keys()):
                pts = predictions[vid]
                if not pts:
                    continue
                cx, cy = pts[0]                     # t_samples[0]=0 → 現在位置
                wp_obs, s_obs = self._closest_wp_and_s(
                    cx, cy, prev_idx=self._wp_match_prev.get(vid))
                self._wp_match_prev[vid] = wp_obs
                # 2026-07-20追加(131-6節②、寸法モデルの一元化): 現在位置(t=0)の
                #   円を前後2個へ分割する(predictions_to_obstacles_capsule)ための
                #   進行方向を、on_pit状態に関わらず(オフセット計算自体はピット中も
                #   安全側に働くため)ここで先に確定する。速度が既存opp_obstacle_speed
                #   (障害物クラス閾値の再利用、新規パラメータ0個)未満の停止/低速車は
                #   速度ベクトルが信頼できないため、参照経路接線(wp.psi)へ
                #   フォールバックする。
                _vx, _vy = self._v2x_tracker.velocity(vid)
                _wpo_h = rp.waypoints[wp_obs]
                if np.hypot(_vx, _vy) >= self._opp_obstacle_speed:
                    _heading = float(np.arctan2(_vy, _vx))
                    _heading_src = "velocity"
                else:
                    _heading = float(_wpo_h.psi)
                    _heading_src = "track_tangent"
                headings[vid] = _heading
                _prev_src = self._capsule_heading_src.get(vid)
                if _prev_src != _heading_src:
                    self.get_logger().info(
                        f"[CAPSULE-HEADING] vid={vid} src={_prev_src}->{_heading_src} "
                        f"heading={_heading:.3f} v=({_vx:.2f},{_vy:.2f}) wp={wp_obs}")
                    self._capsule_heading_src[vid] = _heading_src
                # 速度+走行ラインのマップ学習(前方・後方問わず全車)。速度=max包絡線、
                #   ライン=平均横位置(「どちら側が空くか」の実測根拠。相手の外膨らみも入る)。
                #   ピット走行中は学習しない: 経路スワップで wp_obs がピット経路の番号になり、
                #   レース周回マップの誤バケツへ混入する(監査 2026-07-04)。
                if not self._on_pit:
                    try:
                        _wpo = _wpo_h
                        _vl = _vx * np.cos(_wpo.psi) + _vy * np.sin(_wpo.psi)
                        _lo = (np.cos(_wpo.psi) * (cy - _wpo.y)
                               - np.sin(_wpo.psi) * (cx - _wpo.x))
                        self._opp_map.update(vid, wp_obs, float(_vl),
                                             settled=self._v2x_tracker.is_settled(vid),
                                             lat=float(_lo))
                        # 提案②軽量版(2026-07-11): CV外挿(現在速度で一定と仮定)は前走車が
                        #   コーナーで減速する場面を捉えられないことをコードで確認済み
                        #   (v2x_vehicle_tracker.predict_positions は x+vx*t の等速外挿のみ)。
                        #   opp_mapに学習済みの区間速度包絡線があれば、それをs方向に積分した
                        #   予測へ置き換える(新規学習器は追加せず既存opp_mapを転用)。
                        #   横位置(_lo)は短ホライズン(≤0.5s)では概ね不変と仮定して維持し、
                        #   現在点(t=0)は実測値のまま変えない。未学習車はCV外挿にフォールバック。
                        if len(pts) > 1 and self._opp_map.has_data(vid, wp_obs):
                            _new_pts = self._opp_predict_along_path(
                                vid, wp_obs, s_obs, float(_lo), self._v2x_t_samples)
                            _new_pts[0] = pts[0]
                            pts = _new_pts
                            predictions[vid] = pts
                    except Exception:
                        pass
                ds = (s_obs - s_self + total / 2.0) % total - total / 2.0
                prev = self._map_included.get(vid, False)
                thr = -self._rear_map_exit if prev else -self._rear_map_enter
                # 前方遠方も除外(H1の前方版): ホライズン先端の前方車の予測円がコーナー頂点の
                #   コリドーを削り、egoを外へ押して毎周同地点で外壁ヒット(wp14, 2026-07-04)。
                #   縦方向の安全はICCが担保。マップに入れるのは近傍(追い越し対象域)のみ。
                thr_f = self._fwd_map_exit if prev else self._fwd_map_enter
                inc = (thr < ds < thr_f)
                self._map_included[vid] = inc
                if not inc:
                    del predictions[vid]
        except Exception:
            pass  # 起動直後(wp未確定)等は全車保持=従来挙動にフォールバック
        # 2026-07-20追加(131-6節②、寸法モデルの一元化): 現在位置(t=0)のみ前後
        #   capsule化(along_min_length/2 - vehicle_radius)する版へ切替。
        #   headingsは上のループでvidごとに確定済み(未検知vidはheadings.get既定0.0)。
        self._dynamic_obstacles = predictions_to_obstacles_capsule(
            predictions, self._v2x_vehicle_radius, headings,
            self._along_min_length / 2.0)
        self._obstacles_updated = True

    def _opp_predict_along_path(self, vid, wp0, s0, lat_offset, t_samples):
        """opp_mapの位置別速度包絡線(v_pred)でs方向に積分した将来位置を返す
        (提案②軽量版, 2026-07-11)。戻り値はt_samples[0]用のダミー要素(呼び出し元で
        実測値に上書きする前提)を含む同じ長さのリスト。横位置はlat_offsetで固定
        (短ホライズンでの近似)。opp_mapは全wpについて周回補間済みのため、途中で
        未学習binへ入っても速度0への張り付きは起きない。"""
        rp = self._reference_path
        total = rp.length
        n = len(rp.waypoints)
        dt_micro = 0.05
        s = float(s0)
        wp = int(wp0)
        t_prev = 0.0
        out = []
        for t in t_samples:
            remaining = t - t_prev
            while remaining > 1e-6:
                step = min(dt_micro, remaining)
                v = self._opp_map.v_pred(vid, wp)
                s = (s + (v if v is not None else 0.0) * step) % total
                wp = int(np.searchsorted(self._wp_s_cum, s)) % n
                remaining -= step
            wp_pt = rp.waypoints[wp]
            out.append((float(wp_pt.x - np.sin(wp_pt.psi) * lat_offset),
                        float(wp_pt.y + np.cos(wp_pt.psi) * lat_offset)))
            t_prev = t
        return out


    @staticmethod
    def _ds_priority(ds: float) -> float:
        """2026-07-20追加(129節続報、A-2): 「対象車選択」で使う優先度キー。
        _scan_traffic(1813行目付近)・_follow_speed_limit=icc_stop本体(1864行目付近)
        の両方が`ds < best[0]`という生値比較で対象車を選んでおり、ds規約
        (前+/後-)のもとでは後方の車(dsがより負)が前方の車より常に優先される
        という物理的に逆転した選択になっていた。0720-2実測(wp330、d2:ds=1.94
        dlat=1.87/d3:ds=-1.98 dlat=2.36)でd3(後方)が誤選択され、本来
        engage_lat_max(2.0)以内で追従できたはずのd2が無視されてSTOPPING-NO-VSAFE
        (127節)の空白に陥っていたことを確認した。前方(ds>=0)を常に優先し、
        前方候補が無い場合のみ後方をゼロに近い順(along_min_length許容窓内での
        代替)で選ぶ。"""
        return ds if ds >= 0.0 else (1e9 - ds)

    def _dlat_closing_trend(self, fwd_dlat: Optional[float], dlat_v_ema: float,
                             dlat_shrink_run: int,
                             footprint_risk: bool = False) -> bool:
        """2026-07-20追加(141節、フェーズ1)。「この相手との横間隔が縮み続けており
        このままでは何秒で接触するか」を判定する式そのもの(131-6節①/138節で
        ENGAGEゲート専用に導入した式を、他レイヤーからも再利用できるよう1箇所へ
        抽出。計算内容は無変更)。LAT-TTCが既に使うttc_critical_s(0.8秒)・
        min_trend_cycles(3周期)をそのまま再利用する(新規パラメータ0個)。

        2026-07-22追加(issue⑤③): footprint_risk(呼び出し元が_fwd_dlat_val<
        along_min_widthかつfwd_ds<along_min_lengthから毎周期・状態非依存で
        算出済み、127/163節)がTrueの場合は、横方向のトレンド成立条件
        (shrink_run/dlat_v_ema<0)によらず常にTrueを返す。0722-4/5実測で、
        TTC判定がttc_critical_s(0.8秒)をわずかに上回るだけで通過した瞬間、
        fwd_ds/fwd_dlatは既にfootprint_risk相当の物理的接触リスク域に入って
        いたことを確認した(ENGAGE後0.02〜0.04秒でfootprint_risk giveup)。
        本関数の出力(is_closing_trend)はENGAGEゲート(_dlat_ttc_veto)・
        G2-RELEASE(_g2_release_ready)・force_include_vid(ICC近接除外)の
        3箇所で共有されており、いずれも「既に物理的接触リスクがある間は
        より保守的に振る舞う」という同じ意味を持つため、消費箇所ごとに
        個別のor条件を足すのではなく共有元の本関数で1回だけ拡張する。"""
        if footprint_risk:
            return True
        return (dlat_shrink_run >= self._lat_ttc.min_trend_cycles
                and dlat_v_ema < 0.0
                and fwd_dlat is not None
                and (fwd_dlat / max(abs(dlat_v_ema), 1e-6))
                    <= self._lat_ttc.ttc_critical_s)

    def _build_opponent_situation(self, scan, lat_dec,
                                   footprint_risk: bool = False) -> OpponentSituation:
        """2026-07-20追加(141節、フェーズ1)。自車・相手・両者のトレンドに関する
        既存の計算結果を1回だけ集約する読み取り専用スナップショット。
        _dlat_closing_trend(既存式の抽出のみ)を除き新規計算は行わない。
        _control()内で_lat_dec確定直後に1回だけ構築し、以降の判定はここを参照する
        (第一弾はENGAGEゲートのみ移行)。

        2026-07-22追加(issue⑤③): footprint_riskは呼び出し元(_control())が
        _lat_dec確定直前に既に算出済みの値をそのまま受け取って
        _dlat_closing_trendへ渡すのみで、ここでも新規計算は行わない。"""
        _fwd_dlat = scan.get("fwd_dlat")
        _is_closing_trend = self._dlat_closing_trend(
            _fwd_dlat, lat_dec.dlat_v_ema, lat_dec.dlat_shrink_run, footprint_risk)
        # 190-5節(2026-07-26追加、診断専用): is_closing_trendが連続してTrueの周期数と、
        #   footprint_risk起因かトレンド起因かを1箇所で計測する。ENGAGEゲート/
        #   G2-RELEASE/force_include_vidの3消費先いずれかが長時間ブロックされている
        #   場合、この値の継続時間がその根本原因(共有元)の滞留時間を示す。
        if _is_closing_trend:
            self._dlat_trend_true_cycles += 1
            if self._dlat_trend_true_cycles == 1:
                self._dlat_trend_true_via_fp = footprint_risk
        elif self._dlat_trend_true_cycles > 0:
            self.get_logger().info(
                f"[DLAT-TREND-CLEAR] duration={self._dlat_trend_true_cycles / self._mpc_cfg.control_rate:.2f}s "
                f"via_footprint_risk={int(self._dlat_trend_true_via_fp)} "
                f"fwd_vid={scan.get('fwd_vid')} wp={self._mpc.model.wp_id}")
            self._dlat_trend_true_cycles = 0
        return OpponentSituation(
            fwd_vid=scan.get("fwd_vid"),
            fwd_ds=scan.get("fwd_ds"),
            fwd_dlat=_fwd_dlat,
            fwd_vopp=scan.get("fwd_vopp"),
            dlat_v_ema=lat_dec.dlat_v_ema,
            dlat_shrink_run=lat_dec.dlat_shrink_run,
            is_closing_trend=_is_closing_trend)

    def _evaluate_engage_readiness(self, scan, pass_worth, v_odom,
                                    left_ok, right_ok, being_overtaken, lat_dec,
                                    opp_sit, now, footprint_risk: bool = False) -> EngageEval:
        """2026-07-21追加(148節、ENGAGE判定の純粋スリム化フェーズ1)。旧_control()
        インラインの一連の判定(cheap_ok9条件→_plan_pass→dlat_ttc_veto→gate=ログ)を
        そのまま抽出しただけで、計算内容・呼び出し順序・self状態の変更点は一切
        変えていない(回帰テストでソース比較により確認)。
        2026-07-22追加(00節監査、ユーザー指摘「自車/相手情報を共有できるはず」):
        fwd_vopp/fwd_ds/fwd_dlat/fwd_vidは、141節で構築済みのOpponentSituation
        (opp_sit)が同じscanから集約した値をそのまま持っているため、scanから
        個別に読み直す代わりにopp_sit経由で参照するよう統一した(値は完全に
        同一、挙動不変の純粋リファクタ)。scan.get("fwd_wp")のみ、opp_sitに
        該当フィールドが無いためscanから直接読む。
        2026-07-22追加(issue⑤③): footprint_riskは判定ロジックには使わない
        (opp_sit.is_closing_trend側で既に折り込み済み、_dlat_closing_trend参照)。
        [DLAT-TTC-VETO]ログがTTCトレンド起因かfootprint_risk起因かを区別できる
        よう、診断表示専用としてのみ受け取る。"""
        _fwd_vid_worth = opp_sit.fwd_vid
        if _fwd_vid_worth != self._ot_worth_prev_vid:
            self._ot_worth_count = 0
        self._ot_worth_prev_vid = _fwd_vid_worth
        self._ot_worth_count = self._ot_worth_count + 1 if pass_worth else 0

        _on_path = (opp_sit.fwd_dlat is not None
                    and opp_sit.fwd_dlat <= self._ot_engage_lat_max)
        # 190-6節(2026-07-26追加、診断専用): _on_path=Falseの継続時間・その間の
        #   fwd_dlatの変化量(開始時→現在)を計測する。相手が停止中(fwd_vopp)かどうか
        #   も一緒に記録し、「両者静止中なのにfwd_dlatが伸び続けた」ケースを
        #   実地ログから直接確認できるようにする。判定ロジックへの影響なし。
        if not _on_path:
            if self._on_path_false_cycles == 0:
                self._on_path_false_start_dlat = opp_sit.fwd_dlat
            self._on_path_false_cycles += 1
        elif self._on_path_false_cycles > 0:
            self.get_logger().info(
                f"[ON-PATH-CLEAR] duration={self._on_path_false_cycles / self._mpc_cfg.control_rate:.2f}s "
                f"fwd_dlat_start={self._on_path_false_start_dlat} fwd_dlat_end={opp_sit.fwd_dlat} "
                f"fwd_vopp={opp_sit.fwd_vopp} fwd_vid={opp_sit.fwd_vid} "
                f"wp={self._mpc.model.wp_id}")
            self._on_path_false_cycles = 0
            self._on_path_false_start_dlat = None
        _ego_ready = (opp_sit.fwd_vopp is None
                      or opp_sit.fwd_vopp < self._opp_obstacle_speed
                      or v_odom > opp_sit.fwd_vopp - self._engage_ego_margin)
        _closing_est = ((self._v_pot - opp_sit.fwd_vopp)
                         if opp_sit.fwd_vopp is not None else self._v_pot)
        _closing_est = max(_closing_est, self._opp_min_closing)
        _engage_dist_dynamic = min(
            self._fwd_max_consider,
            max(self._ot_engage_max_dist,
                _closing_est * self._ot_t_lateral + self._ot_pass_clear))
        _t_reach_profile = None
        _is_stopped_for_profile = (opp_sit.fwd_vopp is not None
                                    and opp_sit.fwd_vopp < self._opp_obstacle_speed)
        if _is_stopped_for_profile and scan.get("fwd_wp") is not None:
            _t_reach_profile = self._predicted_time_to_wp(
                int(self._mpc.model.wp_id), int(scan["fwd_wp"]), self._fwd_max_consider)
        if _t_reach_profile is not None:
            _t_reach_thr = self._ot_t_lateral + self._ot_pass_clear / _closing_est
            _close_enough = _t_reach_profile <= _t_reach_thr
        else:
            _close_enough = (opp_sit.fwd_ds is not None
                              and opp_sit.fwd_ds <= _engage_dist_dynamic)

        # 2026-07-21追加(148節②): footprint_risk起因のcooldown中は固定タイマー
        # (self._ot_engage_cooldown==0)ではなく、footprint_risk条件自体が
        # engage_debounce(既存のフリッカー防止デバウンス、新規パラメータ0個)周期
        # 連続で不成立になったかで解除する。他の理由のgiveupは従来通り固定タイマーのまま
        # (139節の元の設計意図を維持)。
        # 2026-07-26追加(190-4節): footprint_risk起因のcooldown中、_fp_near_zoneが
        #   相手に追従し続ける限り真になり続け(icc_stopが相手と同一ライン上へ収束させる
        #   ため、fwd_dlatが物理的に0近傍へ張り付く)、_ot_footprint_risk_clear_countが
        #   0からリセットされ続けて永久に解除されない自己ロックを5日分18ログ中5件で確認
        #   (0722-03/0724-01/0724-02/0725-02/0726-02、最長383秒未解決)。
        #   self._ot_engage_cooldownは既にfootprint_risk起因かどうかに関わらず毎周期
        #   無条件でデクリメントされており(139節でfootprint_risk起因時は2倍≈8秒に設定
        #   済み)、単に本判定式が参照していなかっただけだった。デバウンスカウント方式
        #   (148節②、高速解除)はそのまま残し、それが機能しない場合の上限としてこの
        #   既存タイマーをORで追加する(新規パラメータ0個、新規状態変数0個)。
        _cd_clear = (
            (self._ot_footprint_risk_clear_count >= self._ot_engage_debounce
             or self._ot_engage_cooldown == 0)
            if self._ot_footprint_risk_gated
            else self._ot_engage_cooldown == 0)
        # 検証ロギング(148節②): footprint_risk起因のcooldownが「実測解消」経路で
        # 解除された瞬間を1回だけ記録する。従来の固定8秒より早く/遅く解除されたかを
        # 次回ログのengage_cooldown残り周期数(cd_timer_remain)から確認できる。
        if (self._ot_footprint_risk_gated and _cd_clear
                and not self._ot_fp_clear_logged):
            self._ot_fp_clear_logged = True
            self.get_logger().info(
                f"[FP-COOLDOWN-CLEAR] footprint_risk条件が{self._ot_engage_debounce}周期"
                f"連続で不成立となり解除(cd_timer_remain={self._ot_engage_cooldown}周期)")
        _cheap_ok = (self._ot_enable and (left_ok or right_ok)
                     and self._ot_infeasible_latch == 0
                     and _cd_clear
                     and self._ot_worth_count >= self._ot_engage_debounce
                     and _on_path and _ego_ready and _close_enough
                     and not being_overtaken)
        if _cheap_ok:
            _prefer_side = 0
            if (self._ot_prev_side != 0
                    and self._ot_prev_side_vid is not None
                    and self._ot_prev_side_vid == opp_sit.fwd_vid
                    and self._ot_prev_side_time is not None):
                _elapsed = (now - self._ot_prev_side_time).nanoseconds / 1e9
                if _elapsed <= self._ot_side_flip_hyst_s:
                    _prefer_side = self._ot_prev_side
            _plan_ok, _plan_side, _plan_req = self._plan_pass(scan, _prefer_side)
        else:
            _plan_ok, _plan_side = False, 0
            self._dbg_plan_reason = "cheap_ok_fail"
            self._dbg_plan_lf = float('nan')
            self._dbg_plan_rf = float('nan')

        _dlat_ttc_veto = opp_sit.is_closing_trend
        if _plan_ok and _dlat_ttc_veto:
            self._dbg_plan_reason = "dlat_ttc"
        _can_engage = _cheap_ok and _plan_ok and not _dlat_ttc_veto
        _dlat_ttc_veto_effective = _plan_ok and _dlat_ttc_veto
        if _dlat_ttc_veto_effective and not self._dlat_ttc_veto_active:
            self.get_logger().warn(
                f"[DLAT-TTC-VETO] fwd_vid={opp_sit.fwd_vid} "
                f"fwd_dlat={opp_sit.fwd_dlat} "
                f"dlat_v_ema={lat_dec.dlat_v_ema:.3f} "
                f"shrink_run={lat_dec.dlat_shrink_run} "
                f"ttc_critical_s={self._lat_ttc.ttc_critical_s} "
                f"footprint_risk={int(footprint_risk)} "
                f"wp={self._mpc.model.wp_id}")
        if _dlat_ttc_veto_effective:
            self._dlat_ttc_veto_active_cycles += 1
        elif self._dlat_ttc_veto_active:
            # 190-5節: ENGAGEゲートがdlat_ttc_vetoによって実際にブロックされていた
            #   継続時間。3消費先(ENGAGE/G2-RELEASE/force_include_vid)のうちどれが
            #   実際の長時間停止の原因だったかを切り分けるための診断ログ。
            self.get_logger().info(
                f"[DLAT-TTC-VETO-CLEAR] duration={self._dlat_ttc_veto_active_cycles / self._mpc_cfg.control_rate:.2f}s "
                f"fwd_vid={opp_sit.fwd_vid} wp={self._mpc.model.wp_id}")
            self._dlat_ttc_veto_active_cycles = 0
        self._dlat_ttc_veto_active = _dlat_ttc_veto_effective

        _gate = (
            f"lr={int(left_ok or right_ok)}"
            f",lat={int(self._ot_infeasible_latch == 0)}"
            f",cd={int(_cd_clear)}"
            f",wc={int(self._ot_worth_count >= self._ot_engage_debounce)}"
            f",path={int(_on_path)}"
            f",rdy={int(_ego_ready)}"
            f",cls={int(_close_enough)}"
            f",nbo={int(not being_overtaken)}"
            f",plan={int(_plan_ok)}:{getattr(self, '_dbg_plan_reason', '?')}")

        return EngageEval(
            cheap_ok=_cheap_ok, ego_ready=_ego_ready, close_enough=_close_enough,
            on_path=_on_path, plan_ok=_plan_ok, plan_side=_plan_side,
            can_engage=_can_engage, closing_est=_closing_est,
            engage_dist_dynamic=_engage_dist_dynamic, t_reach_profile=_t_reach_profile,
            gate=_gate)

    def _scan_traffic(self, v_ego: float, ego_lat: float):
        """統一検知: トラッカー全車(現在位置+平滑速度)を1回だけ走査し、追い越し判断・ICC追従・
        守り(被追い越し)が共有する事実を返す。旧3系統(analyzer=予測円/帯6.9m、context=帯3.9m、
        ACC=帯1.5m)が別々の答えを出していた矛盾を解消(2026-07-03一元化)。
        規約: ds=前+/後-[-L/2,L/2), lat=参照ライン基準+左/-右, dlat=|lat-ego_lat|=自車との実横間隔。
        戻り dict:
          n_obs: 他車実台数 / cars: 前方(-along_min_length<ds<=max_consider、
            2026-07-19追加(105節/110節、ds>0の崖対策)、|lat|<=コリドー帯)の
            [(ds,lat,v_long,dlat)]
          fwd_ds/lat/vopp/dlat: 最近傍前方車(無ければNone)
          left_free/right_free: 前方車群に対する実壁基準の左右空き幅(最小=律速)
          being_overtaken: 横並び/後方から「走っている」車が接近(G1: 停止・低速車では発動しない)

        2026-07-19追加(105節発見→110節で2環境目再確認、ユーザー承認済み設計):
        前方車判定の下限を厳密な0.0から車両全長分だけ緩和した。静止/低速の相手に
        ごく至近距離まで詰めた場合、自車の弧長位置がわずかに相手を跨いだ瞬間に
        fwd_ds=Noneへ落ち、OFFSET-RETURN(3468行目付近)が「通過完了」と誤判定して
        全開加速する事象(ローカル・予選の2環境で実測確認)への対処。ds<0側でも
        _g2_speed等の式は(ds-margin_center)がより負に振れるため自然に保守的側へ働き、
        安全性が緩む経路は無い(横方向の半幅であるself._ot_block_halfではなく、縦方向
        の量である車両全長を使うのが物理的に正しい、との当時からの設計意図)。
        2026-07-20修正(128節続報): 実装は自転車モデルのホイールベース
        (self._mpc.model.length=1.087、spatial_bicycle_models.py既存の運動学
        パラメータ)を誤って「全長」として使っていた(公式車両仕様全長200cmとの
        乖離を128節で発見)。footprint_risk用に新設した公式全長ベースのalong_min_length
        (2.00)へ置き換え、コメントが述べる本来の設計意図(車両全長)と実装を一致させる。"""
        import math as _math
        out = {"n_obs": 0, "cars": [], "fwd_ds": None, "fwd_lat": None,
               "fwd_vopp": None, "fwd_dlat": None, "fwd_vid": None, "fwd_wp": None,
               "left_free": None, "right_free": None, "being_overtaken": False,
               "along_lat": None, "along_vlong": None, "along_dlat": None,
               "along_vid": None}
        tracker = getattr(self, "_v2x_tracker", None)
        if tracker is None:
            return out
        vids = tracker.active_vehicle_ids()
        if not vids:
            return out
        out["n_obs"] = len(vids)
        rp = self._reference_path
        try:
            total = rp.length
            s_self = float(self._wp_s_cum[int(self._mpc.model.wp_id)])
        except Exception:
            return out
        # 横帯はコリドー幅基準に統一(C4修正: 旧analyzerは max_width+block_half=全幅相当でコース外まで拾っていた)
        lat_band = self._ot_max_width / 2.0 + self._ot_block_half
        best = None
        for vid in vids:
            try:
                pos = tracker.predict_positions(vid, [0.0])   # t=0 → 現在位置
                if not pos:
                    continue
                cx, cy = pos[0]
                wp_i, s_obs = self._closest_wp_and_s(
                    cx, cy, prev_idx=self._wp_match_prev.get(vid))
                self._wp_match_prev[vid] = wp_i
            except Exception:
                continue
            wp = rp.waypoints[wp_i]
            lat = _math.cos(wp.psi) * (cy - wp.y) - _math.sin(wp.psi) * (cx - wp.x)
            vx, vy = tracker.velocity(vid)
            v_long = max(0.0, vx * _math.cos(wp.psi) + vy * _math.sin(wp.psi))
            ds = (s_obs - s_self + total / 2.0) % total - total / 2.0
            dlat = abs(lat - ego_lat)
            # 真横の車(速度不問): 並走ねばり=レーン継続性チェックの対象。
            #   バグ修正(2026-07-04): dlat下限(def_alongside_lat)を追加。下限なしだと「真正面の追従相手」
            #   (|ds|≈margin3m, dlat≈0)も対象になり、狭いコーナーで縦列追従中に誤って強減速していた。
            #   縦列の相手はICCの管轄。ここは「実際に横に並んでいる」車だけを見る。
            #   バグ修正(2026-07-26): abs(ds)だと自分の後方にいるだけの車も対象になり、
            #   「並走ねばり」(相手の縦速度に合わせて減速、下記along_vlong参照)が後方車の
            #   停止/後退につられて自車を不要に停止させていた(実測: 動画42s、後方車のバック
            #   と自車停止が一致)。すぐ下のbeing_overtaken判定(2528行目)は既に
            #   「後方は自分より速い場合のみ」と正しく限定しており、この非対称性の解消として
            #   ds>=0(前方〜真横のみ)に限定する。新規パラメータは追加しない。
            if (0.0 <= ds <= self._def_alongside_dist
                    and self._def_alongside_lat <= dlat <= 3.0):
                if out["along_dlat"] is None or dlat < out["along_dlat"]:
                    out["along_lat"] = lat
                    out["along_vlong"] = v_long
                    out["along_dlat"] = dlat
                    out["along_vid"] = vid
            # 守り(G1): 「走っている」車のみ対象。停止/低速(<obstacle速度)は抜く相手であり
            #   道を譲る相手ではない(駐車車に譲って停止するバグの根治)。
            if v_long >= self._opp_obstacle_speed:
                if abs(ds) <= self._def_alongside_dist and dlat >= self._def_alongside_lat:
                    out["being_overtaken"] = True          # 横並び(実横間隔あり)
                elif -self._def_rear_dist <= ds < 0.0 and v_long > v_ego + self._def_rear_faster:
                    out["being_overtaken"] = True          # 後方から速い車が接近
            # 前方車(コース近傍のみ)
            # 2026-07-20修正(128節続報): self._mpc.model.length(1.087、ホイールベース)は
            #   車両全長ではなかった(公式仕様全長200cmとの乖離を発見)。along_min_length
            #   (公式全長ベース、footprint_risk用に新設)へ置き換え、この行の元々の設計意図
            #   (縦方向の量=車両全長を使う、上記docstring参照)と実装を一致させる。
            if -self._along_min_length < ds <= self._ot_max_consider and abs(lat) <= lat_band:
                # vid/wp_i は ICC予測制動(速度マップ参照)用に添付(2026-07-04)
                out["cars"].append((ds, lat, v_long, dlat, vid, wp_i))
                # 実壁基準の左右空き幅(占有格子由来 wp.ub/lb から他車の横幅を除く)
                lf = max(0.0, float(wp.ub) - (lat + self._ot_block_half))
                rf = max(0.0, (lat - self._ot_block_half) - float(wp.lb))
                # 2026-07-20修正(129節続報、A-2): 生のds比較(ds<best[0])は後方の車を
                #   常に優先する逆転バグだった(_ds_priority docstring参照)。
                if best is None or self._ds_priority(ds) < self._ds_priority(best[0]):
                    best = (ds, lat, v_long, dlat, vid, wp_i)
                    # 対象車(最近傍=実際に抜く相手)自身の空き幅のみを採用(2026-07-08修正)。
                    #   旧実装は前方全車にわたる最小値(プール)を使っており、視野内に2台目の
                    #   車がいると無関係な空き幅が混入して側選択を誤らせていた(実測: 停止車の
                    #   奥に別の車がいる場面で、広い側を無視して狭い側を選び続ける事例を確認)。
                    out["left_free"] = lf
                    out["right_free"] = rf
        if best is not None:
            (out["fwd_ds"], out["fwd_lat"], out["fwd_vopp"], out["fwd_dlat"],
             out["fwd_vid"], out["fwd_wp"]) = best
        return out

    def _g2_speed(self, v_fwd: float, ds: float) -> float:
        """G2式(単独メソッド化、2026-07-12): v = sqrt(max(0, v_fwd² + 2a(ds - margin)))。
        _follow_speed_limitの制動計算本体と、ICCが対象車を見失った際のフォールバック
        (下記_control()のSTOPPING分岐)の両方から呼ぶ。定数(_fwd_a_brake/_fwd_margin_center)
        を1箇所に保ち、将来のチューニングが二重管理にならないようにする。"""
        rad = v_fwd * v_fwd + 2.0 * self._fwd_a_brake * (ds - self._fwd_margin_center)
        return float(np.sqrt(max(0.0, rad)))

    def _follow_speed_limit(self, scan, path_offset: float = 0.0, near_sep: float = None,
                             force_include_vid: str = None):
        """統一ICC: スキャン結果から「衝突コース上の最近傍前方車」を選び追従速度上限を返す。
        対象選び(F1): 近距離(ds<near_range)は実横間隔 dlat<near_sep(=ぶつかる横関係のみ)、
          遠方は進路帯 |lat-path_offset|<=halfwidth(追い越しオフセット分は帯ごとずらす)。
        near_sep: 通常 min_lat_sep(1.8)。パス対象クリア済(Fix-2)は reacquire(1.6)で再接近のみ捕捉。
        force_include_vid(2026-07-15追加、0715-02実測で確認したswitchback直後の見失いバグ対策):
          側反転(switchback)直後はego自身のオフセットがまだ新側へ移動し切っておらず(alpha低)、
          相手のdlatは旧側にいた頃の実測値をそのまま引きずっているため、たまたまnear_sep以上
          あると「もう十分離れた」と誤判定してこの対象を除外し、ICCが空振り(_vlim=None)して
          全開速度へ抜けてしまう(実測: t=434.35, dlat=2.31が旧側の値のままnear_sep(1.8)超過、
          alpha=-0.311相当でほぼ真後ろなのにvsafe=4.166、0.5秒後に衝突)。呼び出し元がalpha未
          到達と判定した場合にのみ、この1台(直前まで並走していた対象車のvid)だけを除外判定
          から免除し、通常の追従計算(g2_speed)へ必ず含める。他の無関係な車には影響しない。
        式(G2): _g2_speed参照。ds=margin で v=v_fwd(等速追従) / ds<margin では v<v_fwd となり
          車間を開け直す(brake-check対応)。
        戻り: (v_safe, target(ds,lat,v_long,dlat))。対象なしは (None, None)。"""
        if near_sep is None:
            near_sep = self._fwd_min_lat_sep
        best = None
        for ds, lat, v_long, dlat, vid, wp_i in scan["cars"]:
            if ds > self._fwd_max_consider:
                continue
            if vid != force_include_vid:
                if ds < self._fwd_near_range:
                    if dlat >= near_sep:
                        continue
                else:
                    if abs(lat - path_offset) > self._fwd_lateral_halfwidth:
                        continue
            # 2026-07-20修正(129節続報、A-2): 生のds比較(ds<best[0])は後方の車を
            #   常に優先する逆転バグだった(_ds_priority docstring参照)。
            if best is None or self._ds_priority(ds) < self._ds_priority(best[0]):
                best = (ds, lat, v_long, dlat, vid, wp_i)
        if best is None:
            return None, None
        ds, v_long = best[0], best[2]
        # ICC予測制動(2026-07-04, 速度マップ消費): 相手の"この先~25m"の学習速度(包絡線)が現在
        #   速度より低ければ低い方で制動計算。相手がヘアピンで減速するのを1周期遅れで追う
        #   のではなく事前に織り込む(s≈69で相手のコーナー減速に突っ込んだ接触の根治)。
        #   min()なので安全側にしか働かない。1周目(has_data無し)は従来通り現在速度。
        v_eff = v_long
        _om = getattr(self, "_opp_map", None)
        if _om is not None and _om.has_data(best[4], best[5]):
            _ahead = _om.v_pred_ahead(best[4], best[5], 25)
            if _ahead is not None and len(_ahead):
                v_eff = min(v_long, float(np.min(_ahead)))
        # 2026-07-16追加(80節): 本式(G2)はds(縦距離)のみで速度上限を決めており、
        #   dlat(既に確保できている横間隔)を一切見ていない。既に側方へ十分離れて
        #   いても「真後ろ」の場合と同じ強さで絞られ続け、0716-01実測でLap1第3コーナー
        #   (停止車の追い越し成功例)がwp61〜127の間終始5km/h前後(icc_f3クリープ床)に
        #   留まる原因となっていた。F3-TAPER(_est_gap=fwd_ds+fwd_dlat、0714-03で導入済み、
        #   「相手サイドへの接近自体にはペナルティが無く、横に十分離れていれば縦距離に
        #   関わらず安全」というドメイン仕様に基づく)と全く同じ考え方をここでも再利用し、
        #   実効距離をds+dlatとする(新規パラメータ0個)。dlatが実質0の通常追従
        #   (STOPPING時のicc_stop等)では_ds_eff≈dsとなり挙動は不変。
        _ds_eff = ds + best[3]
        return self._g2_speed(v_eff, _ds_eff), best

    def _closest_wp_and_s(self, x: float, y: float, prev_idx=None):
        """キャッシュ済み _waypoint_xy / _wp_s_cum を使った最近傍ウェイポイント探索。
        spatial model の get_closest_waypoint と同一インデックス規約。戻り値 (wp_id, s)。

        2026-07-14追加(ユーザー指摘: 「壁の向こう側にいる相手」誤認識対策): prev_idx
        (このID/自車が前回実際にマッチしたwp_id)が与えられた場合、探索をその近傍
        (±wp_match_radius_m、既存position_jump_thresholdを再利用)に限定する。
        ヘアピン等でコースが壁一枚を挟んで自分自身に近接する箇所では、全waypoint
        からの単純な(x,y)最近傍探索だと、弧長的に無関係な(壁の反対側の)waypointへ
        誤ってマッチしうる。prev_idx=None(初回、基準点が無い)場合のみ従来通り
        全waypointから探索する。"""
        if prev_idx is None:
            d = self._waypoint_xy - np.array([x, y], dtype=np.float64)
            idx = int(np.argmin(np.einsum('ij,ij->i', d, d)))
            return idx, float(self._wp_s_cum[idx])
        n = len(self._waypoint_xy)
        radius_idx = max(1, int(np.ceil(
            self._wp_match_radius_m / max(self._reference_path.resolution, 1e-3))))
        idxs = np.arange(prev_idx - radius_idx, prev_idx + radius_idx + 1) % n
        d = self._waypoint_xy[idxs] - np.array([x, y], dtype=np.float64)
        local = int(np.argmin(np.einsum('ij,ij->i', d, d)))
        idx = int(idxs[local])
        return idx, float(self._wp_s_cum[idx])

    def _side_blocked_by_other_car(self, scan, side, target_vid, ds_end,
                                    room: float, wp_o, need: float = None) -> bool:
        """選んだ側の走査区間内(0<ds≤ds_end)に、対象車(target_vid)以外の車が
        同じ側(lat符号が一致)に存在するか確認する(2026-07-09追加, K)。
        新規の検知機構は起こさず、ICC用に毎周期収集済みの scan["cars"] を流用する。
        4台走行(対向最大3台)では、対象1台だけを見て側を決めると、選んだレーンの
        先に2台目・3台目がいても事前検知できない構造的な盲点があった
        (2026-07-08 3台ローカル走行で確認、衝突はしないがchurnで機会損失)。

        2026-07-13修正: 旧実装は「符号が一致する車が1台でもいれば即ブロック」という
        粗い判定で、実際にどれだけ近い/重なっているかを一切見ていなかった。相手が2台
        近接して停止しているケース(0713-03実測: 全エンゲージゲート通過・持続空きも
        along_lane_need以上あったにも関わらずkvetoのみで10秒以上完全停止→経路3スタック
        に至った)で、本来抜けるはずの側まで過剰にブロックしていたと判明。既存のlf0/rf0
        計算(対象車1台に対して壁基準で算出する空き幅)と同じ考え方を2台目にも適用し、
        「2台目を踏まえてもなお along_lane_need 以上の空きが残るか」で判定するよう改めた
        (符号一致のみでの即ブロックをやめ、真に重なって狭くなる場合のみブロックする)。"""
        # 2026-07-14再修正(フローチャートで洗い出したギャップ③): needが省略された場合は
        #   従来通りalong_lane_need(1.85m、「走行中の相手」分岐向け)を既定にするが、
        #   障害物分岐(停止/低速な相手を低相対速度ですり抜ける場面)の呼び出し側からは
        #   59節と同じalong_min_width(1.45m)を渡す。2台目が近くにいる局面で、59節の
        #   緩和がこのK-checkだけ据え置きのため無効化されていた問題への対処。
        if need is None:
            need = self._along_lane_need
        wps = self._reference_path.waypoints
        wp_t = wps[wp_o % len(wps)]
        for c_ds, c_lat, _c_v, _c_dlat, c_vid, _c_wp in scan["cars"]:
            if c_vid == target_vid:
                continue
            if not (0.0 < c_ds <= ds_end):
                continue
            if (c_lat > 0.0) != (side > 0.0):
                continue
            if side > 0:
                c_room = max(0.0, float(wp_t.ub) - (c_lat + self._ot_block_half))
            else:
                c_room = max(0.0, (c_lat - self._ot_block_half) - float(wp_t.lb))
            _combined = min(room, c_room)
            if _combined < need:
                self.get_logger().info(
                    f"[K-CHECK] blocked side={side} c_vid={c_vid} c_lat={c_lat:.2f} "
                    f"c_room={c_room:.2f} room={room:.2f} combined={_combined:.2f} "
                    f"need={need:.2f} wp={wp_o}")
                return True
        return False

    def _plan_obs_log(self, vopp: float, lf: float, rf: float, side: int,
                       result: str, wp_o) -> None:
        """[PLAN-OBS]診断ログ(2026-07-13追加): 障害物分岐の適用条件拡張(closing速度基準)
        により、本来「走行中の相手」分岐に入っていたはずのケースが障害物分岐(実測ベース・
        短い窓)を通った際にのみ出力する。理由(result)が変化した周期は必ずログし、同じ
        結果が続く間は1-in-5で間引く(既存のLAT-TTC/PLAN-FAILログと同じ間引き方針)。
        2026-07-17追加(86/87節、検証ロギングのみ・判定ロジックは無変更): _plan_passが
        room>=along_min_widthで承認した幅が、MPCソルバー自身が要求するsafety_margin
        (core/MPC.py get_control()と同一の優先順位: safety_margin_override優先)を
        差し引いた実効幅でも本当に足りているかを、次回ログで目視突き合わせできるよう
        margin値をそのまま記録する(このログ自体はveto判定を一切変更しない)。"""
        _changed = (result != self._plan_obs_prev_result)
        if _changed or self._plan_obs_log_count % 5 == 0:
            _sm = (self._mpc.safety_margin_override
                   if self._mpc.safety_margin_override is not None
                   else self._mpc.model.safety_margin)
            self.get_logger().info(
                f"[PLAN-OBS] result={result} side={side} lf={lf:.2f} rf={rf:.2f} "
                f"vopp={vopp:.2f} margin={_sm:.3f} wp={wp_o}")
        self._plan_obs_log_count = 0 if _changed else self._plan_obs_log_count + 1
        self._plan_obs_prev_result = result

    def _room_debounce_ok(self, vid, side: int, room: float, need: float = None,
                           counter_key: str = "primary") -> bool:
        """2026-07-14追加(事象C対策): 55節で追加したmin-width veto(境界の単発比較)は、
        fwd_lat/対象車位置の測位ノイズにより境界付近でengage可否が周期ごとに反転しうる
        (v_safeのチャーン=事象Cの新たな発生源になりかねない、自己点検で発見)。
        対象車(vid)・側(side)が変わらない限り、roomが閾値(need、既定はalong_lane_need)
        以上である状態がengage_debounce周期連続して初めてOKとする。案B(0710-06、
        _ot_prev_side/_ot_prev_side_vidによる側フリップ抑制ヒステリシス)と全く同じ
        「対象車が同一なら持ち越す」考え方の再利用であり、新規のしきい値・パラメータは
        追加しない(既存のengage_debounceをそのまま流用)。
        need(2026-07-14追加、0714-03実測): 呼び出し元の文脈に応じた閾値を渡せるようにした。
        障害物分岐(停止/低速な相手を低相対速度ですり抜ける場面)ではalong_min_width
        (カート幅未満の物理下限)を渡す — along_lane_need(高速すれ違い時の並走継続余裕)は
        この文脈では過剰に保守的なため(k_corner vetoの閾値変更と対になる修正)。

        counter_key(190-7節、2026-07-26追加): 既定"primary"は従来通り
        self._plan_room_ok_count(単一スカラー)をそのまま使い、挙動は完全に無変更
        (byte-for-byteの後方互換)。反対側フォールバック用の呼び出しのみ別の
        counter_key("fallback")を渡し、独立した状態(self._plan_room_ok_count_by_key
        辞書)に持ち越すことで、主系統のvid/side変化リセット挙動には一切影響しない。"""
        if need is None:
            need = self._along_lane_need
        if counter_key == "primary":
            if vid != self._plan_room_prev_vid or side != self._plan_room_prev_side:
                self._plan_room_ok_count = 0
                self._plan_room_prev_vid = vid
                self._plan_room_prev_side = side
            if room >= need:
                self._plan_room_ok_count += 1
            else:
                self._plan_room_ok_count = 0
            return self._plan_room_ok_count >= self._ot_engage_debounce
        if (vid != self._plan_room_prev_vid_by_key.get(counter_key)
                or side != self._plan_room_prev_side_by_key.get(counter_key)):
            self._plan_room_ok_count_by_key[counter_key] = 0
            self._plan_room_prev_vid_by_key[counter_key] = vid
            self._plan_room_prev_side_by_key[counter_key] = side
        if room >= need:
            self._plan_room_ok_count_by_key[counter_key] = (
                self._plan_room_ok_count_by_key.get(counter_key, 0) + 1)
        else:
            self._plan_room_ok_count_by_key[counter_key] = 0
        return self._plan_room_ok_count_by_key[counter_key] >= self._ot_engage_debounce

    def _predicted_time_to_wp(self, from_wp: int, to_wp: int, max_dist: float):
        """2026-07-15追加(ユーザー提案: コース形状は既知のため、各コーナーでの
        計画減速/加速を踏まえれば「どこで追いつくか」は逆算できるはず): 固定の
        v_pot(自車の絶対最高速度)で一定速走行すると仮定する_engage_dist_dynamicより
        精度の高い、「自車が実際にto_wpへ到達するまでの予測時間」を、既知の区間速度
        プロファイル(ref_vel_configulator、コーナーごとの計画速度が事前に分かっている)
        を弧長積分して計算する。

        stopped_opponent(相手が完全停止/低速)の場合、相手の位置(wp)は実質固定であり、
        自車がそこへ到達するまでの所要時間は、区間ごとの計画速度さえ分かれば経路形状
        から一意に計算できる、というユーザーの指摘をそのまま実装したもの。

        ref_vel_configulatorが無い場合や、to_wpがmax_dist(既存fwd_max_consider、
        走査対象と同じ範囲)より遠い場合はNoneを返し、呼び出し側は
        _engage_dist_dynamic(v_pot一定近似)へフォールバックする。"""
        if self._ref_vel_configulator is None:
            return None
        rp = self._reference_path
        n = len(self._wp_s_cum)
        total_t = 0.0
        total_d = 0.0
        wp = int(from_wp) % n
        to_wp = int(to_wp) % n
        for _ in range(n):
            if wp == to_wp:
                return total_t
            nxt = (wp + 1) % n
            seg = float(self._wp_s_cum[nxt]) - float(self._wp_s_cum[wp])
            if rp.circular and seg < 0.0:
                seg += rp.length
            try:
                v_kmh = self._ref_vel_configulator.get_ref_vel(wp)
            except Exception:
                v_kmh = float(self._cfg.mpc.v_max)  # type: ignore
            v = max(kmh_to_m_per_sec(v_kmh), 0.1)
            total_t += seg / v
            total_d += seg
            if total_d > max_dist:
                return None
            wp = nxt
        return None

    def _switchback_curvature_veto(self, new_side: int) -> bool:
        """2026-07-15追加(76節: 0715-06実測でhas_rescued導入後にswitchback頻度が
        倍増し、COLLISION-SUSPECTEDが0→7回に増加した回帰への対処案①)。

        LAT-TTCのswitchback判定(通常branch=A・A_rescue共通)は、その瞬間の
        space/opp_space(=left_free/right_free)の比較のみで反転先を決めており、
        _plan_passのk_corner先読みveto相当のロジックが無い。実測(wp297のA_rescue
        ×2件、別周回で再現)では、反転5〜6秒後にRfreeが1.0〜1.9mまで縮小して
        いるにもかかわらずフルスピードのままCOLLISION-SUSPECTEDが発火していた。

        _plan_passのk_corner検出(1912-1935行目付近)と全く同じ考え方・同じ閾値
        (_ot_pass_block_kappa)を、直近_fwd_max_consider窓内で軽量に再走査する
        (新規パラメータ0個)。反転先(new_side: +1=左/-1=右、_ot_side規約に準拠)を
        閉じる方向の強いコーナー(|kappa|>=_ot_pass_block_kappa)が窓内にあれば
        True(反転を抑制)を返す。"""
        try:
            rp = self._reference_path
            wps = rp.waypoints
            n = len(wps)
            i0 = int(self._mpc.model.wp_id)
            s0 = float(self._wp_s_cum[i0])
            total = rp.length
        except Exception:
            return False
        for d in range(1, n):
            i = (i0 + d) % n
            seg = float(self._wp_s_cum[i]) - s0
            if rp.circular and seg < 0.0:
                seg += total
            if seg > self._fwd_max_consider:
                break
            k = float(wps[i].kappa)
            if abs(k) >= self._ot_pass_block_kappa:
                # k>0(左コーナー)は左(+1)を閉じる、k<0(右コーナー)は右(-1)を閉じる
                # (_plan_passのk_corner vetoと同一の符号規約)
                return (k > 0.0 and new_side > 0) or (k < 0.0 and new_side < 0)
        return False

    def _corr_bound_ahead(self, side: int) -> float:
        """2026-07-21追加(147節、壁激突の深掘り対処): オフセット目標のクランプを
        「今この瞬間の1点」(dbg_corr_ub0/lb0)ではなく、125節で公開済みの動的
        コリドー配列全体(dbg_corr_ub_arr/lb_arr、_switchback_wall_vetoと同一
        データソース)の先読み最小値へ変更するためのヘルパー。

        実測(0720-07 wp270→282、インサイドオーバーテイク中の壁激突)で、
        オフセット目標が単一点クランプのため「壁側コリドーが実際に狭まった
        瞬間」に初めて追従を始め、車両側の横方向応答が追いつかないまま
        壁マージンがゼロまで悪化する事象を確認した。配列全体の最小値を使う
        ことで、収縮が始まる前段階からオフセット目標自体を早めに緩め、
        車両に反応する時間的余裕を与える(新規計算・新規パラメータ0個、
        既存配列の再利用のみ)。"""
        arr = (self._mpc.dbg_corr_ub_arr if side > 0 else self._mpc.dbg_corr_lb_arr)
        if arr is None or len(arr) == 0:
            self._dbg_corr_bound_at_m = float('nan')
            return float(getattr(self._mpc, "dbg_corr_ub0" if side > 0 else "dbg_corr_lb0",
                                  float('inf') if side > 0 else -float('inf')))
        # 診断用(2026-07-22、153節): 採用した最小値(=このステップでオフセット目標を
        #   制約している地点)が現在位置から何m先かを記録する。argmin/argmaxを経由する
        #   だけで返り値の計算式自体はnp.min/np.maxと数値的に同一(挙動は無変更)。
        idx = int(np.argmin(arr)) if side > 0 else int(np.argmax(arr))
        try:
            i0 = int(self._mpc.model.wp_id)
            rp = self._reference_path
            n = len(self._wp_s_cum)
            i_at = (i0 + idx) % n
            d = float(self._wp_s_cum[i_at]) - float(self._wp_s_cum[i0])
            if rp.circular and d < 0.0:
                d += rp.length
            self._dbg_corr_bound_at_m = d
        except Exception:
            self._dbg_corr_bound_at_m = float('nan')
        return float(arr[idx]) if side > 0 else float(-arr[idx])

    def _switchback_wall_veto(self) -> bool:
        """2026-07-20追加(125節、A-1: switchbackが壁マージンを見ない盲点への対処)。

        _switchback_curvature_veto(静的トラックkappaのみ)は壁形状・対向車の
        occupancyを一切見ない。ユーザー指摘(「どのみち空いている隙間からしか
        抜けないのだから、素直に隙間の有無を見るべきでは」)を受け、静的な
        per-waypointテーブルではなく、MPC自身が毎周期実際に解いている動的コリドー
        (壁+占有格子込み、wall_slow(124節)が消費するdbg_corr_ub0/lb0と同一の
        _corridor()計算)の配列全体(dbg_corr_ub_arr/lb_arr、MPC.py側で新規計算
        ゼロで公開済み、約N*resolution[m]先読み)を再利用する。

        設計上の制約(ユーザーへ開示済み): update_path_constraints()は複数の
        空きセグメントがある場合、面積最大の経路を1本だけ選ぶため、この配列は
        「new_side固有の空き」ではなく「MPCが現在計画している単一経路」の幅を
        表す。よって本関数はnew_sideを引数に取らず、「その計画経路自体が
        along_min_width(カート幅未満の物理下限、_opponent_room_aheadと同一の
        既存閾値)を下回るほど狭い区間が先読み内にあるか」を見る、保守的な
        wall_slowの先読み版として機能する。

        戻り値: True=先読み内にalong_min_width未満の区間あり(反転を抑制)。
        """
        ub_arr = self._mpc.dbg_corr_ub_arr
        lb_arr = self._mpc.dbg_corr_lb_arr
        if ub_arr is None or lb_arr is None or len(ub_arr) == 0:
            return False
        for i in range(len(ub_arr)):
            if (float(ub_arr[i]) - float(lb_arr[i])) < self._along_min_width:
                return True
        return False

    def _opponent_room_ahead(self, vid, wp_id, side, n_ahead):
        """2026-07-18追加(103節、Phase 0)→107節案Cで判定入力へ昇格: 対象車両ID込みの
        room先読み。_plan_pass(2205〜2235行目付近)が既に持つ
        「OpponentSpeedMap.lat_mean(vid, i)で学習済みの相手の走行ラインを使い、
        waypoint毎にroomを算出する」というパターンを、_switchback_curvature_vetoと
        同じ軽量な先読み窓(呼び出し元がn_ahead[m]を渡す、既存の_fwd_max_consider
        を再利用する想定)に絞って切り出した軽量版。_plan_passのclosingベースの
        動的窓(w_max)は使わない(Stage1.5計装結果で主犯・従犯だったmpc/prep区間の
        コリドー走査系と同種の負荷を、CPUに最も余裕がない危険域へ追加しないため)。

        side=+1(左)/-1(右)について、om.lat_mean(vid, i)が学習済みのwaypointのみを
        対象に、_plan_passと同一の式(2228/2229行目)でroomを算出し、窓内の最小値を
        返す。未学習のwaypointは(フォールバックせず)単純にスキップする——学習データが
        無い場合にフォールバック値で判定を汚染しない(107節案C: new_side_room_blocked
        はfail-open、Falseのまま=素通し)。

        戻り値: (room_min, wp_at_min, n_sampled)。学習済みwaypointが窓内に1つも
        無ければ(None, None, 0)。
        """
        om = getattr(self, "_opp_map", None)
        if om is None or vid is None:
            return None, None, 0
        try:
            rp = self._reference_path
            wps = rp.waypoints
            n = len(wps)
            i0 = int(wp_id) % n
            s0 = float(self._wp_s_cum[i0])
            total = rp.length
        except Exception:
            return None, None, 0
        room_min = None
        wp_at_min = None
        n_sampled = 0
        for d in range(1, n):
            i = (i0 + d) % n
            seg = float(self._wp_s_cum[i]) - s0
            if rp.circular and seg < 0.0:
                seg += total
            if seg > n_ahead:
                break
            lat_o = om.lat_mean(vid, i)
            if lat_o is None:
                continue
            n_sampled += 1
            if side > 0:
                room = float(wps[i].ub) - (lat_o + self._ot_block_half)
            else:
                room = (lat_o - self._ot_block_half) - float(wps[i].lb)
            if room_min is None or room < room_min:
                room_min = room
                wp_at_min = i
        return room_min, wp_at_min, n_sampled

    def _plan_pass(self, scan, prefer_side=0):
        """追い越し計画(2026-07-04 一元化): 「どれだけ並走が必要か」と「その区間どちら側が
        空き続けるか」を一度に判定する。戻り (terrain_ok, side, req_m)。side=0は側自由。
        prefer_side(案B, 2026-07-11): 直近の側消失STOPPINGから短時間での再エンゲージ時、
        前回側がまだ物理的に許容範囲内なら反転させずに維持する(0以外で有効)。
        1) closing = v_pot − max(現在vopp, マップ区間平均)(相手の加速を織込み)。
        2) 持続空き幅: 相手の学習済み走行ライン lat(s) 基準に左右の最小空き幅(外膨らみも学習値)。
        3) 遠回り: オフセット線の弧長差 ∫max(0,−κ·d)ds を側別に計上し、所要時間
           T = t_lateral + (gap+遠回り)/closing が pass_t_max 以内の側のみ成立。
           両側可なら T が短い側(=ロス最小)。不成立=追従の方が速い(仕掛けない)。
        未学習wp(1周目)は「相手=レースライン上(lat=0)」を事前分布に同じ式で評価(壁・狭窄が
        常に反映される)。地形判定はここに一元化(worth内コーナー分岐/外側原則を統合・廃止済)。"""
        # N3診断(2026-07-11): Falseを返す全地点に理由タグを付け、[OT]ログのplan=から
        #   直接特定できるようにする(既存のgate=診断と同じ考え方)。挙動は変えない。
        self._dbg_plan_reason = "ok"
        self._dbg_plan_trace = []  # 診断用(2026-07-19): 今回呼び出し分のみ保持(edge整合)
        ds = scan["fwd_ds"]; vopp = scan["fwd_vopp"]
        vid = scan["fwd_vid"]; wp_o = scan["fwd_wp"]
        if ds is None or vopp is None:
            self._dbg_plan_reason = "no_scan"
            return False, 0, 0.0
        rp = self._reference_path
        wps = rp.waypoints
        n = len(wps)
        # 障害物分岐の適用条件拡張(2026-07-13): 従来はvopp<obstacle_speed(絶対速度)のみだったが、
        #   エンゲージ可否を決める外側ゲート_pass_worthは既に
        #   「vopp<obstacle_speed or (v_pot-vopp)>opp_min_closing」というclosing速度基準の
        #   OR条件を持っていた。_plan_pass内部の分岐選択がこれと不一致(絶対速度のみ)だったため、
        #   worth=1でエンゲージが試みられても「走行中の相手」分岐(closing依存のw0/req_l/req_r
        #   計算式)に入り、closingが小さい(自車と相手の巡航速度がほぼ同じ)場面でreq_l/req_rが
        #   非現実的な値(最大82m・20秒)まで膨張して持続空き判定に失敗し続ける事象を実測で確認
        #   (0713-01/0713-02、[PLAN-FAIL] reason=L:room/R:room が126件・77件)。
        #   _pass_worthと同じ式を再利用して分岐条件を統一し、この矛盾を解消する
        #   (design_docs/plan_pass_obstacle_unification_20260713.md参照)。
        _via_closing_ext = vopp >= self._opp_obstacle_speed  # 診断用: 拡張条件側で入ったか
        if vopp < self._opp_obstacle_speed or (self._v_pot - vopp) > self._opp_min_closing:
            # 停止/低速=障害物(2026-07-08修正): 対象車自身の実測空き幅+コーナー前内側可否で
            #   側を決める。旧実装は即 side=0 で呼び出し元(_scan_traffic の空き幅)へ委譲していたが、
            #   走行車が持つ「対象1台の実測に基づく判定」を欠いたまま呼び出し元へ渡していた。
            #   停止車は必ずいつか抜く必要があるため、両側ともコーナーで不可でも side=0 にはしない
            #   (その場合は生の空き幅で広い側を選ぶ=安全側フォールバック)。
            fwd_lat = scan["fwd_lat"]
            if fwd_lat is None or wp_o is None:
                # 2026-07-09修正(J): 位置不明時は ok=True,side=0 で呼び出し元の死に体
                # フォールバック(_choose_overtake_side/_prev_locked再利用、左右比較なし)へ
                # 委譲していたが、これが「狭い側を誤って再選択」の原因になっていた
                # (実測: t=183s、広い右3.46mを差し置いて狭い左2.73mを再選択→即失敗)。
                # 位置不明という稀なケースでは、安全側に倒して今回は仕掛けない。
                self._dbg_plan_reason = "lat_unknown"
                return False, 0, 0.0

            wp_t = wps[wp_o % n]
            lf0 = max(0.0, float(wp_t.ub) - (fwd_lat + self._ot_block_half))
            rf0 = max(0.0, (fwd_lat - self._ot_block_half) - float(wp_t.lb))
            # 2026-07-09再修正: 旧実装は|kappa|>=閾値で問答無用に内側を禁止する粗いveto
            #   だったが、これは「動いている相手を追い越す際、内側車線がコーナーのアペックスで
            #   物理的に消滅する」動的収束問題への対策であり、静止した障害物には同じ理屈が
            #   当てはまらない。実測(2026-07-09予選 t=793.92): Rfree=3.32m(対象車位置=コーナー
            #   直前の実測値、既に壁境界を反映済み)にもかかわらずkappa閾値のみでvetoされ、
            #   狭い左2.10mへ誤って倒れていた。「壁境界そのもの(wp.ub/wp.lb)を窓内で実測し
            #   最小値を取る」方式に置き換える — コーナーが本当に道を狭めるならこの実測値に
            #   直接反映されるため、閾値の要不要判断が不要になり、かつ物理的により正確。
            lf_min, rf_min = lf0, rf0
            # 「長い壁」化(2026-07-13): 相手が低速でも前進する分だけ窓を延伸する。相手の
            #   横移動フェーズ(t_lateral)の間に進む距離を足し、静止前提の固定窓より現実的な
            #   評価にする。新規定数は増やさず既存のt_lateralを再利用。
            clear_at = ds + self._ot_pass_clear + vopp * self._ot_t_lateral
            k_corner = None    # 「走行中の相手」分岐と同型のコーナーアペックスveto用
            d_corner = 0.0
            _lf_at_corner = None  # 2026-07-14追加: コーナー地点そのものでの実測空き幅
            _rf_at_corner = None  #   (相手の現在位置fwd_latを反映済み)を記録し、幅ベースveto判定に使う
            try:
                i0 = int(self._mpc.model.wp_id)
                s0 = float(self._wp_s_cum[i0]); total = rp.length
                for d in range(1, n):
                    i = (i0 + d) % n
                    seg = float(self._wp_s_cum[i]) - s0
                    if rp.circular and seg < 0.0:
                        seg += total
                    if seg > clear_at:
                        break
                    lf_i = max(0.0, float(wps[i].ub) - (fwd_lat + self._ot_block_half))
                    rf_i = max(0.0, (fwd_lat - self._ot_block_half) - float(wps[i].lb))
                    lf_min = min(lf_min, lf_i)
                    rf_min = min(rf_min, rf_i)
                    _k = float(wps[i].kappa)
                    # 診断用(2026-07-19): fwd_lat固定・壁境界(ub/lb)は各waypointの実値、
                    #   というこの計算過程自体をwaypoint単位で記録する(wp176-178ウェッジ調査)。
                    self._dbg_plan_trace.append(
                        (i, round(seg, 2), round(_k, 3), round(float(wps[i].ub), 2),
                         round(float(wps[i].lb), 2), round(lf_i, 2), round(rf_i, 2)))
                    if k_corner is None and abs(_k) >= self._ot_pass_block_kappa:
                        k_corner = _k
                        d_corner = seg
                        _lf_at_corner = lf_i
                        _rf_at_corner = rf_i
            except Exception:
                pass
            # k_corner veto(2026-07-13追加 → 2026-07-14修正): 障害物分岐にも「走行中の相手」
            #   分岐と同じ「コーナー前に抜き切れるか」安全装置を適用する(3-3節)。
            #   2026-07-14修正: 旧実装はkappa閾値の有無のみで一律veto(相手の現在位置を
            #   無視)しており、0713-05実測(wp168、Rfree=3.15>Lfree=2.24なのに
            #   planRf=-1e9でk_corner veto、狭い左へ強制されF3クリープで10秒以上停止→
            #   相手発進後にswitchback→壁際激突、という連鎖の起点となっていた)。
            #   相手の現在位置(fwd_lat)は既にlf_i/rf_iの壁基準実測に反映されているため、
            #   「コーナー地点そのものでの実測空き幅」が閾値未満の場合のみ
            #   vetoするよう変更し、相手が既にその側から避けている場合は誤って締め出さない。
            # 2026-07-14再修正(0714-03実測、ユーザー指摘): 本分岐(障害物=vopp<obstacle_speed)
            #   はそもそも「相手と同じ車線を高速ですれ違う」場面ではなく、低相対速度で
            #   すり抜ける場面である。along_lane_need(1.85m)は「走行中の相手」分岐(高速
            #   すれ違いのため並走継続に必要な余裕)向けの閾値であり、静止/低速な相手を
            #   ゆっくり回避するここでは過剰に保守的だった。実測(wp270-277): アウト側は
            #   相手車体で0.8〜1.0mまで狭まり、インはコーナー地点の実測が1.85m未満という
            #   理由だけでk_cornerによりveto(-1e9)、結果イン/アウト両方が締め出されて
            #   完全停止・追従に陥っていた。ここでのvetoの本質は「物理的に嵌まるか」
            #   (=既存alongside_min_width=カート幅未満の物理下限)であり、「並走を
            #   維持できる余裕があるか」(along_lane_need)ではない。同じ理由で下の
            #   最終幅チェック(_room_debounce_ok)にもalong_min_widthを渡す(2箇所で
            #   閾値を統一、新規定数は増やさない)。
            if k_corner is not None and clear_at > d_corner:
                if (k_corner > 0.0 and _lf_at_corner is not None
                        and _lf_at_corner < self._along_min_width):
                    lf_min = -1e9       # 左コーナーの内側=左、実測でも物理的に狭すぎるため不可
                elif (k_corner < 0.0 and _rf_at_corner is not None
                        and _rf_at_corner < self._along_min_width):
                    rf_min = -1e9       # 右コーナーの内側=右、実測でも物理的に狭すぎるため不可
            lf, rf = lf_min, rf_min
            _side = 1 if lf >= rf else -1
            # 案B(2026-07-11): 前回側がまだ_along_lane_need以上残っているなら反転させない
            #   (0710-06実測: wp176-178でside+1→-1反転直後に実速度が0.18m/sへ8秒張り付いた事象。
            #   既存の並走継続しきい値alongside_lane_needを再利用し、新規定数は増やさない)。
            if prefer_side != 0 and prefer_side != _side:
                _pref_free = lf if prefer_side > 0 else rf
                if _pref_free >= self._along_lane_need:
                    _side = prefer_side
            self._dbg_plan_lf = lf; self._dbg_plan_rf = rf  # 診断用(2026-07-09): [OT]ログへ出力
            # 2026-07-14追加(0714-02実測): 「走行中の相手」分岐(1954-1960行目)は
            #   l_ok/r_ok で選んだ側自身の絶対幅(along_lane_need以上)を必ず検証しているが、
            #   この障害物分岐(vopp<obstacle_speed)は `_side = 1 if lf>=rf else -1` という
            #   相対比較のみで側を確定しており、k_corner veto等で片側が-1e9になった際、
            #   もう片方が「相対的に大きい」というだけの理由でalong_lane_need(1.85m)未満でも
            #   通してしまう非対称なギャップがあった(実測: wp166 planLf=1.6m<1.85mのまま
            #   engage、その後のOVERTAKING継続中に幅不足が響き、switchback連鎖の起点になった
            #   可能性が高い)。既存along_lane_need・既存l_ok/r_okと同じ判定をここにも適用し、
            #   両分岐の非対称性を解消する(新規定数なし)。
            _room = lf if _side > 0 else rf
            # 2026-07-14追加(事象C対策): 単発比較ではなくデバウンス済みの結果でvetoする
            #   (自己点検: 55節で追加したこの境界チェック自体が測位ノイズでチャーンしうる
            #   ことに気付いたため、案Bと同じ既存ヒステリシス方式を適用する)。
            # 2026-07-14再修正(0714-03実測、ユーザー指摘): k_corner vetoと同じ理由で、
            #   ここもalong_lane_needではなくalong_min_width(物理下限)を渡す。
            _min_width_need = self._along_min_width
            # 190-7節(2026-07-26追加): argmaxで選んだ側(_side)がnarrow/kvetoで失敗した
            #   場合、反対側も同じ2つのチェックで即座に試す。5日分18ログ横断調査(190節)
            #   で、選んだ側が2台目(kveto)に塞がれている間、反対側を一切試さないまま
            #   最長35秒以上完全停止し続けた実例(0722-04)を確認した。反対側の
            #   room_debounce_okは独立counter_key="fallback"で持ち越すため、主系統の
            #   反転抑制ヒステリシス(vid/side変化で即リセット)には一切影響しない。
            def _side_fail_reason(check_side, check_room, counter_key):
                if not self._room_debounce_ok(vid, check_side, check_room,
                                               need=_min_width_need, counter_key=counter_key):
                    return "narrow"
                if self._side_blocked_by_other_car(scan, check_side, vid, clear_at,
                                                    room=check_room, wp_o=wp_o,
                                                    need=_min_width_need):
                    return "kveto"
                return None

            _fail_reason = _side_fail_reason(_side, _room, "primary")
            if _fail_reason is not None:
                _fb_side = -_side
                _fb_room = rf if _fb_side > 0 else lf
                if _side_fail_reason(_fb_side, _fb_room, "fallback") is None:
                    self.get_logger().info(
                        f"[SIDE-FALLBACK] side={_side}->{_fb_side} room={_fb_room:.2f} "
                        f"need={_min_width_need:.2f} orig_reason={_fail_reason} "
                        f"vid={vid} wp={wp_o}")
                    _side, _room = _fb_side, _fb_room
                    _fail_reason = None

            if _fail_reason == "narrow":
                self._dbg_plan_reason = "narrow"
                _reason_changed = (self._dbg_plan_reason != self._plan_fail_prev_reason)
                if _reason_changed or self._plan_fail_log_count % 5 == 0:
                    # 2026-07-17追加(86/87節、検証ロギングのみ): MPCソルバーが実際に使う
                    #   safety_margin(get_control()と同一の優先順位)を併記し、
                    #   need(along_min_width)だけでは見えない実効幅の不足を次回ログで
                    #   確認できるようにする。判定ロジック自体は無変更。
                    _sm_veto = (self._mpc.safety_margin_override
                                if self._mpc.safety_margin_override is not None
                                else self._mpc.model.safety_margin)
                    self.get_logger().info(
                        f"[PLAN-VETO] MIN-WIDTH FAIL side={_side} room={_room:.2f} "
                        f"need={_min_width_need:.2f} ok_count={self._plan_room_ok_count} "
                        f"debounce={self._ot_engage_debounce} lf={lf:.2f} rf={rf:.2f} "
                        f"vopp={vopp:.2f} margin={_sm_veto:.3f} wp={wp_o}")
                self._plan_fail_log_count = 0 if _reason_changed else self._plan_fail_log_count + 1
                self._plan_fail_prev_reason = self._dbg_plan_reason
                if _via_closing_ext:
                    self._plan_obs_log(vopp, lf, rf, _side, "narrow", wp_o)
                return False, 0, 0.0
            if _fail_reason == "kveto":
                self._dbg_plan_reason = "kveto"
                if _via_closing_ext:
                    self._plan_obs_log(vopp, lf, rf, _side, "kveto", wp_o)
                return False, 0, 0.0
            # 案A(2026-07-11実装 → 同日廃止): dbg_corr_wmin(ホライズン内最小幅)によるveto
            #   を試みたが、ローカル3台走行(0711 06:49台)で「対戦車皆無・単独巡航中でも
            #   wmin≈1.15〜1.19mになる静的コース狭窄区間」が実在すると判明し、そこでの
            #   正当な追い越し機会(Lfree=3.27m)を21秒間ブロックし続け、経路3の完全停止
            #   デッドロックを誘発する新規リグレッションとなったため撤去。
            #   さらに本来の対象だったwp176-178ウェッジ(0710-06)を再検証したところ、
            #   その事象中はwmin=5.1〜6.2mと終始「広い」判定であり、そもそもdbg_corr_wmin
            #   では検知不可能な種類の不具合だったことが判明(閾値の再校正でも解決不可)。
            #   wp176-178型の対策は別アプローチで再検討する。
            # 2026-07-20追加→即日revert(131節、Gap①対処の第1案):
            #   scan["fwd_dlat"]の絶対値でengageを拒否する案を試みたが、「相手が
            #   ちょうど正面にいる、ごく普通のengage直前の状態」でも同様に
            #   fwd_dlatが小さくなるため、正常なオーバーテイク開始の大半を誤って
            #   ブロックしてしまうことが既存回帰テスト(test_plan_pass_kcorner.py)
            #   で判明しrevertした。0720-02実測wp284の再分析で、真因は距離の絶対値
            #   ではなく「距離が縮まる速度(トレンド)」であり、これは_plan_pass
            #   単体では捕捉できない情報(トレンド追跡はOVERTAKING開始後のLAT-TTCの
            #   み保持)と判明。対処方針は再検討中(131節参照)。
            if _via_closing_ext:
                self._plan_obs_log(vopp, lf, rf, _side, "ok", wp_o)
            return True, _side, 0.0
        # --- 1) closing と完遂距離(相手のこの先の速度=マップ区間平均で補正) ---
        # 到達性検証ログ(2026-07-13追加): 分岐条件統一(3-1節)により、この分岐に入るのは
        #   (v_pot-vopp)<=opp_min_closingの場合のみだが、closing(=v_pot-v_opp_eff、
        #   v_opp_eff>=vopp)は数式上必ず同じか更に厳しい値になるため、以降のroom/tmax/
        #   k_corner veto計算(fix1のlat_mean処理を含む)は理論上到達不能になったはず。
        #   削除するかどうかは実走ログで到達しないことを確認してから判断する(保留中)。
        if self._plan_moving_log_count % 20 == 0:
            self.get_logger().info(
                f"[PLAN-MOVING-ENTER] vopp={vopp:.2f} v_pot={self._v_pot:.2f} wp={wp_o}")
        self._plan_moving_log_count += 1
        v_opp_eff = vopp
        v_seg = None    # 診断用(2026-07-13追加): [PLAN-FAIL]でv_seg_mean嵩上げの有無を確認するため保持
        om = getattr(self, "_opp_map", None)
        if om is not None and vid is not None and om.has_data(vid, wp_o):
            v_seg = om.v_seg_mean(vid, wp_o, 30)
            if v_seg is not None:
                v_opp_eff = max(vopp, v_seg)   # 相手が立ち上がって速くなるなら所要が延びる
        closing = self._v_pot - v_opp_eff
        if closing <= self._opp_min_closing:
            # N3診断(2026-07-11): v_opp_effがマップ区間平均(v_seg_mean)で嵩上げされている
            #   場合、現在のvoppでは追いつけても「相手が先で加速する前提」の見積りで
            #   仕掛けそびれる可能性がある。両者を区別できるようタグ分け。
            self._dbg_plan_reason = ("closing_seg" if v_opp_eff > vopp + 0.1 else "closing_raw")
            return False, 0, 0.0           # 追いつけない(蛇行しない・追従)
        # ここに到達した場合、上記の予想(理論上到達不能)が誤りだったことを意味する。
        # 間引き無しで必ずWARNで残す(想定外経路のため頻度が低いはず)。
        self.get_logger().warn(
            f"[PLAN-MOVING-ENTER] result=PASSED_CLOSING_CHECK(理論上到達不能のはず) "
            f"closing={closing:.2f} v_opp_eff={v_opp_eff:.2f} vopp={vopp:.2f} wp={wp_o}")
        gap = max(ds - 1.0, 0.0)
        w0 = self._v_pot * (self._ot_t_lateral + gap / closing)   # 遠回りゼロ仮定の基本窓
        # --- 2) 区間走査: 持続空き幅 + 追い越しライン(オフセット線)の遠回り距離 ---
        #   遠回り(2026-07-04): オフセット線の弧長は ∫(1−κ·d)ds。外側では |κ|·d_off だけ
        #   毎メートル長くなる(R=6ヘアピン+3m ≈ 5m余計 = closing1.4で3.6秒の追い付きが消える)。
        #   実測「外回りしたが全く抜けない」の正体。内側短縮は相手がアペックスを塞ぐため当てにしない。
        try:
            i0 = int(self._mpc.model.wp_id)
            s0 = float(self._wp_s_cum[i0]); total = rp.length
        except Exception:
            self._dbg_plan_reason = "wp_exc"
            return False, 0, w0
        _persist = 2.0 * self._v_pot                  # ねばり継続分(真横到達後も~2秒は並走が続く)
        w_max = w0 + self._v_pot * (10.0 / closing) + _persist
        steps = []                                     # (走行距離, 左room, 右room)
        extra_l = extra_r = 0.0
        prev_seg = 0.0
        k_corner = None                                # 窓内で最初のきついコーナー(符号付きκ)
        d_corner = 0.0
        _lf_at_corner = None  # 2026-07-14追加(水平展開: 障害物分岐と同型のk_corner veto盲点):
        _rf_at_corner = None  #   コーナー地点そのものでの実測空き幅(相手のlat_o反映済み)
        # コールドスタート対処(2026-07-12): 未学習wpの実測フォールバック値。fwd_latが無ければ
        #   従来通り0.0(中央仮定)。ds/vopp非Noneは関数冒頭(1561行目)で既にガード済みなので、
        #   ガレージ帯等そもそも対象車がいない場面ではこの分岐自体に到達しない。
        _fwd_lat_fallback = scan["fwd_lat"] if scan.get("fwd_lat") is not None else 0.0
        _near_lat_o = None      # 診断用(2026-07-12追加): 窓内最近傍wpのlat_o実値
        _near_lat_src = None    # "learned"/"fallback"
        for d in range(1, n):
            i = (i0 + d) % n
            seg = float(self._wp_s_cum[i]) - s0
            if rp.circular and seg < 0.0:
                seg += total
            if seg > w_max:
                break
            # 未学習wp(1周目)は、相手の実測横位置(fwd_lat)を事前分布として使う(2026-07-12修正)。
            #   旧実装は「相手はレースライン上(lat=0)」固定だったため、1周目に相手が実際に
            #   避けている方向を無視して側選択することがあった(0712-03、コーナー3でアウトへ
            #   避けた相手に対しアウトから追い越しを試行)。fwd_latが取れない場合のみ旧来の
            #   0.0(中央仮定、壁・狭窄=wp.ub/lbは常に反映されるため安全側)へフォールバックする。
            lat_o = om.lat_mean(vid, i) if (om is not None and vid is not None) else None
            _is_fallback = lat_o is None
            if _is_fallback:
                lat_o = _fwd_lat_fallback
            if _near_lat_o is None:
                _near_lat_o = lat_o
                _near_lat_src = "fallback" if _is_fallback else "learned"
            steps.append((seg,
                          float(wps[i].ub) - (lat_o + self._ot_block_half),
                          (lat_o - self._ot_block_half) - float(wps[i].lb)))
            _k = float(wps[i].kappa)
            if k_corner is None and abs(_k) >= self._ot_pass_block_kappa:
                k_corner = _k                          # 最初のきついコーナー(内側可否判定用)
                d_corner = seg
                _lf_at_corner = steps[-1][1]
                _rf_at_corner = steps[-1][2]
            if seg <= w0:                              # 遠回りは基本窓内で計上
                _dl = seg - prev_seg
                extra_l += max(0.0, -_k * self._ot_d_off) * _dl   # 左=右コーナー(κ<0)で遠回り
                extra_r += max(0.0, +_k * self._ot_d_off) * _dl   # 右=左コーナー(κ>0)で遠回り
            prev_seg = seg
        # --- 3) 側別の所要時間と成立判定: T = t_lateral + (gap+遠回り)/closing。
        #   予算(pass_t_max)超過は「追従の方がロスが少ない」ので仕掛けない(2026-07-04)。 ---
        t_l = self._ot_t_lateral + (gap + extra_l) / closing
        t_r = self._ot_t_lateral + (gap + extra_r) / closing
        req_l = w0 + extra_l * self._v_pot / closing
        req_r = w0 + extra_r * self._v_pot / closing
        # 持続空きは「真横到達+ねばり継続分」まで見る(旧: 真横到達までで、並走のまま突入する
        #   コーナー区間が窓の外だった=s66ヘアピン内側罠の一因 2026-07-05)
        left_room = min((lr for sg, lr, _ in steps if sg <= req_l + _persist), default=-1e9)
        right_room = min((rr for sg, _, rr in steps if sg <= req_r + _persist), default=-1e9)
        # A: 内側可否=「コーナー前に抜き切れるか」(2026-07-05 ユーザー要件: 直線で並び切れる
        #   なら内側OK)。抜き切り(相手の前へ pass_clear 出る)がきついコーナー進入までに完了
        #   しないなら、そのコーナーの内側は選ばない(内側レーンはアペックスで構造的に消滅、
        #   減速では解決しない。実測: lane 0.61m@s72)。外側は遠回り+時間予算が引き続き律する。
        # 2026-07-14修正(水平展開: 障害物分岐のk_corner veto盲点=0713-05実測と同型のバグを
        #   この「走行中の相手」分岐でも確認): 旧実装はkappa閾値+タイミングのみで一律veto
        #   しており、コーナー地点そのものでの実測空き幅(_lf_at_corner/_rf_at_corner、相手の
        #   実際の位置lat_oを反映済み。上のsteps.append計算と同一値)を見ていなかった。相手が
        #   既にその側から避けている場合まで誤って締め出さないよう、実測が閾値未満の場合のみ
        #   vetoする。本分岐は「高速すれ違いのため並走を維持できる余裕があるか」を問う文脈
        #   (l_ok/r_okも同じalong_lane_needを使用、障害物分岐のalong_min_widthとは別文脈)
        #   のため、閾値はalong_lane_need(新規定数無し、この分岐の既存基準をそのまま流用)。
        if k_corner is not None:
            req_clear = self._v_pot * (self._ot_t_lateral
                                       + (gap + self._ot_pass_clear) / closing)
            if req_clear > d_corner:
                if (k_corner > 0.0 and _lf_at_corner is not None
                        and _lf_at_corner < self._along_lane_need):
                    left_room = -1e9       # 左コーナーの内側=左、実測でも狭すぎるため不可
                elif (k_corner < 0.0 and _rf_at_corner is not None
                        and _rf_at_corner < self._along_lane_need):
                    right_room = -1e9      # 右コーナーの内側=右、実測でも狭すぎるため不可
        # 側の成立 = 持続空き(閾値=並走成立幅) AND 瞬時空き(min_gap) AND 時間予算内
        l_ok = (left_room >= self._along_lane_need
                and (scan["left_free"] or 0.0) >= self._ot_min_gap
                and t_l <= self._ot_pass_t_max)
        r_ok = (right_room >= self._along_lane_need
                and (scan["right_free"] or 0.0) >= self._ot_min_gap
                and t_r <= self._ot_pass_t_max)
        if not (l_ok or r_ok):
            # N3診断(2026-07-11): room(持続空き不足)/gap(瞬時空き不足)/tmax(所要時間予算超過)を
            #   左右それぞれについて特定し、[OT]ログのplan=から直接原因を切り分けられるようにする。
            def _fail(room_ok, gap_ok, tmax_ok):
                if not room_ok:
                    return "room"
                if not gap_ok:
                    return "gap"
                if not tmax_ok:
                    return "tmax"
                return "ok"
            _lfail = _fail(left_room >= self._along_lane_need,
                           (scan["left_free"] or 0.0) >= self._ot_min_gap,
                           t_l <= self._ot_pass_t_max)
            _rfail = _fail(right_room >= self._along_lane_need,
                           (scan["right_free"] or 0.0) >= self._ot_min_gap,
                           t_r <= self._ot_pass_t_max)
            self._dbg_plan_reason = f"L:{_lfail}/R:{_rfail}"
            # 診断用(2026-07-09)を失敗時にも更新する(2026-07-13修正): 従来は成功時にしか
            #   更新されず、[OT]ログのplanLf/planRfが失敗し続ける間ずっと直近成功時の古い値の
            #   まま表示され、実際に何が不足していたのか追えなかった(0712-05、連続シケイン
            #   区間wp216-277でplanLf/planRfが約60周期同一値のまま変化せず誤解を招いた)。
            self._dbg_plan_lf = left_room; self._dbg_plan_rf = right_room
            # [PLAN-FAIL]診断ログ(2026-07-13追加): 「room不足で仕掛けられない」状態が長時間
            #   持続する事象(0712-05実測、worth=1のままplan=L:room/R:room が60周期以上継続)の
            #   原因切り分け用。理由が変化した周期は必ずログ、同じ理由が続く間は1-in-5で間引く
            #   (既存のLAT-TTCログと同じ間引き方針)。
            _reason_changed = (self._dbg_plan_reason != self._plan_fail_prev_reason)
            if _reason_changed or self._plan_fail_log_count % 5 == 0:
                self.get_logger().info(
                    f"[PLAN-FAIL] reason={self._dbg_plan_reason} "
                    f"left_room={left_room:.2f} right_room={right_room:.2f} "
                    f"req_l={req_l:.2f} req_r={req_r:.2f} t_l={t_l:.2f} t_r={t_r:.2f} "
                    f"k_corner={k_corner} lat_o={_near_lat_o} lat_src={_near_lat_src} "
                    # 2026-07-13追加: v_seg_mean嵩上げの有無を直接確認するため
                    #   vopp(実測)/v_seg(区間平均)/v_opp_eff(採用値)/closingを併記する。
                    f"vopp={vopp:.2f} v_seg={v_seg} v_opp_eff={v_opp_eff:.2f} "
                    f"closing={closing:.2f} "
                    f"vid={vid} wp={wp_o}")
            self._plan_fail_log_count = 0 if _reason_changed else self._plan_fail_log_count + 1
            self._plan_fail_prev_reason = self._dbg_plan_reason
            return False, 0, w0            # 空きなし/時間超過 → 追従で機会待ち
        if l_ok and r_ok:
            # 両側可: 所要時間が短い側(=ロス最小)。僅差は空きが広い側
            if abs(t_l - t_r) > 0.5:
                _side, _req = (+1 if t_l < t_r else -1), min(req_l, req_r)
            else:
                _side, _req = (+1 if left_room >= right_room else -1), min(req_l, req_r)
        else:
            _side, _req = (+1, req_l) if l_ok else (-1, req_r)
        self._dbg_plan_lf = left_room; self._dbg_plan_rf = right_room  # 診断用(2026-07-09)
        # K(2026-07-09): 選んだ側の窓内(走査済みの区間全体)に対象車以外がいれば
        #   今回は仕掛けない(既存の「両側塞がり/時間超過」パターンと同じ扱い)。
        if self._side_blocked_by_other_car(scan, _side, vid, w_max,
                                            room=(left_room if _side > 0 else right_room),
                                            wp_o=wp_o):
            self._dbg_plan_reason = "kveto"
            return False, 0, w0
        # 2026-07-20追加→即日revert(131節、Gap①対処の第1案): 障害物分岐と同じ
        #   fwd_dlat絶対値ゲートをここにも追加していたが、既存回帰テスト
        #   (test_plan_pass_kcorner.py)で「相手が正面にいる、ごく普通のengage
        #   直前」の正常ケースまで誤ってブロックすることが判明しrevertした
        #   (障害物分岐側は同日中にrevert済みだったが、本分岐側の削除が漏れて
        #   いたことを132節Phase0の実装検証で発見・是正した)。詳細は
        #   design_docs 132節参照。
        # 側選択診断ログ(2026-07-12修正): 0712-04で判明した通り、コールドスタート
        #   (未学習→fallback)は稀にしか起きず、学習済みlat_meanの値そのものが実際の
        #   相手位置とズレている可能性が浮上した。フォールバック発生時のみだった間引きを
        #   やめ、エンゲージ成立の都度、窓内最近傍wpのlat_o実値(学習/fallbackどちらの
        #   出所か明示)と実測fwd_latを併記して出力し、両者の乖離を直接比較できるようにする。
        self.get_logger().info(
            f"[PLAN-LAT] side={_side} lat_src={_near_lat_src} lat_o={_near_lat_o} "
            f"fwd_lat={scan.get('fwd_lat')} planLf={left_room:.2f} planRf={right_room:.2f} "
            f"vid={vid} wp={wp_o}")
        return True, _side, _req

    def _offset_line_speed_cap(self, ey: float):
        """内側/オフセットライン用の動的速度上限(2026-07-05, ユーザー要件:「内側は通常より
        小さいRを曲がるのでしっかり減速」)。平行オフセット線の実効曲率 κ_eff=κ/(1−κ·e_y) に
        ay_profile 予算を適用し、前方~15mの各wpに対し「今から a_min で減速して間に合う速度」
        の min を返す。min適用=下げ方向のみ(外側でRが広がっても上げには使わない)。
        追い越し中のみ呼ばれる(~16反復、40Hz負荷への影響は微小)。
        2026-07-17追加(97節): 自車wp_id(離散インデックス)が1つ進むだけで先読み窓全体が
        1点分シフトし、コーナー頂点付近の急な曲率スパイクが窓の遠端(15m境界)で出入りする
        ことで、生の値が周期ごとに大きく揺れる(0717-02実測、icc_f3とのv_safe_srcチャタリング
        24回を確認)。既存のalong_lane_ema/v_corridor_emaと同じ考え方・同じ時定数
        (_ot_ema_alpha)で出力値自体を平滑化し、この離散化起因の揺れを均す(新規パラメータ0個)。
        非該当(呼び出し元がOVERTAKING外/新規エンゲージ等)の場合はEMA状態も呼び出し元で
        リセットされる想定。
        2026-07-19追加(112節): ey引数は呼び出し元で_cur_ey(現在の実位置)から_cur_off
        (確定済みのオフセット目標)へ変更された。本メソッド自体はどちらの意味の値を
        渡されても同じ計算をするだけ(このメソッド内にロジック変更はない)。呼び出し元の
        変更理由は3828行目付近のコメント・design_docs 112節(79節の未解決課題への回答)
        を参照。"""
        try:
            rp = self._reference_path
            wps = rp.waypoints
            n = len(wps)
            i0 = int(self._mpc.model.wp_id)
            s0 = float(self._wp_s_cum[i0])
            total = rp.length
        except Exception:
            self._line_cap_ema = None
            return None
        a_br = abs(float(self._mpc_cfg.a_min))
        cap = None
        for d in range(0, 16):
            i = (i0 + d) % n
            seg = float(self._wp_s_cum[i]) - s0
            if rp.circular and seg < 0.0:
                seg += total
            if seg > 15.0:
                break
            k = float(wps[i].kappa)
            den = 1.0 - k * ey
            if den < 0.2:
                den = 0.2                  # オフセットが回転中心へ近づく特異点のガード
            keff = abs(k / den)
            if keff < 1e-3:
                continue
            v_here = float(np.sqrt(self._ay_profile / keff))
            v_now = float(np.sqrt(v_here * v_here + 2.0 * a_br * max(seg, 0.0)))
            cap = v_now if cap is None else min(cap, v_now)
        if cap is None:
            self._line_cap_ema = None
            return None
        if self._line_cap_ema is None:
            self._line_cap_ema = cap
        else:
            self._line_cap_ema += self._ot_ema_alpha * (cap - self._line_cap_ema)
        return self._line_cap_ema

    def _filter_obstacles_to_corridor(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        if not obstacles or self._waypoint_xy.size == 0:
            return obstacles
        thr_sq = self._v2x_corridor_threshold_sq
        wps = self._waypoint_xy
        kept: List[Obstacle] = []
        for ob in obstacles:
            dxy = wps - np.array([ob.cx, ob.cy], dtype=np.float64)
            if np.min(np.einsum('ij,ij->i', dxy, dxy)) <= thr_sq:
                kept.append(ob)
        return kept

    def _border_cells_callback(self, msg: BorderCells):
        self._reference_path.set_border_cells(
            msg.dynamic_upper_bounds, msg.dynamic_lower_bounds, msg.rows, msg.cols)

    def _trajectory_callback(self, msg):
        self._trajectory = msg

    def _awsim_status_callback(self, msg):
        laps = int(msg.data[1])
        lap_time = msg.data[2]
        # section = int(msg.data[3])

        if self._current_laps is None:
            self._current_laps = 1 if laps == 0 else laps

        if laps > self._current_laps:
            self.get_logger().info(f'\033[32mLap {self._current_laps} completed! Lap time: {self._last_lap_time} s\033[0m')
            self._lap_times[self._current_laps] = self._last_lap_time
            self._current_laps = laps
            # 検証: 周回ごとに相手速度マップの学習状況をログdump(bagトピック設定に依存せず必ず残る)
            _om = getattr(self, "_opp_map", None)
            if _om is not None:
                self.get_logger().info(f"[OPP_MAP] {_om.summary()}")

        self._last_lap_time = lap_time

    def _condition_callback(self, msg: Int32):
        if self._last_condition is None:
            self._last_condition = msg.data

        diff_condition = msg.data - self._last_condition
        if diff_condition > 30.0:
            self._last_colliding_time = self.get_clock().now()
            self.get_logger().warning(f"Collision detected!")
        self._last_condition = msg.data

    def _stop_request_callback(self, msg: Empty) -> None:
        if self._enable_control:
            self.get_logger().warn(f"Stop request received {self._enable_control}")
            self._enable_control = False

    def _wait_until_clock_received(self) -> None:
        if self.use_sim_time:
            self.get_logger().info(f"wait until clock received...")
            rate = self.create_rate(10)
            rate.sleep()
            self.get_logger().info(f">> OK!")

    def _wait_until_message_received(self, message_getter, message_name: str, timeout: float, rate_hz: int = 30) -> None:

        t_start = self.get_clock().now()
        rate = self.create_rate(rate_hz)

        self.get_logger().info(f"wait until {message_name} received...")

        while message_getter() is None:
            now = self.get_clock().now()
            if (now - t_start).nanoseconds > timeout * 1e9:
                self.get_logger().info(f"now: {now}, t_start: {t_start}")
                raise TimeoutError(f"Timeout while waiting for {message_name} message")
            rate.sleep()

        self.get_logger().info(f">> OK!")

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        self._wait_until_message_received(lambda: self._odom, 'odometry', timeout)

    def _wait_until_trajectory_received(self, timeout: float = 30.) -> None:
        if self._cfg.reference_path.update_by_topic:
            self._wait_until_message_received(lambda: self._trajectory, 'trajectory', timeout)

    def _wait_until_path_constraints_received(self, timeout: float = 30.) -> None:
        if self.USE_OBSTACLE_AVOIDANCE and self._cfg.reference_path.use_path_constraints_topic: # type: ignore
            self._wait_until_message_received(lambda: self._reference_path.path_constraints, 'path constraints', timeout)

    def _publish_mpc_pred_marker(self, x_pred, y_pred):
        # Stage1.7 R3(2026-07-07): 18個のSPHERE(deepcopy×18)→SPHERE_LIST 1個へ。
        #   見た目は同じ白点列のまま、メッセージ構築コスト(実測max10.6ms)を1/10以下に。
        #   注: bag解析側は「マーカ1個のpoints[]」として復号すること(旧: pose×18)。
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "mpc_pred"
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale = Vector3(x=0.5, y=0.5, z=0.5)
        m.color = self._pred_marker_color
        m.points = [Point(x=float(x), y=float(y), z=0.0)
                    for x, y in zip(x_pred, y_pred)]
        pred_marker_array = MarkerArray()
        pred_marker_array.markers.append(m)  # type: ignore
        self._mpc_pred_pub.publish(pred_marker_array)
        self._mpc_pred_pub_dummy.publish(pred_marker_array)

    def _publish_opponent_speed_map(self) -> None:
        """検証出力: 相手速度マップを Float32MultiArray(bag復号用)+ MarkerArray(RViz色プロット)で publish。
        Stage1.8 S1(2026-07-08): 予選機で1回30〜44ms(=残存欠落62回/分の主犯)を実測したため軽量化。
        ①データ: to_flatはnumpy .tolist()済みなので再ループ(2,800要素)を撤廃(1Hz維持=bag解析互換)
        ②マーカ: 車両毎SPHERE×百個超→SPHERE_LIST 1個(点毎color、見た目同等)
        ③マーカは5回に1回(≈5秒毎)に間引き(RVizの見た目は静的マップなので実害なし)"""
        msg = Float32MultiArray()
        msg.data = self._opp_map.to_flat()
        self._opp_map_pub.publish(msg)

        self._opp_map_marker_skip = getattr(self, "_opp_map_marker_skip", 0) + 1
        if self._opp_map_marker_skip < 5:
            return
        self._opp_map_marker_skip = 0
        ma = MarkerArray()
        wpxy = self._waypoint_xy
        cap_all = max((self._opp_map.cap(v) for v in self._opp_map.vids()), default=1.0) or 1.0
        for mid, vid in enumerate(self._opp_map.vids()):
            m = Marker()
            m.header.frame_id = "map"
            m.ns = f"opp_speed_{vid}"
            m.id = mid
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale = Vector3(x=0.4, y=0.4, z=0.4)
            for w in range(0, min(len(wpxy), self._opp_map.n), 3):
                if not self._opp_map.has_data(vid, w):
                    continue
                vp = self._opp_map.v_pred(vid, w)
                if vp is None:
                    continue
                r = float(np.clip(vp / cap_all, 0.0, 1.0))
                m.points.append(Point(x=float(wpxy[w][0]), y=float(wpxy[w][1]), z=0.2))
                m.colors.append(ColorRGBA(r=r, g=0.1, b=1.0 - r, a=0.8))
            if m.points:
                ma.markers.append(m)  # type: ignore
        if ma.markers:
            self._opp_map_marker_pub.publish(ma)

    def _publish_overtake_status(self, dbg, u):
        """gate2: 追い越し状態の診断を Float32MultiArray で publish（解析用、低頻度）。"""
        import math as _math
        state_map = {"NORMAL": 0.0, "OVERTAKING": 1.0, "STOPPING": 2.0}

        def _f(v):
            return float(v) if v is not None else _math.nan

        msg = Float32MultiArray()
        msg.data = [
            state_map.get(dbg.get("state"), -1.0),
            float(dbg.get("side", 0) or 0),
            float(dbg.get("n_fwd", 0) or 0),
            _f(dbg.get("d_min")),
            _f(dbg.get("left_free")),
            _f(dbg.get("right_free")),
            float(u[0]) if (u is not None and len(u) >= 1) else _math.nan,
            _f(dbg.get("offset")),   # [m] 実効横オフセット（alpha*target, 右=負）
            # --- 案X corridor debug (idx 8..) ---
            _f(dbg.get("corr_ub0")), _f(dbg.get("corr_lb0")), _f(dbg.get("corr_xr0")),
            _f(dbg.get("corr_wmin")), _f(dbg.get("corr_src")),
            float(dbg.get("nseg0", 0) or 0), float(dbg.get("nseg1", 0) or 0),
            float(dbg.get("nseg2", 0) or 0),
            _f(dbg.get("psi_bias")),   # idx16: ヘディング参照バイアス[deg]（右=負）
        ]
        self._overtake_status_pub.publish(msg)

    def _publish_ref_path_marker(self, ref_path: ReferencePath):
        WP_SPHERE_ENABLED = False

        ref_path_marker_array = MarkerArray()

        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "ref_path"
        m_base.type = Marker.LINE_STRIP
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        # 2026-07-23修正(166節続報): RVIZで視認しづらい(細い・青くて薄い)というユーザー指摘により、
        #   緑・太めへ変更(RaceTrajectory表示は/mpc/ref_pathへ張替え済み、166-24節)。
        m_base.scale.x = 0.5
        m_base.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.95)

        for i in range(len(ref_path.waypoints) - 1):
            m = copy.deepcopy(m_base)
            m.id = i
            start = Point()
            start.x = ref_path.waypoints[i].x
            start.y = ref_path.waypoints[i].y
            end = Point()
            end.x = ref_path.waypoints[i + 1].x
            end.y = ref_path.waypoints[i + 1].y
            m.points.append(start) # type: ignore
            m.points.append(end) # type: ignore
            ref_path_marker_array.markers.append(m) # type: ignore

        if WP_SPHERE_ENABLED:
            spheres = Marker()
            spheres.header.frame_id = "map"
            spheres.ns = "ref_path_point"
            spheres.type = Marker.SPHERE_LIST
            spheres.action = Marker.ADD
            radius = 0.2
            spheres.scale = Vector3(x=radius, y=radius, z=radius)
            spheres.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)
            for i in range(len(ref_path.waypoints) - 1):
                p = Point()
                p.x = ref_path.waypoints[i].x
                p.y = ref_path.waypoints[i].y
                p.z = 0.
                spheres.points.append(p) #type: ignore
            ref_path_marker_array.markers.append(spheres) # type: ignore

        self._ref_path_pub.publish(ref_path_marker_array)
        self._ref_path_pub_dummy.publish(ref_path_marker_array)

    # --- Stage1.5 区間計装(2026-07-06) -----------------------------------
    #   目的: 予選40FPSの制御欠落105回/分の犯人特定。initは既に5.2msで犯人でないことが
    #   確定済み(/clock健全・tick充足率95.7%)のため、_control全体を区間分割して実測する。
    #   仕組み: 順次チェックポイント(_pf_mark=直前markからの経過を区間に積算)+
    #   非順次区間(_pf_add: v2x/GC/opp_pub)。~10秒毎に[PERF]で avg/max を出力し窓リセット。
    #   オーバーヘッドはµs級。特定完了後は _pf_report_every=0 で無効化できる。
    def _pf_init(self):
        self._pf_acc = {}
        self._pf_last = _time.perf_counter()
        self._pf_cycles = 0
        self._pf_work_sum = 0.0
        self._pf_work_max = 0.0
        self._pf_over25 = 0
        self._pf_report_every = 400   # ≈10秒(40Hz)
        self._pf_gc_t0 = None
        self._pf_obs_max = 0   # 診断用(2026-07-09): 区間内でcorridorへ投入された障害物数の最大値
        # 2026-07-25追加(179節続報): dev3ローカル実測でdocker statsからCPU競合(cpus=3上限
        #   への張り付き)を直接確認できたが、予選環境はdocker host側の計測手段が無い。
        #   同じ診断能力をプロセス内部から得るため、resource.getrusage(RUSAGE_SELF)の
        #   実CPU時間(ru_utime+ru_stime)と不随意コンテキストスイッチ回数(ru_nivcsw、
        #   CPUを横取りされた=競合で追い出された回数の直接証拠)を[PERF]と同じ窓で追跡する。
        #   壁時計時間(work avg)に対し実CPU時間が明らかに少なければ「計算コストではなく
        #   スケジューリング待ちで遅れている」ことを直接示せる。
        self._pf_rusage_prev = _resource.getrusage(_resource.RUSAGE_SELF)
        _gc.callbacks.append(self._pf_gc_cb)

    def _pf_gc_cb(self, phase, info):
        if phase == 'start':
            self._pf_gc_t0 = _time.perf_counter()
        elif self._pf_gc_t0 is not None:
            self._pf_add('gc', _time.perf_counter() - self._pf_gc_t0)
            self._pf_gc_t0 = None

    def _pf_add(self, name, dt):
        acc = self._pf_acc.get(name)
        if acc is None:
            self._pf_acc[name] = [dt, dt, 1]
        else:
            acc[0] += dt
            acc[2] += 1
            if dt > acc[1]:
                acc[1] = dt

    def _pf_mark(self, name):
        t = _time.perf_counter()
        self._pf_add(name, t - self._pf_last)
        self._pf_last = t

    def _pf_cycle_end(self, work):
        self._pf_cycles += 1
        self._pf_work_sum += work
        if work > self._pf_work_max:
            self._pf_work_max = work
        if work > 0.025:
            self._pf_over25 += 1
        self._pf_obs_max = max(self._pf_obs_max, getattr(self, "_dbg_n_dynobs", 0))
        if self._pf_report_every and self._pf_cycles >= self._pf_report_every:
            k = self._pf_cycles
            parts = ' | '.join(
                '%s a=%.1f m=%.1f' % (n, a[0] / max(a[2], 1) * 1000, a[1] * 1000)
                for n, a in sorted(self._pf_acc.items()) if n != 'sleep')
            print('[PERF] n=%d work avg=%.1fms max=%.1fms >25ms=%d回 n_dynobs_max=%d | %s'
                  % (k, self._pf_work_sum / k * 1000, self._pf_work_max * 1000,
                     self._pf_over25, self._pf_obs_max, parts), flush=True)
            # 179節続報: docker statsが使えない予選環境でも同じ診断ができるよう、
            #   このプロセス自身の実CPU時間と不随意コンテキストスイッチ数を[PERF]と
            #   同じ窓で報告する。cpu_ratio(実CPU時間/壁時計時間)が1に近ければ純粋な
            #   計算コスト、大きく下回っていればスケジューリング待ち(競合)が支配的。
            _ru = _resource.getrusage(_resource.RUSAGE_SELF)
            _prev = self._pf_rusage_prev
            _cpu_time = ((_ru.ru_utime + _ru.ru_stime)
                         - (_prev.ru_utime + _prev.ru_stime))
            _nivcsw = _ru.ru_nivcsw - _prev.ru_nivcsw
            _nvcsw = _ru.ru_nvcsw - _prev.ru_nvcsw
            _wall_time = self._pf_work_sum
            _cpu_ratio = _cpu_time / _wall_time if _wall_time > 1e-9 else float('nan')
            print('[PERF-RUSAGE] n=%d cpu_time=%.2fs wall_time=%.2fs cpu_ratio=%.2f '
                  'nivcsw=%d nvcsw=%d'
                  % (k, _cpu_time, _wall_time, _cpu_ratio, _nivcsw, _nvcsw), flush=True)
            self._pf_rusage_prev = _ru
            self._pf_acc = {}
            self._pf_cycles = 0
            self._pf_work_sum = 0.0
            self._pf_work_max = 0.0
            self._pf_over25 = 0
            self._pf_obs_max = 0

    def _g2_release_ready(self, scan, fwd_vopp, vtgt, left_free, right_free,
                           v_safe_pre_for_log, is_closing_trend: bool = False) -> bool:
        """2026-07-20抽出(143節続報、スリム化)。G/G-2/G-3(2026-07-08〜18節、複数回の
        実測事故を経て段階的に確立した「側方確保できたので前車追従の速度上限から
        解放してよいか」の判定)を_control()から1メソッドへ抽出した。ロジック・
        デバウンス状態(_g2_clear_on_count等)・[G2-RELEASE]ログとも完全に無変更。
        各条件の詳細な経緯(実測値・事故事例)はdesign_docs/stage15_perf_20260707.html
        の該当節を参照(コード内コメントの重複を避けるため詳細はここに書かない)。

        2026-07-20追加(143節続報、フェーズ2): is_closing_trend(OpponentSituation、
        Phase1の既存判定を再利用)がTrueの間は、side_room/fwd_dlat等の静的な瞬時値
        だけを見るこの判定固有の盲点(P0①、0720-05実測wp139で確認)を塞ぐため、
        解放を即座にブロックする。デバウンスカウンタ自体は生値(_side_clear_raw)の
        推移をそのまま追い続け(トレンドが収まった瞬間に不要な再デバウンス待ちを
        発生させないため)、最終的な解放判定にのみis_closing_trendをANDで効かせる
        (壁際減速等、既存の「解放方向は緩やか・制限方向は即時」という非対称設計と
        同じ考え方)。新規パラメータ0個。"""
        _side_room = None
        if (self._ot_side != 0 and scan.get("fwd_vid") is not None
                and scan["fwd_vid"] == vtgt[4]):
            _side_room = left_free if self._ot_side > 0 else right_free
        _stopped_opponent = (fwd_vopp is not None
                              and fwd_vopp < self._opp_obstacle_speed)
        _side_room_ok_now = (_side_room is not None
                              and _side_room >= self._along_min_width)
        _fwd_dlat_now = scan.get("fwd_dlat")
        _actual_lat_clear_now = (_fwd_dlat_now is not None
                                  and _fwd_dlat_now >= self._along_min_width)
        _offset_committed = self._ot_alpha >= 1.0 - 1e-3
        _side_clear_raw = (_side_room is not None
                            and _side_room >= self._along_lane_need
                            and self._ot_cleared) or (
                                _stopped_opponent and _side_room_ok_now
                                and (_actual_lat_clear_now or _offset_committed))
        if _side_clear_raw:
            self._g2_clear_on_count += 1
        else:
            self._g2_clear_on_count = 0
            self._g2_release_debounced = False
        if (not self._g2_release_debounced
                and self._g2_clear_on_count >= self._ot_engage_debounce):
            self._g2_release_debounced = True
        _side_clear = self._g2_release_debounced and not is_closing_trend
        if _side_clear != self._g2_release_prev:
            self.get_logger().info(
                f"[G2-RELEASE] {'ON' if _side_clear else 'OFF'} "
                f"raw={_side_clear_raw} on_count={self._g2_clear_on_count} "
                f"side_room={_side_room} ot_cleared={self._ot_cleared} "
                f"stopped_bypass={_stopped_opponent and _side_room_ok_now and (_actual_lat_clear_now or _offset_committed)} "
                f"actual_lat_clear_now={_actual_lat_clear_now} "
                f"offset_committed={_offset_committed} alpha={self._ot_alpha:.3f} "
                f"is_closing_trend={is_closing_trend} "
                f"fwd_dlat={scan['fwd_dlat']} fwd_ds={scan['fwd_ds']} "
                f"vopp={fwd_vopp} v_safe_prev={v_safe_pre_for_log} "
                f"wp={self._mpc.model.wp_id}")
            self._g2_release_prev = _side_clear
        return _side_clear

    def _f3_taper_speed(self, vtgt, eff_v_cap: float, vlim: float) -> float:
        """2026-07-20抽出(143節続報、スリム化)。F3クリープ床(2026-07-13〜15節、
        停止/低速車の後ろでのデッドロック・バンバン振動対策)の計算を_control()
        から1メソッドへ抽出した。ロジック・[F3-TAPER]ゾーン遷移ログとも完全に
        無変更。詳細な経緯はdesign_docs参照。"""
        _est_gap = float(vtgt[0]) + float(vtgt[3])
        if _est_gap >= self._ot_f3_taper_gap:
            _floor = self._ot_v_creep
        elif _est_gap <= self._ot_hard_stop_gap:
            _floor = 0.0
        else:
            _frac = ((_est_gap - self._ot_hard_stop_gap)
                      / (self._ot_f3_taper_gap - self._ot_hard_stop_gap))
            _floor = self._ot_v_creep * _frac
        _v_safe = min(eff_v_cap, max(_floor, vlim))
        _f3_zone = ("stop" if _est_gap <= self._ot_hard_stop_gap
                    else "creep" if _est_gap >= self._ot_f3_taper_gap
                    else "taper")
        if _f3_zone != self._f3_taper_zone_prev:
            self.get_logger().info(
                f"[F3-TAPER] zone={_f3_zone} est_gap={_est_gap:.2f} "
                f"floor={_floor:.2f} v_safe_pre={_v_safe:.2f} "
                f"vopp={vtgt[2]:.2f} wp={self._mpc.model.wp_id}")
            self._f3_taper_zone_prev = _f3_zone
        return _v_safe

    def _control(self):
        if not hasattr(self, '_pf_acc'):
            self._pf_init()
        self._pf_last = _time.perf_counter()
        now = self.get_clock().now()
        t = (now - self._t_start).nanoseconds / 1e9
        dt = (now - self._last_t).nanoseconds / 1e9

        self._last_t = now
        self._loop += 1

        # record and print execution stats
        if self.use_stats:
            self._stats.record()

        self._control_rate.sleep()
        self._pf_mark('sleep')
        _pf_work0 = _time.perf_counter()

        if self._loop % 100 == 0:
            # update reference path
            if self._cfg.reference_path.update_by_topic: # type: ignore
                _t0 = _time.perf_counter()
                new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                if new_referece_path is not None:
                    self._car.reference_path = new_referece_path
                    self._car.update_reference_path(self._car.reference_path)
                self._pf_add('refresh', _time.perf_counter() - _t0)

            def plot_reference_path(car):
                import matplotlib.pyplot as plt
                import sys
                fig, ax = plt.subplots(1, 1)
                car.reference_path.show(ax)
                plt.show()
                sys.exit(1)
            # plot_reference_path(self._car)

        if self.USE_OBSTACLE_AVOIDANCE and self._obstacles_updated:
            _t_raster = _time.perf_counter()
            self._obstacles_updated = False
            filtered_dynamic = self._filter_obstacles_to_corridor(self._dynamic_obstacles)
            # 診断用(2026-07-09): コリドー計算(corr)コストが障害物点数に連動するか検証するため、
            #   実際にコリドーへ投入される障害物数(=filtered_dynamic、全追跡台数のn_obsとは別物、
            #   コリドー近傍に絞られた後の点数)を保持し、[OT]ログへ出力する。
            self._dbg_n_dynobs = len(filtered_dynamic)
            # P1: 再描画デッドバンド。前回描画から全障害物の移動が閾値未満なら再描画しない。
            #   V2X(≈13Hz)毎の微小ゆらぎで占有マップ→コリドー(ub/lb)が跳び、ステアがパタつくのを防ぐ。
            #   台数変化 or どれか1つでも閾値以上動いたら従来通り再描画（安全側）。
            _new_pos = [(ob.cx, ob.cy) for ob in filtered_dynamic]
            _prev = getattr(self, "_last_rasterized_pos", None)
            _db = getattr(self, "_rebuild_deadband", None)
            if _db is None:
                _db = float(getattr(self._cfg.v2x_obstacle_avoidance, "rebuild_deadband", 0.3))  # type: ignore
                self._rebuild_deadband = _db
            _need = (_prev is None or len(_prev) != len(_new_pos) or any(
                (ax - bx) ** 2 + (ay - by) ** 2 > _db * _db
                for (ax, ay), (bx, by) in zip(_new_pos, _prev)))
            if _need:
                self._map.reset_map()
                self._map.add_obstacles(self._static_obstacles + filtered_dynamic)
                self._reference_path.reset_dynamic_constraints()
                self._last_rasterized_pos = _new_pos
            self._pf_add('raster', _time.perf_counter() - _t_raster)

        is_colliding = False
        if self._last_colliding_time is not None:
            elapsed_from_last_colliding = (now - self._last_colliding_time).nanoseconds / 1e9
            if elapsed_from_last_colliding < 5.0:
                is_colliding = True

        # センシング切り分け計装D(2026-07-19、118節続報): USE_OBSTACLE_AVOIDANCEの
        #   有無に関わらず常に評価する(オーバーテイク非依存の一般センシング診断のため)。
        #   間引きは既存の「約1秒ごと」イディオム(4172行目 _opp_map_pub_loop と同じ考え方)。
        if self._loop % int(max(1, self._mpc_cfg.control_rate)) == 0:
            self._maybe_log_gnss_ekf_xcorr()
            self._maybe_log_steer_xcorr()
        # 208節続報: ピーク値追跡のため、上記の1秒間引きとは別に毎周期呼ぶ
        # (ログ自体は対象wp通過時のみ、診断専用で制御へは無影響)。
        self._maybe_log_hotspot_deviation()

        pose = odom_to_pose_2d(self._odom) # type: ignore
        pose = self._apply_pit_localization(pose)  # ピット内のみGNSS位置で補正(位置のみ)
        v = self._odom.twist.twist.linear.x

        # スタック検知バック(2026-07-09): 復帰中は他の全処理をバイパスし、
        #   ギア+固定コマンドを直接publishする(Pattern A: MPCを一時的に手放す)。
        if self._stuck_state != "NORMAL":
            # 復帰中(ギア切替・後退加速等)は正当な大きな速度変化が起きるため、
            #   衝突検知の比較対象をリセットし、NORMAL復帰直後の誤検知を防ぐ。
            self._collision_check_v_prev = None
            self._collision_v_window.clear()
            self._handle_stuck_recovery(now, pose)
            return
        # 自動衝突検知(2026-07-10追加): 1周期での実速度の急落を監視。閾値の根拠は__init__参照。
        if (self._collision_check_v_prev is not None
                and (v - self._collision_check_v_prev) < -self._collision_suspect_dv):
            self.get_logger().warn(
                f"[COLLISION-SUSPECTED] v drop {self._collision_check_v_prev:.2f}"
                f"->{v:.2f} m/s in 1 cycle")
        self._collision_check_v_prev = v
        # 累積版(2026-07-11追加): 低速域での接触は1周期0.8m/s超に届かず複数周期に分散する
        #   可能性がある(0711ローカルで2件の衝突を目視確認したがCOLLISION-SUSPECTEDが
        #   1度も発火しなかったため)。直近collision_cum_window周期の最大値からの下落幅を見る。
        self._collision_v_window.append(v)
        if len(self._collision_v_window) == self._collision_v_window.maxlen:
            _cum_drop = max(self._collision_v_window) - v
            if _cum_drop >= self._collision_suspect_cum_dv:
                self.get_logger().warn(
                    f"[COLLISION-SUSPECTED-CUM] v drop {_cum_drop:.2f} m/s over "
                    f"{self._collision_v_window.maxlen} cycles (window={list(self._collision_v_window)})")
                self._collision_v_window.clear()
        _v_odom_now = abs(v)
        # 起動猶予期間(2026-07-10追加): 過去の予選ログ実測で、START後の実発進まで
        #   一貫して7.5〜8.75秒かかる(P→D待ち等の正常な起動シーケンス)。この間は
        #   指令速度>実速度の乖離が生じるのが正常なため、スタック判定を行わない
        #   (実測: 猶予なしだとSTART後わずか3秒で誤検知していた)。
        _since_start_s = (now - self._t_start).nanoseconds / 1e9
        if _since_start_s < self._stuck_startup_grace_s:
            self._stuck_count = 0
            self._stuck_stall_count = 0
            self._ghost_block_logged = False
        else:
            if self._stuck_u0_last > self._stuck_u0_thr and _v_odom_now < self._stuck_v_thr:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
                self._ghost_block_logged = False
            # GHOST-BLOCK(2026-07-11追加): 「知覚と現実の乖離」の早期検知。経路1と同じ
            #   カウンタ(u0高いのに動けない)を、経路1本発動(stuck_hold_cycles=120=3秒)より
            #   軽い閾値(ghost_block_hold_cycles、既定40=1秒)でログのみ出す。復帰動作は
            #   起こさない(既存の経路1/2/3の判定・挙動には一切影響しない)。
            #   実測(0711ローカル d2): fwd=0・コリドーwmin=4.7m(モデル上は正常)なのに
            #   実速度が0まで落ちて経路1が発動した事例があり、3秒待たずに前兆を記録し
            #   自己回復するケース(経路1まで至らない近接遭遇)も含めて可視化する狙い。
            if (self._stuck_count >= self._ghost_block_hold_cycles
                    and not self._ghost_block_logged):
                self._ghost_block_logged = True
                # 2026-07-20追加(139節続報、致命的な「前方車無しで動けない」謎の
                #   診断強化): 0720-03実測(wp187、72秒間の完全停止)で、fwd=0・
                #   n_dynobs=0(追跡上は近くに誰もいない)・EKF横偏差も小さい
                #   (0.16〜0.25m)にもかかわらず、前進も後退も実速度がほぼ0のまま
                #   72秒続く事例を発見した。当初「対戦車に挟まれている」と推測
                #   したが、opp座標を参照経路の弧長へ逆算した結果egoから約73m
                #   離れておりこの仮説は否定された(訂正済み、design_docs 140節)。
                #   真因(AWSIM上の物理的な引っ掛かり/アクチュエータ飽和/未知の
                #   固着)を次回ログで特定するため、既存の値のみを集約する
                #   (新規購読・新規スキャン処理0個): ①ギア状態(self._gear_report、
                #   STUCK-BACKUPが既に使う_gear_label再利用)、②直近の操舵指令
                #   (self._last_u[1]、既存の低域通過フィルタ後の値)、③MPC不可解
                #   カウンタ(self._mpc.infeasibility_counter)、④動的コリドー
                #   ([OT]ログと同じdbg_corr_ub0/lb0の取得方法)、⑤egoの現在位置
                #   における占有格子の値(self._map.data、モデルが「壁の中にいる」
                #   と誤認識していないかを直接確認する)、⑥生のpose(x/y/theta、
                #   mcapの実位置と突き合わせるため)。
                _occ_val = None
                try:
                    _px, _py = self._map.w2m(pose.x, pose.y)
                    if 0 <= _py < self._map.data.shape[0] and 0 <= _px < self._map.data.shape[1]:
                        _occ_val = int(self._map.data[_py, _px])
                except Exception:
                    pass
                self.get_logger().warn(
                    f"[GHOST-BLOCK] u0_last={self._stuck_u0_last:.2f} v={_v_odom_now:.2f} "
                    f"count={self._stuck_count} wp={self._mpc.model.wp_id} "
                    f"ot_state={self._ot_state} "
                    f"gear={self._gear_label(self._gear_report.report)} "
                    f"steer_cmd={self._last_u[1]:.3f} "
                    f"infeas={self._mpc.infeasibility_counter} "
                    f"corr_ub0={getattr(self._mpc, 'dbg_corr_ub0', float('nan')):.2f} "
                    f"corr_lb0={getattr(self._mpc, 'dbg_corr_lb0', float('nan')):.2f} "
                    f"pose_x={pose.x:.2f} pose_y={pose.y:.2f} pose_theta={pose.theta:.3f} "
                    f"occ_at_pose={_occ_val} "
                    f"opp[{self._opp_snapshot_str()}]")
            # 経路3のカウント。実速度が観測された時点でそのスタック"episode"は解消したとみなし、
            #   リトライ予算の起点(_stuck_stall_first_trigger_time)もここでリセットする
            #   (別の場所・別の相手で新たに完全停止した場合に、前回の予算消費を持ち越さないため)。
            if _v_odom_now < self._stuck_stall_v_thr:
                self._stuck_stall_count += 1
            else:
                self._stuck_stall_count = 0
                self._stuck_stall_first_trigger_time = None
                self._stuck_stall_budget_exhausted_logged = False
        # 経路2(2026-07-10追加): infeasibility_counterが張り付いたまま長時間続く詰まり。
        #   実測(2026-07-10未明ログ): 復帰直後にu0=0のままinfeasが31秒間・1649まで際限なく
        #   増加し続けた事例を確認。H4-lite(マージン緩和, unlock_after=80)は動作していたが
        #   それだけでは解けなかった。既存の経路1(u0高いのに動けない)は「MPCが指令を出せて
        #   いる」場合のみ捕捉するため、「解自体が出せずu0=0になる」この詰まり方は素通りして
        #   いた。infeasibility_counterはそれ自体が連続失敗周期数を表すため、追加のホールド
        #   カウンタは設けず、H4-liteに猶予を与えた上でこの閾値に達したら即座に発動する。
        _infeas_stuck = (_since_start_s >= self._stuck_startup_grace_s
                          and self._mpc.infeasibility_counter >= self._stuck_infeas_thr)
        # 経路3(2026-07-10追加): 相手車(複数)に塞がれ続け、MPCが正しく安全停止を選び続ける
        #   (u0=0・infeas=0のまま)デッドロック。経路1/2はどちらもu0またはinfeasの異常値を
        #   前提とするため構造的に発動しない(実測: 予選0710-02で363秒間完全停止・0周)。
        #   u0/infeasを問わず実速度のみで判定する。
        _stall_stuck = (_since_start_s >= self._stuck_startup_grace_s
                         and self._stuck_stall_count >= self._stuck_stall_hold_cycles)
        if self._stuck_count >= self._stuck_hold_cycles or _infeas_stuck:
            self._stuck_trigger_path = 1 if self._stuck_count >= self._stuck_hold_cycles else 2
            self.get_logger().warn(
                f"[STUCK] detected (u0_last={self._stuck_u0_last:.2f} "
                f"v={_v_odom_now:.2f} count={self._stuck_count} "
                f"infeas={self._mpc.infeasibility_counter} path={self._stuck_trigger_path}) -> WAIT_REVERSE")
            self._stuck_update_shuffle_cycle(now, pose)  # 184節追加
            self._stuck_state = "WAIT_REVERSE"
            self._stuck_count = 0
            self._stuck_stall_count = 0
            self._stuck_gear_wait_count = 0
            self._ghost_block_logged = False
            self._handle_stuck_recovery(now, pose)
            return
        elif _stall_stuck:
            # リトライ予算(2026-07-10, ユーザー承認済み): 経路3の初回発火から
            #   stall_retry_budget_s(既定360s=6分)以内のみPUSHを再試行する(無限リトライ回避)。
            _budget_ok = True
            if self._stuck_stall_first_trigger_time is None:
                self._stuck_stall_first_trigger_time = now
            else:
                _elapsed_budget = (now - self._stuck_stall_first_trigger_time).nanoseconds / 1e9
                _budget_ok = _elapsed_budget <= self._stuck_stall_retry_budget_s
            if _budget_ok:
                self._stuck_trigger_path = 3
                self.get_logger().warn(
                    f"[STUCK] 完全停止検知(v={_v_odom_now:.2f} count={self._stuck_stall_count} "
                    f"path=3) -> WAIT_REVERSE(→PUSH予定)")
                self._stuck_update_shuffle_cycle(now, pose)  # 184節追加
                self._stuck_state = "WAIT_REVERSE"
                self._stuck_count = 0
                self._stuck_stall_count = 0
                self._stuck_gear_wait_count = 0
                self._ghost_block_logged = False
                self._handle_stuck_recovery(now, pose)
                return
            elif not self._stuck_stall_budget_exhausted_logged:
                self._stuck_stall_budget_exhausted_logged = True
                self.get_logger().warn(
                    "[STUCK] 経路3リトライ予算(stall_retry_budget_s)を超過。PUSH再試行を打ち切ります")

        # ピット→コースイン切替: ピット経路走行中、レースラインに十分近づいたら
        # （＝合流＝コースイン）、レースライン(周回)へ一度だけ切り替える（ヒステリシス付き）。
        if self._pit_enable and self._on_pit:
            _d_race = self._race_line_min_dist(pose.x, pose.y)
            if _d_race < self._pit_course_in_dist:
                self._pit_course_in_acc += 1
                if self._pit_course_in_acc >= self._pit_course_in_count:
                    self._set_active_path(self._race_ref_path, on_pit=False)
                    self.get_logger().info(
                        f"COURSE-IN (race-line dist={_d_race:.1f}m) -> switch to race line")
            else:
                self._pit_course_in_acc = 0

        # 2026-07-14追加(水平展開: 他車と同型の壁越え誤認識バグを自車側でも確認):
        #   update_states内部のget_closest_waypointは従来、全waypointからの単純な
        #   (x,y)最近傍探索で、弧長連続性を見ていなかった(_closest_wp_and_s修正前と
        #   全く同じ構造)。以前は「MPC空間モデルが逐次的にwp_idを管理しているため
        #   壁越え耐性が既にある」という前提を下のコメントに書いていたが、これは誤りで
        #   あったと判明。既存_wp_match_radius_m(V2X position_jump_threshold流用)を
        #   ここでも再利用し、前回wp_idを基準にした窓探索へ切り替える。
        _prev_wp_id = int(self._mpc.model.wp_id)
        self._car.update_states(pose.x, pose.y, pose.theta,
                                 prev_idx=_prev_wp_id, radius_m=self._wp_match_radius_m)

        # --- gate2: 前方車解析＋ステートマシン（NORMAL/OVERTAKING/STOPPING）---
        # 方式A: OVERTAKING/NORMAL では回避ON維持で、MPCのコリドーが毎周期ライブのV2X位置から
        # 空き側を自動選択する（動く相手にも追従）。STOPPING(=両側塞がり, gate1相当)では従来通り
        # 回避OFF＋v_safeで減速停止し、gate1の挙動を保全する。
        _v_safe_pre = None
        # 診断用(2026-07-09): 速度上限は複数の機構(ICC/G-2/コーナー減速/壁際防御/並走ねばり)が
        #   min()で連鎖し、最終値だけでは「どれが効いたか」が分からずデバッグが困難だった。
        #   各機構が候補を出す箇所で (名前, 値) を積み、最後に最小値の出所をログへ出す。
        _v_safe_cand = []
        _fwd_dbg = {}
        if self.USE_OBSTACLE_AVOIDANCE:
            try:
                _v_odom = abs(self._odom.twist.twist.linear.x)
            except Exception:
                _v_odom = 0.0
            # 自車の現在 e_y(統一スキャン・psi_bias・壁ガバナで共有。+左/-右)
            # 2026-07-14追加: MPC空間モデルのwp_id(直上のupdate_states呼び出しで、
            #   同じ窓探索により壁越え耐性を持つよう修正済み)を探索窓の基準点として
            #   再利用する。新規の状態変数は増やさない。
            _idx, _ = self._closest_wp_and_s(
                pose.x, pose.y, prev_idx=int(self._mpc.model.wp_id))
            _wp = self._reference_path.waypoints[_idx]
            _cur_ey = float(np.cos(_wp.psi) * (pose.y - _wp.y)
                            - np.sin(_wp.psi) * (pose.x - _wp.x))

            # 診断用(2026-07-19、センシング切り分け): 「コーナーで狙いが狂う」がEKF起因
            #   (design_docs記録済みのS2、コーナーでの横位置過小報告)か制御(追従)起因かを
            #   次回ログで切り分けるため、同一waypoint基準・同一射影式でGNSS生値の
            #   e_yも独立に計算し、EKF側と並べて記録する。self._gnss_pose は既存購読
            #   (元々ピット内補正用)を再利用し、新規トピック・新規パラメータは追加しない。
            #   周期はデバッグログ(4366/4372行目)と同じ既存の間引きイディオムを再利用。
            if (self._gnss_pose is not None
                    and self._loop % (self._mpc_cfg.control_rate // 4) == 0):
                _gp = self._gnss_pose.pose.pose.position
                _gnss_ey = float(np.cos(_wp.psi) * (_gp.y - _wp.y)
                                  - np.sin(_wp.psi) * (_gp.x - _wp.x))
                self.get_logger().info(
                    f"[LOC-XCHECK] wp={_idx} kappa={_wp.kappa:.3f} "
                    f"ekf_ey={_cur_ey:.3f} gnss_ey={_gnss_ey:.3f} "
                    f"v={_v_odom:.2f} ot={self._ot_state}")

            # === 統一検知: 唯一の他車スキャン(全ての判断がこの結果を共有) ===
            self._pf_mark('prep')
            _scan = self._scan_traffic(_v_odom, _cur_ey)
            # 2026-07-09追加: 'traffic_ot'一括計装(scan~along_lat全体)を細分化。
            #   STOPPING状態が欠落と過剰相関(時間占有14.5%に対し欠落23.0%)する原因区間を
            #   特定するため、scan_traffic/follow_speed_limit/壁際/along_latを個別計測する。
            self._pf_mark('scan')
            _n_fwd = len(_scan["cars"])
            _left_free = _scan["left_free"]; _right_free = _scan["right_free"]
            # 被追い越し判定のデバウンス(2026-07-05): 生値は |ds|≈3m・dlat≈1.0m・v≈6km/h の
            #   境界素通し判定で、追従中はまさにその境界付近の幾何になり毎周期 True/False が
            #   フラップする。この値は STOPPING の use_obstacle_avoidance(=コリドー生成元
            #   動的↔静的)を直接切り替えるため、フラップ=横コリドーの離散ジャンプ=蛇行になる。
            #   他の状態遷移と同様にヒステリシスを付与(ON=def_enter_cycles連続/OFF=def_exit_cycles連続)。
            if _scan["being_overtaken"]:
                self._def_on_count += 1
                self._def_off_count = 0
            else:
                self._def_off_count += 1
                self._def_on_count = 0
            if (not self._def_active
                    and self._def_on_count >= self._def_enter_cycles):
                self._def_active = True
            elif (self._def_active
                    and self._def_off_count >= self._def_exit_cycles):
                self._def_active = False
            _being_overtaken = self._def_active
            _fwd_vopp = _scan["fwd_vopp"]; _fwd_ds = _scan["fwd_ds"]

            # 2026-07-20追加(127節続報、0720-01予選ログwp173の異常接近分析): LAT-TTCの
            #   space/opp_space(_scan_traffic内のlf/rf、壁〜相手の隙間の広さ)は自車の
            #   現在位置を式に含まず、fwd_dlat(自車〜相手の実測横間隔)が0.2m級まで
            #   縮んでいても3m超の「安全」を報告する矛盾が実測で確認された(126/127節)。
            #   また縦間隔(fwd_ds)には対応する「車体全長ベースの物理下限」判定が
            #   一度も存在しなかった。fwd_dlat<along_min_width(既存、両車半幅合計)かつ
            #   fwd_ds<along_min_length(新規、両車半長合計)を「実際に車体が重なる
            #   リスクがある」状態と定義する。_lat_ttc.update()呼び出し前に計算し、
            #   giveup判定・v_safe合成の両方で使い回す(新規スキャン処理0個、既存の
            #   _scan結果を再利用するのみ)。
            _fwd_dlat_val = _scan["fwd_dlat"]
            # 2026-07-22追加(issue⑤②、ENGAGE即時再失敗ループの対処): footprint_risk
            #   本体(ds<along_min_length)と154節のfootprint_taper(along_min_length<=
            #   ds<ot_pass_clear)を合わせた「危険域全体」(dlatが狭いままds<pass_clear)
            #   を1回だけ計算し、_footprint_risk本体・下記のcooldown解除判定(152節)・
            #   後段のfootprint_taper判定の3箇所で同じ値を再利用する(新規スキャン処理
            #   0個)。単一の場所で定義することで、taper側の閾値が将来変わっても
            #   3箇所が自動的に同期する(159節と同じ「同じ周期の同じ値を使う」原則)。
            _fp_near_zone = (_fwd_dlat_val is not None and _fwd_ds is not None
                              and _fwd_dlat_val < self._along_min_width
                              and abs(_fwd_ds) < self._ot_pass_clear)
            _footprint_risk = _fp_near_zone and abs(_fwd_ds) < self._along_min_length

            # 横方向TTC監視(2026-07-11実装、2026-07-12実挙動統合): 1周期1回だけ呼ぶ
            #   (以前はここより後段でも呼んでおり、内部のEMA/連続縮小カウンタが二重更新
            #   される潜在バグがあった)。ここで計算した_lat_decを、この後のgiveup判定・
            #   v_safe合成・ログの全てで使い回す。有効フラグに関わらず毎周期計算・ログは
            #   継続する(実挙動へ反映するかどうかだけをフラグで切り替える設計のため)。
            if self._ot_side == 0:
                self._lat_ttc.reset_episode()
            _lat_space = _left_free if self._ot_side > 0 else _right_free
            _lat_opp_space = _right_free if self._ot_side > 0 else _left_free
            # 2026-07-16追加(79節): 反転先(-side)が直近のカーブで閉じるかどうかを、
            #   LateralTTCMonitor.update()を呼ぶ前に計算して渡す。旧実装(77節)は
            #   update()の戻り値(側反転)を受け取った後に別途veto判定しており、
            #   has_switched/has_rescuedが既に消費された後だったため、vetoされる
            #   たびにこのエピソードの反転トークンを浪費していた(0715-08実測で
            #   確認、詳細はdesign_docs 78/79節)。判定をupdate()の入力に一本化する
            #   ことで、_switchback_eligible/_rescue_eligible自体を不成立にし、
            #   トークンを温存する。
            _new_side_blocked = (self._switchback_curvature_veto(-self._ot_side)
                                  if self._ot_side != 0 else False)
            # 2026-07-22追加(157節、0722-03予選ログ分析): _switchback_curvature_veto
            #   は静的トラック曲率のみを見ており、その瞬間の実測空き(_lat_opp_space、
            #   反転先の壁〜相手の隙間)が既に十分広い場合でも一律に反転を抑制していた
            #   (0721-02実測でC2緊急giveupの約50〜56%がこのパターン、0722-03 wp75-77
            #   でも再確認)。通常のswitchback自体が既に「反転先の実測opp_spaceが
            #   switchback_space_m以上」を要求している(498-500行目付近)ため、この
            #   既存の実測ベース閾値を満たしている場合に限り、静的曲率の懸念を上書き
            #   する。82/83節の教訓(switchback適格性の広範な制限緩和は重大な回帰を
            #   招いた)を踏まえ、適格性判定式自体(lateral_ttc_monitor.py側)には
            #   一切手を入れず、既に検証済みの閾値を満たす場合のみに限定した狭い
            #   上書きとする。新規パラメータ0個(既存switchback_space_m・既存
            #   _lat_opp_spaceを再利用)。
            _new_side_curvature_override = (
                _lat_opp_space is not None
                and _lat_opp_space >= self._lat_ttc.switchback_space_m)
            # 2026-07-20追加(125節、A-1): _switchback_curvature_veto(静的kappaのみ)
            #   と同じスコープ(通常switchback/A_lookahead、厳密なA_rescue両方)へ、
            #   動的コリドー(壁+占有格子込み)ベースの壁veto(new_side自体には
            #   依存しない、MPCが計画中の経路の幅チェック)を追加する。
            _new_side_wall_blocked = (self._switchback_wall_veto()
                                       if self._ot_side != 0 else False)
            # 2026-07-22追加(159節、判定層と実行層の指標不一致対策): A_rescue/switchback
            #   の可否判定(space/opp_space・new_side_wall_blocked=コリドー全体幅)は、
            #   実際にオフセット目標を動かす層(_corr_bound_ahead=反転先方向への実測
            #   先読み最小値)とは異なる指標を使っていた。0722-03実測で、A_rescueが
            #   成立し_ot_sideが反転したにも関わらず、直後のcorr_bound_ahead(新側)が
            #   負値でオフセット目標が実質ゼロへクランプされ、「側だけ反転し車両は
            #   動かない」という内部矛盾状態が発生し、対象車切替→footprint_risk誘発の
            #   連鎖を招いた(157/158節)。判定層に実行層と全く同じ関数・同じ閾値
            #   (along_min_width、既存の物理下限)を追加することで両層を一致させる。
            #   dbg_corr_ub_arr/lb_arrはこの周期のget_control()(4831行目付近、この
            #   時点ではまだ呼ばれていない)が更新するまで前周期の値のまま凍結されて
            #   いるため、ここで計算した値と4173行目付近のオフセット目標計算(反転成立後は
            #   同じ側を引数に取る)は同一配列を参照する決定論的な関数として必ず
            #   一致する(明示的なキャッシュは不要)。新規パラメータ0個。
            _new_side_corr_bound = (self._corr_bound_ahead(-self._ot_side)
                                     if self._ot_side != 0 else float('inf'))
            _new_side_offset_blocked = _new_side_corr_bound < self._along_min_width
            # 2026-07-16追加(84節②、カーブ先回り切り替え、ユーザー承認済み設計):
            #   _switchback_curvature_veto()は「反転先が閉じるか」しか見ていなかったが、
            #   同じ関数を現在側(self._ot_side)へ適用すれば「現在側がこの先閉じるか」も
            #   新規スキャン処理無しで分かる。現在側が閉じ、かつ反対側(_new_side_blocked)
            #   は閉じないと分かっている場合のみ、通常のmargin/cleared_margin判定を
            #   待たずに早めの反転(branch=A_lookahead)を許可する信号として渡す。
            #   直線走行中(前方にきついカーブが無い)は_current_side_closing_ahead自体が
            #   Falseのままのため、無駄な反転(コーナーと無関係な左右切替)は増えない。
            _current_side_closing_ahead = (self._switchback_curvature_veto(self._ot_side)
                                            if self._ot_side != 0 else False)
            # 2026-07-22修正(157節): _new_side_curvature_override成立時はlookahead
            #   経路(84節②)でも通常経路(544行目)と同じ基準で反転を許可し、両経路の
            #   挙動を一致させる(新規計算なし、上で算出済みの値を再利用)。
            _lookahead_favor_switch = _current_side_closing_ahead and (
                not _new_side_blocked or _new_side_curvature_override)
            # 2026-07-18追加(100節、Tier1裁定の外出し): 旧update()引数
            #   fwd_is_obstacle_class(92節続報)を廃止し、ここで裁定用ローカル変数
            #   として保持する(既存opp_obstacle_speed閾値の再利用、新規スキャン処理
            #   0個)。LAT-TTCのC1候補(v_safe_cap)自体は障害物クラスに関わらず常に
            #   計算されて返るため、この変数はv_safe候補集約側([TIER1-C1-YIELD]、
            #   後述)で「その値を使うかどうか」を裁定するためだけに使う。
            _fwd_is_obstacle_class = (_fwd_vopp is not None
                                       and _fwd_vopp < self._opp_obstacle_speed)
            # 2026-07-18追加(107節案C、103節Phase 1): A_rescue_relaxed(最終救済の
            #   適格緩和)の判定「前」に、反転先(-side)について対象車両ID込みの
            #   room先読みを計算する。103節Phase 0の事後診断呼び出し(旧:
            #   branch=="A_rescue_relaxed"確定後に計算・[REVERSE-ROOM-CHECK]ログ
            #   のみ)をここへ統合し、同じ結果を(a)update()へのveto入力、
            #   (b)診断ログ、の両方に使い回す(呼び出し回数は従来通り1回、
            #   新規計算コストなし)。has_switched/is_side_by_side等でどのみち
            #   反転しない周期にも毎回計算することになるが、_opponent_room_ahead
            #   自体が新規スキャン処理0個(既存_fwd_max_consider窓の再利用)で
            #   設計済みのため、CPU予算への追加影響はない。
            _room_min, _room_wp, _room_n = (
                self._opponent_room_ahead(
                    _scan["fwd_vid"], self._mpc.model.wp_id, -self._ot_side,
                    self._fwd_max_consider)
                if self._ot_side != 0 else (None, None, 0))
            _new_side_room_blocked = (_room_min is not None
                                       and _room_min < self._along_min_width)
            # 2026-07-26追加(191節、AXIS03: switchback/A_rescueの縦方向盲点対処):
            #   既存のfootprint_risk本体(_fwd_ds由来)と同じalong_min_length閾値を、
            #   fwd_dlatは問わず縦距離のみで判定する(反転可否そのものを塞ぐため)。
            #   新規パラメータ0個(既存along_min_lengthを再利用)。
            _fwd_ds_overlap_risk = (_fwd_ds is not None
                                     and abs(_fwd_ds) < self._along_min_length)
            _lat_dec = self._lat_ttc.update(
                side=self._ot_side, space=_lat_space, opp_space=_lat_opp_space,
                fwd_dlat=_scan["fwd_dlat"], fwd_ds=_fwd_ds, vopp=_fwd_vopp, dt=dt,
                fwd_vid=_scan["fwd_vid"], cleared=self._ot_cleared,
                new_side_blocked=_new_side_blocked,
                new_side_curvature_override=_new_side_curvature_override,
                lookahead_favor_switch=_lookahead_favor_switch,
                # 2026-07-17追加(92節①): 既存の_current_side_closing_ahead(84節②で
                #   lookahead用に算出済み、新規スキャン処理0個)をそのまま再利用する。
                current_side_closing_ahead=_current_side_closing_ahead,
                new_side_room_blocked=_new_side_room_blocked,
                new_side_wall_blocked=_new_side_wall_blocked,
                new_side_offset_blocked=_new_side_offset_blocked,
                footprint_risk=_footprint_risk,
                fwd_ds_overlap_risk=_fwd_ds_overlap_risk)

            # 2026-07-20追加(141節、フェーズ1): 共有状況スナップショットをここで
            #   1回だけ構築する(_lat_dec確定直後)。以降のENGAGEゲート等はこれを
            #   参照する。詳細はOpponentSituationのdocstring参照。
            _opp_sit = self._build_opponent_situation(_scan, _lat_dec, _footprint_risk)

            # 2026-07-20追加(132節、Gap①Phase0、診断専用・ENGAGE判定への影響なし):
            #   side==0(未エンゲージ)の間にfwd_dlat縮小トレンドが確立した瞬間
            #   (立ち上がりのみ)を記録する。0720-02実測wp284(giveup直後に同一対象車
            #   d3を4.2秒後に再エンゲージし0.55秒後に強制giveup)で、その間の間合いの
            #   推移が一切記録されていなかったことへの対処。次回予選ログでこの値を
            #   実測し、閾値を数値検証してからPhase1(ENGAGE判定への配線)を検討する。
            _dlat_trend_alert = (self._ot_side == 0
                                  and _lat_dec.dlat_shrink_run >= self._lat_ttc.min_trend_cycles)
            if _dlat_trend_alert and not self._dlat_trend_alert_active:
                self.get_logger().info(
                    f"[DLAT-TREND] pre-engage closing trend detected "
                    f"fwd_vid={_scan['fwd_vid']} fwd_dlat={_scan['fwd_dlat']} "
                    f"dlat_v_ema={_lat_dec.dlat_v_ema:.3f} "
                    f"shrink_run={_lat_dec.dlat_shrink_run} wp={self._mpc.model.wp_id}")
            self._dlat_trend_alert_active = _dlat_trend_alert

            # チャタリング防止ヒステリシス(進入=min_gap / 維持=min_gap-gap_hys)
            _was_ot = (self._ot_state == "OVERTAKING")
            _thr = (self._ot_min_gap - self._ot_gap_hys) if _was_ot else self._ot_min_gap
            # 停止/低速車(<6km/h)対象の幅基準は物理下限(along_min_width)へ緩和
            #   (2026-07-05, 源流で一元化。2026-07-14再修正: フローチャートで洗い出した
            #   ギャップ①): 停止車は狭所に停まりがちで通常基準(2.5/2.0)では入れない/
            #   通り切れない。予選実測: エンゲージ緩和(1.85)と維持側(2.0)の不整合により
            #   「L=1.94の狭所に入れるが0.5秒でside_block強制離脱→4秒クールダウン→
            #   再接近…」の失敗ループ×11回。ここ(_thr)で緩和すれば進入・維持の両方に
            #   一貫し、入れる幅=通り切れる幅になる。走行車への基準(2.5/2.0)は不変。
            #   安全担保: クリープ+実効曲率減速+hard_stop_gap。
            #   2026-07-14再修正: 59節で_plan_pass内部(k_corner veto・min-width veto)を
            #   along_lane_need(1.85m)からalong_min_width(1.45m)へ緩和したが、この
            #   一つ手前の「安いゲート」は据え置きのままだった。幅1.5〜1.84mの区間は
            #   _plan_passへ到達する前にここで lr=0 として弾かれ、59節の緩和が
            #   無効化されていた(フローチャートのギャップ①)。同じ閾値に揃える。
            if (_fwd_vopp is not None
                    and _fwd_vopp < self._opp_obstacle_speed):
                _thr = min(_thr, self._along_min_width)
            _left_ok = (_left_free is not None and _left_free >= _thr)
            _right_ok = (_right_free is not None and _right_free >= _thr)
            # 安全STOPラッチ消化（>0の間はOVERTAKING禁止＝再挑戦抑制）
            if self._ot_infeasible_latch > 0:
                self._ot_infeasible_latch -= 1
            # 失敗離脱クールダウン消化(>0の間は再エンゲージしない=追従で仕切り直す)
            if self._ot_engage_cooldown > 0:
                self._ot_engage_cooldown -= 1
            # 2026-07-21追加(148節②): footprint_risk起因のcooldown中は、実際にfootprint_risk
            # 条件(_footprint_risk、上記で毎周期計算済み)が連続で不成立になった周期数を数える。
            # 既存の_ot_engage_debounce(フリッカー防止、8周期≈0.2秒)をそのまま再利用し、
            # 新規パラメータは追加しない。
            # 2026-07-22修正(issue⑤②): 判定基準を_footprint_risk本体(ds<along_min_length)
            #   から_fp_near_zone(footprint_taperの危険域込み、ds<ot_pass_clear)へ拡張。
            #   実測(0722-2ログ、d1 25件中大半)で、_footprint_risk自体が不成立になった
            #   直後(ただしtaper域=至近距離のまま)にcooldownが解除され、間合いが
            #   回復する間もなく即ENGAGE→1秒未満で再度footprint_risk、という無意味な
            #   再挑戦ループが繰り返されていた。危険域(taper込み)を完全に抜けるまで
            #   解除しないことで、次にENGAGEする時点で実際に間合いが回復していることを
            #   保証する。
            if self._ot_footprint_risk_gated:
                self._ot_footprint_risk_clear_count = (
                    0 if _fp_near_zone else self._ot_footprint_risk_clear_count + 1)

            # 攻めの価値判定(2026-07-04 純化): worth =「closingで追いつける相手か」のみ。
            #   closing は「自分が出せる速度(v_pot)」基準(C1: 現在速度だと追従減速で0になり
            #   永久に抜けなくなる)。地形(コーナー・完遂距離・側)は _plan_pass に一元化
            #   (旧: worth内コーナー分岐 + engage外側原則 の二重判定を統合)。
            _pass_worth = False
            if _fwd_vopp is not None and _fwd_ds is not None:
                _pass_worth = (_fwd_vopp < self._opp_obstacle_speed          # 障害物=常に抜く
                               or (self._v_pot - _fwd_vopp) > self._opp_min_closing)
            _fwd_dbg["vopp"] = round(_fwd_vopp * 3.6, 1) if _fwd_vopp is not None else None
            _fwd_dbg["pass_worth"] = int(_pass_worth); _fwd_dbg["def"] = int(_being_overtaken)
            # 2026-07-22追加(160節続報、issue⑤①: STOPPING中の能動的空き確保):
            #   _evaluate_engage_readiness()はOVERTAKING継続中・前方クリア確定時は
            #   呼ばれず_evalが未定義のままになるため、既定でNoneにしておき、
            #   後段(4133行目相当)で「今回計算されなかった=バイアス適用しない」
            #   という安全側デフォルトにする(古い周期の使い回しはしない)。
            _eval = None
            # 2026-07-24追加(168節): OVERTAKING継続中サブブランチで計算した委託側の
            #   corr_bound_ahead()を、後段のオフセットクランプ(4312行目相当)と共有し、
            #   同一値の二重計算を避けるための周期スコープ変数(非冗長性)。
            _room_ahead_locked = None
            if _n_fwd > 0:
                self._fwd_clear_count = 0
                if self._ot_state == "OVERTAKING":
                    # Fix-1 コミット: 一度エンゲージしたら「通過完了(NORMAL)/恒久失敗(infeasible)/
                    #   明確に無理(closing<giveup が giveup_cycles 連続)/側消失」まで維持。
                    #   1周期のworth反転でSTOPPINGへ落ちない(ランプ出戻り・Q切替の往復=ギクシャクの根絶)。
                    # 2026-07-17追加(94節、トークン整合性監査): scan_traffic の fwd_vopp は
                    #   ロック中の対象車ではなく「その周期で最も近い車」から毎回選び直される
                    #   ため、複数車が視界内にいる場面で対象車IDが入れ替わっても、closing判定
                    #   だけを見ているこのカウンタは気付かず連続扱いしてしまう。対象車IDが
                    #   変わった周期は仕切り直す(_ot_worth_countと同一の考え方)。
                    _fwd_vid_giveup = _opp_sit.fwd_vid
                    if _fwd_vid_giveup != self._ot_giveup_prev_vid:
                        self._ot_giveup_count = 0
                    self._ot_giveup_prev_vid = _fwd_vid_giveup
                    if (_opp_sit.fwd_vopp is not None
                            and _opp_sit.fwd_vopp >= self._opp_obstacle_speed
                            and (self._v_pot - _opp_sit.fwd_vopp) < self._opp_giveup_closing):
                        self._ot_giveup_count += 1
                    else:
                        self._ot_giveup_count = 0
                    # 側コミット(2026-07-04): 側は engage で1回だけ選ぶ。mid-pass再選択は廃止
                    #   (コーナー/自オフセット移動で left/right_free が数周期凹むたび反対側へ
                    #   スイング(±3m往復)→gap拡大→スタック の根治)。ロック側が持続的に塞がれた
                    #   時は反対側へ飛ばず追従へ離脱(真後ろ≤6mから始める今、mid-pass側変更は
                    #   幾何的にほぼ失敗確定。正しい応答はリトライ)。
                    _locked = self._ot_side_locked
                    # 2026-07-12: 分岐A(電撃スイッチバック)の実行。1エンゲージ1回のみ
                    #   (has_switchedラッチはLateralTTCMonitor内部で管理済み)。
                    if _lat_dec.side_override is not None:
                        # 2026-07-16修正(79節): curvature vetoはLateralTTCMonitor.update()
                        #   内部(new_side_blocked引数)へ統合済みのため、side_overrideが
                        #   返ってきた時点で既にcurvature-safeであることが保証される。
                        #   77節にあった実行時の後付けveto判定(has_switched/has_rescued
                        #   消費後に別途vetoしてトークンを浪費するバグの原因だった)は削除。
                        _locked = _lat_dec.side_override
                        self._ot_side_locked = _locked
                        self._ot_alpha = 0.0  # 既存H3ガード2と同じ再ランプ(急ハンドル防止)
                        # 2026-07-24追加(168節): 側反転につき、旧側のroom_exhausted計数・
                        #   凍結オフセットは新側と無関係なので持ち越さない。
                        self._ot_room_exhausted_count = 0
                        self._ot_last_valid_target_mag = None
                        # 2026-07-14追加: 側が入れ替わったので_ot_clearedもリセットする。
                        #   他の側変更点(STOPPING遷移・NORMAL復帰・infeasible-stop)は全て
                        #   _ot_cleared=Falseを伴っており、ここだけ漏れていた。本節で
                        #   _ot_clearedをオフセットランプ縮小(_a_target)にも使うようにした
                        #   ため、リセット漏れがあると「反転直後なのにcleared=True(旧側の
                        #   dlatを引き継ぐ)でalphaが0のまま=新側へ寄れない」という
                        #   switchbackの意図(側を切り替えて仕切り直す)を破壊するバグになる。
                        self._ot_cleared = False
                        self._ot_reacquire_count = 0  # 2026-07-14追加: デバウンスも側反転で仕切り直す
                        # 2026-07-12追加: 案②(ds縦マージンのハードロック)検討用の診断ロギング。
                        #   挙動には影響しない(ログ出力のみ)。fwd_dsはis_side_by_side判定が
                        #   Noneでも成立し得るため、発火時にNoneのままswitchbackする
                        #   ケースが実在するかも含めて確認できるようにする。
                        _fwd_ds_s = f"{_opp_sit.fwd_ds:.2f}" if _opp_sit.fwd_ds is not None else "None"
                        _fwd_dlat_s = (f"{_opp_sit.fwd_dlat:.2f}"
                                       if _opp_sit.fwd_dlat is not None else "None")
                        _vopp_s = f"{_opp_sit.fwd_vopp:.2f}" if _opp_sit.fwd_vopp is not None else "None"
                        _ds_ge_2_5 = (_opp_sit.fwd_ds is not None and _opp_sit.fwd_ds >= 2.5)
                        self.get_logger().warn(
                            f"[LAT-TTC-ACT] switchback branch={_lat_dec.branch} side={_locked} "
                            f"space={_lat_space:.2f} opp_space={_lat_opp_space:.2f} "
                            f"fwd_ds={_fwd_ds_s} fwd_dlat={_fwd_dlat_s} vopp={_vopp_s} "
                            f"v_ego={_v_odom:.2f} ds_ge_2.5={_ds_ge_2_5} "
                            f"lookahead={_lat_dec.lookahead_favor_switch} "
                            # 2026-07-18追加(102節続報): A_rescue_relaxedがどの閾値
                            #   (switchback_space_m厳格 or cleared_space_m緩和)で
                            #   発火したかを、既存のcritical_curvature_runの値から
                            #   次回ログで判別できるようにする(0=相手駆動で緩和閾値、
                            #   >0=カーブ駆動で厳格閾値が使われたことを示す)。
                            f"curvature_run={_lat_dec.critical_curvature_run} "
                            # 2026-07-22追加(157節): 静的曲率は懸念ありだったが実測空きで
                            #   上書きして反転できたケースを次回ログから直接追跡できるように
                            #   する(switchback_curvature_blocked自体は生の曲率判定のまま
                            #   無変更のため、この反転が本当にoverride起因だったかはこの
                            #   フィールドでのみ判別できる)。
                            f"curvature_overridden={_lat_dec.switchback_curvature_overridden} "
                            f"wp={self._mpc.model.wp_id}")
                        # 2026-07-18追加(103節Phase 0→107節案Cでveto判定へ統合):
                        #   room先読み(_room_min/_room_wp/_room_n)はupdate()呼び出し
                        #   前に既に計算済み(_locked==-self._ot_side と同一条件で
                        #   算出、上記new_side_room_blocked参照)のため、ここでは
                        #   再計算せずそのままログするだけ(呼び出し回数は従来通り
                        #   1回)。A_rescue_relaxed発火時のみ記録し、
                        #   new_side_room_blockedが実際に判定へ影響したかどうかを
                        #   room_blocked=で次回ログから確認できるようにする。
                        if _lat_dec.branch == "A_rescue_relaxed":
                            self.get_logger().warn(
                                f"[REVERSE-ROOM-CHECK] vid={_opp_sit.fwd_vid} "
                                f"opp_space={_lat_opp_space:.2f} room_ahead_min={_room_min} "
                                f"room_ahead_wp={_room_wp} n_sampled={_room_n} "
                                f"room_blocked={_new_side_room_blocked} "
                                f"wp={self._mpc.model.wp_id}")
                    elif _lat_dec.switchback_suppressed:
                        # 2026-07-14追加: margin不足(opp_space<space)で分岐Aへの反転を
                        #   抑制した瞬間を記録する。過去ログ検証(61件中21件が無駄な反転)
                        #   の効果を次回以降のログで定量確認するための診断ログ。
                        # 2026-07-16修正(79節): curvatureブロックによる抑制も同じログ行
                        #   のreasonで区別できるようにする(旧: 別タグswitchback_vetoed、
                        #   新: 同一タグでreason違いにより一本化、追跡しやすくする)。
                        # 2026-07-16追加(81節、82節でcleared中一律禁止を実装→83節でrevert):
                        # 2026-07-16追加(84節①、margin 0.5m案): cleared中にmarginが
                        #   既存のswitchback_space_m-giveup_space_m(0.5m)未満だったことのみを
                        #   理由に抑制された周期を、cleared_marginとして区別する。
                        # 2026-07-19追加(103/106/107節の非対称性解消、120節続報):
                        #   room(相手車位置認識ベースのveto)をreasonへ追加。
                        _reason = ("cleared_margin" if _lat_dec.switchback_cleared_margin_blocked
                                   else "k_corner" if _lat_dec.switchback_curvature_blocked
                                   else "wall" if _lat_dec.switchback_wall_blocked
                                   else "room" if _lat_dec.switchback_room_blocked
                                   # 2026-07-22追加(159節): 判定層と実行層の指標
                                   #   不一致対策(new_side_offset_blocked)による
                                   #   抑制を専用reasonで区別する。
                                   else "offset" if _lat_dec.switchback_offset_blocked
                                   # 2026-07-26追加(191節、AXIS03対処): 縦距離
                                   #   (fwd_ds)起因の抑制を専用reasonで区別する。
                                   else "ds" if _lat_dec.switchback_ds_blocked
                                   else "margin")
                        self.get_logger().info(
                            f"[LAT-TTC-ACT] switchback_suppressed reason={_reason} "
                            f"side={_locked} space={_lat_space:.2f} "
                            f"opp_space={_lat_opp_space:.2f} "
                            f"lookahead={_lat_dec.lookahead_favor_switch} "
                            f"wp={self._mpc.model.wp_id}")
                    # 案3(2026-07-12)→2026-07-17に旧EMA判定(_ot_side_block_ema)を削除し完全統一:
                    #   「空きが危険なほど狭まったか」の判断はLAT-TTCのC2
                    #   (TTC≤0.8秒、閾値到達を待たない先読み)のみで行う。
                    # 2026-07-24追加(168節、wp161スタック再発対策): 上記C2はTTC/closing/
                    #   footprint_riskのみを見ており、オフセット目標を実際に制約する
                    #   _corr_bound_ahead()の非正転落(先読み内に正の隙間が皆無)を一切
                    #   フィードバックしていなかった。委託側(_locked)の先読みroomが
                    #   _ot_giveup_cycles(既存の断念デバウンス、≈1s)連続で非正のまま
                    #   なら、C2と同じ_side_blocked合流点へ折り込む(新規の状態機械・
                    #   新規閾値は追加せず、既存のgiveup_count/giveup_cyclesの発想を
                    #   専用カウンタで踏襲するのみ)。
                    _room_ahead_locked = (
                        self._corr_bound_ahead(_locked) if _locked != 0 else float('inf'))
                    if _locked != self._ot_room_exhausted_prev_side:
                        self._ot_room_exhausted_count = 0
                    self._ot_room_exhausted_prev_side = _locked
                    if np.isfinite(_room_ahead_locked) and _room_ahead_locked <= 0.0:
                        self._ot_room_exhausted_count += 1
                    else:
                        self._ot_room_exhausted_count = 0
                    _room_exhausted = self._ot_room_exhausted_count >= self._ot_giveup_cycles
                    if _room_exhausted and self._ot_room_exhausted_count == self._ot_giveup_cycles:
                        self.get_logger().warn(
                            f"[OT-ROOM-EXHAUSTED] side={_locked} "
                            f"corr_bound={_room_ahead_locked:.3f} "
                            f"count={self._ot_room_exhausted_count} -> giveup合流 "
                            f"wp={self._mpc.model.wp_id}")
                    _side_blocked = _lat_dec.force_giveup or _room_exhausted
                    if (self._ot_giveup_count >= self._ot_giveup_cycles
                            or _locked == 0 or _side_blocked):
                        # 案B(2026-07-11): 「側消失」による離脱のみ、次回再エンゲージでの
                        #   反転抑制ヒステリシス用に側・対象車・時刻を記録する。giveup(相手が
                        #   速すぎる)は側と無関係の理由のため対象外(再選択して問題ない)。
                        if _side_blocked and _locked != 0:
                            self._ot_prev_side = _locked
                            self._ot_prev_side_vid = _opp_sit.fwd_vid
                            self._ot_prev_side_time = now
                        if _side_blocked:
                            # 2026-07-13追加: branch(C2/C2_cleared)をそのまま出し、cleared緩和後の
                            #   物理下限割れによるgiveupか、通常閾値によるgiveupかを区別できるようにする。
                            # 2026-07-16追加(79節): curvatureが救済反転をブロックした
                            #   結果の緊急giveupかどうかをログに残す(トークン浪費バグの
                            #   再発有無を今後のログで確認できるようにするための診断)。
                            # 2026-07-24追加(168節): force_giveupを伴わずroom_exhaustedのみで
                            #   合流した場合を区別できるようにtriggerラベルを分岐する。
                            _giveup_trigger = ("room_exhausted" if (_room_exhausted and not _lat_dec.force_giveup)
                                                else f"lat_ttc_{_lat_dec.branch}")
                            self.get_logger().warn(
                                f"[LAT-TTC-ACT] giveup trigger={_giveup_trigger} "
                                f"ttc={_lat_dec.ttc_lat} cleared={_lat_dec.cleared} "
                                f"curvature_blocked={_lat_dec.switchback_curvature_blocked} "
                                f"wall_blocked={_lat_dec.switchback_wall_blocked} "
                                f"footprint_risk={_lat_dec.footprint_risk_triggered} "
                                f"side={_locked} space={_lat_space:.2f} wp={self._mpc.model.wp_id} "
                                # 2026-07-21追加(149節続報、③診断): footprint_risk等の発火時、
                                #   自車〜相手の実測間隔(fwd_dlat)のトレンドが判定にどう見えて
                                #   いたかを記録する。dlat_trend_reset_reasonは"none"=正常に
                                #   トレンド計算済み、それ以外は_update_dlat_trend内のどの
                                #   リセット経路が発火したか(dlat_none/vid_changed/warmup)を示す
                                #   (これまでこのログ行が出していたv_ema/shrink_runはv_corridor_ema/
                                #   shrink_run=壁ベースの別トレンドで、footprint_risk自身が
                                #   発火時に明示的に0リセットする値だったため、実際のdlat
                                #   トレンドとは無関係だった)。
                                f"dlat_v_ema={_lat_dec.dlat_v_ema:.3f} "
                                f"dlat_shrink_run={_lat_dec.dlat_shrink_run} "
                                f"dlat_trend_reset_reason={_lat_dec.dlat_trend_reset_reason}")
                        # 断念(相手が速い) or 側消失(持続) → 追従へ。無効と判明した側は持ち越さず、
                        #   クールダウン中は再エンゲージしない(0.2秒debounceだけで反対側に飛ぶ
                        #   エピソード間スイングの抑制。仕切り直してICCで詰め直す)。
                        self._ot_state = "STOPPING"
                        self._ot_side = 0
                        self._ot_side_locked = 0
                        self._ot_giveup_count = 0
                        # 2026-07-20追加(138-5節②、停止車への繰り返しENGAGE失敗の是正):
                        #   実測(0720-04 wp240-243)で、完全停止した相手車の狭所にegoが
                        #   3回以上ENGAGEを試み、いずれもfootprint_riskで0.5〜1秒以内に
                        #   断念する往復を約9秒間繰り返していた。_plan_passの静的room
                        #   計算(デバウンス込み)は「わずかに間に合う」と判定するが、
                        #   footprint_risk(実測ベース)は毎回すぐに危険と判定しており、
                        #   両者の認識がズレたまま即座に再試行していたことが原因。
                        #   footprint_risk起因のgiveupの場合のみ、既存engage_cooldown_cycles
                        #   を2倍にする(92節①で確立済みの「min_trend_cycles*2」という
                        #   既存の倍化パターンを踏襲、新規パラメータ0個)。相手が速すぎる
                        #   等の他のgiveup理由は従来通りの長さのまま(再選択して問題ない
                        #   ケースまで不要に待たせない)。
                        self._ot_engage_cooldown = (
                            self._ot_engage_cooldown_cycles * 2
                            if _lat_dec.footprint_risk_triggered
                            else self._ot_engage_cooldown_cycles)
                        # 2026-07-21追加(148節②): footprint_risk起因の場合、以降の再エンゲージ
                        #   判定は上記の固定タイマーではなく、footprint_risk条件自体が実際に
                        #   解消したか(_ot_footprint_risk_clear_count)で決める。
                        self._ot_footprint_risk_gated = _lat_dec.footprint_risk_triggered
                        self._ot_footprint_risk_clear_count = 0
                        self._ot_fp_clear_logged = False
                        self._ot_cleared = False
                    else:
                        self._ot_side = _locked
                else:
                    # エンゲージ判定(2026-07-21、148節でヘルパー_evaluate_engage_readinessへ
                    #   抽出。cheap_ok9条件+_plan_pass+dlat_ttc_vetoの計算内容・呼び出し順序は
                    #   一切変えていない、純粋スリム化)。
                    _eval = self._evaluate_engage_readiness(
                        _scan, _pass_worth, _v_odom, _left_ok, _right_ok,
                        _being_overtaken, _lat_dec, _opp_sit, now, _footprint_risk)
                    _fwd_dbg["gate"] = _eval.gate
                    if _eval.can_engage:
                        # 2026-07-17追加(91節、③エンゲージ判定再設計のログ収集用): この分岐は
                        #   state!=OVERTAKINGの間のみ評価されるため、_can_engage=Trueの瞬間は
                        #   常に新規エンゲージそのもの(エッジ検知用のprevフラグ不要)。
                        #   engage_max_dist動的化(69/91節)の実走効果検証と、
                        #   engage_cooldown固定値の妥当性検討に使う実測値を記録する。
                        self.get_logger().info(
                            f"[ENGAGE] side={_eval.plan_side} fwd_ds={_fwd_ds} "
                            f"fwd_dlat={_scan.get('fwd_dlat')} vopp={_fwd_vopp} "
                            # 2026-07-20追加(132節、Gap①Phase0、診断専用): engage判定の
                            #   瞬間、直前の間合い推移(fwd_dlat縮小トレンド)が既に危険域
                            #   だったかを遡及検証できるよう記録する。ENGAGE可否には未反映。
                            f"dlat_v_ema={_lat_dec.dlat_v_ema:.3f} "
                            f"dlat_shrink_run={_lat_dec.dlat_shrink_run} "
                            f"closing_est={_eval.closing_est:.2f} "
                            f"engage_dist_dynamic={_eval.engage_dist_dynamic:.2f} "
                            f"t_reach_profile={_eval.t_reach_profile} "
                            f"path={'profile' if _eval.t_reach_profile is not None else 'dynamic'} "
                            f"wp={self._mpc.model.wp_id} "
                            # 診断用(2026-07-19、wp176-178ウェッジ再調査): 障害物分岐が
                            #   実際に選んだ側を決めるまでの窓内lf_i/rf_i推移(waypoint,seg,
                            #   kappa,ub,lb,lf,rf)。「走行中の相手」分岐ではtrace=[]のまま
                            #   (空リストは非該当を意味する、判定ロジック自体は無変更)。
                            f"trace={self._dbg_plan_trace}")
                        self._ot_state = "OVERTAKING"
                        self._ot_giveup_count = 0
                        self._ot_cleared = False
                        # 2026-07-24追加(168節): 新規エンゲージにつき、前回エピソード
                        #   (別側/別相手)のroom_exhausted計数・凍結オフセットは持ち越さない。
                        self._ot_room_exhausted_count = 0
                        self._ot_last_valid_target_mag = None
                        self._lat_ttc.reset_episode()  # シャドウ検証(2026-07-11): 新規エンゲージ毎にリセット
                        # 2026-07-17追加(97節): line_cap EMAも新規エンゲージ毎に仕切り直す
                        #   (前回のオーバーテイクの平滑化値を持ち越さない、既存原則の踏襲)。
                        self._line_cap_ema = None
                        # 2026-07-09修正(J): _plan_pass(J修正後)は can_engage=True のとき
                        #   常に非ゼロの側を返すため、「ロック側を左右比較なしで再利用する」
                        #   死に体フォールバック(_choose_overtake_side/_prev_locked再利用)を
                        #   削除し、常に_plan_passの計画(区間で空き続ける側)を採用する。
                        #   「今この瞬間広い側」は罠(内側=行き止まり)になり得るため使わない。
                        _prev_locked = self._ot_side_locked
                        self._ot_side = _eval.plan_side
                        self._ot_side_locked = _eval.plan_side
                        # 2026-07-20追加(131-6節④、対象車の一意性): _plan_passが実際に
                        #   計画対象とした相手車ID(scan["fwd_vid"]、_plan_pass冒頭2241行目
                        #   のvidと同一値)をエンゲージのたびに記録する。オフセット復帰判定
                        #   (下記_offset_return_ok)がこのIDと比較することで、「無関係な
                        #   別の車の存在」で復帰がブロックされるのを防ぐ。新規スキャン処理0個
                        #   (既存_scanをそのまま再利用)。
                        self._ot_target_vid = _scan.get("fwd_vid")
                        # H3(ガード2): 再エンゲージで側が変わった場合も α=0 から再ランプ
                        if _prev_locked != 0 and self._ot_side != _prev_locked:
                            self._ot_alpha = 0.0
                    else:
                        # 追従(ICC) / 両側塞がり / 被追い越し / ラッチ中
                        self._ot_state = "STOPPING"
                        self._ot_side = 0
            else:
                # 前方クリアが連続したら NORMAL 復帰（ハンチング防止）
                self._fwd_clear_count += 1
                if self._fwd_clear_count >= self._ot_exit_clear:
                    self._ot_state = "NORMAL"
                    self._ot_side = 0
                    self._ot_side_locked = 0   # A: 通過完了 → 側コミット解除
                    self._ot_worth_count = 0
                    self._ot_giveup_count = 0
                    self._ot_cleared = False

            # 一時的な infeasible では完全停止せず OVERTAKING を維持（後段のクリープで前進）。
            # infeasible_stop 回 連続で解けない＝実際に通れない時のみ安全STOPへ落とす（最後の保険）。
            # 2026-07-22修正(issue④①横展開、ユーザー指摘による一貫性検証): 従来は
            #   _ot_infeasible_latch(再エンゲージ禁止ラッチ)が「OVERTAKING起因のinfeasibility
            #   委譲」の時にしかセットされず、STOPPING中に発生したinfeasibility(下記v_safe
            #   テーパーが主に扱う状況)では根本の混雑が解消していなくても即再ENGAGEを
            #   許してしまう見落としがあった。counter==_ot_infeasible_stop(1周期ごとに+1
            #   されるため確実に一度だけ発火するエッジ検知)でラッチのセットを状態非依存にし、
            #   側の状態リセット(side資産の解除)だけをOVERTAKING起因の場合に限定して残す
            #   (OVERTAKINGケースの従来の発火タイミング・挙動は完全に同一のまま)。
            if self._mpc.infeasibility_counter == self._ot_infeasible_stop:
                self._ot_infeasible_latch = self._ot_infeasible_latch_cycles
                if self._ot_state == "OVERTAKING":
                    self._ot_state = "STOPPING"
                    self._ot_side = 0
                    self._ot_side_locked = 0   # A: 恒久失敗（実際に通れない）→ 側コミット解除して次で再選択
                    self._ot_giveup_count = 0
                    self._ot_cleared = False

            # 2026-07-22追加(160節続報、issue⑤①): 155節のRAMP-BYPASS判定・下記の新規
            #   STOPPING分岐の両方が参照するため、if/elifより前に1回だけ計算する
            #   (同一周期内で同じ値を共有し、160節の教訓=判定層と実行層の指標不一致を防ぐ)。
            _stopped_opp = (_opp_sit.fwd_vopp is not None
                             and _opp_sit.fwd_vopp < self._opp_obstacle_speed)
            # 方式B: 空き側へ e_y 目標をオフセット。OVERTAKING確定(=近接ゲートで真後ろに詰めて
            #   から)と同時にランプ開始。旧「前車から遠く早期に寄り始める」はコーナー跨ぎで壁に
            #   刺さったため撤廃し、「追いついてから寄る」へ変更(2026-07-04)。
            #   ランプで alpha を ramp_time かけて 0↔1 へ漸増漸減し横ジャークを防ぐ。
            if self._ot_state == "OVERTAKING" and self._ot_side != 0:
                # 2026-07-14修正: 真横到達済み(_ot_cleared)ならオフセットをレースライン側へ
                #   戻し始める。_ot_clearedは既にG-2/G-3(ICC速度解放)・LAT-TTC B_cleared
                #   (C1バイパス)の2箇所で「もう横方向の危険は去った」の判定に使っている
                #   既存ラッチであり、ここで3箇所目として再利用する(新規状態は増やさない)。
                #   従来はcleared後も_ot_state=="OVERTAKING"である限りalpha=1.0を維持し
                #   続けており、d_off(3.0m、実コリドーより大きい設定)がコリドー境界クランプ
                #   (2026-07-09追加)によって常に「壁そのもの」に張り付く一方、cleared後は
                #   LAT-TTCの速度キャップも外れる(45節)ため、「既に安全に離れているのに
                #   壁際を全開で通過する」という新規の危険な組み合わせを生んでいた
                #   (0713-06実測 wp136/wp243、いずれもcl=1で壁境界=オフセット目標=全開速度)。
                # 2026-07-15追加(71節、0715-02/0715-03実測で確認): _ot_clearedはfwd_dlat
                #   (横間隔)のみで判定しており、相手を縦に抜き終えたか(まだ真横/前方に
                #   いないか)を一切見ていなかった。真横到達(fwd_ds<=clear_ds_beside等)を
                #   満たした直後にオフセットが中央へ戻り始めると、それは自分から相手へ
                #   幅寄せする動きと同義になり、0715-02では実際の追突、0715-03では
                #   LAT-TTC C2_clearedの強制giveup(急ブレーキ)を招いていた。
                #   _scan_trafficは0<ds(自車より前方)の車のみをcarsに含める既存の仕様
                #   (1624行目)を再利用し、「追跡中の前方車がいない」ことを縦方向の完了
                #   確認の代理指標とする。新規状態変数は不要、G-2/G-3・LAT-TTC
                #   B_clearedバイパス用の_ot_cleared自体は無変更のまま(そちらは横間隔
                #   だけで解放してよい、既存の設計判断を維持)。
                # 2026-07-20修正(131-6節④、対象車の一意性、131-3節で発見): 上記の
                #   「0<dsの車のみ」という前提は、129節がfootprint_risk向けに後方許容窓を
                #   -along_min_lengthまで拡張したことで静かに崩れていた(_scan_traffic自体は
                #   無変更でも、cars/best選択の対象がds<0側にも広がった)。実測(0720-02
                #   wp338→339): 追い越し直後の相手がds=-1.99(既に自車の後方)まで下がった
                #   状態でも fwd_ds is not None が成立し、オフセット復帰開始0.83秒後に
                #   誤って再拡大していた。また元々「無関係な別の車」(3台以上のレースで
                #   別対象が窓内に入った場合)による阻害も未対処だった。fwd_ds>0(実際に
                #   前方にいる)かつfwd_vidが今回のエンゲージ対象(_ot_target_vid)と一致する
                #   場合のみ「まだクリアしていない」とみなす。対象車ID不明時(起動直後等)は
                #   従来通りfwd_ds is not Noneへフォールバックする(安全側、退行なし)。
                _fwd_ds_now = _scan.get("fwd_ds")
                _fwd_vid_now = _scan.get("fwd_vid")
                if self._ot_target_vid is not None:
                    _still_ahead = (_fwd_ds_now is not None and _fwd_ds_now > 0.0
                                     and _fwd_vid_now == self._ot_target_vid)
                else:
                    _still_ahead = _fwd_ds_now is not None
                _offset_return_ok = self._ot_cleared and not _still_ahead
                _a_target = 0.0 if _offset_return_ok else 1.0
                if _offset_return_ok != self._ot_offset_return_prev:
                    self.get_logger().info(
                        f"[OFFSET-RETURN] {'ON(alpha->0開始)' if _offset_return_ok else 'OFF(alpha->1再開)'} "
                        f"cleared={self._ot_cleared} fwd_ds={_fwd_ds_now} "
                        f"fwd_vid={_fwd_vid_now} target_vid={self._ot_target_vid} "
                        f"side={self._ot_side} alpha={self._ot_alpha:.2f} "
                        f"wp={self._mpc.model.wp_id}")
                    self._ot_offset_return_prev = _offset_return_ok
                # 2026-07-09修正: 固定d_off(3.0m)が実コリドー境界を超える目標を出し続けていた
                #   (実測2026-07-09予選: offset=3.0要求 vs 実コリドーub0=0.645mで2.36m超過、
                #   左壁衝突。右側でも同型を確認、左右対称の構造的バグ)。直近周期のコリドー境界
                #   (dbg_corr_ub0/lb0、占有格子ベースで4台走行でも全車の情報が既に反映済み)で
                #   クランプし、無理な目標を追わせない。ub0/lb0は既にsafety_margin_overtake
                #   (0.8m)込みの値のため、追加マージンは重ねない(二重マージン化を回避)。
                # 2026-07-21修正(147節、壁激突の深掘り対処): 「今この瞬間の1点」(dbg_corr_ub0/lb0)
                #   ではなく、_corr_bound_ahead()経由で動的コリドー配列全体(125節で公開済みの
                #   dbg_corr_ub_arr/lb_arr)の先読み最小値を使う。実測(0720-07 wp270→282、
                #   インサイドオーバーテイク中の壁激突)で、単一点クランプは壁側コリドーが
                #   実際に狭まった瞬間にしか反応できず、車両の横方向応答が追いつかないまま
                #   壁マージンがゼロまで悪化していた。新規パラメータ0個。
                # 2026-07-24追加(168節): 既にOVERTAKING継続中サブブランチ(giveup判定)で
                #   同一側のcorr_bound_ahead()を計算済みならそれを再利用し、二重計算を避ける
                #   (新規エンゲージ直後の1周期のみ_room_ahead_locked=Noneのため個別に計算)。
                _corr_bound = (_room_ahead_locked if _room_ahead_locked is not None
                               else self._corr_bound_ahead(self._ot_side))
                # 診断用(2026-07-22、153節): [OT]ログへ出力するため保持(_plan_pass由来の
                #   planLf/planRfとの乖離を、発生地点(何m先)込みで次回ログから直接判別する)。
                _fwd_dbg["corr_bound"] = round(_corr_bound, 3) if np.isfinite(_corr_bound) else _corr_bound
                _fwd_dbg["corr_bound_at"] = round(self._dbg_corr_bound_at_m, 2)
                _target_mag = self._ot_d_off
                if np.isfinite(_corr_bound):
                    if _corr_bound > 0.0:
                        _target_mag = min(_target_mag, _corr_bound)
                        self._ot_last_valid_target_mag = _target_mag
                    elif self._ot_last_valid_target_mag is not None:
                        # 2026-07-24追加(168節、wp161スタック再発対策): corr_boundが非正転落
                        #   (=先読み内に正の隙間が皆無、幾何学的に不可能)した瞬間、従来は
                        #   max(0.0, corr_bound)により目標を即座に0(直進)へ落としていた。
                        #   これは_ot_side/_ot_stateが「まだ継続中」を主張したまま実際の
                        #   指令だけが直進化する非矛盾性違反であり、しかも直進先は隙間皆無を
                        #   作っている相手車の方向そのもの(0724-01実測 wp160-163、offset
                        #   -1.196→-0.710→-0.242→-0.000の3周期で崩壊、直後にGHOST-BLOCK/
                        #   STUCK再発)。giveup合流(_ot_room_exhausted_count、上記)が実際に
                        #   OVERTAKINGを離脱させるまでの間は、直近の有効(正マージン)時の
                        #   目標量を凍結保持し、無理に直進化させない。
                        _target_mag = self._ot_last_valid_target_mag
                    else:
                        _target_mag = 0.0
                self._mpc.lateral_target = float(self._ot_side) * _target_mag
                _lat_active_side = self._ot_side
            # 2026-07-22追加(160節続報、issue⑤①: STOPPING中の能動的空き確保、issue⑤
            #   検討開始節参照): footprint_risk等の反応的検知を待たず、ENGAGE試行と同一の
            #   _plan_pass判定(_eval.plan_side/plan_ok、_evaluate_engage_readiness内で
            #   毎周期計算済み・新規判定式0個)が地形的に成立している間、停止/低速の相手
            #   (_stopped_opp、上でOVERTAKING分岐と共有計算済み)に対してのみ小さく
            #   先行して寄せる。側の値は本エンゲージ時にself._ot_sideへ採用される値と
            #   完全に同一のためswitchbackのような判定層/実行層の乖離(159節)は生じない。
            #   量は_ot_d_offではなく小さい_ot_proactive_bias_maxを上限とし、
            #   _corr_bound_ahead()で動的コリドークランプする点はOVERTAKING分岐と同一。
            #   cooldown中(_ot_engage_cooldown>0)は_eval.plan_okがcheap_ok経由でFalseに
            #   なるため自動的にバイアスも0になり、152節の適応的cooldown設計と矛盾しない。
            elif (self._ot_state == "STOPPING" and _eval is not None
                    and _eval.plan_ok and _eval.plan_side != 0 and _stopped_opp):
                _corr_bound = self._corr_bound_ahead(_eval.plan_side)
                _fwd_dbg["corr_bound"] = round(_corr_bound, 3) if np.isfinite(_corr_bound) else _corr_bound
                _fwd_dbg["corr_bound_at"] = round(self._dbg_corr_bound_at_m, 2)
                _target_mag = self._ot_proactive_bias_max
                if np.isfinite(_corr_bound):
                    _target_mag = min(_target_mag, max(0.0, _corr_bound))
                self._mpc.lateral_target = float(_eval.plan_side) * _target_mag
                _a_target = 1.0
                _lat_active_side = _eval.plan_side
            else:
                _a_target = 0.0
                _lat_active_side = 0
            # 検証ロギング(2026-07-22、160節続報、issue⑤①): [OT]ログのoffset=は
            #   本エンゲージ後の_ot_alpha*lateral_targetと同じ式のため既に非ゼロ値が
            #   出るが、それが本追い越しか今回追加した先行バイアスかを区別できるよう
            #   専用フィールドを追加する。次回ログでバイアスの発火頻度・収束地点
            #   (corr_bound_at)を直接確認できる。
            _fwd_dbg["proactive_bias_side"] = (
                _lat_active_side if self._ot_state == "STOPPING" else 0)
            # 2026-07-22追加(155節、なるべく減速せず完遂するための対処): 停止/低速の
            #   相手(vopp<opp_obstacle_speed、ENGAGE判定[_is_stopped_for_profile]と
            #   同一の既存閾値を再利用)に対し寄せる最中(_a_target>0)は、
            #   _ot_ramp_time(2.5秒、横ジャーク防止の目標漸増)を経由せず目標へ即座に
            #   到達させる。MPC自身のQP制約(max_steering_rate=既存のκレート上限、
            #   core/MPC.py _rate_bounds)が実際の操舵変化速度を既に制限しているため、
            #   目標側でさらにゆっくり出す必要は元々薄く、ゆっくり出すこと自体が
            #   実オフセットの成長(icc_stopのds_eff=ds+dlatが伸びるタイミング)を遅らせ、
            #   不要な減速を招いていた(154節の実座標検証で確認)。走行中の相手への
            #   通常の高速すれ違い、およびオフセット復帰(_a_target=0)側は従来通り
            #   ランプを維持し退行を避ける。2026-07-22修正(160節続報): 下記STOPPING分岐
            #   でも共有するため、if/elifより前で計算済みの_stopped_oppをそのまま使う。
            if _a_target > 0.0 and _stopped_opp:
                if self._ot_alpha < 1.0:
                    self.get_logger().info(
                        f"[RAMP-BYPASS] 停止/低速の相手(vopp={_opp_sit.fwd_vopp})に対し"
                        f"ランプ省略、alpha即時1.0 side={_lat_active_side} "
                        f"state={self._ot_state} "
                        f"wp={self._mpc.model.wp_id}")
                self._ot_alpha = 1.0
            else:
                _ramp_step = dt / max(self._ot_ramp_time, 1e-3)
                self._ot_alpha += float(np.clip(_a_target - self._ot_alpha, -_ramp_step, _ramp_step))
                self._ot_alpha = float(np.clip(self._ot_alpha, 0.0, 1.0))
            self._mpc.lateral_blend = self._ot_alpha
            _cur_off = self._ot_alpha * self._mpc.lateral_target  # 現在の実効オフセット(e_y, 右=負)

            # 2026-07-14追加: safety_margin_overrideもオフセットと同じ_ot_ramp_timeで滑らかに
            #   遷移させる(0714-01 事象A対策、787行目コメント参照)。OVERTAKING中は縮小側
            #   (_ot_safety_margin)、それ以外(STOPPING/NORMAL)は通常側(_ot_margin_full)を
            #   目標にし、目標が変わった瞬間に飛び付かせず_ramp_step(オフセットランプと同一の
            #   時定数)で追従させる。速度[m/s]は2値の差分/ramp_timeから導出するのみで新規の
            #   チューニング値は増やさない。
            _margin_target = (self._ot_safety_margin if self._ot_state == "OVERTAKING"
                               else self._ot_margin_full)
            _margin_rate = abs(self._ot_margin_full - self._ot_safety_margin) / max(
                self._ot_ramp_time, 1e-3)
            _margin_step = _margin_rate * dt
            self._ot_margin_cur += float(np.clip(
                _margin_target - self._ot_margin_cur, -_margin_step, _margin_step))
            _is_ramping = abs(_margin_target - self._ot_margin_cur) >= 1e-3
            if _is_ramping != self._ot_margin_ramping_prev:
                self.get_logger().info(
                    f"[MARGIN-RAMP] {'START' if _is_ramping else 'DONE'} "
                    f"target={_margin_target:.3f} cur={self._ot_margin_cur:.3f} "
                    f"state={self._ot_state} wp={self._mpc.model.wp_id}")
                self._ot_margin_ramping_prev = _is_ramping

            # B-lite核心: ヘディング参照(e_psi目標)を開き側へ傾け、支配項Q[e_psi]に右へ操舵させる。
            #   バイアス = asin(残り横ギャップ / 先読み距離)。目標に寄るほど0へ収束＝過操舵抑制。
            if self._ot_state == "OVERTAKING" and self._ot_side != 0:
                # 自然ライン化(2026-07-05): 目標は「最終±3m」でなく「ランプ済みオフセット
                #   (_cur_off)」。旧実装はエンゲージ瞬間に (±3−0)/Leff→±20°の急バイアスで
                #   急ハンドル+失速していた。ランプ済み目標なら参照と車体が一緒に動き、
                #   バイアスは自然に小さく立ち上がる(*alpha の二重減衰は廃止)。
                _Leff = max((_fwd_ds if _fwd_ds is not None else 8.0) - 2.0, 3.0)
                _pb = float(np.arcsin(np.clip((_cur_off - _cur_ey) / _Leff, -0.99, 0.99)))
                _pb = float(np.clip(_pb, -self._ot_psi_max, self._ot_psi_max))
                self._mpc.lateral_psi_bias = _pb
            else:
                self._mpc.lateral_psi_bias = 0.0

            # 横方向TTC監視ログ(2026-07-12): 実際のupdate()呼び出しは本メソッド冒頭
            #   (_lat_dec算出箇所)へ移設済み。ここでは間引きログのみを行う
            #   (enabled=Falseでも判定内容は引き続きログし、次回以降の比較検証に使えるようにする)。
            #   2026-07-11修正: 単純な1-in-5間引きだと、分岐A(スイッチバック)のような
            #   1エピソードにつき1回しか発火しない単発イベントが、たまたま非サンプル
            #   周期に起きた場合ログに一切残らないという欠陥があった。branch遷移
            #   (前周期と異なる値になった瞬間)は必ずログし、同じ危険branchが継続する
            #   場合のみ1-in-5で間引く。
            # 2026-07-14追加: v_inst物理妥当性クランプが実際に発動した周期を無間引きで
            #   記録する(0713-05 wp16-21・0713-06 wp243-246のような外れ値混入を次回
            #   ログで直接検証できるようにするため)。稀な事象である想定のため間引かない。
            if _lat_dec.v_inst_clamped:
                self.get_logger().warn(
                    f"[LAT-TTC-CLAMP] v_inst clamped side={self._ot_side} "
                    f"space={_lat_space} fwd_dlat={_scan['fwd_dlat']} "
                    f"fwd_vid={_scan['fwd_vid']} v_ema_after={_lat_dec.v_corridor_ema} "
                    f"wp={self._mpc.model.wp_id}")
            _lat_changed = (_lat_dec.branch != self._lat_ttc_prev_branch)
            if _lat_dec.branch not in ("none", "warmup", "stable"):
                if _lat_changed or self._lat_ttc_log_count % 5 == 0:
                    self.get_logger().info(
                        f"[LAT-TTC] branch={_lat_dec.branch} ttc={_lat_dec.ttc_lat} "
                        f"side={self._ot_side} space={_lat_space} opp_space={_lat_opp_space} "
                        f"would_switch={_lat_dec.side_override} would_vcap={_lat_dec.v_safe_cap} "
                        f"would_giveup={_lat_dec.force_giveup} "
                        f"is_sbs={_lat_dec.is_side_by_side} has_switched={_lat_dec.has_switched} "
                        f"fwd_dlat={_scan['fwd_dlat']} fwd_ds={_scan['fwd_ds']} "
                        f"fwd_vid={_scan['fwd_vid']} v_ema={_lat_dec.v_corridor_ema} "
                        f"shrink_run={_lat_dec.shrink_run} "
                        # 2026-07-13追加: cleared緩和(B_cleared/C2_cleared)がなぜ選ばれたかを
                        #   このログ単体で追えるようにする(cleared済みかthr値の裏付け)。
                        f"cleared={_lat_dec.cleared} "
                        f"thr={(self._along_min_width if _lat_dec.cleared else self._along_lane_need)} "
                        # 2026-07-17追加(92節①): branch=C1_deferredが何周期継続中かを
                        #   このログ単体で追えるようにする(猶予の消費具合を次回ログで検証するため)。
                        f"curvature_run={_lat_dec.critical_curvature_run}")
                self._lat_ttc_log_count += 1
            else:
                self._lat_ttc_log_count = 0
            self._lat_ttc_prev_branch = _lat_dec.branch
            _fwd_dbg["psi_bias"] = float(np.degrees(self._mpc.lateral_psi_bias))

            # Fix-2: パス対象のクリア判定(ヒステリシス)。scanの最近傍前方車の実横間隔で更新。
            #   解放: dlat≥2.1 or (dlat≥1.8 かつ ds≤1.0=真横到達) / 再取得: dlat<1.6 に再接近。
            #   旧「1.8で即・全開」はコーナーのライン収束で再接近し接触(0703_02で3件)、かつ
            #   1.8境界のトグルで v_safe がサージ(28%)していた。
            if self._ot_state == "OVERTAKING":
                _fd = _scan["fwd_dlat"]; _fs = _scan["fwd_ds"]
                if _fd is not None:
                    if _fd < self._clear_lat_reacquire:
                        # 2026-07-14追加(事象③追補): 単発のdlat落ち込み(コーナー形状由来の
                        #   一時的なもの)で即座にclearedを解除せず、engage_debounce周期連続
                        #   (既存パターン、_ot_worth_count/_room_debounce_okと同一思想)で
                        #   初めて再取得を確定する。真に相手が再接近する場合は継続的にdlatが
                        #   小さいままのためengage_debounce周期以内に確実に検知でき、安全性は
                        #   犠牲にしない(最大でも約0.2秒の再取得遅延のみ)。
                        self._ot_reacquire_count += 1
                        if self._ot_reacquire_count >= self._ot_engage_debounce:
                            self._ot_cleared = False
                    else:
                        self._ot_reacquire_count = 0  # 再取得ゾーンを抜けたのでデバウンスをリセット
                        if (_fd >= self._clear_lat_release
                                or (_fd >= self._fwd_min_lat_sep
                                    and _fs is not None and _fs <= self._clear_ds_beside)):
                            # 2026-07-14追加(0714-02実測、事象③): fwd_dlatは「対象車との絶対横間隔」
                            #   であり、自車がどちら側にいるかを区別しない。switchback(側反転)直後は
                            #   alphaが0へリセットされ実際にはまだ新側へ寄っていないにも関わらず、
                            #   fwd_dlatがたまたま閾値を満たしていると即座にcleared=Trueへ戻ってしまい
                            #   (実測: wp175でswitchback側=-1発生の直後、alpha=0.08の時点で早くも
                            #   [OFFSET-RETURN] ONが発火し、新側へのランプ立ち上げをその場で潰していた)、
                            #   側反転そのものが機能しなくなっていた。実際に新側への横移動が完了
                            #   (alpha≈1、オフセットランプ完了)して初めて「真横到達」を認めるようにし、
                            #   側反転直後の数百ミリ秒だけclearedへの昇格を遅らせる(新規パラメータなし、
                            #   既存の_ot_alphaを再利用)。
                            if self._ot_alpha >= 1.0 - 1e-3:
                                self._ot_cleared = True
            else:
                self._ot_cleared = False
                self._ot_reacquire_count = 0

            # === 統一ICC: 唯一の速度制限(1回だけ計算し、状態で味付け) ===
            #   対象選び: 近距離=実横間隔<near_sep / 遠方=進路帯(OVERTAKING中はオフセット分ずらす)
            #   near_sep はクリア済なら1.6(再接近のみ捕捉)、未クリアなら1.8。
            #   式(G2): √(max(0, v_fwd²+2a(ds−margin))) — margin内では v_fwd より遅く=車間を開け直す
            _near_sep = (self._clear_lat_reacquire
                         if (self._ot_state == "OVERTAKING" and self._ot_cleared)
                         else self._fwd_min_lat_sep)
            # 2026-07-15追加: switchback直後(オフセットがまだ新側へ移動し切っていない=
            #   alpha未到達)は、旧側にいた頃の実測dlatをそのまま使ったnear_sep除外判定で
            #   ICCが対象車を誤って見失う(実測t=434.35で確認)。この間は現在追跡中の対象車
            #   (fwd_vid)だけを除外対象から免除する。
            # 2026-07-20追加(143節続報、フェーズ2、P0①本体対処): near_sepの静的ゲート
            #   (現在の横距離のみ)が、LAT-TTCが横方向closingトレンドとして継続追跡中の
            #   対象車(_opp_sit.is_closing_trend、既存のPhase1判定を再利用)を除外し、
            #   eff_v_cap(前車なし=無制限速度)へ抜けてしまう問題(0720-05実測wp139、
            #   単一サイクルでv=4.18→2.02m/sの実質衝突)に対処する。上記switchback-alpha
            #   救済と同じ仕組み(force_include_vid)・同じ対象車(_scan["fwd_vid"]==
            #   _opp_sit.fwd_vid、143節続報のスリム化点検で同一window内であることを
            #   確認済み)をOR条件で拡張する。新規パラメータ0個。
            # 2026-07-20追加(145節続報、フェーズ3①、STOPPING側の同型盲点対処): 144節の
            #   スリム化点検で、STOPPING側のicc_stop(_vlim直接使用)も全く同じnear_sep
            #   静的ゲートを共有しているにもかかわらず、is_closing_trend救済は
            #   OVERTAKING状態限定のままだったことを発見した(icc_stop_fallback/
            #   STOPPING-NO-VSAFEブリッジという段階的な保険はあるが、それぞれ
            #   engage_lat_max(2.0m)/along_min_length(2.0m)というnear_sep(1.8m)より
            #   さらに狭い窓でしか対象車を捕捉できず、trend自体は見ていない)。
            #   switchback-alpha救済(_ot_side/_ot_alpha)はOVERTAKING固有の概念のため
            #   スコープを変えず、is_closing_trend救済のみSTOPPINGへ拡張する
            #   (_vlimはOVERTAKING/STOPPING共通の「統一ICC」であるため、同じ救済を
            #   同じ理由で適用するのが一貫している)。新規パラメータ0個。
            _force_include_via_trend = (
                self._ot_state in ("OVERTAKING", "STOPPING") and _opp_sit.is_closing_trend)
            _force_include_vid = (
                _scan.get("fwd_vid")
                if ((self._ot_state == "OVERTAKING"
                     and self._ot_side != 0 and self._ot_alpha < 1.0 - 1e-3)
                    or _force_include_via_trend)
                else None)
            # 190-5節(2026-07-26追加、診断専用): force_include_vidがis_closing_trend
            #   起因(alpha救済ではなく)で発火した瞬間のみを記録する。従来この消費先には
            #   ログが皆無で、ENGAGEゲート/G2-RELEASEと違い実際に発火しているか確認
            #   できなかった。
            if _force_include_via_trend and not self._force_include_vid_trend_active:
                self.get_logger().info(
                    f"[FORCE-INCLUDE-VID-TREND] fwd_vid={_scan.get('fwd_vid')} "
                    f"ot_state={self._ot_state} wp={self._mpc.model.wp_id}")
            self._force_include_vid_trend_active = _force_include_via_trend
            _vlim, _vtgt = self._follow_speed_limit(
                _scan, path_offset=(_cur_off if self._ot_state == "OVERTAKING" else 0.0),
                near_sep=_near_sep, force_include_vid=_force_include_vid)
            self._pf_mark('icc')
            if self._ot_state == "OVERTAKING":
                self._mpc.use_obstacle_avoidance = True   # コリドーが空き側へ自動誘導
                self._mpc.lateral_funnel_steps = self._ot_funnel_steps  # 「徐々に右」
                self._mpc.safety_margin_override = self._ot_margin_cur  # 実寸まで滑らかに縮小して通す(2026-07-14: 即時値→ランプ値)
                self._ot_returning = True  # 追い越し後にライン復帰が必要、とアーム
                # v_cap を v_max に連動させる(2026-07-08修正): 固定6.0m/s(21.6km/h)のままだと、
                #   将来 v_max を18→24km/hへ引き上げた際、追い越し中だけ通常巡航より遅くなる
                #   逆転が起きる(v_maxに対する安全上のブレーキという本来の役割を超えて主速度の
                #   足枷になる)。設定値と現在のv_maxの大きい方を採用し、上限としての意味だけ残す。
                # OVERTAKING中のv_safe候補は3択排他(いずれか1つ、優先順位固定):
                #   ①前車なし→全開 ②側方確保済み→解放(G/G-2/G-3、143節続報で
                #   _g2_release_readyへ抽出) ③前車追従+クリープ床(F3、143節続報で
                #   _f3_taper_speedへ抽出)。各判定の詳細な実測事故の経緯は
                #   design_docs/stage15_perf_20260707.html参照(2026-07-08〜18節)。
                _eff_v_cap = max(self._ot_v_cap, float(self._mpc.input_constraints["umax"][0]))
                if _vlim is None:
                    _v_safe_pre = _eff_v_cap
                    _v_safe_cand.append(("eff_v_cap(前車なし)", _eff_v_cap))
                elif self._g2_release_ready(_scan, _fwd_vopp, _vtgt, _left_free,
                                             _right_free, _v_safe_pre,
                                             is_closing_trend=_opp_sit.is_closing_trend):
                    _v_safe_pre = _eff_v_cap
                    _v_safe_cand.append(("eff_v_cap(G-2側確保解放)", _eff_v_cap))
                else:
                    _v_safe_pre = self._f3_taper_speed(_vtgt, _eff_v_cap, _vlim)
                    _v_safe_cand.append(("icc_f3(前車追従+クリープ床)", _v_safe_pre))
                # B: 内側ライン減速(2026-07-05): オフセット走行中はレースライン基準の
                #   エンベロープでは速すぎる(内側=R縮小)。実効曲率ベースの上限をminで重畳。
                # 2026-07-19追加(112節、79節の未解決課題への回答): 引数を_cur_ey(現在の
                #   実位置、まだ目標へ到達していない可能性がある)から_cur_off(既に確定
                #   済みのオフセット目標、self._ot_alpha*lateral_target)へ変更。0719-01
                #   実測(wp202-204、1周目・2周目とも同一地点でCOLLISION-SUSPECTED)で、
                #   offset=-3.0へ収束中の約9秒間、line_capが実質無効(v_safe=v_max)の
                #   まま推移し、壁際減速(wall_slow、現在wp1点のみの反応式)が衝突の
                #   約1秒後にしか検知できていなかったことを確認した。79節がwall_slow
                #   自身の先読み追加をrevertした理由(「自車の現在のeyを固定して先の
                #   waypointと比較するのは誤り、能動的に経路追従するため実態と乖離する」)
                #   とは異なり、_cur_offは車両の物理追従を待たず時間(ramp_time)だけで
                #   確定する制御目標であり、「今どこにいるか」ではなく「どこへ向かえと
                #   命令されているか」を使うため、79節の教訓とは矛盾しない。新規パラメータ
                #   0個(既存の_cur_offを再利用)。
                _line_cap = self._offset_line_speed_cap(_cur_off)
                if _line_cap is not None:
                    _v_safe_pre = _line_cap if _v_safe_pre is None \
                        else min(_v_safe_pre, _line_cap)
                    _v_safe_cand.append(("line_cap(内側ライン曲率減速)", _line_cap))
                # 分岐C1(2026-07-12実挙動統合): 既存のv_safe合成パターンにそのまま乗せる。
                #   分岐B(並走中)はv_safe_cap=None(意図的に速度介入なし)なのでここでは
                #   何もしない。新規ログは追加しない(v_safe_srcに自動的に載るため)。
                # 2026-07-18追加(100節、Tier1裁定の外出し): 旧C1_obstacle_yield分岐
                #   (92節続報、lateral_ttc_monitor.py内)をここへ移設した。障害物クラス
                #   の間はC1のv_cap自体は計算されて返るが、F3-TAPER(icc_f3)へ委譲する
                #   ため候補には含めない(挙動は92節続報と完全に同一、判定の置き場所の
                #   みを変更)。
                _lat_c1_yielded = (_lat_dec.branch == "C1" and _fwd_is_obstacle_class
                                    and _lat_dec.v_safe_cap is not None)
                if _lat_c1_yielded != self._lat_ttc_c1_yield_prev:
                    self.get_logger().info(
                        f"[TIER1-C1-YIELD] {self._lat_ttc_c1_yield_prev} -> {_lat_c1_yielded} "
                        f"vopp={_fwd_vopp} v_cap_would_be={_lat_dec.v_safe_cap} "
                        f"wp={self._mpc.model.wp_id}")
                self._lat_ttc_c1_yield_prev = _lat_c1_yielded
                if _lat_dec.v_safe_cap is not None and not _lat_c1_yielded:
                    _v_safe_pre = _lat_dec.v_safe_cap if _v_safe_pre is None \
                        else min(_v_safe_pre, _lat_dec.v_safe_cap)
                    _v_safe_cand.append((_lat_dec.v_safe_cap_label, _lat_dec.v_safe_cap))
            elif self._ot_state == "STOPPING":
                # 追従(ICC)。被追い越し中、または上記の能動的空き確保バイアスが今回作動中
                #   (_lat_active_side!=0、160節続報issue⑤①)は回避ON=相手を障害物として
                #   避けつつ自ライン維持(押し込まない)。
                # 2026-07-22追加(160節続報): バイアス作動中にuse_obstacle_avoidance=Falseの
                #   ままだと、_corr_bound_ahead()が読むdbg_corr_ub_arr/lb_arrは静的テーブル
                #   (相手車の存在を一切反映しない)のままになり、クランプが実質機能しない
                #   まま「安全にクランプ済み」と誤認する上流-下流不一致が生じるため、
                #   バイアス発動条件(_lat_active_side!=0)と同一周期で必ずTrueにする。
                self._mpc.use_obstacle_avoidance = bool(_being_overtaken) or _lat_active_side != 0
                # 2026-07-22追加(160節続報): バイアス作動中はOVERTAKING開始時と同じ理由
                #   (静的→動的コリドーへの急変によるinfeasible化防止、147節近傍参照)で
                #   既存のfunnel_steps(新規パラメータ0個)を流用する。
                self._mpc.lateral_funnel_steps = (
                    self._ot_funnel_steps if _lat_active_side != 0 else 0)
                self._mpc.safety_margin_override = self._ot_margin_cur  # 2026-07-14: None即時復帰→ランプ値
                _v_safe_pre = _vlim
                if _vlim is not None:
                    _v_safe_cand.append(("icc_stop(追従)", _vlim))
                # ICC見失いフォールバック(2026-07-12): _follow_speed_limitはdlat<near_sep(1.8m)
                #   でしか対象車を捕捉しない。直前のswitchback/giveup試行の残存オフセットで
                #   fwd_dlatが一時的にnear_sepを超えていると、目の前(fwd_ds僅か数m)の相手を
                #   完全に見失い、STOPPING中にもかかわらず無制限速度のまま追突する事例を実測
                #   (0712-03、C2発火直後の約2秒間v_safe_src=Noneでu0=最高速のまま接近)。
                #   OVERTAKING分岐の「_vlim=None→全開」(2684-2686行目)とは対照的にSTOPPING中の
                #   _vlim=Noneは「自分のオフセット経路が空いている」ではなく「dlatゲートで
                #   見失っただけで実際は目の前にいる」ことを意味しうるため、_scan_traffic側の
                #   より緩い最近傍車判定(fwd_ds、dlat条件なし)を保険として使う。近距離
                #   (fwd_near_range=6.0m)限定、既存のG2式(_g2_speed)をそのまま再利用。
                # 2026-07-14修正(0714-04実測、H2型デッドロックの再発対策): 上記フォールバックは
                #   dlat条件を一切持たず、明確に進路外(H2/2026-07-09修正でengageの基準とした
                #   engage_lat_max=2.0m超)の相手にも縦方向のみでv=0まで強制減速していた。
                #   H2修正は「ICCとengageは同じ相手を同じ基準(dlat)で評価する」という不変条件を
                #   確立したが、その3日後に追加された本フォールバックはdlatを見ないため、
                #   このフォールバック経路からH2と同型のデッドロック(ICCはv=0を強制し続ける
                #   のにengageも起きない永久停止)が再発していた(実測: fwd_dlat=2.15〜2.45m、
                #   Rfree=5.44mと十分な空きがあるにも関わらずd_min=0.9994mに固着し15秒以上
                #   完全停止、STUCK-BACKUP復帰でのみ解消)。既存engage_lat_max(H2が既に
                #   確立した基準そのもの)を再利用し、フォールバックが介入してよい範囲を
                #   「engageが正当に評価対象とする相手」に限定する(新規パラメータ0個)。
                _fwd_dlat_val = _scan["fwd_dlat"]
                _icc_fallback_candidate = (_vlim is None and _fwd_ds is not None
                                            and _fwd_vopp is not None
                                            and _fwd_ds <= self._fwd_near_range)
                _icc_fallback_on_path = (_fwd_dlat_val is None
                                         or _fwd_dlat_val <= self._ot_engage_lat_max)
                _icc_fallback_on = _icc_fallback_candidate and _icc_fallback_on_path
                if _icc_fallback_on:
                    _v_fallback = self._g2_speed(_fwd_vopp, _fwd_ds)
                    _v_safe_pre = _v_fallback
                    _v_safe_cand.append(("icc_stop_fallback(near_sep見失い保険)", _v_fallback))
                if _icc_fallback_on != self._icc_fallback_prev:
                    if _icc_fallback_on:
                        self.get_logger().info(
                            f"[ICC-FALLBACK] ON fwd_ds={_fwd_ds:.2f} "
                            f"fwd_dlat={_scan['fwd_dlat']} vopp={_fwd_vopp:.2f} "
                            f"v_fallback={_v_fallback:.2f} wp={self._mpc.model.wp_id}")
                    else:
                        self.get_logger().info(
                            f"[ICC-FALLBACK] OFF wp={self._mpc.model.wp_id}")
                    self._icc_fallback_prev = _icc_fallback_on
                # [ICC-FALLBACK-SKIP]診断ログ(2026-07-14追加): 縦距離だけ見ればフォールバックの
                #   対象になり得た(ds<=fwd_near_range)が、dlatがengage_lat_maxを超えるため
                #   介入を見送った瞬間を記録する。事後に「なぜこの相手では減速しなかったか」
                #   (=明確に進路外と判定されたため)を検証できるようにする。
                _icc_fallback_skip = _icc_fallback_candidate and not _icc_fallback_on_path
                if _icc_fallback_skip != self._icc_fallback_skip_prev:
                    if _icc_fallback_skip:
                        self.get_logger().info(
                            f"[ICC-FALLBACK-SKIP] fwd_ds={_fwd_ds:.2f} "
                            f"fwd_dlat={_fwd_dlat_val:.2f} "
                            f"engage_lat_max={self._ot_engage_lat_max:.2f} "
                            f"vopp={_fwd_vopp:.2f} wp={self._mpc.model.wp_id}")
                    else:
                        self.get_logger().info(
                            f"[ICC-FALLBACK-SKIP] cleared wp={self._mpc.model.wp_id}")
                    self._icc_fallback_skip_prev = _icc_fallback_skip
                # 2026-07-18追加(109節続報、診断専用・挙動へ影響なし): STOPPING中に
                #   icc_stop(_vlim)・icc_stop_fallbackのいずれも成立せずv_safe_pre=None
                #   のまま全開(u0=4.17)になる瞬間を記録する。ローカル3台走行実測
                #   (output/20260718-172517、t≈426.35秒、wp272→277)で、fwd_dlat=2.76〜
                #   3.47m(near_sep=1.8・engage_lat_max=2.0のいずれも超過、既存H2/0714-04
                #   設計によりicc_stop/fallback双方が正しく対象外とした結果)の相手を
                #   「明確に進路外」として無視した直後、Rfree≈0(実壁基準の左空き幅が
                #   ほぼゼロ)・[COLLISION-SUSPECTED](v drop 3.99→3.08)を確認した。
                #   _scan["cars"]は「best(=_vlim/fallbackが検討する唯一の最近傍車)」
                #   より広い全前方車リストを既に保持しているため(1624行目)、新規スキャン
                #   処理無しでこの瞬間の全候補(ds/dlat/vid)をログし、除外されたfwd_vid以外
                #   に、より近い・より危険な相手が存在していたかを次回ログで確認する
                #   (推測せず計装で実測、Stage1.5方針)。
                if _v_safe_pre is None:
                    if not self._stopping_no_vsafe_prev:
                        _cars_s = ", ".join(
                            f"(vid={c[4]} ds={c[0]:.2f} dlat={c[3]:.2f})"
                            for c in _scan["cars"])
                        self.get_logger().warn(
                            f"[STOPPING-NO-VSAFE] ON fwd_vid={_scan.get('fwd_vid')} "
                            f"fwd_ds={_fwd_ds} fwd_dlat={_fwd_dlat_val} "
                            f"near_sep={self._fwd_min_lat_sep} "
                            f"engage_lat_max={self._ot_engage_lat_max} "
                            f"n_cars={len(_scan['cars'])} cars=[{_cars_s}] "
                            f"wp={self._mpc.model.wp_id}")
                    self._stopping_no_vsafe_prev = True
                    # 2026-07-20追加(131-6節⑤、なめらかな断念): 実測(0720-02 wp13、
                    #   t=609.34)で、OVERTAKING中の速度モデル(eff_v_cap)からSTOPPINGの
                    #   速度モデル(icc_stop/fallback)への切替の瞬間、両方とも不成立になると
                    #   v_safe_pre=Noneのまま、MPC自身の最適化が無制限速度(u0=v_max)を
                    #   出力することを直接確認した(OTログ: v_safe_src=None なのに
                    #   u0=4.1667)。0.6秒後にwall_slowが追いついた時点で壁マージンは
                    #   既にマイナス(wall=-0.42)まで悪化していた。footprint_risk・
                    #   wall_slowの完全キャップが既に再利用している既存定数
                    #   wall_slow_speedを、状態遷移の隙間を埋める保守速度としてここでも
                    #   再利用する(新規パラメータ0個)。これにより「誰も速度を能動的に
                    #   決めない空白」を解消し、後続の各候補(wall_slow等)が追いつくまでの
                    #   ブリッジとして働く。
                    # 2026-07-20再修正(138-5節①、過剰発火の是正): 0720-04実測(wp47-52、
                    #   ds=5〜7m・dlat=2.8〜3.1m)で、除外車が明確に遠い場合にも一律に
                    #   ブリッジが発動し、可視的な減速チャタリングを起こすことが判明した。
                    #   正当化事例(wp13、ds=1.0m)との違いはfwd_ds(縦方向の近さ)のみ
                    #   だったため、footprint_riskが既に使う縦方向の物理下限
                    #   along_min_length(既存、新規パラメータ0個)より近い場合のみ
                    #   ブリッジを適用する。遠い/対象車無しの場合は従来のMPC最適化に
                    #   委ねる(この場合はそもそも「近くに見落としている危険な相手がいる」
                    #   という前提が成立しないため、u0=v_maxのままでも実害の証拠がない)。
                    if _fwd_ds is not None and abs(_fwd_ds) < self._along_min_length:
                        _v_safe_pre = self._wall_slow_speed
                        _v_safe_cand.append(("stopping_no_vsafe(状態遷移ブリッジ)", self._wall_slow_speed))
                elif self._stopping_no_vsafe_prev:
                    self.get_logger().warn(
                        f"[STOPPING-NO-VSAFE] OFF v_safe={_v_safe_pre:.2f} "
                        f"wp={self._mpc.model.wp_id}")
                    self._stopping_no_vsafe_prev = False
            else:  # NORMAL（前方車なし）→ 通常のレースライン走行 / 追い越し後の復帰
                self._mpc.use_obstacle_avoidance = True
                self._mpc.safety_margin_override = self._ot_margin_cur  # 2026-07-14: None即時復帰→ランプ値
                _v_safe_pre = None
                # 戻りfunnel: 追い越し後にライン外(|e_y|大)なら、コリドーを現在位置から
                #   ラインへ徐々に寄せて feasible に復帰させる（外側でinfeasible→内壁へ流れるのを防ぐ）。
                if self._ot_returning and abs(_cur_ey) < self._ot_return_done:
                    self._ot_returning = False  # ライン復帰完了
                self._mpc.lateral_funnel_steps = self._ot_funnel_steps if self._ot_returning else 0

            # H4-lite(2026-07-04): infeasible自己ロック解除(前進のみ・後退なし=AWSIM制約)。
            #   壁際でMPCが解けず u0=0 のまま20秒級で停止する事象(実測2回で計45秒損失)への
            #   最小リカバリ: マージン最小化+強funnelで「今いる場所から」解ける問題に作り直し
            #   這い出す。通常走行では inf がここまで積み上がらないため誤発動しない。
            if (self._mpc.infeasibility_counter >= self._unlock_after
                    and _v_odom < 0.3):
                if self._unlock_left == 0:
                    self.get_logger().warn(
                        f"[UNLOCK] infeasible-lock recovery (inf="
                        f"{self._mpc.infeasibility_counter})")
                self._unlock_left = self._unlock_hold
            if self._unlock_left > 0:
                self._unlock_left -= 1
                self._mpc.safety_margin_override = 0.2
                self._mpc.lateral_funnel_steps = max(
                    self._mpc.lateral_funnel_steps, 24)

            # 追い越し中のみ Q[e_y] を上げ、計画した右経路を前倒し実行（状態変化時のみ即時 update_Q、
            # 既存挙動を無変更）。ピット走行中は下の専用ブロックで Q[e_y] を設定するためここでは触らない。
            if self._on_pit:
                pass
            elif self._ot_state == "OVERTAKING":
                if self._ot_q_applied != "overtake":
                    _q = list(self._cfg.mpc.Q)  # [e_y, e_psi, t]（rqt変更も反映）
                    _q[0] = self._ot_q_ey
                    self._mpc.update_Q(sparse.diags(_q))
                    self._ot_q_applied = "overtake"
            else:
                # 2026-07-24再撤去(174節、Q曲率スケジュールA/Bテスト): 171節で再導入した
                #   スケジュール(v5+量子化ゲート)を、Q[e_y]ベース値(3M、確定値・不変)は
                #   維持したまま単独変数として一時的にOFFにする。予選環境の「全体的な蛇行」
                #   の原因切り分けのため。static Q(rqt変更も反映)へ戻すのみ。
                if self._ot_q_applied != "normal":
                    self._mpc.update_Q(sparse.diags(list(self._cfg.mpc.Q)))
                    self._ot_q_applied = "normal"
            self._pf_mark('state_v_safe')

            # C 守り: 壁近接減速。「カート端-壁」が小さければ減速(ぶつかりそうなら減速)。
            # 2026-07-14修正: 従来は被追い越し中(_being_overtaken)限定だったが、0713-05実測で
            #   (a) OVERTAKING中の大オフセット委託時(wp127-136、offset目標がub0=壁境界その
            #       ものまで伸びきり、その間ずっとno_limit/eff_v_cap=全開速度)、(b) 完全ソロ
            #       走行中(wp270-280、fwd=0・n_dynobs=0、トラフィック追跡と無関係にコリドー
            #       自体が0.5〜0.7mまで自然に狭まる区間)の両方で、実測壁マージンを一切見ずに
            #       全開のまま壁へ接近し衝突する事象を確認した(「第2コーナー立ち上がり後の
            #       左壁への衝突が2週連続」の直接原因と推定)。トラフィック追跡状態・
            #       被追い越し中かどうかに関わらず常時この実測壁マージン監視を適用する
            #       (既存のwall_slow_margin/wall_slow_speedをそのまま再利用、新規パラメータなし)。
            # 2026-07-15再修正(76節対処案②) → 2026-07-16revert(79節): 現在wp1点のみの
            #   評価では、コーナー進入直前の急激な壁接近に反応が間に合わない(実測
            #   wp204-234)という仮説のもと、_cur_eyを固定して直近窓内のub/lbを走査する
            #   先読みを一時追加した。しかし実走行(0715-07/08)で、この「自車の現在の
            #   横偏差を固定して先のwaypointと比較する」手法自体が誤りだったことが判明:
            #   自車はMPCが能動的にコリドーへ追従するため、実際にコーナーへ差し掛かる
            #   頃にはeyは変化しているのに「今のeyのまま」で先を評価すると、コーナーの
            #   きつさに応じて見かけ上のマージンが物理的にあり得ないほど負に振れる
            #   (対向車位置lat_oを固定するalong_lat先読み(3621行目付近)には他車がどう
            #   動くか分からないので妥当な近似だが、能動的に経路追従する自車自身には
            #   不適切だった)。実測でwall=の50〜65%が負値(最悪-2.41m)を記録し、
            #   障害物が一切ない完全にクリアな直線でも誤発動しCOLLISION-SUSPECTEDまで
            #   発火したため、44/45節の実装(現在wp1点のみの評価)へrevertする。
            #   なお元々の「wp204-234で発動していたのに間に合わなかった」事象自体は、
            #   実際には検知(wmargin<0.5m)は正しく機能しており(0.08〜0.47mの非負実測値)、
            #   先読みの有無とは別の問題(応答量・応答遅れ)だった可能性が高いと判明した
            #   ため、この課題は別途独立して検討する(design_docs 79節参照)。
            # 2026-07-19修正(122節、Sランク根本原因、ユーザー承認済み設計): 上記の
            #   wp.ub/lb(起動時1回計算の静的テーブル、safety_margin未控除)を使う方式は、
            #   MPC自身が実際にQPで拘束・追従するcorridor(dbg_corr_ub0/lb0、毎周期
            #   update_path_constraints()で動的計算、safety_margin=NORMAL時width/√2
            #   ≈1.626m・OVERTAKING時0.8m控除済み)とは完全に独立した別ソースだった。
            #   wall_slowが「余裕あり」(wall=None)と判定していても、MPCが実際に追従
            #   するcorridorは既に極小、ということが起こり得た。0719-04実測(wp330-336、
            #   1周目最終コーナー、kappa最大0.147)で、dbg_corr_ub0=0.484mの状態でも
            #   wall=Noneのままユーザー目視で右壁接触を確認(全4周で再現)。wall_slowの
            #   判定式をdbg_corr_ub0/lb0ベースへ置き換え、MPCが認識するcorridorと同一の
            #   ものを監視する(新規パラメータ・新規計算0個)。dbg_corr_ub0/lb0は既に
            #   safety_margin込みのため、hw(半幅)の追加減算は二重マージン化になり
            #   行わない(3603-3606行目のoffsetクランプ処理と同じ既存規約)。
            #   dbg_corr_ub0/lb0はself._mpc.get_control()(このブロックより後で呼ばれる)
            #   が更新するため前周期の値になるが、これは3608行目・4227行目の既存箇所と
            #   同じ許容済みの遅延特性(1周期≈25ms、40Hzループでの移動量は僅か)。
            # 2026-07-19修正(124節、ユーザー承認済み設計、15km/hでの健全な走行を確立する
            #   一環): 123節で閾値を0.15へ再較正したが、実測(0719-05/run_perffix_*)で
            #   margin=+0.01〜+0.15(実際にはまだ逸脱していない)の27箇所全てで一律に
            #   wall_slow_speed(2.0m/s)まで急減速しており、これら全てが122節以前(静的
            #   wp.ub/lb、閾値0.5)では発火していなかったことを遡及検証で確認した
            #   (旧式margin換算で+0.79〜+1.05と十分な余裕があった)。1周あたり約4秒の
            #   不要な減速(83秒→87秒)の直接原因と判定。判定条件(コリドーが狭まって
            #   いること自体)は正しいが、応答が「介入なし/2.0m/s」の二値でありすぎる
            #   ことが問題だった。既存のicc_f3(前車追従+クリープ床、3844行目付近)が
            #   採用する hard_stop_gap〜f3_taper_gap 間の線形テーパーと全く同じ考え方を
            #   適用し、wall_slow_margin(0.15、テーパー開始点として再利用)〜
            #   wall_slow_margin_hard(新規、実際の境界到達点)の間を線形補間する。
            #   上限速度はself._mpc.input_constraints["umax"][0](現在のv_max、既存値の
            #   再利用)とすることで、将来v_maxを引き上げた場合も自動的に追従する
            #   (新規パラメータ1個のみ: wall_slow_margin_hard)。
            try:
                _corr_ub0 = self._mpc.dbg_corr_ub0
                _corr_lb0 = self._mpc.dbg_corr_lb0
                if self._wall_slow_enable and np.isfinite(_corr_ub0) and np.isfinite(_corr_lb0):
                    _m_left = _corr_ub0 - _cur_ey   # 左コリドー境界までの余裕
                    _m_right = _cur_ey - _corr_lb0  # 右コリドー境界までの余裕
                    _wmargin = min(_m_left, _m_right)
                    if _wmargin < self._wall_slow_margin:
                        if _wmargin <= self._wall_slow_margin_hard:
                            _wall_cap = self._wall_slow_speed
                        else:
                            _frac = ((_wmargin - self._wall_slow_margin_hard)
                                      / (self._wall_slow_margin - self._wall_slow_margin_hard))
                            _umax = float(self._mpc.input_constraints["umax"][0])
                            _wall_cap = self._wall_slow_speed + _frac * (_umax - self._wall_slow_speed)
                        _v_safe_pre = (_wall_cap if _v_safe_pre is None
                                       else min(_v_safe_pre, _wall_cap))
                        _fwd_dbg["wall_slow"] = round(_wmargin, 2)
                        _v_safe_cand.append(("wall_slow(壁際減速)", _wall_cap))
            except Exception:
                pass

            # 2026-07-20追加(127節続報): _footprint_risk(スキャン直後に計算済み、上記
            #   3189行目付近参照)をstate/branchに関わらず常時wall_slowと同じ層で速度
            #   キャップする(wall_slow_speedを再利用、新規速度定数0個)。
            if _footprint_risk:
                _v_safe_pre = (self._wall_slow_speed if _v_safe_pre is None
                               else min(_v_safe_pre, self._wall_slow_speed))
                _fwd_dbg["footprint_risk"] = 1
                _v_safe_cand.append(("footprint_risk(車体重なりリスク)", self._wall_slow_speed))

            # 2026-07-22追加(153節、第3コーナー繰り返しfootprint_risk根治): footprint_risk
            #   自体(fwd_dlat<along_min_widthかつfwd_ds<along_min_length)は物理的接触
            #   リスクの検知としては正しいが、応答が二値の急停止しか無い。実測(0721-03
            #   wp172-176)では、車両の実位置(_cur_ey)がオフセット目標へ追いつくのに
            #   数秒〜9秒規模かかる(112節で既知、target=-3.0mで実測9秒)一方、
            #   fwd_dsは1〜2秒で危険域(along_min_length)に達しており、間に合わないまま
            #   footprint_riskで何度も強制停止に陥っていた(実測波形で確認、offset目標が
            #   -0.55→-1.53mへ動く間もfwd_dlatは0.26m付近でほぼ不変)。124節でwall_slowに
            #   適用した「二値→線形テーパー」と同じ設計を、footprint_risk本体が発火する
            #   手前(まだdlatが物理下限未満のまま接近している間)にも適用し、実オフセットが
            #   育つ時間を確保する。テーパー開始点は既存_ot_pass_clear(3.0m、抜き切り
            #   距離として既に定義済み)を再利用し、新規パラメータは0個。dlat条件を
            #   footprint_risk自身と同じalong_min_widthでゲートすることで、既に十分
            #   離れている(側並走が安定した)場面には一切作用しない。
            # 2026-07-22修正(issue⑤②): 条件式をここで独立に再定義せず、_footprint_risk
            #   本体と同じ場所で計算済みの_fp_near_zone(危険域全体、ds<ot_pass_clear)を
            #   再利用する(_footprint_risk=Trueの場合はifで既に処理済みのため、ここでは
            #   実質的にalong_min_length<=ds<ot_pass_clearの範囲のみ該当)。cooldown解除
            #   判定(152節)と同一の式を共有することで、閾値変更時に3箇所が自動的に
            #   同期する(159節と同じ「同じ周期の同じ値を使う」原則)。
            elif _fp_near_zone:
                _fp_frac = ((abs(_fwd_ds) - self._along_min_length)
                            / (self._ot_pass_clear - self._along_min_length))
                _umax = float(self._mpc.input_constraints["umax"][0])
                _fp_taper_cap = self._wall_slow_speed + _fp_frac * (_umax - self._wall_slow_speed)
                _v_safe_pre = (_fp_taper_cap if _v_safe_pre is None
                               else min(_v_safe_pre, _fp_taper_cap))
                _fwd_dbg["footprint_taper"] = round(abs(_fwd_ds), 2)
                _v_safe_cand.append(("footprint_taper(接触リスク接近テーパー)", _fp_taper_cap))

            # 2026-07-22追加(issue④①、fallback_forwardの操舵盲目化対策): core/MPC.py
            #   のget_control()はinfeasibleが続く間、前回成功時の計画軌道を最大N-2周期
            #   (≈0.45秒)先送りするが、それを超えると操舵を強制的にゼロ固定する
            #   (mpc_controller.py側のfallback_forward分岐)。速度側は_v_safe_preで
            #   既にキャップされるが、操舵側には対応する安全網が無く、非ゼロ速度のまま
            #   コリドー・相手車を無視して直進し続けうる(実測: 0722-4ログd2、infeas=282
            #   まで悪化する間u0最大2.78m/sを確認)。_ot_infeasible_stop(5、OVERTAKING
            #   中は既にこの周期数でSTOPPINGへ委譲している既存閾値)を開始点、
            #   self._mpc.N-2(操舵が完全に盲目になる点、core/MPC.py get_control()の
            #   先送りバッファ計算式と同一)を終了点とし、124/154節と同じ「二値→線形
            #   テーパー」でv_safeをそこまでに0へ収束させる。state/branchに関わらず
            #   常時適用(wall_slow/footprint_riskと同じ設計)。新規パラメータ0個。
            _infeas_now = self._mpc.infeasibility_counter
            if _infeas_now > self._ot_infeasible_stop:
                _blind_at = max(self._ot_infeasible_stop + 1, self._mpc.N - 2)
                _infeas_frac = min(1.0, (_infeas_now - self._ot_infeasible_stop)
                                    / (_blind_at - self._ot_infeasible_stop))
                _infeas_cap = (1.0 - _infeas_frac) * float(self._mpc.input_constraints["umax"][0])
                _v_safe_pre = (_infeas_cap if _v_safe_pre is None
                               else min(_v_safe_pre, _infeas_cap))
                # 検証ロギング(③): 専用フィールドは追加しない。[OT]ログの既存infeas=
                #   (post-solve値、こちらはpre-solve値のため最大1周期ずれるのみ)と
                #   v_safe_src="infeas_taper(...)"の組み合わせで、発火状況・実際に
                #   採用されたかは既存フィールドのみで判別できる(②非冗長性)。
                _v_safe_cand.append(("infeas_taper(操舵盲目化への減速)", _infeas_cap))

            # 並走ねばり(2026-07-03): 真横に車がいても「自分のレーンが確保できる限り減速しない」。
            #   自分側レーン幅 = 相手の自分側エッジ〜自分側の壁 を現在+先読みwpで評価(コーナー狭窄を事前検知)。
            #   lane >= lane_need → 並走継続(ねばる) / lane < lane_need → 走行継続困難:
            #     相手より少し遅くして後ろへ下がる(横並び解消→前方車になればICCが引き継ぐ)。
            #     lane < カート幅 → 物理的に不可 → wall_slow_speed まで強く減速。
            # 2026-07-09シンプル化: 自分がOVERTAKING中(攻めている最中)はスコープ外にする。
            #   本来は防御用(自分が抜かれる側)の機構で、攻めの速度決定はG-2に一元化済み。
            #   両者が同時に_v_safe_preへ介入すると「なぜこの速度か」の追跡が煩雑になっていた。
            if self._ot_state != "OVERTAKING" and _scan["along_lat"] is not None:
                try:
                    _a_lat = _scan["along_lat"]
                    _opp_right = (_a_lat < _cur_ey)   # 相手が自分の右か
                    _wps = self._reference_path.waypoints
                    _n_wp = len(_wps)
                    _n_ahead = max(1, int(self._along_lookahead /
                                          max(self._reference_path.resolution, 1e-3)))
                    _lane_min = float("inf")
                    for _k in range(_n_ahead):
                        _w = _wps[(_idx + _k) % _n_wp]
                        if _opp_right:
                            _lane = float(_w.ub) - (_a_lat + self._ot_block_half)  # 相手左端→左壁
                        else:
                            _lane = (_a_lat - self._ot_block_half) - float(_w.lb)  # 右壁→相手右端
                        _lane_min = min(_lane_min, _lane)
                    # 2026-07-10簡素化: 瞬時値をEMAで平滑化(side_blockと同じ時定数)してから
                    #   閾値判定。コーナー通過中の一瞬の凹みで急減速しないようにする。
                    # 2026-07-17追加(94節): along車の対象IDが変わった周期は、別の車の
                    #   lane_minが混入しないよう新しい基準値へ静かに再スタートする
                    #   (LAT-TTCの_vid_changed処理と同一の考え方)。
                    _along_vid_now = _scan.get("along_vid")
                    if (self._along_lane_ema is not None
                            and _along_vid_now != self._along_lane_prev_vid):
                        self._along_lane_ema = None
                    self._along_lane_prev_vid = _along_vid_now
                    if self._along_lane_ema is None:
                        self._along_lane_ema = _lane_min
                    else:
                        self._along_lane_ema += self._ot_ema_alpha * (
                            _lane_min - self._along_lane_ema)
                    _lane_eff = self._along_lane_ema
                    _fwd_dbg["lane"] = round(_lane_eff, 2)
                    if _lane_eff < self._along_min_width:      # カート幅未満=物理的に通れない
                        _v_yield = self._wall_slow_speed
                    elif _lane_eff < self._along_lane_need:    # 狭い=継続困難→後ろへ下がる
                        _v_yield = max(0.0, float(_scan["along_vlong"]) - 0.5)
                    else:
                        _v_yield = None                        # レーン確保→ねばる(減速しない)
                    if _v_yield is not None:
                        _v_safe_pre = _v_yield if _v_safe_pre is None \
                            else min(_v_safe_pre, _v_yield)
                        _v_safe_cand.append(("along_lat(並走レーン不足)", _v_yield))
                except Exception:
                    pass
            else:
                self._along_lane_ema = None  # 非適用状態への遷移時はリセット(次回新規スタート)
                self._along_lane_prev_vid = None
            self._pf_mark('along_lat')

            _fwd_dbg["state"] = self._ot_state
            _fwd_dbg["side"] = self._ot_side
            _fwd_dbg["offset"] = self._ot_alpha * self._mpc.lateral_target
            _fwd_dbg["n_obs"] = _scan["n_obs"]
            _fwd_dbg["n_fwd"] = _n_fwd
            _fwd_dbg["d_min"] = _fwd_ds
            # 診断用(2026-07-09): エンゲージ判定基準(Fix I, dlat)を直接出力。
            #   長時間STOPPINGのまま未エンゲージの原因切り分けに、これまでbag/CSV突合が
            #   必要だった(2026-07-09予選 t=931〜951の20秒未エンゲージ事象で原因未確定)。
            _fwd_dbg["fwd_dlat"] = _scan.get("fwd_dlat")
            # 診断用(2026-07-09): オフセット目標(lateral_target)が直近コリドー境界を
            #   超えていないか(壁マージン)を直接ログへ出力。実測(t=1783589006.76)で
            #   offset=3.0m要求 vs 実コリドーub0=0.645mの2.36m超過を、これまでは
            #   corr[]との手動突合で発見していた。1行で即座に分かるようにする。
            _corr_ub0_prev = getattr(self._mpc, "dbg_corr_ub0", float('nan'))
            _corr_lb0_prev = getattr(self._mpc, "dbg_corr_lb0", float('nan'))
            if self._ot_state == "OVERTAKING" and self._ot_side != 0:
                _bound = _corr_ub0_prev if self._ot_side > 0 else -_corr_lb0_prev
                _fwd_dbg["offset_margin"] = round(_bound - abs(_fwd_dbg["offset"]), 3)
            else:
                _fwd_dbg["offset_margin"] = None
            # 診断用(2026-07-09): wp_id・_plan_passの窓内実測最小幅(scanの単点Lfree/Rfreeとは
            #   別物)をログへ出力。次回ログのみでbagデコード無しにコーナー位置・veto判断根拠を
            #   検証できるようにする(前回はbag位置→CSV曲率の突合が必要だった)。
            _fwd_dbg["wp_id"] = int(self._mpc.model.wp_id)
            _fwd_dbg["plan_lf"] = round(self._dbg_plan_lf, 2)
            _fwd_dbg["plan_rf"] = round(self._dbg_plan_rf, 2)
            _fwd_dbg["n_dynobs"] = self._dbg_n_dynobs
            _fwd_dbg["left_free"] = _left_free
            _fwd_dbg["right_free"] = _right_free

            # 相手速度マップの検証出力(~1Hz, 40Hzループ外の間引き。データはbagで復号・可視化はRViz)
            self._opp_map_pub_loop += 1
            if self._opp_map_pub_loop >= int(self._mpc_cfg.control_rate):  # ≈1秒ごと
                self._opp_map_pub_loop = 0
                if self._opp_map.vids():
                    _t0 = _time.perf_counter()
                    self._publish_opponent_speed_map()
                    self._pf_add('opp_pub', _time.perf_counter() - _t0)
        # --- end gate2 state machine ---

        # ピット走行中は「純経路追従」にする。
        #   占有マップは車庫/ピットをドライブ可能帯として持たないため、回避コリドー(update_path_constraints)が
        #   「No feasible free segment」で崩壊→MPC infeasible→無操舵で壁へ、となる。
        #   ピットに相手はいないので回避は不要。回避OFF＝静的コリドー(経路追従)で素直に経路を辿る。
        if self._pit_enable and self._on_pit:
            self._mpc.use_obstacle_avoidance = False
            # from_garage 中心線を厳密に追従するため Q[e_y] を強化
            if getattr(self, "_ot_q_applied", None) != "pit":
                _qp = list(self._cfg.mpc.Q)
                _qp[0] = self._pit_q_ey
                self._mpc.update_Q(sparse.diags(_qp))
                self._ot_q_applied = "pit"
            # 速度上限を低速に（最終キャップ。ref_vel/ MPC v_max より優先）
            _v_safe_pre = self._pit_v_max if _v_safe_pre is None else min(_v_safe_pre, self._pit_v_max)

        # 対策①: レースライン追従。通常走行(非ピット・追い越し非アクティブ)では横目標を
        #   レースライン(e_y=0)へブレンドする（既存 lateral_blend/lateral_target を再利用）。
        #   line_follow_w=0 で従来(中心追従)に完全一致。追い越し中/ランプ中(_ot_alpha>0)は
        #   既存挙動を維持し、追い越し終了で alpha が 0 に戻ったら自動的にレースライン追従へ復帰。
        _ot_active = bool(self.USE_OBSTACLE_AVOIDANCE) and (getattr(self, "_ot_alpha", 0.0) > 1e-6)
        if (not self._on_pit) and (not _ot_active):
            self._mpc.lateral_blend = self._line_follow_w
            self._mpc.lateral_target = 0.0

        # 176節続報(曲率スイング検知→R[delta]動的引き上げ): OT/pit状態と無関係に毎周期
        #   計算する(宣言箇所の初期化コメント参照)。前方lookahead_wp点のkappaのmax-min
        #   (=swing)を求め、時間方向EMA(v5の教訓、166-24節)で経路自体の局所ノイズを
        #   均してからsmoothstepで連続的にR[delta]へ反映する。
        # 2026-07-25追加(177節続報): dev3実走行で処理落ち悪化を観測した際、量子化ゲートを
        #   通過した`update_R`自体(r_delta_swing_update)は計測していたが、その手前の
        #   kappa取得ループ+EMA+smoothstepは毎周期無条件に走るにもかかわらず未計測だった。
        #   ユーザー指摘(「各処理にかかる時間を計測できていますか」)を受け、ブロック全体を
        #   1本のタイマー(r_delta_swing_total)で囲み、量子化ゲートで弾かれた周期も含めた
        #   本機構の実コスト全体を漏れなく計測する。
        _t0_total = _time.perf_counter()
        _kappas_fwd = [
            self._reference_path.get_waypoint(self._mpc.model.wp_id + _i).kappa
            for _i in range(self._r_delta_swing_lookahead_wp + 1)]
        _swing_raw = max(_kappas_fwd) - min(_kappas_fwd)
        if self._r_delta_swing_ema is None:
            self._r_delta_swing_ema = _swing_raw  # 初回はEMAを生値で初期化
        else:
            self._r_delta_swing_ema = (
                self._r_delta_swing_ema_beta * _swing_raw
                + (1 - self._r_delta_swing_ema_beta) * self._r_delta_swing_ema)
        _swing = self._r_delta_swing_ema
        _t_sw = min(1.0, max(0.0, (_swing - self._r_delta_swing_kappa_lo)
                             / (self._r_delta_swing_kappa_hi - self._r_delta_swing_kappa_lo)))
        _smooth_sw = _t_sw * _t_sw * (3 - 2 * _t_sw)  # smoothstep(離散切替を避ける、166-24節の教訓)
        _r = list(self._cfg.mpc.R)  # [v, delta]（rqt変更も反映）
        _r[1] = _r[1] + self._r_delta_swing_boost * _smooth_sw
        # 検証ロギング③④: 167節で「毎周期無条件update_Qが予選環境の処理落ちを悪化させた」
        #   回帰を発見した教訓を最初から適用し、量子化ゲート(目標値がboostの1%以上動いた
        #   時のみ実際にupdate_Rを呼ぶ)を組み込む。呼び出し回数も計装し[R-DELTA-SWING]
        #   ログで直接検証できるようにする(167節は事後にしか気付けなかった反省)。
        if (self._r_delta_applied_value is None
                or abs(_r[1] - self._r_delta_applied_value) > self._r_delta_swing_boost * 0.01):
            _t0 = _time.perf_counter()
            self._mpc.update_R(sparse.diags(_r))
            self._pf_add('r_delta_swing_update', _time.perf_counter() - _t0)
            self._r_delta_applied_value = _r[1]
            self._r_delta_swing_update_count += 1
        self._pf_add('r_delta_swing_total', _time.perf_counter() - _t0_total)
        self._r_delta_swing_dbg_loop += 1
        if self._r_delta_swing_dbg_loop % int(max(1, self._mpc_cfg.control_rate)) == 0:
            self.get_logger().info(
                f"[R-DELTA-SWING] wp_id={self._mpc.model.wp_id} swing={_swing:.3f} "
                f"smooth={_smooth_sw:.2f} r_delta_target={_r[1]:.1f} "
                f"r_delta_applied={self._r_delta_applied_value:.1f} "
                f"updates={self._r_delta_swing_update_count}")

        # 2026-07-26追加(徹底解析: 遅延補償の動的スケーリング): 現在速度でwp_id_offset
        #   (1点≈1m)がT_delayをカバーできているか毎周期チェックし、不足時のみ底上げする。
        #   ceil()で切り上げる(round()だと約27km/hまで変化が出ず、既に悪化を実測済みの
        #   20km/hに間に合わないため)。既存wp_id_offset(inside-cut対策、_wp_id_offset_base)
        #   を下回ることはない(max())。R-DELTA-SWINGと同じ量子化ゲート(値が実際に
        #   変わった周期のみupdate)+edge-triggeredログのパターンを踏襲する。
        # 2026-07-27追加(199節)→撤去(200節、2026-07-27): 予選環境との差分60msをこの
        #   先読み補正機構へdebug_extra_actuator_delay_sとして上乗せする拡張を一時追加した
        #   が、実ログ照合の結果、実際のwp_spacing(self._reference_path.resolution)は
        #   レース速度域で早期に閾値飽和し、追加分を拾う余地がほとんど残らないことが判明
        #   (量子化が粗く実効性なし)。加えて速度依存の幾何学的補正を積み増す方向は、
        #   Q/R速度スケジュールが過去に振動源として不採用となった教訓とも相性が悪い。
        #   ユーザー判断により、この差分はQP自体には触れる先読み補正ではなく、
        #   debug_extra_actuator_delay_s=0.055を注入した状態を標準チューニング条件とした
        #   Q/R再チューニングで吸収する方針に転換したため、拡張部分のみ撤去し元のT_delay
        #   単体ロジックに戻す。
        _v_odom_now = abs(self._odom.twist.twist.linear.x)
        _wp_spacing = max(self._reference_path.resolution, 1e-3)
        _wp_id_offset_needed = int(np.ceil(_v_odom_now * self._delay_t_delay_s / _wp_spacing))
        _wp_id_offset_target = max(self._wp_id_offset_base, _wp_id_offset_needed)
        if _wp_id_offset_target != self._wp_id_offset_applied:
            self._mpc.update_wp_id_offset(_wp_id_offset_target)
            self.get_logger().info(
                f"[WP-OFFSET-DELAY] v={_v_odom_now:.2f}m/s "
                f"T_delay={self._delay_t_delay_s * 1000:.0f}ms "
                f"wp_id_offset: {self._wp_id_offset_applied} -> {_wp_id_offset_target}")
            self._wp_id_offset_applied = _wp_id_offset_target

        self._pf_mark('traffic_ot')
        with self._stats.time_block("control"):
            u, max_delta = self._mpc.get_control()
        self._pf_mark('mpc')
        # 178節続報(2026-07-25): 「mpc」区間が障害物数(n_dynobs_max)と無相関と判明した
        #   ため、Python側の行列組み立て(setup)とOSQPソルバー本体(solve)のどちらが
        #   支配的かを切り分ける。MPC.get_control()側で既に計測済みの値を[PERF]へ転記
        #   するだけ(新規計測ロジックはMPC.py側、ここでは読み出しのみ)。リトライ回数は
        #   _pf_addが時間[秒]前提(×1000でms表示)のため転記せず、既存の
        #   "Relaxed safety margin"ログで引き続き追跡する。
        self._pf_add('mpc_setup', getattr(self._mpc, 'last_setup_time', 0.0))
        self._pf_add('mpc_solve', getattr(self._mpc, 'last_solve_time', 0.0))
        # 179節続報: setupの内訳をさらに切り分け(線形化ループ vs コリドー光線走査)
        self._pf_add('mpc_linearize', getattr(self._mpc, 'last_linearize_time', 0.0))
        self._pf_add('mpc_corridor', getattr(self._mpc, 'last_corridor_time', 0.0))

        # --- 案X: get_control 後にMPCコリドー診断を収集（記録のみ）---
        if self.USE_OBSTACLE_AVOIDANCE:
            _fwd_dbg["corr_ub0"] = getattr(self._mpc, "dbg_corr_ub0", float('nan'))
            _fwd_dbg["corr_lb0"] = getattr(self._mpc, "dbg_corr_lb0", float('nan'))
            _fwd_dbg["corr_xr0"] = getattr(self._mpc, "dbg_corr_xr0", float('nan'))
            _fwd_dbg["corr_wmin"] = getattr(self._mpc, "dbg_corr_wmin", float('nan'))
            _fwd_dbg["corr_src"] = getattr(self._mpc, "dbg_corr_src", -1.0)
            _rp = self._mpc.model.reference_path
            _fwd_dbg["nseg0"] = getattr(_rp, "dbg_nseg0", 0)
            _fwd_dbg["nseg1"] = getattr(_rp, "dbg_nseg1", 0)
            _fwd_dbg["nseg2"] = getattr(_rp, "dbg_nseg2", 0)

        # --- forward obstacle longitudinal stop/follow (v2e, 2-stage) ---
        if self.USE_OBSTACLE_AVOIDANCE:
            v_safe = _v_safe_pre
            _v_max_clip = float(self._mpc.input_constraints["umax"][0])  # m/s (v_max)
            if v_safe is not None:
                v_safe = min(v_safe, _v_max_clip)
            _branch = "none"
            if v_safe is not None:
                if (u is None) or (len(u) < 2) or (abs(u[0]) < 1e-3 and self._mpc.infeasibility_counter > 0):
                    _steer = u[1] if (u is not None and len(u) >= 2) else 0.0
                    u = np.array([v_safe, _steer])
                    _branch = "fallback_forward"
                else:
                    if v_safe < u[0]:
                        _branch = "decelerate"
                    else:
                        _branch = "no_limit"
                    u[0] = min(u[0], v_safe)
            # 診断用(2026-07-09): 積んだ候補のうち実際に最小値を出した機構名を確定。
            _v_safe_src = (min(_v_safe_cand, key=lambda kv: kv[1])[0]
                           if _v_safe_cand else None)
            # 2026-07-17追加(91節、④v_safe候補の相互作用のログ収集用): [OT]ログは1Hzに
            #   間引かれており、複数のv_safe候補(wall_slow/icc_f3/line_cap/LAT-TTC C1等)
            #   が周期単位で入れ替わるチャーンを1Hz解像度では追えない(86節③で観測した
            #   0.0→2.4→0.8km/hの細かい往復の原因切り分けに、変化した周期そのものが必要)。
            #   OVERTAKING中のみ、v_safe_srcが前周期と変化した瞬間を間引かずに記録する。
            if (self._ot_state == "OVERTAKING"
                    and _v_safe_src != self._v_safe_src_prev):
                self.get_logger().info(
                    f"[V-SAFE-SRC-CHANGE] {self._v_safe_src_prev} -> {_v_safe_src} "
                    f"v_safe={v_safe} offset={_fwd_dbg.get('offset')} "
                    f"wp={self._mpc.model.wp_id}")
            self._v_safe_src_prev = _v_safe_src
            if getattr(self, "_ot_debug", False):
                self._ot_dbg_loop += 1
                if self._ot_dbg_loop % int(max(1, self._mpc_cfg.control_rate)) == 0:
                    _u0 = (u[0] if (u is not None and len(u) >= 2) else None)
                    # 2026-07-17追加(86/87節、検証ロギングのみ・判定ロジック無変更):
                    #   core/MPC.py get_control()の`sm`計算式と完全に同一の優先順位で、
                    #   MPCソルバーが今回実際に使うsafety_marginを毎周期記録する。
                    #   [MPC] Relaxed safety marginの発生頻度・タイミングと、この値・
                    #   corr[wmin]・planLf/planRfを次回ログで直接突き合わせるための
                    #   純粋な観測用フィールド。
                    _sm_ot = (self._mpc.safety_margin_override
                              if self._mpc.safety_margin_override is not None
                              else self._mpc.model.safety_margin)
                    self.get_logger().info(
                        f"[OT] state={_fwd_dbg.get('state')} side={_fwd_dbg.get('side')} "
                        f"obs={_fwd_dbg.get('n_obs')} fwd={_fwd_dbg.get('n_fwd')} "
                        f"wp_id={_fwd_dbg.get('wp_id')} "
                        f"d_min={_fwd_dbg.get('d_min')} "
                        f"Lfree={_fwd_dbg.get('left_free')} Rfree={_fwd_dbg.get('right_free')} "
                        f"planLf={_fwd_dbg.get('plan_lf')} planRf={_fwd_dbg.get('plan_rf')} "
                        f"n_dynobs={_fwd_dbg.get('n_dynobs')} "
                        f"fwd_dlat={_fwd_dbg.get('fwd_dlat')} offset_margin={_fwd_dbg.get('offset_margin')} "
                        f"v_safe_src={_v_safe_src} "
                        f"offset={_fwd_dbg.get('offset')} psi_bias={_fwd_dbg.get('psi_bias')} "
                        f"corr_bound={_fwd_dbg.get('corr_bound')}@{_fwd_dbg.get('corr_bound_at')}m "
                        f"corr[ub0={_fwd_dbg.get('corr_ub0')} lb0={_fwd_dbg.get('corr_lb0')} "
                        f"xr0={_fwd_dbg.get('corr_xr0')} wmin={_fwd_dbg.get('corr_wmin')} "
                        f"src={_fwd_dbg.get('corr_src')} nseg0/1/2={_fwd_dbg.get('nseg0')}/{_fwd_dbg.get('nseg1')}/{_fwd_dbg.get('nseg2')}] "
                        f"v_safe={v_safe} u0={_u0} infeas={self._mpc.infeasibility_counter} "
                        f"branch={_branch} vopp={_fwd_dbg.get('vopp')} worth={_fwd_dbg.get('pass_worth')} "
                        f"def={_fwd_dbg.get('def')} wall={_fwd_dbg.get('wall_slow')} "
                        f"fp_taper={_fwd_dbg.get('footprint_taper')} "
                        f"proactive_bias_side={_fwd_dbg.get('proactive_bias_side')} "
                        f"lane={_fwd_dbg.get('lane')} cl={int(self._ot_cleared)} "
                        f"margin={_sm_ot:.3f} "
                        f"gate={_fwd_dbg.get('gate')}")
        # --- end forward obstacle ---

        # ピット走行中は ref_vel(レースライン用) を適用しない（経路が別・wp_id不整合のため）。
        if self._ref_vel_configulator is not None and not self._on_pit:
            _sec_kmh = self._ref_vel_configulator.get_ref_vel(self._mpc.model.wp_id)  # 区間値[km/h]
            _cap_mps = min(kmh_to_m_per_sec(_sec_kmh), self._mpc_cfg.v_max)           # 上限[m/s]
            self._mpc.update_v_max(_cap_mps)
            # 曲率エンベロープと要素毎min: 旧フラット上書きはヘアピン減速を消していた
            #   (毎周壁接触の一因)。直線・緩コーナーは _cap_mps のまま不変。
            _env = getattr(self, "_v_envelope", None)
            if _env is not None and len(_env) == len(self._reference_path.waypoints):
                v_ref: List[float] = np.minimum(_env, _cap_mps).tolist()
            else:
                v_ref = [_cap_mps] * len(self._reference_path.waypoints)
            self._reference_path.set_v_ref(v_ref)

        # override by brake command if control is disabled
        if not self._enable_control:
            last_v_cmd = self._last_u[0]
            if last_v_cmd < 0.5:
                u[0] = 0.0
            else:
                decel_v = last_v_cmd + self._mpc_cfg.a_min * dt
                u[0] = np.clip(decel_v, 0.0, self._mpc_cfg.v_max)

        if len(u) == 0:
            self.get_logger().error("No control signal", throttle_duration_sec=1)
            u = [0.0, 0.0]
            # continue

        acc = 0.
        bug_acc_enabled = False
        if self.USE_BUG_ACC:
            def deg2rad(deg):
                return deg * np.pi / 180.0

            if abs(v) > kmh_to_m_per_sec(44.0) or \
             (abs(v) > kmh_to_m_per_sec(38.0) and abs(max_delta) > deg2rad(12.0)):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_min / 3.0 * 2.0
                self._pred_marker_color = RED
            elif abs(v) > kmh_to_m_per_sec(41.0) or abs(u[1]) > deg2rad(10.0):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_max
                self._pred_marker_color = YELLOW
            else:
                bug_acc_enabled = True
                acc = 500.0
                self._pred_marker_color = CYAN
        # forward obstacle: 静的コリドーは起動時に1回だけ生成する
        # （update_simple_path_constraints は n_waypoints×N の二重ループで重く、
        #   毎ループ呼ぶと制御周期が約12Hzまで低下するため初回のみ実行に変更）
        if not getattr(self, '_static_corridor_ready', False):
            try:
                # ピット走行中は壁(lanelet境界)が近いため専用の小さめマージンでコリドー化する
                # （既定 width/√2≈1.63m は狭路でコリドーが崩壊するため）。
                _sm = self._pit_safety_margin if (self._pit_enable and self._on_pit) \
                    else self._car.safety_margin
                self._car.reference_path.update_simple_path_constraints(
                    self._mpc.N, _sm)
                self._static_corridor_ready = True
            except Exception as _e:
                self.get_logger().warn(f'static corridor init failed: {_e}')

        # 縦方向 P 制御は毎ループ必須。コリドー生成の成否から独立させる。
        acc = self.KP * (u[0] - v)
        acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)
        # u[0] = np.clip(last_u[0] + acc * dt, 0.0, self._mpc_cfg.v_max)

        # apply low pass filter to control signal
        acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
        u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        # 2026-07-22追加(issue④③): _last_acc/_last_uの更新は_publish_control_command内へ
        #   一本化した(呼び出し元によらず実発行値と一致することを保証するため)。ここでの
        #   代入は削除(挙動は同一、通常フローは従来通りpublish直前に最新値へ更新される)。

        # update car state (use v for feedback actual speed)
        self._car.drive([v, u[1]])

        # Publish control command
        # 2026-07-15追加(Stage1.9 T2、design_docs 8-3節で承認待ちのまま未実装だった提案の実装):
        #   pubtail区間は従来"mpcマーク〜pubtailマーク間の全経過時間"としてしか計測されておらず、
        #   command_pub/overtake_statusのどちらが実体かを区別できなかった(計測法の注記、8-2節)。
        #   個別に_pf_addすることで、次回[PERF]ログで内訳を直接確認できるようにする。
        _t0 = _time.perf_counter()
        self._publish_control_command(now, u, acc, bug_acc_enabled)
        self._pf_add('command_pub', _time.perf_counter() - _t0)
        self._stuck_u0_last = float(u[0])  # スタック検知用(2026-07-09): 次周期の指令/実速度比較に使用

        # Log states
        _t0 = _time.perf_counter()
        self._sim_logger.log(self._car, u, t)
        self._sim_logger.plot_animation(t, self._loop, self._current_laps, self._lap_times, is_colliding, u, self._mpc, self._car)
        self._pf_add('simlog', _time.perf_counter() - _t0)

        # 約 0.25 秒ごとに予測結果を表示
        if (self._mpc.current_prediction is not None) and (self._loop % (self._mpc_cfg.control_rate // 4) == 0):
            _t0 = _time.perf_counter()
            self._publish_mpc_pred_marker(self._mpc.current_prediction[0], self._mpc.current_prediction[1]) # type: ignore
            self._pf_add('pred_pub', _time.perf_counter() - _t0)

        # gate2: 追い越し診断トピック（低頻度）
        if self.USE_OBSTACLE_AVOIDANCE and (self._loop % (self._mpc_cfg.control_rate // 4) == 0):
            _t0 = _time.perf_counter()
            self._publish_overtake_status(_fwd_dbg, u)
            self._pf_add('overtake_status', _time.perf_counter() - _t0)
        self._pf_mark('pubtail')
        self._pf_cycle_end(_time.perf_counter() - _pf_work0)

    def run(self) -> None:
        self._wait_until_clock_received()
        self._wait_until_odom_received()
        self._wait_until_trajectory_received()
        self._wait_until_path_constraints_received()

        # initialize car states
        pose = odom_to_pose_2d(self._odom) # type: ignore
        self._car.update_states(pose.x, pose.y, pose.theta)
        self._car.update_reference_path(self._car.reference_path)

        # 発進地点判別: レースラインから遠ければ(車庫/ピット発進)ピット経路を参照に。
        # gate3/eval(車庫発進)→ピット経路、gate1/2(コース上発進)→レースライン のまま。
        if self._pit_enable:
            _d0 = self._race_line_min_dist(pose.x, pose.y)
            if _d0 > self._pit_enter_dist:
                # 発進地点近傍から始まるピット経路を構築
                self._pit_ref_path = self._build_pit_ref_path(pose.x, pose.y)
                self._set_active_path(self._pit_ref_path, on_pit=True)
                self.get_logger().info(
                    f"PIT start (race-line dist={_d0:.1f}m) -> follow pit path at low speed")
            else:
                self.get_logger().info(
                    f"course start (race-line dist={_d0:.1f}m) -> follow race line")

        if self._ref_vel_configulator is None:
            self._publish_ref_path_marker(self._car.reference_path)

        self._pred_marker_color = CYAN

        # for i in range(10):
        #     self._obstacle_manager.push_next_obstacle()

        # initialize control states
        self._control_rate = self.create_rate(self._mpc_cfg.control_rate)
        self._sim_logger = SimulationLogger(
            self.get_logger(),
            self._car.temporal_state.x, self._car.temporal_state.y, self._cfg.sim_logger.animation_enabled, self.SHOW_PLOT_ANIMATION, self.PLOT_RESULTS, self.ANIMATION_INTERVAL) # type: ignore

        self._loop = 0
        self._last_acc = 0.0
        self._last_u = np.array([0.0, 0.0])
        self._t_start = self.get_clock().now()
        self._last_t = self._t_start

        # gate2: STOPPING(回避OFF) は静的コリドー path_constraints を参照するため、
        # 制御ループ開始前に1回生成しておく（初手から STOPPING になっても None 参照で落ちない）。
        if self.USE_OBSTACLE_AVOIDANCE and not getattr(self, '_static_corridor_ready', False):
            try:
                self._car.reference_path.update_simple_path_constraints(
                    self._mpc.N, self._car.safety_margin)
                self._static_corridor_ready = True
            except Exception as _e:
                self.get_logger().warn(f'static corridor pre-init failed: {_e}')

        # Stage1.7 R2(2026-07-07): GC大停止(実測max130ms=制御欠落の主犯②)対策。
        #   初期化で生成した恒久オブジェクトをGC追跡から除外(freeze)し、世代2走査を激減させる。
        #   閾値も引き上げて収集頻度を下げる(効果と副作用は [PERF] gc と [GCTUNE] ログで監視)。
        _gc.collect()
        _gc.freeze()
        _gc.set_threshold(50000, 100, 100)
        self.get_logger().info(
            f"[GCTUNE] freeze={_gc.get_freeze_count()}obj thresholds={_gc.get_threshold()}")

        self.get_logger().info("----------------------")
        self.get_logger().info("START!")
        self.get_logger().info("----------------------")

        while rclpy.ok() and (not self._sim_logger.stop_requested()):
            self._control()

    def stop(self):
        # Wait for stopping
        self.get_logger().warn("----------------------")
        self.get_logger().warn("Stopping...")
        self.get_logger().warn("----------------------")
        timeout_time = self.get_clock().now() + rclpy.time.Duration(seconds=5)
        while self._odom.twist.twist.linear.x > 0.1 and self.get_clock().now() < timeout_time:
            self._enable_control = False
            self._control()

        # Publish zero command to stop the car completely
        zero_cmd = self._create_ackerman_control_command(self.get_clock().now(), [0.0, 0.0], 0.0, False)
        self._command_pub.publish(zero_cmd)

        self.get_logger().warn(">> Stop Completed!")

        # show results
        self._sim_logger.show_results(self._current_laps, self._lap_times, self._car)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
