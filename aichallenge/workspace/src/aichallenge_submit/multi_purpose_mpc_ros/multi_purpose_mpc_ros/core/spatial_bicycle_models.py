import numpy as np
from abc import abstractmethod
try:
    from abc import ABC
except:
    # for Python 2.7
    from abc import ABCMeta

    class ABC(object):
        __metaclass__ = ABCMeta
        pass
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math

# Colors
CAR = '#F1C40F'
CAR_COLLIDING = '#FF0000'
CAR_OUTLINE = '#B7950B'


#########################
# Temporal State Vector #
#########################

class TemporalState:
    def __init__(self, x, y, psi):
        """
        Temporal State Vector containing car pose (x, y, psi)
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: yaw angle | [rad]
        """
        self.x = x
        self.y = y
        self.psi = psi

        self.members = ['x', 'y', 'psi']

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """
        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self


########################
# Spatial State Vector #
########################

class SpatialState(ABC):
    """
    Spatial State Vector - Abstract Base Class.
    """

    @abstractmethod
    def __init__(self):
        self.members = None
        self.e_y = None
        self.e_psi = None

    def __getitem__(self, item):
        if isinstance(item, int):
            members = [self.members[item]]
        else:
            members = self.members[item]
        return [vars(self)[key] for key in members]

    def __setitem__(self, key, value):
        vars(self)[self.members[key]] = value

    def __len__(self):
        return len(self.members)

    def __iadd__(self, other):
        """
        Overload Sum-Add operator.
        :param other: numpy array to be added to state vector
        """

        for state_id in range(len(self.members)):
            vars(self)[self.members[state_id]] += other[state_id]
        return self

    def list_states(self):
        """
        Return list of names of all states.
        """
        return self.members


class SimpleSpatialState(SpatialState):
    def __init__(self, e_y=0.0, e_psi=0.0, t=0.0, delta_actual=0.0):
        """
        Simplified Spatial State Vector containing orthogonal deviation from
        reference path (e_y), difference in orientation (e_psi) and velocity
        :param e_y: orthogonal deviation from center-line | [m]
        :param e_psi: yaw angle relative to path | [rad]
        :param t: time | [s]
        :param delta_actual: actuator's actual (lagged) steering input state,
            first-order-lag toward the commanded input | [rad or curvature,
            same unit as input[1]]. 2026-07-27再実装(201/202節、AXIS06):
            アクチュエータの実際の物理応答遅れ(tau、環境に依らず一定=130ms、
            198節で一度撤回・202節続報でtau=55ms差分案も無効と判明の末に再設計)を
            QPの内部モデルへ明示的に組み込むための第4状態。予選環境特有の追加遅延
            (55-60ms)はこのtauではなくdebug_extra_actuator_delay_s(196節、別機構)
            で扱う。
        """
        super(SimpleSpatialState, self).__init__()

        self.e_y = e_y
        self.e_psi = e_psi
        self.t = t
        self.delta_actual = delta_actual

        self.members = ['e_y', 'e_psi', 't', 'delta_actual']


####################################
# Spatial Bicycle Model Base Class #
####################################

class SpatialBicycleModel(ABC):
    def __init__(self, reference_path, length, width, Ts):
        """
        Abstract Base Class for Spatial Reformulation of Bicycle Model.
        :param reference_path: reference path object to follow
        :param length: length of car in m
        :param width: width of car in m
        :param Ts: sampling time of model
        """

        # Precision
        self.eps = 1e-12

        # Car Parameters
        self.length = length
        self.width = width
        self.safety_margin = self._compute_safety_margin()

        # Reference Path
        self.reference_path = reference_path

        # Set initial distance traveled
        self.s = 0.0

        # Set sampling time
        self.Ts = Ts

        # Set initial waypoint ID
        self.wp_id = 0

        # Set initial waypoint
        self.current_waypoint = self.reference_path.waypoints[self.wp_id]

        # Declare spatial state variable | Initialization in sub-class
        self.spatial_state = None

        # Declare temporal state variable | Initialization in sub-class
        self.temporal_state = None

    def s2t(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state given a reference waypoint.
        :param reference_waypoint: waypoint object to use as reference
        :param reference_state: state vector as np.array to use as reference
        :return Temporal State equivalent to reference state
        """

        # Compute temporal state variables
        if isinstance(reference_state, np.ndarray):
            x = reference_waypoint.x - reference_state[0] * np.sin(
                reference_waypoint.psi)
            y = reference_waypoint.y + reference_state[0] * np.cos(
                reference_waypoint.psi)
            psi = reference_waypoint.psi + reference_state[1]
        elif isinstance(reference_state, SpatialState):
            x = reference_waypoint.x - reference_state.e_y * np.sin(
                reference_waypoint.psi)
            y = reference_waypoint.y + reference_state.e_y * np.cos(
                reference_waypoint.psi)
            psi = reference_waypoint.psi + reference_state.e_psi
        else:
            print('Reference State type not supported!')
            x, y, psi = None, None, None
            exit(1)

        return TemporalState(x, y, psi)

    def t2s(self, reference_waypoint, reference_state):
        """
        Convert spatial state to temporal state. Either convert self.spatial_
        state with current waypoint as reference or provide reference waypoint
        and reference_state.
        :return Spatial State equivalent to reference state
        """

        # Compute spatial state variables
        if isinstance(reference_state, np.ndarray):
            e_y = np.cos(reference_waypoint.psi) * \
                  (reference_state[1] - reference_waypoint.y) - \
                  np.sin(reference_waypoint.psi) * (reference_state[0] -
                                                    reference_waypoint.x)
            e_psi = reference_state[2] - reference_waypoint.psi

            # Ensure e_psi is kept within range (-pi, pi]
            e_psi = np.mod(e_psi + math.pi, 2 * math.pi) - math.pi
        elif isinstance(reference_state, TemporalState):
            e_y = np.cos(reference_waypoint.psi) * \
                  (reference_state.y - reference_waypoint.y) - \
                  np.sin(reference_waypoint.psi) * (reference_state.x -
                                                    reference_waypoint.x)
            e_psi = reference_state.psi - reference_waypoint.psi

            # Ensure e_psi is kept within range (-pi, pi]
            e_psi = np.mod(e_psi + math.pi, 2 * math.pi) - math.pi
        else:
            print('Reference State type not supported!')
            e_y, e_psi = None, None
            exit(1)

        # AXIS07 State Sanitizer(220節続報): EKFの横方向誤差は曲率に比例することが
        # 実測済み(diff=ekf_ey-gnss_ey ≈ -slope*kappa-intercept)。EKF本体やローカリ
        # ゼーションには一切触れず、MPCへ渡す直前のe_yのみをここで補正する。
        if self.use_curvature_bias_correction:
            e_y = e_y + self.curvature_bias_slope * reference_waypoint.kappa \
                + self.curvature_bias_intercept

        # time state can be set to zero since it's only relevant for the MPC
        # prediction horizon
        t = 0.0

        # delta_actual(実際の遅延応答舵角)は(x,y,psi)から幾何学的に導出できない内部
        # フィルタ状態のため、位置更新のたびにリセットせず前回値を引き継ぐ(2026-07-27
        # 再実装)。self.spatial_stateは呼び出し時点でまだ更新前の値を保持している。
        delta_actual = getattr(self.spatial_state, 'delta_actual', 0.0)

        return SimpleSpatialState(e_y, e_psi, t, delta_actual)

    def drive(self, u):
        """
        Drive.
        :param u: input vector containing [v, delta]
        """

        # Get input signals
        v, delta = u

        # Compute temporal state derivatives
        x_dot = v * np.cos(self.temporal_state.psi)
        y_dot = v * np.sin(self.temporal_state.psi)
        psi_dot = v / self.length * np.tan(delta)
        temporal_derivatives = np.array([x_dot, y_dot, psi_dot])

        # Update spatial state (Forward Euler Approximation)
        self.temporal_state += temporal_derivatives * self.Ts

        # Compute velocity along path
        s_dot = 1 / (1 - self.spatial_state.e_y * self.current_waypoint.kappa) \
                * v * np.cos(self.spatial_state.e_psi)

        # Update distance travelled along reference path
        self.s += s_dot * self.Ts

    def _compute_safety_margin(self):
        """
        Compute safety margin for car if modeled by its center of gravity.
        """

        # Model ellipsoid around the car
        # safety_margin = self.width
        # safety_margin = self.width / 2.0
        safety_margin = self.width / np.sqrt(2)
        # safety_margin = self.width / np.sqrt(2) / 2.0
        # safety_margin = 0.0

        return safety_margin

    def get_current_waypoint(self):
        """
        Get closest waypoint on reference path based on car's current location.
        """

        # Compute cumulative path length
        length_cum = np.cumsum(self.reference_path.segment_lengths)
        # Get first index with distance larger than distance traveled by car
        # so far
        greater_than_threshold = length_cum > self.s
        next_wp_id = greater_than_threshold.searchsorted(True)

        # check end of path
        if next_wp_id == len(length_cum):
            self.wp_id = len(length_cum) - 1
            self.current_waypoint = self.reference_path.waypoints[self.wp_id]
            return

        # Get previous index
        prev_wp_id = next_wp_id - 1

        # Get distance traveled for both enclosing waypoints
        s_next = length_cum[next_wp_id]
        s_prev = length_cum[prev_wp_id]

        if np.abs(self.s - s_next) < np.abs(self.s - s_prev):
            self.wp_id = next_wp_id
            self.current_waypoint = self.reference_path.waypoints[next_wp_id]
        else:
            self.wp_id = prev_wp_id
            self.current_waypoint = self.reference_path.waypoints[prev_wp_id]

    def get_closest_waypoint(self, x, y, prev_idx=None, radius_m=None):
        """
        Get the index of the closest waypoint to the given x, y coordinates.
        :param x: x coordinate
        :param y: y coordinate
        :param prev_idx: previous matched waypoint index (search anchor). None -> full global search.
        :param radius_m: search window radius in meters around prev_idx. Ignored if prev_idx is None.
        :return: Index of the closest waypoint

        2026-07-14追加(水平展開: mpc_controller.py._closest_wp_and_sと同一パターン)。
        従来は全waypointからの単純な(x,y)最近傍探索のみで、弧長的な連続性を考慮して
        いなかった。ヘアピン等、コースが壁一枚を挟んで自分自身に近接する箇所では、
        自車の生座標が壁の反対側のwaypointへ誤ってマッチし得る(他車のwall誤認識と
        全く同じ構造)。prev_idx(前回マッチしたwp_id)とradius_mが与えられた場合のみ
        探索をその近傍に限定する。prev_idx=None(初回起動時、基準点が無い)は従来通り
        全waypointから探索する(呼び出し側のupdate_statesが判断する)。
        """
        waypoints = self.reference_path.waypoints
        n = len(waypoints)
        if prev_idx is None or radius_m is None:
            distances = np.sqrt((np.array([wp.x for wp in waypoints]) - x)**2 +
                                (np.array([wp.y for wp in waypoints]) - y)**2)
            return int(np.argmin(distances))

        radius_idx = max(1, int(np.ceil(radius_m / max(self.reference_path.resolution, 1e-3))))
        idxs = np.arange(prev_idx - radius_idx, prev_idx + radius_idx + 1) % n
        xs = np.array([waypoints[i].x for i in idxs])
        ys = np.array([waypoints[i].y for i in idxs])
        distances = np.sqrt((xs - x)**2 + (ys - y)**2)
        local = int(np.argmin(distances))
        return int(idxs[local])

    def get_s_at_waypoint(self, wp_id):
        """
        Calculate the distance s along the reference path corresponding to the
        waypoint.
        :param wp_id: waypoint id
        :return: Distance s along the reference path
        """
        # Compute cumulative path length
        length_cum = np.cumsum(self.reference_path.segment_lengths)

        # Distance s at the closest waypoint
        s_at_closest_wp = length_cum[wp_id]

        return s_at_closest_wp

    def show(self, is_colliding: bool, ax):
        """
        Display car on the provided axis.
        :param ax: Matplotlib axis object to plot on
        """

        # Get car's center of gravity
        cog = (self.temporal_state.x, self.temporal_state.y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(self.temporal_state.psi)

        facecolor = CAR_COLLIDING if is_colliding else CAR

        # Draw rectangle
        car = plt_patches.Rectangle(cog, width=self.length, height=self.width,
                                    angle=yaw, facecolor=facecolor,
                                    edgecolor=CAR_OUTLINE, zorder=20)

        # Shift center rectangle to match center of the car
        car.set_x(car.get_x() - (self.length / 2 *
                                 np.cos(self.temporal_state.psi) -
                                 self.width / 2 *
                                 np.sin(self.temporal_state.psi)))
        car.set_y(car.get_y() - (self.width / 2 *
                                 np.cos(self.temporal_state.psi) +
                                 self.length / 2 *
                                 np.sin(self.temporal_state.psi)))

        # Add rectangle to provided axis
        ax.add_patch(car)

    @abstractmethod
    def get_spatial_derivatives(self, state, input, kappa):
        pass

    @abstractmethod
    def linearize(self, v_ref, kappa_ref, delta_s):
        pass


#################
# Bicycle Model #
#################

class BicycleModel(SpatialBicycleModel):
    def __init__(self, reference_path, length, width, Ts, actuator_lag_tau_s=0.16,
                 actuator_gain=1.0,
                 use_curvature_bias_correction=False,
                 curvature_bias_slope=0.772, curvature_bias_intercept=0.016):
        """
        Simplified Spatial Bicycle Model. Spatial Reformulation of Kinematic
        Bicycle Model. Uses Simplified Spatial State.
        :param reference_path: reference path model is supposed to follow
        :param length: length of the car in m
        :param width: with of the car in m
        :param Ts: sampling time of model in s
        :param use_curvature_bias_correction: AXIS07(EKFの曲率依存横方向誤差)への
            State Sanitizer対処。既定False。Trueの場合、t2s()で算出したe_yに対し
            `e_y += curvature_bias_slope*kappa + curvature_bias_intercept`を適用する。
            既定係数0.772/0.016は211節(|kappa| vs |ekf_ey-gnss_ey|回帰、n=774、r=0.702)由来。
            220節続報で符号付き回帰による検証(0728-02、n=2101)を実施し、
            diff(ekf_ey-gnss_ey) = -0.747*kappa - 0.012(r=-0.703、符号は+0.747/+0.012が
            正しい補正方向であることを確認)、既定値と近い値であることを確認済み。
            EKF本体・ローカリゼーションには一切触れず、MPCへ渡す直前の値のみを補正する。
        :param curvature_bias_slope: 上記補正式の傾き(kappaに対する係数)。
        :param curvature_bias_intercept: 上記補正式の切片。
        :param actuator_lag_tau_s: delta_actual状態(第4状態)の一次遅れ時定数 | [s]。
            2026-07-27再実装(201節続報、AXIS06): 差分のみ(tau=55ms)は、MPCのホライズン
            ステップが表す実時間(delta_s/v_ref、典型的に0.1-0.4s)より常に小さく、
            alphaが常に1.0へ飽和して旧3状態モデルと数値的に区別できないと判明した
            (202節)。202節続報: tauをローカル環境の実測フル遅延(130ms)へ変更し、
            Q/Rをこのモデルに対して再チューニングする方針へ転換した。203節続報:
            実装バグ(e_psi行がdelta_actual_k+1でなくdelta_actual_kを使っていた、
            修正済み)の影響で198節のtau=130ms/190ms実験結果が過小評価されていた
            疑いが生じたため、修正後の実装でtau=190ms(予選環境相当のフル遅延)を
            探索的に再検証する(0.13から一時変更)。0.0を指定するとdelta_actual_next=
            delta_cmd(瞬時追従)となり旧3状態モデルと数学的に完全一致する
            (下位互換の保証、テストで検証)。213節でtau=240msを試したが悪化し190msへ
            復元、208節でAXIS06クローズ(190ms確定)。2026-08-03、`analyze_actuator_delay.py`
            のFOPDT実測(v_max=15/20km/h×Q=700k/1.0Mの4条件中3条件で一致)でtau=160msと
            判明したため、190ms→160msへ変更(design_docs/axis06_gain_correction_design_
            20260803.md参照、緩和策①[ゲイン]棄却後の次善策)。150ms・Q=1.0Mとの組み合わせも
            ローカルA/Bで比較した結果、160ms・Q=700k(現行Q維持)が最良(wp269-282ホットスポット
            PASS、対数減衰率0.88)と確定し、予選環境での検証へ進む。
        :param actuator_gain: delta_actualが定常状態で収束する先を、指令deltaの
            何倍にするかのゲイン | 無次元(既定1.0、下位互換)。2026-08-03、
            `analyze_actuator_delay.py`のestimate_gain_continuousによる実測
            (通常走行域|指令振幅|0-30°で4条件(v_max=15/20km/h×Q=700k/1.0M)すべて
            0.63-0.78、既知値0.67と整合)に基づき導入。連続時間モデルを
            d(delta_actual)/dt = (actuator_gain*delta - delta_actual)/tauへ変更し、
            定常状態でdelta_actual→actuator_gain*deltaに収束するようにする。
            1.0を指定すると全ての変更箇所で乗算が1倍になり、旧モデルと数学的に
            完全一致する(下位互換の保証、テストで検証)。設計根拠:
            design_docs/axis06_gain_correction_design_20260803.md参照。
        """

        # Initialize base class
        super(BicycleModel, self).__init__(reference_path, length=length,
                                           width=width, Ts=Ts)

        self.actuator_lag_tau_s = max(0.0, actuator_lag_tau_s)
        self.actuator_gain = float(actuator_gain)
        self.use_curvature_bias_correction = bool(use_curvature_bias_correction)
        self.curvature_bias_slope = float(curvature_bias_slope)
        self.curvature_bias_intercept = float(curvature_bias_intercept)

        # Initialize spatial state
        self.spatial_state = SimpleSpatialState()

        # Number of spatial state variables
        self.n_states = len(self.spatial_state)

        # Initialize temporal state
        self.temporal_state = self.s2t(reference_state=self.spatial_state,
                                       reference_waypoint=self.current_waypoint)

    def update_reference_path(self, reference_path):
        # Update Reference Path
        self.reference_path = reference_path

        # Update the current waypoint based on the new reference path
        self.wp_id = self.get_closest_waypoint(self.temporal_state.x, self.temporal_state.y)

        # Update the distance s along the reference path based on the closest waypoint
        self.s = self.get_s_at_waypoint(self.wp_id)

    def update_states(self, x, y, psi, prev_idx=None, radius_m=None):
        self.temporal_state.x = x
        self.temporal_state.y = y
        self.temporal_state.psi = psi

        # Update the current waypoint based on the new reference path
        self.wp_id = self.get_closest_waypoint(self.temporal_state.x, self.temporal_state.y,
                                                prev_idx=prev_idx, radius_m=radius_m)

        # Update the distance s along the reference path based on the closest waypoint
        self.s = self.get_s_at_waypoint(self.wp_id)

    def get_temporal_derivatives(self, state, input, kappa):
        """
        Compute relevant temporal derivatives needed for state update.
        :param state: state vector for which to compute derivatives
        :param input: input vector
        :param kappa: curvature of corresponding waypoint
        :return: temporal derivatives of distance, angle and velocity
        """

        # Get state and input variables
        e_y, e_psi, t, delta_actual = state
        v, delta = input

        # Compute velocity along path
        s_dot = 1 / (1 - (e_y * kappa)) * v * np.cos(e_psi)

        # Compute yaw angle rate of change from the actual(lagged) steering
        # state, not the instantaneously commanded input (2026-07-27再実装)
        psi_dot = v / self.length * np.tan(delta_actual)

        return s_dot, psi_dot

    def get_spatial_derivatives(self, state, input, kappa):
        """
        Compute spatial derivatives of all state variables for update.
        :param state: state vector for which to compute derivatives
        :param input: input vector
        :param kappa: curvature of corresponding waypoint
        :return: numpy array with spatial derivatives for all state variables
        """

        # Get state and input variables
        e_y, e_psi, t, delta_actual = state
        v, delta = input

        # Compute temporal derivatives
        s_dot, psi_dot = self.get_temporal_derivatives(state, input, kappa)

        # Compute spatial derivatives
        d_e_y_d_s = v * np.sin(e_psi) / s_dot
        d_e_psi_d_s = psi_dot / s_dot - kappa
        d_t_d_s = 1 / s_dot

        # delta_actualの一次遅れ(連続時間: d(delta_actual)/dt=(actuator_gain*delta-delta_actual)/tau)
        # を空間微分へ変換(2026-07-27再実装、2026-08-03にactuator_gain追加)。
        # tau=0(無効)はゼロ除算を避けepsを使う。
        tau = max(self.actuator_lag_tau_s, self.eps)
        d_delta_actual_d_s = (self.actuator_gain * delta - delta_actual) / (tau * s_dot)

        return np.array([d_e_y_d_s, d_e_psi_d_s, d_t_d_s, d_delta_actual_d_s])

    def linearize(self, v_ref, kappa_ref, delta_s):
        """
        Linearize the system equations around provided reference values.
        :param v_ref: velocity reference around which to linearize
        :param kappa_ref: kappa of waypoint around which to linearize
        :param delta_s: distance between current waypoint and next waypoint

        2026-07-27再実装(201節続報、AXIS06のdelta_actual状態拡張、198節撤回分の
        再挑戦): e_psiの更新を駆動する項(旧: b_2[1]=delta_s、指令入力から瞬時に反映)
        を、delta_actual_{k+1}(今回ステップで更新された後の実舵角)経由の寄与へ置き換
        える。delta_actual_{k+1}=(1-alpha)*delta_actual_k + alpha*delta_k
        (alpha=clip(delta_s/(tau*v_ref),0,1)、一次遅れの離散化率)を e_psi の式へ
        代入すると、e_psi_{k+1}の delta_actual_k 係数は delta_s*(1-alpha)
        (a_2[3])、delta_k係数は delta_s*alpha (b_2[1])になる。
        (delta_actual_kをそのまま使うと、直近1周期分の値をさらに1周期遅らせてしまう
        バグになるため、必ず更新後の値を代入すること。)
        tau→0(alpha→1、v_ref>0)では a_2[3]→0・b_2[1]→delta_s となり、旧3状態
        モデルのb_2=[0,delta_s]と数学的に完全一致する(下位互換、テストで検証)。
        v_ref==0(速度ゼロ)ではalphaを定義できないため0扱い(delta_actual凍結、
        t状態のv_ref==0特別扱いと同じ思想)とする。
         """

        ###################
        # System Matrices #
        ###################

        tau = max(self.actuator_lag_tau_s, self.eps)

        # Handle v_ref == 0 case
        if v_ref == 0:
            alpha = 0.0
        else:
            alpha = min(1.0, max(0.0, delta_s / (tau * v_ref)))

        # Construct Jacobian Matrix
        a_1 = np.array([1, delta_s, 0, 0])
        a_2 = np.array([-kappa_ref ** 2 * delta_s, 1, 0, delta_s * (1 - alpha)])

        b_1 = np.array([0, 0])
        # delta_actual_{k+1} = (1-alpha)*delta_actual_k + alpha*actuator_gain*delta_k
        # (2026-08-03、actuator_gain追加)。e_psi行(b_2)もdelta_actual_{k+1}経由で
        # 同じゲインを受け取る。a_2[3](delta_actual_kの係数)は前ステップ状態の持続
        # なのでactuator_gainの影響を受けない。
        b_2 = np.array([0, delta_s * alpha * self.actuator_gain])

        if v_ref == 0:
            a_3 = np.array([0, 0, 1, 0])
            b_3 = np.array([0, 0])
            a_4 = np.array([0, 0, 0, 1])
            b_4 = np.array([0, 0])
            f = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            a_3 = np.array([-kappa_ref / v_ref * delta_s, 0, 1, 0])
            b_3 = np.array([-1 / (v_ref ** 2) * delta_s, 0])
            a_4 = np.array([0, 0, 0, 1 - alpha])
            b_4 = np.array([0, alpha * self.actuator_gain])
            f = np.array([0.0, 0.0, 1 / v_ref * delta_s, 0.0])

        A = np.stack((a_1, a_2, a_3, a_4), axis=0)
        B = np.stack((b_1, b_2, b_3, b_4), axis=0)

        return f, A, B
