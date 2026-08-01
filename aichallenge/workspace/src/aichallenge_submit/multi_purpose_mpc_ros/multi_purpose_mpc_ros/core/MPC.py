"""空間バイシクルモデル用 MPC (OSQP)。

QP構造(変数 z = [x_0..x_N | u_0..u_{N-1}], nx=self.model.n_states
  (通常3=e_y,e_psi,t。2026-07-27再実装のdelta_actual状態拡張使用時は4)、nu=2(v,κ)):

    最小化  (1/2) zᵀPz + qᵀz
        P = blockdiag(Q×N, QN, R×N) + 2λ·SᵀS      (λ=r_drate, S=隣接κ差分行列)
    制約
        [ Aeq ]        力学:  kron(I,-I) + A_lin(下副対角ブロック) | B_lin
        [  I  ]  z ∈   box :  x0固定 / コリドー[lb,ub] / 入力[umin, umax_dyn]
        [  S  ]        レート: |Δκ_i| ≤ rate × Δt_i   (P3: Δt=Δs/v_ref, cap 0.15s)

実行経路(Stage1, 2026-07-05):
  fast   : 構造凍結テンプレート + データ配列書込 + osqp.update()。
           適用条件 = N==self.N かつ 循環経路(レース周回)。毎周期の疎行列生成を全廃。
  legacy : 毎周期フル構築 + setup()。ピット(可変N/非循環)・フォールバック・シャドー検証用。
両経路は同一の _stage_data() の出力から組み立てるため数値的に等価
(起動直後 shadow_cycles 周期はレガシーと突き合わせ、最大差をログで機械確認)。

設計ノート:
- A の「値」はLTV線形化のため毎周期変わるが「疎パターン」は固定 → タグ法でdataインデックス
  を一度だけ特定し、以後は値の書込のみ(dense→csc変換の値ゼロ落ち罠を構造的非ゼロで回避)。
- コリドー/マージン緩和リトライは l,u,q のみ変化 → reuse機構で力学の再計算・Ax再書込をしない。
- P はコスト変更イベント(update_Q等)時のみ再計算し update(Px=)。パターンは diag∪SᵀS で不変。
- 既知バグ修正: 旧実装はリトライ毎に wp_id_offset を再加算し参照が最大+5wpずれていた。
"""
import time as _time
from typing import Optional, Tuple

import numpy as np
import osqp
from scipy import sparse

PREDICTION = '#BA4A00'   # 予測軌道の描画色(オフラインシム用)

_TAG0 = 1.0e12           # テンプレートのタグ値の基点(実データと衝突しない巨大値)

# Stage1.9(2026-07-08): OSQPの正式ステータスで解の可否を判定する(solved/solved_inaccurateのみ可)。
#   定数APIが無い旧バージョンへのフォールバックとして {1: solved, 2: solved_inaccurate} を既定値に。
try:
    _OSQP_OK = {osqp.constant('OSQP_SOLVED'), osqp.constant('OSQP_SOLVED_INACCURATE')}
except Exception:
    _OSQP_OK = {1, 2}


def _resize_to_length(arr, n):
    """2026-08-02追加(262節続報、Part A: MPC.py初期化クラッシュ防御)。長さが`n`と
    異なる配列を安全側に整形する: 長すぎれば先頭n件を使用(コリドーは先頭=直近の
    waypointほど重要なため)、短すぎれば末尾値で埋める(=直近既知の安全な幅を
    延長するだけで、範囲外を「無限に安全」等の危険な値で埋めない)。空配列や
    n<=0はそのまま返す(呼び出し側でxmin_dyn[0]等の別経路が処理する特殊ケース)。"""
    arr = np.asarray(arr, dtype=float)
    if n <= 0 or len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    if len(arr) == 0:
        # 手がかりが全く無い場合のみ、_corridor()の既存の「infeasible」慣例
        # (ub<lbをub=lb=0.0にする、上記_corridor参照)に倣い0.0で埋める。
        return np.zeros(n)
    pad = np.full(n - len(arr), arr[-1])
    return np.concatenate([arr, pad])


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, max_steering_rate, wp_id_offset, use_obstacle_avoidance,
                 use_path_constraints_topic, use_max_kappa_pred=True,
                 r_drate=0.0, use_osqp_update=True, shadow_cycles=50):
        """
        :param model: 空間バイシクルモデル
        :param N: 予測ホライズン
        :param Q/R/QN: コスト重み(対角 sparse)
        :param ay_max: 曲率ベース速度上限に使う横加速度
        :param max_steering_rate: κレート上限 [1/s 相当 raw]
        :param wp_id_offset: 制御遅れ補償の参照先読み [wp]
        :param r_drate: ホライズン内 Δκ² ソフトペナルティ λ (0=無効)
        :param use_osqp_update: fast経路(テンプレート+update)を使うか
        :param shadow_cycles: 起動直後にfast/legacyを突き合わせる周期数(0=無効)
        """
        self.model = model
        self.N = N
        self.Q = Q
        self.R = R
        self.QN = QN
        self.nx = self.model.n_states
        self.nu = 2
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints
        self.ay_max = ay_max
        self.wp_id_offset = wp_id_offset
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic
        self.use_max_kappa_pred = use_max_kappa_pred

        # --- 操舵系 ---
        self.max_steering_rate = max_steering_rate
        self.previous_steering = 0.0     # 出力側レートクリップのアンカー(δ)
        # r_drate: ホライズン内の隣接κ差のみ罰する(前回κアンカーは位相遅れ→蛇行のため不採用)
        self.r_drate = float(r_drate)

        # --- 追い越し用 横目標(コントローラが毎周期設定) ---
        self.lateral_blend = 0.0         # 0=コリドー中心 / 1=lateral_target
        self.lateral_target = 0.0        # 参照ラインからの目標横位置(+左/-右)
        self.lateral_psi_bias = 0.0      # e_psi参照バイアス(開き側へ操舵させる)
        self.lateral_funnel_steps = 0    # コリドーを現在e_yから滑らかに合流(0=無効)
        self.safety_margin_override = None  # 追い越し中のマージン縮小(None=通常)

        # --- 診断(コリドー実効値) ---
        self.dbg_corr_lb0 = float('nan')
        self.dbg_corr_ub0 = float('nan')
        self.dbg_corr_xr0 = float('nan')
        self.dbg_corr_wmin = float('nan')
        self.dbg_corr_src = -1.0
        # コリドー配列全体(N点、壁+占有格子込み、約N*resolution[m]先読み)。
        # dbg_corr_ub0/lb0(先頭要素のみ)と同じub/lbから複製するだけで新規計算は無し。
        self.dbg_corr_ub_arr = None
        self.dbg_corr_lb_arr = None

        # --- ソルバ状態 ---
        self.current_prediction = None
        self.infeasibility_counter = 0
        self.last_solved_wp_id = 0
        self.current_control = np.zeros(self.nu * self.N)
        self.optimizer = osqp.OSQP()     # legacy経路用(毎周期作り直し)

        # --- fast経路(Stage1) ---
        self.use_osqp_update = bool(use_osqp_update)
        self._fast = None                # テンプレート・インデックスマップ・OSQPインスタンス
        self._cost_dirty = True          # Q/R/QN/r_drate変更 → 次周期にPx更新
        self._shadow_left = int(shadow_cycles) if self.use_osqp_update else 0
        self._shadow_worst = 0.0

        # レート系の定数キャッシュ(legacy経路用): N -> {A_ineq, S, StS2}
        self._rate_cache = {}
        # 2026-07-25追加(178節続報): 「mpc」区間(平均14-20ms、処理落ちの最大要因)が
        #   QP自体の大きさ(障害物数)に相関しないと判明したため、Python側の行列組み立て
        #   (_init_problem、以下"setup")とOSQPソルバー本体(.solve()、以下"solve")の
        #   どちらが支配的かを切り分ける計装。リトライ(最大2回の再solve)分も合算する。
        self.last_setup_time = 0.0
        self.last_solve_time = 0.0
        self.last_retry_count = 0
        # 179節続報: setup内訳の切り分け(線形化ループ vs コリドー光線走査)
        self.last_linearize_time = 0.0
        self.last_corridor_time = 0.0
        # P3のセグメント長フォールバック(経路が変わった時のみ再計算)
        self._seg_fallback = 0.6
        self._seg_fb_n = None

        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N, self.model.safety_margin)

    # ------------------------------------------------------------------
    # 外部からのパラメータ更新(コストはイベント扱い=fast経路のPx更新トリガ)
    # ------------------------------------------------------------------
    def update_v_max(self, v_max: float):
        self.input_constraints['umax'][0] = v_max

    def update_ay_max(self, ay_max: float):
        self.ay_max = ay_max

    def update_wp_id_offset(self, wp_id_offset: int):
        self.wp_id_offset = wp_id_offset

    def update_Q(self, Q):
        self.Q = Q
        self._cost_dirty = True

    def update_R(self, R):
        self.R = R
        self._cost_dirty = True

    def update_QN(self, QN):
        self.QN = QN
        self._cost_dirty = True

    def reset_solver_state(self):
        """内部ウォームスタート状態のリセット(Stage R: D復帰1歩目用。setup不要のμs級)。"""
        if self._fast is not None:
            n = self._fast['nvar']
            m = self._fast['ncon']
            try:
                self._fast['prob'].warm_start(x=np.zeros(n), y=np.zeros(m))
            except Exception:
                self._fast = None   # 失敗時はテンプレート再構築(=フルsetup)へフォールバック

    # ------------------------------------------------------------------
    # ① データ計算: QPの「中身」を全て数値配列として揃える(経路非依存の共通部)
    # ------------------------------------------------------------------
    def _rate_matrices(self, N):
        """κ差分行列S・legacy用A_ineq・2λSᵀS(全て定数、N別キャッシュ)。"""
        rc = self._rate_cache.get(N)
        if rc is None:
            nvar = self.nx * (N + 1) + self.nu * N
            m = np.zeros((N - 1, nvar))
            for i in range(N - 1):
                m[i, self.nx * (N + 1) + self.nu * i + 1] = -1.0
                m[i, self.nx * (N + 1) + self.nu * (i + 1) + 1] = 1.0
            S = sparse.csc_matrix(m)
            rc = {
                'S': S,
                'A_ineq': sparse.vstack([sparse.eye(nvar), S], format='csc'),
                'StS2': ((2.0 * self.r_drate) * (S.T @ S)).tocsc()
                        if self.r_drate > 0.0 else None,
            }
            self._rate_cache[N] = rc
        return rc

    def _rate_bounds(self, N):
        """P3: |Δκ_i| ≤ rate × Δt_i, Δt=Δs/v_ref(前方セグメント, cap 0.15s)。
        非循環経路は末尾クランプ / 周回継ぎ目(segment_lengths[0]=0)は代表間隔で代替。"""
        rp = self.model.reference_path
        nwp = rp.n_waypoints
        segl = rp.segment_lengths
        if self._seg_fb_n != nwp:
            pos = [s for s in segl if s > 1e-6]
            self._seg_fallback = float(np.median(pos)) if pos else 0.6
            self._seg_fb_n = nwp
        wp0 = self.model.wp_id
        circ = rp.circular
        dt = np.empty(N - 1)
        for i in range(N - 1):
            w = (wp0 + i) % nwp if circ else min(wp0 + i, nwp - 1)
            wn = (w + 1) % nwp if circ else min(w + 1, nwp - 1)
            ds = float(segl[wn])
            if ds <= 1e-6:
                ds = self._seg_fallback
            vr = getattr(rp.waypoints[w], 'v_ref', None)
            dt[i] = ds / max(float(vr) if vr else 0.0, 1.0)
        return self.max_steering_rate * np.clip(dt, 0.0, 0.15)

    def _corridor(self, N, safety_margin):
        """e_yコリドー(lb, ub)。回避ON=動的(占有格子+現姿勢) / OFF=静的テーブル。"""
        rp = self.model.reference_path
        if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
            ub, lb, _ = rp.update_path_constraints(
                self.model.wp_id + 1,
                [self.model.temporal_state.x, self.model.temporal_state.y,
                 self.model.temporal_state.psi],
                N, self.model.length, self.model.width, safety_margin)
        else:
            ref_wp_id = (self.model.wp_id + 1) % len(rp.path_constraints[0])
            ub = rp.path_constraints[0][ref_wp_id]
            lb = rp.path_constraints[1][ref_wp_id]
            rp.border_cells.current_wp_id = ref_wp_id
            if self.model.safety_margin != safety_margin:
                diff = safety_margin - self.model.safety_margin
                ub = ub - diff
                lb = lb + diff
                bad = ub < lb
                ub[bad] = 0.0
                lb[bad] = 0.0

        # funnel: 現在e_yを含む幅から元コリドーへ滑らかに合流(即・端要求→infeasibleの防止)
        if self.lateral_funnel_steps > 0 and len(ub) > 0:
            e_y0 = self.model.spatial_state.e_y
            w = np.clip(np.arange(len(ub)) / float(self.lateral_funnel_steps), 0.0, 1.0)
            ub = (1.0 - w) * np.maximum(ub, e_y0) + w * ub
            lb = (1.0 - w) * np.minimum(lb, e_y0) + w * lb
        return np.asarray(ub, dtype=float), np.asarray(lb, dtype=float)

    def _stage_data(self, N, safety_margin, reuse: Optional[dict] = None) -> dict:
        """QPを組むための全数値を計算して返す。
        reuse: マージン緩和リトライ用。力学(A/B/uq/ur/レート境界)は前回値を再利用し、
               コリドー依存部(lb/ub/xr/境界ベクトル)だけを再計算する。
               ※旧実装はリトライ毎に wp_id_offset を再加算するバグがあった(修正済)。"""
        nx, nu = self.nx, self.nu
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']

        if reuse is not None:
            # Stage1.7 R1(2026-07-07): マージン緩和リトライ経路。
            #   コリドー再計算(実測4ms×5回=リトライ周期28ms→制御欠落の主犯①)を廃し、
            #   初回コリドーを「マージン差分だけ算術的に広げる」(緩和=広げる方向のみ=安全側)。
            #   静的コリドー(path_constraints_topic)と同じ手法。力学・q系も再利用し l,u,q のみ更新。
            d = reuse
            diff = d['margin0'] - safety_margin
            ub = d['ub0'] + diff
            lb = d['lb0'] - diff
        else:
            d = {'N': N}
            # 制御遅れ補償(get_control毎に1回だけ適用)
            self.model.wp_id += self.wp_id_offset

            kappa_pred = np.tan(np.append(
                np.array(self.current_control[3::nu]),
                self.current_control[-1])) / self.model.length

            _t0 = _time.perf_counter()
            A_blk = np.empty((N, nx, nx))
            B_blk = np.empty((N, nx, nu))
            uq = np.empty(N * nx)
            ur = np.empty(N * nu)
            umax_dyn = np.kron(np.ones(N), umax)
            for n in range(N):
                wp = self.model.reference_path.get_waypoint(self.model.wp_id + n)
                wp_next = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
                delta_s = wp_next - wp
                kappa_ref = wp.kappa
                v_ref = np.clip(wp.v_ref, umin[0], umax[0])
                f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
                A_blk[n] = A_lin
                B_blk[n] = B_lin
                ur[n * nu:(n + 1) * nu] = [v_ref, kappa_ref]
                uq[n * nx:(n + 1) * nx] = B_lin.dot([v_ref, kappa_ref]) - f
                # 曲率(予測κ)ベースの速度上限
                if self.use_max_kappa_pred:
                    kp = np.max(np.abs(kappa_pred[n:]))
                else:
                    kp = np.abs(kappa_pred[n])
                umax_dyn[nu * n] = min(np.sqrt(self.ay_max / (kp + 1e-12)),
                                       umax_dyn[nu * n])
            d['A_blk'] = A_blk
            d['B_blk'] = B_blk
            d['uq'] = uq
            d['ur'] = ur
            d['umax_dyn'] = umax_dyn
            d['rate_hi'] = self._rate_bounds(N)
            d['x0'] = np.array(self.model.spatial_state[:])
            # 2026-07-25追加(179節続報): 「mpc_setup」がsolveより支配的と判明したため、
            #   この関数内の2大ブロック(①線形化ループ=物理モデルの逐次線形化、
            #   ②コリドー計算=占有格子への光線走査)のどちらが重いかを切り分ける。
            self.last_linearize_time += _time.perf_counter() - _t0

            # --- コリドー(初回のみ実計算。リトライは上の算術緩和で済ませる) ---
            _t0 = _time.perf_counter()
            ub, lb = self._corridor(N, safety_margin)
            self.last_corridor_time += _time.perf_counter() - _t0
            d['ub0'] = ub
            d['lb0'] = lb
            d['margin0'] = safety_margin

        # 2026-08-02追加(262節続報、Part A: MPC.py初期化クラッシュ防御): ごく稀に
        #   (実測1/10回程度、非循環経路[ピット等]終端付近で発生)`_corridor()`が
        #   返すlb/ubの長さがNと一致しないことがあり、直後の`xmin_dyn[nx::nx]=lb`で
        #   ValueError: could not broadcast input array (例: shape(20,)→shape(1,))と
        #   なりノードごとクラッシュしていた。原因は本関数冒頭の
        #   `self.model.wp_id += self.wp_id_offset`が、呼び出し元`get_control()`で
        #   既に確定させたN(旧wp_id基準)より後に効くため、非循環経路の終端付近では
        #   wp_id+N が n_waypoints を超えうること(高確度の仮説、コード解析による特定。
        #   `update_path_constraints`内部の分岐まで完全にはトレースし切れていないため
        #   「仮説」と明記する)。起動シーケンス(wp_id_offset加算タイミング)自体への
        #   変更は本ガードのスコープ外とし、ここでは長さ不一致を検出した場合のみ
        #   安全側に整形して大惨事(クラッシュ=数秒間の無制御)を防ぐ。長さ一致時
        #   (通常の全周期)はこの分岐に入らず、既存の数値・挙動に一切影響しない。
        if len(lb) != N or len(ub) != N:
            print(f'[MPC-GUARD] corridor length mismatch: len(lb)={len(lb)} '
                  f'len(ub)={len(ub)} N={N} wp_id={self.model.wp_id} -> '
                  f'clamping to avoid crash', flush=True)
            lb = _resize_to_length(lb, N)
            ub = _resize_to_length(ub, N)

        xmin_dyn = np.kron(np.ones(N + 1), self.state_constraints['xmin'])
        xmax_dyn = np.kron(np.ones(N + 1), self.state_constraints['xmax'])
        xmin_dyn[0] = xmax_dyn[0] = self.model.spatial_state.e_y
        xmin_dyn[nx::nx] = lb
        xmax_dyn[nx::nx] = ub

        # e_y参照: コリドー中心 or 追い越しオフセット(コリドーへクランプ)
        xr = np.zeros(nx * (N + 1))
        center = (lb + ub) / 2.0
        if self.lateral_blend > 0.0:
            tgt = (1.0 - self.lateral_blend) * center + self.lateral_blend * self.lateral_target
            xr[nx::nx] = np.clip(tgt, lb, ub)
        else:
            xr[nx::nx] = center
        if self.lateral_psi_bias != 0.0:
            xr[nx + 1::nx] = self.lateral_psi_bias

        d['xmin_dyn'] = xmin_dyn
        d['xmax_dyn'] = xmax_dyn
        d['xr'] = xr

        # 診断
        self.dbg_corr_src = 1.0 if (self.use_obstacle_avoidance
                                    and not self.use_path_constraints_topic) else 0.0
        self.dbg_corr_lb0 = float(lb[0]) if len(lb) else float('nan')
        self.dbg_corr_ub0 = float(ub[0]) if len(ub) else float('nan')
        self.dbg_corr_ub_arr = ub.copy()
        self.dbg_corr_lb_arr = lb.copy()
        self.dbg_corr_xr0 = float(xr[nx])
        w = ub - lb
        self.dbg_corr_wmin = float(np.min(w)) if len(w) else float('nan')
        return d

    # ------------------------------------------------------------------
    # ② ベクトル組み立て(q, l, u は両経路で共通の式)
    # ------------------------------------------------------------------
    def _vectors(self, N, d):
        nx, nu = self.nx, self.nu
        umin = self.input_constraints['umin']
        leq = np.hstack([-d['x0'], d['uq']])
        l = np.hstack([leq,
                       d['xmin_dyn'], np.kron(np.ones(N), umin),
                       -d['rate_hi']])
        u = np.hstack([leq,
                       d['xmax_dyn'], d['umax_dyn'],
                       d['rate_hi']])
        q = np.hstack([
            -np.tile(np.diag(self.Q.toarray()), N) * d['xr'][:-nx],
            -self.QN.dot(d['xr'][-nx:]),
            -np.tile(np.diag(self.R.toarray()), N) * d['ur'],
        ])
        return q, l, u

    def _cost_diag(self, N):
        """P対角(blockdiag(Q×N, QN, R×N))を1本のベクトルで。"""
        return np.hstack([
            np.tile(np.diag(self.Q.toarray()), N),
            np.diag(self.QN.toarray()),
            np.tile(np.diag(self.R.toarray()), N),
        ])

    # ------------------------------------------------------------------
    # ③-a legacy経路: 毎周期フル構築 + setup()
    # ------------------------------------------------------------------
    def _assemble_legacy(self, N, d):
        nx, nu = self.nx, self.nu
        nx_N = nx * (N + 1)
        nu_N = nu * N
        A = np.zeros((nx_N, nx_N))
        B = np.zeros((nx_N, nu_N))
        for n in range(N):
            A[(n + 1) * nx:(n + 2) * nx, n * nx:(n + 1) * nx] = d['A_blk'][n]
            B[(n + 1) * nx:(n + 2) * nx, n * nu:(n + 1) * nu] = d['B_blk'][n]
        Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx)) + sparse.csc_matrix(A)
        Aeq = sparse.hstack([Ax, sparse.csc_matrix(B)])
        rc = self._rate_matrices(N)
        A_full = sparse.vstack([Aeq, rc['A_ineq']], format='csc')

        P = sparse.diags(self._cost_diag(N)).tocsc()
        if rc['StS2'] is not None:
            P = (P + rc['StS2']).tocsc()
        q, l, u = self._vectors(N, d)
        return P, q, A_full, l, u

    def _setup_legacy(self, N, d):
        P, q, A_full, l, u = self._assemble_legacy(N, d)
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A_full, l=l, u=u, verbose=False)
        return self.optimizer

    # ------------------------------------------------------------------
    # ③-b fast経路: 構造凍結テンプレート + update()
    # ------------------------------------------------------------------
    def _build_fast(self, N, d):
        """タグ法でAのdataインデックスマップを構築し、OSQPを一度だけsetupする。
        A/Bブロックの全成分を構造的非ゼロとしてパターンに焼き込む(値ゼロ落ち罠の回避)。"""
        nx, nu = self.nx, self.nu
        nx_N = nx * (N + 1)
        nu_N = nu * N
        n_dyn = N * (nx * nx + nx * nu)          # 毎周期書き込むスロット数

        # タグ入り力学ブロックでテンプレートを組む(書込順=タグ順)
        tagged = dict(d)
        tags = _TAG0 + np.arange(1, n_dyn + 1, dtype=float)
        tagged['A_blk'] = tags[:N * nx * nx].reshape(N, nx, nx)
        tagged['B_blk'] = tags[N * nx * nx:].reshape(N, nx, nu)
        _, _, A_t, _, _ = self._assemble_legacy(N, tagged)

        # タグ値 → data位置 のマップ(タグは一意なのでソートで対応付け)
        data = A_t.data
        cand = np.where(data >= _TAG0)[0]
        order = np.argsort(data[cand])           # data[cand]昇順 = タグ順
        idx_dyn = cand[order]
        assert len(idx_dyn) == n_dyn, 'fastテンプレート: 力学スロット数が不一致'

        # 定数部を確定(タグをゼロクリア。-I/box/S行の±1はそのまま残る)
        A_buf = data.copy()
        A_buf[idx_dyn] = 0.0

        # Pパターン(diag ∪ 2λSᵀS)と対角インデックス。
        # 注意: OSQPはsetup時にPを上三角(triu)へ変換して保持するため、update(Px=)の
        #   dataレイアウトを一致させるには最初からtriuパターンで組む必要がある。
        rc = self._rate_matrices(N)
        nvar = nx_N + nu_N
        if rc['StS2'] is not None:
            Pp = sparse.triu(sparse.diags(np.ones(nvar)) + rc['StS2'], format='csc')
            sts_diag = np.asarray(rc['StS2'].diagonal()).ravel()
        else:
            Pp = sparse.diags(np.ones(nvar)).tocsc()
            sts_diag = np.zeros(nvar)
        diag_idx = np.empty(nvar, dtype=int)
        for c in range(nvar):
            rows = Pp.indices[Pp.indptr[c]:Pp.indptr[c + 1]]
            diag_idx[c] = Pp.indptr[c] + int(np.where(rows == c)[0][0])
        P_base = Pp.data.copy()
        P_base[diag_idx] = sts_diag              # 非対角=SᵀS上三角値 / 対角=SᵀS対角のみ

        P_buf = P_base.copy()
        P_buf[diag_idx] = P_base[diag_idx] + self._cost_diag(N)

        q, l, u = self._vectors(N, d)
        A_pat = A_t.copy()
        A_pat.data = A_buf.copy()
        P_pat = Pp.copy()
        P_pat.data = P_buf.copy()

        prob = osqp.OSQP()
        prob.setup(P=P_pat, q=q, A=A_pat, l=l, u=u, verbose=False, warm_start=True)

        self._fast = {
            'N': N, 'nvar': nvar, 'ncon': A_t.shape[0],
            'prob': prob,
            'A_buf': A_buf, 'idx_dyn': idx_dyn,
            'P_base': P_base, 'diag_idx': diag_idx,
        }
        self._cost_dirty = False

    def _apply_fast(self, N, d, dynamics_changed: bool):
        """値の書込 + update()。疎行列オブジェクトの生成は一切行わない。"""
        f = self._fast
        q, l, u = self._vectors(N, d)
        kw = {'q': q, 'l': l, 'u': u}
        if dynamics_changed:
            # タグ順(=idx_dynの並び)は「A_blk全スロット → B_blk全スロット」(各n-major, 行-major)。
            # 書込値も同じ並びで連結すること(並び不一致は等価性テストが検出する)。
            vals = np.concatenate([d['A_blk'].ravel(), d['B_blk'].ravel()])
            f['A_buf'][f['idx_dyn']] = vals
            kw['Ax'] = f['A_buf']
        if self._cost_dirty:
            P_buf = f['P_base'].copy()
            P_buf[f['diag_idx']] = f['P_base'][f['diag_idx']] + self._cost_diag(N)
            kw['Px'] = P_buf
            self._cost_dirty = False
        f['prob'].update(**kw)
        return f['prob']

    # ------------------------------------------------------------------
    # ディスパッチ(+シャドー検証)
    # ------------------------------------------------------------------
    def _init_problem(self, N, safety_margin, reuse: Optional[dict] = None) -> dict:
        d = self._stage_data(N, safety_margin, reuse=reuse)

        fast_ok = (self.use_osqp_update
                   and N == self.N
                   and self.model.reference_path.circular)
        if fast_ok:
            if self._fast is None or self._fast['N'] != N:
                self._build_fast(N, d)
                # 構築直後も必ず実データを書き込む(テンプレートの力学スロットは0のため)
                self._active = self._apply_fast(N, d, dynamics_changed=True)
            else:
                self._active = self._apply_fast(N, d, dynamics_changed=(reuse is None))
            if self._shadow_left > 0:
                self._shadow_check(N, d)
                self._shadow_left -= 1
        else:
            self._active = self._setup_legacy(N, d)
        return d

    def _shadow_check(self, N, d):
        """起動直後のみ: fastが組んだQPとlegacy組立の一致を機械検証(等価性ゲート)。"""
        try:
            P_l, q_l, A_l, l_l, u_l = self._assemble_legacy(N, d)
            f = self._fast
            # パターンが異なる(fastは構造的非ゼロの上位集合)ため密行列で比較する
            A_fast_dense = self._fast_A_dense(N)
            dA = float(np.abs(A_fast_dense - A_l.toarray()).max())
            P_buf = f['P_base'].copy()
            P_buf[f['diag_idx']] = f['P_base'][f['diag_idx']] + self._cost_diag(N)
            P_fast = sparse.csc_matrix(
                (P_buf, *self._fast_P_pattern(N)), shape=P_l.shape)
            # fast側はtriu保持 → legacyもtriuに揃えて比較
            dP = float(np.abs(P_fast.toarray()
                              - sparse.triu(P_l, format='csc').toarray()).max())
            q_f, l_f, u_f = self._vectors(N, d)
            dv = max(float(np.abs(q_f - q_l).max()),
                     float(np.abs(l_f - l_l).max()),
                     float(np.abs(u_f - u_l).max()))
            worst = max(dA, dP, dv)
            self._shadow_worst = max(self._shadow_worst, worst)
            if self._shadow_worst >= 1e-9:
                print(f'[OSQP-SHADOW] 不一致検出(今回max差={worst:.2e} '
                      f'累計max差={self._shadow_worst:.2e}) → legacy経路へフォールバック',
                      flush=True)
                self.use_osqp_update = False
                self._fast = None
        except Exception as e:  # シャドーは検証専用: 失敗しても走行は止めない
            print(f'[OSQP-SHADOW] 検証例外: {e!r} → legacyへフォールバック', flush=True)
            self.use_osqp_update = False
            self._fast = None

    def _fast_A_dense(self, N):
        f = self._fast
        pat = self._fast.get('_A_pattern')
        if pat is None:
            # パターン保持用に一度だけタグ組立を再現(シャドー期間のみ使用)
            nx, nu = self.nx, self.nu
            tagged_d = {'A_blk': np.ones((N, nx, nx)), 'B_blk': np.ones((N, nx, nu)),
                        'uq': np.zeros(N * nx), 'ur': np.zeros(N * nu),
                        'umax_dyn': np.zeros(N * nu), 'rate_hi': np.zeros(N - 1),
                        'x0': np.zeros(nx), 'xmin_dyn': np.zeros(nx * (N + 1)),
                        'xmax_dyn': np.zeros(nx * (N + 1)), 'xr': np.zeros(nx * (N + 1))}
            tags = _TAG0 + np.arange(1, N * (nx * nx + nx * nu) + 1, dtype=float)
            tagged_d['A_blk'] = tags[:N * nx * nx].reshape(N, nx, nx)
            tagged_d['B_blk'] = tags[N * nx * nx:].reshape(N, nx, nu)
            _, _, A_t, _, _ = self._assemble_legacy(N, tagged_d)
            f['_A_pattern'] = (A_t.indices.copy(), A_t.indptr.copy(), A_t.shape)
            pat = f['_A_pattern']
        indices, indptr, shape = pat
        return sparse.csc_matrix((f['A_buf'], indices, indptr), shape=shape).toarray()

    def _fast_P_pattern(self, N):
        rc = self._rate_matrices(N)
        nvar = self.nx * (N + 1) + self.nu * N
        if rc['StS2'] is not None:
            Pp = sparse.triu(sparse.diags(np.ones(nvar)) + rc['StS2'], format='csc')
        else:
            Pp = sparse.diags(np.ones(nvar)).tocsc()
        return Pp.indices, Pp.indptr

    # ------------------------------------------------------------------
    # 制御計算(公開API)
    # ------------------------------------------------------------------
    def get_control(self) -> Tuple[np.ndarray, float]:
        nx, nu = self.nx, self.nu

        self.model.get_current_waypoint()
        N = min(self.N, self.model.reference_path.n_waypoints - self.model.wp_id) \
            if not self.model.reference_path.circular else self.N
        self.model.spatial_state = self.model.t2s(
            reference_state=self.model.temporal_state,
            reference_waypoint=self.model.current_waypoint)

        sm = self.safety_margin_override if self.safety_margin_override is not None \
            else self.model.safety_margin

        self.last_setup_time = 0.0
        self.last_solve_time = 0.0
        self.last_retry_count = 0
        self.last_linearize_time = 0.0
        self.last_corridor_time = 0.0

        _t0 = _time.perf_counter()
        d = self._init_problem(N, sm)
        self.last_setup_time += _time.perf_counter() - _t0

        try:
            _t0 = _time.perf_counter()
            dec = self._active.solve()
            self.last_solve_time += _time.perf_counter() - _t0

            # Stage1.9(2026-07-08): リトライ発火をOSQPの正式ステータスで判定する。
            #   旧実装は `not np.all(kappa列)` = 「ホライズン内にexactly 0.0のκが1つでもあれば
            #   失敗とみなす」という誤ったヒューリスティックだった。直線区間ではκ=0.0が正当な
            #   最適解であり(実測で status='solved' かつ x=[0,0] を確認)、そのたびに無駄な
            #   コリドー再計算付きリトライを起動していた(予選の retry 発生回数の主要因と推定)。
            #   さらに本物のprimal infeasible時はdec.xに桁違いの値(実測 ~2.1e9)が返るが、これは
            #   非ゼロのため旧判定はすり抜けさせ、無検証のまま出力していた安全上の抜けもあった。
            #   新判定は status_val を直接見るため両方を同時に解消する。
            solved = dec.info.status_val in _OSQP_OK
            if not solved:
                # マージン緩和リトライ(Stage1.7 R1): コリドーは算術緩和(reuse)で再計算なし。
                #   段数5→2(0.4×sm→0): 中間段での成功はログ上ほぼ皆無で、最終段0が実質の
                #   フォールバックだった。1リトライ周期 実測28ms→数msへ(制御欠落の主犯①対策)。
                #   最終段の前に warm_start をクリアし、悪い初期点を引きずったままの反復増加
                #   (交通が濃い局面での retry 再肥大の一因)を防ぐ。
                for i, relaxed in enumerate((sm * 0.4, 0.0)):
                    self.last_retry_count += 1
                    if i == 1:
                        self.reset_solver_state()
                    _t0 = _time.perf_counter()
                    self._init_problem(N, relaxed, reuse=d)
                    self.last_setup_time += _time.perf_counter() - _t0
                    _t0 = _time.perf_counter()
                    dec = self._active.solve()
                    self.last_solve_time += _time.perf_counter() - _t0
                    solved = dec.info.status_val in _OSQP_OK
                    if solved:
                        if self.infeasibility_counter == 0 and self.last_solved_wp_id != self.model.wp_id:
                            print(f'Relaxed safety margin to {relaxed:.2f} '
                                  f'to solve the problem')
                        break
            if not solved:
                # 全リトライ失敗: dec.xは信頼できない値(garbage)なので使わず、
                # except節と同じ「前回解の先送り」フォールバックに合流させる。
                raise ValueError('MPC solve failed after relax retries')

            control_signals = np.array(dec.x[-N * nu:])

            # κ→δ変換と出力側レートクリップ(アクチュエータ実時間基準)
            control_signals[1::2] = np.arctan(control_signals[1::2] * self.model.length)
            v = control_signals[0]
            delta = control_signals[1]
            max_delta_change = self.max_steering_rate * self.model.Ts
            delta = np.clip(delta,
                            self.previous_steering - max_delta_change,
                            self.previous_steering + max_delta_change)
            self.previous_steering = delta

            self.current_control = control_signals
            x = np.reshape(dec.x[:(N + 1) * nx], (N + 1, nx))
            self.current_prediction = self.update_prediction(x, N)

            u = np.array([v, delta])
            max_delta = np.max(np.abs(control_signals[1:len(control_signals) // 3 * 2:2]))

            if self.infeasibility_counter > (N - 1):
                print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
            self.infeasibility_counter = 0
            self.last_solved_wp_id = self.model.wp_id

        except (TypeError, ValueError):
            # 解なし: 前回解の先を順送りで使う(旧実装の `except TypeError or ValueError`
            # は ValueError を捕捉できていなかったバグを修正)
            idx = nu * (self.infeasibility_counter + 1)
            if idx + 2 < len(self.current_control):
                u = np.array(self.current_control[idx:idx + 2])
                max_delta = np.abs(u[1])
            else:
                u = np.array([0.0, 0.0])
                max_delta = 0.0
            self.infeasibility_counter += 1
            # 2026-07-22追加(issue④②、previous_steeringの陳腐化対策): previous_steering
            #   はtry節(628行目)でのみ更新されており、infeasibleが続く間はQP再解決成功前の
            #   古い値のまま凍結されていた。再解決成功時のレートクランプ(615-617行目)は
            #   このprevious_steeringを基準にするため、凍結中に実際に出力していた操舵
            #   (このexcept節のu[1]、先送りされた過去解 or 0.0)とは無関係な古い基準で
            #   クランプされ、再解決直後に実態と乖離した補正がかかりうる。try節と同じ
            #   意味(「直近に実際に出力した操舵」)になるよう、この節でも必ず更新する。
            self.previous_steering = float(u[1])

        if self.infeasibility_counter > (N - 1) and self.infeasibility_counter % 100 == 0:
            print('No control signal computed!')

        return u, max_delta

    # ------------------------------------------------------------------
    # 可視化ユーティリティ
    # ------------------------------------------------------------------
    def update_prediction(self, spatial_state_prediction, N):
        """予測状態列を(x, y)座標列へ変換(可視化用)。"""
        x_pred, y_pred = [], []
        for n in range(2, N):
            wp = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            ts = self.model.s2t(wp, spatial_state_prediction[n, :])
            x_pred.append(ts.x)
            y_pred.append(ts.y)
        return x_pred, y_pred

    def show_prediction(self, ax):
        """予測軌道を描画(オフラインシム用。ROSノードではmatplotlibを読み込まない)。"""
        if self.current_prediction is not None:
            ax.plot(self.current_prediction[0], self.current_prediction[1], c=PREDICTION)
