#!/usr/bin/env python3
"""
analyze_bag.py — レース走行ログ(rosbag/mcap) 統合分析エントリ

このスクリプトを実行すると、登録された全アナライザが俯瞰的に走る。
システム健全性（ノイズ・指令と実現の乖離・操舵遅れ）は「コアアナライザ」として
--only/--category で絞った場合でも毎回必ず実行される。単独走行でも複数台走行でも、
これらの基礎監視は常に行われる。

個別の分析ロジックと bag 読み込みは analyze_core.py（ライブラリ）にある。
新しい観点は @analyzer を付けた関数を1つ足すだけで増やせる（パッチ不要）。

使い方:
    python3 analyze_bag.py <bag> [ref_csv]               # 全アナライザ（俯瞰）
    python3 analyze_bag.py <bag> [ref_csv] --only hotspot_zoom
    python3 analyze_bag.py <bag> [ref_csv] --category 乖離
    python3 analyze_bag.py <bag> [ref_csv] --core-only   # システム健全性のみ
    python3 analyze_bag.py --list

新しいアナライザの足し方:
    @analyzer("key", "表示名", "カテゴリ", topics=["cmd", "odom"], core=False)
    def _a_xxx(ctx):
        # ctx.msgs[T["cmd"]], ctx.ey_series(), ctx.odom_arrays() などを使う
        ...
"""
import sys
import math
import argparse
from pathlib import Path
import numpy as np

# 同ディレクトリの analyze_core.py（旧 analyze_bag.py）を下位ライブラリとして使う
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_core as ab  # noqa: E402

# ---- トピック短縮名 → 正式名 ----
T = {
    "cmd":          "/control/command/control_cmd",
    "cmd_raw":      "/control/command/control_cmd_raw",
    "odom":         "/localization/kinematic_state",
    "steer":        "/vehicle/status/steering_status",
    "vel":          "/vehicle/status/velocity_status",
    "actu_cmd":     "/control/command/actuation_cmd",
    "actu_st":      "/vehicle/status/actuation_status",
    "imu":          "/sensing/imu/imu_raw",
    "cond":         "/aichallenge/pitstop/condition",
    "collision_gt": "/awsim/ground_truth/on_collision",
    "v2x":          "/v2x/vehicle_positions/markers",
    "awsim":        "/awsim/status",
}

# カテゴリの表示順（俯瞰時のグルーピング順）
CATEGORY_ORDER = ["ノイズ・異常", "指令と実現の乖離", "衝突・接触", "複数台・対戦"]

# ---- アナライザ登録機構 ----
REGISTRY = []


def analyzer(key, title, category, topics, core=False, optional=None):
    """アナライザ登録デコレータ。
    core=True のものは --only/--category で絞っても毎回必ず実行される。
    optional は「あれば使うが、無くても実行する」トピック（例: imu）。"""
    def deco(fn):
        REGISTRY.append(dict(key=key, title=title, category=category,
                             topics=list(topics), optional=list(optional or []),
                             fn=fn, core=core))
        return fn
    return deco


# ---- 共通コンテキスト（重い計算は一度だけ・キャッシュ）----
class Ctx:
    def __init__(self, msgs, recv, ref_csv):
        self.msgs = msgs
        self.recv = recv
        self.ref_csv = ref_csv
        self._ref = None
        self._odom = None
        self._ey = None
        self._refprof = None
        self._refprof_signed = None

    def has(self, *keys):
        return all(self.msgs.get(T[k]) for k in keys)

    def corner_turn_sign(self, s_center, half_window=4.0):
        """指定 s 周辺でコーナーが左(+1)/右(-1)どちらに曲がるかを返す（不明なら0）。
        参照CSVの kappa 符号を使う。kappa>0=左旋回, <0=右旋回（このCSVの規約）。
        steering_tire_angle 規約（+左/-右）と符号が揃う。"""
        prof = self.ref_profile()
        if prof is None:
            return 0
        ss, kk_abs, vv = prof
        # ref_profile は |kappa| を返すので、符号付きを別途読む
        if self._refprof_signed is None:
            try:
                import csv as _csv
                ssig, ksig = [], []
                with open(self.ref_csv) as f:
                    rd = _csv.DictReader(f)
                    cols = {c.strip(): c for c in rd.fieldnames}
                    s_c, k_c = cols.get("s_m"), cols.get("kappa_radpm")
                    if not (s_c and k_c):
                        self._refprof_signed = False
                        return 0
                    for row in rd:
                        ssig.append(float(row[s_c]))
                        ksig.append(float(row[k_c]))
                self._refprof_signed = (np.array(ssig), np.array(ksig))
            except Exception:
                self._refprof_signed = False
                return 0
        if self._refprof_signed is False:
            return 0
        ss2, ksig = self._refprof_signed
        sel = np.abs(ss2 - s_center) <= half_window
        if sel.sum() == 0:
            return 0
        # 最も曲率の大きい点の符号
        idx = np.where(sel)[0]
        j = idx[int(np.argmax(np.abs(ksig[idx])))]
        return 1 if ksig[j] > 0 else -1

    def ref(self):
        if self._ref is None and self.ref_csv:
            od = self.msgs[T["odom"]]
            ox = np.array([m.pose.pose.position.x for m in od])
            oy = np.array([m.pose.pose.position.y for m in od])
            rx, ry, _, _ = ab._load_ref_xy(self.ref_csv, ox, oy)
            seg = np.hypot(np.diff(rx), np.diff(ry))
            s_wp = np.concatenate([[0.0], np.cumsum(seg)])
            track_len = s_wp[-1] + math.hypot(rx[0] - rx[-1], ry[0] - ry[-1])
            self._ref = (rx, ry, s_wp, track_len)
        return self._ref

    def ref_profile(self):
        """参照CSVから s, |kappa|, vx_ref を読む。列が無ければ None。
        入口アンダー診断（理論限界速度・必要操舵角）に使う。"""
        if self._refprof is not None:
            return self._refprof if self._refprof is not False else None
        if not self.ref_csv:
            self._refprof = False
            return None
        try:
            import csv as _csv
            ss, kk, vv = [], [], []
            with open(self.ref_csv) as f:
                rd = _csv.DictReader(f)
                cols = {c.strip(): c for c in rd.fieldnames}
                s_c = cols.get("s_m")
                k_c = cols.get("kappa_radpm")
                v_c = cols.get("vx_mps")
                if not (s_c and k_c):
                    self._refprof = False
                    return None
                for row in rd:
                    ss.append(float(row[s_c]))
                    kk.append(abs(float(row[k_c])))
                    vv.append(float(row[v_c]) if v_c else float("nan"))
            self._refprof = (np.array(ss), np.array(kk), np.array(vv))
            return self._refprof
        except Exception:
            self._refprof = False
            return None

    def corner_geometry(self, s_center, half_window=2.0, ay_max=12.0, L=1.087):
        """指定 s 周辺で最タイト点を探し、(s*, |κ|, R, 必要操舵[deg],
        理論限界速度[km/h], 目標速度vx_ref[km/h]) を返す。"""
        prof = self.ref_profile()
        if prof is None:
            return None
        ss, kk, vv = prof
        sel = np.abs(ss - s_center) <= half_window
        if sel.sum() == 0:
            i = int(np.argmin(np.abs(ss - s_center)))
            sel = np.zeros(len(ss), bool)
            sel[i] = True
        idx = np.where(sel)[0]
        j = idx[int(np.argmax(kk[idx]))]
        k = max(kk[j], 1e-6)
        R = 1.0 / k
        need = math.degrees(math.atan(L * k))
        vlim = math.sqrt(ay_max / k) * 3.6
        vref = vv[j] * 3.6 if not math.isnan(vv[j]) else float("nan")
        return (ss[j], k, R, need, vlim, vref)

    def odom_arrays(self):
        if self._odom is None:
            od = self.msgs[T["odom"]]
            t = np.array(self.recv[T["odom"]], dtype=float) * 1e-9
            x = np.array([m.pose.pose.position.x for m in od])
            y = np.array([m.pose.pose.position.y for m in od])
            v = np.array([m.twist.twist.linear.x for m in od])
            o = np.argsort(t)
            self._odom = (t[o], x[o], y[o], v[o])
        return self._odom

    def ey_series(self):
        if self._ey is None:
            ref = self.ref()
            if ref is None:
                return None
            rx, ry, s_wp, _ = ref
            t, x, y, v = self.odom_arrays()
            n = len(rx)
            ss = np.empty(len(x))
            ey = np.empty(len(x))
            for k in range(len(x)):
                i = int(np.argmin((rx - x[k]) ** 2 + (ry - y[k]) ** 2))
                j = (i + 1) % n
                tx, ty = rx[j] - rx[i], ry[j] - ry[i]
                nrm = math.hypot(tx, ty) + 1e-9
                ey[k] = -(ty / nrm) * (x[k] - rx[i]) + (tx / nrm) * (y[k] - ry[i])
                ss[k] = s_wp[i]
            self._ey = (t, ss, ey, x, y, v)
        return self._ey

    def signal_recv(self, topic_key, getter):
        ms = self.msgs[T[topic_key]]
        t = np.array(self.recv[T[topic_key]], dtype=float) * 1e-9
        val = np.array([getter(m) for m in ms], dtype=float)
        o = np.argsort(t)
        return t[o], val[o]


# ============================================================
# コアアナライザ（システム健全性・毎回必ず実行）
#   ノイズ＝ジッタ / 指令と実現の乖離＝速度・操舵の追従
# ============================================================
@analyzer("jitter", "周期ジッタ", "ノイズ・異常", ["cmd"], core=True)
def _a_jitter(ctx):
    ab.report_jitter(ctx.recv[T["cmd"]])


@analyzer("speed", "速度到達・追従", "指令と実現の乖離", ["cmd", "odom"], core=True)
def _a_speed(ctx):
    ab.report_speed(ctx.msgs[T["cmd"]], ctx.msgs[T["odom"]])


@analyzer("steer_sat", "操舵飽和・追従", "指令と実現の乖離", ["cmd", "steer", "odom"], core=True)
def _a_steer_sat(ctx):
    ab.report_steer_saturation(ctx.msgs[T["cmd"]], ctx.msgs[T["steer"]],
                               ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv)


# ============================================================
# 拡張アナライザ（俯瞰実行／個別指定で動く）
# ============================================================
@analyzer("ey", "e_y追従誤差", "指令と実現の乖離", ["odom"])
def _a_ey(ctx):
    if not ctx.ref_csv:
        print("[e_y追従誤差] ref_csv 未指定でスキップ")
        return
    ab.compute_ey(ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv, plot=None)


@analyzer("hotspot_zoom", "ホットスポット区間ズーム", "指令と実現の乖離",
          ["cmd", "odom", "steer"])
def _a_hotspot_zoom(ctx, window_s=1.5, n_rows=16):
    """e_y 最大点を自動検出し、その通過前後 window_s 秒の
    指令操舵・実操舵・速度・e_y を時系列で並べる。"""
    series = ctx.ey_series()
    if series is None:
        print("[区間ズーム] ref_csv が必要です")
        return
    t, ss, ey, x, y, v = series
    aey = np.abs(ey)
    kmax = int(np.argmax(aey))
    tc0 = t[kmax]
    print(f"[区間ズーム] e_y最大点  s={ss[kmax]:.0f}m  |e_y|={aey[kmax]:.2f}m  "
          f"t={tc0 - t[0]:.1f}s  v={v[kmax] * 3.6:.1f}km/h")

    t_cmd, d_cmd = ctx.signal_recv("cmd", lambda m: math.degrees(m.lateral.steering_tire_angle))
    t_st, d_st = ctx.signal_recv("steer", lambda m: math.degrees(m.steering_tire_angle))

    lo, hi = tc0 - window_s, tc0 + window_s
    grid = np.linspace(lo, hi, n_rows)
    g_cmd = np.interp(grid, t_cmd, d_cmd)
    g_st = np.interp(grid, t_st, d_st)
    g_ey = np.interp(grid, t, ey)
    g_v = np.interp(grid, t, v) * 3.6

    print(f"  通過前後 ±{window_s:.1f}s の時系列（δ=操舵角[deg], 正=左, t=0が最大点）:")
    print("      t[s]   δ指令    δ実   指令-実   e_y[m]  v[km/h]")
    for i in range(n_rows):
        mark = "  <-max" if abs(grid[i] - tc0) <= (hi - lo) / (2 * n_rows) else ""
        print(f"    {grid[i] - tc0:+5.2f}  {g_cmd[i]:+6.1f}  {g_st[i]:+6.1f}  "
              f"{g_cmd[i] - g_st[i]:+6.1f}   {g_ey[i]:+5.2f}  {g_v[i]:5.1f}{mark}")

    def max_rate(tt, vv):
        d = np.diff(vv)
        dt = np.diff(tt)
        ok = dt > 1e-3
        return np.abs(d[ok] / dt[ok]).max() if ok.any() else 0.0

    win_c = (t_cmd >= lo) & (t_cmd <= hi)
    win_s = (t_st >= lo) & (t_st <= hi)
    rate_cmd = max_rate(t_cmd[win_c], d_cmd[win_c]) if win_c.sum() > 2 else 0.0
    rate_st = max_rate(t_st[win_s], d_st[win_s]) if win_s.sum() > 2 else 0.0
    dev_at_max = abs(np.interp(tc0, t_cmd, d_cmd) - np.interp(tc0, t_st, d_st))
    print(f"  [診断] 指令操舵の最大変化率={rate_cmd:.0f}deg/s  実操舵={rate_st:.0f}deg/s")
    print(f"         最大点での指令-実 乖離={dev_at_max:.1f}deg")
    if rate_st < rate_cmd * 0.6 or dev_at_max > 8.0:
        print("         → 実操舵が指令に追従しきれていない。"
              "対策候補: 進入速度を落とす / 先読みを伸ばし操舵をなだらかに / 操舵レート上限の見直し")
    else:
        print("         → 操舵追従は概ね良好。膨らみは操舵以外（進入速度/ライン）が主因の可能性")


@analyzer("collision_gt", "衝突(真値)", "衝突・接触", ["collision_gt", "odom"])
def _a_collision(ctx):
    ab.report_collision_gt(ctx.msgs[T["collision_gt"]], ctx.recv[T["collision_gt"]],
                           ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv)


# config の steer_rate_max(0.35 rad/s) を操舵レート比較の基準にする
STEER_RATE_MAX_DEG = math.degrees(0.35)  # ≈ 20.05 deg/s


@analyzer("collision_trace", "崩壊トレース（衝突原因まで遡及）", "衝突・接触",
          ["cmd", "odom", "steer"], optional=["imu"])
def _a_collision_trace(ctx, ey_trigger=2.4, ey_recover=0.5, max_lookback_s=8.0,
                       max_rows=24, v_move=2.0):
    """『走行中(v≥v_move)に |e_y| が ey_trigger を超えて急拡大した起点』を崩壊として検出し、
    そこから膨らみ始め(|e_y|<ey_recover)まで遡る。壁衝突→90度回転→停止の後処理データを
    起点に拾わないよう、膨らみが始まった瞬間に速度が出ていたものだけを崩壊とみなす。
    各点で指令操舵・実操舵・変化率・速度・e_y を表示し、進入時の操舵方向とコーナーの
    曲がる向きを比較する（直線で逆向きに切っていた等を検出）。"""
    series = ctx.ey_series()
    if series is None:
        print("[崩壊トレース] ref_csv が必要です")
        return
    t, ss, ey, x, y, v = series
    aey = np.abs(ey)
    over = aey >= ey_trigger
    if not over.any():
        print(f"[崩壊トレース] |e_y|≥{ey_trigger}m の崩壊なし（最大{aey.max():.2f}m）")
        return

    t_cmd, d_cmd = ctx.signal_recv("cmd", lambda m: math.degrees(m.lateral.steering_tire_angle))
    t_st, d_st = ctx.signal_recv("steer", lambda m: math.degrees(m.steering_tire_angle))

    # 連続する崩壊領域ごとに、閾値を「最初に超えた瞬間(k_enter)」を崩壊の顕在点とし、
    # そこから膨らみ始め(|e_y|<ey_recover)まで遡る。膨らみ進行中(k0→k_enter)に
    # 走行していた(最高速≥v_move)ものだけを採用＝衝突後の停止データを起点にしない。
    events = []
    i, n = 0, len(over)
    while i < n:
        if over[i]:
            j = i
            while j < n and over[j]:
                j += 1
            k_enter = i
            t_enter = t[k_enter]
            k0 = k_enter
            while k0 > 0 and (t_enter - t[k0]) < max_lookback_s and aey[k0] >= ey_recover:
                k0 -= 1
            v_during = (v[k0:k_enter + 1].max() if k_enter > k0 else v[k_enter]) * 3.6
            if v_during >= v_move * 3.6:
                events.append((k_enter, k0))
            i = j
        else:
            i += 1

    if not events:
        print(f"[崩壊トレース] 走行中(v≥{v_move*3.6:.0f}km/h)の |e_y|≥{ey_trigger}m 崩壊なし")
        print("  （閾値超えはあるが、膨らみ進行中に停止＝衝突後の後処理データのみ）")
        return

    print(f"[崩壊トレース] 走行中の |e_y|≥{ey_trigger}m 崩壊 {len(events)}件。膨らみ始めまで遡及")
    print(f"  操舵レート基準: config steer_rate_max=0.35rad/s = {STEER_RATE_MAX_DEG:.0f}deg/s（'!'=超過）")
    for ev, (k_enter, k0) in enumerate(events, 1):
        t_enter = t[k_enter]
        t_start = t[k0]
        # 進入の直線局面も見えるよう、起点の少し手前から崩壊顕在化の少し後まで表示
        lo, hi = t_start - 0.5, t_enter + 0.5
        rows = int(np.clip((hi - lo) / 0.1, 8, max_rows))
        grid = np.linspace(lo, hi, rows)
        g_cmd = np.interp(grid, t_cmd, d_cmd)
        g_st = np.interp(grid, t_st, d_st)
        g_ey = np.interp(grid, t, ey)
        g_v = np.interp(grid, t, v) * 3.6
        r_cmd = np.gradient(g_cmd, grid)
        r_st = np.gradient(g_st, grid)
        print(f"\n  ── イベント{ev}: 崩壊顕在 s={ss[k_enter]:.0f}m |e_y|={aey[k_enter]:.2f}m "
              f"t={t_enter - t[0]:.1f}s v={v[k_enter] * 3.6:.1f}km/h  "
              f"（膨らみ始めから{t_enter - t_start:.1f}s）──")
        print("      t[s]   δ指令  δ実   δ\u0307指令  δ\u0307実    e_y   v[km/h]")
        for r in range(rows):
            f_c = "!" if abs(r_cmd[r]) > STEER_RATE_MAX_DEG else " "
            f_s = "!" if abs(r_st[r]) > STEER_RATE_MAX_DEG else " "
            mark = " <-顕在" if abs(grid[r] - t_enter) <= (hi - lo) / (2 * rows) else ""
            print(f"    {grid[r] - t_enter:+5.2f} {g_cmd[r]:+6.1f} {g_st[r]:+6.1f} "
                  f"{r_cmd[r]:+5.0f}{f_c} {r_st[r]:+5.0f}{f_s} {g_ey[r]:+5.2f} {g_v[r]:5.1f}{mark}")
        ksp = int(np.argmin(np.abs(t - t_start)))
        v_entry = v[ksp] * 3.6
        cmd_entry = np.interp(t_start, t_cmd, d_cmd)
        st_entry = np.interp(t_start, t_st, d_st)
        print(f"  [起点] 膨らみ始め t={t_start - t[0]:.1f}s: e_y={ey[ksp]:+.2f}m  "
              f"指令δ={cmd_entry:+.1f}deg  実δ={st_entry:+.1f}deg  v={v_entry:.1f}km/h")

        # コーナーの曲がる向き（参照ラインの psi 変化＝旋回方向）を求める
        geo = ctx.corner_geometry(ss[k_enter])
        turn_sign = ctx.corner_turn_sign(ss[k_enter])  # +1=左旋回, -1=右旋回, 0=不明
        if geo is not None:
            s_star, kappa, R, need, vlim, vref = geo
            print(f"  [入口診断] コーナー最タイト s={s_star:.0f}m: "
                  f"|κ|={kappa:.3f}(R={R:.1f}m)  必要操舵={need:.1f}deg  "
                  f"理論限界v={vlim:.1f}km/h" +
                  (f"  目標v_ref={vref:.1f}km/h" if not math.isnan(vref) else ""))
            # 進入操舵の向きが、コーナーの曲がる向きと逆/不足か
            if turn_sign != 0:
                turn_name = "左" if turn_sign > 0 else "右"
                # 操舵符号: +=左, -=右（steering_tire_angle 規約）
                if st_entry * turn_sign < 0:
                    print(f"    → 起点で{turn_name}コーナーなのに逆向き（実δ={st_entry:+.0f}°）に操舵。"
                          f"【直線の蛇行補正が尾を引き、コーナーで切り遅れ】")
                elif abs(st_entry) < need - 1.0:
                    print(f"    → {turn_name}コーナーに同方向だが実δ{abs(st_entry):.0f}<必要{need:.0f}°で不足。"
                          f"【A:操舵不足/切り遅れ】")
                else:
                    print(f"    → 進入操舵は向き・量とも妥当。膨らみは速度/立ち上がり遅れ")
            over_speed = v_entry > vlim
            if over_speed:
                print(f"    → 補足: 進入{v_entry:.0f} > 理論限界{vlim:.0f}km/h（速度超過も寄与）")

        # 崩壊後半に「舵を戻さず切り続けて逆へ」＝オーバー転化が起きていないか
        # 顕在点(k_enter)以降で、実δがピークから戻らずに e_y が反対符号へ向かう兆候
        seg_after = (t >= t_enter) & (t <= t_enter + 1.5)
        if seg_after.sum() >= 3 and turn_sign != 0:
            st_after = np.interp(t[seg_after], t_st, d_st)
            # コーナー方向に振り切ったまま戻していない（同符号で大きいまま）か
            held = np.median(st_after) * turn_sign
            if held > 10.0:  # コーナー方向に10°以上入れたまま
                print(f"    → 崩壊後: 実δが{turn_name}に入れっぱなし（中央{np.median(st_after):+.0f}°、"
                      f"戻し不足）→ 膨らみ後に内側へ切れ込むオーバー転化のリスク")

        # imu があれば U/O 判定
        if ctx.msgs.get(T["imu"]):
            t_im, wz = ctx.signal_recv("imu", lambda m: m.angular_velocity.z)
            t_sr, d_sr = ctx.signal_recv("steer", lambda m: m.steering_tire_angle)
            seg = (t_im >= t_start) & (t_im <= t_enter)
            if seg.sum() >= 3:
                d_seg = np.interp(t_im[seg], t_sr, d_sr)
                v_seg = np.interp(t_im[seg], t, v)
                w_exp = v_seg * np.tan(d_seg) / 1.087
                ok = np.abs(w_exp) > 0.05
                if ok.sum() >= 3:
                    r = np.median(wz[seg][ok] / w_exp[ok])
                    tag = ("アンダー(回頭不足)" if r < 0.85
                           else "オーバー(回頭過剰)" if r > 1.15 else "中立")
                    print(f"  [U/O] 崩壊区間のヨーレート比 中央値={r:.2f} → {tag}")
    print("  → 起点で『コーナーと逆向き/不足の操舵』なら直線蛇行の尾引き・参照ライン、"
          "『戻し不足』ならオーバー転化、『指令δ\u0307が!連発』なら指令の乱高下")


@analyzer("steer_scale", "操舵スケール検査（指令δ vs 実δ）", "指令と実現の乖離",
          ["cmd", "steer", "odom"])
def _a_steer_scale(ctx, sat_band=1.0, v_min=3.0):
    """control_cmd(指令δ) と steering_status(実δ) を同時刻で対応付け、
    線形フィット(実δ≈a*指令δ+b)と頭打ち点を出す。21°上限が『指令何度で頭打ちか』
    『一定スケール係数があるか』を明らかにする。停止区間(v<v_min)は操舵と車両応答が
    物理的に対応しないため除外する（最小限のフィルタ。除外数も表示）。"""
    t_cmd, d_cmd = ctx.signal_recv("cmd", lambda m: math.degrees(m.lateral.steering_tire_angle))
    t_st, d_st = ctx.signal_recv("steer", lambda m: math.degrees(m.steering_tire_angle))
    t_od, _, _, v_od = ctx.odom_arrays()
    # 実操舵時刻に指令と速度を内挿
    d_cmd_at = np.interp(t_st, t_cmd, d_cmd)
    v_at = np.interp(t_st, t_od, v_od)

    n_all = len(d_st)
    moving = v_at >= v_min
    n_excl = n_all - int(moving.sum())
    d_st_m = d_st[moving]
    d_cmd_m = d_cmd_at[moving]
    a_st = np.abs(d_st_m)
    a_cmd = np.abs(d_cmd_m)
    if len(d_st_m) < 20:
        print("[操舵スケール検査] 走行サンプル不足（停止除外後）")
        return
    sat_level = a_st.max()

    print("[操舵スケール検査]（指令δ → 実δ の対応, 停止除外）")
    print(f"  サンプル: 全{n_all} / 使用{len(d_st_m)}（v<{v_min:.0f}m/s の停止 {n_excl} を除外）")
    print(f"  指令δ レンジ: {d_cmd_m.min():+.1f} 〜 {d_cmd_m.max():+.1f} deg")
    print(f"  実δ  レンジ: {d_st_m.min():+.1f} 〜 {d_st_m.max():+.1f} deg（|max|={sat_level:.1f}）")

    nonsat = a_st < (sat_level - sat_band)
    if nonsat.sum() > 10:
        A = np.vstack([d_cmd_m[nonsat], np.ones(nonsat.sum())]).T
        slope, intercept = np.linalg.lstsq(A, d_st_m[nonsat], rcond=None)[0]
        print(f"  非飽和域の線形フィット: 実δ ≈ {slope:.3f} × 指令δ {intercept:+.2f}  "
              f"（非飽和 {int(nonsat.sum())}点）")
        if slope < 0.9:
            print(f"    → 指令の {slope:.2f} 倍に縮小（スケール{1/slope:.2f}分の1）")
            print(f"       config gain=1.639 の逆数 1/1.639={1/1.639:.3f} と比較せよ")
        elif slope > 1.1:
            print(f"    → 指令の {slope:.2f} 倍に拡大")
        else:
            print("    → ほぼ等倍（スケールは効いていない）")

    near_sat = a_st >= (sat_level - sat_band)
    if near_sat.sum() > 3:
        cmd_at_sat = a_cmd[near_sat]
        print(f"  頭打ち: 実δが|{sat_level:.0f}|°付近のとき、指令δは "
              f"{cmd_at_sat.min():.0f}〜{cmd_at_sat.max():.0f}°（中央{np.median(cmd_at_sat):.0f}°, "
              f"{int(near_sat.sum())}点）")
        print(f"    → 実δ {sat_level:.0f}° は固定上限。指令をこれ以上出しても実舵角は増えない")
    print(f"  [参照] config delta_max_deg=32° / gain=1.639 → 32/1.639={32/1.639:.1f}°  "
          f"vehicle_info max_steer_angle=0.64rad={math.degrees(0.64):.1f}°")


@analyzer("understeer", "アンダー/オーバー判定（ヨーレート比）", "指令と実現の乖離",
          ["odom", "steer"], optional=["imu"])
def _a_understeer(ctx, v_min=2.0, d_min=5.0, L=1.087):
    """実操舵δと速度vから期待ヨーレート ω_exp=v·tan(δ)/L を計算し、imu実測ヨーレート
    ω_act(angular_velocity.z) と比較。比<1=アンダー(回頭不足)、>1=オーバー(回頭過剰)。
    全体傾向とコーナー別(s)を集計する。imu_raw が必要。"""
    if not ctx.msgs.get(T["imu"]):
        print("[アンダー/オーバー判定] imu_raw 未収録でスキップ（次回収集で録ること）")
        return
    t, x, y, v = ctx.odom_arrays()
    t_st, d_st = ctx.signal_recv("steer", lambda m: m.steering_tire_angle)     # rad
    t_im, wz = ctx.signal_recv("imu", lambda m: m.angular_velocity.z)          # rad/s
    d_at = np.interp(t_im, t_st, d_st)
    v_at = np.interp(t_im, t, v)
    w_exp = v_at * np.tan(d_at) / L
    valid = (np.abs(np.degrees(d_at)) >= d_min) & (v_at >= v_min) & (np.abs(w_exp) > 0.05)
    if valid.sum() < 10:
        print("[アンダー/オーバー判定] 有効サンプル不足（操舵+速度が出ている区間が少ない）")
        return
    ratio = wz[valid] / w_exp[valid]
    med = np.median(ratio)
    print(f"[アンダー/オーバー判定]（実測ヨーレート / 期待ヨーレート, L={L:.3f}m）")
    print(f"  有効サンプル {valid.sum()}（|δ|≥{d_min:.0f}° かつ v≥{v_min:.0f}m/s）")
    print(f"  比 中央値={med:.2f}  p25={np.percentile(ratio,25):.2f}  p75={np.percentile(ratio,75):.2f}")
    if med < 0.85:
        print(f"  [診断] 全体的にアンダーステア傾向（回頭が操舵に対し {(1-med)*100:.0f}% 不足）")
    elif med > 1.15:
        print(f"  [診断] 全体的にオーバーステア傾向（回頭が {(med-1)*100:.0f}% 過剰）")
    else:
        print("  [診断] 概ねニュートラル（比≈1）")

    series = ctx.ey_series()
    if series is not None:
        ts, ss, ey, sx, sy, sv = series
        s_at = np.interp(t_im, ts, ss)[valid]
        ref = ctx.ref()
        track_len = ref[3] if ref else (s_at.max() if len(s_at) else 1.0)
        n_bins = 36
        bins = np.linspace(0, track_len, n_bins + 1)
        which = np.clip(np.digitize(s_at, bins) - 1, 0, n_bins - 1)
        binmed = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = ratio[which == b]
            if len(sel) >= 3:
                binmed[b] = np.median(sel)
        order = np.argsort(np.where(np.isnan(binmed), 9e9, binmed))[:6]
        print("  [区間別 ヨーレート比（36分割・アンダー強い順 上位6）]")
        for b in order:
            if np.isnan(binmed[b]):
                continue
            tag = "アンダー" if binmed[b] < 0.85 else ("オーバー" if binmed[b] > 1.15 else "中立")
            print(f"    s={bins[b]:5.0f}-{bins[b+1]:5.0f}m  比={binmed[b]:.2f}  {tag}")


@analyzer("steer_chain", "操舵チェーン段間比較", "指令と実現の乖離",
          ["cmd"], optional=["cmd_raw", "actu_cmd", "actu_st", "steer"])
def _a_steer_chain(ctx):
    """操舵指令が MPC生→gain後→actuation指示→actuation実現→実操舵 の
    どの段で頭打ちするかを比較し、実操舵21°上限がどの段（converter/車両）で
    課されているかを特定する。"""
    stages = [
        ("MPC生 raw",      "cmd_raw", lambda m: m.lateral.steering_tire_angle),
        ("gain後 cmd",     "cmd",     lambda m: m.lateral.steering_tire_angle),
        ("actuation指示",  "actu_cmd", lambda m: m.actuation.steer_cmd),
        ("actuation実現",  "actu_st",  lambda m: m.status.steer_status),
        ("実操舵 status",  "steer",    lambda m: m.steering_tire_angle),
    ]
    print("[操舵チェーン段間比較] 各段の操舵角[deg]（どこで頭打ちするか）")
    print("    段                max|δ|  p99|δ|  p50|δ|   サンプル")
    prev_max = None
    for label, key, getter in stages:
        if not ctx.msgs.get(T[key]):
            print(f"    {label:16s}    （未収録）")
            continue
        try:
            g = getter
            _, val = ctx.signal_recv(key, lambda m, _g=g: math.degrees(_g(m)))
            a = np.abs(val)
            mark = ""
            if prev_max is not None and a.max() < prev_max * 0.7:
                mark = "  ← ここで頭打ち（前段比 {:.0f}%）".format(a.max() / prev_max * 100)
            print(f"    {label:16s} {a.max():6.1f}  {np.percentile(a,99):6.1f}  "
                  f"{np.percentile(a,50):6.1f}   {len(a)}{mark}")
            prev_max = a.max()
        except Exception as e:
            print(f"    {label:16s}   取得失敗: {e}")
    print("  → max|δ| が急に縮む段が操舵上限を課している箇所。")
    print("    gain後cmdが大きいのに actuation/実操舵が21°なら、converter か車両側のクリップ。")
    print("    config: delta_max_deg=32° / gain=1.639 → 32/1.639≈19.5°（実測21°に近い＝gain補正の疑い）")


@analyzer("corner_sequence", "コーナー連鎖（弧長sで一望）", "衝突・接触",
          ["odom", "steer", "cmd"])
def _a_corner_sequence(ctx, s_from=185.0, s_to=222.0, n_rows=32, pick="worst"):
    """指定区間(s_from..s_to)を弧長sに沿って一望する。デフォルトは
    ヘアピン出口(s≈189)→直線→左コーナー(s≈197-210)→問題の右コーナー(s≈211-)。
    各点で参照κの向き(L/R)・e_y・実操舵・操舵レート(deg/s)・速度・方向整合を表示。
    pick='worst'(崩壊周) / 'best'(正常周) で見る周を選ぶ。
    操舵レートを config steer_rate_max(20deg/s) と比較し、連続コーナーの
    切り返しが間に合っているかを見る。"""
    series = ctx.ey_series()
    if series is None:
        print("[コーナー連鎖] ref_csv が必要です")
        return
    t, ss, ey, x, y, v = series
    prof = ctx.ref_profile()
    if prof is None:
        print("[コーナー連鎖] 参照CSVに kappa 列が必要です")
        return
    ksig = None
    ctx.corner_turn_sign(s_from)  # 副作用で signed プロファイル生成
    if ctx._refprof_signed not in (None, False):
        ss2, ksig = ctx._refprof_signed

    t_st, d_st = ctx.signal_recv("steer", lambda m: math.degrees(m.steering_tire_angle))

    in_zone = (ss >= s_from) & (ss <= s_to)
    if in_zone.sum() < 5:
        print(f"[コーナー連鎖] s={s_from:.0f}-{s_to:.0f}m を通過したデータが不足")
        return
    tz = np.sort(t[in_zone])
    gaps = np.where(np.diff(tz) > 3.0)[0]
    segments = np.split(tz, gaps + 1) if len(gaps) else [tz]
    # 各通過の最悪|e_y|を評価
    cand = []
    for seg in segments:
        if len(seg) < 3:
            continue
        m = (t >= seg.min()) & (t <= seg.max()) & in_zone
        if m.any():
            cand.append((np.abs(ey[m]).max(), seg.min(), seg.max()))
    if not cand:
        print("[コーナー連鎖] 有効な通過が見つかりません")
        return
    cand.sort()
    if pick == "best":
        worst, t0, t1 = cand[0]   # 最も e_y が小さい＝正常な通過
        label = "正常通過(最小e_y)"
    else:
        worst, t0, t1 = cand[-1]  # 崩壊通過
        label = "崩壊通過(最大e_y)"
    print(f"[コーナー連鎖] s={s_from:.0f}-{s_to:.0f}m を弧長で一望 "
          f"[{label}] t={t0-t[0]:.1f}〜{t1-t[0]:.1f}s, 最悪|e_y|={worst:.2f}m  "
          f"（pick='worst'/'best'で周を切替, 全{len(cand)}通過）")
    print("  区間構成: s189出口=ヘアピン / s197-210=左コーナー / s211-=問題の右コーナー")
    print("  操舵レート基準 steer_rate_max=20deg/s（'!'=超過）  κ向き: L=左 R=右 -=直")
    print("     s[m]  κ   e_y[m]  実δ[°] δ̇[°/s] v[km/h]  整合")

    s_grid = np.linspace(s_from, s_to, n_rows)
    msk = (t >= t0) & (t <= t1)
    ss_m, ey_m, v_m, t_m = ss[msk], ey[msk], v[msk], t[msk]
    o = np.argsort(t_m)
    ss_m, ey_m, v_m, t_m = ss_m[o], ey_m[o], v_m[o], t_m[o]

    # 衝突後の停止・回転データを除外：時間順に歩き、前進走行が続く区間だけ残す。
    # 主信号は「持続的な停止(v低下)」。レース中の崩壊は急減速→停止で終わるため。
    # 補助的に大きなs後退も崩壊後とみなす。
    if len(ss_m) > 3:
        keep = len(ss_m)
        s_peak = ss_m[0]
        stuck = 0
        slow = 0
        for i in range(1, len(ss_m)):
            if v_m[i] * 3.6 < 1.5:
                slow += 1
            else:
                slow = 0
            if ss_m[i] >= s_peak - 0.5:
                s_peak = max(s_peak, ss_m[i])
                stuck = 0
            else:
                stuck += 1
            # 持続停止(0.3s相当)か、大きなs後退で崩壊後と判定
            if slow >= 3 or s_peak - ss_m[i] > 3.0 or stuck >= 6:
                keep = i
                break
        if keep < len(ss_m):
            cut_s = ss_m[keep - 1]
            print(f"  （注: s≈{cut_s:.0f}m で走行停止。以降は衝突後の停止・回転データとして除外）")
            ss_m, ey_m, v_m, t_m = ss_m[:keep], ey_m[:keep], v_m[:keep], t_m[:keep]

    prev_d, prev_s = None, None
    switch_info = []
    s_last_real = ss_m.max() if len(ss_m) else s_from
    for sg in s_grid:
        if len(ss_m) == 0:
            break
        if sg > s_last_real + 1.0:
            break  # 切り詰めた走行ぶんを越えたら表示しない
        idx = int(np.argmin(np.abs(ss_m - sg)))
        if abs(ss_m[idx] - sg) > 4.0:
            continue
        e = ey_m[idx]
        vv = v_m[idx] * 3.6
        dd = np.interp(t_m[idx], t_st, d_st)
        # 操舵レート: この時刻周辺の実操舵の変化率
        tt = t_m[idx]
        win = (t_st >= tt - 0.15) & (t_st <= tt + 0.15)
        if win.sum() >= 2:
            tw, dw = t_st[win], d_st[win]
            ow = np.argsort(tw)
            rate = (dw[ow][-1] - dw[ow][0]) / max(tw[ow][-1] - tw[ow][0], 1e-3)
        else:
            rate = 0.0
        kk = np.interp(sg, ss2, ksig) if ksig is not None else 0.0
        kdir = "L" if kk > 0.01 else ("R" if kk < -0.01 else "-")
        align = ""
        if kdir == "L":
            align = "逆!" if dd < -1 else ("OK" if dd > 1 else "")
        elif kdir == "R":
            align = "逆!" if dd > 1 else ("OK" if dd < -1 else "")
        f_r = "!" if abs(rate) > STEER_RATE_MAX_DEG else " "
        print(f"   {sg:6.1f}  {kdir}  {e:+5.2f}  {dd:+5.1f}  {rate:+5.0f}{f_r} {vv:5.1f}   {align}")
        # 操舵符号反転（切り返し）を検出
        if prev_d is not None and prev_d * dd < 0 and abs(prev_d) > 1 and abs(dd) > 1:
            switch_info.append((prev_s, sg, prev_d, dd))
        prev_d, prev_s = dd, sg

    # 切り返し（左→右など符号反転）に要した距離を要約
    if switch_info:
        print("  [切り返し] 操舵の符号反転（左↔右の切り替え）:")
        for s_a, s_b, d_a, d_b in switch_info:
            print(f"    s={s_a:.0f}→{s_b:.0f}m で {d_a:+.0f}°→{d_b:+.0f}°（{abs(s_b-s_a):.0f}m かけて切替）")
    print("  → 左コーナーでe_yが膨らみ→右コーナー入口で『逆!』が続けば切替遅れ。"
          "δ̇に『!』が出れば steer_rate_max(20deg/s)が切り返しのボトルネック。")


# ============================================================
# 周回比較・崩壊トレース（既出）の下に続く
# ============================================================
def _lap_boundaries(ctx):
    """/awsim/status の lapCount(data[1]) 変化から周回境界の時刻[s]を返す。
    戻り値: [(lap, t_start, t_end), ...]（受信時刻ベース）。"""
    if not ctx.msgs.get(T["awsim"]):
        return None
    ms = ctx.msgs[T["awsim"]]
    tt = np.array(ctx.recv[T["awsim"]], dtype=float) * 1e-9
    lap = np.array([int(m.data[1]) if len(m.data) > 1 else 0 for m in ms])
    o = np.argsort(tt)
    tt, lap = tt[o], lap[o]
    bounds = []
    uniq = np.unique(lap)
    for L in uniq:
        sel = np.where(lap == L)[0]
        bounds.append((int(L), tt[sel[0]], tt[sel[-1]]))
    return bounds


@analyzer("lap_compare", "周回比較（崩壊周vs成功周）", "衝突・接触",
          ["cmd", "odom", "steer", "awsim"], core=False, optional=["imu"])
def _a_lap_compare(ctx, s_lo=285.0, s_hi=305.0, ey_fail=2.4):
    """崩壊コーナー区間(s_lo..s_hi)を毎周抽出し、各周を崩壊/成功に自動分類して
    進入速度・最悪e_y・操舵の立ち上がり・周期ジッタ・横G(あれば)を周ごとに比較。
    『なぜ割れるか』を、崩壊群と成功群の差として要約する。"""
    series = ctx.ey_series()
    if series is None:
        print("[周回比較] ref_csv が必要です")
        return
    t, ss, ey, x, y, v = series
    aey = np.abs(ey)

    bounds = _lap_boundaries(ctx)
    if not bounds:
        print("[周回比較] /awsim/status(lapCount) 未収録のため周回分割不可")
        return

    # 操舵・横G信号（受信時刻）
    t_cmd, d_cmd = ctx.signal_recv("cmd", lambda m: math.degrees(m.lateral.steering_tire_angle))
    has_imu = bool(ctx.msgs.get(T["imu"]))
    if has_imu:
        # IMU linear_acceleration.y を横G として使う（frame: base_link, y=左右）
        t_imu, ay_imu = ctx.signal_recv("imu", lambda m: m.linear_acceleration.y)

    print(f"[周回比較] 崩壊コーナー s={s_lo:.0f}-{s_hi:.0f}m を周ごとに抽出（崩壊判定 |e_y|≥{ey_fail}m）")
    rows = []
    for (L, t0, t1) in bounds:
        in_lap = (t >= t0) & (t <= t1)
        in_seg = in_lap & (ss >= s_lo) & (ss <= s_hi)
        if in_seg.sum() < 3:
            continue
        seg_aey = aey[in_seg]
        seg_v = v[in_seg]
        seg_t = t[in_seg]
        worst = seg_aey.max()
        failed = worst >= ey_fail
        # 進入速度: 区間入口側 (先頭20%) の平均
        k = max(1, int(len(seg_v) * 0.2))
        v_entry = np.sort(seg_t)
        order = np.argsort(seg_t)
        v_in = seg_v[order][:k].mean()
        # 進入位置 e_y: 区間入口の e_y
        ey_in = ey[in_seg][order][0]
        # 操舵の立ち上がり: 区間内の指令操舵 最大変化率
        wc = (t_cmd >= t0) & (t_cmd <= t1)
        tc, dc = t_cmd[wc], d_cmd[wc]
        # s で絞るため、cmd 時刻に対応する s を内挿
        sc = np.interp(tc, t, ss)
        wseg = (sc >= s_lo) & (sc <= s_hi)
        if wseg.sum() > 2:
            tcs, dcs = tc[wseg], dc[wseg]
            oo = np.argsort(tcs)
            rate = np.abs(np.gradient(dcs[oo], tcs[oo]))
            steer_rate_max = rate.max()
            steer_pk = np.abs(dcs).max()
        else:
            steer_rate_max = steer_pk = float("nan")
        # 周期ジッタ: 区間内 cmd 間隔の最大
        if len(tcs) > 2:
            jit = np.diff(np.sort(tcs)) * 1000.0
            jit_max = jit.max() if len(jit) else float("nan")
        else:
            jit_max = float("nan")
        # 横G
        ay_pk = float("nan")
        if has_imu:
            wi = (t_imu >= seg_t.min()) & (t_imu <= seg_t.max())
            if wi.sum() > 0:
                ay_pk = np.abs(ay_imu[wi]).max()
        rows.append(dict(lap=L, failed=failed, worst=worst, v_in=v_in, ey_in=ey_in,
                         steer_pk=steer_pk, steer_rate=steer_rate_max, jit=jit_max, ay=ay_pk))

    if not rows:
        print("  該当区間を通過した周がありません")
        return

    hdr = "  周  判定   最悪e_y  進入v   進入e_y  操舵pk  操舵レートpk  ジッタmax"
    if has_imu:
        hdr += "  横Gpk"
    print(hdr)
    for r in rows:
        line = (f"  {r['lap']:2d}  {'崩壊' if r['failed'] else '成功'}  "
                f"{r['worst']:6.2f}m  {r['v_in']*3.6:5.1f}  {r['ey_in']:+6.2f}m  "
                f"{r['steer_pk']:5.1f}d  {r['steer_rate']:7.0f}d/s  {r['jit']:6.1f}ms")
        if has_imu:
            line += f"  {r['ay']:5.1f}"
        print(line)

    # 崩壊群 vs 成功群の差分要約
    fail = [r for r in rows if r["failed"]]
    ok = [r for r in rows if not r["failed"]]
    print(f"\n  [集計] 崩壊 {len(fail)}周 / 成功 {len(ok)}周（崩壊率 {len(fail)/len(rows)*100:.0f}%）")
    if fail and ok:
        def mean(g, k):
            vals = [r[k] for r in g if not math.isnan(r[k])]
            return sum(vals) / len(vals) if vals else float("nan")
        print("  [崩壊群 vs 成功群 の平均差]")
        print(f"    進入速度    : 崩壊 {mean(fail,'v_in')*3.6:5.1f} vs 成功 {mean(ok,'v_in')*3.6:5.1f} km/h")
        print(f"    進入e_y     : 崩壊 {mean(fail,'ey_in'):+5.2f} vs 成功 {mean(ok,'ey_in'):+5.2f} m")
        print(f"    操舵レートpk : 崩壊 {mean(fail,'steer_rate'):5.0f} vs 成功 {mean(ok,'steer_rate'):5.0f} deg/s")
        print(f"    ジッタmax   : 崩壊 {mean(fail,'jit'):5.1f} vs 成功 {mean(ok,'jit'):5.1f} ms")
        if has_imu:
            print(f"    横Gpk       : 崩壊 {mean(fail,'ay'):5.1f} vs 成功 {mean(ok,'ay'):5.1f} m/s^2")
        # 最も差が出た指標を指摘
        dv = abs(mean(fail, 'v_in') - mean(ok, 'v_in')) * 3.6
        de = abs(mean(fail, 'ey_in') - mean(ok, 'ey_in'))
        dj = abs(mean(fail, 'jit') - mean(ok, 'jit'))
        hints = []
        if dv > 1.0:
            hints.append(f"進入速度が崩壊周で{dv:.1f}km/h違う→速度プロファイル/前コーナー脱出")
        if de > 0.3:
            hints.append(f"進入位置が崩壊周で{de:.2f}m違う→前コーナーのライン/姿勢")
        if dj > 15.0:
            hints.append(f"ジッタが崩壊周で{dj:.0f}ms違う→周期スパイク/localizationの混入")
        if hints:
            print("  [示唆] " + " / ".join(hints))
        else:
            print("  [示唆] 群間で目立つ差が小さい→確率的な求解不安定の可能性。"
                  "マージン（ay_max下げ or steer_rate_max上げ）で安定化を検討")
    elif not fail:
        print("  → この走行では崩壊周なし。全周成功")
    else:
        print("  → この走行では成功周なし。全周崩壊（設定が常に限界超過）")


# ============================================================
# 複数台走行用（単独の課題が片付いたらここに追加）
#   コアアナライザは複数台でも毎回走るので、ノイズ・操舵遅れは常に監視される。
# ------------------------------------------------------------
# @analyzer("rival_relative", "対戦相手との相対", "複数台・対戦", ["v2x", "odom"])
# def _a_rival(ctx):
#     ...  # 区間速度の相対比較・最接近・追い抜き地点 など
# ============================================================


# ============================================================
# 実行オーケストレータ
# ============================================================
def _category_rank(cat):
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)


def run(bag, ref_csv, only=None, category=None, core_only=False, extra_kwargs=None):
    all_topics = sorted({T[k] for a in REGISTRY
                         for k in (a["topics"] + a.get("optional", []))})
    msgs, recv = ab.read_bag(bag, all_topics)
    ctx = Ctx(msgs, recv, ref_csv)
    extra_kwargs = extra_kwargs or {}

    core = [a for a in REGISTRY if a["core"]]
    if core_only:
        chosen = list(core)
    elif only:
        keys = set(only.split(","))
        chosen = core + [a for a in REGISTRY if a["key"] in keys]
    elif category:
        chosen = core + [a for a in REGISTRY if category in a["category"]]
    else:
        chosen = list(REGISTRY)

    # 重複排除（順序保持）→ カテゴリ順に整列
    seen, uniq = set(), []
    for a in chosen:
        if a["key"] not in seen:
            seen.add(a["key"])
            uniq.append(a)
    uniq.sort(key=lambda a: (_category_rank(a["category"]), 0 if a["core"] else 1))

    last_cat = None
    for a in uniq:
        if a["category"] != last_cat:
            print(f"\n========== [{a['category']}] ==========")
            last_cat = a["category"]
        missing = [T[k] for k in a["topics"] if not msgs.get(T[k])]
        if missing:
            print(f"[{a['title']}] スキップ（未収録: {missing}）")
            continue
        try:
            kw = extra_kwargs.get(a["key"], {})
            a["fn"](ctx, **kw)
        except Exception as e:
            import traceback
            print(f"[{a['title']}] エラー: {e}")
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="レース走行ログ 統合分析エントリ")
    ap.add_argument("bag", nargs="?", help="rosbag ディレクトリ または .mcap")
    ap.add_argument("ref_csv", nargs="?", default=None, help="参照ライン CSV（任意）")
    ap.add_argument("--only", default=None, help="追加実行するアナライザkey（カンマ区切り, コアは常時）")
    ap.add_argument("--category", default=None, help="カテゴリ部分一致で絞る（コアは常時）")
    ap.add_argument("--core-only", action="store_true", help="システム健全性(コア)のみ実行")
    ap.add_argument("--list", action="store_true", help="アナライザ一覧を表示して終了")
    ap.add_argument("--seq-pick", default="worst", choices=["worst", "best"],
                    help="corner_sequence で見る周: worst=崩壊周 / best=正常周")
    ap.add_argument("--seq-range", default=None,
                    help="corner_sequence の弧長範囲 s_from,s_to（例: 185,222）")
    args = ap.parse_args()

    if args.list:
        print("登録アナライザ（★=コア:毎回必ず実行）:")
        for a in sorted(REGISTRY, key=lambda a: (_category_rank(a["category"]), 0 if a["core"] else 1)):
            star = "★" if a["core"] else " "
            print(f"  {star} {a['key']:14s} [{a['category']:8s}] {a['title']}  topics={a['topics']}")
        return
    if not args.bag:
        ap.error("bag を指定してください（一覧は --list）")
    seq_kw = {"pick": args.seq_pick}
    if args.seq_range:
        try:
            a0, a1 = args.seq_range.split(",")
            seq_kw["s_from"], seq_kw["s_to"] = float(a0), float(a1)
        except Exception:
            ap.error("--seq-range は s_from,s_to 形式（例: 185,222）")
    run(args.bag, args.ref_csv, only=args.only, category=args.category,
        core_only=args.core_only, extra_kwargs={"corner_sequence": seq_kw})


if __name__ == "__main__":
    main()
