#!/usr/bin/env python3
"""
race_analyzer.py — レース走行ログ(rosbag/mcap) プラグイン型分析基盤

「何が起きたか」を事実から逆算するための分析を、独立した "アナライザ" として
登録・実行する。新しい観点は @analyzer を付けた関数を1つ足すだけで増やせる
（analyze_bag.py へのパッチは不要）。

設計:
  - bag は1回だけ読み、必要トピックを集約して全アナライザで共有する。
  - 共通計算（参照ライン s, 各点 e_y, odom 時系列）は Ctx が一度だけ計算しキャッシュ。
  - 既存 analyze_bag.py の検証済みロジックは「ライブラリ」として import 再利用する。
  - 時刻はすべて受信時刻(recv)=実時間で統一（msg.stamp は use_sim_time で量子化され
    トピック間でズレるため、複数トピックを重ねる分析では使わない）。

使い方:
    python3 race_analyzer.py <bag> [ref_csv]                 # 全アナライザ実行
    python3 race_analyzer.py <bag> [ref_csv] --only hotspot_zoom,steer_sat
    python3 race_analyzer.py <bag> [ref_csv] --category 乖離  # 部分一致でカテゴリ絞り
    python3 race_analyzer.py --list                          # アナライザ一覧

新しいアナライザの足し方:
    @analyzer("key", "表示名", "カテゴリ", topics=["cmd", "odom"])
    def _a_xxx(ctx):
        # ctx.msgs[T["cmd"]], ctx.ey_series(), ctx.odom_arrays() などを使う
        ...
"""
import sys
import os
import math
import argparse
from pathlib import Path
import numpy as np

# 同ディレクトリの analyze_bag.py を下位ライブラリとして再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_bag as ab  # noqa: E402

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

# ---- アナライザ登録機構 ----
REGISTRY = []


def analyzer(key, title, category, topics):
    """アナライザ登録デコレータ。topics は T のキーのリスト（必須トピック）。"""
    def deco(fn):
        REGISTRY.append(dict(key=key, title=title, category=category,
                             topics=list(topics), fn=fn))
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

    def has(self, *keys):
        return all(self.msgs.get(T[k]) for k in keys)

    def ref(self):
        """参照ライン (rx, ry, s_wp, track_len) を一度だけ計算。ref_csv 必須。"""
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

    def odom_arrays(self):
        """odom を (t, x, y, v) のソート済み配列で返す（t=受信時刻[s]）。"""
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
        """各 odom サンプルの (t, s, e_y, x, y, v) を返す（t=受信時刻）。ref 必須。"""
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
        """任意トピックを (t_recv[s], 値) のソート済み配列で返す。"""
        ms = self.msgs[T[topic_key]]
        t = np.array(self.recv[T[topic_key]], dtype=float) * 1e-9
        val = np.array([getter(m) for m in ms], dtype=float)
        o = np.argsort(t)
        return t[o], val[o]


# ============================================================
# 既存ロジックのラッパー（analyze_bag の検証済み関数を再利用）
# ============================================================
@analyzer("jitter", "周期ジッタ", "ノイズ・異常", ["cmd"])
def _a_jitter(ctx):
    ab.report_jitter(ctx.recv[T["cmd"]])


@analyzer("speed", "速度到達", "指令と実現の乖離", ["cmd", "odom"])
def _a_speed(ctx):
    ab.report_speed(ctx.msgs[T["cmd"]], ctx.msgs[T["odom"]])


@analyzer("steer_sat", "操舵飽和・追従", "指令と実現の乖離", ["cmd", "steer", "odom"])
def _a_steer_sat(ctx):
    ab.report_steer_saturation(ctx.msgs[T["cmd"]], ctx.msgs[T["steer"]],
                               ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv)


@analyzer("collision_gt", "衝突(真値)", "衝突・接触", ["collision_gt", "odom"])
def _a_collision(ctx):
    ab.report_collision_gt(ctx.msgs[T["collision_gt"]], ctx.recv[T["collision_gt"]],
                           ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv)


@analyzer("ey", "e_y追従誤差", "指令と実現の乖離", ["odom"])
def _a_ey(ctx):
    if not ctx.ref_csv:
        print("[e_y追従誤差] ref_csv 未指定でスキップ")
        return
    ab.compute_ey(ctx.msgs[T["odom"]], ctx.recv[T["odom"]], ctx.ref_csv, plot=None)


# ============================================================
# 第1号 新規アナライザ: ホットスポット区間ズーム
# ============================================================
@analyzer("hotspot_zoom", "ホットスポット区間ズーム", "指令と実現の乖離",
          ["cmd", "odom", "steer"])
def _a_hotspot_zoom(ctx, window_s=1.5, n_rows=16):
    """e_y 最大点を自動検出し、その通過前後 window_s 秒の
    指令操舵・実操舵・速度・e_y を時系列で並べる。
    『指令の急峻さ』対『実操舵の遅れ』を可視化し、崩壊コーナーの機序を特定する。"""
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

    # 指令操舵 / 実操舵（受信時刻で統一）
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

    # 立ち上がり速度（窓内の最大変化率）と最大点での乖離で機序を診断
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
        print("         → 実操舵が指令に追従しきれず膨らんでいる。"
              "対策候補: 進入速度を一段落とす / 先読みを伸ばし操舵をなだらかに / 操舵レート上限の見直し")
    else:
        print("         → 操舵追従は概ね良好。膨らみは操舵以外（進入速度/ライン）が主因の可能性")


# ============================================================
# 実行オーケストレータ
# ============================================================
def run(bag, ref_csv, only=None, category=None):
    all_topics = sorted({T[k] for a in REGISTRY for k in a["topics"]})
    msgs, recv = ab.read_bag(bag, all_topics)
    ctx = Ctx(msgs, recv, ref_csv)

    selected = REGISTRY
    if only:
        keys = set(only.split(","))
        selected = [a for a in REGISTRY if a["key"] in keys]
    elif category:
        selected = [a for a in REGISTRY if category in a["category"]]

    for a in selected:
        missing = [T[k] for k in a["topics"] if not msgs.get(T[k])]
        if missing:
            print(f"[{a['title']}] スキップ（未収録: {missing}）")
            continue
        try:
            a["fn"](ctx)
        except Exception as e:
            import traceback
            print(f"[{a['title']}] エラー: {e}")
            traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="レース走行ログのプラグイン型分析基盤")
    ap.add_argument("bag", nargs="?", help="rosbag ディレクトリ または .mcap")
    ap.add_argument("ref_csv", nargs="?", default=None, help="参照ライン CSV（任意）")
    ap.add_argument("--only", default=None, help="実行するアナライザkey（カンマ区切り）")
    ap.add_argument("--category", default=None, help="カテゴリ部分一致で絞る")
    ap.add_argument("--list", action="store_true", help="アナライザ一覧を表示して終了")
    args = ap.parse_args()

    if args.list:
        print("登録アナライザ:")
        for a in REGISTRY:
            print(f"  {a['key']:14s} [{a['category']:8s}] {a['title']}  "
                  f"topics={a['topics']}")
        return
    if not args.bag:
        ap.error("bag を指定してください（一覧は --list）")
    run(args.bag, args.ref_csv, only=args.only, category=args.category)


if __name__ == "__main__":
    main()
