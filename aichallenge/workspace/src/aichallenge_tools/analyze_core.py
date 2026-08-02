#!/usr/bin/env python3
"""
rosbag(mcap) 解析: 周期ジッタ / 速度追従遅延 / GNSS実効遅延 / e_y 追従誤差

ホストでもコンテナでも動く（ROS 不要）。rosbags が bag 埋め込みの型定義(IDL/msg)から
Autoware 独自型も自動復元するため、ROS や Autoware パッケージのインストールは不要。

準備:
    pip install rosbags numpy

使い方:
    python3 analyze_bag.py <bag_dir_or_mcap> [reference_csv]

  <bag_dir_or_mcap> : ros2 bag ディレクトリ（metadata.yaml と *.mcap を含む）
                      または .mcap ファイル（親ディレクトリを自動使用）
  reference_csv     : 省略可。traj_mincurv.csv 等（s,x,y,... 形式でも列を自動判定）
"""
import sys
import os
import glob
import math
import argparse
from pathlib import Path
import numpy as np

TOPIC_CMD = "/control/command/control_cmd"        # AckermannControlCommand（時刻は msg.stamp）
TOPIC_ODOM = "/localization/kinematic_state"       # nav_msgs/Odometry
TOPIC_VEL = "/vehicle/status/velocity_status"      # VelocityReport（低遅延の車輪速）
TOPIC_GNSS = "/sensing/gnss/pose_with_covariance"  # PoseWithCovarianceStamped（遅延あり）
TOPIC_V2X = "/v2x/vehicle_positions/markers"       # MarkerArray（他車位置, ns=v2x_vehicles）
TOPIC_COND = "/aichallenge/pitstop/condition"      # Int32（衝突で急増。_condition_callback と同じ）
TOPIC_STEER = "/vehicle/status/steering_status"        # SteeringReport（実操舵角）
TOPIC_GT_COLLISION = "/awsim/ground_truth/on_collision"  # Bool（衝突真値）


# ----------------------------- bag 読み出し（ROS不要） -----------------------------
def _find_mcap_files(uri):
    if os.path.isfile(uri) and uri.endswith(".mcap"):
        return [uri]
    if os.path.isdir(uri):
        return sorted(glob.glob(os.path.join(uri, "*.mcap")))
    return []


def read_bag_anyreader(bagdir, topics):
    """metadata.yaml がある正規 rosbag2 を AnyReader で読む。"""
    from rosbags.highlevel import AnyReader
    tset = set(topics)
    msgs = {t: [] for t in topics}
    recv = {t: [] for t in topics}
    with AnyReader([Path(bagdir)]) as reader:
        missing = tset - {c.topic for c in reader.connections}
        if missing:
            print(f"[bag] 未収録トピック: {sorted(missing)}")
        conns = [c for c in reader.connections if c.topic in tset]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msgs[conn.topic].append(reader.deserialize(raw, conn.msgtype))
            recv[conn.topic].append(t_ns)
    print(f"[bag] AnyReader 読み出し（ROS不要）: {bagdir}")
    return msgs, recv


def read_bag_mcap_raw(uri, topics):
    """metadata.yaml が無い生 .mcap を直接読む（ros2idl/ros2msg 両対応, ROS不要）。"""
    from mcap.reader import make_reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_idl, get_types_from_msg
    files = _find_mcap_files(uri)
    if not files:
        raise FileNotFoundError(f".mcap が見つかりません: {uri}")
    ts = get_typestore(Stores.ROS2_HUMBLE)
    registered = set()
    tset = set(topics)
    msgs = {t: [] for t in topics}
    recv = {t: [] for t in topics}
    for fp in files:
        with open(fp, "rb") as f:
            for schema, channel, message in make_reader(f).iter_messages(topics=list(topics)):
                if channel.topic not in tset:
                    continue
                if schema.name not in registered:
                    text = schema.data.decode()
                    try:
                        types = (get_types_from_idl(text) if schema.encoding == "ros2idl"
                                 else get_types_from_msg(text, schema.name))
                        ts.register({k: v for k, v in types.items() if k not in ts.types})
                    except Exception as e:
                        print(f"[warn] 型登録失敗 {schema.name}: {e}")
                    registered.add(schema.name)
                msgs[channel.topic].append(ts.deserialize_cdr(message.data, schema.name))
                recv[channel.topic].append(message.log_time)
    print(f"[bag] 生mcap直接読み出し（{len(files)}ファイル, metadata.yaml不要, ROS不要）")
    return msgs, recv


def read_bag(uri, topics):
    p = Path(uri)
    bagdir = p.parent if p.is_file() else p
    if (bagdir / "metadata.yaml").exists():
        try:
            return read_bag_anyreader(bagdir, topics)
        except Exception as e:
            print(f"[bag] AnyReader 失敗({e}) → 生mcapにフォールバック")
    return read_bag_mcap_raw(uri, topics)


def list_topics(uri):
    """bag 内の全トピックを (件数, トピック名, 型) で一覧。障害物トピック発見用。"""
    p = Path(uri)
    bagdir = p.parent if p.is_file() else p
    print(f"[topics] {bagdir}")
    if (bagdir / "metadata.yaml").exists():
        from rosbags.highlevel import AnyReader
        with AnyReader([bagdir]) as reader:
            for topic, mt, cnt in sorted((c.topic, c.msgtype, c.msgcount) for c in reader.connections):
                print(f"  {cnt:8d}  {topic:52s} {mt}")
    else:
        from mcap.reader import make_reader
        for fp in _find_mcap_files(uri):
            with open(fp, "rb") as f:
                summ = make_reader(f).get_summary()
                cnts = summ.statistics.channel_message_counts if summ.statistics else {}
                for cid, c in sorted(summ.channels.items(), key=lambda kv: kv[1].topic):
                    name = summ.schemas[c.schema_id].name if c.schema_id in summ.schemas else "?"
                    print(f"  {cnts.get(cid, 0):8d}  {c.topic:52s} {name}")


# ----------------------------- 共通ユーティリティ -----------------------------
def stamp_s(msg):
    # AckermannControlCommand は header を持たず msg.stamp に時刻がある。
    # Odometry / VelocityReport / PoseWithCovarianceStamped は msg.header.stamp。
    s = msg.header.stamp if hasattr(msg, "header") else msg.stamp
    return s.sec + s.nanosec * 1e-9


def _sorted(t, v):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    idx = np.argsort(t)
    return t[idx], v[idx]


def _xcorr_lag(a, b, fs, max_lag_s):
    """a に対する b の遅れ[s]を ±max_lag_s に制限して相互相関で推定。
    戻り値: (tau[s], 正規化ピーク[0-1], 境界張り付きフラグ)。
    b(t) ≈ a(t - tau) のとき tau>0。"""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    corr = np.correlate(b, a, mode="full") / denom
    center = len(a) - 1
    max_lag = int(max_lag_s * fs)
    lo, hi = max(0, center - max_lag), center + max_lag + 1
    sub = corr[lo:hi]
    k = int(np.argmax(sub))
    lag = (k + lo - center) / fs
    at_boundary = (k == 0 or k == len(sub) - 1)
    return lag, float(sub[k]), at_boundary


# ----------------------------- 各解析 -----------------------------
def report_speed(cmd_msgs, odom_msgs):
    """指令速度(control_cmd) と実速度(kinematic_state) を比較し、到達速度・頭打ちを診断。
    実加速度も推定し、a_max に張り付くか（MPC律速）/ 手前で頭打ちか（車両律速）を切り分ける。"""
    v_cmd = np.array([m.longitudinal.speed for m in cmd_msgs])
    v_act = np.array([m.twist.twist.linear.x for m in odom_msgs])
    t_act = np.array([stamp_s(m) for m in odom_msgs])
    kmh = 3.6
    cmd_max, act_max = v_cmd.max(), v_act.max()
    thr = np.percentile(v_act, 80)
    hi = v_act >= thr
    print("[速度到達]（指令 vs 実速度）")
    print(f"  指令 max={cmd_max:.2f} m/s ({cmd_max*kmh:.1f} km/h)   実 max={act_max:.2f} m/s ({act_max*kmh:.1f} km/h)")
    print(f"  高速域(実上位20%) 平均: 実={v_act[hi].mean():.2f} m/s ({v_act[hi].mean()*kmh:.1f} km/h)")
    if cmd_max > act_max + 0.5:
        print(f"  [診断] 指令は {cmd_max*kmh:.1f}km/h まで出ているのに実速度が {act_max*kmh:.1f}km/h で頭打ち → 車両側律速の疑い")
    else:
        print(f"  [診断] 指令自体が {cmd_max*kmh:.1f}km/h 止まり（実とほぼ一致）→ MPC側律速")

    # 実加速度（実速度の数値微分、軽く平滑化）
    order = np.argsort(t_act)
    ts, vs = t_act[order], v_act[order]
    dt = np.diff(ts)
    ok = dt > 1e-3
    acc = np.diff(vs)[ok] / dt[ok]
    # 移動平均で微分ノイズを抑制
    if len(acc) >= 5:
        k = 5
        acc_s = np.convolve(acc, np.ones(k) / k, mode="same")
    else:
        acc_s = acc
    a_p95 = np.percentile(acc_s, 95)
    a_p99 = np.percentile(acc_s, 99)
    print(f"  [実加速度] p95={a_p95:.2f} m/s^2  p99={a_p99:.2f} m/s^2  max={acc_s.max():.2f} m/s^2")
    print("         （config の a_max と比較: a_max付近に張り付く→a_max律速 / 大きく下→accel_map等の車両律速）")

    # 減速側（負の加速度）の統計 — brake_map が指令通り減速できているかの判定材料
    dec = acc_s[acc_s < 0]
    if len(dec) > 0:
        d_p5 = np.percentile(acc_s, 5)     # 下位5%（強い減速側）
        d_p1 = np.percentile(acc_s, 1)
        print(f"  [実減速度] p5={d_p5:.2f} m/s^2  p1={d_p1:.2f} m/s^2  min(最大減速)={acc_s.min():.2f} m/s^2")
        print("         （config a_min / brake_map 最大減速 と比較: 届かない→brake_map が弱い疑い）")

    # リミッター検出: 実速度が「頭打ち速度」付近のとき、指令加速度がどうなっているか
    try:
        a_cmd = np.array([m.longitudinal.acceleration for m in cmd_msgs])
        t_cmd = np.array([stamp_s(m) for m in cmd_msgs])
        v_at_cmd = np.interp(t_cmd, ts, vs)  # 指令時刻における実速度
        cap = act_max - 0.3                  # 頭打ち速度の少し下
        near = v_at_cmd >= cap
        if near.sum() > 5:
            mean_acmd = a_cmd[near].mean()
            neg_ratio = float((a_cmd[near] < -0.05).mean())
            print(f"  [リミッター検査] 実速度≧{cap*kmh:.1f}km/h のとき:")
            print(f"      指令加速度 平均={mean_acmd:+.2f} m/s^2  / 負(減速)の割合={neg_ratio*100:.0f}%")
            if neg_ratio > 0.3 or mean_acmd < -0.05:
                print(f"      → 頭打ち速度付近で減速指令あり＝約{act_max*kmh:.0f}km/h に速度制限の疑い")
            else:
                print("      → 減速指令は出ていない。加速不足（車両駆動力 or 直線長）が原因")
    except AttributeError:
        print("  [リミッター検査] control_cmd に acceleration フィールド無し（スキップ）")
    return cmd_max, act_max


def report_stops(odom_msgs, recv_ns, v_thresh=0.3, min_dur=0.5):
    """実速度が閾値未満の停止イベントを検出（障害物停止ゲートの判定用）。"""
    v = np.array([m.twist.twist.linear.x for m in odom_msgs])
    t = np.array(recv_ns, dtype=float) * 1e-9
    x = np.array([m.pose.pose.position.x for m in odom_msgs])
    y = np.array([m.pose.pose.position.y for m in odom_msgs])
    order = np.argsort(t)
    v, t, x, y = v[order], t[order], x[order], y[order]
    stopped = v < v_thresh
    events = []
    i, n = 0, len(v)
    while i < n:
        if stopped[i]:
            j = i
            while j < n and stopped[j]:
                j += 1
            dur = t[j - 1] - t[i]
            if dur >= min_dur:
                events.append((t[i] - t[0], dur, x[i], y[i]))
            i = j
        else:
            i += 1
    print(f"[停止イベント] 実速度<{v_thresh}m/s が {min_dur}s 以上継続: {len(events)}件")
    for (ts, dur, xx, yy) in events[:10]:
        print(f"    t={ts:6.1f}s  停止 {dur:5.1f}s  位置({xx:.1f}, {yy:.1f})")
    return events


def report_obstacle_distance(odom_msgs, recv_odom, v2x_marker_msgs, recv_v2x, collide_dist=1.5):
    """V2X他車(markers の ns=v2x_vehicles)と自車の距離推移・最接近・追突瞬間を分析。"""
    # 各時刻の他車位置リストを抽出
    obst = []  # (t, [(x,y),...])
    for m, t in zip(v2x_marker_msgs, recv_v2x):
        pts = []
        for mk in m.markers:
            ns = getattr(mk, "ns", "")
            act = getattr(mk, "action", 0)
            if ns == "v2x_vehicles" and act == 0:
                pts.append((mk.pose.position.x, mk.pose.position.y))
        if pts:
            obst.append((t * 1e-9, pts))
    if not obst:
        print("[障害物距離] v2x_vehicles マーカーが見つかりません")
        return
    ot = np.array([o[0] for o in obst])

    # 自車の時系列
    et = np.array(recv_odom, dtype=float) * 1e-9
    ex = np.array([m.pose.pose.position.x for m in odom_msgs])
    ey = np.array([m.pose.pose.position.y for m in odom_msgs])
    ev = np.array([m.twist.twist.linear.x for m in odom_msgs])
    order = np.argsort(et)
    et, ex, ey, ev = et[order], ex[order], ey[order], ev[order]

    # 各自車サンプルで、直近の他車集合との最小距離
    mind = np.full(len(et), np.inf)
    for k in range(len(et)):
        j = int(np.argmin(np.abs(ot - et[k])))
        for (ox, oy) in obst[j][1]:
            d = math.hypot(ex[k] - ox, ey[k] - oy)
            if d < mind[k]:
                mind[k] = d
    kmin = int(np.argmin(mind))
    print("[障害物距離]（V2X他車との最接近）")
    print(f"  他車台数(最新): {len(obst[-1][1])}  サンプル数: {len(et)}")
    print(f"  最接近 = {mind[kmin]:.2f} m  @ t={et[kmin]-et[0]:.1f}s  自車速度={ev[kmin]:.2f} m/s ({ev[kmin]*3.6:.1f} km/h)")

    # START時(最初の2秒)の前方障害物距離と v_safe 予測
    a_brake = 3.0          # m/s^2（a_min 相当）
    margin_center = 3.0    # 中心間マージン = 表面間1.0m + 車長2.0m
    t0 = et[0]
    early = et - t0 <= 2.0
    if early.any():
        d_start = mind[early].min()  # START付近の最接近(中心間)
        usable = max(0.0, d_start - margin_center)
        v_safe_start = math.sqrt(2.0 * a_brake * usable)
        print(f"  [START時] 前方障害物まで(中心間) {d_start:.1f} m → 使える距離 {usable:.1f} m")
        print(f"            v_safe = sqrt(2*{a_brake}*{usable:.1f}) = {v_safe_start:.2f} m/s ({v_safe_start*3.6:.1f} km/h)")
        if v_safe_start > 0.5:
            print(f"            → フォールバックで発進可能（{v_safe_start*3.6:.1f}km/hで前進し、接近で減速・手前で停止）")
        else:
            print("            → START時点で既にマージン内。最初から停止が正解挙動")

    # 追突判定
    hit = mind <= collide_dist
    if hit.any():
        first = int(np.argmax(hit))
        print(f"  [追突] 距離≤{collide_dist}m に到達: t={et[first]-et[0]:.1f}s  そのとき速度={ev[first]*3.6:.1f} km/h")
        # 減速がいつ始まったか（最接近の手前で速度が落ち始めた点）
        pre = ev[:first+1]
        if len(pre) > 3:
            vpk = int(np.argmax(pre))
            print(f"        減速開始の目安: t={et[vpk]-et[0]:.1f}s(速度{pre[vpk]*3.6:.1f}km/h) → 追突まで {et[first]-et[vpk]:.1f}s")
            print(f"        → この区間で止まりきれずに追突。検出/減速の早期化が必要")
    else:
        print(f"  [OK] 最接近 {mind[kmin]:.2f} m で {collide_dist}m 以内に入らず（追突なし）")


def report_collisions(cond_msgs, recv_cond, odom_msgs, recv_odom, ref_csv=None, jump=30.0):
    """/aichallenge/pitstop/condition の急増(>jump)を衝突として検出。
    発生時刻・速度・位置（ref があれば弧長s・e_y）を出す。
    mpc_controller._condition_callback と同じ diff>30 ロジック。"""
    if not cond_msgs:
        print("[衝突イベント] condition トピック未収録のためスキップ")
        return []
    cv = np.array([m.data for m in cond_msgs], dtype=float)
    ct = np.array(recv_cond, dtype=float) * 1e-9
    o = np.argsort(ct)
    cv, ct = cv[o], ct[o]
    ot = np.array(recv_odom, dtype=float) * 1e-9
    ox = np.array([m.pose.pose.position.x for m in odom_msgs])
    oy = np.array([m.pose.pose.position.y for m in odom_msgs])
    ov = np.array([m.twist.twist.linear.x for m in odom_msgs])
    oo = np.argsort(ot)
    ot, ox, oy, ov = ot[oo], ox[oo], oy[oo], ov[oo]

    hits = np.where(np.diff(cv) > jump)[0] + 1
    print(f"[衝突イベント]（condition 急増 > {jump:.0f}）: {len(hits)}件")
    if len(hits) == 0:
        print("  → 衝突は検出されず")
        return []

    s_info = None
    if ref_csv:
        try:
            rx, ry, _, _ = _load_ref_xy(ref_csv, ox, oy)
            seg = np.hypot(np.diff(rx), np.diff(ry))
            s_wp = np.concatenate([[0.0], np.cumsum(seg)])
            s_info = (rx, ry, s_wp)
        except Exception as e:
            print(f"  [warn] s計算用 ref 読み込み失敗: {e}")

    t0 = ot[0]
    events = []
    for h in hits:
        th = ct[h]
        k = int(np.argmin(np.abs(ot - th)))
        x, y, v = ox[k], oy[k], ov[k]
        line = f"  t={th-t0:6.1f}s  速度={v*3.6:5.1f}km/h  位置({x:.1f},{y:.1f})"
        if s_info is not None:
            rx, ry, s_wp = s_info
            i = int(np.argmin((rx - x) ** 2 + (ry - y) ** 2))
            j = (i + 1) % len(rx)
            tx, ty = rx[j] - rx[i], ry[j] - ry[i]
            nrm = math.hypot(tx, ty) + 1e-9
            ey = -(ty / nrm) * (x - rx[i]) + (tx / nrm) * (y - ry[i])
            line += f"  s={s_wp[i]:.0f}m  e_y={ey:+.2f}m"
        print(line)
        events.append((th - t0, x, y, v))
    print("  → 衝突地点の s が e_y ホットスポットと一致すれば、膨らみ→コース外接触が確定")
    return events


def report_steer(cmd_msgs, odom_msgs=None):
    """ステア指令(steering_tire_angle)の振動を定量化し、蛇行(お釣り)を検出する。
    符号反転頻度・振幅・レートを出す。周期高速化後のゲイン過敏を疑う材料。"""
    st = np.array([m.lateral.steering_tire_angle for m in cmd_msgs], dtype=float)
    tt = np.array([stamp_s(m) for m in cmd_msgs])
    o = np.argsort(tt)
    st, tt = st[o], tt[o]
    if len(tt) < 3:
        print("[ステア指令の振動] サンプル不足でスキップ")
        return
    dur = tt[-1] - tt[0]
    deg = np.rad2deg(st)
    sign = np.sign(st)
    sign[sign == 0] = 1
    zc = int(np.sum(np.abs(np.diff(sign)) > 0))
    zc_rate = zc / max(dur, 1e-9)
    dt = np.diff(tt)
    ok = dt > 1e-3
    rate = np.abs(np.diff(st)[ok] / dt[ok]) if ok.any() else np.array([])
    print("[ステア指令の振動]（蛇行/お釣りの定量化）")
    print(f"  振幅: std={deg.std():.2f}deg  p95|delta|={np.percentile(np.abs(deg),95):.2f}deg  max|delta|={np.abs(deg).max():.2f}deg")
    print(f"  符号反転: {zc}回 / {dur:.0f}s = {zc_rate:.2f} 回/s")
    if len(rate):
        print(f"  ステアレート: p95={np.rad2deg(np.percentile(rate,95)):.0f}deg/s  max={np.rad2deg(rate.max()):.0f}deg/s")
    if zc_rate > 2.0:
        print(f"  [診断] 符号反転 {zc_rate:.2f}回/s は高め。左右に振れる蛇行（お釣り）の疑い")
    else:
        print("  [診断] 符号反転は低頻度。蛇行は目立たない")
    print("  ※ 周期 12->39Hz 化でステア応答が速くなり、ゲインが過敏だと振動が増える")


def report_steer_saturation(cmd_msgs, steer_msgs, odom_msgs=None, recv_odom=None, ref_csv=None, limit_deg=30.0):
    """control_cmd の目標操舵角が入力上限(±limit_deg = AWSIM Max Steer Angle Input)に
    飽和する割合と、実操舵(steering_status)との追従乖離を出す。
    ref があれば s 別飽和率も出し、追従不良(操舵不足)の発生区間を特定する。"""
    cmd_d = np.rad2deg(np.array([m.lateral.steering_tire_angle for m in cmd_msgs], dtype=float))
    t_cmd = np.array([stamp_s(m) for m in cmd_msgs])
    o = np.argsort(t_cmd)
    cmd_d, t_cmd = cmd_d[o], t_cmd[o]
    acmd = np.abs(cmd_d)
    sat = acmd >= limit_deg * 0.98
    print(f"[操舵飽和]（目標操舵角 vs 入力上限 ±{limit_deg:.0f}°）")
    print(f"  目標 max|delta|={acmd.max():.1f}deg  p99={np.percentile(acmd,99):.1f}deg  p95={np.percentile(acmd,95):.1f}deg")
    print(f"  |delta|>={limit_deg:.0f}deg(飽和)の割合: {sat.mean()*100:.1f}%")
    if sat.mean() > 0.02:
        print("  [診断] 飽和が発生＝MPCが上限超の操舵を要求。該当区間で曲がりきれず膨らむ")
    else:
        print("  [診断] 飽和はほぼ無し。膨らみの主因は操舵上限ではない")

    if steer_msgs:
        act = np.rad2deg(np.array([m.steering_tire_angle for m in steer_msgs], dtype=float))
        t_act = np.array([stamp_s(m) for m in steer_msgs])
        oa = np.argsort(t_act)
        act, t_act = act[oa], t_act[oa]
        cmd_at_act = np.interp(t_act, t_cmd, cmd_d)
        dev = np.abs(cmd_at_act - act)
        print(f"  実操舵 max|delta|={np.abs(act).max():.1f}deg")
        print(f"  指令-実 乖離: p50={np.percentile(dev,50):.1f}deg  p95={np.percentile(dev,95):.1f}deg  max={dev.max():.1f}deg")
        if np.percentile(dev, 95) > 5.0:
            print("  [診断] 指令に実操舵が追従しきれていない（アクチュエータ応答/レート制限の疑い）")

    if ref_csv and odom_msgs is not None and recv_odom is not None:
        ox = np.array([m.pose.pose.position.x for m in odom_msgs])
        oy = np.array([m.pose.pose.position.y for m in odom_msgs])
        ot = np.array(recv_odom, dtype=float) * 1e-9
        oo = np.argsort(ot)
        ox, oy, ot = ox[oo], oy[oo], ot[oo]
        try:
            rx, ry, _, _ = _load_ref_xy(ref_csv, ox, oy)
        except Exception as e:
            print(f"  [s別飽和] ref 読み込み失敗: {e}")
            return
        seg = np.hypot(np.diff(rx), np.diff(ry))
        s_wp = np.concatenate([[0.0], np.cumsum(seg)])
        track_len = s_wp[-1] + math.hypot(rx[0] - rx[-1], ry[0] - ry[-1])
        d_at_odom = np.abs(np.interp(ot, t_cmd, cmd_d))
        ss = np.empty(len(ox))
        for k in range(len(ox)):
            i = int(np.argmin((rx - ox[k]) ** 2 + (ry - oy[k]) ** 2))
            ss[k] = s_wp[i]
        n_bins = 36
        bins = np.linspace(0, track_len, n_bins + 1)
        which = np.clip(np.digitize(ss, bins) - 1, 0, n_bins - 1)
        satrate = np.zeros(n_bins)
        dmax = np.zeros(n_bins)
        for b in range(n_bins):
            sel = d_at_odom[which == b]
            if len(sel):
                satrate[b] = (sel >= limit_deg * 0.98).mean() * 100
                dmax[b] = sel.max()
        order = np.argsort(satrate)[::-1][:8]
        print(f"  [区間別 操舵飽和率（{n_bins}分割・上位8区間）]")
        for b in order:
            if satrate[b] <= 0:
                continue
            print(f"    s={bins[b]:5.0f}-{bins[b+1]:5.0f}m  飽和率={satrate[b]:3.0f}%  目標max|delta|={dmax[b]:.1f}deg")
        print("  → e_y ホットスポット(s≈307m)で飽和率が高ければ、操舵上限が追従不良の主因")


def report_collision_gt(coll_msgs, recv_coll, odom_msgs, recv_odom, ref_csv=None):
    """ground_truth/on_collision(Bool) の False→True 立ち上がりを衝突として検出し、
    時刻・速度・位置（ref があれば弧長s・e_y）を出す。"""
    if not coll_msgs:
        print("[衝突(真値)] on_collision 未収録のためスキップ")
        return []
    cb = np.array([1 if m.data else 0 for m in coll_msgs], dtype=int)
    ct = np.array(recv_coll, dtype=float) * 1e-9
    o = np.argsort(ct)
    cb, ct = cb[o], ct[o]
    ot = np.array(recv_odom, dtype=float) * 1e-9
    ox = np.array([m.pose.pose.position.x for m in odom_msgs])
    oy = np.array([m.pose.pose.position.y for m in odom_msgs])
    ov = np.array([m.twist.twist.linear.x for m in odom_msgs])
    oo = np.argsort(ot)
    ot, ox, oy, ov = ot[oo], ox[oo], oy[oo], ov[oo]
    rising = np.where(np.diff(cb) > 0)[0] + 1
    print(f"[衝突(真値 on_collision)]: {len(rising)}件")
    if len(rising) == 0:
        print("  → 衝突なし")
        return []
    s_info = None
    if ref_csv:
        try:
            rx, ry, _, _ = _load_ref_xy(ref_csv, ox, oy)
            seg = np.hypot(np.diff(rx), np.diff(ry))
            s_wp = np.concatenate([[0.0], np.cumsum(seg)])
            s_info = (rx, ry, s_wp)
        except Exception as e:
            print(f"  [warn] s計算用 ref 失敗: {e}")
    t0 = ot[0]
    events = []
    for h in rising:
        th = ct[h]
        k = int(np.argmin(np.abs(ot - th)))
        x, y, v = ox[k], oy[k], ov[k]
        line = f"  t={th-t0:6.1f}s  速度={v*3.6:5.1f}km/h  位置({x:.1f},{y:.1f})"
        if s_info is not None:
            rx, ry, s_wp = s_info
            i = int(np.argmin((rx - x) ** 2 + (ry - y) ** 2))
            j = (i + 1) % len(rx)
            tx, ty = rx[j] - rx[i], ry[j] - ry[i]
            nrm = math.hypot(tx, ty) + 1e-9
            ey = -(ty / nrm) * (x - rx[i]) + (tx / nrm) * (y - ry[i])
            line += f"  s={s_wp[i]:.0f}m  e_y={ey:+.2f}m"
        print(line)
        events.append((th - t0, x, y, v))
    print("  → 衝突 s が e_y/操舵飽和ホットスポットと一致するか確認")
    return events


def report_jitter(recv_ns):
    # 受信(記録)時刻＝実時間に近い。header.stamp は use_sim_time でクロック量子化されるため使わない。
    ts = np.sort(np.array(recv_ns, dtype=float)) * 1e-9
    dt = np.diff(ts) * 1000.0  # ms
    print(f"[周期ジッタ {TOPIC_CMD}]（受信時刻＝実時間ベース）")
    print(f"  サンプル数: {len(dt)}")
    if len(dt) < 2:
        print("  （サンプル不足: control_cmd がほぼ出ていない＝発進前/求解失敗の可能性）")
        return
    print(f"  p50={np.percentile(dt,50):.2f}ms  p99={np.percentile(dt,99):.2f}ms  "
          f"max={dt.max():.2f}ms  min={dt.min():.2f}ms  (目標 25ms=40Hz)")


def estimate_speed_lag(cmd_msgs, odom_msgs, fs=100.0, max_lag_s=1.0):
    t_cmd, v_cmd = _sorted([stamp_s(m) for m in cmd_msgs],
                           [m.longitudinal.speed for m in cmd_msgs])
    t_act, v_act = _sorted([stamp_s(m) for m in odom_msgs],
                           [m.twist.twist.linear.x for m in odom_msgs])
    t0, t1 = max(t_cmd[0], t_act[0]), min(t_cmd[-1], t_act[-1])
    grid = np.arange(t0, t1, 1.0 / fs)
    c = np.interp(grid, t_cmd, v_cmd)
    a = np.interp(grid, t_act, v_act)
    tau, peak, boundary = _xcorr_lag(c, a, fs, max_lag_s)
    print(f"[速度追従遅延] 約 {tau*1000:.0f} ms （+ は actual が command に遅れる量, ±{int(max_lag_s*1000)}ms内, 相関{peak:.2f}）")
    if boundary or peak < 0.3:
        print("  [警告] 相関が弱い/境界張り付き。信頼度低（加減速のある走行で録り直し推奨）")
    print("  ※駆動・アクチュエータ応答の総遅延。GNSS センサ遅延とは別物")
    return tau


def estimate_gnss_delay(vel_msgs, gnss_msgs, fs=50.0, max_lag_s=1.0):
    """低遅延の車輪速 vs GNSS位置微分速度 の相互相関で GNSS 実効遅延を推定。"""
    t_v, v_wheel = _sorted([stamp_s(m) for m in vel_msgs],
                           [m.longitudinal_velocity for m in vel_msgs])
    t_g = np.array([stamp_s(m) for m in gnss_msgs])
    px = np.array([m.pose.pose.position.x for m in gnss_msgs])
    py = np.array([m.pose.pose.position.y for m in gnss_msgs])
    order = np.argsort(t_g)
    t_g, px, py = t_g[order], px[order], py[order]
    dt = np.diff(t_g)
    spd_g = np.hypot(np.diff(px), np.diff(py)) / np.maximum(dt, 1e-3)
    t_g = t_g[1:]

    t0, t1 = max(t_v[0], t_g[0]), min(t_v[-1], t_g[-1])
    if t1 <= t0:
        print("[GNSS 実効遅延] 時間重複が無く推定不可")
        return None
    grid = np.arange(t0, t1, 1.0 / fs)
    a = np.interp(grid, t_v, v_wheel)   # 低遅延（車輪速）
    b = np.interp(grid, t_g, spd_g)     # GNSS（遅延あり）
    lag, peak, boundary = _xcorr_lag(a, b, fs, max_lag_s)
    print(f"[GNSS 実効遅延（bag推定）] 約 {lag*1000:.0f} ms （±{int(max_lag_s*1000)}ms内, 相関{peak:.2f}）")
    if boundary or peak < 0.3:
        print("  [警告] 相関が弱い/境界張り付き。信頼度低。加減速のある走行で録り直すこと")
    else:
        print("  → pose_additional_delay_var の初期値候補。e_y スイープで微調整")
    return lag


def _load_ref_xy(ref_csv, ox, oy):
    """参照CSVから x,y 列を頑健に取得。区切り(,/;)・ヘッダ行を自動処理し、
    odom の座標中心に最も近い列を x,y として自動判定（s,x,y,... 形式に対応）。"""
    with open(ref_csv) as f:
        first = f.readline()
    delim = ";" if first.count(";") > first.count(",") else ","
    ref = np.genfromtxt(ref_csv, delimiter=delim, comments="#")
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    ref = ref[~np.isnan(ref).any(axis=1)]   # 非数値（ヘッダ）行を除去
    centers = ref.mean(axis=0)
    xcol = int(np.argmin(np.abs(centers - ox.mean())))
    ycol = int(np.argmin(np.abs(centers - oy.mean())))
    return ref[:, xcol], ref[:, ycol], xcol, ycol


def _plot_track(rx, ry, ox, oy, aey, ss, bins, order, binmax, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    vmax = float(np.percentile(aey, 99.5))

    # 左: コース上のトラジェクトリを |e_y| で着色
    axL.plot(rx, ry, color="0.78", lw=1.2, zorder=1, label="reference line")
    sc = axL.scatter(ox, oy, c=aey, cmap="turbo", s=5, vmin=0, vmax=vmax, zorder=2)
    cb = fig.colorbar(sc, ax=axL, shrink=0.85)
    cb.set_label("|e_y| [m]")
    w = int(np.argmax(aey))
    axL.scatter([ox[w]], [oy[w]], s=160, facecolors="none", edgecolors="red", lw=2.2, zorder=4)
    axL.annotate(f"max |e_y|={aey[w]:.2f}m (s={ss[w]:.0f}m)", (ox[w], oy[w]),
                 textcoords="offset points", xytext=(10, 10), color="red", fontsize=9, weight="bold")
    for b in order[:5]:
        sel = (ss >= bins[b]) & (ss < bins[b + 1])
        if not sel.any():
            continue
        k = int(np.where(sel)[0][np.argmax(aey[sel])])
        if k == w:
            continue
        axL.annotate(f"s≈{bins[b]:.0f}m\n{binmax[b]:.2f}m", (ox[k], oy[k]),
                     textcoords="offset points", xytext=(6, -14), fontsize=8, color="black")
    axL.set_aspect("equal")
    axL.set_title("track colored by |e_y|")
    axL.set_xlabel("x [m]"); axL.set_ylabel("y [m]")
    axL.legend(loc="best", fontsize=8)

    # 右: 弧長 s 対 |e_y|（全周回を重ね描き）＋ 区間max線
    axR.scatter(ss, aey, s=3, alpha=0.18, color="steelblue")
    centers = (bins[:-1] + bins[1:]) / 2
    axR.step(centers, binmax, where="mid", color="red", lw=1.4, label="bin max|e_y|")
    axR.set_xlabel("s [m] (arc length)"); axR.set_ylabel("|e_y| [m]")
    axR.set_title("|e_y| profile along the track")
    axR.grid(alpha=0.3)
    axR.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def compute_ey(odom_msgs, recv_ns, ref_csv, n_bins=36, top_n=15, dump=None, plot=None):
    ox = np.array([m.pose.pose.position.x for m in odom_msgs])
    oy = np.array([m.pose.pose.position.y for m in odom_msgs])
    rx, ry, xcol, ycol = _load_ref_xy(ref_csv, ox, oy)

    print("[e_y 追従誤差]")
    print(f"  [自動判定] 参照CSV x=col{xcol}, y=col{ycol}")
    print(f"  [診断] odom x:[{ox.min():.1f},{ox.max():.1f}] y:[{oy.min():.1f},{oy.max():.1f}]")
    print(f"  [診断] ref  x:[{rx.min():.1f},{rx.max():.1f}] y:[{ry.min():.1f},{ry.max():.1f}]")
    off_x = (rx.min() + rx.max()) / 2 - (ox.min() + ox.max()) / 2
    off_y = (ry.min() + ry.max()) / 2 - (oy.min() + oy.max()) / 2
    if abs(off_x) > 50 or abs(off_y) > 50:
        print(f"  [警告] 列自動判定後も原点ズレ（dx≈{off_x:.1f}, dy≈{off_y:.1f}）。フレーム不一致の可能性。")
        return None

    n = len(rx)
    # 参照ラインの弧長 s（周回コース）
    seg = np.hypot(np.diff(rx), np.diff(ry))
    s_wp = np.concatenate([[0.0], np.cumsum(seg)])
    track_len = s_wp[-1] + math.hypot(rx[0] - rx[-1], ry[0] - ry[-1])

    eys = np.empty(len(ox))
    ss = np.empty(len(ox))
    for k, (x, y) in enumerate(zip(ox, oy)):
        i = int(np.argmin((rx - x) ** 2 + (ry - y) ** 2))
        j = (i + 1) % n
        tx, ty = rx[j] - rx[i], ry[j] - ry[i]
        nrm = math.hypot(tx, ty) + 1e-9
        tx, ty = tx / nrm, ty / nrm
        eys[k] = -ty * (x - rx[i]) + tx * (y - ry[i])  # 左正の符号付き横偏差
        ss[k] = s_wp[i]
    aey = np.abs(eys)
    times = ((np.array(recv_ns, dtype=float) - recv_ns[0]) * 1e-9) if recv_ns else np.zeros(len(ox))

    print(f"  bias={eys.mean():+.3f}m  std={eys.std():.3f}m  max|e_y|={aey.max():.3f}m  (track≈{track_len:.0f}m)")

    # e_y 上位点の発生地点
    idx = np.argsort(aey)[::-1][:top_n]
    print(f"  [e_y 上位{top_n}点]   s[m]    e_y[m]    t[s]     (x, y)")
    for r in idx:
        print(f"    s={ss[r]:6.1f}   e_y={eys[r]:+.3f}   t={times[r]:6.1f}   ({ox[r]:.1f}, {oy[r]:.1f})")

    # 区間別 max|e_y|
    bins = np.linspace(0, track_len, n_bins + 1)
    which = np.clip(np.digitize(ss, bins) - 1, 0, n_bins - 1)
    binmax = np.zeros(n_bins)
    bincnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = aey[which == b]
        if len(sel):
            binmax[b] = sel.max()
            bincnt[b] = len(sel)
    order = np.argsort(binmax)[::-1][:8]
    ov_ey = np.array([m.twist.twist.linear.x for m in odom_msgs])
    print(f"  [区間別 max|e_y|（{n_bins}分割・上位8区間）]")
    for b in order:
        vsel = ov_ey[which == b]
        vavg = vsel.mean() * 3.6 if len(vsel) else 0.0
        print(f"    s={bins[b]:5.0f}-{bins[b+1]:5.0f}m  max={binmax[b]:.3f}m  n={bincnt[b]}  v_avg={vavg:.1f}km/h")

    # 系統的 / 分散 の判定（上位5%の外れが少数の区間=コーナーに集中するか）
    thr = np.percentile(aey, 95)
    big = which[aey >= thr]
    if len(big):
        counts = np.bincount(big, minlength=n_bins)
        topbins = np.argsort(counts)[::-1]
        top3 = counts[topbins[:3]].sum()
        frac3 = top3 / len(big)
        corner_s = ", ".join(f"s≈{(bins[b]+bins[b+1])/2:.0f}m" for b in topbins[:3] if counts[b] > 0)
        print(f"  [判定] 大きい外れ(上位5%)の {frac3*100:.0f}% が上位3区間（{corner_s}）に集中")
        if frac3 >= 0.5:
            print("         → 系統的（特定コーナーで毎周）。該当コーナーのライン/速度/先読みを対策")
        else:
            print("         → 分散（ランダム）。ジッタ/localizationスパイクの安定化を対策")

    if dump:
        import csv
        with open(dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "s_m", "x", "y", "e_y"])
            for k in range(len(ox)):
                w.writerow([f"{times[k]:.3f}", f"{ss[k]:.2f}", f"{ox[k]:.3f}", f"{oy[k]:.3f}", f"{eys[k]:.4f}"])
        print(f"  [dump] 全サンプルを {dump} に出力（プロット用: s vs e_y, x-y 着色）")

    if plot:
        try:
            _plot_track(rx, ry, ox, oy, aey, ss, bins, order, binmax, plot,
                        title=f"e_y on track  (max={aey.max():.2f}m, std={eys.std():.3f}m, track≈{track_len:.0f}m)")
            print(f"  [plot] コース着色図を {plot} に出力")
        except Exception as e:
            print(f"  [plot] 描画スキップ: {e}（matplotlib 未導入なら pip install matplotlib）")
    return eys


# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser(description="rosbag(mcap) 解析（ホスト/コンテナ両対応・ROS不要）")
    p.add_argument("bag", help="rosbag ディレクトリ または .mcap ファイル")
    p.add_argument("reference_csv", nargs="?", default=None, help="参照ライン CSV（任意）")
    p.add_argument("--dump-ey", default=None, help="全サンプル(t,s,x,y,e_y)をCSV出力（プロット用）")
    p.add_argument("--plot", default=None, help="コース着色図PNGの出力先（既定: <bag名>_ey.png を自動生成）")
    p.add_argument("--no-plot", action="store_true", help="PNG自動生成を無効化")
    p.add_argument("--list", action="store_true", help="bag内の全トピックを一覧表示して終了（障害物トピック発見用）")
    args = p.parse_args()

    if args.list:
        list_topics(args.bag)
        return

    msgs, recv = read_bag(args.bag, [TOPIC_CMD, TOPIC_ODOM, TOPIC_VEL, TOPIC_GNSS, TOPIC_V2X, TOPIC_COND, TOPIC_STEER, TOPIC_GT_COLLISION])
    cmd, odom = msgs[TOPIC_CMD], msgs[TOPIC_ODOM]
    if not cmd or not odom:
        print("必要なトピック（control_cmd / kinematic_state）が見つかりません。")
        sys.exit(1)

    report_jitter(recv[TOPIC_CMD])
    report_speed(cmd, odom)
    report_stops(odom, recv[TOPIC_ODOM])
    report_collisions(msgs[TOPIC_COND], recv[TOPIC_COND], odom, recv[TOPIC_ODOM], args.reference_csv)
    report_steer(cmd, odom)
    report_steer_saturation(cmd, msgs[TOPIC_STEER], odom, recv[TOPIC_ODOM], args.reference_csv)
    report_collision_gt(msgs[TOPIC_GT_COLLISION], recv[TOPIC_GT_COLLISION], odom, recv[TOPIC_ODOM], args.reference_csv)
    if msgs[TOPIC_V2X]:
        report_obstacle_distance(odom, recv[TOPIC_ODOM], msgs[TOPIC_V2X], recv[TOPIC_V2X])
    estimate_speed_lag(cmd, odom)
    if msgs[TOPIC_VEL] and msgs[TOPIC_GNSS]:
        estimate_gnss_delay(msgs[TOPIC_VEL], msgs[TOPIC_GNSS])
    else:
        print("[GNSS 実効遅延] velocity_status / gnss pose 未収録のためスキップ")
    if args.reference_csv:
        # 参照CSVがあれば、PNGを既定で自動生成（--no-plot で無効化）
        plot_path = None
        if not args.no_plot:
            bagname = Path(args.bag.rstrip("/")).stem
            plot_path = args.plot or str(Path.cwd() / f"{bagname}_ey.png")
        compute_ey(odom, recv[TOPIC_ODOM], args.reference_csv,
                   dump=args.dump_ey, plot=plot_path)


if __name__ == "__main__":
    main()
