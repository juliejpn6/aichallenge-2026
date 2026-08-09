"""
wp349-18(実質wp342-25の3コブ:wp346/wp3/wp16-17)を、頂点wp6付近を持つ
1つの滑らかなコーナーへ統合する(2026-08-10、ユーザー提案)。

traj_mincurv_straightened.csv(wp135-166直線化済み)をベースに、Laplacian平滑化
(kaleidoscope smooth_all_pointsと同アルゴリズム)を該当区間へ複数pass適用し、
自然な平坦点(wp341/wp342境界、wp25付近)でブレンドして元経路へ接続する。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import read_traj, write_traj, recompute_geometry  # noqa: E402

KALEIDO_DIR = (
    "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/"
    "tools/kaleidoscope"
)
sys.path.insert(0, KALEIDO_DIR)
from kaleidoscope import trajectory_clearance as tc  # noqa: E402

BASE_DIR = "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
TRAJ_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv_straightened.csv"
MAP_YAML = f"{BASE_DIR}/env/final_ver3/occupancy_grid_map.yaml"
OUT_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv_straightened.csv"

# 統合対象コア(ユーザー指定): wp349-18。ブレンド用バッファを342-25まで拡張
# (いずれも元kappaがほぼゼロに近い自然な平坦点)。
CORE_LO, CORE_HI = 349, 18
# 2026-08-10訂正: BUF_HI=25では終端ブレンドの助走が短すぎて、wp25-40帯
# (元々kappa≈0で自然に平坦、ユーザー指摘「ハンドル操作不要」と一致)で
# 逆に暴れが発生していた。ブレンド終端を、元kappaが十分小さい自然な点wp40まで延長。
BUF_LO, BUF_HI = 338, 40


def _raised_cosine(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def kap(xs, ys, i, n):
    im, ip = (i - 1) % n, (i + 1) % n
    x0, y0 = xs[im], ys[im]
    x1, y1 = xs[i % n], ys[i % n]
    x2, y2 = xs[ip], ys[ip]
    a = math.hypot(x1 - x0, y1 - y0)
    b = math.hypot(x2 - x1, y2 - y1)
    c = math.hypot(x2 - x0, y2 - y0)
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    denom = a * b * c
    return 2.0 * cross / denom if denom > 1e-9 else 0.0


def consolidate(xs, ys, n, alpha, passes):
    idxs = list(range(BUF_LO, n)) + list(range(0, BUF_HI + 1))
    m = len(idxs)
    core_lo_pos = idxs.index(CORE_LO)
    core_hi_pos = idxs.index(CORE_HI)

    smoothed = {i: (xs[i], ys[i]) for i in idxs}
    for _ in range(passes):
        src = dict(smoothed)
        for k, i in enumerate(idxs):
            if k == 0 or k == m - 1:
                continue
            ip, im = idxs[k + 1], idxs[k - 1]
            avg_x = 0.5 * (src[ip][0] + src[im][0])
            avg_y = 0.5 * (src[ip][1] + src[im][1])
            sx, sy = src[i]
            smoothed[i] = ((1 - alpha) * sx + alpha * avg_x, (1 - alpha) * sy + alpha * avg_y)

    xs2, ys2 = list(xs), list(ys)
    for k, i in enumerate(idxs):
        # core内は完全にsmoothed、buffer内はraised-cosineでブレンド
        if core_lo_pos <= k <= core_hi_pos:
            w = 1.0
        elif k < core_lo_pos:
            w = _raised_cosine(k / core_lo_pos) if core_lo_pos > 0 else 1.0
        else:
            w = _raised_cosine((m - 1 - k) / (m - 1 - core_hi_pos)) if (m - 1 - core_hi_pos) > 0 else 1.0
        sx, sy = smoothed[i]
        xs2[i] = (1 - w) * xs[i] + w * sx
        ys2[i] = (1 - w) * ys[i] + w * sy
    return xs2, ys2, idxs


def main():
    rows = read_traj(TRAJ_PATH)
    n = len(rows)
    xs0 = [r["x_m"] for r in rows]
    ys0 = [r["y_m"] for r in rows]

    grid = tc.load_occupancy_grid(MAP_YAML)
    vehicle = tc.VehicleFootprintSpec(
        reference_point="rear_axle", wheel_base_m=1.087, front_overhang_m=0.467,
        rear_overhang_m=0.510, wheel_tread_m=1.12, left_overhang_m=0.09,
        right_overhang_m=0.09, margin_left_m=0.5, margin_right_m=0.5,
        margin_front_m=0.2, margin_rear_m=0.2,
    )

    accepted = None
    for alpha, passes in [(0.25, 10), (0.2, 15), (0.15, 20), (0.1, 25), (0.08, 15)]:
        xs2, ys2, idxs = consolidate(xs0, ys0, n, alpha, passes)
        poses = []
        for i in range(n):
            ip, im = (i + 1) % n, (i - 1) % n
            yaw = math.atan2(ys2[ip] - ys2[im], xs2[ip] - xs2[im])
            poses.append(tc.Pose2D(x_m=xs2[i], y_m=ys2[i], yaw_rad=yaw))
        report = tc.validate_clearance(grid, poses, vehicle, tc.ValidationOptions(circular=True))
        # 単純な曲率悪化チェック(peak count確認込み)
        kappas = [kap(xs2, ys2, i, n) for i in idxs]
        peak_i = min(range(len(kappas)), key=lambda k: kappas[k])
        peak_wp = idxs[peak_i]
        print(f"alpha={alpha} passes={passes}: is_safe={report.is_safe} "
              f"min_clearance={report.minimum_clearance_m} peak_kappa_wp={peak_wp} peak_kappa={kappas[peak_i]:.4f}")
        if report.is_safe:
            accepted = (alpha, passes, xs2, ys2)
            break

    if not accepted:
        print("!!! 安全な統合候補が見つかりませんでした")
        return

    alpha, passes, xs2, ys2 = accepted
    print(f"\n採用: alpha={alpha}, passes={passes}")

    print("\n=== kappaプロファイル(統合前 -> 後) ===")
    for i in list(range(342, 350)) + list(range(0, 26)):
        kb = kap(xs0, ys0, i, n)
        ka = kap(xs2, ys2, i, n)
        print(f"wp{i}: {kb:+.4f} -> {ka:+.4f}")

    out_rows = [dict(r) for r in rows]
    for i, r in enumerate(out_rows):
        r["x_m"] = xs2[i]
        r["y_m"] = ys2[i]
    recompute_geometry(out_rows, closed=True)
    write_traj(OUT_PATH, out_rows)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
