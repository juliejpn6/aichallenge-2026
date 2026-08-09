"""
ユーザー指示による直線化(2026-08-10):
  - wp330(コーナー出口)からwp5(コーナー入口)まで(S字wp340-40帯)
  - wp130付近からwp170付近まで

各区間をchord(直線)へ置き換えた新しいベースジオメトリを生成し、
kaleidoscope.trajectory_clearanceで実マップ+実車体寸法に対し安全性を検証する。
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
TRAJ_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv.csv"
MAP_YAML = f"{BASE_DIR}/env/final_ver3/occupancy_grid_map.yaml"
OUT_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv_straightened.csv"

REGIONS = [
    # 2026-08-10: S字wp325-9はwp0-5・329-342・347-349の広範囲で実壁衝突を確認
    # (単純な境界ブレンド不連続ではなく、区間内の広い範囲でクリアランス不足)。
    # このトラック区間は実際に湾曲が必要と判断し、直線化を見送り原形維持。
    # (325, 9, "S字wp325-9(旧wp340-40帯、平坦点採用)"),
    (135, 166, "wp135-166(平坦点採用)"),
]


def _raised_cosine(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def straighten_blended(xs, ys, n, start, end, blend_pts=6):
    """start,endの手前blend_pts点は元経路と直線chordのブレンド(raised-cosine)、
    中間は完全な直線。境界でのヘディング急変(=衝突の原因)を避ける。"""
    idxs = list(range(start, n)) + list(range(0, end + 1)) if end < start else list(range(start, end + 1))
    x0, y0 = xs[idxs[0]], ys[idxs[0]]
    x1, y1 = xs[idxs[-1]], ys[idxs[-1]]
    seg_lens = [0.0]
    for a, b in zip(idxs[:-1], idxs[1:]):
        seg_lens.append(seg_lens[-1] + math.hypot(xs[b] - xs[a], ys[b] - ys[a]))
    total = seg_lens[-1]
    xs2, ys2 = list(xs), list(ys)
    m = len(idxs)
    for k, i in enumerate(idxs):
        t = seg_lens[k] / total if total > 0 else 0.0
        line_x, line_y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        # 両端からのブレンド重み(0=元経路のまま, 1=直線)
        w_start = min(1.0, k / blend_pts) if blend_pts > 0 else 1.0
        w_end = min(1.0, (m - 1 - k) / blend_pts) if blend_pts > 0 else 1.0
        w = _raised_cosine(min(w_start, w_end))
        xs2[i] = (1 - w) * xs[i] + w * line_x
        ys2[i] = (1 - w) * ys[i] + w * line_y
    xs2[idxs[0]], ys2[idxs[0]] = x0, y0
    xs2[idxs[-1]], ys2[idxs[-1]] = x1, y1
    return xs2, ys2, idxs


def straighten(xs, ys, n, start, end):
    idxs = list(range(start, n)) + list(range(0, end + 1)) if end < start else list(range(start, end + 1))
    x0, y0 = xs[idxs[0]], ys[idxs[0]]
    x1, y1 = xs[idxs[-1]], ys[idxs[-1]]
    seg_lens = [0.0]
    for a, b in zip(idxs[:-1], idxs[1:]):
        seg_lens.append(seg_lens[-1] + math.hypot(xs[b] - xs[a], ys[b] - ys[a]))
    total = seg_lens[-1]
    xs2, ys2 = list(xs), list(ys)
    for k, i in enumerate(idxs):
        t = seg_lens[k] / total if total > 0 else 0.0
        xs2[i] = x0 + t * (x1 - x0)
        ys2[i] = y0 + t * (y1 - y0)
    xs2[idxs[0]], ys2[idxs[0]] = x0, y0
    xs2[idxs[-1]], ys2[idxs[-1]] = x1, y1
    return xs2, ys2, idxs


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


def main():
    rows = read_traj(TRAJ_PATH)
    n = len(rows)
    xs0 = [r["x_m"] for r in rows]
    ys0 = [r["y_m"] for r in rows]

    xs, ys = list(xs0), list(ys0)
    all_idxs = []
    for start, end, label in REGIONS:
        # 2026-08-10訂正: straighten_blended()はブレンドの実装ミスで区間"内部"
        # (wp160-165)まで元のカーブへ引き戻してしまい、逆に大きなキンク(kappa最大0.33)
        # を作っていた(区間の途中で「まっすぐ」と「元のコーナー形状」を混ぜてしまう
        # バグ)。region2(wp135-166)はそもそも境界衝突が無かった区間なので、
        # ブレンド無しの単純な直線化(straighten)へ戻す。
        xs, ys, idxs = straighten(xs, ys, n, start, end)
        all_idxs.append((idxs, label))
        length = sum(
            math.hypot(xs[b] - xs[a], ys[b] - ys[a])
            for a, b in zip(idxs[:-1], idxs[1:])
        )
        print(f"[{label}] wp{idxs[0]}->wp{idxs[-1]} を直線化(区間長{length:.2f}m)")

    print("\n=== 境界+区間内の曲率チェック ===")
    worst = 0.0
    worst_i = None
    for i in range(n):
        kb, ka = kap(xs0, ys0, i, n), kap(xs, ys, i, n)
        if abs(ka) - abs(kb) > worst:
            worst = abs(ka) - abs(kb)
            worst_i = i
    print(f"最大曲率悪化: {worst:.4f} rad/m @ wp{worst_i}")
    maxb = max(abs(kap(xs0, ys0, i, n)) for i in range(n))
    maxa = max(abs(kap(xs, ys, i, n)) for i in range(n))
    print(f"max|kappa| 直線化前={maxb:.4f} 直線化後={maxa:.4f}")

    print("\n=== kaleidoscope clearance検証(実マップ+実車体寸法) ===")
    grid = tc.load_occupancy_grid(MAP_YAML)
    vehicle = tc.VehicleFootprintSpec(
        reference_point="rear_axle",
        wheel_base_m=1.087, front_overhang_m=0.467, rear_overhang_m=0.510,
        wheel_tread_m=1.12, left_overhang_m=0.09, right_overhang_m=0.09,
        margin_left_m=0.5, margin_right_m=0.5, margin_front_m=0.2, margin_rear_m=0.2,
    )
    poses = []
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        dx, dy = xs[ip] - xs[im], ys[ip] - ys[im]
        yaw = math.atan2(dy, dx)
        poses.append(tc.Pose2D(x_m=xs[i], y_m=ys[i], yaw_rad=yaw))
    report = tc.validate_clearance(grid, poses, vehicle, tc.ValidationOptions(circular=True))
    print(f"is_safe={report.is_safe}, min_clearance={report.minimum_clearance_m}")
    if not report.is_safe:
        print("issues:")
        for issue in report.issues[:10]:
            print(f"  {issue}")

    out_rows = [dict(r) for r in rows]
    for i, r in enumerate(out_rows):
        r["x_m"] = xs[i]
        r["y_m"] = ys[i]
    recompute_geometry(out_rows, closed=True)
    write_traj(OUT_PATH, out_rows)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
