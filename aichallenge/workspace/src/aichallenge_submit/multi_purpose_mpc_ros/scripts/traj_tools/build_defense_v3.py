"""
25km/hディフェンストラジェクトリ生成、v3(2026-08-10)。

v2からの変更点(経緯: v2実測でwp63/88/192/196/199/211/280/306の8箇所に
新規スパイクを発見。うち6箇所は隣接するGENTLE_CORNERS同士の隙間[2-4pt]、
2箇所は単一コーナー境界[taperなしの生バイアスが急に始まる/終わる点]。
v2は「taperなしの生バイアス+全体への軽いLaplacian平滑化で不連続を均す」
設計だったが、平滑化(alpha0.15-0.25・6-10pass)だけでは吸収しきれない
段差が系統的に残ることが判明した):

  1. 各ゆるいコーナーの内部で、バイアス振幅をsmoothstep(3t^2-2t^3)で
     0→最大→0とテーパーする(taper幅はコーナー長の25%、最小2pt)。
     smoothstepは両端で導関数=0なので、コーナー境界で原ジオメトリと
     位置・接線方向がなめらかに一致し、隣接コーナーとの隙間が数ptしか
     なくても不連続を作らない。
  2. 全体へのLaplacian平滑化は「仕上げ」として軽く残す(段差除去の主役
     ではなく微細ノイズ除去用、v2よりalpha/passesを控えめに)。
  3. kaleidoscope.trajectory_clearanceでの実マップ検証はv2と同じ。

taper窓を使った過去の失敗(タイトコーナー拡幅でのraised-cosine taper、
3回不採用)はいずれも「壁に極めて近いタイトコーナーの半径そのものを拡大する」
操作だった。今回は半径拡大ではなくゆるいコーナー内側への小さな平行移動
(1.02m基準)であり、taper境界での曲率悪化のリスク構造が異なる
(taper区間内で目標カーブが原カーブから大きく乖離しないため)。
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
DEFENSE_OUT = f"{BASE_DIR}/env/final_ver3/traj_defense_25kmh.csv"

INSIDE_GAP_M = 1.02

TIGHT_CORNERS = [(224, 230), (247, 257)]
GENTLE_CORNERS = [
    (1, 9, "R"), (12, 21, "R"),
    (64, 87, "R"), (109, 132, "L"),
    (167, 190, "R"), (193, 195, "R"),
    (215, 222, "R"),
    (232, 239, "R"), (242, 242, "L"),
    (259, 276, "L"), (281, 305, "R"), (307, 319, "R"),
    (327, 340, "L"), (343, 347, "R"),
]
# 2026-08-10(v3): (200,210,"L")は原本ピーク曲率0.146(R≈6.8m)と「ゆるい」というより
# タイトな部類で、内側へのオフセットカーブが曲率を増幅する性質(offset curve理論、
# R-dで曲率1/(R-d)に増加)と相性が悪く、taper長を伸ばしても境界(wp201/209)で
# 原本ピークを超える曲率(0.20/0.18 vs 原本最大0.146)が残った。無理に均そうとする
# より除外する方が安全という判断(protected_points()同様の除外方針)。


def protected_points():
    pts = set()
    for lo, hi in TIGHT_CORNERS:
        for i in range(lo - 1, hi + 2):
            pts.add(i)
    for i in list(range(320, 350)) + list(range(0, 10)):
        pts.add(i % 350)
    for i in range(133, 168):
        pts.add(i)
    for i in range(18, 48):
        pts.add(i)
    return pts


def normals(xs, ys, n):
    nrm = []
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        dx, dy = xs[ip] - xs[im], ys[ip] - ys[im]
        norm = math.hypot(dx, dy)
        nrm.append((-dy / norm, dx / norm) if norm > 1e-9 else (0.0, 0.0))
    return nrm


MIN_GAP_TO_PROTECTED = 6


def _corner_too_close(lo, hi, protected, n):
    for i in range(lo, hi + 1):
        for d in range(1, MIN_GAP_TO_PROTECTED + 1):
            if (i - d) % n in protected or (i + d) % n in protected:
                return True
    return False


def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _taper_factor(i, lo, hi, taper_len):
    """コーナー内での位置iにおけるバイアス係数[0,1]。
    両端からtaper_len点でsmoothstepにより0→1→0とテーパーする。"""
    from_start = i - lo
    from_end = hi - i
    edge_dist = min(from_start, from_end)
    if edge_dist >= taper_len:
        return 1.0
    return _smoothstep(edge_dist / taper_len)


def apply_tapered_bias(xs, ys, n, protected):
    xs, ys = list(xs), list(ys)
    nrm = normals(xs, ys, n)
    skipped = []
    for lo, hi, side in GENTLE_CORNERS:
        if _corner_too_close(lo, hi, protected, n):
            skipped.append((lo, hi))
            continue
        sign = 1.0 if side == "L" else -1.0
        length = hi - lo + 1
        taper_len = max(4, length // 3)
        if 2 * taper_len >= length:
            # コーナーが短すぎてtaperの余地(core区間)が確保できない場合は
            # 中途半端に段差を作るよりバイアス自体を見送る(2026-08-10発見:
            # taper=2ptでは1点あたり0.28m級の急変化が生じ新規スパイクの原因になった)。
            skipped.append((lo, hi, "短すぎ"))
            continue
        for i in range(lo, hi + 1):
            if i in protected:
                continue
            factor = _taper_factor(i, lo, hi, taper_len)
            nx, ny = nrm[i]
            xs[i] += nx * INSIDE_GAP_M * sign * factor
            ys[i] += ny * INSIDE_GAP_M * sign * factor
    if skipped:
        print(f"  [skip: タイトコーナーに近接しすぎ] {skipped}")
    return xs, ys


def laplacian_smooth(xs, ys, n, alpha, passes, protected):
    xs, ys = list(xs), list(ys)
    for _ in range(passes):
        sx, sy = list(xs), list(ys)
        for i in range(n):
            if i in protected:
                continue
            ip, im = (i + 1) % n, (i - 1) % n
            avg_x = 0.5 * (xs[ip] + xs[im])
            avg_y = 0.5 * (ys[ip] + ys[im])
            sx[i] = (1 - alpha) * xs[i] + alpha * avg_x
            sy[i] = (1 - alpha) * ys[i] + alpha * avg_y
        xs, ys = sx, sy
    return xs, ys


def to_poses(xs, ys, n):
    poses = []
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        dx, dy = xs[ip] - xs[im], ys[ip] - ys[im]
        yaw = math.atan2(dy, dx)
        poses.append(tc.Pose2D(x_m=xs[i], y_m=ys[i], yaw_rad=yaw))
    return poses


def find_spikes(rows, n, thresh_abs=0.03, ratio=1.6):
    spikes = []
    for i in range(n):
        ip, im = (i + 1) % n, (i - 1) % n
        k = abs(rows[i]["kappa_radpm"])
        kp, km = abs(rows[ip]["kappa_radpm"]), abs(rows[im]["kappa_radpm"])
        avg_neighbor = (kp + km) / 2
        if k > thresh_abs and avg_neighbor > 1e-6 and k > avg_neighbor * ratio:
            spikes.append((i, k, avg_neighbor))
    return spikes


def main():
    rows = read_traj(TRAJ_PATH)
    n = len(rows)
    xs0 = [r["x_m"] for r in rows]
    ys0 = [r["y_m"] for r in rows]
    protected = protected_points()

    print("=== occupancy grid / vehicle footprint 読み込み ===")
    grid = tc.load_occupancy_grid(MAP_YAML)
    vehicle = tc.VehicleFootprintSpec(
        reference_point="rear_axle",
        wheel_base_m=1.087,
        front_overhang_m=0.467,
        rear_overhang_m=0.510,
        wheel_tread_m=1.12,
        left_overhang_m=0.09,
        right_overhang_m=0.09,
        margin_left_m=0.5,
        margin_right_m=0.5,
        margin_front_m=0.2,
        margin_rear_m=0.2,
    )
    print(f"grid: {grid.width}x{grid.height} @ {grid.spec.resolution_m}m/px")

    base_poses = to_poses(xs0, ys0, n)
    base_report = tc.validate_clearance(
        grid, base_poses, vehicle, tc.ValidationOptions(circular=True)
    )
    print(f"baseline(元ジオメトリ) is_safe={base_report.is_safe}, "
          f"min_clearance={base_report.minimum_clearance_m}")

    # 元ジオメトリ自体が持つ自然なスパイク(急なコーナー頂点等、欠陥ではない)を
    # 基準として控除する。「新規に増えたスパイク」だけを合否判定に使う。
    base_rows_for_spike = [dict(r) for r in rows]
    for i, r in enumerate(base_rows_for_spike):
        r["x_m"] = xs0[i]
        r["y_m"] = ys0[i]
    recompute_geometry(base_rows_for_spike, closed=True)
    baseline_spike_wps = {s[0] for s in find_spikes(base_rows_for_spike, n)}
    print(f"baseline由来スパイク(欠陥ではない、控除対象): {sorted(baseline_spike_wps)}")

    amplitudes = [1.0, 0.85, 0.7, 0.55, 0.4]
    # taperが境界連続性を担うため、平滑化なし(0,0)を最優先で試す。
    # 平滑化パスは逆に「テーパー済み境界」と「未着手の隣接点」を混ぜて
    # 新たな段差を作りうることが実測で判明したため、あくまで保険として残す。
    smooth_configs = [(0.0, 0), (0.10, 2), (0.10, 3)]
    accepted = None
    for amp in amplitudes:
        global INSIDE_GAP_M
        saved_gap = INSIDE_GAP_M
        INSIDE_GAP_M = saved_gap * amp
        xs1, ys1 = apply_tapered_bias(xs0, ys0, n, protected)
        INSIDE_GAP_M = saved_gap
        for alpha, passes in smooth_configs:
            xs2, ys2 = laplacian_smooth(xs1, ys1, n, alpha, passes, protected)
            poses = to_poses(xs2, ys2, n)
            report = tc.validate_clearance(
                grid, poses, vehicle, tc.ValidationOptions(circular=True)
            )
            # スパイク検査も合否条件に加える(v2はclearanceのみでスパイク見逃し)
            tmp_rows = [dict(r) for r in rows]
            for i, r in enumerate(tmp_rows):
                r["x_m"] = xs2[i]
                r["y_m"] = ys2[i]
            recompute_geometry(tmp_rows, closed=True)
            all_spikes = find_spikes(tmp_rows, n)
            new_spikes = [s for s in all_spikes if s[0] not in baseline_spike_wps]
            print(f"  amp={amp:.2f} alpha={alpha} passes={passes}: "
                  f"is_safe={report.is_safe} min_clearance={report.minimum_clearance_m} "
                  f"新規spikes={len(new_spikes)} {[s[0] for s in new_spikes]}")
            if report.is_safe and not new_spikes:
                accepted = (amp, alpha, passes, xs2, ys2, report)
                break
        if accepted:
            break

    if not accepted:
        print("\n!!! 安全かつスパイクなしの候補が見つかりませんでした。defenseファイルは生成しません。")
        return

    amp, alpha, passes, xs2, ys2, report = accepted
    print(f"\n採用: amp={amp:.2f}, alpha={alpha}, passes={passes}, "
          f"min_clearance={report.minimum_clearance_m:.2f}m")

    max_protected_disp = max(
        math.hypot(xs2[i] - xs0[i], ys2[i] - ys0[i]) for i in protected
    )
    print(f"タイトコーナーcore最大変位(0であるべき): {max_protected_disp:.6f}m")

    defense_rows = [dict(r) for r in rows]
    for i, r in enumerate(defense_rows):
        r["x_m"] = xs2[i]
        r["y_m"] = ys2[i]
    recompute_geometry(defense_rows, closed=True)
    write_traj(DEFENSE_OUT, defense_rows)
    print(f"wrote {DEFENSE_OUT}")


if __name__ == "__main__":
    main()
