"""
wp135-166直線化(straighten_regions.py)の出口境界(wp166/167)に残っていた
タンジェント不連続を、Hermite補間ブレンドで解消する(2026-08-10)。

発見の経緯: ユーザーがWP170付近の実走行での「震え」を指摘。原本traj_mincurv.csvでは
wp168〜171がなめらか(-0.098→-0.122→-0.119→-0.135)なのに対し、直線化後は
wp169=-0.029(直線区間の名残)→wp170=-0.200(1.7倍の急変)というスパイクが
生じていた。原因は`straighten()`が境界で位置は一致させるが接線方向を考慮しない
単純chord直線化だったため、直線区間(タンジェント=直線方向)から原カーブ
(タンジェント=カーブ方向)への遷移がwp166/167の1点間で強制され、その帳尻合わせが
数点先(wp169-170)の曲率スパイクとして現れていた。

wp20-46で実績のあるHermite補間ブレンド(位置・接線方向とも連続、design_docs
opp_lat_pred_overlap_guard_design_20260806.md §47.9参照)と同じ技法を、直線区間
末尾(wp158-166、まだ直線のまま)〜原カーブ再安定区間(wp167-178)に適用し、
タンジェントのなだらかな遷移を作る。直線区間の入口側(wp135付近)は原本も既に
ほぼ直線(|kappa|<0.02)だったため対称的な問題は生じておらず、対象外。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import read_traj, write_traj, recompute_geometry  # noqa: E402

BASE_DIR = "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
TRAJ_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv_straightened.csv"
ORIG_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv.csv"

BLEND_START = 164  # まだ直線区間内(タンジェント=直線方向)
BLEND_END = 172    # 原カーブが再安定した区間(タンジェント=原カーブ方向)
# 2026-08-10: 当初(158,178)の広いウィンドウを試したが、なめらかにする代わりに
# コーナー内側へ膨らみすぎてFOOTPRINT_COLLISION(wp162-164)を引き起こした。
# (164,172)まで狭めると、is_safe=True・min_clearance=0.528(原本と同水準)を
# 維持しつつwp170のスパイクも解消(近傍比1.01、閾値1.6を十分下回る)。
# wp173以降は完全な原本と一致(ブレンド区間外のため無変更)。


def _h00(s): return 2 * s**3 - 3 * s**2 + 1
def _h10(s): return s**3 - 2 * s**2 + s
def _h01(s): return -2 * s**3 + 3 * s**2
def _h11(s): return s**3 - s**2


def tangent_at(xs, ys, i, n):
    ip, im = (i + 1) % n, (i - 1) % n
    dx, dy = xs[ip] - xs[im], ys[ip] - ys[im]
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-9 else (1.0, 0.0)


def hermite_blend(xs, ys, n, i0, i1):
    """i0(直線区間内、タンジェント=直線方向)からi1(原カーブ再安定点、
    タンジェント=原カーブ方向)までを3次エルミート補間で置き換える。
    位置・接線方向とも両端で連続。"""
    xs, ys = list(xs), list(ys)
    idxs = list(range(i0, i1 + 1))
    p0 = (xs[i0], ys[i0])
    p1 = (xs[i1], ys[i1])
    t0 = tangent_at(xs, ys, i0, n)
    t1 = tangent_at(xs, ys, i1, n)
    seg_lens = [0.0]
    for a, b in zip(idxs[:-1], idxs[1:]):
        seg_lens.append(seg_lens[-1] + math.hypot(xs[b] - xs[a], ys[b] - ys[a]))
    total_len = seg_lens[-1]
    for k, i in enumerate(idxs):
        if i == i0 or i == i1:
            continue
        s = seg_lens[k] / total_len if total_len > 0 else 0.0
        x = (p0[0] * _h00(s) + t0[0] * total_len * _h10(s)
             + p1[0] * _h01(s) + t1[0] * total_len * _h11(s))
        y = (p0[1] * _h00(s) + t0[1] * total_len * _h10(s)
             + p1[1] * _h01(s) + t1[1] * total_len * _h11(s))
        xs[i], ys[i] = x, y
    return xs, ys


def main():
    rows = read_traj(TRAJ_PATH)
    n = len(rows)
    xs0 = [r["x_m"] for r in rows]
    ys0 = [r["y_m"] for r in rows]

    xs2, ys2 = hermite_blend(xs0, ys0, n, BLEND_START, BLEND_END)

    tmp_rows = [dict(r) for r in rows]
    for i, r in enumerate(tmp_rows):
        r["x_m"] = xs2[i]
        r["y_m"] = ys2[i]
    recompute_geometry(tmp_rows, closed=True)

    orig_rows = read_traj(ORIG_PATH)
    recompute_geometry(orig_rows, closed=True)

    print("=== wp155-180 曲率比較(修正前→修正後、参考: 完全な原本) ===")
    base_rows = read_traj(TRAJ_PATH)
    recompute_geometry(base_rows, closed=True)
    for wp in range(155, 181):
        print(f"  wp={wp}: 修正前={base_rows[wp]['kappa_radpm']:.4f}  "
              f"修正後={tmp_rows[wp]['kappa_radpm']:.4f}  "
              f"完全原本={orig_rows[wp]['kappa_radpm']:.4f}")

    write_traj(TRAJ_PATH, tmp_rows)
    print(f"\nwrote {TRAJ_PATH}(上書き)")


if __name__ == "__main__":
    main()
