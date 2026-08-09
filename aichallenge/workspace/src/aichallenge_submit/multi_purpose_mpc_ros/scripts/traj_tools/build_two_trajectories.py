"""
25km/hディフェンス・35km/hオフェンスの2トラジェクトリを生成する。

【重要な前提(2026-08-09、mpc_controller.py/reference_path.py/utils.py実装確認済み)】
  utils.load_ref_path()はpsi_rad/kappa_radpm列も返すが、呼び出し側
  (mpc_controller.py:532, path_constraints_provider.py:126)は
  `wp_x, wp_y, _, _ = load_ref_path(...)` と受け取り、psi/kappaを直ちに破棄している。
  ReferencePathクラスはx,yのみから自前でリサンプリング・平滑化・psi/kappa再計算・
  速度プロファイル計算(compute_speed_profile)を行う。
  → traj_mincurv.csv系ファイルで実際に効くのは x_m, y_m 列のみ。
    s_m/psi_rad/kappa_radpm/vx_mps/ax_mps2 は非使用(外部ツール向けの体裁のみ)。
  よって本スクリプトはx_m,y_mの編集に専念し、他列はgeometry.pyで整合性のため
  再計算するだけでよい。

出力(いずれも新規ファイル、既存traj_mincurv.csvは無変更):
  - env/final_ver3/traj_offense_35kmh.csv : 現行ジオメトリをそのままコピー
    (vx_mpsプロファイルが既にay_max~12/直線66km/h相当で設計されており、
    現状のmincurv線が概ね曲率最小と判断。速度自体はconfig.yaml側のv_max/ay_max/
    ay_profileで引き出す)
  - env/final_ver3/traj_defense_25kmh.csv : 以下2種の編集を適用
    (a) タイトコーナー2箇所(wp224-230, wp247-257)を、実測トラック幅
        (aic_2024.csv、拡幅の余地を確認済み)の範囲内で平滑化し曲率を緩和
    (b) ゆるいコーナー17区間で、進行方向イン側へ実車幅0.7台分(1.02m)の
        ギャップを残すディフェンスバイアスを付与(方向[L/R]考慮、
        トラック幅からの安全マージン1.2m以上を確保)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import read_traj, write_traj, recompute_geometry, load_track_bounds  # noqa: E402

BASE_DIR = "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
TRAJ_PATH = f"{BASE_DIR}/env/final_ver3/traj_mincurv.csv"
TRACK_BOUNDS_PATH = "/home/yoshihito/global_racetrajectory_optimization/inputs/tracks/aic_2024.csv"

OFFENSE_OUT = f"{BASE_DIR}/env/final_ver3/traj_offense_35kmh.csv"
DEFENSE_OUT = f"{BASE_DIR}/env/final_ver3/traj_defense_25kmh.csv"

AY_MAX = 12.0
V25 = 25.0 / 3.6
SAFETY_MARGIN_M = 1.2  # トラック境界からの最低クリアランス(corridor safety_margin~1.15mよりやや広めに取る)
INSIDE_GAP_M = 1.02    # 実車幅1.45m x 0.7台分

# 2026-08-09解析(design_docs記録予定)で特定した2タイトコーナー(index=raw CSV行=wp_id)
TIGHT_CORNERS = [
    {"name": "wp224-230", "core": (224, 230), "window": (204, 250)},
    {"name": "wp247-257", "core": (247, 257), "window": (227, 277)},
]

# 「ゆるいコーナー」17区間(|kappa|>0.05、タイト2区間を除く)。dirはインバイアス方向の目安
# (L=左カーブ=インは進行方向左=法線+側, R=右カーブ=インは進行方向右=法線-側)
GENTLE_CORNERS = [
    (1, 9, "R"), (12, 21, "R"), (34, 35, "R"),
    (64, 87, "R"), (109, 132, "L"),
    (158, 164, "L"), (167, 190, "R"), (193, 195, "R"),
    (200, 210, "L"), (215, 222, "R"),
    (232, 239, "R"), (242, 242, "L"),
    (259, 276, "L"), (281, 305, "R"), (307, 319, "R"),
    (327, 340, "L"), (343, 347, "R"),
]


def dist_to_bound(pt, track_bounds):
    bx, by, br, bl = min(track_bounds, key=lambda t: (t[0] - pt[0]) ** 2 + (t[1] - pt[1]) ** 2)
    return br, bl  # 右境界までの距離, 左境界までの距離 (どちらも中心線からの正の距離)


def normals(xs, ys, n, closed=True):
    """各点での進行方向左向き法線ベクトル(単位ベクトル)を返す。"""
    nrm = []
    for i in range(n):
        if closed:
            ip, inx = (i - 1) % n, (i + 1) % n
        else:
            ip, inx = max(i - 1, 0), min(i + 1, n - 1)
        dx, dy = xs[inx] - xs[ip], ys[inx] - ys[ip]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            nrm.append((0.0, 0.0))
            continue
        tx, ty = dx / norm, dy / norm
        # 進行方向に対して左向き法線 = (-ty, tx)
        nrm.append((-ty, tx))
    return nrm


def raised_cosine_taper(i, lo, core_lo, core_hi, hi):
    """[lo,core_lo]で0→1、[core_lo,core_hi]で1、[core_hi,hi]で1→0となる重み。"""
    if core_lo <= i <= core_hi:
        return 1.0
    if lo <= i < core_lo:
        t = (i - lo) / max(1, (core_lo - lo))
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if core_hi < i <= hi:
        t = (hi - i) / max(1, (hi - core_hi))
        return 0.5 - 0.5 * math.cos(math.pi * t)
    return 0.0


def smooth_window(xs, ys, lo, hi, win, n):
    """[lo,hi]区間のx,yを移動平均(窓win)で平滑化した結果を返す(閉路対応)。"""
    sx, sy = list(xs), list(ys)
    half = win // 2
    for i in range(lo, hi + 1):
        acc_x, acc_y, cnt = 0.0, 0.0, 0
        for j in range(i - half, i + half + 1):
            jj = j % n
            acc_x += xs[jj]
            acc_y += ys[jj]
            cnt += 1
        sx[i % n] = acc_x / cnt
        sy[i % n] = acc_y / cnt
    return sx, sy


def apply_widen(xs, ys, track_bounds, n, verbose):
    """タイトコーナー2箇所を平滑化して曲率を緩和。安全マージンを侵さない範囲で
    smoothing windowを段階的に強めるサーチを行う。"""
    xs, ys = list(xs), list(ys)
    for tc in TIGHT_CORNERS:
        lo, hi = tc["window"]
        core_lo, core_hi = tc["core"]
        best_win = 1
        for win in [3, 5, 7, 9, 11, 13, 15]:
            sx, sy = smooth_window(xs, ys, lo, hi, win, n)
            # taperブレンド
            bx, by = list(xs), list(ys)
            for i in range(lo, hi + 1):
                w = raised_cosine_taper(i, lo, core_lo, core_hi, hi)
                ii = i % n
                bx[ii] = (1 - w) * xs[ii] + w * sx[ii]
                by[ii] = (1 - w) * ys[ii] + w * sy[ii]
            # クリアランス確認
            ok = True
            for i in range(lo, hi + 1):
                ii = i % n
                br, bl = dist_to_bound((bx[ii], by[ii]), track_bounds)
                if br < SAFETY_MARGIN_M or bl < SAFETY_MARGIN_M:
                    ok = False
                    break
            if ok:
                best_win = win
            else:
                break
        sx, sy = smooth_window(xs, ys, lo, hi, best_win, n)
        for i in range(lo, hi + 1):
            w = raised_cosine_taper(i, lo, core_lo, core_hi, hi)
            ii = i % n
            xs[ii] = (1 - w) * xs[ii] + w * sx[ii]
            ys[ii] = (1 - w) * ys[ii] + w * sy[ii]
        if verbose:
            print(f"  [widen] {tc['name']}: smoothing window={best_win} (安全マージン{SAFETY_MARGIN_M}m維持の最大値)")
    return xs, ys


def _protected_core_points():
    """タイトコーナーのcore範囲に属する点indexの集合(前後1ptバッファ込み)。"""
    pts = set()
    for tc in TIGHT_CORNERS:
        lo, hi = tc["core"]
        for i in range(lo - 1, hi + 2):
            pts.add(i)
    return pts


def _dist_to_nearest_protected(i, protected, n, max_search):
    """点iから最も近い保護点までの距離(点数)。max_search以上ならmax_searchを返す。"""
    for d in range(0, max_search + 1):
        if (i - d) in protected or (i + d) in protected:
            return d
    return max_search


def apply_defense_bias(xs, ys, track_bounds, n, verbose):
    """ゆるいコーナー17区間にイン側1.02mバイアスを付与(taper込み)。

    2026-08-09、実装中に2回のバグを発見・修正:
    (1) taper窓が単純固定幅(±4pt)だとタイトコーナーcoreへ滲み出し曲率悪化。
    (2) (1)の対策でtaper窓をcore境界で単純クリップすると、クリップ位置の重みが
        1.0のまま隣接未処理点(重み0)へ直結し、1.02m級の不連続ジャンプが発生、
        かえって曲率が悪化した(wp222で1.02m変位→wp223で0mの段差)。
    最終対策: 保護点(タイトコーナーcore±1pt)からの距離に応じて重み自体を
    追加で減衰させる(distance_ramp)。taper長がぶつかる場合は自動的に短縮され、
    重みが必ず0へなめらかに収束してから保護点へ到達する。
    """
    xs, ys = list(xs), list(ys)
    base_xs, base_ys = list(xs), list(ys)
    protected = _protected_core_points()
    taper = 4
    nrm = normals(base_xs, base_ys, n)
    for lo, hi, side in GENTLE_CORNERS:
        wlo, whi = max(0, lo - taper), min(n - 1, hi + taper)
        sign = 1.0 if side == "L" else -1.0
        for i in range(wlo, whi + 1):
            if i in protected:
                continue  # 保護点は完全ノータッチ
            w = raised_cosine_taper(i, wlo, lo, hi, whi)
            # 保護点への近接度に応じた追加減衰(距離0で0、taper点離れていれば1.0)
            d = _dist_to_nearest_protected(i, protected, n, taper)
            w *= 0.5 - 0.5 * math.cos(math.pi * min(1.0, d / taper))
            offset = sign * INSIDE_GAP_M * w
            nx, ny = nrm[i]
            # クリアランス確認(offsetを縮小しながら安全を確保)
            scale = 1.0
            while scale > 0.0:
                cx = base_xs[i] + nx * offset * scale
                cy = base_ys[i] + ny * offset * scale
                br, bl = dist_to_bound((cx, cy), track_bounds)
                if br >= SAFETY_MARGIN_M and bl >= SAFETY_MARGIN_M:
                    break
                scale -= 0.1
            xs[i] = base_xs[i] + nx * offset * scale
            ys[i] = base_ys[i] + ny * offset * scale
        if verbose:
            print(f"  [defense-bias] wp{lo}-{hi} side={side}: applied (taper±{taper}pt, 保護点近接減衰込み)")
    return xs, ys


def report_corner_speeds(xs_before, ys_before, xs_after, ys_after, n, label):
    print(f"\n=== {label}: タイトコーナー速度到達点比較(ay_max={AY_MAX}) ===")
    for tc in TIGHT_CORNERS:
        lo, hi = tc["core"]
        best_before, best_after = 0.0, 0.0
        for src_x, src_y, store in [(xs_before, ys_before, "before"), (xs_after, ys_after, "after")]:
            max_k = 0.0
            for i in range(lo - 1, hi + 2):
                im, ip = (i - 1) % n, (i + 1) % n
                x0, y0 = src_x[im], src_y[im]
                x1, y1 = src_x[i % n], src_y[i % n]
                x2, y2 = src_x[ip], src_y[ip]
                a = math.hypot(x1 - x0, y1 - y0)
                b = math.hypot(x2 - x1, y2 - y1)
                c = math.hypot(x2 - x0, y2 - y0)
                cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                denom = a * b * c
                k = abs(2.0 * cross / denom) if denom > 1e-9 else 0.0
                max_k = max(max_k, k)
            v_cap = math.sqrt(AY_MAX / max_k) if max_k > 1e-9 else 999
            if store == "before":
                best_before = v_cap
            else:
                best_after = v_cap
        print(f"  {tc['name']}: v_cap {best_before*3.6:.1f}km/h -> {best_after*3.6:.1f}km/h")


def main():
    rows = read_traj(TRAJ_PATH)
    n = len(rows)
    xs0 = [r["x_m"] for r in rows]
    ys0 = [r["y_m"] for r in rows]
    track_bounds = load_track_bounds(TRACK_BOUNDS_PATH)

    # --- OFFENSE: 現行ジオメトリをそのままコピー ---
    offense_rows = [dict(r) for r in rows]
    recompute_geometry(offense_rows, closed=True)
    write_traj(
        OFFENSE_OUT, offense_rows,
        header_comment=(
            "traj_offense_35kmh.csv (2026-08-09生成)\n"
            "現行traj_mincurv.csvのx_m,y_mジオメトリをそのまま使用。\n"
            "vx_mpsプロファイルが既にay_max~12(直線最大66km/h相当)で設計済みと確認\n"
            "(最タイトコーナーwp224でvx_mps=19.9-20.0km/h=sqrt(12*R2.57m)と厳密一致)。\n"
            "35km/h到達はconfig.yaml側のv_max/ay_max/ay_profileで引き出す想定。\n"
            "s_m/psi_rad/kappa_radpm/vx_mps/ax_mps2は幾何再計算値(参考、実行時は未使用。\n"
            "utils.load_ref_path()の戻り値psi/kappaはmpc_controller.py/path_constraints_provider.py\n"
            "で破棄されx_m,y_mのみが使われるため)。"
        ),
    )
    print(f"[offense] wrote {OFFENSE_OUT} (n={len(offense_rows)}, x,y unchanged)")

    # --- DEFENSE: このスクリプトでの生成は廃止 ---
    # 2026-08-09、タイトコーナー拡幅(移動平均+raised-cosine taperブレンド)を試行したが、
    # taper境界付近でピーク曲率がむしろ悪化する箇所が複数発生(wp224の頂点|kappa|0.20が
    # 隣接wp225-228で0.24-0.26まで悪化)することを実測で確認、不採用とした。
    # 単純な座標移動平均は曲率単調減少を保証しないため、この用途には不適(2次元座標への
    # box-filterは形状によってはピークをずらすだけで悪化させうる)。
    #
    # 2026-08-10重要な注意: 本スクリプトのDEFENSE生成コードは意図せずtraj_defense_25kmh.csv
    # を上書きする事故を一度起こしている(build_defense_v2.pyで生成した正しい版を、この
    # スクリプトの再実行が古いロジックの出力で上書きした)。DEFENSE版は必ず
    # build_defense_v2.py(kaleidoscopeのclearance検証込み)を使うこと。このスクリプトは
    # OFFENSE版生成専用として運用する(下記のapply_widen/apply_defense_biasは参考として
    # 残置しているが呼び出さない)。
    apply_widen  # noqa: F401 (未使用、上記理由により無効化。関数自体は残置)
    apply_defense_bias  # noqa: F401 (未使用。build_defense_v2.pyへ移行済み)
    track_bounds  # noqa: F401 (DEFENSE生成廃止に伴い本スクリプト内では未使用、参考として残置)


if __name__ == "__main__":
    main()
