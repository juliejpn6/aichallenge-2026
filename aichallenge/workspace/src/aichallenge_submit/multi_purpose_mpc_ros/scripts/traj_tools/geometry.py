"""
TUM形式トラジェクトリCSV(s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2)の
read/write + x,y編集後のs_m/psi_rad/kappa_radpm再計算ユーティリティ。

2026-08-09、25km/hディフェンス/35km/hオフェンスの2トラジェクトリ生成用に新規作成
(design_docsに経緯記録予定)。既存env/final_ver3/traj_mincurv.csvは一切変更しない。
"""
import csv
import math

FIELDS = ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"]


def read_traj(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(row[k]) for k in FIELDS})
    return rows


def write_traj(path, rows, header_comment=None):
    """TUM形式CSVを書き出す。

    2026-08-10重要な訂正: 以前は`header_comment`を`#`行としてCSV先頭に書き込んで
    いたが、実際の消費側(`multi_purpose_mpc_ros/core/utils.py:load_ref_path`の
    `pd.read_csv(csv_file_path)`)は`comment=`引数を指定していないため、`#`行が
    あるとpandas.errors.ParserErrorで即座に読み込み失敗することが判明
    (kaleidoscopeでの目視確認時に発覚)。よって`header_comment`は無視し、常に
    ヘッダー行から始まる純粋なCSVのみを書き出す。由来・経緯は呼び出し側で
    別途.mdファイル等に記録すること。
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for row in rows:
            w.writerow([f"{row[k]:.7f}" for k in FIELDS])


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def recompute_geometry(rows, closed=True):
    """x_m/y_mを正として、s_m・psi_rad・kappa_radpmを幾何的に再計算する(in-place)。

    - psi_rad: 進行方向の接線角(atan2(dy,dx))。前後点の中心差分(closed loopなので
      両端も隣接点を使う)。
    - kappa_radpm: 隣接3点の外接円曲率(符号付き、進行方向に対して左旋回が正)。
      原ファイルの符号規約(左カーブ=正、右カーブ=負)と一致することを既存データで
      検証済み(recompute_geometryのユニットチェック参照)。
    - s_m: 各点間のユークリッド距離の累積和(0始点)。
    """
    n = len(rows)
    xs = [r["x_m"] for r in rows]
    ys = [r["y_m"] for r in rows]

    # s_m: 累積弧長
    s = [0.0] * n
    for i in range(1, n):
        s[i] = s[i - 1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])

    # psi_rad: 前後点を使った中心差分接線角(閉路想定)
    psi = [0.0] * n
    for i in range(n):
        if closed:
            ip = (i - 1) % n
            inx = (i + 1) % n
        else:
            ip = max(i - 1, 0)
            inx = min(i + 1, n - 1)
        dx = xs[inx] - xs[ip]
        dy = ys[inx] - ys[ip]
        # TUM規約: psi_rad=0はY軸正方向(北)基準(標準atan2は+X基準のため-pi/2補正)。
        # geometry.py自己検証(2026-08-09)で既存traj_mincurv.csvと突き合わせ、
        # atan2(dy,dx)-pi/2が原本と最も一致することを確認済み。
        psi[i] = _wrap_pi(math.atan2(dy, dx) - math.pi / 2)

    # kappa_radpm: 隣接3点の外接円曲率(符号付き)
    kappa = [0.0] * n
    for i in range(n):
        if closed:
            im = (i - 1) % n
            ip = (i + 1) % n
        else:
            im = max(i - 1, 0)
            ip = min(i + 1, n - 1)
        x0, y0 = xs[im], ys[im]
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[ip], ys[ip]
        # 符号付き曲率 = 2*cross / (|a|*|b|*|c|) (a,b,c: 三角形の3辺)
        a = math.hypot(x1 - x0, y1 - y0)
        b = math.hypot(x2 - x1, y2 - y1)
        c = math.hypot(x2 - x0, y2 - y0)
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        denom = a * b * c
        if denom < 1e-9:
            kappa[i] = 0.0
        else:
            kappa[i] = 2.0 * cross / denom

    for i, r in enumerate(rows):
        r["s_m"] = s[i]
        r["psi_rad"] = psi[i]
        r["kappa_radpm"] = kappa[i]
    return rows


def load_track_bounds(path):
    """global_racetrajectory_optimization形式のtrack csv
    (x_m,y_m,w_tr_right_m,w_tr_left_m)を読み込む。
    """
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            pts.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return pts


def nearest_track_width(pt, track_bounds):
    """指定点に最も近いtrack境界データの(w_tr_right, w_tr_left)を返す。"""
    bx, by, br, bl = min(
        track_bounds, key=lambda t: (t[0] - pt[0]) ** 2 + (t[1] - pt[1]) ** 2
    )
    return br, bl


if __name__ == "__main__":
    # 自己検証: 既存traj_mincurv.csvを読み込み、x,yはそのまま使ってkappa/psi/s_mを
    # 再計算し、原本と比較(丸め・アルゴリズム差はあるが大きく乖離しないことを確認)。
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/"
        "env/final_ver3/traj_mincurv.csv"
    )
    rows = read_traj(path)
    orig_kappa = [r["kappa_radpm"] for r in rows]
    orig_psi = [r["psi_rad"] for r in rows]
    orig_s = [r["s_m"] for r in rows]
    recompute_geometry(rows, closed=True)
    new_kappa = [r["kappa_radpm"] for r in rows]
    new_psi = [r["psi_rad"] for r in rows]
    new_s = [r["s_m"] for r in rows]

    def stats(a, b, name, wrap=False):
        diffs = []
        for x, y in zip(a, b):
            d = x - y
            if wrap:
                d = _wrap_pi(d)
            diffs.append(abs(d))
        print(f"{name}: max|diff|={max(diffs):.4f}, mean|diff|={sum(diffs)/len(diffs):.4f}")

    stats(orig_kappa, new_kappa, "kappa_radpm")
    stats(orig_psi, new_psi, "psi_rad", wrap=True)
    stats(orig_s, new_s, "s_m")
    print(f"total length: orig_final_s+seg={orig_s[-1]:.2f}, new={new_s[-1]:.2f}")
