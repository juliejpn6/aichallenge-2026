#!/usr/bin/env python3
"""evaluate_mpc_param_sets_20260807.py

25km/h蛇行対策パラメータセット(S1/S2/S3)比較の判定書生成(2026-08-07)。

外部AI(Claude)提案の2段階判定手法を実装する:
  Phase 1: ハード制約(衝突・STUCK)をポアソン帯で判定(件数でなくレート基準)。
           観測件数 > 期待値 + 2*sqrt(期待値) の場合のみ「悪化」と認定する。
  Phase 2: ソフト指標(蛇行、距離正規化cm/m)を線形アンカー式でスコア化。
           score = 100*(基準25km/h値 − 実測)/(基準25km/h値 − 15km/h値)
           ガードレール(|ekf_ey| p95)は加重せず、有意悪化のみ拒否権として扱う。
           有意差判定は5分ブロック分割によるブロック間SEを使う。

制御には一切関与しない、オフライン分析専用ツール。30km/h帯実験でも
再利用できるよう、条件(ログパス・basemetrics等)は全て引数/定数で分離している。

使い方:
    python3 evaluate_mpc_param_sets_20260807.py
"""
import math
import re
import statistics
import sys

LOC_XCHECK_PAT = re.compile(
    r"\[(\d+\.\d+)\].*\[LOC-XCHECK\] wp=(\d+) kappa=(-?[\d.]+) "
    r"ekf_ey=(-?[\d.]+) gnss_ey=(-?[\d.]+) v=([\d.]+) ot=(\w+)"
)


# ---------------------------------------------------------------------------
# Phase 1: ハード制約(ポアソン帯)
# ---------------------------------------------------------------------------

def count_hard_events(log_path):
    n_collision = 0
    n_stuck = 0
    n_wall = 0
    for line in open(log_path, errors="replace"):
        if "[COLLISION-SUSPECTED] v drop" in line:
            n_collision += 1
        if "[STUCK] detected" in line:
            n_stuck += 1
        if "[WALL" in line:
            n_wall += 1
    return {"collision": n_collision, "stuck": n_stuck, "wall": n_wall}


def poisson_judgment(observed, rate_per_min, duration_min):
    expected = rate_per_min * duration_min
    threshold = expected + 2 * math.sqrt(expected) if expected > 0 else 0.0
    verdict = "FAIL(悪化)" if observed > threshold else "PASS"
    return {
        "observed": observed,
        "expected": round(expected, 2),
        "threshold": round(threshold, 2),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Phase 2: 蛇行指標(距離正規化、ブロック分割)
# ---------------------------------------------------------------------------

def parse_loc_xcheck_rows(log_path, skip_until_t=None):
    rows = []
    for line in open(log_path, errors="replace"):
        m = LOC_XCHECK_PAT.search(line)
        if not m:
            continue
        t, wp, kappa, ekf_ey, gnss_ey, v, ot = m.groups()
        t = float(t)
        if skip_until_t is not None and t < skip_until_t:
            continue
        rows.append({
            "t": t, "wp": int(wp), "kappa": float(kappa),
            "ekf_ey": float(ekf_ey), "v": float(v), "ot": ot,
        })
    return rows


def split_into_blocks(rows, block_seconds=300):
    if not rows:
        return []
    t0 = rows[0]["t"]
    blocks = {}
    for r in rows:
        idx = int((r["t"] - t0) // block_seconds)
        blocks.setdefault(idx, []).append(r)
    return [blocks[k] for k in sorted(blocks.keys())]


def wobble_stats_for_rows(rows, ot_filter="NORMAL", v_min=0.5):
    """隣接サンプル差分std(cm/cycle)と距離正規化(cm/m)を直線/コーナー別に返す。"""
    straight_dv, corner_dv = [], []
    straight_ds, corner_ds = [], []
    prev = None
    for r in rows:
        if r["v"] < v_min or (ot_filter is not None and r["ot"] != ot_filter):
            prev = None
            continue
        if prev is not None:
            dt = r["t"] - prev["t"]
            if 0 < dt < 1.0:
                dv_cm = (r["ekf_ey"] - prev["ekf_ey"]) * 100.0
                ds_m = r["v"] * dt
                if abs(r["kappa"]) < 0.03:
                    straight_dv.append(dv_cm)
                    straight_ds.append(ds_m)
                elif abs(r["kappa"]) > 0.08:
                    corner_dv.append(dv_cm)
                    corner_ds.append(ds_m)
        prev = r

    def summarize(dvs, dss):
        if len(dvs) < 2:
            return {"n": len(dvs), "cm_per_cycle": float("nan"), "cm_per_m": float("nan")}
        cm_per_cycle = statistics.pstdev(dvs)
        mean_ds = statistics.mean(dss) if dss else float("nan")
        cm_per_m = cm_per_cycle / mean_ds if mean_ds and mean_ds > 0 else float("nan")
        return {"n": len(dvs), "cm_per_cycle": cm_per_cycle, "cm_per_m": cm_per_m}

    return {
        "straight": summarize(straight_dv, straight_ds),
        "corner": summarize(corner_dv, corner_ds),
    }


def block_wobble_series(log_path, skip_until_t=None, block_seconds=300):
    rows = parse_loc_xcheck_rows(log_path, skip_until_t=skip_until_t)
    blocks = split_into_blocks(rows, block_seconds=block_seconds)
    straight_series, corner_series = [], []
    for block_rows in blocks:
        stats = wobble_stats_for_rows(block_rows)
        if stats["straight"]["n"] >= 5:
            straight_series.append(stats["straight"]["cm_per_m"])
        if stats["corner"]["n"] >= 5:
            corner_series.append(stats["corner"]["cm_per_m"])
    return straight_series, corner_series


def mean_se(series):
    series = [x for x in series if x == x]  # drop NaN
    if len(series) < 2:
        return (series[0] if series else float("nan")), float("nan")
    m = statistics.mean(series)
    se = statistics.pstdev(series) / math.sqrt(len(series))
    return m, se


def guardrail_abs_ey(log_path, skip_until_t=None, ot_filter="NORMAL", v_min=0.5):
    rows = parse_loc_xcheck_rows(log_path, skip_until_t=skip_until_t)
    vals = [abs(r["ekf_ey"]) for r in rows if r["v"] >= v_min and
            (ot_filter is None or r["ot"] == ot_filter)]
    if not vals:
        return {"mean": float("nan"), "p95": float("nan"), "n": 0}
    vals_sorted = sorted(vals)
    p95 = vals_sorted[int(len(vals_sorted) * 0.95)]
    return {"mean": statistics.mean(vals), "p95": p95, "n": len(vals)}


def linear_anchor_score(actual, baseline_25kmh, target_15kmh):
    denom = baseline_25kmh - target_15kmh
    if denom == 0:
        return float("nan")
    return 100.0 * (baseline_25kmh - actual) / denom


def significant_diff(mean_a, se_a, mean_b, se_b):
    if se_a != se_a or se_b != se_b:  # NaN check
        return None  # 判定不能
    thresh = 2 * math.sqrt(se_a ** 2 + se_b ** 2)
    return abs(mean_a - mean_b) > thresh, thresh


# ---------------------------------------------------------------------------
# メイン: 実データへの適用
# ---------------------------------------------------------------------------

BASELINE_LOG = "/home/yoshihito/aichallenge-racingkart/output/20260807-065209/d3/autoware.log"
BASELINE_DURATION_MIN = 36.6
# 速度差分実験(delay=0、パラメータ変更なし、design_docs §22.2)のログから、
# cm/cycleとcm/m両方のアンカー値をこのスクリプト自身の関数で再計算する
# (単位不一致バグの修正: 既存design_docsの数値はcm/cycleのみで、cm/m版は
# 保存されていなかったため、同一ログ・同一手法で作り直す)。
TARGET_15KMH_LOG = "/home/yoshihito/aichallenge-racingkart/output/20260807-065209/d1/autoware.log"
BASE_25KMH_LOG = "/home/yoshihito/aichallenge-racingkart/output/20260807-065209/d3/autoware.log"
# 065209runのv_max更新完了(起動後約6分)以降のみアンカー計算に使う
ANCHOR_SKIP_SECONDS = 360


def compute_anchor(log_path):
    rows_all = parse_loc_xcheck_rows(log_path)
    if not rows_all:
        return {"straight": {"cm_per_cycle": float("nan"), "cm_per_m": float("nan")},
                "corner": {"cm_per_cycle": float("nan"), "cm_per_m": float("nan")}}
    skip_t = rows_all[0]["t"] + ANCHOR_SKIP_SECONDS
    rows = [r for r in rows_all if r["t"] >= skip_t]
    return wobble_stats_for_rows(rows)


RUNS = {
    "S1": "/home/yoshihito/aichallenge-racingkart/output/20260807-091734/d1/autoware.log",
    "S2": "/home/yoshihito/aichallenge-racingkart/output/20260807-091734/d2/autoware.log",
    "S3": "/home/yoshihito/aichallenge-racingkart/output/20260807-091734/d3/autoware.log",
}
RUN_DURATION_MIN = 30.0
# パラメータ投入完了は起動から約3.5分後(150秒待機+投入処理)。以降のみ集計に使う。
SKIP_SECONDS_FROM_START = 210


def main():
    target_15 = compute_anchor(TARGET_15KMH_LOG)
    base_25 = compute_anchor(BASE_25KMH_LOG)
    TARGET_15KMH_CM_PER_M = {"straight": target_15["straight"]["cm_per_m"],
                              "corner": target_15["corner"]["cm_per_m"]}
    BASE_25KMH_CM_PER_M = {"straight": base_25["straight"]["cm_per_m"],
                            "corner": base_25["corner"]["cm_per_m"]}
    print("=" * 78)
    print("アンカー値(cm/m、同一手法で再計算)")
    print("=" * 78)
    print(f"15km/h(目標=100点): 直線={TARGET_15KMH_CM_PER_M['straight']:.2f}cm/m "
          f"コーナー={TARGET_15KMH_CM_PER_M['corner']:.2f}cm/m")
    print(f"25km/h(基準=0点): 直線={BASE_25KMH_CM_PER_M['straight']:.2f}cm/m "
          f"コーナー={BASE_25KMH_CM_PER_M['corner']:.2f}cm/m\n")

    base_counts = count_hard_events(BASELINE_LOG)
    base_rate = {
        "collision": base_counts["collision"] / BASELINE_DURATION_MIN,
        "stuck": base_counts["stuck"] / BASELINE_DURATION_MIN,
    }

    print("=" * 78)
    print("Phase 1: ハード制約(ポアソン帯判定)")
    print("=" * 78)
    print(f"基準(デフォルト25km/h、{BASELINE_DURATION_MIN}分): "
          f"衝突={base_counts['collision']} STUCK={base_counts['stuck']} WALL={base_counts['wall']}")
    print(f"  → レート: 衝突={base_rate['collision']:.4f}/分 STUCK={base_rate['stuck']:.4f}/分\n")

    hard_verdicts = {}
    for name, path in RUNS.items():
        counts = count_hard_events(path)
        j_coll = poisson_judgment(counts["collision"], base_rate["collision"], RUN_DURATION_MIN)
        j_stuck = poisson_judgment(counts["stuck"], base_rate["stuck"], RUN_DURATION_MIN)
        wall_fail = counts["wall"] > 0
        overall = "FAIL" if (j_coll["verdict"].startswith("FAIL") or
                              j_stuck["verdict"].startswith("FAIL") or wall_fail) else "PASS"
        hard_verdicts[name] = overall
        print(f"[{name}] ({RUN_DURATION_MIN}分)")
        print(f"  衝突: 観測={j_coll['observed']} 期待値={j_coll['expected']} "
              f"閾値={j_coll['threshold']} → {j_coll['verdict']}")
        print(f"  STUCK: 観測={j_stuck['observed']} 期待値={j_stuck['expected']} "
              f"閾値={j_stuck['threshold']} → {j_stuck['verdict']}")
        print(f"  WALL: 観測={counts['wall']} → {'FAIL' if wall_fail else 'PASS'}")
        print(f"  総合判定: {overall}\n")

    print("=" * 78)
    print("Phase 2: ソフト指標(蛇行、距離正規化cm/m、線形アンカースコア)")
    print("=" * 78)

    skip_until_t_cache = {}
    for name, path in RUNS.items():
        rows_all = parse_loc_xcheck_rows(path)
        skip_until_t_cache[name] = rows_all[0]["t"] + SKIP_SECONDS_FROM_START if rows_all else None

    results = {}
    for name, path in RUNS.items():
        skip_t = skip_until_t_cache[name]
        straight_series, corner_series = block_wobble_series(path, skip_until_t=skip_t)
        s_mean, s_se = mean_se(straight_series)
        c_mean, c_se = mean_se(corner_series)
        s_score = linear_anchor_score(s_mean, BASE_25KMH_CM_PER_M["straight"], TARGET_15KMH_CM_PER_M["straight"])
        c_score = linear_anchor_score(c_mean, BASE_25KMH_CM_PER_M["corner"], TARGET_15KMH_CM_PER_M["corner"])
        avg_score = (s_score + c_score) / 2 if s_score == s_score and c_score == c_score else float("nan")
        guardrail = guardrail_abs_ey(path, skip_until_t=skip_t)
        results[name] = {
            "straight_mean": s_mean, "straight_se": s_se, "straight_score": s_score,
            "corner_mean": c_mean, "corner_se": c_se, "corner_score": c_score,
            "avg_score": avg_score,
            "guardrail_p95": guardrail["p95"], "guardrail_mean": guardrail["mean"],
            "n_blocks_straight": len(straight_series), "n_blocks_corner": len(corner_series),
        }
        print(f"[{name}] ハード判定={hard_verdicts[name]}")
        print(f"  直線: mean={s_mean:.2f}cm/m SE={s_se:.2f} (n_block={len(straight_series)}) "
              f"score={s_score:.1f}")
        print(f"  コーナー: mean={c_mean:.2f}cm/m SE={c_se:.2f} (n_block={len(corner_series)}) "
              f"score={c_score:.1f}")
        print(f"  総合スコア(直線・コーナー平均) = {avg_score:.1f}")
        print(f"  ガードレール |ekf_ey|: mean={guardrail['mean']:.3f}m p95={guardrail['p95']:.3f}m\n")

    print("=" * 78)
    print("S2 vs S3 有意差判定")
    print("=" * 78)
    survivors = [n for n in RUNS if hard_verdicts[n] == "PASS"]
    print(f"ハード制約PASS(生存者): {survivors}")
    if len(survivors) == 2:
        a, b = survivors
        for axis in ["straight", "corner"]:
            ma, sa = results[a][f"{axis}_mean"], results[a][f"{axis}_se"]
            mb, sb = results[b][f"{axis}_mean"], results[b][f"{axis}_se"]
            sig = significant_diff(ma, sa, mb, sb)
            if sig is None:
                print(f"  [{axis}] SE算出不能(ブロック数不足) → 判定不能")
            else:
                is_sig, thresh = sig
                print(f"  [{axis}] |{ma:.2f}-{mb:.2f}|={abs(ma-mb):.2f} "
                      f"閾値(2*sqrt(SEa^2+SEb^2))={thresh:.2f} → "
                      f"{'有意差あり' if is_sig else '有意差なし'}")
        avg_diff = abs(results[a]["avg_score"] - results[b]["avg_score"])
        print(f"  総合スコア差: {a}={results[a]['avg_score']:.1f} vs "
              f"{b}={results[b]['avg_score']:.1f} (差={avg_diff:.1f})")


if __name__ == "__main__":
    sys.exit(main())
