#!/usr/bin/env python3
"""Phase 0(候補④設計、design_docs predictive_control_overtake_development_plan_20260805.md
7-8節参照): 新ENGAGE予測式(候補④)の発火距離 vs icc制動開始距離(_g2_speed/G2式)を、
既存ログの[ENGAGE]イベントに対して机上計算で比較する。コード変更ゼロ・制御には一切
影響しないオフライン分析(既存ログの再解析のみ)。

目的: margin(新ENGAGE式の余裕時間)をどの値に設定すれば、icc_stopの制動開始距離
(ds_icc)より先に新ENGAGE距離(ds_engage)へ到達する(=icc制動が発火する前にENGAGE
できる)割合が十分高くなるかを実測する。

前提定数(mpc_controller.pyから転記、変更ゼロ):
  a_brake = 1.3        # [m/s^2] G2式の安全減速度
  margin_center = 4.0   # [m] icc追従車間(中心間)
  t_lateral = 3.0        # [s] 横移動フェーズ時間(既存_ot_t_lateral)
  pass_clear = 3.0       # [m] 抜き切りクリアランス(既存_ot_pass_clear)
  ramp_time = 0.5        # [s] オフセットランプ所要時間(既存_ot_ramp_time)

計算式(2026-08-05訂正: 初版はramp_time(0.5s)のみでt_lateralを含めておらず、
既存_engage_dist_dynamicより過小な距離になっていたバグを修正):
  v_rel = v_pot - vopp  (closing_est相当、自車ポテンシャル速度基準)
  ds_engage(margin) = v_rel * (t_lateral + margin) + pass_clear   # 新ENGAGE発火距離
    (既存_engage_dist_dynamic = v_rel*t_lateral+pass_clearに、追加マージンmarginを
    上乗せした形。margin=0が現行の_engage_dist_dynamicと同一)
  ds_icc = margin_center + (v_pot^2 - vopp^2) / (2*a_brake)         # icc制動開始距離
         = margin_center + v_rel * (v_pot + vopp) / (2*a_brake)

  「iccが先に発火」 <=> ds_icc > ds_engage(margin)

使い方: python3 analyze_engage_vs_icc_ordering.py [logdir1] [logdir2] ...
  (引数省略時は2026-08-05当日のdev3ラン3本を既定で解析)

実測結果サマリ(2026-08-05、dev3当日3ラン合算、n=460 ENGAGE):
  高v_rel(3.0m/s以上、全体の約60%): margin=0.5秒で順序保証ほぼ完璧(0%がicc先発火)
  低v_rel(1.0-1.5m/s帯): margin=1.5秒でもなお55.6%がicc先発火——構造的な限界。
  margin=0.0(=現行_engage_dist_dynamicそのまま)でも全体40.7%が既にicc後発火と
  判明、これは新機構導入前から存在する現状の実態。
  詳細はdesign_docs predictive_control_overtake_development_plan_20260805.md
  9節参照。
"""
import glob
import re
import sys

A_BRAKE = 1.3
MARGIN_CENTER = 4.0
T_LATERAL = 3.0
PASS_CLEAR = 3.0
RAMP_TIME = 0.5

ENGAGE_RE = re.compile(
    r'\[ENGAGE\] side=(\S+) fwd_ds=(\S+) fwd_dlat=(\S+) vopp=(\S+) '
)


def default_v_pot():
    """v_pot(自車ポテンシャル速度)。config.yamlの現行有効ブロック(中速20km/h、
    166行目付近)に合わせた固定値。TODO: 起動ログのmpc.v_max行を確実に判別できる
    正規表現が定まれば、ログから自動抽出する形へ拡張する。"""
    return 20.0 / 3.6


def collect_engage_events(logdirs):
    events = []
    for logdir in logdirs:
        for path in sorted(glob.glob(f"{logdir}/d*/autoware.log")):
            with open(path, errors="ignore") as f:
                for line in f:
                    m = ENGAGE_RE.search(line)
                    if not m:
                        continue
                    try:
                        fwd_ds = float(m.group(2))
                        vopp = float(m.group(4))
                    except ValueError:
                        continue
                    events.append({"path": path, "fwd_ds": fwd_ds, "vopp": vopp})
    return events


def analyze(events, v_pot, margins):
    print(f"v_pot={v_pot:.3f} m/s ({v_pot*3.6:.1f}km/h)  n_events={len(events)}")
    print(f"定数: a_brake={A_BRAKE} margin_center={MARGIN_CENTER} "
          f"t_lateral={T_LATERAL} pass_clear={PASS_CLEAR} ramp_time={RAMP_TIME}\n")

    # v_relビン分割
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0)]
    for margin in margins:
        print(f"=== margin={margin}s ===")
        rows = {b: {"n": 0, "icc_first": 0} for b in bins}
        for e in events:
            v_rel = v_pot - e["vopp"]
            if v_rel <= 0:
                continue  # 自車の方が遅い/同速はENGAGE対象外(既存_ego_ready相当)
            ds_engage = v_rel * (T_LATERAL + margin) + PASS_CLEAR
            ds_icc = MARGIN_CENTER + v_rel * (v_pot + e["vopp"]) / (2.0 * A_BRAKE)
            icc_first = ds_icc > ds_engage
            for b in bins:
                if b[0] <= v_rel < b[1]:
                    rows[b]["n"] += 1
                    if icc_first:
                        rows[b]["icc_first"] += 1
                    break
        print(f"{'v_rel[m/s]':>14} {'n':>6} {'icc先発火':>10} {'割合':>8}")
        total_n = 0
        total_icc = 0
        for b in bins:
            n = rows[b]["n"]
            ic = rows[b]["icc_first"]
            total_n += n
            total_icc += ic
            pct = f"{ic/n*100:.1f}%" if n else "-"
            print(f"{b[0]:>5.1f}-{b[1]:<5.1f} {n:>6} {ic:>10} {pct:>8}")
        pct_all = f"{total_icc/total_n*100:.1f}%" if total_n else "-"
        print(f"{'全体':>14} {total_n:>6} {total_icc:>10} {pct_all:>8}\n")


if __name__ == "__main__":
    logdirs = sys.argv[1:] if len(sys.argv) > 1 else [
        "/home/yoshihito/aichallenge-racingkart/output/20260805-153141",
        "/home/yoshihito/aichallenge-racingkart/output/20260805-162758",
        "/home/yoshihito/aichallenge-racingkart/output/20260805-170835",
    ]
    v_pot = default_v_pot()
    events = collect_engage_events(logdirs)
    analyze(events, v_pot, margins=[0.0, 0.5, 1.0, 1.5, 2.0])
