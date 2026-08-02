#!/usr/bin/env python3
"""analyze_perf_gap.py

263節(2026-08-02、予選環境ギャップ分析の準備Phase 4): [PERF]系ログ
([PERF]/[PERF-DT]/[PERF-RUSAGE]/[PERF-SPIKE]/[PERF-PLATFORM])から
J(ジッタ)・work/work_cpu分布・プラットフォーム構成・スパイク帰属を
一発生成する分析ツール。制御には一切関与しない、オフライン分析専用。

単一環境モード: ログ1本を渡すと、そのログの特性(J・work/work_cpu・
プラットフォーム・スパイク帰属)を整形して出力する。
    python3 analyze_perf_gap.py <log>

比較モード: ローカルログと予選ログの2本を渡すと、比較表と72Hz成立予測
(go/no-go)を出力する(1本目=ローカル、2本目=予選、という順で解釈する)。
    python3 analyze_perf_gap.py <local_log> <qualifying_log>

ログ形式は263節時点のmpc_controller.py計装を前提とするが、計装追加の
歴史があり収集時期によってフィールドが増減するため、無い項目は全てN/A
(Noneまたは'N/A'表示)で通す寛容なパーサにしてある。1つの行の1フィールドが
読めなくても、その行・その集計は「部分的に」使う(全部か無かではない —
これは[PERF-SPIKE]計装のnseg等とは違い、ログ解析側では欠損の影響が
局所的なため)。

集計方針(既存の分析プロンプト・design docでの用法に合わせた選択、複数の
選び方がありうる場所での本ツールの立場):
  - dtパーセンタイル(p50/p95/p99/p999)は複数の[PERF-DT]窓の**n重み付け平均**を
    採用する(263節続報Part A-2で変更。旧実装の「窓ごとの最悪値のmax集計」は
    走行時間[=窓の数]が長いほど単調に悪化する統計であり、run長の異なる
    走行のJを比較できない問題があった。加重平均は真の全区間パーセンタイル
    [生サンプル非公開のため厳密には計算不能]の実用的な近似であり、窓数が
    増えても定常過程なら真の期待値へ収束する)。
  - max_msだけは「観測された真の最大値」という定義上、従来通りmax集計を
    維持する(走行時間とともに悪化しうるのが正しい挙動のため)。
  - p99の窓別分布(min/median/max)を参考情報として別途保持する
    (「どの窓が悪かったか」の特定用、旧実装のmax集計が果たしていた役割)。
  - eff_rateは窓ごとのnで重み付けした加重平均を採用する(スループットの
    実態を表すため)。
  - work/work_cpu/cpu_ratio/run_delayは可能な限り生データ(sum/n)から
    再計算した加重平均を使う(窓ごとの単純平均の平均は窓サイズが不揃いだと
    歪むため)。
"""
import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# パーサ: 各[PERF*]タグの1行をdictへ変換する。フィールドが見つからなければ
# キー自体を辞書に含めない(呼び出し側でdict.get(key)がNoneを返す形に統一)。
# ---------------------------------------------------------------------------

def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    if m is None:
        return None
    val = m.group(1)
    if val == 'N/A':
        return None
    try:
        return cast(val)
    except ValueError:
        return None


def parse_perf_dt_lines(text):
    """[PERF-DT] n=%d p50=%.2fms p95=%.2fms p99=%.2fms p999=%.2fms max=%.2fms
    eff_rate=%.2fHz over_budget=%.2f%% max_consec_over=%d margin=%.3fms"""
    windows = []
    for line in re.findall(r'\[PERF-DT\][^\n]*', text):
        w = {
            'n': _num(r'\bn=(\d+)', line, int),
            'p50_ms': _num(r'p50=([\d.]+)ms', line),
            'p95_ms': _num(r'p95=([\d.]+)ms', line),
            'p99_ms': _num(r'p99=([\d.]+)ms', line),
            'p999_ms': _num(r'p999=([\d.]+)ms', line),
            'max_ms': _num(r'\bmax=([\d.]+)ms', line),
            'eff_rate_hz': _num(r'eff_rate=([\d.]+)Hz', line),
            'over_budget_pct': _num(r'over_budget=([\d.]+)%', line),
            'max_consec_over': _num(r'max_consec_over=(\d+)', line, int),
            'margin_ms': _num(r'margin=([\d.]+)ms', line),
        }
        windows.append(w)
    return windows


def parse_perf_lines(text):
    """[PERF] n=%d work avg=%.1fms max=%.1fms >%.1fms=%d回
    work_cpu avg=%.1fms max=%.1fms >%.1fms=%d回 n_dynobs_max=%d | ..."""
    windows = []
    for line in re.findall(r'\[PERF\] n=\d+[^\n]*', text):
        budgets = re.findall(r'>([\d.]+)ms=', line)
        overs = re.findall(r'>[\d.]+ms=(\d+)\s*回', line)
        w = {
            'n': _num(r'\[PERF\] n=(\d+)', line, int),
            'work_avg_ms': _num(r'work avg=([\d.]+)ms', line),
            'work_max_ms': _num(r'work avg=[\d.]+ms max=([\d.]+)ms', line),
            'work_cpu_avg_ms': _num(r'work_cpu avg=([\d.]+)ms', line),
            'work_cpu_max_ms': _num(r'work_cpu avg=[\d.]+ms max=([\d.]+)ms', line),
            'budget_ms': float(budgets[0]) if budgets else None,
            'work_over': int(overs[0]) if len(overs) > 0 else None,
            'work_cpu_over': int(overs[1]) if len(overs) > 1 else None,
        }
        windows.append(w)
    return windows


def parse_perf_rusage_lines(text):
    """[PERF-RUSAGE] n=%d cpu_time=%.2fs wall_time=%.2fs cpu_ratio=%.2f
    nivcsw=%d nvcsw=%d freq_avg=...MHz freq_min=...MHz freq_max=...MHz
    run_delay_avg=...ms run_delay_max=...ms"""
    windows = []
    for line in re.findall(r'\[PERF-RUSAGE\][^\n]*', text):
        w = {
            'n': _num(r'\bn=(\d+)', line, int),
            'cpu_time_s': _num(r'cpu_time=([\d.]+)s', line),
            'wall_time_s': _num(r'wall_time=([\d.]+)s', line),
            'cpu_ratio': _num(r'cpu_ratio=([\d.]+|nan)', line),
            'nivcsw': _num(r'nivcsw=(\d+)', line, int),
            'nvcsw': _num(r'nvcsw=(\d+)', line, int),
            'freq_avg_mhz': _num(r'freq_avg=([\d.]+)MHz', line),
            'run_delay_avg_ms': _num(r'run_delay_avg=([\d.]+)ms', line),
            'run_delay_max_ms': _num(r'run_delay_max=([\d.]+)ms', line),
        }
        windows.append(w)
    return windows


_SPIKE_COMPONENT_RE = re.compile(r'(?<=\| )(.*)$')


# mpc_controller.py側の計装で「親(ロールアップ)タイマー」として意図的に
# 子タイマーの合計を兼ねているキー。dominant_component抽出からは除外する
# (除外しないと、数値の性質上ほぼ必ず親キーが「最大」になり、raster/prep/
# traffic_ot等の実質的な原因が隠れてしまう——手動分析(263節本編)で
# 実際にmpc_setup/mpc_corridor/traffic_ot等の子レベルで帰属を見た方針に合わせる)。
_SPIKE_ROLLUP_COMPONENTS = frozenset({'mpc', 'r_delta_swing_total'})


def parse_perf_spike_lines(text):
    """[PERF-SPIKE] loop=%d ... work=%.2fms work_cpu=%.2fms budget=%.2fms
    ... freq_* run_delay=... | <component>=<ms> ... (帰属内訳)"""
    spikes = []
    for line in re.findall(r'\[PERF-SPIKE\] loop=[^\n]*', text):
        components = {}
        tail = _SPIKE_COMPONENT_RE.search(line)
        if tail:
            for key, val in re.findall(r'([a-zA-Z_0-9]+)=([\d.]+)', tail.group(1)):
                components[key] = float(val)
        leaf_components = {k: v for k, v in components.items()
                            if k not in _SPIKE_ROLLUP_COMPONENTS}
        dominant = (max(leaf_components.items(), key=lambda kv: kv[1])
                    if leaf_components else (None, None))
        s = {
            'loop': _num(r'loop=(\d+)', line, int),
            'work_ms': _num(r'\bwork=([\d.]+)ms', line),
            'work_cpu_ms': _num(r'work_cpu=([\d.]+)ms', line),
            'budget_ms': _num(r'budget=([\d.]+)ms', line),
            'nivcsw_diff': _num(r'nivcsw_diff=(-?\d+)', line, int),
            'gen2_gc': 'gen2_gc=True' in line,
            'run_delay_ms': _num(r'\brun_delay=([\d.]+)ms', line),
            'dominant_component': dominant[0],
            'dominant_component_ms': dominant[1],
        }
        spikes.append(s)
    return spikes


def parse_perf_platform(text):
    """[PERF-PLATFORM]系の複数行(governor行・colocated_affinity行・
    cgroup行・availability行)を1つのdictへマージする。263節Phase 1/2より
    前のログにはcgroup/use_sim_time/availability行が無いため、全てN/A/None
    のまま返す(存在するフィールドだけ埋まる)。"""
    blob = '\n'.join(re.findall(r'\[PERF-PLATFORM\][^\n]*', text))

    def _s(pattern):
        m = re.search(pattern, blob)
        if m is None:
            return None
        val = m.group(1).strip()
        return None if val == 'N/A' else val

    def _s_raw(pattern):
        """availability系専用: 'N/A'自体が意味のある値(項目が計装されて
        いるが読み取れなかった)であり、行が存在しない場合のNoneと区別する
        ため、_s()と違って'N/A'をそのまま返す。"""
        m = re.search(pattern, blob)
        return m.group(1).strip() if m else None

    return {
        'governor': _s(r'governor=(\S+)'),
        'scaling_max_freq': _s(r'scaling_max_freq=(\S+)'),
        'cores_sampled': _s(r'cores_sampled=(\d+)'),
        'cpu_affinity': _s(r'cpu_affinity=(\[[^\]]*\])'),
        'use_sim_time': _s(r'use_sim_time=(\S+)'),
        'colocated_affinity': _s(r"colocated_affinity=(\{[^\n]*\})"),
        'cgroup_version': _s(r'\bcgroup=(\S+)'),
        'cpu_quota_cores': _s(r'cpu_quota_cores=(\S+)'),
        'cpuset_cpus': _s(r'cpuset_cpus=(\S+)'),
        'memory_max': _s(r'memory_max=(\S+)'),
        'cpu_model': _s(r'cpu_model="([^"]*)"'),
        'cpu_count': _s(r'cpu_count=(\S+)'),
        'availability_scaling_cur_freq': _s_raw(r'scaling_cur_freq=(OK|N/A)'),
        'availability_sched_schedstats': _s_raw(r'sched_schedstats=(OK|N/A)'),
        'availability_cgroup_cpu_quota': _s_raw(r'cgroup_cpu_quota=(OK|N/A)'),
        'availability_cgroup_cpuset': _s_raw(r'cgroup_cpuset=(OK|N/A)'),
        'availability_cgroup_memory_max': _s_raw(r'cgroup_memory_max=(OK|N/A)'),
    }


# ---------------------------------------------------------------------------
# 集計: 複数窓 -> 1つの代表値。集計方針は本ファイル冒頭のdocstring参照。
# ---------------------------------------------------------------------------

def aggregate_dt(windows):
    """263節続報Part A-2で集計方針を変更した: [PERF-DT]は窓(既定400/720周期)
    ごとのパーセンタイルしかログに残らず、生のdtサンプル全件は取得できない
    (ダンプするとログ量が爆発するため)。旧実装はp50/p95/p99/p999を「窓ごとの
    最悪値のmax集計」としていたが、これは走行時間(=窓の数)が長いほど
    「たまたま悪い窓」に当たる機会が増え、単調に悪化する統計だった
    (C4'の40Hzミニベースラインが会話の合間で想定の倍以上走ってしまい、
    Jが同じ設定のC4のベースラインと比較不能になった実例で発覚)。

    そこで、パーセンタイル系(p50/p95/p99/p999)は「窓ごとの値をn(窓内サンプル数)で
    重み付け平均する」方式へ変更した。これは真の全区間p99(生サンプルへの
    アクセスが無いため厳密には計算不能)の実用的な近似であり、かつ窓数が
    増えても値が単調悪化しない(定常的なプロセスなら窓を増やすほどむしろ
    真の期待値へ収束する)という望ましい性質を持つ。max_msだけは「観測された
    真の最大値」という定義上、従来通りmax集計を維持する(これは正しく走行
    時間とともに悪化しうる統計であり、それ自体が正しい挙動)。

    p99の窓別分布(min/median/max)は「どの窓が悪かったか」を特定するための
    参考情報として別途保持する(旧実装のmax集計が果たしていた役割はここへ
    移した)。"""
    windows = [w for w in windows if w.get('n')]
    if not windows:
        return None
    total_n = sum(w['n'] for w in windows)

    def _max_field(key):
        vals = [w[key] for w in windows if w.get(key) is not None]
        return max(vals) if vals else None

    def _weighted_avg(key):
        pairs = [(w[key], w['n']) for w in windows if w.get(key) is not None]
        if not pairs:
            return None
        return sum(v * n for v, n in pairs) / sum(n for _, n in pairs)

    p99_vals = sorted(w['p99_ms'] for w in windows if w.get('p99_ms') is not None)
    if p99_vals:
        mid = len(p99_vals) // 2
        p99_window_median = (p99_vals[mid] if len(p99_vals) % 2 == 1
                              else (p99_vals[mid - 1] + p99_vals[mid]) / 2)
        p99_window_min, p99_window_max = p99_vals[0], p99_vals[-1]
    else:
        p99_window_median = p99_window_min = p99_window_max = None

    return {
        'total_n': total_n,
        'num_windows': len(windows),
        'p50_ms': _weighted_avg('p50_ms'),
        'p95_ms': _weighted_avg('p95_ms'),
        'p99_ms': _weighted_avg('p99_ms'),
        'p999_ms': _weighted_avg('p999_ms'),
        'max_ms': _max_field('max_ms'),
        'eff_rate_hz': _weighted_avg('eff_rate_hz'),
        'over_budget_pct_worst': _max_field('over_budget_pct'),
        'max_consec_over': _max_field('max_consec_over'),
        'margin_ms': windows[0].get('margin_ms'),
        'p99_window_min_ms': p99_window_min,
        'p99_window_median_ms': p99_window_median,
        'p99_window_max_ms': p99_window_max,
    }


def aggregate_perf(windows):
    windows = [w for w in windows if w.get('n')]
    if not windows:
        return None
    total_n = sum(w['n'] for w in windows)
    total_work_over = sum(w['work_over'] for w in windows if w.get('work_over') is not None)
    total_work_cpu_over = sum(
        w['work_cpu_over'] for w in windows if w.get('work_cpu_over') is not None)
    budgets = [w['budget_ms'] for w in windows if w.get('budget_ms') is not None]

    def _weighted_avg(key):
        pairs = [(w[key], w['n']) for w in windows if w.get(key) is not None]
        if not pairs:
            return None
        return sum(v * n for v, n in pairs) / sum(n for _, n in pairs)

    def _max_field(key):
        vals = [w[key] for w in windows if w.get(key) is not None]
        return max(vals) if vals else None

    return {
        'total_n': total_n,
        'budget_ms': budgets[-1] if budgets else None,
        'work_avg_ms': _weighted_avg('work_avg_ms'),
        'work_max_ms': _max_field('work_max_ms'),
        'work_over_pct': (100.0 * total_work_over / total_n) if total_n else None,
        'work_cpu_avg_ms': _weighted_avg('work_cpu_avg_ms'),
        'work_cpu_max_ms': _max_field('work_cpu_max_ms'),
        'work_cpu_over_pct': (100.0 * total_work_cpu_over / total_n) if total_n else None,
    }


def aggregate_rusage(windows):
    windows = [w for w in windows if w.get('n')]
    if not windows:
        return None
    total_cpu = sum(w['cpu_time_s'] for w in windows if w.get('cpu_time_s') is not None)
    total_wall = sum(w['wall_time_s'] for w in windows if w.get('wall_time_s') is not None)
    total_n = sum(w['n'] for w in windows)
    nivcsw_vals = [w['nivcsw'] for w in windows if w.get('nivcsw') is not None]
    run_delay_avg_vals = [
        w['run_delay_avg_ms'] for w in windows if w.get('run_delay_avg_ms') is not None]
    run_delay_max_vals = [
        w['run_delay_max_ms'] for w in windows if w.get('run_delay_max_ms') is not None]
    return {
        'cpu_ratio_overall': (total_cpu / total_wall) if total_wall else None,
        'nivcsw_avg_per_window': (sum(nivcsw_vals) / len(nivcsw_vals)) if nivcsw_vals else None,
        'nivcsw_total': sum(nivcsw_vals) if nivcsw_vals else None,
        'run_delay_avg_ms': (
            sum(run_delay_avg_vals) / len(run_delay_avg_vals)) if run_delay_avg_vals else None,
        'run_delay_max_ms': max(run_delay_max_vals) if run_delay_max_vals else None,
        'run_delay_available': bool(run_delay_avg_vals),
        'total_n': total_n,
    }


def aggregate_spikes(spikes):
    if not spikes:
        return {'count': 0, 'dominant_components': {}}
    dom_counts = {}
    for s in spikes:
        key = s.get('dominant_component') or 'N/A'
        dom_counts[key] = dom_counts.get(key, 0) + 1
    return {
        'count': len(spikes),
        'dominant_components': dict(
            sorted(dom_counts.items(), key=lambda kv: -kv[1])),
        'work_ms_max': max((s['work_ms'] for s in spikes if s.get('work_ms') is not None),
                            default=None),
    }


# ---------------------------------------------------------------------------
# 1本のログをまとめて読み、上記アグリゲータへ通す。
# ---------------------------------------------------------------------------

_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')


def analyze_log(path):
    text = _ANSI_ESCAPE_RE.sub('', Path(path).read_text(errors='replace'))
    dt = aggregate_dt(parse_perf_dt_lines(text))
    perf = aggregate_perf(parse_perf_lines(text))
    rusage = aggregate_rusage(parse_perf_rusage_lines(text))
    spikes = aggregate_spikes(parse_perf_spike_lines(text))
    platform = parse_perf_platform(text)
    budget_ms = (perf or {}).get('budget_ms')
    j_ms = None
    if dt and budget_ms:
        j_ms = dt['p99_ms'] - budget_ms
    return {
        'path': str(path),
        'dt': dt,
        'perf': perf,
        'rusage': rusage,
        'spikes': spikes,
        'platform': platform,
        'budget_ms': budget_ms,
        'j_ms': j_ms,
    }


# ---------------------------------------------------------------------------
# 出力整形
# ---------------------------------------------------------------------------

def _fmt(v, unit='', nd=2):
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.{nd}f}{unit}'
    return f'{v}{unit}'


def format_single_report(result):
    lines = []
    lines.append(f"# [PERF]系ログ分析: {result['path']}")
    lines.append('')
    dt, perf, rusage, spikes, platform = (
        result['dt'], result['perf'], result['rusage'], result['spikes'], result['platform'])

    lines.append('## ジッタ(J = dt p99 - 予算)')
    if dt and result['budget_ms']:
        lines.append(f"budget={_fmt(result['budget_ms'], 'ms')}  "
                      f"J={_fmt(result['j_ms'], 'ms')}")
        lines.append(f"p50={_fmt(dt['p50_ms'], 'ms')} p95={_fmt(dt['p95_ms'], 'ms')} "
                      f"p99={_fmt(dt['p99_ms'], 'ms')} p999={_fmt(dt['p999_ms'], 'ms')} "
                      f"max={_fmt(dt['max_ms'], 'ms')}")
        lines.append(f"eff_rate(加重平均)={_fmt(dt['eff_rate_hz'], 'Hz')}  "
                      f"over_budget(最悪窓)={_fmt(dt['over_budget_pct_worst'], '%')}  "
                      f"max_consec_over={_fmt(dt['max_consec_over'])}  "
                      f"(集計対象={dt['num_windows']}窓, n={dt['total_n']})")
        lines.append(f"[参考]窓別p99分布: min={_fmt(dt['p99_window_min_ms'], 'ms')} "
                      f"median={_fmt(dt['p99_window_median_ms'], 'ms')} "
                      f"max={_fmt(dt['p99_window_max_ms'], 'ms')} "
                      "(問題窓の特定用。Jの算出には使わない)")
    else:
        lines.append("[PERF-DT]行が見つからないか、budgetを特定できませんでした(N/A)。")
    lines.append('')

    lines.append('## work(wall) / work_cpu 分布')
    if perf:
        lines.append(f"work avg={_fmt(perf['work_avg_ms'], 'ms')} "
                      f"max={_fmt(perf['work_max_ms'], 'ms')} "
                      f">予算={_fmt(perf['work_over_pct'], '%')}")
        lines.append(f"work_cpu avg={_fmt(perf['work_cpu_avg_ms'], 'ms')} "
                      f"max={_fmt(perf['work_cpu_max_ms'], 'ms')} "
                      f">予算={_fmt(perf['work_cpu_over_pct'], '%')}")
        if perf['work_avg_ms'] is not None and perf['work_cpu_avg_ms'] is not None:
            gap = perf['work_avg_ms'] - perf['work_cpu_avg_ms']
            lines.append(f"wall-cpuギャップ(avg)={_fmt(gap, 'ms')} "
                         "(大きいほどスケジューラ横取り待ちが支配的)")
    else:
        lines.append("[PERF]行が見つかりませんでした(N/A)。")
    if rusage:
        lines.append(f"cpu_ratio(全体)={_fmt(rusage['cpu_ratio_overall'], '', 3)}  "
                      f"nivcsw(窓平均)={_fmt(rusage['nivcsw_avg_per_window'], '', 1)}")
        if rusage['run_delay_available']:
            lines.append(f"run_delay avg={_fmt(rusage['run_delay_avg_ms'], 'ms')} "
                          f"max={_fmt(rusage['run_delay_max_ms'], 'ms')} "
                          "(スケジューラのランキュー待ち、実測)")
            if perf and perf['work_avg_ms'] is not None and perf['work_cpu_avg_ms'] is not None:
                residual = (perf['work_avg_ms'] - perf['work_cpu_avg_ms']
                            - rusage['run_delay_avg_ms'])
                lines.append(
                    f"wall内訳(avg): work_cpu={_fmt(perf['work_cpu_avg_ms'], 'ms')} + "
                    f"run_delay={_fmt(rusage['run_delay_avg_ms'], 'ms')} + "
                    f"残差(block等)={_fmt(residual, 'ms')} = work={_fmt(perf['work_avg_ms'], 'ms')}")
        else:
            lines.append("run_delay=N/A(sched_schedstats無効、またはこのログは263節Part B以前)")
    lines.append('')

    lines.append('## [PERF-SPIKE]帰属')
    lines.append(f"件数={spikes['count']}")
    if spikes['count']:
        for comp, cnt in spikes['dominant_components'].items():
            lines.append(f"  支配的コスト={comp}: {cnt}件")
        lines.append(f"work最大={_fmt(spikes['work_ms_max'], 'ms')}")
    lines.append('')

    lines.append('## [PERF-PLATFORM]')
    lines.append(f"governor={platform['governor'] or 'N/A'}  "
                  f"cpu_model={platform['cpu_model'] or 'N/A'}  "
                  f"cpu_count={platform['cpu_count'] or 'N/A'}  "
                  f"use_sim_time={platform['use_sim_time'] or 'N/A'}")
    lines.append(f"cgroup={platform['cgroup_version'] or 'N/A'}  "
                  f"cpu_quota_cores={platform['cpu_quota_cores'] or 'N/A'}  "
                  f"cpuset_cpus={platform['cpuset_cpus'] or 'N/A'}  "
                  f"memory_max={platform['memory_max'] or 'N/A'}")
    lines.append(f"cpu_affinity={platform['cpu_affinity'] or 'N/A'}  "
                  f"colocated_affinity={platform['colocated_affinity'] or 'N/A'}")
    avail = [
        ('scaling_cur_freq', platform['availability_scaling_cur_freq']),
        ('sched_schedstats', platform['availability_sched_schedstats']),
        ('cgroup_cpu_quota', platform['availability_cgroup_cpu_quota']),
        ('cgroup_cpuset', platform['availability_cgroup_cpuset']),
        ('cgroup_memory_max', platform['availability_cgroup_memory_max']),
    ]
    lines.append('availability: ' + ' '.join(
        f'{k}={v or "N/A"}' for k, v in avail))
    return '\n'.join(lines)


def _recommend(local, qual, target_hz):
    """72Hz成立予測のgo/no-go判定。判定ロジックは単純なしきい値規則であり
    ブラックボックスにしない(根拠を全てreasonsへ積む)。"""
    budget_new_ms = 1000.0 / target_hz
    reasons = []
    verdict = 'GO'

    j_qual = qual['j_ms']
    if j_qual is not None:
        predicted_p99 = budget_new_ms + j_qual
        reasons.append(
            f"dt p99予測 = {budget_new_ms:.3f}ms(新予算) + J_予選({j_qual:.2f}ms) "
            f"= {predicted_p99:.2f}ms。改訂基準(≦予算+J)は定義上満たす(判定材料にはならない)。")
    else:
        reasons.append("J_予選が算出できず(N/A)、dt p99予測は評価不能。予選側ログの再確認が必要。")
        verdict = '追加情報必要'

    qual_perf = qual['perf']
    if qual_perf and qual_perf['work_cpu_avg_ms'] is not None:
        avg = qual_perf['work_cpu_avg_ms']
        mx = qual_perf['work_cpu_max_ms']
        headroom_avg = budget_new_ms - avg
        reasons.append(
            f"work_cpu(予選実測、現行rateのまま) avg={avg:.2f}ms×{target_hz:.0f}Hz="
            f"{avg * target_hz:.0f}ms/s vs 新予算合計1000ms/s "
            f"(余裕={headroom_avg:.2f}ms/cycle)。"
            + ("avgは新予算内。" if headroom_avg > 0 else "avgの時点で新予算を超過、要再検討。"))
        if mx is not None and mx > budget_new_ms:
            reasons.append(
                f"ただしwork_cpu max={mx:.2f}msは新予算{budget_new_ms:.3f}msを超えており、"
                "瞬間的な計算コスト増があると新予算を割る周期が発生しうる"
                "(窓のavg/maxしか無くパーセンタイル分布が無いため、正確な超過率は"
                "予測不可——72Hz実測でしか確定できない)。")
            if verdict == 'GO':
                verdict = '追加情報必要'
    else:
        reasons.append("予選ログにwork_cpuデータが無く(N/A)、計算余裕を評価できない。")
        verdict = '追加情報必要'

    local_rusage, qual_rusage = local['rusage'], qual['rusage']
    if local_rusage and qual_rusage:
        ld, qd = local_rusage.get('run_delay_avg_ms'), qual_rusage.get('run_delay_avg_ms')
        if ld is not None and qd is not None:
            reasons.append(
                f"run_delay avg: ローカル={ld:.3f}ms vs 予選={qd:.3f}ms "
                f"(差={qd - ld:+.3f}ms、正ならスケジューラ競合が予選側でより深刻)")
        elif qd is None:
            reasons.append(
                "予選ログにrun_delayが無い(sched_schedstats未計測、または263節Part B以前の"
                "ログ)。次回収集ではsudo sysctl kernel.sched_schedstats=1を検討すること。")
        ln, qn = local_rusage.get('nivcsw_avg_per_window'), qual_rusage.get(
            'nivcsw_avg_per_window')
        if ln is not None and qn is not None:
            reasons.append(
                f"nivcsw(窓平均、横取り回数): ローカル={ln:.1f} vs 予選={qn:.1f} "
                f"(差={qn - ln:+.1f})")

    local_dt, qual_dt = local['dt'], qual['dt']
    if local_dt and qual_dt:
        reasons.append(
            f"eff_rate(現行rateでの実効値、加重平均): ローカル="
            f"{_fmt(local_dt['eff_rate_hz'], 'Hz')} vs 予選={_fmt(qual_dt['eff_rate_hz'], 'Hz')}")

    return verdict, reasons


def format_comparison_report(local, qual, target_hz):
    lines = []
    lines.append(f"# 比較: ローカル({local['path']}) vs 予選({qual['path']})")
    lines.append('')
    lines.append('## 比較表')
    lines.append(f"{'指標':<28}{'ローカル':>16}{'予選':>16}")

    def row(label, lv, qv, unit=''):
        lines.append(f"{label:<28}{_fmt(lv, unit):>16}{_fmt(qv, unit):>16}")

    row('budget(ms、現行rate)', local['budget_ms'], qual['budget_ms'], 'ms')
    row('J = p99-budget(ms)', local['j_ms'], qual['j_ms'], 'ms')
    if local['dt'] and qual['dt']:
        row('dt p99(ms、窓n加重平均)', local['dt']['p99_ms'], qual['dt']['p99_ms'], 'ms')
        row('  [参考]窓別p99 max(ms)', local['dt']['p99_window_max_ms'],
            qual['dt']['p99_window_max_ms'], 'ms')
        row('eff_rate(Hz)', local['dt']['eff_rate_hz'], qual['dt']['eff_rate_hz'], 'Hz')
    if local['perf'] and qual['perf']:
        row('work_cpu avg(ms)', local['perf']['work_cpu_avg_ms'],
            qual['perf']['work_cpu_avg_ms'], 'ms')
        row('work_cpu max(ms)', local['perf']['work_cpu_max_ms'],
            qual['perf']['work_cpu_max_ms'], 'ms')
    if local['rusage'] and qual['rusage']:
        row('cpu_ratio(全体)', local['rusage']['cpu_ratio_overall'],
            qual['rusage']['cpu_ratio_overall'])
        row('nivcsw(窓平均)', local['rusage']['nivcsw_avg_per_window'],
            qual['rusage']['nivcsw_avg_per_window'])
        row('run_delay avg(ms)', local['rusage']['run_delay_avg_ms'],
            qual['rusage']['run_delay_avg_ms'], 'ms')
    row('cgroup', local['platform']['cgroup_version'], qual['platform']['cgroup_version'])
    row('cpu_quota_cores', local['platform']['cpu_quota_cores'],
        qual['platform']['cpu_quota_cores'])
    row('cpu_affinity', local['platform']['cpu_affinity'], qual['platform']['cpu_affinity'])
    lines.append('')

    verdict, reasons = _recommend(local, qual, target_hz)
    lines.append(f'## {target_hz:.0f}Hz成立予測: {verdict}')
    for r in reasons:
        lines.append(f"- {r}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='[PERF]系ログの単一環境分析、または2本比較+72Hz成立予測')
    parser.add_argument('log', help='分析対象ログ(単一環境モード)、または比較モードの1本目(ローカル)')
    parser.add_argument('qualifying_log', nargs='?', default=None,
                         help='比較モードの2本目(予選)。省略時は単一環境モード')
    parser.add_argument('--target-hz', type=float, default=72.0,
                         help='成立予測の対象レート(既定72Hz)')
    parser.add_argument('--json', action='store_true', help='整形テキストの代わりにJSONで出力')
    args = parser.parse_args(argv)

    local_result = analyze_log(args.log)
    if args.qualifying_log is None:
        if args.json:
            print(json.dumps(local_result, ensure_ascii=False, indent=2))
        else:
            print(format_single_report(local_result))
        return 0

    qual_result = analyze_log(args.qualifying_log)
    if args.json:
        verdict, reasons = _recommend(local_result, qual_result, args.target_hz)
        print(json.dumps(
            {'local': local_result, 'qualifying': qual_result,
             'verdict': verdict, 'reasons': reasons},
            ensure_ascii=False, indent=2))
    else:
        print(format_comparison_report(local_result, qual_result, args.target_hz))
    return 0


if __name__ == '__main__':
    sys.exit(main())
