"""Unit tests for the `_plan_pass` obstacle-branch lookahead trace diagnostic
(2026-07-19、wp176-178ウェッジ再調査、診断計装のみ・判定ロジック無変更)。

背景: design_docs 24節/26節(2026-07-11)で、wp176-178が「wmin(コリドー幅)は
5.1〜6.2mと終始『広い』判定だったにもかかわらず8秒間ウェッジ(実速度0.18m/sに
張り付き)した」という、コリドー幅ベースの指標では原理的に検知できない事象が
記録され、対策は「別アプローチで再検討する」として棚上げされたまま(80〜112節
では再検討されず)だった。0719-02実測で同一waypoint(wp176)にて再度スタックが
発生し、ユーザーから「アウトサイド(選ばれた側)には隙間がなく、インサイドが
がら空きだった」という新しい目視情報が得られたことを受け、エンゲージ時の
側選択(_plan_pass障害物分岐)がどのような窓内計算過程を経て`_side`を決めたかを
直接観測できるよう、[ENGAGE]ログへ窓内lf_i/rf_i推移(trace)を追加した。

このラウンドはStage1.5方針(推測で対策しない、まず計装して実測する)に従い、
対策実装ではなく診断計装のみを対象とする。テストは以下を検証する:
  1) 配線: traceが_plan_pass呼び出しごとにリセットされ(edge整合)、窓内ループの
     既存計算値(lf_i/rf_i、非冗長)をそのまま記録していること。
  2) [ENGAGE]ログにtrace=が追加され、既存フィールドは変更されていないこと。
  3) 非冗長性: trace追加はlf_min/rf_min/_sideの計算式そのものには一切手を
     入れていないこと(判定ロジック不変)。
  4) 診断が捉えられる現象の数式的デモ: フリーズしたfwd_latを、カーブに伴い
     ub/lbが変化する窓内waypointへそのまま適用すると、lf_i/rf_iが初期値
     (lf0/rf0)から大きく乖離しうることを合成データで示す(これは診断計装が
     「何を可視化できるか」の実証であり、wp176-178の実際の根本原因を
     確定するものではない — 確定は次回ログのtrace実測を待つ)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: 窓内トレース構築ロジックの複製
# ---------------------------------------------------------------------------

def _lookahead_trace(fwd_lat, block_half, ub_list, lb_list):
    """mpc_controller.py の障害物分岐ループ内トレース記録(2112〜2127行目付近)の複製ミラー。
    lf_i/rf_iの計算式自体は_plan_passの既存式(lf0/rf0と同一形)を再利用している。"""
    trace = []
    lf_min = rf_min = float("inf")
    for ub, lb in zip(ub_list, lb_list):
        lf_i = max(0.0, ub - (fwd_lat + block_half))
        rf_i = max(0.0, (fwd_lat - block_half) - lb)
        lf_min = min(lf_min, lf_i)
        rf_min = min(rf_min, rf_i)
        trace.append((lf_i, rf_i))
    return trace, lf_min, rf_min


BLOCK_HALF = 0.35  # self._ot_block_half相当(カート半幅+マージンのオーダー)


def test_trace_reuses_same_formula_as_lf0_rf0_no_new_calculation():
    """非冗長性: トレースのlf_i/rf_iはlf0/rf0(wp_oでの初期値)と同一の式
    (ub-(fwd_lat+block_half) / (fwd_lat-block_half)-lb)で計算されていることを確認する。"""
    fwd_lat = 0.5
    ub0, lb0 = 2.0, -2.0
    lf0 = max(0.0, ub0 - (fwd_lat + BLOCK_HALF))
    rf0 = max(0.0, (fwd_lat - BLOCK_HALF) - lb0)
    trace, _lf_min, _rf_min = _lookahead_trace(fwd_lat, BLOCK_HALF, [ub0], [lb0])
    assert trace[0] == (lf0, rf0)


def test_curving_boundary_causes_trace_to_diverge_from_initial_values():
    """診断が捉えられる現象のデモ(結論を断定するものではない): フリーズした
    fwd_latを、コーナーによりub/lbが変化していく窓内waypointへそのまま
    適用すると、後方のwaypointでのlf_i/rf_iが初期値(lf0/rf0)から大きく
    乖離しうる。wp176-178のような「エンゲージ時点の側選択がその後の実態と
    合わなくなる」事象の一因として、この乖離幅を次回ログのtraceで直接
    確認できるようにするのが今回の計装の目的。"""
    fwd_lat = 0.5  # 相手車の実測横位置(wp_oで固定)
    # 右カーブでコリドー中心線が左へ寄っていく想定(ub縮小・lb拡大)
    ub_list = [2.0, 1.6, 1.2, 0.9]
    lb_list = [-2.0, -1.8, -1.6, -1.4]
    trace, lf_min, rf_min = _lookahead_trace(fwd_lat, BLOCK_HALF, ub_list, lb_list)
    lf0, rf0 = trace[0]
    lf_last, rf_last = trace[-1]
    assert lf_last < lf0  # ub縮小に伴いlf_iは初期値より確実に狭くなる
    assert lf_min < lf0   # 窓内最小値も初期点より狭い(=初期点だけ見ると過大評価)
    assert lf_min == lf_last


def test_trace_is_empty_when_window_has_no_waypoints():
    """境界: 窓(clear_at)が短くループが1回も回らない場合はtrace=[]のまま
    (lf_min/rf_minはlf0/rf0のみで確定する、既存挙動を変えない)。"""
    trace, _lf_min, _rf_min = _lookahead_trace(0.5, BLOCK_HALF, [], [])
    assert trace == []


# ---------------------------------------------------------------------------
# 2) mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_dbg_plan_trace_initialized_in_init():
    idx = _SRC.index("self._dbg_plan_trace = []")
    assert idx > 0


def test_dbg_plan_trace_reset_at_top_of_plan_pass_edge_triggered():
    """edge整合: _plan_passが呼ばれるたびに(障害物分岐に入るかどうかに
    関わらず)traceは今回呼び出し分のみを反映するようリセットされる
    (55節のplan_lf/rf nanリセットと同じ考え方)。"""
    idx_def = _SRC.index("def _plan_pass(self, scan, prefer_side=0):")
    idx_reason = _SRC.index('self._dbg_plan_reason = "ok"', idx_def)
    idx_reset = _SRC.index("self._dbg_plan_trace = []", idx_def)
    # _dbg_plan_reasonのリセットと同じブロック内(直後)にあることを確認
    assert idx_reason < idx_reset < idx_reason + 100


def test_trace_appended_inside_obstacle_branch_lookahead_loop():
    idx_loop = _SRC.index("rf_i = max(0.0, (fwd_lat - self._ot_block_half) - float(wps[i].lb))")
    idx_append = _SRC.index("self._dbg_plan_trace.append(", idx_loop)
    idx_kcorner = _SRC.index("if k_corner is None and abs(_k) >= self._ot_pass_block_kappa:", idx_loop)
    assert idx_loop < idx_append < idx_kcorner


def test_trace_append_records_same_lf_i_rf_i_variables_no_recomputation():
    """非冗長性: appendの引数はループ内で既に計算済みのlf_i/rf_i/_k/wps[i].ub/lbを
    そのまま使っており、別の計算式を新たに導入していないことを確認する。"""
    idx = _SRC.index("self._dbg_plan_trace.append(")
    snippet = _SRC[idx:idx + 250]
    assert "round(_k, 3)" in snippet
    assert "round(lf_i, 2)" in snippet
    assert "round(rf_i, 2)" in snippet
    assert "round(float(wps[i].ub), 2)" in snippet
    assert "round(float(wps[i].lb), 2)" in snippet


def test_engage_log_includes_trace_field():
    idx = _SRC.index('f"[ENGAGE] side={_eval.plan_side}')
    # 2026-07-20追加(132節、Gap①Phase0)のdlat_v_ema/dlat_shrink_runフィールド分、窓を拡大。
    snippet = _SRC[idx:idx + 1400]
    assert 'f"trace={self._dbg_plan_trace}")' in snippet


def test_engage_log_existing_fields_unchanged_regression():
    """回帰: trace追加によって既存フィールド(91節のengage_dist_dynamic等)が
    削られたり順序以外の点で変更されたりしていないことを確認する。"""
    idx = _SRC.index('f"[ENGAGE] side={_eval.plan_side}')
    # 2026-07-20追加(132節、Gap①Phase0)のdlat_v_ema/dlat_shrink_runフィールド分、窓を拡大。
    snippet = _SRC[idx:idx + 1100]
    for field in ("fwd_ds={_fwd_ds}", "fwd_dlat={_scan.get('fwd_dlat')}",
                  "vopp={_fwd_vopp}", "closing_est={_eval.closing_est:.2f}",
                  "engage_dist_dynamic={_eval.engage_dist_dynamic:.2f}",
                  "t_reach_profile={_eval.t_reach_profile}",
                  "wp={self._mpc.model.wp_id}"):
        assert field in snippet


def test_lf_min_rf_min_side_selection_formula_unchanged_regression():
    """非冗長性/非矛盾性の核心確認: トレース追加はlf_min/rf_min更新や
    _sideの選択式(`1 if lf >= rf else -1`)には一切触れていないことを確認する。"""
    idx = _SRC.index("lf_min = min(lf_min, lf_i)")
    snippet = _SRC[idx:idx + 400]
    assert "rf_min = min(rf_min, rf_i)" in snippet
    idx_side = _SRC.index("_side = 1 if lf >= rf else -1")
    assert idx_side > idx
