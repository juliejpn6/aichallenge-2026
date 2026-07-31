"""255節続報(2026-07-31、レートスケーリングのクローズ作業Phase 2): 制御ループの
実効周期(dt、_control()呼び出し開始時刻の連続差分)の分布計装[PERF-DT]。

背景: 制御ループは`while rclpy.ok(): self._control()` + `rclpy.Rate.sleep()`
構造であり、周期超過は「コールバック欠落」ではなく「その周期の実行が後ろ倒しに
なるだけ」として現れる(実装報告のGeminiレビューで確認済みの前提)。既存の[PERF]は
_control()内の処理時間(sleep除く)のみを見ており、rclpy.Rate.sleep()自体の
ジッタを含む「実際に何秒おきに呼ばれたか」は別の量として計測されていなかった。
72Hz切替の判定基準(実効平均レート・dtのp99・連続超過回数)を実測するための
専用計装として_dtperf_record()を追加した。

mpc_controller.pyはrclpy依存(かつautoware_auto_control_msgs等、本環境では
未インストールのメッセージ型に依存)のため直接importできない。ロジック自体は
純粋な算術(print以外の副作用は_dtperf_*属性への代入のみ)のためミラー関数で
検証し、mpc_controller.py側の実装(呼び出し配線・ログ書式・窓リセット時の
ストリーク不リセット)はソーステキスト構造検証で確認する(既存test_rate_scaling_254.py
と同じ方針)。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def mirror_dtperf_window(dts, over_budget_s, prev_streak=0):
    """_dtperf_record()の集計ロジックのミラー(1ウィンドウ分)。
    prev_streakは窓開始時点で持ち越されている連続超過ストリーク数
    (_dtperf_cur_streakのウィンドウをまたいだ引き継ぎ)。
    戻り値: (p50, p95, p99, dt_max, effective_rate, over_ratio, max_streak, final_streak)
    """
    n = len(dts)
    over_count = 0
    cur_streak = prev_streak
    max_streak = prev_streak
    for dt in dts:
        if dt > over_budget_s:
            over_count += 1
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    sorted_dts = sorted(dts)
    dt_sum = sum(sorted_dts)
    effective_rate = n / dt_sum if dt_sum > 1e-9 else 0.0
    p50 = sorted_dts[int(n * 0.50)]
    p95 = sorted_dts[min(n - 1, int(n * 0.95))]
    p99 = sorted_dts[min(n - 1, int(n * 0.99))]
    dt_max = sorted_dts[-1]
    over_ratio = over_count / n
    return p50, p95, p99, dt_max, effective_rate, over_ratio, max_streak, cur_streak


# ---------------------------------------------------------------------------
# ①非矛盾性: 集計ロジックそのものの正しさ
# ---------------------------------------------------------------------------

def test_percentiles_on_uniform_dt_all_equal():
    """全周期が同一dtなら、p50/p95/p99/maxは全て同じ値になる。"""
    dts = [0.025] * 400
    p50, p95, p99, dt_max, *_ = mirror_dtperf_window(dts, over_budget_s=0.0139)
    assert p50 == p95 == p99 == dt_max == 0.025


def test_percentiles_known_index_based_values():
    """0.000〜0.399の400サンプル(sorted済み、index=値*1000)でp50/p95/p99を検証する。
    実装はint(n*0.95)等のインデックスベース近似(補間なし)であることを踏まえた
    期待値を計算する。"""
    dts = [i / 1000.0 for i in range(400)]
    p50, p95, p99, dt_max, *_ = mirror_dtperf_window(dts, over_budget_s=1.0)
    n = 400
    assert p50 == dts[int(n * 0.50)]
    assert p95 == dts[int(n * 0.95)]
    assert p99 == dts[int(n * 0.99)]
    assert dt_max == dts[-1] == 0.399


def test_percentile_index_clamped_at_n_minus_1_for_small_n():
    """小さいnでint(n*0.99)がn以上になりうる場合でも、末尾を超えてIndexErrorに
    ならないこと(min(n-1, ...)によるクランプ)を確認する。"""
    dts = [0.020, 0.021]
    p50, p95, p99, dt_max, *_ = mirror_dtperf_window(dts, over_budget_s=1.0)
    assert p99 == dt_max == 0.021


def test_effective_rate_known_value():
    """40周期の合計dtが1.0秒ちょうどなら、実効平均レートは40.0Hzになる。"""
    dts = [0.025] * 40
    _, _, _, _, effective_rate, *_ = mirror_dtperf_window(dts, over_budget_s=1.0)
    assert effective_rate == pytest.approx(40.0)


def test_effective_rate_degrades_when_dt_inflated():
    """周期が予算を超えて間延びしていれば、実効レートは目標より低く出る
    (72Hz狙いでdtが平均14ms=71.4Hzのケースを模擬)。"""
    dts = [0.014] * 100
    _, _, _, _, effective_rate, *_ = mirror_dtperf_window(dts, over_budget_s=1.0)
    assert effective_rate == pytest.approx(1.0 / 0.014)


def test_over_budget_ratio_known_value():
    """400周期中40周期が予算超過なら、超過率はちょうど10%になる。"""
    dts = [0.010] * 360 + [0.020] * 40  # budget=0.0139
    _, _, _, _, _, over_ratio, *_ = mirror_dtperf_window(dts, over_budget_s=0.0139)
    assert over_ratio == 40 / 400


def test_max_consecutive_streak_known_pattern():
    """超過(True)/非超過(False)のパターンから最大連続超過数を検証する。
    パターン: T,T,T,F,T,T,F,T,T,T,T,T → 最大連続は5(末尾のTTTTT)。"""
    pattern = [True, True, True, False, True, True, False, True, True, True, True, True]
    dts = [0.020 if over else 0.005 for over in pattern]
    *_, max_streak, final_streak = mirror_dtperf_window(dts, over_budget_s=0.0139)
    assert max_streak == 5
    assert final_streak == 5  # 最後のサンプルまで超過が続いたまま窓が終わる


def test_streak_resets_on_any_non_over_cycle():
    """1周期でも予算内に収まれば、ストリークはゼロへリセットされる。"""
    pattern = [True, True, True, True, False]
    dts = [0.020 if over else 0.005 for over in pattern]
    *_, max_streak, final_streak = mirror_dtperf_window(dts, over_budget_s=0.0139)
    assert max_streak == 4
    assert final_streak == 0


def test_streak_carries_across_window_boundary():
    """核心: ウィンドウ1の末尾3周期が連続超過中のまま窓が切れ、ウィンドウ2の
    冒頭2周期も引き続き超過だった場合、真の連続超過数は5であり、
    ウィンドウ1単独のmax(3)・ウィンドウ2単独のmax(2)のいずれよりも大きい。
    prev_streakを正しく引き継ぐ設計(_dtperf_reset_windowがcur_streakを
    リセットしない)でなければ、この5連続超過を検出できない。"""
    window1 = [True, True, True]       # 窓1の末尾、超過が続いたまま窓が切れる
    window2 = [True, True, False]      # 窓2の冒頭も超過が継続、3周期目で途切れる
    dts1 = [0.020 if o else 0.005 for o in window1]
    dts2 = [0.020 if o else 0.005 for o in window2]

    _, _, _, _, _, _, max1, final1 = mirror_dtperf_window(dts1, over_budget_s=0.0139, prev_streak=0)
    assert max1 == 3
    assert final1 == 3

    # ウィンドウ2はウィンドウ1終了時点のストリーク(final1=3)を引き継ぐ
    _, _, _, _, _, _, max2, final2 = mirror_dtperf_window(dts2, over_budget_s=0.0139, prev_streak=final1)
    assert max2 == 5  # 3(窓1から持ち越し)+2(窓2冒頭) = 5
    assert final2 == 0  # 窓2の3周期目で途切れる


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_dtperf_record_is_called_right_after_dt_is_computed_in_control():
    idx = _SRC.index("dt = (now - self._last_t).nanoseconds / 1e9")
    idx_end = _SRC.index("\n", idx)
    next_line_end = _SRC.index("\n", idx_end + 1)
    following = _SRC[idx_end:next_line_end]
    assert "self._dtperf_record(dt)" in following


def test_dtperf_record_reuses_existing_pf_report_every_and_over_budget_s():
    """新規パラメータを追加せず、既存の_pf_report_every/self._pf_over_budget_s
    (レートスケーリング機構が既に用意している値)をそのまま再利用していることを
    確認する。"""
    idx = _SRC.index("def _dtperf_record(self, dt):")
    idx_end = _SRC.index("\n    def _g2_release_ready", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._pf_over_budget_s" in snippet
    assert "self._pf_report_every" in snippet


def test_dtperf_reset_window_does_not_reset_cur_streak():
    """窓境界をまたぐ連続超過を検出し続けるための核心的な設計: _dtperf_reset_window
    は_dtperf_cur_streakへの代入を含まない(cycles/dts/over_count/max_streakのみ
    リセットする)ことを確認する。"""
    idx = _SRC.index("def _dtperf_reset_window(self):")
    idx_end = _SRC.index("\n    def _dtperf_record", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._dtperf_cur_streak = " not in snippet
    assert "self._dtperf_cycles = 0" in snippet
    assert "self._dtperf_dts = []" in snippet
    assert "self._dtperf_over_count = 0" in snippet
    assert "self._dtperf_max_streak = 0" in snippet


def test_dtperf_init_does_reset_cur_streak():
    """_dtperf_init(初回のみのlazy初期化)は_dtperf_reset_windowと異なり、
    _dtperf_cur_streakも含め全状態をゼロから始める。"""
    idx = _SRC.index("def _dtperf_init(self):")
    idx_end = _SRC.index("\n    def _dtperf_reset_window", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._dtperf_cur_streak = 0" in snippet


def test_perf_dt_log_line_contains_all_required_fields():
    idx = _SRC.index("'[PERF-DT]")
    idx_end = _SRC.index("flush=True)", idx)
    snippet = _SRC[idx:idx_end]
    for field in ("p50=", "p95=", "p99=", "max=", "eff_rate=", "over_budget=", "max_consec_over="):
        assert field in snippet, f"missing {field!r} in [PERF-DT] log line"


def test_dtperf_sort_happens_only_inside_report_block_not_every_cycle():
    """性能確認: sorted()呼び出しがレポート出力ブロック内(400周期に1回)のみで
    行われ、毎周期実行されるホットパスには出現しないことを確認する。"""
    idx = _SRC.index("def _dtperf_record(self, dt):")
    idx_end = _SRC.index("\n    def _g2_release_ready", idx)
    snippet = _SRC[idx:idx_end]
    assert snippet.count("sorted(") == 1
