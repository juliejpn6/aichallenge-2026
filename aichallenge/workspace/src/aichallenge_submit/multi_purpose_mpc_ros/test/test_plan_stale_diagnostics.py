"""Unit test for the stale `_dbg_plan_lf`/`_dbg_plan_rf` diagnostic fix (2026-07-14).

0714-01実測: `_cheap_ok`が`rdy=0`で13秒以上連続して失敗する間、`_plan_pass`は
一度も呼ばれないのに`[OT]`ログの`planLf`/`planRf`は最後に`_plan_pass`が実際に
評価した時点の値(k_corner veto等による-1e9を含む)のまま表示され続けていた。
これは「今このコーナーが締め出している」ように誤認させる表示バグであり、
`_cheap_ok`が失敗した周期では`_dbg_plan_lf`/`_dbg_plan_rf`をnanへリセットする
よう修正した。ここではその分岐のみをミラーリングして検証する
(実際の`_cheap_ok`はROS型に依存する多数のゲートの組み合わせのため)。
"""
import math


def cheap_ok_else_branch(dbg_plan_lf, dbg_plan_rf):
    """mpc_controller.py の `else: ... self._dbg_plan_lf = float('nan') ...` のミラー。"""
    plan_ok, plan_side = False, 0
    dbg_plan_reason = "cheap_ok_fail"
    dbg_plan_lf = float('nan')
    dbg_plan_rf = float('nan')
    return plan_ok, plan_side, dbg_plan_reason, dbg_plan_lf, dbg_plan_rf


def test_cheap_ok_failure_resets_stale_plan_lf_rf_to_nan():
    """0714-01再現: 直前の_plan_pass呼び出しでplanLf=-1e9(k_corner veto等)が
    設定されていても、cheap_okが失敗した今回の周期ではnanへリセットされる。"""
    stale_lf, stale_rf = -1_000_000_000.0, 0.88  # 直前の実際の評価値(古い)
    _ok, _side, _reason, new_lf, new_rf = cheap_ok_else_branch(stale_lf, stale_rf)
    assert math.isnan(new_lf)
    assert math.isnan(new_rf)


def test_cheap_ok_failure_marks_reason_correctly_regression():
    """回帰: 失敗理由タグ自体は従来通りcheap_ok_failのまま。"""
    _ok, side, reason, _lf, _rf = cheap_ok_else_branch(1.0, 1.0)
    assert reason == "cheap_ok_fail"
    assert side == 0
