"""Structural regression guard for the safety_margin diagnostic logging (86/87節, 2026-07-17).

Background: 86節で、`_plan_pass`のalong_min_width幅検証がMPCソルバー自身の
safety_margin(core/MPC.py get_control()の`sm`計算)と独立しており、幅は足りると
判定されたコリドーがMPC側では解けず「Relaxed safety margin」の緩和リトライへ
頻繁に落ちている可能性を発見した。87節で過去ログを照合したところ、この特定の
不整合(safety_marginとalong_min_widthの関係)自体は過去に検討された形跡が無く、
新規の懸念として扱うことにした。

ユーザー指示: まずは判定ロジックを変更せず、次回ログで実際の相関を確認できる
よう検証ロギングのみを追加する。本ファイルはmpc_controller.py(rclpy依存のため
直接importできない)に対する構造的なソーステキスト検証で、3箇所(_plan_obs_log・
[PLAN-VETO]・[OT])全てが、core/MPC.py get_control()の`sm`計算式(
safety_margin_override優先、Noneならmodel.safety_marginへフォールバック)と
同一の優先順位で値を取得していることを保証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
_MPC_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "MPC.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()
with open(_MPC_SRC_PATH) as _f:
    _MPC_SRC = _f.read()


def test_core_mpc_get_control_priority_is_override_then_model_default():
    """前提確認: core/MPC.py get_control()の`sm`計算が
    safety_margin_override優先→Noneならmodel.safety_marginという優先順位で
    あることを確認する(mpc_controller.py側のロギングがこれと一致すべき基準)。"""
    idx = _MPC_SRC.index("sm = self.safety_margin_override if self.safety_margin_override is not None")
    snippet = _MPC_SRC[idx:idx + 150]
    assert "else self.model.safety_margin" in snippet


def test_plan_obs_log_includes_margin_field_with_correct_priority():
    """[PLAN-OBS]ログがmarginフィールドを、core/MPC.pyと同一の優先順位
    (safety_margin_override優先)で計算・出力していることを確認する。"""
    idx = _SRC.index("def _plan_obs_log")
    snippet = _SRC[idx:idx + 1200]
    assert "self._mpc.safety_margin_override" in snippet
    assert "self._mpc.model.safety_margin" in snippet
    assert "margin={_sm:.3f}" in snippet


def test_plan_veto_log_includes_margin_field():
    """[PLAN-VETO] MIN-WIDTH FAILログがmarginフィールドを出力していることを
    確認する(along_min_widthのneedだけでは見えない実効幅不足を次回ログで
    確認するための観測用フィールド)。"""
    idx = _SRC.index('[PLAN-VETO] MIN-WIDTH FAIL side=')
    snippet = _SRC[max(0, idx - 400):idx + 400]
    assert "_sm_veto" in snippet
    assert "margin={_sm_veto:.3f}" in snippet


def test_ot_log_includes_margin_field():
    """毎周期出力される[OT]ログにもmarginフィールドが含まれ、
    [MPC] Relaxed safety marginの発生タイミングと直接突き合わせられることを
    確認する(state/plan_pass呼び出しの有無に関わらず必ず出力される点が
    [PLAN-OBS]/[PLAN-VETO]との違い)。"""
    idx = _SRC.index('f"[OT] state=')
    snippet = _SRC[max(0, idx - 300):idx + 1800]
    assert "_sm_ot" in snippet
    assert "margin={_sm_ot:.3f}" in snippet


def test_all_three_margin_reads_use_identical_priority_expression():
    """回帰: 4箇所(_plan_obs_log・PLAN-VETO・OT・184節で追加した_fresh_gap_target)が
    全く同じ優先順位の式を使っており、どれか1箇所だけ将来変更されて食い違う
    (既存のギャップと同種の新しい非整合を生む)ことを防ぐ。
    2026-07-26更新(184節): _fresh_gap_target()がsafety_marginを読む際、既存3箇所と
    同一の優先順位式(override優先、Noneならmodel.safety_margin)を新規に作らず
    そのまま再利用したため、期待件数を3から4へ更新した。"""
    count = _SRC.count(
        "if self._mpc.safety_margin_override is not None\n")
    assert count == 4
