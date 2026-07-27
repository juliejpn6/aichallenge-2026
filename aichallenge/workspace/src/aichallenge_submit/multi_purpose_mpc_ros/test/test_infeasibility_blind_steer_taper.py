"""Unit tests for issue④①: fallback_forwardの操舵盲目化に対するv_safeテーパー
+ _ot_infeasible_latchの状態非依存化(2026-07-22)。

背景: core/MPC.pyのget_control()はQP infeasibleが続く間、前回成功時の計画軌道を
最大N-2周期(≈0.45秒)先送りするが、それを超えると操舵を強制的にゼロ固定する
(mpc_controller.py側のfallback_forward分岐)。速度側は_v_safe_preで既にキャップ
されるが、操舵側には対応する安全網が無く、非ゼロ速度のままコリドー・相手車を
無視して直進し続けうる。実測(0722-4ログ、4台走行、d2)でinfeasibility_counterが
282(約7秒)まで悪化する間、u0が最大2.78m/sの非ゼロ値を取り続けていたことを確認した。

対処①: _ot_infeasible_stop(5、OVERTAKING中は既にこの周期数でSTOPPINGへ委譲して
いる既存閾値)を開始点、self._mpc.N-2(操舵が完全に盲目になる点)を終了点とし、
124/154節と同じ「二値→線形テーパー」でv_safeを0へ収束させる。state/branchに
関わらず常時適用。

対処②(横展開、ユーザー指摘による一貫性検証): _ot_infeasible_latch(再エンゲージ
禁止ラッチ)が従来OVERTAKING起因のinfeasibility委譲時にしかセットされておらず、
STOPPING中に発生したinfeasibility(対処①が主に扱う状況)では根本の混雑が解消
していなくても即再ENGAGEを許してしまう見落としがあった。ラッチのセット条件を
counter==_ot_infeasible_stop(状態非依存のエッジ検知)へ変更し、側の状態リセット
だけをOVERTAKING起因の場合に限定して残した。

mpc_controller.pyはrclpy依存のため直接importできないため、
test_footprint_taper.pyと同じ方針(純Pythonミラー関数+ソーステキストによる
構造的検証)を用いる。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

OT_INFEASIBLE_STOP = 5
N = 20
UMAX = 4.1667


def infeas_taper_cap(infeas_now, ot_infeasible_stop=OT_INFEASIBLE_STOP, n=N, umax=UMAX):
    """mpc_controller.pyのinfeas_taper計算式の複製ミラー。Noneはテーパー非適用を示す。"""
    if infeas_now <= ot_infeasible_stop:
        return None
    blind_at = max(ot_infeasible_stop + 1, n - 2)
    frac = min(1.0, (infeas_now - ot_infeasible_stop) / (blind_at - ot_infeasible_stop))
    return (1.0 - frac) * umax


def infeasible_latch_transition(infeas_now, ot_state, ot_infeasible_stop=OT_INFEASIBLE_STOP):
    """mpc_controller.pyの_ot_infeasible_latch設定式(状態非依存化後)の複製ミラー。
    戻り値: (latch_set: bool, new_ot_state)。"""
    latch_set = (infeas_now == ot_infeasible_stop)
    new_state = "STOPPING" if (latch_set and ot_state == "OVERTAKING") else ot_state
    return latch_set, new_state


# --- ①非矛盾性: テーパーの境界・単調性 ---

def test_no_taper_at_or_below_infeasible_stop_threshold():
    """infeas<=_ot_infeasible_stopの間はテーパー非適用(既存のOVERTAKING委譲閾値と
    同じタイミングまでは介入しない)。"""
    assert infeas_taper_cap(0) is None
    assert infeas_taper_cap(OT_INFEASIBLE_STOP) is None


def test_taper_starts_just_above_threshold():
    """閾値を1超えた瞬間からテーパーが開始する(ほぼv_maxに近い値)。"""
    cap = infeas_taper_cap(OT_INFEASIBLE_STOP + 1)
    assert cap is not None
    assert cap < UMAX
    assert cap > UMAX * 0.9  # まだ緩やか


def test_taper_reaches_zero_at_blind_point():
    """操舵が完全に盲目になる点(N-2)でv_safeキャップは0に達する。"""
    cap = infeas_taper_cap(N - 2)
    assert cap == pytest.approx(0.0, abs=1e-9)


def test_taper_stays_zero_beyond_blind_point():
    """N-2を超えて更にinfeasibilityが悪化しても0のまま(負にならない、既存の
    STUCK検知(300周期)まで安全側で待機し続ける)。"""
    assert infeas_taper_cap(N - 2 + 50) == pytest.approx(0.0, abs=1e-9)
    assert infeas_taper_cap(282) == pytest.approx(0.0, abs=1e-9)  # 実測値(0722-4ログ)


def test_taper_monotonically_decreasing():
    """単調減少(段階的に悪化するほど、より厳しく減速する)。"""
    caps = [infeas_taper_cap(x) for x in range(OT_INFEASIBLE_STOP + 1, N - 2 + 1)]
    assert all(a >= b for a, b in zip(caps, caps[1:]))


def test_retroactive_0722_4_log_d2_scenario():
    """遡及検証: 実測(0722-4ログ、d2、wp235-246)のinfeas=38, 78, 118, 158, 198, 238, 278は
    いずれも新テーパー導入後はv_safe=0に収束しており、実測u0(1.14〜2.78m/s)のような
    非ゼロ値は今後発生しないことを確認する。"""
    for infeas_val in (38, 78, 118, 158, 198, 238, 278):
        assert infeas_taper_cap(infeas_val) == pytest.approx(0.0, abs=1e-9)


# --- ①非矛盾性: ラッチの状態非依存化 ---

def test_latch_set_when_crossing_threshold_during_stopping():
    """②横展開の核心: STOPPING中にinfeasibilityが閾値へ達した場合も、従来は
    セットされなかったラッチが正しくセットされるようになったことを確認する。"""
    latch_set, new_state = infeasible_latch_transition(OT_INFEASIBLE_STOP, "STOPPING")
    assert latch_set is True
    assert new_state == "STOPPING"  # 側の状態遷移はOVERTAKING起因の場合のみ


def test_latch_set_when_crossing_threshold_during_normal():
    """NORMAL中に閾値へ達した場合も同様にラッチがセットされる(状態非依存)。"""
    latch_set, new_state = infeasible_latch_transition(OT_INFEASIBLE_STOP, "NORMAL")
    assert latch_set is True
    assert new_state == "NORMAL"


def test_latch_and_state_transition_both_fire_for_overtaking_unchanged():
    """回帰: OVERTAKING起因のケースは、ラッチセットと状態遷移(STOPPINGへ)が
    従来通り同時に発火する(挙動不変)。"""
    latch_set, new_state = infeasible_latch_transition(OT_INFEASIBLE_STOP, "OVERTAKING")
    assert latch_set is True
    assert new_state == "STOPPING"


def test_latch_not_reset_every_cycle_while_infeasible_persists():
    """エッジ検知(==)により、閾値を超えて infeasibility が続く間は毎周期
    再セットされない(一度だけ発火、従来のOVERTAKINGケースと同じタイミング)。"""
    latch_set_at_threshold, _ = infeasible_latch_transition(OT_INFEASIBLE_STOP, "STOPPING")
    latch_set_after, _ = infeasible_latch_transition(OT_INFEASIBLE_STOP + 10, "STOPPING")
    assert latch_set_at_threshold is True
    assert latch_set_after is False


def test_no_latch_or_transition_below_threshold():
    """閾値未満では何も起きない(通常時の回帰確認)。"""
    latch_set, new_state = infeasible_latch_transition(OT_INFEASIBLE_STOP - 1, "OVERTAKING")
    assert latch_set is False
    assert new_state == "OVERTAKING"


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_source_infeas_taper_reuses_existing_constants_no_new_parameters():
    """②非冗長性: 新規パラメータを使わず、既存の_ot_infeasible_stop/self._mpc.N/
    input_constraints["umax"]を再利用していることを確認する。"""
    idx = _SRC.index("_infeas_now = self._mpc.infeasibility_counter")
    snippet = _SRC[idx:idx + 700]
    assert "self._ot_infeasible_stop" in snippet
    assert "self._mpc.N - 2" in snippet
    assert 'self._mpc.input_constraints["umax"][0]' in snippet


def test_source_infeas_taper_is_state_independent():
    """①非矛盾性: infeas_taperの計算がif self._ot_state==...のようなガードに
    包まれておらず、wall_slow/footprint_riskと同じ「常時適用」であることを確認する。"""
    idx = _SRC.index("_infeas_now = self._mpc.infeasibility_counter")
    idx_prev_nl = _SRC.rfind("\n", 0, idx)
    line_start = _SRC.rfind("\n", 0, idx_prev_nl) + 1
    snippet_before = _SRC[line_start:idx]
    assert "if self._ot_state" not in snippet_before


def test_source_infeas_taper_appended_to_v_safe_candidates():
    """v_safe候補スタックへ追加され、他の候補と同じmin()合成に参加することを確認する。"""
    idx = _SRC.index("_infeas_now = self._mpc.infeasibility_counter")
    snippet = _SRC[idx:idx + 900]
    assert '_v_safe_cand.append(("infeas_taper' in snippet


def test_source_latch_edge_triggered_on_equality_not_gte():
    """②横展開: ラッチのセット条件が状態非依存のエッジ検知(==)になっており、
    旧来の「_ot_state=="OVERTAKING" and >=」という組み合わせ条件ではないことを
    確認する。"""
    idx = _SRC.index("self._ot_infeasible_latch = self._ot_infeasible_latch_cycles")
    snippet_before = _SRC[max(0, idx - 300):idx]
    assert "self._mpc.infeasibility_counter == self._ot_infeasible_stop" in snippet_before


def test_source_side_reset_still_scoped_to_overtaking_only():
    """回帰: side資産のリセット(_ot_side等)は引き続きOVERTAKING起因の場合のみに
    スコープされており、STOPPING/NORMAL起因では発火しないことを確認する
    (無関係な側コミットを誤って解除しない)。"""
    idx = _SRC.index("self._ot_infeasible_latch = self._ot_infeasible_latch_cycles")
    snippet = _SRC[idx:idx + 400]
    assert 'if self._ot_state == "OVERTAKING":' in snippet
    assert "self._ot_side = 0" in snippet
