"""Unit tests for 190-5節(2026-07-26): is_closing_trendの3消費先(ENGAGEゲート/
G2-RELEASE/force_include_vid)のうち、実際に長時間ブロックの原因になっているのが
どれかを切り分けるための診断ロギング追加。

背景: 5日分18ログの機械的横断調査(190節)で、`dlat_ttc`ゲートが「相手停止時に
完全停止し再発進不可」症状の一因として3ログ(0723-06/0724-04/0726-03)で
確認された。根は`_dlat_closing_trend()`(141節で3消費先=ENGAGEゲート
(`_dlat_ttc_veto`)/G2-RELEASE(`_g2_release_ready`)/force_include_vidへ一元共有)
の`if footprint_risk: return True`分岐で、これは`cd`ゲート(190-4節)と同じ
「footprint_riskゾーンに居続ける限り抜け出す唯一の手段=ENGAGE自体が塞がれる」
構造的デッドロックと同根だった。ただしこの関数は3箇所で共有されており、
どの消費先が実際の長時間ブロックの犯人かは既存ログからは切り分けられなかった
(ENGAGEゲートは発動の瞬間のみ、G2-RELEASEは遷移時の値のみ、force_include_vidは
ログ皆無)。

対処: 判定ロジック自体は一切変更せず(診断専用)、以下3点のログを追加した。
  1. `_build_opponent_situation()`: is_closing_trendが連続True/footprint_risk
     起因かの計測、Falseに戻った瞬間に`[DLAT-TREND-CLEAR] duration=...`
  2. ENGAGEゲート: 既存`[DLAT-TTC-VETO]`(発動時のみ)に加え、解除時に
     `[DLAT-TTC-VETO-CLEAR] duration=...`
  3. force_include_vid: 従来皆無だった`[FORCE-INCLUDE-VID-TREND]`を新設
     (alpha救済とtrend救済を区別し、trend起因の発火のみ記録)

新規パラメータ0個(全て診断用カウンタ・ログのみ)。

mpc_controller.pyはrclpy依存で直接importできないため、カウンタ更新ロジックを
ミラー実装した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

CONTROL_RATE_HZ = 40.0


def mirror_duration_tracker(sequence):
    """`_dlat_trend_true_cycles`更新ロジックのミラー。sequenceはbool列(各周期の
    is_closing_trend値)。戻り値: [(clear_duration_s, via_fp), ...] Falseに戻る
    たびに記録される。"""
    cycles = 0
    via_fp_first = None
    events = []
    for is_true, is_fp in sequence:
        if is_true:
            cycles += 1
            if cycles == 1:
                via_fp_first = is_fp
        elif cycles > 0:
            events.append((cycles / CONTROL_RATE_HZ, via_fp_first))
            cycles = 0
    return events


# --- ①非矛盾性: 継続時間カウンタは単純増分・Falseで確定記録してリセット ---

def test_duration_tracker_records_on_false_transition():
    seq = [(True, True)] * 40 + [(False, False)]  # 40周期=1.0秒 継続
    events = mirror_duration_tracker(seq)
    assert len(events) == 1
    assert events[0][0] == 1.0
    assert events[0][1] is True  # footprint_risk起因


def test_duration_tracker_no_event_while_still_true():
    seq = [(True, False)] * 100  # まだFalseに戻っていない
    events = mirror_duration_tracker(seq)
    assert events == []


def test_duration_tracker_handles_multiple_episodes_independently():
    seq = [(True, True)] * 10 + [(False, False)] * 5 + [(True, False)] * 20 + [(False, False)]
    events = mirror_duration_tracker(seq)
    assert len(events) == 2
    assert events[0] == (10 / CONTROL_RATE_HZ, True)
    assert events[1] == (20 / CONTROL_RATE_HZ, False)


def test_via_fp_flag_captures_the_value_at_episode_start_not_end():
    """トレンド起因で始まりfootprint_risk状態が変化しても、記録されるのは
    エピソード開始時点の値(現在の実装は最初の1周期でのみ確定させるため)。"""
    seq = [(True, False)] + [(True, True)] * 39 + [(False, False)]
    events = mirror_duration_tracker(seq)
    assert events[0][1] is False  # 開始時点はfootprint_risk起因ではなかった


# ---------------------------------------------------------------------------
# ソーステキスト検証: 実際の追加箇所
# ---------------------------------------------------------------------------

def test_source_build_opponent_situation_tracks_duration():
    idx = _SRC.index("def _build_opponent_situation(")
    idx_end = _SRC.index("def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._dlat_trend_true_cycles += 1" in snippet
    assert "[DLAT-TREND-CLEAR]" in snippet
    assert "self._dlat_trend_true_cycles = 0" in snippet


def test_source_build_opponent_situation_still_returns_unmodified_is_closing_trend():
    """④遡及効果: 診断ロギング追加によって、実際にOpponentSituationへ渡される
    is_closing_trendの値自体(判定ロジック)は変更されていないことを確認する。"""
    idx = _SRC.index("def _build_opponent_situation(")
    idx_end = _SRC.index("def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "is_closing_trend=_is_closing_trend)" in snippet
    assert "_is_closing_trend = self._dlat_closing_trend(" in snippet


def test_source_dlat_ttc_veto_clear_logs_duration():
    idx = _SRC.index("_dlat_ttc_veto_effective = _plan_ok and _dlat_ttc_veto")
    snippet = _SRC[idx:idx + 1500]
    assert "[DLAT-TTC-VETO-CLEAR]" in snippet
    assert "self._dlat_ttc_veto_active_cycles += 1" in snippet
    assert "self._dlat_ttc_veto_active_cycles = 0" in snippet


def test_source_dlat_ttc_veto_activation_log_unchanged():
    """④遡及効果: 既存の[DLAT-TTC-VETO](発動時)ログ自体は無変更のまま残っている。"""
    idx = _SRC.index("_dlat_ttc_veto_effective = _plan_ok and _dlat_ttc_veto")
    snippet = _SRC[idx:idx + 1500]
    assert '"[DLAT-TTC-VETO] fwd_vid={opp_sit.fwd_vid} "' in snippet


def test_source_force_include_vid_trend_logs_activation():
    idx = _SRC.index("_force_include_via_trend = (")
    snippet = _SRC[idx:idx + 1100]
    assert "[FORCE-INCLUDE-VID-TREND]" in snippet
    assert "not self._force_include_vid_trend_active" in snippet


def test_source_force_include_vid_value_computation_unchanged():
    """④遡及効果: _force_include_vidの実際の値(alpha救済 OR trend救済という
    既存条件式)は変更されておらず、ロギングのために発火条件そのものを
    変えていないことを確認する。"""
    idx = _SRC.index("_force_include_via_trend = (")
    snippet = _SRC[idx:idx + 1100]
    assert 'self._ot_side != 0 and self._ot_alpha < 1.0 - 1e-3)' in snippet
    assert "or _force_include_via_trend)" in snippet


def test_source_no_new_config_parameters_introduced():
    """②非冗長性: 190-5節は診断ロギングのみで新規config.yamlパラメータを
    導入していないことを確認する(既存の_mpc_cfg.control_rateのみ再利用)。"""
    idx = _SRC.index("def _build_opponent_situation(")
    idx_end = _SRC.index("def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "_otget(" not in snippet
    assert "self._mpc_cfg.control_rate" in snippet
