"""Unit tests for 190-6節(2026-07-26): `_on_path`(engage_lat_max=2.0m)ゲートが
継続してFalseの間の診断ロギング追加。

背景: 5日分18ログの機械的横断調査(190節)で、0726-05ログにおいて両者が静止中
(fwd_vopp≈0、v_odom≈0)にもかかわらずfwd_dlatが1.15m→2.87m→2.99mと単調に
増加し続け、`_on_path`(fwd_dlat<=engage_lat_max)が11秒間成立しなかった
ケースが見つかった。`_scan_traffic()`の`dlat`計算式(相手側waypoint基準の
局所座標系と自車側waypoint基準の局所座標系を単純に差し引いている)が、直線
では無視できる誤差でも曲率のある区間では系統的に蓄積する可能性が根本原因
候補として浮上した。

ただし`_scan_traffic()`はcars/left_free/right_free/being_overtaken/ICC/
footprint_riskなどOTロジックのほぼ全ての判断が依存する基盤関数であり、
82/83節の教訓(switchback関連の広範な条件変更が完走率半減・衝突4倍という
重大な回帰を招いた前例)を踏まえ、直接改修する前にまず実地頻度・大きさを
計測する必要がある。本節は判定ロジックを一切変更せず、`_on_path=False`の
継続時間・その間のfwd_dlat変化量・相手速度を記録する診断ログのみを追加する。

mpc_controller.pyはrclpy依存で直接importできないため、カウンタ更新ロジックを
ミラー実装した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

CONTROL_RATE_HZ = 40.0


def mirror_on_path_tracker(sequence):
    """`_on_path_false_cycles`更新ロジックのミラー。sequenceは
    (on_path: bool, fwd_dlat: float)のタプル列。戻り値: on_pathがFalse→Trueへ
    戻るたびに記録される[(duration_s, dlat_start, dlat_end), ...]。"""
    cycles = 0
    dlat_start = None
    events = []
    for on_path, dlat in sequence:
        if not on_path:
            if cycles == 0:
                dlat_start = dlat
            cycles += 1
        elif cycles > 0:
            events.append((cycles / CONTROL_RATE_HZ, dlat_start, dlat))
            cycles = 0
            dlat_start = None
    return events


# --- ①非矛盾性: on_path=Falseの継続だけを計測し、Trueに戻ると確定記録してリセット ---

def test_tracker_records_duration_and_dlat_drift_on_clear():
    seq = [(False, 1.15)] * 20 + [(False, 2.0)] * 20 + [(True, 2.99)]
    events = mirror_on_path_tracker(seq)
    assert len(events) == 1
    duration, dlat_start, dlat_end = events[0]
    assert duration == 1.0  # 40周期
    assert dlat_start == 1.15
    assert dlat_end == 2.99  # Trueに戻った瞬間のfwd_dlat


def test_tracker_no_event_while_still_blocked():
    seq = [(False, 1.5)] * 100
    assert mirror_on_path_tracker(seq) == []


def test_tracker_handles_multiple_independent_stalls():
    seq = ([(False, 1.5)] * 10 + [(True, 1.0)] * 5
           + [(False, 2.5)] * 15 + [(True, 1.8)])
    events = mirror_on_path_tracker(seq)
    assert len(events) == 2
    assert events[0] == (10 / CONTROL_RATE_HZ, 1.5, 1.0)
    assert events[1] == (15 / CONTROL_RATE_HZ, 2.5, 1.8)


def test_retroactive_0726_05_monotonic_drift_scenario():
    """遡及検証: 0726-05実測相当(両者静止中にfwd_dlatが1.15m→2.99mへ単調増加、
    約11秒間継続)を再現し、開始/終了dlatの差が計測できることを確認する。"""
    seq = [(False, 1.15 + i * 0.04) for i in range(440)] + [(True, 2.99)]  # 11秒@40Hz
    events = mirror_on_path_tracker(seq)
    assert len(events) == 1
    duration, dlat_start, dlat_end = events[0]
    assert duration == 11.0
    assert dlat_end > dlat_start  # 単調増加(離れる方向)を確認できる


# ---------------------------------------------------------------------------
# ソーステキスト検証: 実際の追加箇所
# ---------------------------------------------------------------------------

def test_source_on_path_tracks_duration_and_dlat():
    idx = _SRC.index("_on_path = (opp_sit.fwd_dlat is not None")
    snippet = _SRC[idx:idx + 1300]
    assert "self._on_path_false_cycles += 1" in snippet
    assert "[ON-PATH-CLEAR]" in snippet
    assert "self._on_path_false_cycles = 0" in snippet
    assert "fwd_dlat_start=" in snippet
    assert "fwd_dlat_end=" in snippet
    assert "fwd_vopp=" in snippet


def test_source_on_path_computation_itself_unchanged():
    """④遡及効果: _on_path自体の判定式(engage_lat_max比較)は無変更のまま。"""
    idx = _SRC.index("_on_path = (opp_sit.fwd_dlat is not None")
    snippet = _SRC[idx:idx + 120]
    assert "opp_sit.fwd_dlat <= self._ot_engage_lat_max)" in snippet


def test_source_no_new_config_parameters():
    """②非冗長性: 190-6節は診断ロギングのみで新規config.yamlパラメータを
    導入していない。"""
    idx = _SRC.index("_on_path = (opp_sit.fwd_dlat is not None")
    snippet = _SRC[idx:idx + 1300]
    assert "_otget(" not in snippet
