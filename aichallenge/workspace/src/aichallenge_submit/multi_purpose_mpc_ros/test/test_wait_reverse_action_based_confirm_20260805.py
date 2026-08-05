"""WAIT_REVERSEの行動ベース確認方式への転換(294節続報、2026-08-05)。

背景: 293-294節でgear_settle_cycles拡大(20→50→200)・エッジトリガー化・中間ギア
PARK→NEUTRAL修正と対処を重ねたが、REVERSE自体のGearReport確認成功率は
dev3実測で一貫して0%のままだった(NEUTRALは100%まで改善)。外部AI(Gemini/Claude)へ
相談した結果、最有力仮説として「WAIT_REVERSEが速度指令ゼロで確認を待つ設計自体が、
AWSIM側の『シフト受理後も運動指令が継続しないと静止+無入力とみなし自動的にPARKへ
戻す』可能性のある挙動と根本的に相性が悪い」という分析を得た(手動操作は
「Rに入れて即アクセル」の連続動作であり confirm待ちの空白が無いこととも整合)。

対処: WAIT_REVERSEでGearReport確認を待たず、REVERSE要求と同時に後退運動指令
(backup_speed)を送り始める。成功シグナルを「GearReport==REVERSE確認」
「実速度が後退方向へ動き始めた(reverse_move_confirm_v以下)」「タイムアウト」の
3種類に拡張し、いずれか1つでも満たせばBACKUPへ遷移する(既存の確認方式・
タイムアウト方式は削除せず、行動ベースの確認を追加する形)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ロジックのミラー実装+ソーステキスト検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG = _f.read()


def _wait_reverse_transition_step(gear_report_is_reverse, v_now, wait_count,
                                   settle_cycles, move_confirm_v):
    """WAIT_REVERSE分岐の遷移判定のミラー実装。
    戻り値: (transitions_to_backup: bool, reason: str)。"""
    confirmed = gear_report_is_reverse
    moving_backward = v_now <= move_confirm_v
    if confirmed or moving_backward or wait_count >= settle_cycles:
        reason = ("confirmed" if confirmed else
                  "moving_backward" if moving_backward else "timeout")
        return True, reason
    return False, None


# ---------------------------------------------------------------------------
# ①ロジック検証: 3種類の成功シグナルそれぞれが単独で遷移を引き起こす
# ---------------------------------------------------------------------------

def test_gear_report_confirmed_alone_transitions():
    ok, reason = _wait_reverse_transition_step(
        gear_report_is_reverse=True, v_now=0.0, wait_count=1,
        settle_cycles=200, move_confirm_v=-0.05)
    assert ok and reason == "confirmed"


def test_moving_backward_alone_transitions_even_without_gear_confirm():
    """行動ベース確認の核心: GearReportが未確認でも、実速度が後退していれば遷移する。"""
    ok, reason = _wait_reverse_transition_step(
        gear_report_is_reverse=False, v_now=-0.10, wait_count=1,
        settle_cycles=200, move_confirm_v=-0.05)
    assert ok and reason == "moving_backward"


def test_timeout_alone_transitions_when_neither_signal_fires():
    ok, reason = _wait_reverse_transition_step(
        gear_report_is_reverse=False, v_now=0.0, wait_count=200,
        settle_cycles=200, move_confirm_v=-0.05)
    assert ok and reason == "timeout"


def test_no_signal_does_not_transition():
    ok, _ = _wait_reverse_transition_step(
        gear_report_is_reverse=False, v_now=0.0, wait_count=50,
        settle_cycles=200, move_confirm_v=-0.05)
    assert not ok


def test_positive_velocity_does_not_count_as_moving_backward():
    """前進方向の速度(正値)は後退確認にカウントしない。"""
    ok, reason = _wait_reverse_transition_step(
        gear_report_is_reverse=False, v_now=0.10, wait_count=1,
        settle_cycles=200, move_confirm_v=-0.05)
    assert not ok


def test_boundary_exactly_at_move_confirm_v_counts():
    ok, reason = _wait_reverse_transition_step(
        gear_report_is_reverse=False, v_now=-0.05, wait_count=1,
        settle_cycles=200, move_confirm_v=-0.05)
    assert ok and reason == "moving_backward"


# ---------------------------------------------------------------------------
# ②ソーステキスト構造検証: WAIT_REVERSEが常に後退運動指令を送ること
# ---------------------------------------------------------------------------

def _wait_reverse_body():
    idx = _SRC.index('elif self._stuck_state == "WAIT_REVERSE":')
    idx_end = _SRC.index('elif self._stuck_state == "BACKUP":')
    return _SRC[idx:idx_end]


def test_wait_reverse_always_commands_backup_motion_unconditionally():
    """u/accの代入が、if/elseの分岐の外側(=常に実行される位置)にあることを、
    最後のif関連行より後ろに現れる位置関係で確認する。"""
    snippet = _wait_reverse_body()
    idx_last_if_body = snippet.rindex("後方の相手車を考慮し後退距離を")
    idx_u_assign = snippet.index("u = [self._stuck_backup_speed, 0.0]")
    idx_acc_assign = snippet.index("acc = self._stuck_backup_accel")
    assert idx_u_assign > idx_last_if_body
    assert idx_acc_assign > idx_last_if_body
    # 退行防止: 旧来のゼロ保持パターン(hold_accelでの待機)がWAIT_REVERSE本体に
    #   残っていないこと(WAIT_PARK側は別途0.0保持へ変更済みのため対象外)。
    assert "acc = self._stuck_hold_accel" not in snippet


def test_wait_reverse_checks_velocity_feedback():
    snippet = _wait_reverse_body()
    assert "self._odom.twist.twist.linear.x" in snippet
    assert "self._stuck_reverse_move_confirm_v" in snippet
    assert "_moving_backward = _v_now <= self._stuck_reverse_move_confirm_v" in snippet


def test_wait_reverse_transition_condition_includes_all_three_signals():
    snippet = _wait_reverse_body()
    idx = snippet.index("if _confirmed or _moving_backward or self._stuck_gear_wait_count")
    assert idx >= 0


def test_wait_park_kept_zero_hold_not_backup_motion():
    """WAIT_PARK(NEUTRAL要求フェーズ)は本節の対象外であり、引き続き速度ゼロで
    待機すること(後退運動を送るのはWAIT_REVERSE以降のみ)。"""
    idx_park = _SRC.index('if self._stuck_state == "WAIT_PARK":')
    idx_reverse = _SRC.index('elif self._stuck_state == "WAIT_REVERSE":')
    snippet = _SRC[idx_park:idx_reverse]
    assert "u = [0.0, 0.0]" in snippet
    assert "acc = 0.0" in snippet


# ---------------------------------------------------------------------------
# ③config配線・既定値確認
# ---------------------------------------------------------------------------

def test_config_declares_reverse_move_confirm_v_default():
    idx = _CFG.index("reverse_move_confirm_v:")
    snippet = _CFG[idx:idx + 60]
    assert "reverse_move_confirm_v: -0.05" in snippet


def test_controller_reads_reverse_move_confirm_v_with_default():
    assert ('self._stuck_reverse_move_confirm_v = float(\n'
            '                _stkget("reverse_move_confirm_v", -0.05))' in _SRC
            or '_stkget("reverse_move_confirm_v", -0.05)' in _SRC)
