"""WAIT_REVERSEのゼロ加速度待機方式(294節続報の再訂正、2026-08-05)。

背景: 293-294節でgear_settle_cycles拡大・エッジトリガー化・中間ギアPARK→NEUTRAL
修正と対処を重ねたが、REVERSE自体のGearReport確認成功率はdev3実測で一貫して0%の
ままだった(NEUTRALは100%まで改善)。外部AI(Gemini/Claude)へ相談した結果、
「WAIT_REVERSEが速度指令ゼロで確認を待つ設計自体が、AWSIM側の『シフト受理後も
運動指令が継続しないと静止+無入力とみなし自動的にPARKへ戻す』可能性のある挙動と
相性が悪い」という仮説を得て、確認を待たずREVERSE要求と同時に後退運動指令を送る
「行動ベース確認方式」へ一度転換した(コミット809cb09)。

しかし別の外部AI(TPACさん)から「t=0のREVERSEはManual操作の残存の可能性がある」
「longitudinal.speedはAWSIM未使用」という重要な指摘を受け、本線コントローラを
完全に停止した最小再現ノードで因果分離試験を実施した。結果:
- N→Rの遷移をacceleration=0.0固定(運動指令なし)で完了させた場合 → Rが安定して
  5秒以上維持された
- 確認前から同時にacceleration!=0(運動指令あり)を送った場合 → 1秒以内に
  強制的にPARKへ戻った

これにより「行動ベース確認方式」(確認前から運動指令を送る)は**逆効果**であり、
AWSIM側は「ギア遷移中に運動指令が混ざっている」ことを拒否条件としている
(実車のインターロックに近い挙動)と判明した。本節で、WAIT_REVERSEを
「confirmするまでは速度・加速度とも完全ゼロで待つ」設計へ再訂正する
(中間ギアがNEUTRAL[293節続報で修正済み]である点は288節以前の原設計と異なる)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _wait_reverse_body():
    idx = _SRC.index('elif self._stuck_state == "WAIT_REVERSE":')
    idx_end = _SRC.index('elif self._stuck_state == "BACKUP":')
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①ソーステキスト構造検証: confirm前は完全ゼロで待機すること
# ---------------------------------------------------------------------------

def test_wait_reverse_holds_zero_before_confirmed():
    """confirmedでもタイムアウトでもない(else)分岐は、速度・加速度とも
    完全ゼロで待機する(294節続報の再訂正の核心)。"""
    snippet = _wait_reverse_body()
    idx_else = snippet.rindex("else:")
    tail = snippet[idx_else:]
    assert "u = [0.0, 0.0]" in tail
    assert "acc = 0.0" in tail


def test_wait_reverse_only_sends_backup_motion_after_transition_decided():
    """後退運動指令(backup_speed/backup_accel)は、状態遷移(BACKUPへ進む)が
    決定したif分岐の内側でのみ送られ、confirm前のelse分岐には無いこと。"""
    snippet = _wait_reverse_body()
    idx_else = snippet.rindex("else:")
    head = snippet[:idx_else]
    tail = snippet[idx_else:]
    assert "u = [self._stuck_backup_speed, 0.0]" in head
    assert "acc = self._stuck_backup_accel" in head
    assert "self._stuck_backup_speed" not in tail
    assert "self._stuck_backup_accel" not in tail


def test_wait_reverse_transition_condition_is_confirmed_or_timeout_only():
    """294節続報で追加した行動ベース確認(moving_backward)は、確認前から
    運動指令を送ることを前提とするため本節の再訂正で削除された。
    confirmedとタイムアウトの2条件のみへ戻っていることを確認する。"""
    snippet = _wait_reverse_body()
    assert 'if _confirmed or self._stuck_gear_wait_count >= self._stuck_gear_settle_cycles:' in snippet
    assert "_moving_backward" not in snippet
    assert "reverse_move_confirm_v" not in snippet.lower() or "self._stuck_reverse_move_confirm_v" not in snippet


def test_wait_reverse_still_publishes_reverse_gear_request():
    """ゼロ加速度待機に戻っても、REVERSE要求自体(エッジトリガー化ヘルパー経由)は
    引き続き毎周期行われること(293節続報の成果は維持)。"""
    snippet = _wait_reverse_body()
    assert "self._publish_gear_cmd_throttled(now, GearCommand.REVERSE)" in snippet


def test_wait_reverse_still_checks_gear_report_confirmed():
    snippet = _wait_reverse_body()
    assert "_confirmed = (self._gear_report.report == GearReport.REVERSE)" in snippet


# ---------------------------------------------------------------------------
# ②最小再現ノードでの因果分離試験結果を裏付ける回帰(config定義は残置、
#   本線ロジックからは参照されなくなったことを明示的に確認)
# ---------------------------------------------------------------------------

def test_reverse_move_confirm_v_config_read_removed_from_wait_reverse_logic():
    """_stuck_reverse_move_confirm_vの読み込み自体(__init__)は将来の再利用に
    備えて残すが、WAIT_REVERSEの遷移判定からは参照されなくなったことを確認する。"""
    assert 'self._stuck_reverse_move_confirm_v = float(' in _SRC
    snippet = _wait_reverse_body()
    assert "self._stuck_reverse_move_confirm_v" not in snippet
