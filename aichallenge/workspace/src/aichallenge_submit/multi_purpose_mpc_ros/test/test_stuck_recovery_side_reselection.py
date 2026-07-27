"""Structural regression guard for STUCK-recovery side re-selection (123節, 2026-07-19).

mpc_controller.py imports rclpy/autoware message types at module scope, and
`_handle_stuck_recovery` is a ~230-line state machine deep inside the class with
many ROS-typed free variables (GearCommand/GearReport/Odometry), so full
instantiation / execution is impractical (same rationale as
test_stuck_backup_blocked.py and test_switchback_token_wiring.py before it).
This file does a structural source-text check instead.

Background: 0719-05実測(qualifying log)で、wp332-333にてBACKUP+PUSHサイクルが
4回連続で発生した。ユーザーが目視で確認し、STUCK復帰(BACKUP/PUSH)はステア0固定の
直進のみで側の再検討を一切行わないため、復帰後に self._ot_side がSTUCK発生時のまま
引き継がれ、毎回同じ側へ再突入して同一地点で繰り返しSTUCKしていたことが判明した。
ユーザー指示:「スタックしてバックしたときには改めて壁との距離、相手の位置を把握し、
一番隙間が広いサイドを選択できるようにしましょう」。

対処: 新規ヘルパー `_reset_ot_side_for_fresh_replan()` を追加し、STUCK復帰が完了する
4箇所すべてで呼び出す。これは既存の「前方クリア連続」「infeasible」によるNORMAL復帰
(3548-3554行目等)と全く同じ6変数のリセットセットを再利用しており、次周期の
ENGAGE判定(_cheap_ok→_plan_pass、壁+相手車位置ベースの側選択)が自然に再実行される。
新規の側選択ロジックは追加していない(_plan_pass自体は無変更)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_reset_helper_exists_and_resets_the_full_normal_reentry_set():
    """_reset_ot_side_for_fresh_replan()が、既存の「前方クリア連続」NORMAL復帰
    (3548-3554行目相当)と同じ6変数(_ot_state/_ot_side/_ot_side_locked/
    _ot_worth_count/_ot_giveup_count/_ot_cleared)をすべてリセットしていることを
    確認する。一部だけのリセットは、古い側コミット・古いworth/giveupカウントが
    残存し「見かけ上は再評価したが実質的に前回と同じ判断へ収束する」不完全な
    再評価になりうるため、完全一致を要求する。"""
    idx = _SRC.index("def _reset_ot_side_for_fresh_replan(self)")
    idx_end = _SRC.index("def _handle_stuck_recovery(self", idx)
    snippet = _SRC[idx:idx_end]
    assert 'self._ot_state = "NORMAL"' in snippet
    assert "self._ot_side = 0" in snippet
    assert "self._ot_side_locked = 0" in snippet
    assert "self._ot_worth_count = 0" in snippet
    assert "self._ot_giveup_count = 0" in snippet
    assert "self._ot_cleared = False" in snippet


def test_reset_helper_is_called_at_all_stuck_recovery_exit_points():
    """_stuck_state = "NORMAL"(STUCK復帰完了、状態機械からの離脱)が発生する
    箇所は3つあり(BACKUP-BLOCKED断念/BACKUP-TIMEOUT予算超過断念/PUSH完了)、
    そのそれぞれで_reset_ot_side_for_fresh_replan()の効果が及ぶことを確認する。
    2026-07-21修正(148節②): 当時4箇所あった個別実装が共通ヘルパー
    _stuck_recovery_complete()へ統合され、_stuck_state="NORMAL"への遷移も
    _reset_ot_side_for_fresh_replan()の呼び出しも両方このヘルパー内の1箇所に
    集約された。_handle_stuck_recovery側では「全箇所がこのヘルパーを
    呼んでいるか」を、ヘルパー側では「呼ばれれば必ず両方を実行するか」を
    それぞれ確認する(2段階に分けて検証、直接文字列を数える一段階の検証は
    もはやヘルパーの意義=1箇所に集約したこと自体を裏付けられないため)。
    2026-07-24更新(171節続報): 経路1/2専用だったWAIT_DRIVE(ステア0固定の直進
    復帰)がPUSH(低速+最大舵角の回避走行)へ統合され到達不能になったため削除
    され、4箇所→3箇所になった。"""
    idx_def = _SRC.index("def _handle_stuck_recovery(self, now, pose)")
    idx_def_end = _SRC.index("\n    def ", idx_def + 10)
    body = _SRC[idx_def:idx_def_end]
    n_calls = body.count("self._stuck_recovery_complete(")
    assert n_calls == 3, (
        f"想定していた3箇所のNORMAL遷移から数が変わっている(現在{n_calls}箇所)。"
        "新しい遷移経路が追加/削除された場合はこのテスト自体の更新も必要。")

    idx_helper = _SRC.index("def _stuck_recovery_complete(")
    idx_helper_end = _SRC.index("def _handle_stuck_recovery(")
    helper_body = _SRC[idx_helper:idx_helper_end]
    assert 'self._stuck_state = "NORMAL"' in helper_body
    assert helper_body.count("self._reset_ot_side_for_fresh_replan()") == 1


def test_reset_helper_not_called_from_normal_overtaking_side_flip_paths():
    """回帰(非干渉性): _reset_ot_side_for_fresh_replan()はSTUCK復帰専用の
    _stuck_recovery_complete()からのみ呼ばれ、通常のOVERTAKING側フリップ・
    switchback関連コードからは呼ばれていないことを確認する。STUCK復帰以外の
    経路にまで側リセットが波及すると、正常な追い越し継続中に不要な
    再エンゲージ待ちが発生しうる。
    2026-07-21修正(148節②): 唯一の呼び出し元が_stuck_recovery_complete()
    (_handle_stuck_recoveryより前で定義)になったため、境界を
    _stuck_recovery_complete()の定義開始位置へ変更する。"""
    idx_helper = _SRC.index("def _stuck_recovery_complete(")
    idx_stuck_end = _SRC.index("\n    def ", idx_helper + 10)
    idx_stuck_end = _SRC.index("\n    def ", idx_stuck_end + 10)  # _handle_stuck_recovery本体の終端まで
    before = _SRC[:idx_helper]
    after = _SRC[idx_stuck_end:]
    assert "self._reset_ot_side_for_fresh_replan()" not in before
    assert "self._reset_ot_side_for_fresh_replan()" not in after


def test_reset_helper_reuses_plan_pass_no_new_side_selection_logic():
    """非冗長性の確認: _plan_pass自体(側選択の実ロジック)には123節による変更が
    加わっていないことを、_plan_pass定義ブロック中に123節のマーカーが無いことで
    確認する(_reset_ot_side_for_fresh_replan()は既存のENGAGE経路を再トリガー
    するだけで、独自の側選択式を持たない設計であることの裏付け)。"""
    idx = _SRC.index("def _plan_pass(")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "123節" not in snippet
    assert "_reset_ot_side_for_fresh_replan" not in snippet
