"""STUCKエスカレーション欠如バグの経路非依存安全網(291節、2026-08-05)。

背景: 0804(q25_r2、design_docs [[stuck-escalation-gap-confirmed-full-run-loss-0804]])
・0805(tau n=2積み増し中run5)の2回にわたり、STUCK復帰が同一地点で永久ループし
run全体(0805は約30分)を喪失する実害を確認した。

根本原因: shuffle_max_cycles/max_giveup_streakによるエスカレーション上限は
BACKUP-BLOCKED(後退不能)経路にしか実装されておらず、PUSH timeout(実測移動量
dist=0.00mでも`elapsed >= push_timeout_s`だけで無条件に完了扱いされる経路)を
繰り返すケースには上限チェックが一切なかった。この経路は`_stuck_recovery_complete`
を呼んで一旦NORMALへ「完了」扱いで戻るが、車が動いていないため直後にSTUCKが
再検知され、`_stuck_update_shuffle_cycle`が「同一エピソード継続」と判定して
shuffle_cycleを際限なくインクリメントし続ける(実測cycle=138まで確認)。

対処: どの経路でshuffle_cycleが増えたかに関わらず、_stuck_enter_wait_reverse()
(STUCK検知の全3経路[経路1/2/3]が必ず経由する唯一の入口)でハード上限
(shuffle_hard_limit、既定20)をチェックし、到達していれば復帰そのものを断念して
NORMALへ委譲する。既存のBACKUP-BLOCKED専用エスカレーション(shuffle_max_cycles/
max_giveup_streak)には一切手を入れない(通常はこちらが先に到達し、ハード上限は
それより緩い経路[PUSH timeout繰り返し等]に対する最終安全網としてのみ機能する)。

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


def _enter_wait_reverse_step(shuffle_cycle_after_update, hard_limit):
    """_stuck_enter_wait_reverse()冒頭のミラー実装。_stuck_update_shuffle_cycle()
    呼び出し後のshuffle_cycle値を引数で直接与える(インクリメント自体は184節の
    既存ロジックで別途テスト済みのため、ここでは安全網の分岐のみを検証する)。
    戻り値: (abandoned: bool, next_state, shuffle_cycle_reset_to)。"""
    if shuffle_cycle_after_update >= hard_limit:
        return True, "NORMAL", 0
    return False, "WAIT_PARK", shuffle_cycle_after_update


# ---------------------------------------------------------------------------
# ①非矛盾性: 上限未満では通常通りWAIT_PARKへ進む
# ---------------------------------------------------------------------------

def test_below_hard_limit_proceeds_to_wait_park():
    abandoned, nxt, cycle = _enter_wait_reverse_step(shuffle_cycle_after_update=19, hard_limit=20)
    assert not abandoned
    assert nxt == "WAIT_PARK"
    assert cycle == 19


def test_far_below_hard_limit_proceeds_normally():
    abandoned, nxt, _ = _enter_wait_reverse_step(shuffle_cycle_after_update=1, hard_limit=20)
    assert not abandoned
    assert nxt == "WAIT_PARK"


# ---------------------------------------------------------------------------
# ②境界値: ちょうど上限に達したら断念する(以上、超過ではない)
# ---------------------------------------------------------------------------

def test_exactly_at_hard_limit_abandons():
    abandoned, nxt, cycle = _enter_wait_reverse_step(shuffle_cycle_after_update=20, hard_limit=20)
    assert abandoned
    assert nxt == "NORMAL"
    assert cycle == 0  # 断念後はshuffle_cycleを0へリセットし次のエピソードに引きずらない


def test_beyond_hard_limit_abandons():
    abandoned, nxt, _ = _enter_wait_reverse_step(shuffle_cycle_after_update=138, hard_limit=20)
    assert abandoned
    assert nxt == "NORMAL"


# ---------------------------------------------------------------------------
# 遡及効果: ソーステキスト構造検証(実装が経路非依存の安全網として正しく
# 配置されていることを確認する)
# ---------------------------------------------------------------------------

def test_hard_limit_check_immediately_follows_shuffle_cycle_update():
    """_stuck_update_shuffle_cycle()呼び出しの直後(WAIT_PARK突入より前)に
    ハード上限チェックが配置されていることを確認する。"""
    idx = _SRC.index("def _stuck_enter_wait_reverse(self, now, pose) -> None:")
    idx_update = _SRC.index("self._stuck_update_shuffle_cycle(now, pose)", idx)
    idx_check = _SRC.index("self._stuck_shuffle_cycle >= self._stuck_shuffle_hard_limit", idx_update)
    idx_wait_park = _SRC.index('self._stuck_state = "WAIT_PARK"', idx_update)
    assert idx_update < idx_check < idx_wait_park


def test_hard_limit_abandon_calls_stuck_recovery_complete():
    idx = _SRC.index("self._stuck_shuffle_cycle >= self._stuck_shuffle_hard_limit")
    idx_end = _SRC.index('self._stuck_state = "WAIT_PARK"', idx)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_recovery_complete(" in snippet
    assert "reset_backup_state=True" in snippet
    assert "reset_corridor=True" in snippet
    assert "return" in snippet


def test_hard_limit_abandon_resets_shuffle_counters():
    """断念後、shuffle_cycle/giveup_streak/push_side_flipを0/False相当へ戻し、
    次の(新規とみなされる)エピソードへ古い状態を持ち越さないことを確認する。"""
    idx = _SRC.index("self._stuck_shuffle_cycle >= self._stuck_shuffle_hard_limit")
    idx_end = _SRC.index("self._stuck_recovery_complete(", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_shuffle_cycle = 0" in snippet
    assert "self._stuck_giveup_streak = 0" in snippet
    assert "self._stuck_push_side_flip = False" in snippet


def test_hard_limit_does_not_touch_existing_backup_blocked_escalation():
    """既存のBACKUP-BLOCKED専用エスカレーション(shuffle_max_cycles/
    max_giveup_streak)のロジックには一切手を入れていないことを確認する
    (今回の変更が_stuck_enter_wait_reverse側にのみ閉じていること)。"""
    idx_backup_blocked = _SRC.index(
        "elif _zero_v_elapsed >= self._stuck_backup_blocked_confirm_s or _stuck_backup_impact:")
    idx_end = _SRC.index("elif _backup_elapsed >= self._stuck_backup_timeout_s:", idx_backup_blocked)
    snippet = _SRC[idx_backup_blocked:idx_end]
    assert "shuffle_hard_limit" not in snippet
    assert "self._stuck_shuffle_cycle < self._stuck_shuffle_max_cycles" in snippet


def test_config_declares_shuffle_hard_limit_default_20():
    idx = _CFG.index("shuffle_hard_limit:")
    snippet = _CFG[idx:idx + 60]
    assert "shuffle_hard_limit: 20" in snippet


def test_controller_reads_shuffle_hard_limit_with_default_20():
    idx = _SRC.index('self._stuck_shuffle_hard_limit = int(_stkget("shuffle_hard_limit"')
    snippet = _SRC[idx:idx + 80]
    assert "20" in snippet
