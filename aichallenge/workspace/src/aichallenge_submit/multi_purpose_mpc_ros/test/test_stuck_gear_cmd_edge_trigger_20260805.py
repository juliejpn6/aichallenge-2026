"""GearCommandパブリッシュのエッジトリガー化(293節続報、2026-08-05)。

背景: 293節で、ローカル環境のREVERSE確認成功率が1%未満(soloでもdev3でも同様、
CPU負荷は主因でないと確認済み)と判明し、gear_settle_cycles/gear_park_dwell_cyclesを
2倍へ拡大したが、dev3実測(run7)ではこの拡大だけでは明確な改善が確認できなかった。
次の仮説として、_handle_stuck_recovery内でGearCommandを毎周期(40Hz)無条件で
連続再パブリッシュし続けている動作自体が、AWSIM側のギアシフト処理と干渉している
可能性を検証するため、指令値が変化した瞬間(エッジ)は即座に送信し、変化が無い間は
gear_cmd_heartbeat_cycles周期ごとに間引いて再送する方式へ変更した(完全な単発publish
は配信ロス時のリスクがあるため採用せず、低頻度ハートビート再送を残す)。

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


def _throttled_publish_step(command, last, heartbeat_count, heartbeat_cycles):
    """_publish_gear_cmd_throttled()のミラー実装。
    戻り値: (published: bool, next_last, next_heartbeat_count)。"""
    heartbeat_count += 1
    changed = (command != last)
    if not (changed or heartbeat_count >= heartbeat_cycles):
        return False, last, heartbeat_count
    return True, command, 0


# ---------------------------------------------------------------------------
# ①ロジック検証: エッジ(変化時)は即座に送信する
# ---------------------------------------------------------------------------

def test_publishes_immediately_on_change():
    published, last, cnt = _throttled_publish_step(
        command=20, last=22, heartbeat_count=0, heartbeat_cycles=8)
    assert published is True
    assert last == 20
    assert cnt == 0


def test_first_call_from_none_publishes_immediately():
    """新規エピソード突入直後(last=Noneへリセット済み)は必ず1回目が即送信される。"""
    published, last, _ = _throttled_publish_step(
        command=22, last=None, heartbeat_count=0, heartbeat_cycles=8)
    assert published is True
    assert last == 22


# ---------------------------------------------------------------------------
# ②ロジック検証: 変化が無い間は間引かれ、heartbeat_cycles周期で再送する
# ---------------------------------------------------------------------------

def test_suppressed_when_unchanged_and_below_heartbeat():
    published, last, cnt = _throttled_publish_step(
        command=20, last=20, heartbeat_count=3, heartbeat_cycles=8)
    assert published is False
    assert last == 20  # 送信していないので変化なし
    assert cnt == 4


def test_republishes_at_heartbeat_boundary():
    published, _, cnt = _throttled_publish_step(
        command=20, last=20, heartbeat_count=7, heartbeat_cycles=8)
    assert published is True
    assert cnt == 0


def test_heartbeat_count_resets_after_publish():
    """公開後は次周期からまた0スタートでカウントし直す(累積しない)。"""
    published, last, cnt = _throttled_publish_step(
        command=20, last=20, heartbeat_count=7, heartbeat_cycles=8)
    assert published and cnt == 0
    # 直後の周期はカウント1から再スタート
    published2, _, cnt2 = _throttled_publish_step(
        command=20, last=last, heartbeat_count=cnt, heartbeat_cycles=8)
    assert published2 is False
    assert cnt2 == 1


# ---------------------------------------------------------------------------
# ③ソーステキスト構造検証: ヘルパーの実装
# ---------------------------------------------------------------------------

def test_helper_method_exists():
    assert "def _publish_gear_cmd_throttled(self, now, command: int) -> None:" in _SRC


def test_helper_implements_edge_or_heartbeat_condition():
    idx = _SRC.index("def _publish_gear_cmd_throttled(")
    idx_end = _SRC.index("\n    def _handle_stuck_recovery(", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_gear_cmd_heartbeat_count += 1" in snippet
    assert "_changed = (command != self._stuck_gear_cmd_last)" in snippet
    assert "self._gear_cmd_pub.publish(gear_cmd)" in snippet
    assert "self._stuck_gear_cmd_last = command" in snippet
    assert "self._stuck_gear_cmd_heartbeat_count = 0" in snippet


# ---------------------------------------------------------------------------
# ④退行防止: _handle_stuck_recovery内の全呼び出し元がヘルパー経由になっていること
#   (毎周期無条件publishする旧パターンが残っていないこと)
# ---------------------------------------------------------------------------

def test_no_direct_gear_cmd_publish_remains_in_handle_stuck_recovery():
    idx = _SRC.index("def _handle_stuck_recovery(self, now, pose) -> None:")
    idx_end = _SRC.index("\n    def _stuck_target_steer(", idx) if \
        "\n    def _stuck_target_steer(" in _SRC[idx:] else len(_SRC)
    # _handle_stuck_recovery本体の範囲を、次のメソッド定義までとして粗く取る
    idx_end = _SRC.index("\n    def ", idx + 50)
    while _SRC[idx_end + 8:idx_end + 40].strip().startswith("_") is False:
        # 念のため無限ループ回避、最初のdef境界で確定させる
        break
    snippet = _SRC[idx:idx_end]
    assert "self._gear_cmd_pub.publish(gear_cmd)" not in snippet
    # 2026-08-05追加修正(293節続報): 公式仕様(PARK=22は非対応)判明を受け、WAIT_PARK
    #   ステップの送信値をPARKからNEUTRALへ変更した(状態名"WAIT_PARK"自体は維持)。
    assert snippet.count("self._publish_gear_cmd_throttled(now, GearCommand.NEUTRAL)") == 1
    assert snippet.count("self._publish_gear_cmd_throttled(now, GearCommand.REVERSE)") == 2
    assert snippet.count("self._publish_gear_cmd_throttled(now, GearCommand.DRIVE)") == 2


def test_total_helper_call_count_is_five():
    n = _SRC.count("self._publish_gear_cmd_throttled(now, GearCommand.")
    assert n == 5, f"想定は5箇所(NEUTRAL1・REVERSE2・DRIVE2)だが{n}箇所で見つかった"


# ---------------------------------------------------------------------------
# ⑤新規エピソード突入時のリセット(_stuck_enter_wait_reverse)
# ---------------------------------------------------------------------------

def test_new_episode_entry_resets_gear_cmd_tracking_state():
    idx = _SRC.index("def _stuck_enter_wait_reverse(self, now, pose) -> None:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_gear_cmd_last = None" in snippet
    assert "self._stuck_gear_cmd_heartbeat_count = 0" in snippet


# ---------------------------------------------------------------------------
# ⑥config配線・既定値確認
# ---------------------------------------------------------------------------

def test_config_declares_gear_cmd_heartbeat_cycles_default_8():
    idx = _CFG.index("gear_cmd_heartbeat_cycles:")
    snippet = _CFG[idx:idx + 60]
    assert "gear_cmd_heartbeat_cycles: 8" in snippet


def test_controller_reads_gear_cmd_heartbeat_cycles_with_default_8():
    assert ('"gear_cmd_heartbeat_cycles", int(_stkget("gear_cmd_heartbeat_cycles", 8))'
            in _SRC)


def test_init_declares_tracking_state_variables():
    idx = _SRC.index('self._stuck_gear_wait_count = 0\n            self._stuck_gear_cmd_last')
    snippet = _SRC[idx:idx + 300]
    assert "self._stuck_gear_cmd_last = None" in snippet
    assert "self._stuck_gear_cmd_heartbeat_count = 0" in snippet
