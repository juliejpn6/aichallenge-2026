"""Regression tests for the STUCK-recovery infeasibility_counter freeze fix (90節, 2026-07-17).

Background: 0717-01ログ実測で、_control()が_stuck_state != "NORMAL"の間
get_control()を一切呼ばないため、self._mpc.infeasibility_counter(core/MPC.py側の
カウンタ)が復帰処理中ずっと凍結されることが判明した。復帰処理が完了しNORMALへ
戻っても、次周期のSTUCK再判定(経路2: infeasibility_counter>=stuck_infeas_thr(300))は
MPCが一度も再実行される前にこの凍結値(既に300以上)を見て即座に再発火してしまい、
0717-01では207回連鎖・332秒間MPCが一度も再稼働しないまま記録が終わっていた
(BACKUP-BLOCKED完了の14ms後に次のSTUCK detectedが記録されている)。

対処(案C、ユーザー承認済み設計): 復帰完了(NORMALへ戻る)箇所全てで
self._mpc.infeasibility_counter = 0 を明示的にリセットする。新規状態変数・
新規判定分岐は追加せず、既存の「300周期(≈7.5秒)継続したらSTUCK」という
判定式の意味(=復帰後に実際にどれだけ解けなかったか)を守るのみ。

2026-07-24更新(171節続報): 従来は復帰完了箇所が4箇所(BACKUP-BLOCKED断念/
BACKUP-TIMEOUT予算超過/PUSH完了/WAIT_DRIVE完了)あったが、経路1/2専用の
WAIT_DRIVE(ステア0固定の直進復帰)がPUSH(低速+最大舵角の回避走行)へ統合され
到達不能になったため削除され、3箇所になった。

_handle_stuck_recovery()はGearCommand/GearReport(ROS message types)に依存し
単体で import できないため、test_stuck_backup_blocked.py と同じ方針で
(1) ソーステキスト上の構造的検証、(2) 検知条件そのものの純Pythonミラーによる
遡及検証、の2本立てで確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

# _handle_stuck_recovery内で self._stuck_state = "NORMAL" へ遷移する3箇所の
# 直前にある一意なアンカー文字列(それぞれの分岐を特定するため)。
_NORMAL_TRANSITION_ANCHORS = [
    "無理に後退せず停止し復帰断念、NORMAL(通常のMPC/ICC)へ委譲",  # BACKUP-BLOCKED
    "リトライ予算(backup_retry_budget_s=",                        # BACKUP-TIMEOUT予算超過
    "PUSH終了(reason={_reason} dist={dist:.2f}m elapsed={elapsed:.1f}s)",  # PUSH完了(経路1/2/3共通)
]


def test_all_three_normal_transitions_exist_in_source():
    """前提確認: 3箇所のアンカー文字列が全てソース中に(重複なく)存在する。
    アンカー自体が見つからない場合、以降のテストが偽陽性になるため先に確認する。"""
    for anchor in _NORMAL_TRANSITION_ANCHORS:
        assert _SRC.count(anchor) == 1, f"anchor not found exactly once: {anchor!r}"


def test_all_three_normal_transitions_reset_infeasibility_counter():
    """回帰の核心: STUCK復帰が完了しNORMALへ戻る3箇所全てで、カウンタリセットが
    実行されることを確認する。
    2026-07-21修正(148節②): 4箇所(当時)の個別実装が_stuck_recovery_complete()へ
    統合されたため、アンカー直後では直接の代入ではなく共通ヘルパー呼び出し
    (_stuck_recovery_complete(...))を確認する。ヘルパー自体が必ず
    infeasibility_counter=0を実行することはtest_stuck_recovery_complete_helper
    側の専用テストで別途保証する。"""
    for anchor in _NORMAL_TRANSITION_ANCHORS:
        idx = _SRC.index(anchor)
        snippet = _SRC[idx:idx + 700]
        assert "self._stuck_recovery_complete(" in snippet, (
            f"missing _stuck_recovery_complete() call near anchor: {anchor!r}")


def test_exactly_four_stuck_recovery_complete_call_sites():
    """回帰: _stuck_recovery_complete()の呼び出しがちょうど4箇所であることを
    確認する(将来の編集で5箇所目が増えたり、既存の1箇所が削られたりする
    変化を検知するため。定義自体の1箇所は含まないよう呼び出し構文で数える)。
    2026-08-05追加(291節): STUCKエスカレーション欠如バグの安全網
    (_stuck_enter_wait_reverse内、shuffle_hard_limit到達時の断念)で4箇所目が
    増えた——経路非依存で復帰を断念する新しい正当な呼び出し元のため、3→4へ更新。"""
    assert _SRC.count("self._stuck_recovery_complete(") == 4


def test_stuck_recovery_complete_helper_always_resets_counter_unconditionally():
    """_stuck_recovery_complete()自体(148節②で新設)が、reset_backup_state/
    reset_corridorの値に関わらず必ずinfeasibility_counter=0を実行し、
    [STUCK-COUNTER-RESET]をログすることを確認する(90節の意図=復帰完了なら
    常にリセットする、を守っている)。"""
    idx = _SRC.index("def _stuck_recovery_complete(")
    idx_end = _SRC.index("def _handle_stuck_recovery(")
    snippet = _SRC[idx:idx_end]
    assert "self._mpc.infeasibility_counter = 0" in snippet
    assert "[STUCK-COUNTER-RESET]" in snippet
    # 条件分岐(if reset_backup_state/if reset_corridor)の外側(=無条件)にあることを、
    # 最後のif行より後ろに現れる位置関係で確認する。
    idx_last_if = snippet.rindex("if reset_corridor:")
    idx_counter_reset = snippet.index("self._mpc.infeasibility_counter = 0")
    assert idx_counter_reset > idx_last_if


# ---------------------------------------------------------------------------
# 遡及検証: 0717-01実測(BACKUP-BLOCKED完了の14ms後に即再発火)を純Pythonで再現する
# ---------------------------------------------------------------------------
STUCK_INFEAS_THR = 300
STARTUP_GRACE_S = 10.0


def _infeas_stuck(infeasibility_counter, since_start_s):
    """_control()内の_infeas_stuck計算式のミラー(mpc_controller.py該当行と同一)。"""
    return since_start_s >= STARTUP_GRACE_S and infeasibility_counter >= STUCK_INFEAS_THR


def test_retroactive_0717_01_without_reset_would_instantly_refire():
    """遡及検証(修正前の挙動): 0717-01実測の通り、infeasibility_counterが
    凍結されたまま(=300)NORMALへ戻ると、直後のSTUCK再判定は(get_control()が
    一度も挟まらなくても)即座にTrueになる。これが実際に観測された14ms再発火の
    仕組みそのものであることを示す。"""
    frozen_counter_from_before_recovery = 300  # 実測: 復帰前から常にちょうど300
    since_start_s = 220.0  # 起動猶予(10s)は当然超過済み
    assert _infeas_stuck(frozen_counter_from_before_recovery, since_start_s) is True


def test_retroactive_0717_01_with_reset_gives_mpc_a_real_grace_window():
    """遡及検証(修正後の挙動): 復帰完了時にinfeasibility_counter=0へリセットすると、
    NORMAL復帰直後の周期ではSTUCK再判定はFalseになり、MPCに実際の解を試す猶予が
    生まれる。カウンタが0から再度閾値に達するまでには、既存の300周期(≈7.5秒)分の
    「本当の」連続失敗が必要になる(既存の判定式・閾値は一切変更していない)。"""
    reset_counter = 0
    since_start_s = 220.0
    assert _infeas_stuck(reset_counter, since_start_s) is False
    # 閾値未満の任意の値でも同様にFalseのまま(境界のちょうど1つ手前まで確認)。
    assert _infeas_stuck(STUCK_INFEAS_THR - 1, since_start_s) is False
    # 逆に、本当に閾値へ到達すれば(=リセット後も実際に300周期解けなかった場合)、
    # 従来通り正しく検知される(既存の安全機構を弱めていないことの確認)。
    assert _infeas_stuck(STUCK_INFEAS_THR, since_start_s) is True
