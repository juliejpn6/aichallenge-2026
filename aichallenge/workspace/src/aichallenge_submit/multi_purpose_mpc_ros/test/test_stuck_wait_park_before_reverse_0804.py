"""STUCK復帰: WAIT_REVERSEの前段としてWAIT_PARKを経由する変更(2026-08-04)。

背景: dev3対戦車ありA/B検証中、P1(D1)が壁に激突しSTUCK復帰(BACKUP)が
gear_report=PARK固定のまま一切REVERSEへ遷移せず、30サイクル・100秒以上の
無限ループに陥った。当初は[[local-awsim-reverse-gear-unreliable]]と同種の
ローカルAWSIM環境固有の問題(対処不能)と判断していたが、ユーザーから
「一旦Pレンジに入れてからRレンジに入れると入るらしい」「直接DからRには
入らない」「DからP、PからRにシフトする」という具体的な仕様の指摘があった。

従来の実装は、STUCK検知時に(実際のギアがDRIVEのままであっても)いきなり
GearCommand.REVERSEを送り続けていた——これは「Pを経由しないR要求」に相当し、
指摘された仕様と矛盾する。対処として、WAIT_REVERSEへ突入する前に必ず
WAIT_PARK(PARKを送りgear_settle_cycles確定待ち)を経由するよう変更した。

2026-08-04同日追記: 上記修正を実地投入したところ、D→P(WAIT_PARK)は
5件とも成功したがP→R(WAIT_REVERSE)は5件とも失敗した。ログ調査の結果、
WAIT_PARK突入直後(1周期未満)にgear_report.report==PARKが既に真になって
おり、confirmed扱いで即座に通過していたことが判明——前エピソードの残存
状態を拾っているだけの疑いが強い。confirmedを信用せず、
gear_park_dwell_cycles周期は無条件で滞留してからWAIT_REVERSEへ進むよう
再修正した(新規configパラメータ導入、意図的な設計変更——2026-08-04時点の
旧テストが要求していた「新規パラメータなし」の非冗長性方針は、今回の実測
知見によりこの箇所に限り撤回する)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装
した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _wait_park_step(gear_report_is_park, wait_count, dwell_cycles):
    """WAIT_PARKブロックのミラー実装(2026-08-04再修正版)。confirmedはログの
    表示にのみ使い、遷移条件からは外す(常にdwell_cycles周期を無条件で待つ)。
    戻り値: (next_state, u, acc, new_wait_count)。"""
    wait_count += 1
    if wait_count >= dwell_cycles:
        return "WAIT_REVERSE", [0.0, 0.0], "hold", 0
    return "WAIT_PARK", [0.0, 0.0], "hold", wait_count


# --- ①非矛盾性: gear_report=PARK確認だけでは進まない(即時confirmedを信用しない) ---

def test_confirmed_park_does_not_advance_before_dwell_elapsed():
    nxt, u, acc, wc = _wait_park_step(gear_report_is_park=True, wait_count=0, dwell_cycles=40)
    assert nxt == "WAIT_PARK"
    assert wc == 1


# --- ②非矛盾性: dwell_cycles到達で確認有無によらず進む(gear_status配信が無い環境でも進行を保証) ---

def test_dwell_elapsed_advances_regardless_of_confirmation():
    nxt, u, acc, wc = _wait_park_step(gear_report_is_park=False, wait_count=39, dwell_cycles=40)
    assert nxt == "WAIT_REVERSE"
    assert wc == 0

    nxt2, _, _, wc2 = _wait_park_step(gear_report_is_park=True, wait_count=39, dwell_cycles=40)
    assert nxt2 == "WAIT_REVERSE"
    assert wc2 == 0


def test_not_dwell_elapsed_stays_in_wait_park():
    nxt, u, acc, wc = _wait_park_step(gear_report_is_park=False, wait_count=5, dwell_cycles=40)
    assert nxt == "WAIT_PARK"
    assert wc == 6


# --- ③車両は静止したまま(WAIT_REVERSE既存分岐と同じu=[0,0]) ---

def test_stays_stopped_while_waiting():
    _, u, acc, _ = _wait_park_step(gear_report_is_park=False, wait_count=0, dwell_cycles=40)
    assert u == [0.0, 0.0]


# --- ④配線確認: ヘルパーの突入先がWAIT_PARK(WAIT_REVERSE直行ではない)であること ---

def test_enter_wait_reverse_helper_actually_enters_wait_park_first():
    idx = _SRC.index("def _stuck_enter_wait_reverse(self, now, pose)")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert 'self._stuck_state = "WAIT_PARK"' in snippet
    assert 'self._stuck_state = "WAIT_REVERSE"' not in snippet


# --- ⑤配線確認: _handle_stuck_recovery内でWAIT_PARKがWAIT_REVERSEより先に処理され、
#     PARKを要求すること(confirmedはログ表示のみに使い、遷移条件には使わない) ---

def test_wait_park_block_precedes_wait_reverse_block_and_requests_park():
    idx_park = _SRC.index('if self._stuck_state == "WAIT_PARK":')
    idx_reverse = _SRC.index('elif self._stuck_state == "WAIT_REVERSE":')
    assert idx_park < idx_reverse
    snippet = _SRC[idx_park:idx_reverse]
    # 2026-08-05更新(293節続報): GearCommandパブリッシュがエッジトリガー化ヘルパー
    #   (_publish_gear_cmd_throttled)経由になったため、直接代入ではなく呼び出しを確認する。
    # さらに同日、公式仕様(command: 1=NEUTRAL/2=DRIVE/20=REVERSE、PARK=22は非対応)が
    #   判明したため、中間ギアをPARKからNEUTRALへ変更した(状態名"WAIT_PARK"自体は維持)。
    assert "self._publish_gear_cmd_throttled(now, GearCommand.NEUTRAL)" in snippet
    assert "self._gear_report.report == GearReport.NEUTRAL" in snippet
    assert 'self._stuck_state = "WAIT_REVERSE"' in snippet
    # 退行防止: WAIT_PARK内でBACKUPへ直接進む旧経路を誤って残していないこと
    assert 'self._stuck_state = "BACKUP"' not in snippet
    # 2026-08-04再修正: confirmedを遷移条件のor節に使っていないこと(常時滞留の要)
    assert "_confirmed or self._stuck_gear_wait_count" not in snippet


def test_wait_park_uses_dedicated_dwell_param_not_shared_settle_cycles():
    """2026-08-04再修正: WAIT_PARKはWAIT_REVERSE等と共有のgear_settle_cyclesではなく、
    専用のgear_park_dwell_cyclesを使う(confirmedが即時Trueになりうるため、
    共有パラメータのままだと他状態のsettle_cycles調整に巻き込まれてしまう)。"""
    idx_park = _SRC.index('if self._stuck_state == "WAIT_PARK":')
    idx_reverse = _SRC.index('elif self._stuck_state == "WAIT_REVERSE":')
    snippet = _SRC[idx_park:idx_reverse]
    assert "self._stuck_gear_wait_count += 1" in snippet
    assert "self._stuck_gear_wait_count >= self._stuck_gear_park_dwell_cycles" in snippet
    assert "self._stuck_gear_settle_cycles" not in snippet


def test_gear_park_dwell_cycles_param_wired_with_default():
    assert '"gear_park_dwell_cycles", int(_stkget("gear_park_dwell_cycles", 40))' in _SRC
