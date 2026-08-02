"""STUCK検知(経路1/2/3)→WAIT_REVERSE突入処理の共通化(230節続報2、2026-07-29)。

背景: mpc_controller.pyの重複調査(230節続報)で、STUCK経路1/2用と経路3用の
2箇所に、WAIT_REVERSE突入時の全く同じ7行(_stuck_update_shuffle_cycle呼び出し+
_stuck_state/_stuck_count/_stuck_stall_count/_stuck_gear_wait_count/
_ghost_block_loggedのリセット+_handle_stuck_recovery呼び出し)が手作業で
複製されていることが判明した。lateral_target/_ot_alphaの離脱時未リセット
(230節続報)と同種の「複数箇所に手書きされた同一処理」パターンであり、
片方だけ将来修正されて挙動が乖離するリスクがあった。

対処: 新規ヘルパー_stuck_enter_wait_reverse(now, pose)を追加し、経路1/2・経路3
両方の呼び出し元から呼ぶ形に統合した。ログメッセージ(path=1/2 vs path=3で文言が
異なる)と直後のreturnは呼び出し元に残す(制御フローの分岐点を保つため)。

mpc_controller.pyはrclpy非依存のため直接importできず、他の巨大メソッド関連
テスト群と同じ方針(ソーステキストによる構造的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _helper_body():
    idx = _SRC.index("def _stuck_enter_wait_reverse(self, now, pose)")
    idx_end = _SRC.index("\n    def ", idx + 10)
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①非矛盾性: ヘルパー自体が旧来の7行全てを含むこと
# ---------------------------------------------------------------------------

def test_helper_contains_full_reset_set():
    snippet = _helper_body()
    assert "self._stuck_update_shuffle_cycle(now, pose)" in snippet
    assert 'self._stuck_state = "WAIT_REVERSE"' in snippet
    assert "self._stuck_count = 0" in snippet
    assert "self._stuck_stall_count = 0" in snippet
    assert "self._stuck_gear_wait_count = 0" in snippet
    assert "self._ghost_block_logged = False" in snippet
    assert "self._handle_stuck_recovery(now, pose)" in snippet


def test_helper_does_not_include_return_or_logging():
    """②非冗長性: ヘルパーはreturn文・ログ出力(経路ごとに文言が異なるため
    呼び出し元固有の責務)を含まず、状態リセットと復帰処理呼び出しのみに
    責務を限定していることを確認する。"""
    snippet = _helper_body()
    assert "\n        return" not in snippet  # 実際のreturn文(docstring内の言及は除外)
    assert "self.get_logger()" not in snippet


# ---------------------------------------------------------------------------
# ④遡及効果: 経路1/2・経路3の両方の呼び出し元がヘルパー経由になっていること
# ---------------------------------------------------------------------------

def test_path12_calls_helper():
    idx = _SRC.index("-> WAIT_REVERSE\")")
    snippet = _SRC[idx:idx + 200]
    assert "self._stuck_enter_wait_reverse(now, pose)" in snippet
    assert "return" in snippet
    # 旧来のインライン展開が残っていないことも確認
    assert "self._stuck_update_shuffle_cycle(now, pose)  # 184節追加" not in snippet


def test_path3_calls_helper():
    idx = _SRC.index("path=3) -> WAIT_REVERSE(→PUSH予定)")
    snippet = _SRC[idx:idx + 200]
    assert "self._stuck_enter_wait_reverse(now, pose)" in snippet
    assert "return" in snippet


def test_total_call_count_matches_two_known_sites():
    """新しいSTUCK検知経路が追加/削除された場合はこのテスト自体の更新も必要。"""
    n_calls = _SRC.count("self._stuck_enter_wait_reverse(now, pose)")
    assert n_calls == 2, (
        f"想定していた2箇所(経路1/2・経路3)から数が変わっている(現在{n_calls}箇所)。"
        "新しいSTUCK検知経路が追加/削除された場合はこのテスト自体の更新も必要。")


def test_no_hand_duplicated_reset_blocks_remain():
    """回帰防止: 旧来の手作業複製ブロックの一意な目印だった
    `_stuck_update_shuffle_cycle(now, pose)  # 184節追加`(shuffle呼び出し+コメント
    の組み合わせ)が、ファイル全体でヘルパー定義内の1箇所にしか存在しないことを
    確認する(_stuck_gear_wait_count=0/_ghost_block_logged=False単体は
    _handle_stuck_recovery内のギア待ちループ等、無関係な箇所にも正当に存在するため
    個別には検査しない)。"""
    n = _SRC.count("self._stuck_update_shuffle_cycle(now, pose)  # 184節追加")
    assert n == 1, f"想定は1箇所(ヘルパー内)のみだが{n}箇所で見つかった"
