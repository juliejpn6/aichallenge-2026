"""[OT-OUTCOME] success/exit_clearの無限再ログバグ修正(2026-08-06)。

背景: Fix A'(opp_lat_pred根本修正)のdev3ローカル検証中、[OT-OUTCOME]の
outcome=success件数を集計しようとしたところ、30分・3台走行で133338件という
明らかに異常な値が出た。原因は「前方クリアが連続したらNORMAL復帰」の分岐
(_control()内、_n_fwd==0の側)が状態非依存で、self._ot_state=="NORMAL"へ
既に遷移済み(=前方に相手がいない大半の巡航区間)でも
self._fwd_clear_count >= self._ot_exit_clearが真であり続ける限り毎周期
再実行され、[OT-OUTCOME]を無意味に再ログし+_reset_ot_offset_state()等の
リセット処理も無駄に再実行し続けていたこと。

対処: 条件へself._ot_state != "NORMAL"を追加し、実際の
OVERTAKING/STOPPING→NORMAL遷移が起きるその1周期だけで発火するよう限定した。
254節の後方車バグ修正で確立したSTOPPING→NORMAL復帰経路(_n_fwd==0が
毎周期真になり続けるケースを含む)は、遷移の意味・タイミングとも無変更で
維持している(遷移後にself._ot_state=="NORMAL"となるため、以降の周期は
本ガードで自然にスキップされるだけ)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと
同じ「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _exit_clear_block():
    idx = _SRC.index("self._fwd_clear_count += 1")
    idx_end = _SRC.index("self._reset_ot_offset_state()", idx) + 40
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①状態ガード: NORMALへ既に遷移済みの場合は再発火しないこと
# ---------------------------------------------------------------------------

def test_exit_clear_guarded_by_state_not_normal():
    snippet = _exit_clear_block()
    assert 'self._ot_state != "NORMAL"' in snippet
    assert "self._ot_exit_clear" in snippet


def test_guard_and_threshold_combined_with_and():
    """状態ガードと従来のカウンタ閾値判定がANDで結合されている
    (どちらか一方でも欠けると意図通り動かない)ことを確認する。"""
    snippet = _exit_clear_block()
    idx_if = snippet.index("if (")
    idx_colon = snippet.index("):", idx_if)
    cond = snippet[idx_if:idx_colon]
    assert 'self._ot_state != "NORMAL"' in cond
    assert "self._fwd_clear_count >= self._ot_exit_clear" in cond
    assert " and " in cond


# ---------------------------------------------------------------------------
# ②回帰防止: カウンタ増分・遷移後の代入(state="NORMAL"等)自体は無変更
# ---------------------------------------------------------------------------

def test_counter_increment_unchanged():
    snippet = _exit_clear_block()
    assert snippet.startswith("self._fwd_clear_count += 1")


def test_transition_body_unchanged():
    snippet = _exit_clear_block()
    assert 'self._log_ot_outcome("success", self._ot_side, reason="exit_clear")' in snippet
    assert 'self._ot_state = "NORMAL"' in snippet
    assert "self._ot_side = 0" in snippet
    assert "self._ot_side_locked = 0" in snippet
    assert "self._ot_worth_count = 0" in snippet
    assert "self._ot_giveup_count = 0" in snippet
    assert "self._ot_cleared = False" in snippet
    assert "self._reset_ot_offset_state()" in snippet


# ---------------------------------------------------------------------------
# ③254節の後方車バグ修正で確立したSTOPPING→NORMAL復帰経路が維持されていること
#   (ガード条件はself._ot_state!="NORMAL"であり"OVERTAKING"限定ではないため、
#   STOPPING状態からの復帰も引き続き通ることを確認する)
# ---------------------------------------------------------------------------

def test_guard_covers_stopping_state_too_not_only_overtaking():
    """ガードが"OVERTAKING"限定ではなく"NORMAL"以外全てを許可する形に
    なっていることを確認する(STOPPING→NORMAL復帰の回帰防止、254節参照)。"""
    snippet = _exit_clear_block()
    idx_if = snippet.index("if (")
    idx_colon = snippet.index("):", idx_if)
    cond = snippet[idx_if:idx_colon]
    assert '"OVERTAKING"' not in cond
    assert '!= "NORMAL"' in cond


# ---------------------------------------------------------------------------
# ④ミラー検証: ガード追加により「NORMAL巡航中の無限再発火」が実際に止まる
#   ことを、状態遷移ロジックのミニマル模擬実装で数値的に確認する。
# ---------------------------------------------------------------------------

def _simulate_cycles(n_cycles, exit_clear=3, start_state="OVERTAKING"):
    """[OT-OUTCOME] success ログが何回発火するかを模擬する
    (このテストファイル専用のミラー実装、本体ロジックの1:1移植ではない)。"""
    state = start_state
    fwd_clear_count = 0
    log_count = 0
    for _ in range(n_cycles):
        # 全周期_n_fwd==0(前方に相手なし)と仮定
        fwd_clear_count += 1
        if state != "NORMAL" and fwd_clear_count >= exit_clear:
            log_count += 1
            state = "NORMAL"
    return log_count


def test_mirror_fires_exactly_once_over_many_cycles():
    """修正後のロジックは、前方クリアが何周期続いてもsuccessログは1回だけ
    発火することを確認する(修正前は133338件のように周期数分発火していた)。"""
    assert _simulate_cycles(1000, exit_clear=3, start_state="OVERTAKING") == 1
    assert _simulate_cycles(1000, exit_clear=3, start_state="STOPPING") == 1
