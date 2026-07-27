"""Unit tests for 186節続報(2026-07-26): シャッフル上限到達後の無限復帰断念ループ対処。

背景: 0726-01local試験(3台dev3自己対戦)で、車速20km/h化に伴う蛇行が悪化した状態で
コーナーの狭いコリドーを追い越そうとしたd3が壁に接触したとみられる急減速
(COLLISION-SUSPECTED)の直後、STUCK検知→184節のBACKUP-PUSHシャッフルが起動したが
即座にシャッフル上限(6回)へ到達し「無理に後退せず停止し復帰断念、NORMAL(通常の
MPC/ICC)へ委譲」となった。しかしNORMAL側でも車は一切動けず(v=0.00のまま)、約3秒
周期でSTUCKが再検知される度に同一地点判定(shuffle_episode_gap_s=6.0s/radius_m=3.0m)
が成立し続け、シャッフル上限到達→即断念→再STUCK、を95秒以上・cycle=24超まで一度も
回復せず繰り返す無限ループに陥っていた(ログでは末尾まで回復を確認できず)。

副次的に、PUSHの「reason=cleared」判定(コリドー幅+向き一致のみ)がPUSH開始直後
(dist=0.00m、車が全く動いていない)でも成立してしまい、「回避成功」扱いで即NORMAL
復帰→再STUCKを繰り返すことでシャッフル回数を浪費していたことも判明した。

対処:
  1. シャッフル上限(shuffle_max_cycles)に到達しても、別カウンタ
     (_stuck_giveup_streak、max_giveup_streak=3が上限)の範囲内であれば、PUSHの
     操舵方向候補を反転(_stuck_push_side_flip)した上でシャッフルカウンタを
     仕切り直し、もう一巡だけ試す。挟まれ方が非対称な場合、逆方向の方が抜けられる
     可能性があるための追加候補であり、既存の物理妥当性判定(backup_blocked_v_thr/
     confirm_s)・shuffle_max_cycles自体は無変更(新規の安全弁緩和なし)。
  2. PUSHの「reason=cleared」判定に、最小実移動量(push_min_dist_for_cleared=0.15m)
     を追加要求し、動いていない「見かけ上の回避成功」を弾く。
  3. giveup_streak/push_side_flipは、同一地点判定が「新規エピソード」(十分離れた/
     時間が経った)になった瞬間にのみリセットする(既存のshuffle_cycleリセット箇所
     を再利用、新規の位置・時刻トラッキングは追加しない)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装した上で
ソーステキスト検証と組み合わせる(既存テストと同じ方針、184節: test_stuck_gap_target_
steer_184.py参照)。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

MAX_GIVEUP_STREAK = 3
MIN_DIST_FOR_CLEARED = 0.15


def mirror_giveup_decision(giveup_streak_before, max_giveup_streak=MAX_GIVEUP_STREAK):
    """シャッフル上限到達時点(BACKUP-BLOCKED)でのミラー。
    戻り値: (new_streak, should_retry_with_flip)"""
    new_streak = giveup_streak_before + 1
    should_retry = new_streak <= max_giveup_streak
    return new_streak, should_retry


def mirror_cleared(dist, width_ok, heading_ok, min_dist=MIN_DIST_FOR_CLEARED):
    return (dist >= min_dist) and width_ok and heading_ok


# ---------------------------------------------------------------------------
# ①非矛盾性: giveup_streakのエスカレーション判定(境界値を含む)
# ---------------------------------------------------------------------------

def test_first_shuffle_exhaustion_retries_with_flip():
    new_streak, should_retry = mirror_giveup_decision(giveup_streak_before=0)
    assert new_streak == 1
    assert should_retry is True


def test_streak_exactly_at_max_still_retries():
    """max_giveup_streak回目の到達はまだリトライ対象(<=、境界は許可側)。"""
    new_streak, should_retry = mirror_giveup_decision(giveup_streak_before=MAX_GIVEUP_STREAK - 1)
    assert new_streak == MAX_GIVEUP_STREAK
    assert should_retry is True


def test_streak_exceeding_max_gives_up_permanently():
    new_streak, should_retry = mirror_giveup_decision(giveup_streak_before=MAX_GIVEUP_STREAK)
    assert new_streak == MAX_GIVEUP_STREAK + 1
    assert should_retry is False


def test_streak_zero_max_never_retries():
    """max_giveup_streak=0の設定(将来的な運用変更を想定)では、
    従来(186節以前)通り即座に完全断念する。"""
    _, should_retry = mirror_giveup_decision(giveup_streak_before=0, max_giveup_streak=0)
    assert should_retry is False


# ---------------------------------------------------------------------------
# ①非矛盾性: PUSH「cleared」判定の最小移動量要求
# ---------------------------------------------------------------------------

def test_cleared_false_when_geometry_ok_but_zero_displacement():
    """186節続報の核心回帰: 0726-01localで実測した「dist=0.00mのままreason=cleared」
    パターンが再現しないことを確認する。"""
    assert mirror_cleared(dist=0.0, width_ok=True, heading_ok=True) is False


def test_cleared_false_below_threshold_displacement():
    assert mirror_cleared(dist=0.10, width_ok=True, heading_ok=True) is False


def test_cleared_true_at_exactly_threshold_displacement():
    assert mirror_cleared(dist=MIN_DIST_FOR_CLEARED, width_ok=True, heading_ok=True) is True


def test_cleared_still_requires_geometry_even_with_enough_displacement():
    """最小移動量を満たしても、既存のコリドー幅/向き一致要求は無変更のまま残る
    (安全弁の緩和ではなく追加条件であることの確認)。"""
    assert mirror_cleared(dist=1.0, width_ok=False, heading_ok=True) is False
    assert mirror_cleared(dist=1.0, width_ok=True, heading_ok=False) is False


def test_cleared_true_when_all_conditions_met():
    assert mirror_cleared(dist=1.0, width_ok=True, heading_ok=True) is True


# ---------------------------------------------------------------------------
# ②非冗長性・③配線確認: ソーステキスト検証
# ---------------------------------------------------------------------------

def test_giveup_branch_increments_streak_before_deciding():
    idx = _SRC.index("self._stuck_giveup_streak += 1")
    idx_end = idx + 2200
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_max_giveup_streak" in snippet
    assert "self._stuck_push_side_flip = not self._stuck_push_side_flip" in snippet
    assert "self._stuck_shuffle_cycle = 0" in snippet
    assert "self._stuck_recovery_complete(" in snippet


def test_final_giveup_only_reached_in_else_branch_of_streak_check():
    """完全断念(_stuck_recovery_complete呼び出し)が、giveup_streakが上限を
    超えた場合のelse節にのみ存在し、既存のshuffle_cycle判定式自体
    (`if self._stuck_shuffle_cycle < self._stuck_shuffle_max_cycles:`)は
    186節時点のまま無変更であることを確認する(④遡及効果: 通常のシャッフル
    挙動そのものは変えていない)。"""
    idx = _SRC.index('if self._stuck_shuffle_cycle < self._stuck_shuffle_max_cycles:')
    idx_end = idx + 3200
    snippet = _SRC[idx:idx_end]
    assert 'if self._stuck_giveup_streak <= self._stuck_max_giveup_streak:' in snippet
    # 完全断念のrecovery_completeは反転リトライのelse節(=このスニペット後半)にのみ
    # 現れる(反転リトライ枝自体はrecovery_completeを呼ばない)。
    _retry_idx = snippet.index('if self._stuck_giveup_streak <= self._stuck_max_giveup_streak:')
    _else_idx = snippet.index("else:", _retry_idx)
    assert "self._stuck_recovery_complete(" not in snippet[_retry_idx:_else_idx]
    assert "self._stuck_recovery_complete(" in snippet[_else_idx:]


def test_new_episode_resets_giveup_streak_and_push_side_flip():
    """_stuck_update_shuffle_cycle内で、shuffle_cycleが0へ戻る(=新規エピソード)
    のと同じ箇所でgiveup_streak/push_side_flipもリセットされることを確認する
    (新規の位置・時刻トラッキングを増やさず、既存の同一地点判定を再利用)。"""
    idx = _SRC.index("def _stuck_update_shuffle_cycle(")
    idx_end = _SRC.index("def _stuck_recovery_complete(")
    snippet = _SRC[idx:idx_end]
    reset_idx = snippet.index("self._stuck_shuffle_cycle = 0")
    tail = snippet[reset_idx:reset_idx + 300]
    assert "self._stuck_giveup_streak = 0" in tail
    assert "self._stuck_push_side_flip = False" in tail


def test_push_steer_entry_applies_side_flip_after_computing_original_steer():
    idx = _SRC.index("self._stuck_push_steer = self._compute_stuck_push_steer(pose)")
    idx_end = idx + 700
    snippet = _SRC[idx:idx_end]
    assert "if self._stuck_push_side_flip:" in snippet
    assert "self._stuck_push_steer = -self._stuck_push_steer" in snippet
    assert "self._stuck_push_side = -self._stuck_push_side" in snippet


def test_push_cleared_check_requires_minimum_distance_in_both_branches():
    """_fresh_gap_target成功時・失敗時フォールバックの両方に最小移動量要求が
    掛かっていることを確認する(片方だけ対処して抜け穴を残す回帰を防ぐ)。"""
    idx = _SRC.index("_dist_ok = dist >= self._stuck_push_min_dist_for_cleared")
    idx_end = idx + 900
    snippet = _SRC[idx:idx_end]
    assert snippet.count("_dist_ok") >= 3  # 定義1回 + 分岐2箇所での参照
    assert "_cleared = (_dist_ok and _gap_width_chk > self._along_min_width" in snippet
    assert "_cleared = (_dist_ok and self._stuck_push_side != 0" in snippet


def test_new_config_params_defined_with_stkget():
    idx = _SRC.index('self._stuck_max_giveup_streak = int(_stkget("max_giveup_streak"')
    idx2 = _SRC.index('self._stuck_push_min_dist_for_cleared = float(')
    assert idx > 0 and idx2 > idx


def test_config_yaml_documents_new_params():
    _cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(_cfg_path) as f:
        cfg_src = f.read()
    assert "max_giveup_streak:" in cfg_src
    assert "push_min_dist_for_cleared:" in cfg_src


# ---------------------------------------------------------------------------
# ④遡及効果: 184節のシャッフル本体(cycle<maxの通常リトライ枝)は無変更
# ---------------------------------------------------------------------------

def test_normal_shuffle_retry_branch_untouched():
    """shuffle_cycle<maxの通常リトライ(184節本体)は186節続報で一切変更されて
    いないことを、分岐内容の完全一致で確認する。"""
    idx = _SRC.index('if self._stuck_shuffle_cycle < self._stuck_shuffle_max_cycles:')
    idx_end = _SRC.index("else:", idx)
    snippet = _SRC[idx:idx_end]
    assert '_next = "WAIT_DRIVE_PUSH"' in snippet
    assert "[STUCK-BACKUP-BLOCKED-SHUFFLE]" in snippet
    assert "self._stuck_giveup_streak" not in snippet  # このリトライ枝自体は触れない
