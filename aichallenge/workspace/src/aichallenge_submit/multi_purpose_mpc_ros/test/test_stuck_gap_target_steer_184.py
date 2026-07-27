"""Unit tests for 184節(2026-07-26): 隙間狙い操舵+動的後退距離+シャッフル脱出。

背景: 壁リカバリー(AWSIM組み込みの90°自動修正)をoff化した(183節)ことで、
「後退→単純な直進発進」では同じ壁へ再突入しやすくなった。ユーザー提案
(「後退時、進行方向で最も隙間の大きい場所へ先頭を向ける。壁だけでなく複数の
相手車も考慮する」)を受け、以下を実装した。

  1. `_fresh_gap_target()`: STUCK中でも新鮮な自己位置から壁+相手車統合済みの
     隙間中心を計算する(reference_path.update_path_constraintsの単発呼び出し、
     新規の空き区間探索アルゴリズムは増やさない)。
  2. `_stuck_target_steer()`: 隙間中心・経路接線への単純な比例操舵則。後退中は
     kinematic bicycle modelのpsi_dot=v/L*tan(delta)がv<0でもそのまま成り立つ
     ことを実コードで確認した上で、前進基準の操舵角の符号を反転して適用する。
  3. `_rear_clearance_m()`: 後退開始前に後方の相手車から安全な後退距離の上限を
     求める(test_stuck_backup_blocked.pyで詳細をカバー)。
  4. シャッフル(縦列駐車脱出): 短い後退→隙間への微調整前進を繰り返す
     (test_stuck_backup_blocked.pyでカバー)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装
した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import math
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

EY_KP = 0.8
PSI_KP = 1.0
DELTA_MAX = math.radians(52.0)


def mirror_gap_center(ub, lb):
    """`_fresh_gap_target`の隙間中心計算部分のミラー(ub<=lbは全区間ふさがり)。"""
    if ub is None or lb is None or ub <= lb:
        return None
    return (ub + lb) / 2.0, (ub - lb)


def mirror_target_steer(target_ey, wp_psi, cur_ey, cur_psi,
                         ey_kp=EY_KP, psi_kp=PSI_KP, delta_max=DELTA_MAX, reverse=False):
    """`_stuck_target_steer`のミラー。"""
    e_y_err = target_ey - cur_ey
    psi_err = math.atan2(math.sin(wp_psi - cur_psi), math.cos(wp_psi - cur_psi))
    delta = ey_kp * e_y_err + psi_kp * psi_err
    delta = max(-delta_max, min(delta_max, delta))
    return -delta if reverse else delta


# --- ①非矛盾性: 隙間中心の計算 ---

def test_gap_center_of_symmetric_corridor_is_zero():
    center, width = mirror_gap_center(ub=1.0, lb=-1.0)
    assert center == pytest.approx(0.0)
    assert width == pytest.approx(2.0)


def test_gap_center_of_asymmetric_corridor_is_skewed():
    """壁+相手車で片側が狭まっている場合、中心は空いている側へ寄る。"""
    center, width = mirror_gap_center(ub=0.3, lb=-1.7)
    assert center == pytest.approx(-0.7)
    assert width == pytest.approx(2.0)


def test_gap_center_returns_none_when_fully_blocked():
    assert mirror_gap_center(ub=-0.1, lb=0.1) is None


def test_gap_center_returns_none_when_zero_width():
    assert mirror_gap_center(ub=0.5, lb=0.5) is None


# --- 符号規約: target_ey>cur_ey(隙間が左)は正の前進操舵、reverseで符号反転 ---

def test_target_left_of_current_yields_positive_forward_steer():
    steer = mirror_target_steer(target_ey=1.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer > 0.0


def test_target_right_of_current_yields_negative_forward_steer():
    steer = mirror_target_steer(target_ey=-1.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer < 0.0


def test_reverse_flips_sign_for_identical_target():
    """後退中(reverse=True)は、同じ隙間目標でも前進基準の操舵角の符号が
    反転する(kinematic bicycle modelのpsi_dot=v/L*tan(delta)がv<0でもそのまま
    成り立つことに基づく)。"""
    fwd = mirror_target_steer(target_ey=1.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                               reverse=False)
    rev = mirror_target_steer(target_ey=1.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                               reverse=True)
    assert rev == pytest.approx(-fwd)


def test_heading_error_alone_contributes_even_with_zero_lateral_error():
    """横方向誤差が0でも、経路接線とのヨー誤差だけで操舵が生じる。"""
    steer = mirror_target_steer(target_ey=0.0, wp_psi=0.3, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer > 0.0


def test_no_error_yields_zero_steer():
    """誤差が両方0(隙間の中心を向いて既に正対)なら操舵0。"""
    steer = mirror_target_steer(target_ey=0.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer == pytest.approx(0.0)


def test_steer_clamped_at_delta_max_even_with_huge_error():
    steer = mirror_target_steer(target_ey=100.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer == pytest.approx(DELTA_MAX)


def test_steer_clamped_at_negative_delta_max():
    steer = mirror_target_steer(target_ey=-100.0, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer == pytest.approx(-DELTA_MAX)


# --- ②非冗長性: 新規の空き区間探索式を持ち込まず、既存のupdate_path_constraintsを再利用 ---

def test_fresh_gap_target_calls_existing_update_path_constraints():
    idx = _SRC.index("def _fresh_gap_target(")
    idx_end = _SRC.index("def _stuck_target_steer(")
    snippet = _SRC[idx:idx_end]
    assert "self._reference_path.update_path_constraints(" in snippet
    assert "self._closest_wp_and_s(" in snippet


def test_fresh_gap_target_returns_none_on_fully_blocked_corridor():
    idx = _SRC.index("def _fresh_gap_target(")
    idx_end = _SRC.index("def _stuck_target_steer(")
    snippet = _SRC[idx:idx_end]
    assert "if u0 <= l0:" in snippet
    assert "return None" in snippet


# --- ③配線確認: BACKUP/PUSHの両方が新しいヘルパーを実際に使っている ---

def test_backup_branch_uses_fresh_gap_target_and_reverse_steer():
    idx = _SRC.index('elif self._stuck_state == "BACKUP":')
    idx_end = _SRC.index('elif self._stuck_state == "WAIT_DRIVE_PUSH":')
    snippet = _SRC[idx:idx_end]
    assert "self._fresh_gap_target(" in snippet
    assert "self._stuck_target_steer(" in snippet
    assert "reverse=True" in snippet


def test_compute_stuck_push_steer_tries_fresh_gap_target_first():
    idx = _SRC.index("def _compute_stuck_push_steer(")
    idx_end = _SRC.index("def _handle_stuck_recovery(")
    snippet = _SRC[idx:idx_end]
    assert "self._fresh_gap_target(" in snippet
    assert "reverse=False" in snippet


def test_push_end_condition_checks_heading_tolerance():
    idx = _SRC.index('elif self._stuck_state == "PUSH":')
    idx_end = _SRC.index("self._publish_control_command(")
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_push_heading_tol_rad" in snippet
    assert "self._fresh_gap_target(" in snippet
    # 陳腐化フォールバック(既存corr_bound_ahead)も引き続き残っていること
    assert "self._corr_bound_ahead(self._stuck_push_side)" in snippet


def test_backup_entry_computes_rear_clearance():
    idx = _SRC.index('self._stuck_state = "BACKUP"')
    idx_end = idx + 1200
    snippet = _SRC[idx:idx_end]
    assert "self._rear_clearance_m(" in snippet
    assert "self._stuck_backup_dist_eff" in snippet


# --- ④遡及効果: 172節実測(wp332、side不明の壁単独STUCK)相当のシナリオ ---

def test_retroactive_wp332_symmetric_wall_now_uses_gap_center_not_binary_fallback():
    """172節実測(壁に正対して左右がほぼ対称に停止したケース)相当: 従来は
    壁マージンの左右差が僅少(along_min_widthの1/10未満)だとside=0・steer=0.0の
    フォールバックに落ちていた。184節では隙間中心(非対称にわずかでも空きが
    あればそこを狙う)を優先するため、僅かな非対称性(0.05m)でも有向な操舵が
    得られることを確認する。"""
    # 左右がほぼ対称(ub=1.02, lb=-1.00 -> center=+0.01)でも、隙間中心狙いの式は
    # 従来の「差がalong_min_width*0.1(=0.145m)未満なら直進」という閾値を持たず、
    # 僅かでも中心が右か左かに応じて有向な操舵を返す。
    center, _width = mirror_gap_center(ub=1.02, lb=-1.00)
    steer = mirror_target_steer(target_ey=center, wp_psi=0.0, cur_ey=0.0, cur_psi=0.0,
                                 reverse=False)
    assert steer > 0.0
