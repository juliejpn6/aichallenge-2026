"""Unit tests for PUSH-directed steering (148節②、2026-07-21)。

背景: 0721-01実測(wp332、約193秒間完全スタック)で、STUCK復帰のPUSH状態が
操舵0固定の完全直進(u=[push_speed, 0.0])だったため、_ot_state側が既に持つ
状況認識(_plan_passの側選択・動的コリドー)を一切参照せず、車体headingの
偶発的なズレ頼みでしかfwd_dlatを稼げず、1.35m付近で頭打ちしていたことが
判明した。

ユーザー提案(「上流から下流まで同じ計算式・同じ値を使うべき」)を受け、
PUSH開始時に既存の_plan_pass(ENGAGE判定と全く同じ関数)を1回だけ呼び、
決定した側へ小角の操舵を加えるよう変更した(_compute_stuck_push_steer)。
新規の側選択式は作らず、既存関数の呼び出しのみで完結させている。

動的コリドー先読み(_corr_bound_ahead)は「これ以上は超えない安全上限」
としてのみ使う。BACKUP後は自車位置がMPC最終ソルブ時から数m動いており、
このデータ自体が陳腐化している可能性があるため、上限としてのみ使う限り
陳腐化はより保守的な方向にしか作用しない、という設計上の判断をテストで
裏付ける。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー
実装した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _mirror_push_steer(plan_ok, plan_side, room, steer_max_deg=6.0, room_ref=1.0):
    """_compute_stuck_push_steer()のミラー実装。"""
    if not plan_ok or plan_side == 0:
        return 0.0
    room = max(0.0, room)
    scale = min(1.0, room / room_ref)
    mag = np.deg2rad(steer_max_deg) * scale
    return float(mag if plan_side > 0 else -mag)


# --- ①非矛盾性: 側不明・空きなし時は従来通り直進(挙動の後方互換) ---

def test_falls_back_to_straight_when_plan_not_ok():
    assert _mirror_push_steer(plan_ok=False, plan_side=1, room=5.0) == 0.0


def test_falls_back_to_straight_when_side_is_zero():
    assert _mirror_push_steer(plan_ok=True, plan_side=0, room=5.0) == 0.0


def test_falls_back_to_straight_when_room_is_zero():
    assert _mirror_push_steer(plan_ok=True, plan_side=1, room=0.0) == 0.0


# --- 符号規約: plan_side>0(左)は正の操舵角、plan_side<0(右)は負 ---

def test_left_side_yields_positive_steer():
    steer = _mirror_push_steer(plan_ok=True, plan_side=1, room=5.0)
    assert steer > 0.0


def test_right_side_yields_negative_steer():
    steer = _mirror_push_steer(plan_ok=True, plan_side=-1, room=5.0)
    assert steer < 0.0


# --- 上限キャップ: room_refを超える空きでも、steer_max_deg以上には振れない ---

def test_steer_capped_at_max_even_with_abundant_room():
    steer = _mirror_push_steer(plan_ok=True, plan_side=1, room=100.0)
    assert steer == pytest.approx(np.deg2rad(6.0))


def test_steer_scales_down_proportionally_with_small_room():
    """室が上限基準(room_ref)の半分なら、操舵角も概ね半分になる(線形スケール)。"""
    steer_full = _mirror_push_steer(plan_ok=True, plan_side=1, room=1.0)
    steer_half = _mirror_push_steer(plan_ok=True, plan_side=1, room=0.5)
    assert steer_half == pytest.approx(steer_full / 2.0)


# --- ②非冗長性: 側選択は既存_plan_passの再利用のみ、新規式を持たない ---

def test_compute_stuck_push_steer_calls_existing_plan_pass():
    idx = _SRC.index("def _compute_stuck_push_steer(")
    idx_end = _SRC.index("def _handle_stuck_recovery(")
    snippet = _SRC[idx:idx_end]
    assert "self._plan_pass(_scan, self._ot_side)" in snippet
    assert "self._corr_bound_ahead(_plan_side)" in snippet


def test_compute_stuck_push_steer_reuses_scan_traffic_not_new_scan_logic():
    idx = _SRC.index("def _compute_stuck_push_steer(")
    idx_end = _SRC.index("def _handle_stuck_recovery(")
    snippet = _SRC[idx:idx_end]
    assert "self._scan_traffic(" in snippet
    # 独自の側判定式(lf>=rf等)を持ち込んでいないことの裏付け。
    assert "lf" not in snippet.lower() or "lf_min" not in snippet


# --- ③配線確認: PUSH突入時に1回だけ決定し、PUSH中は同じ値を使い続ける ---

def test_push_entry_computes_steer_once():
    idx = _SRC.index('self._stuck_state = "PUSH"')
    # 2026-07-26追加(186節続報): giveup_streak反転リトライ用のside_flip適用ブロックが
    #   steer計算とu=[...]の間に挿入されたため、窓を広げた(意味的な変更はない)。
    # 2026-07-31追加(252節): 衝突疑い検知(v_prev/v_window)のリセット行が
    #   steer計算より前に挿入されたため、さらに窓を広げた(意味的な変更はない)。
    idx_end = idx + 1400
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_push_steer = self._compute_stuck_push_steer(pose)" in snippet
    assert "u = [self._stuck_push_speed, self._stuck_push_steer]" in snippet


def test_push_continuation_reuses_same_stored_steer_value():
    """PUSH状態が継続する周期(dist/timeout未到達)でも、_compute_stuck_push_steerを
    毎回呼び直さず、突入時に決めた値をそのまま使うことを確認する(4Hz以上で
    _plan_pass等を再計算し続けるコストを避ける、既存の軽量化方針と整合)。"""
    idx_class_end = _SRC.index("def _gnss_pose_callback(")
    idx_handle = _SRC.index("def _handle_stuck_recovery(")
    body = _SRC[idx_handle:idx_class_end]
    assert body.count("self._compute_stuck_push_steer(") == 1


def test_backup_state_now_uses_directed_steer_not_fixed_zero():
    """2026-07-26更新(184節): 従来このテストは「BACKUP状態の操舵は0固定のまま」
    (本節=148節②の変更対象外)であることを確認していた。184節でユーザーから
    「壁リカバリー(90°自動修正)をoff化したため、後退時も隙間の中心へ先頭を
    向ける能動操舵が必要」と要望があり、BACKUP自体の操舵も0固定から
    _fresh_gap_target+_stuck_target_steer(reverse=True)による比例制御へ
    変更した。よって本テストは「0固定のまま」ではなく「_backup_steer(隙間
    追従の計算値)を使うようになった」ことを確認する形へ更新する
    (_compute_stuck_push_steerは引き続きBACKUP側からは呼ばない=PUSH開始時
    専用のままという非重複性は維持)。"""
    idx = _SRC.index('elif self._stuck_state == "BACKUP":')
    idx_end = _SRC.index('elif self._stuck_state == "WAIT_DRIVE_PUSH":')
    snippet = _SRC[idx:idx_end]
    assert "u = [self._stuck_backup_speed, 0.0]" not in snippet
    assert "u = [self._stuck_backup_speed, _backup_steer]" in snippet
    assert "self._stuck_target_steer(" in snippet
    assert "reverse=True" in snippet
    assert "self._compute_stuck_push_steer" not in snippet


# --- ④過去ログへの遡及効果 ---

def test_retroactive_0721_01_side_and_room_would_have_produced_directed_steer():
    """0721-01実測(wp332-336): _plan_passは一貫してside=1(左)、lf_min=1.82m>
    rf_min=1.69mを正しく選んでいた。この値をミラーへ投入すると、従来の
    steer=0.0(完全直進)ではなく、左方向への有向な操舵角が得られることを
    確認する(対処前は一度もこの情報を使っていなかった)。"""
    steer = _mirror_push_steer(plan_ok=True, plan_side=1, room=1.82)
    assert steer > 0.0
    assert steer == pytest.approx(np.deg2rad(6.0))  # room(1.82)>room_ref(1.0)なので上限に到達


def test_retroactive_0721_01_plateau_scenario_with_tight_room():
    """0721-01実測終盤(ub0=0.68m程度まで狭まっていた時間帯)相当のroomを
    投入すると、操舵角が上限より小さく抑えられ、無理に大きく切り込まない
    ことを確認する(コリドーが実際に狭い場面での安全側動作)。"""
    steer = _mirror_push_steer(plan_ok=True, plan_side=1, room=0.68)
    assert 0.0 < steer < np.deg2rad(6.0)
