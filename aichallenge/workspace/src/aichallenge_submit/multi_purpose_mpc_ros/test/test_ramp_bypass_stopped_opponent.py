"""Unit tests for the ramp bypass on stopped/slow opponents (155節、2026-07-22)。

背景: 154節で第3コーナー(0721-03、wp172-176)の繰り返しfootprint_risk停止を
実座標データで検証したところ、footprint_risk発火直前の1.76秒間、オフセット
目標は約1m動いていたが、車両の実位置はほぼ前進成分のみで横方向にはほぼ
動いていなかった(112節で先例確立済み、target=-3.0mで実位置収束に約9秒)。

ユーザーから「なるべく減速しないで完遂したい」との要望を受け、_ot_ramp_time
(2.5秒、目標を漸増させ横ジャークを防ぐ設計)が、MPC自身のQP制約
(max_steering_rate、core/MPC.py _rate_bounds、既存のκレート上限)と機能的に
重複しており、目標をゆっくり出すこと自体が実オフセットの成長(ひいては
icc_stopのds_eff=ds+dlatの伸び、不要な減速の回避)を遅らせている可能性がある
ことを確認した。停止/低速の相手(vopp<opp_obstacle_speed、ENGAGE判定と同一の
既存閾値)に対してのみランプを省略し、目標へ即座に到達させる(alpha=1.0を
即時設定)。走行中の相手への通常の高速すれ違い、およびオフセット復帰
(_a_target=0)側は従来通りランプを維持する。

mpc_controller.pyはrclpy依存のため直接importできないため、
test_wall_slow_universal.pyと同じ方針(純Pythonミラー関数+ソーステキストに
よる構造的検証)を用いる。
"""
import os

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

OPP_OBSTACLE_SPEED = 6.0  # [km/h]
RAMP_TIME = 2.5  # [s]


def alpha_update(alpha_prev, a_target, fwd_vopp, dt,
                  opp_obstacle_speed=OPP_OBSTACLE_SPEED, ramp_time=RAMP_TIME):
    """mpc_controller.pyのalphaランプ更新(155節修正後)の複製ミラー。

    戻り値: (alpha_new, bypassed, logged) logged=このステップで[RAMP-BYPASS]相当の
    ログ条件(alpha_prev<1.0からのバイパス)を満たしたか。
    """
    stopped_opp = fwd_vopp is not None and fwd_vopp < opp_obstacle_speed
    if a_target > 0.0 and stopped_opp:
        logged = alpha_prev < 1.0
        return 1.0, True, logged
    ramp_step = dt / max(ramp_time, 1e-3)
    alpha_new = alpha_prev + float(np.clip(a_target - alpha_prev, -ramp_step, ramp_step))
    alpha_new = float(np.clip(alpha_new, 0.0, 1.0))
    return alpha_new, False, False


# --- 中核: 停止/低速相手ではバイパス、走行中の相手では従来通り ---

def test_stopped_opponent_bypasses_ramp_immediately():
    """停止相手(vopp=0.9km/h<6.0)へ寄せる最中は、1周期でalpha=1.0に到達する
    (通常なら2.5秒かけるところを即座に)。"""
    alpha_new, bypassed, _ = alpha_update(alpha_prev=0.0, a_target=1.0, fwd_vopp=0.9, dt=0.025)
    assert bypassed is True
    assert alpha_new == pytest.approx(1.0)


def test_moving_opponent_still_uses_normal_ramp_no_regression():
    """回帰: 走行中の相手(vopp=8.9km/h>=6.0)へ寄せる場合は、従来通りramp_time
    ベースの漸増のまま(1周期(dt=0.025s)でalpha=1.0には到達しない)。"""
    alpha_new, bypassed, _ = alpha_update(alpha_prev=0.0, a_target=1.0, fwd_vopp=8.9, dt=0.025)
    assert bypassed is False
    assert alpha_new < 1.0
    assert alpha_new == pytest.approx(0.025 / RAMP_TIME, abs=1e-6)


def test_vopp_none_falls_back_to_normal_ramp():
    """fwd_vopp不明(相手情報無し)の場合は安全側にフォールバックし、
    従来通りランプを維持する(バイパスしない)。"""
    alpha_new, bypassed, _ = alpha_update(alpha_prev=0.0, a_target=1.0, fwd_vopp=None, dt=0.025)
    assert bypassed is False
    assert alpha_new < 1.0


def test_offset_return_phase_always_uses_ramp_even_for_stopped_opponent():
    """①非矛盾性: オフセット復帰側(_a_target=0、cleared後にレースラインへ戻る)は、
    相手が停止中であってもバイパスされず、従来通りゆっくり戻ることを確認する
    (急な戻りは想定外の挙動のため対象外のまま)。"""
    alpha_new, bypassed, _ = alpha_update(alpha_prev=1.0, a_target=0.0, fwd_vopp=0.9, dt=0.025)
    assert bypassed is False
    assert alpha_new < 1.0
    assert alpha_new == pytest.approx(1.0 - 0.025 / RAMP_TIME, abs=1e-6)


def test_boundary_exactly_at_obstacle_speed_does_not_bypass():
    """境界値: vopp==opp_obstacle_speedちょうどは「停止/低速」ではない
    (厳密<のみ)ため従来通りランプを維持する(ENGAGE判定の_is_stopped_for_profile
    と同一の境界規約)。"""
    alpha_new, bypassed, _ = alpha_update(
        alpha_prev=0.0, a_target=1.0, fwd_vopp=OPP_OBSTACLE_SPEED, dt=0.025)
    assert bypassed is False
    assert alpha_new < 1.0


# --- ログ(エッジトリガー)の確認 ---

def test_log_condition_fires_once_not_every_cycle():
    """③検証ロギング: ログ条件(alpha_prev<1.0)は最初のバイパス周期でのみ真になり、
    以降alphaが1.0に到達した後は多重ログにならないことを確認する。"""
    _, _, logged_1st = alpha_update(alpha_prev=0.0, a_target=1.0, fwd_vopp=0.9, dt=0.025)
    _, _, logged_2nd = alpha_update(alpha_prev=1.0, a_target=1.0, fwd_vopp=0.9, dt=0.025)
    assert logged_1st is True
    assert logged_2nd is False


# --- ④過去ログへの遡及効果: 0721-03実測(wp172-176) ---

def test_retroactive_0721_03_stopped_opponent_would_have_bypassed_the_slow_ramp():
    """遡及検証: 0721-03 wp172でのENGAGE時実測vopp(0.27km/h、footprint_riskの
    繰り返し発火episode)は明確にopp_obstacle_speed(6.0)未満であり、本節の対処が
    有効だった場合、alphaは即座に1.0へ到達していたはず(退行チェックとして、旧方式
    ではdt=0.025sで0.01しか進まなかったことも併せて確認する)。"""
    alpha_new, bypassed, _ = alpha_update(alpha_prev=0.0, a_target=1.0, fwd_vopp=0.2678, dt=0.025)
    assert bypassed is True
    assert alpha_new == pytest.approx(1.0)

    # 旧方式(155節導入前)相当: ランプのみだとこの周期ではまだ0.01程度にしか進まない
    old_ramp_step = 0.025 / RAMP_TIME
    assert old_ramp_step == pytest.approx(0.01, abs=1e-3)


# ---------------------------------------------------------------------------
# ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def _ramp_block_snippet():
    idx = _SRC.index('_stopped_opp = (_opp_sit.fwd_vopp is not None')
    idx_end = _SRC.index("self._mpc.lateral_blend = self._ot_alpha")
    return _SRC[idx:idx_end]


def test_source_bypass_reuses_existing_obstacle_speed_threshold():
    """②非冗長性: 新規閾値を持たず、既存のself._opp_obstacle_speedを再利用する
    ことを確認する。"""
    snippet = _ramp_block_snippet()
    assert "self._opp_obstacle_speed" in snippet
    assert "_a_target > 0.0 and _stopped_opp" in snippet


def test_source_bypass_gated_on_a_target_positive_not_return_phase():
    """①非矛盾性: バイパスが_a_target>0(寄せる最中)にのみ適用され、
    オフセット復帰側(_a_target=0)には及ばないことをソーステキストで確認する。"""
    snippet = _ramp_block_snippet()
    idx_if = snippet.index("if _a_target > 0.0 and _stopped_opp:")
    idx_else = snippet.index("else:", idx_if)
    ramp_snippet = snippet[idx_else:]
    assert "_ramp_step = dt / max(self._ot_ramp_time" in ramp_snippet


def test_source_edge_triggered_log_uses_alpha_prev_check():
    """③検証ロギング: [RAMP-BYPASS]ログがself._ot_alpha<1.0の判定でエッジ
    トリガーされ、多重ログを防いでいることを確認する。"""
    snippet = _ramp_block_snippet()
    assert "if self._ot_alpha < 1.0:" in snippet
    assert "[RAMP-BYPASS]" in snippet


def test_source_max_steering_rate_constraint_exists_justifying_bypass():
    """設計根拠の確認: MPC.py側にmax_steering_rateベースのレート制約
    (_rate_bounds)が実在し、ランプ省略の安全性の根拠(QP自体がジャーク相当を
    既に制限している)が成立していることを確認する。"""
    mpc_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "MPC.py")
    with open(mpc_path) as f:
        mpc_src = f.read()
    assert "def _rate_bounds(self, N):" in mpc_src
    assert "self.max_steering_rate" in mpc_src
