"""Unit tests for the GNSS-EKF time-lag diagnostic D ([GNSS-EKF-XCORR], 118節続報,
2026-07-19)。

背景: 116節で「wp64-87/wp232-239だけEKF可視率が特に低い理由」を、コーナー滞在
時間・横加速度・GNSS共分散との相関で調べたが単一要因を特定できなかった。まだ
検証していない切り口として「EKFの遅延量そのものがコーナーの動特性によって
変動するか」がある。過去の手動解析(2026-06-29、an13.py)では「EKFはGNSSに対し
約60ms先行する(遅延ではない)」と分かっていたが、これは単発のオフライン解析
であり、常設の計装ではなかった。

対処: mpc_controller.pyに、既存の self._odom(EKF)・self._gnss_pose(生GNSS、
既にピット用途で購読済み)それぞれの時刻付き位置履歴を保持し、周期的に
相互相関(ラグ探索)を行い、EKFがGNSSに対して何ms先行/遅延しているかを
[GNSS-EKF-XCORR]としてログする。

符号規約: lag_ms>0 は EKF(t) ≈ GNSS(t-lag) を意味し、EKFがGNSSより遅延している
ことを示す。lag_ms<0 はEKFが先行(2026-06-29の手動解析an13.pyと同じ規約)。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。相互相関
探索アルゴリズム自体は単純な数式(np.interpによるリサンプル+RMS残差の最小化)
のため、同一のロジックを複製したPythonミラーで、既知のラグを注入した合成信号
から正しく復元できることを数式的に実証する。mpc_controller.py側の配線(既存
購読の再利用・ウィンドウ管理・間引き)は構造的なソーステキスト検証で確認する。
"""
import os
import math
import numpy as np

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: ラグ探索アルゴリズム
# ---------------------------------------------------------------------------

def _xcorr_best_lag(ekf_t, ekf_x, ekf_y, gnss_t, gnss_x, gnss_y,
                     lag_max=0.3, lag_step=0.02):
    """mpc_controller.py の _maybe_log_gnss_ekf_xcorr() 内のラグ探索の複製ミラー。"""
    ekf_t = np.asarray(ekf_t); ekf_x = np.asarray(ekf_x); ekf_y = np.asarray(ekf_y)
    gnss_t = np.asarray(gnss_t); gnss_x = np.asarray(gnss_x); gnss_y = np.asarray(gnss_y)
    t_lo = gnss_t[0] + lag_max
    t_hi = gnss_t[-1] - lag_max
    mask = (ekf_t >= t_lo) & (ekf_t <= t_hi)
    ekf_t_m, ekf_x_m, ekf_y_m = ekf_t[mask], ekf_x[mask], ekf_y[mask]

    lags = np.arange(-lag_max, lag_max + 1e-9, lag_step)
    best_lag, best_resid, resid_at_zero = 0.0, None, None
    for lag in lags:
        gx_i = np.interp(ekf_t_m - lag, gnss_t, gnss_x)
        gy_i = np.interp(ekf_t_m - lag, gnss_t, gnss_y)
        resid = float(np.sqrt(np.mean((ekf_x_m - gx_i) ** 2 + (ekf_y_m - gy_i) ** 2)))
        if abs(lag) < 1e-9:
            resid_at_zero = resid
        if best_resid is None or resid < best_resid:
            best_resid, best_lag = resid, float(lag)
    return best_lag, best_resid, resid_at_zero


def _make_synthetic_series(true_lag_s, n=200, dt=0.025, noise=0.0, seed=0):
    """GNSS(t)=円軌道、EKF(t)=GNSS(t - true_lag)(EKFがtrue_lag秒だけ遅延)を合成する。
    true_lag>0 → EKFが遅延、true_lag<0 → EKFが先行、という規約でテストする。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    # 単調でない(直線ではラグの効果が並進とほぼ等価になり退化するため)円軌道を使う
    gx = 5.0 * np.cos(0.5 * t)
    gy = 5.0 * np.sin(0.5 * t)
    ekf_t = t
    ekf_x = 5.0 * np.cos(0.5 * (t - true_lag_s)) + rng.normal(0, noise, n)
    ekf_y = 5.0 * np.sin(0.5 * (t - true_lag_s)) + rng.normal(0, noise, n)
    return ekf_t, ekf_x, ekf_y, t, gx, gy


def test_recovers_known_positive_lag_ekf_delayed():
    """核心: EKFがGNSSに対して0.08s遅延している合成信号から、
    正のlag(遅延)がおおよそ復元できることを確認する。"""
    ekf_t, ekf_x, ekf_y, gt, gx, gy = _make_synthetic_series(true_lag_s=0.08)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(ekf_t, ekf_x, ekf_y, gt, gx, gy)
    assert abs(best_lag - 0.08) <= 0.02 + 1e-9  # 探索刻み(0.02s)以内で一致
    assert best_resid < resid_zero  # 真のラグでの残差はゼロラグ仮定より必ず小さい


def test_recovers_known_negative_lag_ekf_leads():
    """核心: EKFがGNSSに対して0.06s先行している合成信号(2026-06-29のan13.py実測相当)
    から、負のlag(先行)がおおよそ復元できることを確認する。"""
    ekf_t, ekf_x, ekf_y, gt, gx, gy = _make_synthetic_series(true_lag_s=-0.06)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(ekf_t, ekf_x, ekf_y, gt, gx, gy)
    assert abs(best_lag - (-0.06)) <= 0.02 + 1e-9
    assert best_resid < resid_zero


def test_zero_lag_when_series_identical():
    """回帰: ラグが無い(EKF=GNSS)場合、最良ラグは0付近で、resid_at_bestと
    resid_at_zeroがほぼ一致する。"""
    ekf_t, ekf_x, ekf_y, gt, gx, gy = _make_synthetic_series(true_lag_s=0.0)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(ekf_t, ekf_x, ekf_y, gt, gx, gy)
    assert abs(best_lag) <= 0.02 + 1e-9
    assert math.isclose(best_resid, resid_zero, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 2) mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_xcorr_buffers_separate_from_pit_heading_buffer():
    """非矛盾性: 相互相関用の時刻付きバッファ(_xcorr_ekf_hist/_xcorr_gnss_hist)は、
    ピット用途の既存_gnss_hist(時刻を持たない)とは別物であることを確認する。"""
    assert "self._xcorr_ekf_hist: List[Tuple[float, float, float]] = []" in _SRC
    assert "self._xcorr_gnss_hist: List[Tuple[float, float, float]] = []" in _SRC
    assert "self._gnss_hist: List[Tuple[float, float]] = []" in _SRC


def test_odom_callback_appends_ekf_history():
    idx = _SRC.index("def _odom_callback(")
    snippet = _SRC[idx:idx + 500]
    assert "self._xcorr_ekf_hist.append((t, p.x, p.y))" in snippet
    assert "_XCORR_WINDOW_S" in snippet


def test_gnss_pose_callback_appends_xcorr_history():
    idx = _SRC.index("def _gnss_pose_callback(")
    snippet = _SRC[idx:idx + 700]
    assert "self._xcorr_gnss_hist.append((t, p.x, p.y))" in snippet
    assert "_XCORR_WINDOW_S" in snippet


def test_xcorr_evaluated_unconditionally_not_gated_by_obstacle_avoidance():
    """核心: [GNSS-EKF-XCORR]の評価がUSE_OBSTACLE_AVOIDANCEブロックの外側
    (オーバーテイク非依存の一般センシング診断)で行われていることを確認する。"""
    idx_call = _SRC.index("self._maybe_log_gnss_ekf_xcorr()")
    idx_obstacle_block = _SRC.index("if self.USE_OBSTACLE_AVOIDANCE:", idx_call - 2000)
    assert idx_call < idx_obstacle_block


def test_xcorr_log_has_expected_fields():
    idx = _SRC.index('"[GNSS-EKF-XCORR]')
    snippet = _SRC[idx:idx + 250]
    assert "lag_ms=" in snippet
    assert "resid_at_best=" in snippet
    assert "resid_at_zero=" in snippet
    assert "n_ekf=" in snippet
    assert "n_gnss=" in snippet


def test_xcorr_throttle_reuses_existing_rate_idiom_no_new_constant():
    """非冗長性: 間引きは既存のcontrol_rateベースのイディオムを再利用しており、
    新しい間引き定数を導入していないことを確認する。"""
    idx = _SRC.index("self._maybe_log_gnss_ekf_xcorr()")
    snippet = _SRC[max(0, idx - 300):idx]
    assert "int(max(1, self._mpc_cfg.control_rate))" in snippet


def test_xcorr_method_guards_against_insufficient_data():
    """回帰: バッファが十分に溜まっていない(起動直後等)場合、例外を投げず
    早期returnすることを確認する(存在確認のみ、実行はrclpy依存のため不可)。"""
    idx = _SRC.index("def _maybe_log_gnss_ekf_xcorr(self)")
    # 2026-07-26追加(競合状態バグ修正)のdocstringが伸びた分、窓を広げた
    # (意味的な変更はない、test_gnss_ekf_xcorr_race_condition.py参照)。
    snippet = _SRC[idx:idx + 1500]
    assert "if len(ekf_h) < 20 or len(gnss_h) < 10:" in snippet
    assert "return" in snippet
