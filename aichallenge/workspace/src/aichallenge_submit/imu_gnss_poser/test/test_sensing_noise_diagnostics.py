"""Unit tests for the raw sensing-noise diagnostics A ([GNSS-NOISE]/[IMU-NOISE],
118節続報, 2026-07-19).

背景: 118節でGNSS共分散の再較正を行い、EKF可視率は改善したが、118-3節で
「コーナーでの真の逸脱量自体が対処前より大きくなっている区間がある。GNSSを
より強く信頼するようになったことで位置フィードバックがノイジーになった副作用
ではないか」という留保を開示した。この仮説を検証するには、GNSS/IMUの生値
そのもののノイズ(既知の軌道モデルを使わない直接推定)が必要になる。

対処: imu_gnss_poser_node.cpp に、既存の gnss_callback/imu_callback が既に
受信している生値を再利用し、[GNSS-NOISE](直近1秒窓への直交回帰の垂直残差RMS、
wp127-129衝突事象の事後解析「手法1」と同一の数式)と[IMU-NOISE](gyro wzの
標準偏差)を追加した。gnss_heading機能専用の既存バッファ(gnss_hist_、
enable=false/後退中は更新されない)とは別に専用バッファを持たせ、診断が
機能トグルや後退状態に依存しないようにした。

テスト方針: C++ノードはビルド・rclpy依存のため直接importできない。数式部分
(直交回帰の垂直残差RMS・標準偏差)は単純な閉形式のため、同一の式を複製した
Pythonミラーで数式的性質を検証する。C++ソース側の配線(専用バッファの独立性・
既存購読の再利用・間引き)は構造的なソーステキスト検証で確認する。
"""
import math
import os

_PKG_DIR = os.path.join(os.path.dirname(__file__), "..")
_CPP_PATH = os.path.join(_PKG_DIR, "src", "imu_gnss_poser_node.cpp")
with open(_CPP_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: line_fit_perp_residual_rms / stddev_scalar
# ---------------------------------------------------------------------------

def _line_fit_perp_residual_rms(pts):
    """imu_gnss_poser_node.cpp の line_fit_perp_residual_rms() の複製ミラー。
    pts: [(x, y), ...]"""
    n = len(pts)
    if n < 4:
        return -1.0
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    cxx = sum((p[0] - mx) ** 2 for p in pts) / n
    cyy = sum((p[1] - my) ** 2 for p in pts) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    nx, ny = -math.sin(theta), math.cos(theta)
    sum_sq = sum((nx * (p[0] - mx) + ny * (p[1] - my)) ** 2 for p in pts)
    return math.sqrt(sum_sq / n)


def _stddev_scalar(vals):
    """imu_gnss_poser_node.cpp の stddev_scalar() の複製ミラー。"""
    n = len(vals)
    if n < 4:
        return -1.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return math.sqrt(var)


def test_perfectly_straight_line_gives_near_zero_residual():
    """核心: ノイズの無い直線移動なら垂直残差RMSはほぼ0。"""
    pts = [(float(i) * 0.5, float(i) * 0.5) for i in range(10)]  # 45度の直線
    rms = _line_fit_perp_residual_rms(pts)
    assert rms < 1e-9


def test_known_perpendicular_jitter_recovered_approximately():
    """核心: 既知の垂直方向ジッタ(振幅0.1mの正負交互)を注入した直線から、
    残差RMSがその振幅オーダーで検出できることを確認する(実測σ推定の妥当性)。"""
    pts = []
    for i in range(20):
        x = float(i) * 0.5
        y = float(i) * 0.5 + (0.1 if i % 2 == 0 else -0.1)  # 進行方向に直交(y軸)へのジッタ
        pts.append((x, y))
    rms = _line_fit_perp_residual_rms(pts)
    assert 0.05 < rms < 0.15


def test_insufficient_points_returns_sentinel():
    """回帰: 4点未満では計算不能を示す負値を返す(C++側のn<4ガードと同じ)。"""
    assert _line_fit_perp_residual_rms([(0.0, 0.0), (1.0, 1.0)]) == -1.0


def test_stddev_scalar_matches_known_values():
    """核心: 標準偏差計算式そのものの正しさを確認する。"""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    # population std of [1..5] = sqrt(2.0)
    assert abs(_stddev_scalar(vals) - math.sqrt(2.0)) < 1e-9


def test_stddev_scalar_insufficient_points_returns_sentinel():
    assert _stddev_scalar([1.0, 2.0]) == -1.0


# ---------------------------------------------------------------------------
# 2) C++ソース側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_new_params_declared():
    assert 'declare_parameter("sensing_noise.window_s"' in _SRC
    assert 'declare_parameter("sensing_noise.log_interval_s"' in _SRC
    assert 'declare_parameter("sensing_noise.min_disp_m"' in _SRC


def test_noise_buffer_is_separate_from_gnss_heading_buffer():
    """非矛盾性の核心: 診断用バッファ(noise_gnss_hist_)は、gnss_heading専用の
    既存バッファ(gnss_hist_、enable=false/後退中は更新されない)とは別物である
    ことを確認する。診断がheading機能のON/OFFや後退状態に依存してはならない。"""
    assert "std::deque<TimedPoint2D> noise_gnss_hist_;" in _SRC
    assert "std::deque<TimedPoint> gnss_hist_;" in _SRC


def test_noise_update_called_unconditionally_in_gnss_callback():
    """核心: update_and_maybe_log_sensing_noise()の呼び出しが、gnss_heading機能の
    enable/reversingガード(apply_gnss_track_headingの内部)の外側、gnss_callback
    本体から直接呼ばれていることを確認する。"""
    idx_cb = _SRC.index("void gnss_callback(")
    idx_next_fn = _SRC.index("void adjust_covariance(", idx_cb)
    snippet = _SRC[idx_cb:idx_next_fn]
    assert "update_and_maybe_log_sensing_noise(" in snippet


def test_imu_callback_appends_wz_history():
    idx = _SRC.index("void imu_callback(")
    snippet = _SRC[idx:idx + 400]
    assert "noise_wz_val_.push_back(msg->angular_velocity.z)" in snippet
    assert "noise_wz_t_.push_back(" in snippet


def test_gnss_noise_log_has_expected_fields():
    idx = _SRC.index('"[GNSS-NOISE]')
    snippet = _SRC[max(0, idx - 300):idx + 200]
    assert "rms=%.4f" in snippet
    assert "disp >= noise_min_disp_m_" in snippet  # 最小変位ゲート(停止時の誤検知防止)


def test_imu_noise_log_has_expected_fields():
    idx = _SRC.index('"[IMU-NOISE]')
    snippet = _SRC[max(0, idx - 100): idx + 200]
    assert "wz_std=%.5f" in snippet


def test_log_throttle_uses_dedicated_interval_not_every_message():
    """回帰: GNSS(20Hz程度)毎に出力せず、既存のnoise_log_interval_sで間引くことを確認する
    (ログ量の暴走防止)。"""
    idx = _SRC.index("noise_last_log_t_ >= 0.0")
    snippet = _SRC[idx:idx + 150]
    assert "noise_log_interval_s_" in snippet
