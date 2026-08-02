"""Unit tests for the Savitzky-Golay post-processing kappa smoothing
(`use_savgol_kappa`, 215節続報2, AXIS08 原因2, 2026-07-28)。

背景: 215節で、5次内挿スプライン(`use_c2_spline_kappa`)が局所的なキンクを
滑らかにせずむしろ増幅する(厳密内挿ゆえのオーバーシュート)ことが判明した。
Geminiはこれを踏まえ、厳密内挿ではなく局所多項式最小二乗近似である
Savitzky-Golayフィルタを、算出済みのkappa配列に対する後処理として提案した。

実装前に、215節で使った同じ合成シナリオ(単一点突起・緩やかコーナー+微小乱れ・
ヘアピンでのピーク保持)でSavitzky-Golayを検証したところ、C2スプラインとは
逆に一貫して改善することを確認した(単一点突起の隣接差分で約4.7倍改善、
局所stdで約3倍改善、ピーク曲率は潰れず微増)。

`core/reference_path.py`はrclpy非依存のため、実際に`ReferencePath`/
`_apply_savgol_kappa`をimportして数値検証する。
"""
import inspect
import os

import numpy as np
import pytest

from multi_purpose_mpc_ros.core.reference_path import (
    ReferencePath, _apply_savgol_kappa)

_MC_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_MC_SRC_PATH) as _f:
    _MC_SRC = _f.read()

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG = _f.read()


def _make_ref_path_stub(circular=True, smoothing_distance=1, resolution=5.0,
                         use_savgol_kappa=False, savgol_window_length=7,
                         savgol_polyorder=2):
    obj = object.__new__(ReferencePath)
    obj.circular = circular
    obj.smoothing_distance = smoothing_distance
    obj.resolution = resolution
    obj.use_savgol_kappa = use_savgol_kappa
    obj.savgol_window_length = savgol_window_length
    obj.savgol_polyorder = savgol_polyorder
    obj.eps = 1e-12
    return obj


# ---------------------------------------------------------------------------
# 1) _apply_savgol_kappa 単体の数式検証
# ---------------------------------------------------------------------------

def test_output_length_matches_input():
    kappa = np.sin(np.linspace(0, 4 * np.pi, 100)) * 0.1
    out = _apply_savgol_kappa(kappa, window_length=7, polyorder=2, circular=True)
    assert len(out) == len(kappa)


def test_constant_kappa_unchanged():
    """定曲率(円)なら平滑化しても値は変わらないはず(最小二乗多項式フィットが厳密一致)。"""
    kappa = np.full(50, 0.05)
    out = _apply_savgol_kappa(kappa, window_length=7, polyorder=2, circular=True)
    assert np.allclose(out, 0.05, atol=1e-9)


def test_single_spike_jump_reduced_vs_raw():
    """単一点スパイクに対し、隣接差分の最大値が生値より小さくなること
    (215節のC2スプラインとは逆に、悪化ではなく改善することを確認)。"""
    kappa = np.zeros(60)
    kappa[30] = 0.5  # 孤立した単一点スパイク
    out = _apply_savgol_kappa(kappa, window_length=7, polyorder=2, circular=False)
    jump_raw = np.max(np.abs(np.diff(kappa)))
    jump_sg = np.max(np.abs(np.diff(out)))
    assert jump_sg < jump_raw


def test_wrap_mode_seam_continuity():
    """circular=Trueでmode='wrap'を使うことで、始点・終点の接続部が不連続に
    ならないことを確認する(楕円型の非一定曲率データで検証)。"""
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    a, b = 25.0, 12.0
    # 解析的な楕円curvature(参考データとして生成、平滑化の入力に使う)
    dx = -a * np.sin(theta)
    dy = b * np.cos(theta)
    d2x = -a * np.cos(theta)
    d2y = -b * np.sin(theta)
    kappa = (dx * d2y - dy * d2x) / np.power(dx ** 2 + dy ** 2, 1.5)
    out = _apply_savgol_kappa(kappa, window_length=7, polyorder=2, circular=True)
    typical_diff = np.mean(np.abs(np.diff(out)))
    seam_diff = abs(out[0] - out[-1])
    assert seam_diff < typical_diff * 10 + 1e-6


def test_peak_curvature_not_crushed_like_moving_average():
    """Gemini提案の主張(移動平均と異なりピーク曲率を潰さない)を検証する。
    ヘアピン形状(なめらかな半円)のピーク|kappa|が、平滑化後も
    大きく減少しない(90%以上を保持する)ことを確認する。"""
    n = 100
    theta = np.linspace(-np.pi / 2, np.pi / 2, n)
    radius = 5.0
    x = radius * np.sin(theta)
    y = radius * (1 - np.cos(theta))
    dx = np.diff(x)
    dy = np.diff(y)
    psi = np.arctan2(dy, dx)
    kappa = np.zeros(n - 1)
    for i in range(1, n - 1):
        d_ahead = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        d_behind = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        psi_ahead = np.arctan2(d_ahead[1], d_ahead[0])
        psi_behind = np.arctan2(d_behind[1], d_behind[0])
        angle_dif = np.mod(psi_ahead - psi_behind + np.pi, 2 * np.pi) - np.pi
        kappa[i] = angle_dif / (np.linalg.norm(d_ahead) + 1e-12)
    out = _apply_savgol_kappa(kappa, window_length=7, polyorder=2, circular=False)
    peak_raw = np.max(np.abs(kappa))
    peak_sg = np.max(np.abs(out))
    assert peak_sg >= peak_raw * 0.9


# ---------------------------------------------------------------------------
# 2) ReferencePath._construct_path統合: フラグの配線とスコープ限定
# ---------------------------------------------------------------------------

def test_flag_default_value_is_false():
    sig = inspect.signature(ReferencePath.__init__)
    assert sig.parameters["use_savgol_kappa"].default is False
    assert sig.parameters["savgol_window_length"].default == 7
    assert sig.parameters["savgol_polyorder"].default == 2


def test_flag_true_preserves_xy_coordinates_exactly():
    """④遡及効果: フラグtrueでも(x,y)座標は一切変更されない。"""
    n = 150
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = (20.0 * np.cos(theta)).tolist()
    y = (20.0 * np.sin(theta)).tolist()

    stub_false = _make_ref_path_stub(use_savgol_kappa=False)
    stub_true = _make_ref_path_stub(use_savgol_kappa=True)
    wps_false = stub_false._construct_path(list(x), list(y))
    wps_true = stub_true._construct_path(list(x), list(y))

    assert len(wps_false) == len(wps_true)
    for wf, wt in zip(wps_false, wps_true):
        assert wf.x == pytest.approx(wt.x)
        assert wf.y == pytest.approx(wt.y)


def test_flag_true_preserves_psi_unchanged():
    """psiには一切触れない(kappaのみを平滑化する)ことを確認する。"""
    n = 150
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = (20.0 * np.cos(theta)).tolist()
    y = (20.0 * np.sin(theta)).tolist()

    stub_false = _make_ref_path_stub(use_savgol_kappa=False)
    stub_true = _make_ref_path_stub(use_savgol_kappa=True)
    wps_false = stub_false._construct_path(list(x), list(y))
    wps_true = stub_true._construct_path(list(x), list(y))

    for wf, wt in zip(wps_false, wps_true):
        assert wf.psi == pytest.approx(wt.psi)


def test_flag_true_changes_kappa_values():
    n = 200
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = (25.0 * np.cos(theta)).tolist()
    y = (12.0 * np.sin(theta)).tolist()  # 楕円(曲率が一定でない)

    stub_false = _make_ref_path_stub(use_savgol_kappa=False)
    stub_true = _make_ref_path_stub(use_savgol_kappa=True)
    wps_false = stub_false._construct_path(list(x), list(y))
    wps_true = stub_true._construct_path(list(x), list(y))

    diff = np.array([wt.kappa - wf.kappa for wf, wt in zip(wps_false, wps_true)])
    assert np.mean(np.abs(diff)) > 1e-6


def test_flag_true_reduces_synthetic_kink_unlike_c2_spline():
    """215節で使ったのと同じ合成キンクシナリオに対し、C2スプラインとは逆に
    改善する(隣接差分の最大値が小さくなる)ことを確認する。"""
    n = 60
    x = np.linspace(0.0, 60.0, n)
    y = np.zeros(n)
    y[30] += 0.15  # 212節のキンクを模した突起(215節と同一)
    x_list, y_list = x.tolist(), y.tolist()

    stub_false = _make_ref_path_stub(circular=False, use_savgol_kappa=False)
    stub_true = _make_ref_path_stub(circular=False, use_savgol_kappa=True)
    wps_false = stub_false._construct_path(x_list, y_list)
    wps_true = stub_true._construct_path(x_list, y_list)

    kappa_false = np.array([wp.kappa for wp in wps_false])
    kappa_true = np.array([wp.kappa for wp in wps_true])
    peak_jump_false = np.max(np.abs(np.diff(kappa_false)))
    peak_jump_true = np.max(np.abs(np.diff(kappa_true)))
    assert peak_jump_true < peak_jump_false


def test_short_waypoint_list_skips_savgol_without_crashing():
    """waypoint数がsavgol_window_length以下の場合、例外を出さずスキップすること
    (window_lengthを超えるデータ長を要求するsavgol_filterの制約に対する防御)。"""
    x = [0.0, 1.0, 2.0]
    y = [0.0, 0.0, 0.0]
    stub_true = _make_ref_path_stub(circular=False, smoothing_distance=0, resolution=5.0,
                                     use_savgol_kappa=True, savgol_window_length=7)
    waypoints = stub_true._construct_path(x, y)  # 例外が出ないことそのものが検証対象
    assert len(waypoints) >= 1


# ---------------------------------------------------------------------------
# 3) mpc_controller.py / config.yaml側の配線(ソーステキスト検証)
# ---------------------------------------------------------------------------

def test_config_yaml_declares_savgol_params():
    """218節: use_savgol_kappaはfield検証のため一時的にtrueへ変更されており
    (debug_extra_actuator_delay_sと同様、実験フェーズにより現在値が変わる想定の
    ライブトグルのため)、config.yaml側は値そのものではなくキーの宣言・型のみを
    確認する。Pythonデフォルト(False)自体はtest_flag_default_value_is_falseで
    別途固定的に検証済み。"""
    import re
    assert re.search(r"^\s*use_savgol_kappa:\s*(true|false)\s*(#.*)?$", _CFG, re.MULTILINE)
    assert re.search(r"^\s*savgol_window_length:\s*7\s*(#.*)?$", _CFG, re.MULTILINE)
    assert re.search(r"^\s*savgol_polyorder:\s*2\s*(#.*)?$", _CFG, re.MULTILINE)


def test_create_ref_path_both_branches_pass_savgol_flags_through():
    assert _MC_SRC.count(
        'bool(getattr(cfg_ref_path, "use_savgol_kappa", False))') == 2
    assert _MC_SRC.count(
        'int(getattr(cfg_ref_path, "savgol_window_length", 7))') == 2
    assert _MC_SRC.count(
        'int(getattr(cfg_ref_path, "savgol_polyorder", 2))') == 2


def test_startup_log_for_use_savgol_kappa_present():
    assert '"[CONFIG] use_savgol_kappa: "' in _MC_SRC


def test_pit_ref_path_not_affected_by_new_flag():
    idx = _MC_SRC.index("def _build_pit_ref_path")
    snippet = _MC_SRC[idx:idx + 1500]
    assert "use_savgol_kappa" not in snippet
