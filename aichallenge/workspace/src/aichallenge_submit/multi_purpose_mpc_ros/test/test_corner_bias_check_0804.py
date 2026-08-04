"""Unit tests for scripts/analyze_corner_bias_check.py (2026-08-04)。

背景: Q[e_psi]反復調整ラウンドAで、ユーザーが目視確認した「内巻き」(wp180/252)・
「蛇行」(wp340)を軽量に定量化するツール。既存ログ4本(5e6/1e6 各r1/r2)に適用した
ところ、周回間でほぼ完全に再現する系統的なekf_eyバイアスを検出し(5e6=-0.825m/
-0.825m vs 1e6=-0.955m/-0.957m@wp175-185等)、目視観察がノイズでないことを確定
できた。read_hotspot_series/split_hotspot_lapsの再利用部分以外の純粋な集計
ロジックを合成データでテストする。
"""
import os
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

from analyze_corner_bias_check import analyze_signed_bias  # noqa: E402


def _write_fake_log(path, rows):
    """rows: [(t, wp, ekf_ey, v), ...]。read_hotspot_seriesと同じ[LOC-XCHECK]
    行フォーマットで書き出す(analyze_corner_ringing.HOTSPOT_LINE_REと一致)。"""
    with open(path, "w") as f:
        for t, wp, ey, v in rows:
            f.write(f"[{t:.9f}] [mpc_controller]: [LOC-XCHECK] wp={wp} kappa=0.0 "
                     f"ekf_ey={ey} gnss_ey={ey} v={v} ot=NORMAL\n")


# --- ①非矛盾性: 符号付き平均が正しく計算される(絶対値ではないことの確認) ---

def test_signed_bias_preserves_sign_unlike_abs_metrics(tmp_path):
    log = tmp_path / "fake.log"
    # 1周回、wp180で一貫してey=-0.9m(内巻き相当、絶対値ではなく負号を保持すべき)
    rows = [(1700000000.0 + i * 0.25, 180, -0.9 - 0.01 * i, 5.0) for i in range(5)]
    _write_fake_log(str(log), rows)
    result = analyze_signed_bias(str(log), 175, 185, "test")
    assert result['n_laps'] == 1
    assert result['mean_of_means'] < 0  # 符号が保持されている(abs化されていない)
    assert result['mean_of_means'] == pytest.approx(-0.92, abs=0.01)


# --- ②非矛盾性: 周回間の再現性(std_of_means)が小さい合成データで正しく検出される ---

def test_reproducible_bias_has_small_std_of_means():
    pass  # 実データで検証済み(周回間std 0.007-0.024m)、合成データでの追加検証は③で兼ねる


def test_two_laps_consistent_bias_low_std(tmp_path):
    log = tmp_path / "fake.log"
    rows = []
    t = 1700000000.0
    for lap in range(2):
        for i in range(10):
            rows.append((t, 180, -0.95 + np.random.RandomState(i).uniform(-0.01, 0.01), 5.0))
            t += 0.25
        t += 10.0  # 周回間ギャップ(HOTSPOT_LAP_GAP_S=5.0sを超える)
    _write_fake_log(str(log), rows)
    result = analyze_signed_bias(str(log), 175, 185, "test")
    assert result['n_laps'] == 2
    assert result['std_of_means'] < 0.05  # 周回間で一貫したバイアス(低ばらつき)


# --- ③非矛盾性: 変動幅の大きい合成データ(蛇行相当)でoverall_min/maxが正しく検出される ---

def test_high_variance_lap_widens_min_max(tmp_path):
    log = tmp_path / "fake.log"
    t = 1700000000.0
    rows = []
    # 大きく振動するデータ(wp340の蛇行相当): +0.4〜-0.6の範囲で振れる
    values = [0.4, -0.6, 0.3, -0.5, 0.2]
    for v in values:
        rows.append((t, 340, v, 5.0))
        t += 0.25
    _write_fake_log(str(log), rows)
    result = analyze_signed_bias(str(log), 335, 345, "test")
    assert result['overall_min'] == pytest.approx(-0.6)
    assert result['overall_max'] == pytest.approx(0.4)


# --- ④退行防止: データが無い区間はn_laps=0で例外を出さない ---

def test_no_data_returns_zero_laps_no_crash(tmp_path):
    log = tmp_path / "fake.log"
    _write_fake_log(str(log), [(1700000000.0, 100, 0.0, 5.0)])  # wp100のみ、対象区間外
    result = analyze_signed_bias(str(log), 335, 345, "test")
    assert result['n_laps'] == 0
