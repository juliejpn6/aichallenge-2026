"""Unit tests for scripts/analyze_steer_rate_saturation.py (2026-08-04、蛇行対策Phase 1改訂版)。

背景: 統一機構仮説(SWING区間で必要操舵レート dδ/dt≈L_wb·κ'·v が r_max で飽和し
過渡オーバーシュートを生む)を、トラック全域(コーナーだけでなく直線含む)で
成分分解(A: κ'駆動 / B: κ'非依存)して検証するツール。ログ依存部分(mcap/rosbag
読み込み)以外の純粋な計算ロジック(κ'真値・単位較正・回帰・層別集計)を、
scripts/をsys.pathへ追加した上で直接importしてテストする(mcap/PyYAML等の
依存が無い環境でも実行可能な部分に限定)。
"""
import os
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

from analyze_steer_rate_saturation import (  # noqa: E402
    compute_kappa_prime_true, compute_kappa_swing_window, calibrate_r_max_analytical,
    smooth_short_label_runs,
    empirical_r_max_deg_s, build_window_table, regress_component_a_b,
    bucket_by_swing_quartile, straight_quantization_check, horizon_desk_check,
)


# --- ①非矛盾性: κ'真値は既知の解析的曲率プロファイルで正しい符号・大きさになる ---

def test_kappa_prime_true_linear_ramp():
    """kappaがs方向に線形増加(傾き一定)する場合、中央差分κ'は定数(=傾き)に
    一致するはず。"""
    s_m = np.linspace(0, 99, 100)
    kappa = 0.01 * s_m  # 傾き0.01
    kp = compute_kappa_prime_true(s_m, kappa)
    assert kp[10] == pytest.approx(0.01, abs=1e-9)
    assert kp[50] == pytest.approx(0.01, abs=1e-9)


def test_kappa_prime_true_constant_kappa_is_zero():
    """真の直線(kappa一定)ならκ'はほぼゼロになる。"""
    s_m = np.linspace(0, 99, 100)
    kappa = np.full(100, 0.05)
    kp = compute_kappa_prime_true(s_m, kappa)
    assert np.allclose(kp, 0.0, atol=1e-9)


def test_kappa_prime_true_endpoints_use_one_sided_difference():
    """両端は片側差分になる(周回継ぎ目の複雑な処理を避ける簡略化、非冗長性②)。"""
    s_m = np.array([0.0, 1.0, 2.0, 3.0])
    kappa = np.array([0.0, 0.1, 0.2, 0.3])
    kp = compute_kappa_prime_true(s_m, kappa)
    assert kp[0] == pytest.approx(0.1)  # 片側差分: (kappa[1]-kappa[0])/(s[1]-s[0])
    assert kp[-1] == pytest.approx(0.1)  # 片側差分: (kappa[-1]-kappa[-2])/(s[-1]-s[-2])


# --- ②既存swing定義(177節、窓内max-min)との非退行確認 ---

def test_kappa_swing_window_matches_max_minus_min():
    kappa = np.array([0.0, 0.05, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    swing = compute_kappa_swing_window(kappa, window=2)
    # i=1: window=[1,2,3]→[0.05,0.1,0.05]→max-min=0.05
    assert swing[1] == pytest.approx(0.05)
    # i=2: window=[2,3,4]→[0.1,0.05,0.0]→max-min=0.1
    assert swing[2] == pytest.approx(0.1)


# --- ③単位較正: config由来の2候補算出が正しいこと ---

def test_calibrate_r_max_analytical_candidates():
    cfg = {'steer_rate_max_raw': 1.1, 'gain': 1.639, 'wheel_base_m': 1.087, 'N': 20}
    result = calibrate_r_max_analytical(cfg, comment_value_deg_s=109.0)
    rad2deg = 180.0 / np.pi
    assert result['candidate_a_deg_s'] == pytest.approx(1.1 * rad2deg)
    assert result['candidate_b_deg_s'] == pytest.approx(1.1 * 1.639 * rad2deg)
    # B(raw×gain)の方がコメント値(109°/s)に近いはず(実データでの実測結果と整合)
    assert result['closer_candidate'] == 'B(raw×gain)'


def test_empirical_r_max_uses_high_percentile_not_max():
    """④退行防止: 経験的較正は外れ値(単発の巨大スパイク)に引きずられず、
    percentile(既定99.5)を使う——maxをそのまま採用しないこと。"""
    rng = np.concatenate([np.full(1000, 50.0), np.array([99999.0])])  # 極端な外れ値1件
    result = empirical_r_max_deg_s(rng, percentile=99.5)
    assert result['p_value_deg_s'] < 1000.0  # 外れ値に支配されていない
    assert result['max_deg_s'] == pytest.approx(99999.0)  # ただしmaxは別途記録・報告される


# --- ④成分A/B回帰: 既知の合成データで正しく分解できること ---

def test_regress_component_a_b_pure_component_a():
    """wobble_stdが完全にpredicted_rateに比例する合成データなら、R²=1・
    残差=0になるはず(成分Aのみ、成分B無し)。"""
    window_rows = [
        {'predicted_rate_deg_s': x, 'wobble_std_deg': 2.0 * x}
        for x in [1.0, 2.0, 3.0, 4.0, 5.0]
    ]
    result = regress_component_a_b(window_rows)
    assert result['r2'] == pytest.approx(1.0, abs=1e-6)
    assert result['slope'] == pytest.approx(2.0, abs=1e-6)
    for r in window_rows:
        assert r['residual_deg'] == pytest.approx(0.0, abs=1e-6)


def test_regress_component_a_b_pure_component_b():
    """wobble_stdがpredicted_rateとほぼ無関係(小さなノイズのみ、成分Bのみ)の
    場合、原点通過回帰は弱い/悪いフィットになり、残差(=ほぼwobble_std自体)が
    大きく残る——実データ(全7ログでR²が大きく負)と同じ性質を、既知の合成
    データで再現する(ss_tot=0となる完全定数は、既存コードが意図的にNoneを
    返す境界ケースのため避ける——1つ下のtest_regress_..._zero_variance参照)。"""
    window_rows = [
        {'predicted_rate_deg_s': x, 'wobble_std_deg': v}
        for x, v in zip([0.1, 1.0, 2.0, 5.0, 10.0], [9.8, 10.1, 9.9, 10.2, 9.9])
    ]
    result = regress_component_a_b(window_rows)
    assert result['r2'] < -0.5  # 原点通過回帰はほぼ定数のターゲットに対し大きく負のR²になる
    # 残差の大部分がwobble_std自体として残る(=説明できていない、成分Bが支配的)
    mean_abs_residual = np.mean([abs(r['residual_deg']) for r in window_rows])
    assert mean_abs_residual > 3.0


def test_regress_component_a_b_zero_variance_returns_none_not_crash():
    """境界ケース: wobble_stdが厳密に定数(ss_tot=0)の場合はR²が定義不能
    ——例外にせずNoneを返す(実データでは起こらないが、堅牢性として確認)。"""
    window_rows = [{'predicted_rate_deg_s': x, 'wobble_std_deg': 10.0}
                    for x in [0.1, 1.0, 2.0, 5.0, 10.0]]
    result = regress_component_a_b(window_rows)
    assert result['r2'] is None


# --- ⑤4層バケット分割: 四分位境界が正しいこと ---

def test_bucket_by_swing_quartile_splits_into_four_labels():
    window_rows = [
        {'kappa_prime_center': k, 'wobble_std_deg': 10.0, 'saturation_rate': 0.01,
         'residual_deg': 0.0}
        for k in [0.001, 0.002, 0.01, 0.02, 0.05, 0.06, 0.1, 0.2]
    ]
    buckets = bucket_by_swing_quartile(window_rows)
    labels = [b['label'] for b in buckets]
    assert labels == ['直線(Q1)', '低swing(Q2)', '中swing(Q3)', '高swing(Q4)']
    total = sum(b['n_windows'] for b in buckets)
    assert total == len(window_rows)  # 非冗長性: 全窓が必ずどこか1つの層に属する


# --- ⑥直線区間量子化検査: 明らかなアーティファクトを検出できること ---

def test_straight_quantization_check_detects_noisy_kappa_prime():
    s_m = np.linspace(0, 9, 10)
    kappa_prime_clean = np.zeros(10)
    kappa_prime_noisy = np.array([0.0, 0.05, -0.05, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    idx = np.arange(10)
    clean_result = straight_quantization_check(s_m, kappa_prime_clean, idx)
    noisy_result = straight_quantization_check(s_m, kappa_prime_noisy, idx)
    assert clean_result['kappa_prime_rms'] == pytest.approx(0.0)
    assert noisy_result['kappa_prime_rms'] > 0.01
    assert noisy_result['kappa_prime_rms'] > clean_result['kappa_prime_rms']


# --- ⑦ホライズン机上検査: 速度が上がるほど余裕が減ること ---

def test_horizon_desk_check_margin_decreases_with_speed():
    results = horizon_desk_check(N=20, wp_spacing_m=0.6, v_kmh_list=[15.0, 20.0, 35.0])
    margins = [r['margin_s'] for r in results]
    assert margins[0] > margins[1] > margins[2]  # 速度が上がるほど先読み時間が短くなる


# --- ⑧build_window_table: rate系列とangle系列でwp割り当てを別々に扱うこと ---

def test_build_window_table_uses_separate_wp_assignment_for_rate_and_angle():
    """④回帰防止: angle系列(N点)とrate系列(N-1点)のwp割り当てを取り違えて
    ズレたインデックスで参照すると、rate系列が短い分だけ末尾がtruncateされ
    ズレた集計になる。別々の配列として渡すことで正しく対応することを確認する。"""
    s_m = np.linspace(0, 9, 10)
    kappa_prime = np.full(10, 0.01)
    sample_wp = np.array([0, 0, 1, 1, 2, 2])  # angle系列相当(6点)
    sample_angle_deg = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    sample_wp_for_rate = np.array([0, 1, 1, 2, 2])  # rate系列相当(5点、N-1)
    sample_rate_deg_s = np.array([40.0, 40.0, 40.0, 40.0, 40.0])
    sample_v_mps = np.full(6, 5.0)
    rows = build_window_table(
        s_m, kappa_prime, wp_spacing_m=1.0, sample_wp=sample_wp,
        sample_angle_deg=sample_angle_deg, sample_wp_for_rate=sample_wp_for_rate,
        sample_rate_deg_s=sample_rate_deg_s, sample_v_mps=sample_v_mps,
        wheel_base_m=1.087, window_wp=1, r_max_deg_s=100.0, min_samples=1)
    # 例外なく実行でき、各窓が対応するrateサンプル数を正しく参照できていることを
    # 間接確認する(sample_rate_deg_s全要素が40.0<95=r_max*0.95のため飽和率は0)
    assert all(r['saturation_rate'] == 0.0 for r in rows if r['saturation_rate'] is not None)


# --- ⑨smooth_short_label_runs: 短い切り替わりの平滑化(直線vsコーナーPSD"データ不足"対策) ---

def test_smooth_short_label_runs_absorbs_brief_flip():
    """0.5s(閾値2.0s未満)だけ挟まるcornerラベルは、直前のstraightへ吸収される。"""
    series = [(0.0, 'straight'), (0.25, 'straight'), (0.5, 'straight'),
              (0.75, 'corner'), (1.0, 'corner'),  # 0.5s間だけcorner
              (1.25, 'straight'), (1.5, 'straight')]
    result = smooth_short_label_runs(series, min_run_s=2.0)
    labels = [lab for _, lab in result]
    assert labels == ['straight'] * 7  # 短いcorner区間が吸収され全てstraightになる


def test_smooth_short_label_runs_keeps_long_run():
    """min_run_s以上続く区間はそのまま残る(非冗長性: 過剰な平滑化をしない)。"""
    series = [(t, 'straight' if t < 5.0 else 'corner') for t in np.arange(0, 10, 0.25)]
    result = smooth_short_label_runs(series, min_run_s=2.0)
    labels = [lab for _, lab in result]
    assert 'corner' in labels  # 5秒続くcorner区間(閾値2.0s超)は吸収されず残る
    assert labels.count('corner') == labels.count('straight')  # 分割位置がそのまま維持される


def test_smooth_short_label_runs_first_run_untouched_when_too_short():
    """先頭の区間は「直前」が無いため吸収先が無く、そのまま残る(境界条件)。"""
    series = [(0.0, 'corner'), (0.25, 'straight'), (0.5, 'straight')]
    result = smooth_short_label_runs(series, min_run_s=2.0)
    assert result[0][1] == 'corner'  # 先頭の短い区間は吸収されず残る(仕様として許容)


def test_smooth_short_label_runs_empty_input():
    assert smooth_short_label_runs([], min_run_s=2.0) == []
