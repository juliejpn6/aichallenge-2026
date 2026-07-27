"""Unit tests for the gnss_covariance threshold recalibration (117節, 2026-07-19)。

背景: 単独走行ログ(run_perffix_20260719_110700)の実測で、このAWSIM環境の生GNSS共分散
(/sensing/gnss/pose_with_covariance)が全8479サンプルで定数10.0固定と判明した。旧設定
(good_threshold=0.1, moderate_threshold=0.5)ではこの値は常にpoor_value(100.0)に分類され、
2026-07-05にチューニングしたgood_value(0.05)が一度も発火していなかった。

自己申告値と無関係な独立2手法(直線あてはめ残差RMS=平均0.052m/847窓、周回再現性=11地点で
0.015〜0.200m)でGNSS実精度を検証したところ、自己申告が示唆するσ≈3.16m(=sqrt(10.0))の
約60〜180倍良い(実際はσ≈0.05m級)と確認できたため、既知の動作点(10.0)がgood_valueへ
分類されるようgood_thresholdを引き上げた(imu_gnss_poser.param.yaml)。

adjust_covariance()自体(imu_gnss_poser_node.cpp)のロジックは無変更(3段階しきい値判定式
そのまま)で、しきい値の「値」のみを変更した。

テスト方針: C++ノードはビルド・rclpy依存のため直接importできない。3段階判定ロジック自体は
単純なif-elif連鎖のため、同一の式を複製したミラー関数で数式的性質を検証する。C++ソース側の
判定式が今回変更していないことと、YAML側の新しいしきい値の配線を構造的に確認する。
"""
import os
import yaml

_PKG_DIR = os.path.join(os.path.dirname(__file__), "..")
_CPP_PATH = os.path.join(_PKG_DIR, "src", "imu_gnss_poser_node.cpp")
_YAML_PATH = os.path.join(_PKG_DIR, "config", "imu_gnss_poser.param.yaml")

with open(_CPP_PATH) as _f:
    _CPP_SRC = _f.read()
with open(_YAML_PATH) as _f:
    _CFG = yaml.safe_load(_f)["/**"]["ros__parameters"]["gnss_covariance"]


def _adjust_covariance(v, good_thresh, good_value, mod_thresh, mod_value, poor_value):
    """imu_gnss_poser_node.cpp adjust_covariance()内のadjustラムダ(272-278行目付近)の複製ミラー。"""
    if v <= good_thresh:
        return good_value
    if v <= mod_thresh:
        return mod_value
    return poor_value


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: 新しきい値での分類結果を数式的に検証
# ---------------------------------------------------------------------------

def test_known_operating_point_10_0_now_classified_as_good():
    """核心(117節の対処そのもの): このAWSIM環境で実測された既知の動作点raw=10.0が、
    再較正後はgood_value(YAML側の現在値。166節続報で0.05→0.01→0.03と改定済み)に
    分類されることを確認する。"""
    result = _adjust_covariance(
        10.0, _CFG["good_threshold"], _CFG["good_value"],
        _CFG["moderate_threshold"], _CFG["moderate_value"], _CFG["poor_value"])
    assert result == _CFG["good_value"]


def test_known_operating_point_10_0_was_poor_under_old_thresholds_regression():
    """回帰(対処前の状態の記録): 旧しきい値(good=0.1, moderate=0.5)では、raw=10.0は
    常にpoor_value(100.0)に分類されていたことを数値で再現する(対処の必要性の実証)。"""
    old_good_thresh, old_mod_thresh = 0.1, 0.5
    result = _adjust_covariance(10.0, old_good_thresh, 0.05, old_mod_thresh, 0.25, 100.0)
    assert result == 100.0


def test_much_worse_covariance_still_falls_to_poor():
    """回帰: 万一raw共分散が既知動作点(10.0)から大きく劣化した場合(例: 60.0)、
    再較正後のmoderate_threshold(50.0)を超えるため引き続きpoor_valueに分類される
    (階梯を保持し、常時good扱いにはならないことを確認)。"""
    result = _adjust_covariance(
        60.0, _CFG["good_threshold"], _CFG["good_value"],
        _CFG["moderate_threshold"], _CFG["moderate_value"], _CFG["poor_value"])
    assert result == _CFG["poor_value"]


def test_moderate_tier_still_reachable_for_intermediate_values():
    """回帰: good_threshold(12.0)とmoderate_threshold(50.0)の間の値(例: 20.0)は
    moderate_valueに分類され、3段階の階梯構造(good<moderate<poor)が維持されている
    ことを確認する(good_thresholdだけを上げてmoderate_thresholdを据え置くと、この階梯が
    構造的に到達不能になる誤りを防ぐ回帰テスト)。"""
    result = _adjust_covariance(
        20.0, _CFG["good_threshold"], _CFG["good_value"],
        _CFG["moderate_threshold"], _CFG["moderate_value"], _CFG["poor_value"])
    assert result == _CFG["moderate_value"]


def test_threshold_ordering_is_still_valid_good_lt_moderate():
    """非矛盾性: good_threshold < moderate_thresholdが保たれていることを確認する
    (満たされない場合、moderate段が到達不能なdead codeになる)。"""
    assert _CFG["good_threshold"] < _CFG["moderate_threshold"]


def test_real_hardware_low_covariance_still_classified_as_good_regression():
    """回帰: 実機等でGNSSが本来の意味で低分散(例: 0.05)を報告した場合も、
    引き上げ後のgood_threshold(12.0)以下のため引き続きgood_valueに分類される
    (再較正が「常にgoodにする」設計崩壊ではなく、既存の階梯を保った引き上げであることを確認)。"""
    result = _adjust_covariance(
        0.05, _CFG["good_threshold"], _CFG["good_value"],
        _CFG["moderate_threshold"], _CFG["moderate_value"], _CFG["poor_value"])
    assert result == _CFG["good_value"]


# ---------------------------------------------------------------------------
# 2) C++ソース側の判定ロジックが無変更であることを構造的に確認
# ---------------------------------------------------------------------------

def test_adjust_covariance_logic_unchanged_in_cpp_source():
    """非冗長性/非矛盾性: 117節の対処はYAMLのしきい値の値のみを変更しており、
    adjust_covariance()自体のif-elif分岐式には一切手を入れていないことを確認する。"""
    idx = _CPP_SRC.index("void adjust_covariance")
    snippet = _CPP_SRC[idx:idx + 700]
    assert "if (v <= gnss_cov_good_thresh_) return gnss_cov_good_;" in snippet
    assert "if (v <= gnss_cov_mod_thresh_) return gnss_cov_mod_;" in snippet
    assert "return gnss_cov_poor_;" in snippet


def test_yaml_good_threshold_raised_above_old_value():
    assert _CFG["good_threshold"] > 0.1


def test_yaml_good_value_is_tighter_than_original_0705_tuning():
    """回帰: good_value自体は117節(しきい値再較正)では変更していなかったが、166節続報の
    決定実験(0.05→0.01、独立実測の真の精度σ≈0.052m/分散0.0027への接近)以降は意図的に
    改定され続けている(現在0.03、0.01と0.05の中間を検証中)。ここでは「117節当時のまま
    緩んでいない(0.05より小さいまま)」ことのみを確認する(具体値は今後も変わり得るため
    固定値をアサートしない)。"""
    assert _CFG["good_value"] < 0.05


def test_yaml_poor_value_unchanged():
    """回帰: poor_value(100.0)・moderate_value(0.25)は本節の対象外で無変更。"""
    assert _CFG["poor_value"] == 100.0
    assert _CFG["moderate_value"] == 0.25
