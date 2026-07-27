"""Unit tests for _footprint_risk (127節続報, A-2候補, 2026-07-20).

Background: 0720-01予選ログのwp173分析で、LAT-TTCが参照するspace/opp_space
(_scan_traffic内のlf/rf、壁〜相手の隙間の広さ)が自車の現在位置を式に含まないため、
fwd_dlat(自車〜相手の実測横間隔)が0.198mまで縮んでいるのに同一周期でspace=3.12m
(「安全」)を報告する矛盾を実測で確認した(126/127節)。また縦間隔(fwd_ds)には
対応する「車体全長ベースの物理下限」判定が一度も存在しなかった。

ユーザーから提供された公式車両仕様(全長200cm/全幅145cm)より、同型カート2台なら
「両車の半幅合計=1台分の全幅」「両車の半長合計=1台分の全長」となる関係を用い、
fwd_dlat<along_min_width(既存1.45m、再利用)かつfwd_ds<along_min_length
(新規2.00m)を「実際に車体が重なるリスクがある」状態(_footprint_risk)と定義する。
self._mpc.model.length(1.087)はホイールベース(自転車モデルの運動学パラメータ)で
あり車両全長ではないため、当たり判定には流用しない。

mpc_controller.pyはautoware_auto_control_msgs等のROSメッセージ型をモジュール
スコープでimportしており単体テスト環境では直接importできないため、ソーステキストの
構造的検証で配線を確認する(test_switchback_wall_veto.py等、既存の同種テストと
同じ制約)。_footprint_risk自体の判定ロジック(AND条件)はミラー関数で検証する。
"""
import os

import pytest


def _footprint_risk_mirror(fwd_dlat, fwd_ds, along_min_width, along_min_length):
    """mpc_controller.py内の_footprint_risk計算式(3189行目付近)の複製ミラー。"""
    return (fwd_dlat is not None and fwd_ds is not None
            and fwd_dlat < along_min_width
            and abs(fwd_ds) < along_min_length)


ALONG_MIN_WIDTH = 1.45
ALONG_MIN_LENGTH = 2.00


def test_both_lat_and_lon_within_threshold_is_risk():
    """本修正の中核: fwd_dlat<along_min_widthかつfwd_ds<along_min_lengthの
    両方が成立する場合のみTrue。"""
    assert _footprint_risk_mirror(0.198, 0.5, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is True


def test_retroactive_0720_01_wp173_incident_now_detected():
    """遡及検証(0720-01予選ログwp173実測): fwd_dlat=0.198m(space=3.12mと矛盾していた
    実測値)は、along_min_width=1.45m未満のため、fwd_dsが2.00m未満であれば
    footprint_riskとして検知できる。"""
    assert _footprint_risk_mirror(0.1982132638959605, 1.0,
                                   ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is True


def test_wide_lateral_gap_is_not_risk_even_if_longitudinally_close():
    """回帰: 横間隔が十分広ければ(along_min_width以上)、縦間隔がどれだけ近くても
    リスクとしない(横方向にすれ違えているため)。"""
    assert _footprint_risk_mirror(2.0, 0.1, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False


def test_wide_longitudinal_gap_is_not_risk_even_if_laterally_close():
    """回帰: 縦間隔が十分離れていれば(along_min_length以上)、横間隔がどれだけ
    近くてもリスクとしない(まだ縦に遠く並走していないため)。"""
    assert _footprint_risk_mirror(0.1, 5.0, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False


def test_boundary_exactly_at_thresholds_is_not_risk():
    """境界値: ちょうど閾値と等しい場合は抑制しない(<のみ、along_min_widthの
    既存運用と同じ規約)。"""
    assert _footprint_risk_mirror(ALONG_MIN_WIDTH, 1.0,
                                   ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False
    assert _footprint_risk_mirror(0.5, ALONG_MIN_LENGTH,
                                   ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False


def test_negative_fwd_ds_uses_absolute_value():
    """回帰: fwd_dsは自車より僅かに後方(-model.length〜0)になり得るため、絶対値で
    比較する(_scan_traffic の cars 判定窓 -model.length < ds と同じ考え方)。"""
    assert _footprint_risk_mirror(0.3, -0.5, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is True


def test_none_inputs_fail_open():
    """回帰: fwd_dlat/fwd_dsが未取得(None、対象車なし)の場合は安全側
    (リスクなし=fail-open)にフォールバックする。"""
    assert _footprint_risk_mirror(None, 1.0, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False
    assert _footprint_risk_mirror(0.3, None, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False
    assert _footprint_risk_mirror(None, None, ALONG_MIN_WIDTH, ALONG_MIN_LENGTH) is False


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_along_min_length_loaded_and_reuses_wall_slow_speed_no_new_speed_constant():
    """②非冗長性: along_min_length(新規パラメータ1個)のみ追加し、v_safe候補の
    速度キャップにはwall_slow_speed(既存)を再利用する(新規速度定数0個)。"""
    assert 'self._along_min_length = float(_otget("along_min_length", 2.00))' in _SRC
    idx = _SRC.index("footprint_risk(車体重なりリスク)")
    snippet = _SRC[max(0, idx - 300):idx + 100]
    assert "self._wall_slow_speed" in snippet


def test_footprint_risk_computed_once_and_reused_no_duplicate_calculation():
    """②非冗長性: _footprint_riskの代入は1箇所のみ(スキャン直後に一度だけ
    計算し、LAT-TTC呼び出しとv_safe候補集約の両方で使い回す)。2026-07-22修正
    (issue⑤②): _fp_near_zone(footprint_risk本体+154節taperの危険域全体)も
    同じ場所で1回だけ計算し、_footprint_risk自体をその部分集合として定義する
    ようになったため、両方の代入が1箇所ずつであることを確認する。"""
    assert _SRC.count("_footprint_risk = ") == 1
    assert _SRC.count("_fp_near_zone = (") == 1


def test_footprint_risk_computed_before_lat_ttc_update_call():
    """新配線: _footprint_riskの計算はLateralTTCMonitor.update()呼び出しより前に
    実行され、その結果がupdate()の引数として渡される(出現順で確認)。"""
    idx_calc = _SRC.index("_fp_near_zone = (")
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_calc < idx_update


def test_footprint_risk_passed_to_lat_ttc_update_call():
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx_update:idx_update + 900]
    assert "footprint_risk=_footprint_risk" in snippet


def test_footprint_risk_added_to_v_safe_candidates():
    """新配線: _footprint_riskがTrueの場合、v_safe候補スタックへ
    footprint_risk(車体重なりリスク)ラベルで追加される(wall_slowと同じ層、
    state/branchに依存しない)。"""
    idx = _SRC.index('_v_safe_cand.append(("footprint_risk(車体重なりリスク)"')
    snippet = _SRC[max(0, idx - 300):idx]
    assert "if _footprint_risk:" in snippet


def test_footprint_risk_logged_in_giveup_trigger_line():
    idx = _SRC.index('f"[LAT-TTC-ACT] giveup trigger=')
    snippet = _SRC[idx:idx + 600]
    assert "footprint_risk={_lat_dec.footprint_risk_triggered}" in snippet
