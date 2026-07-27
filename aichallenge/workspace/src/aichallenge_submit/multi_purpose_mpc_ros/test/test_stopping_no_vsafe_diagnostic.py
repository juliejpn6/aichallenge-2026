"""Structural tests for [STOPPING-NO-VSAFE] diagnostic (109節続報, 2026-07-18).

背景: ローカル3台走行実測(output/20260718-172517、t≈426.35秒、wp272→277)で、
STOPPING状態にもかかわらずicc_stop(_vlim)・icc_stop_fallbackのいずれも成立せず
v_safe_pre=Noneのままu0が全開(4.17)になる瞬間を発見した。この瞬間、fwd_dlat=
2.76〜3.47m(near_sep=1.8・engage_lat_max=2.0のいずれも超過)の相手がH2/0714-04
設計により正しく「進路外」として除外されていたが、直後にRfree≈0・
[COLLISION-SUSPECTED](v drop 3.99→3.08)が発生していた。

除外された相手より近い/危険な別の相手が_scan["cars"](既存の全前方車リスト、
_vlim/fallbackが検討する「best」1台より広い)に含まれていたかどうかを次回ログで
確認するため、診断専用(挙動へ影響なし)のログを追加した。推測せず計装で実測する
というStage1.5方針に従い、本節では原因の特定・修正は行わない。

2026-07-20追記(131-6節⑤「なめらかな断念」、136節続報): その後の実測(0720-02
wp13、t=609.34)で、OVERTAKING中の速度モデル(eff_v_cap)からSTOPPINGの速度モデル
(icc_stop/fallback)への切替の瞬間に両方とも不成立になると、v_safe_pre=Noneの
ままMPC自身の最適化が無制限速度(u0=v_max)を出力することを直接確認した(OTログ:
v_safe_src=Noneなのにu0=4.1667)。0.6秒後にwall_slowが追いついた時点で壁マージン
は既にマイナス(wall=-0.42)まで悪化していた。診断専用だった本ブロックへ、
footprint_risk/wall_slowが既に再利用しているwall_slow_speedをブリッジ用の
保守速度として追加した(新規パラメータ0個)。以下のテストのうち
test_stopping_no_vsafe_is_diagnostic_only_no_v_safe_assignmentは、この意図的な
挙動変更を反映して更新済み。

mpc_controller.pyはrclpy/autoware型をモジュールスコープでimportするため直接
importできない。ロジック自体はif文1つ(v_safe_preがNoneかどうか)のみで
自己完結しているため、ソーステキストの構造的検証のみで配線を確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_stopping_no_vsafe_state_initialized():
    assert "self._stopping_no_vsafe_prev = False" in _SRC


def test_stopping_no_vsafe_fires_after_both_icc_paths_determined():
    """診断ログが、icc_stop(_vlim)・icc_stop_fallbackの両方の判定が
    完了した後に位置していることを確認する(除外理由の記録が目的のため、
    どちらの経路も評価済みである必要がある)。"""
    idx_vlim = _SRC.index("_v_safe_pre = _vlim")
    idx_fallback = _SRC.index('_v_safe_cand.append(("icc_stop_fallback')
    idx_diag = _SRC.index("[STOPPING-NO-VSAFE] ON")
    assert idx_vlim < idx_diag
    assert idx_fallback < idx_diag


def test_stopping_no_vsafe_now_bridges_with_wall_slow_speed():
    """2026-07-20更新(131-6節⑤、136節続報): 本ブロックは診断専用ではなくなり、
    v_safe_pre=Noneが続く間、既存wall_slow_speedをブリッジ用の保守速度として
    _v_safe_pre/_v_safe_candへ代入するようになったことを確認する(意図的な
    挙動変更、旧テストtest_stopping_no_vsafe_is_diagnostic_only_no_v_safe_assignment
    を置き換え)。新規パラメータ0個(既存self._wall_slow_speedの再利用)。"""
    idx = _SRC.index("if _v_safe_pre is None:")
    idx_end = _SRC.index("elif self._stopping_no_vsafe_prev:")
    snippet = _SRC[idx:idx_end]
    assert "_v_safe_pre = self._wall_slow_speed" in snippet
    assert '_v_safe_cand.append(("stopping_no_vsafe' in snippet


def test_stopping_no_vsafe_reuses_existing_cars_list_no_new_scan():
    """②非冗長性: 新規スキャン処理を追加せず、既存の_scan["cars"]
    (_scan_trafficが毎周期計算済みの全前方車リスト)をそのまま再利用する。"""
    idx = _SRC.index("[STOPPING-NO-VSAFE] ON")
    snippet = _SRC[idx - 400:idx + 600]
    assert '_scan["cars"]' in snippet


def test_stopping_no_vsafe_reuses_existing_thresholds_no_new_parameter():
    """②非冗長性: near_sep/engage_lat_maxとも既存定数(self._fwd_min_lat_sep・
    self._ot_engage_lat_max)の再利用であり、新規パラメータは0個。"""
    idx = _SRC.index("[STOPPING-NO-VSAFE] ON")
    snippet = _SRC[idx - 200:idx + 600]
    assert "self._fwd_min_lat_sep" in snippet
    assert "self._ot_engage_lat_max" in snippet


def test_stopping_no_vsafe_edge_triggered_on_and_off():
    idx = _SRC.index("[STOPPING-NO-VSAFE] ON")
    # 2026-07-20更新: ブリッジ処理のコメント・代入文が追加され窓を広げる必要が生じた。
    snippet = _SRC[idx - 300:idx + 2700]
    assert "[STOPPING-NO-VSAFE] OFF" in snippet
    assert "self._stopping_no_vsafe_prev = True" in snippet
    assert "self._stopping_no_vsafe_prev = False" in snippet
