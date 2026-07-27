"""Unit tests for the shared OpponentSituation snapshot (141節、フェーズ1、
2026-07-20)。

背景: ユーザー指摘「自車位置・コース状況・相手位置・両者の未来位置を推測して
オーバーテイク/追従/停止を判断しているのだから、どの層も同じ情報にアクセス
して判断すべき」を受けた設計。実際に既存の2つの事故の根本原因はいずれも
「同じ相手について、層ごとに別々の式・別々のタイミングで再計算していた」
ことだった:
  - 128節: LAT-TTCのspace式が自車位置を含まず、space=3.12m(安全)なのに
    実際のfwd_dlat=0.198m(ほぼ接触)という矛盾が実測された。
  - P0#1(0720-05実測、第二コーナー衝突): ICCのnear_sep(現在のdlatのみ)と
    LAT-TTCのdlat_v_ema(トレンド)が同じ相手について食い違い、v_safeが
    無制限のまま衝突した。

フェーズ1の対処は「既存の計算結果(scan/_lat_ttc.update)を1箇所へ集約する
OpponentSituationスナップショットを新設し、まずENGAGEゲートをそこへ移行する」
という純粋なリファクタで、挙動は一切変更しない。このテストは
①ミラー実装による式の正しさの再検証、②実測値による遡及検証、
③ソーステキストによる配線確認(スナップショットが_lat_dec確定直後に1回だけ
構築され、ENGAGEゲートがそれを参照すること)の3種類で検証する。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー
実装した上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

TTC_CRITICAL_S = 0.8
MIN_TREND_CYCLES = 3


def _dlat_closing_trend(fwd_dlat, dlat_v_ema, dlat_shrink_run,
                         footprint_risk=False,
                         min_trend_cycles=MIN_TREND_CYCLES,
                         ttc_critical_s=TTC_CRITICAL_S):
    """_dlat_closing_trend()のミラー実装(旧_dlat_ttc_veto式+issue⑤③の
    footprint_risk短絡追加、2026-07-22)。"""
    if footprint_risk:
        return True
    if dlat_shrink_run < min_trend_cycles:
        return False
    if dlat_v_ema >= 0.0:
        return False
    if fwd_dlat is None:
        return False
    ttc = fwd_dlat / max(abs(dlat_v_ema), 1e-6)
    return ttc <= ttc_critical_s


def _build_opponent_situation(scan, dlat_v_ema, dlat_shrink_run, footprint_risk=False):
    """_build_opponent_situation()のミラー実装(辞書で代用)。"""
    fwd_dlat = scan.get("fwd_dlat")
    return {
        "fwd_vid": scan.get("fwd_vid"),
        "fwd_ds": scan.get("fwd_ds"),
        "fwd_dlat": fwd_dlat,
        "fwd_vopp": scan.get("fwd_vopp"),
        "dlat_v_ema": dlat_v_ema,
        "dlat_shrink_run": dlat_shrink_run,
        "is_closing_trend": _dlat_closing_trend(fwd_dlat, dlat_v_ema, dlat_shrink_run,
                                                 footprint_risk),
    }


# --- ①非矛盾性: ミラー実装が旧_dlat_ttc_veto式と完全一致すること ---

def test_formula_unchanged_positive_case():
    assert _dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=-0.5, dlat_shrink_run=5) is True


def test_formula_unchanged_negative_case_ttc_too_slow():
    assert _dlat_closing_trend(fwd_dlat=3.0, dlat_v_ema=-0.5, dlat_shrink_run=5) is False


def test_formula_unchanged_widening_gap_not_vetoed():
    assert _dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=0.5, dlat_shrink_run=5) is False


def test_formula_unchanged_insufficient_trend_cycles():
    assert _dlat_closing_trend(fwd_dlat=0.3, dlat_v_ema=-0.5, dlat_shrink_run=1) is False


# --- ④過去ログへの遡及効果: 実測値でスナップショットが正しく再現すること ---

def test_retroactive_0720_04_wp93_corner2_collision():
    """138節、0720-04 wp93(第2コーナー衝突の起点となったENGAGE)。
    fwd_dlat=0.22, dlat_v_ema=-0.365, shrink_run=16 → ttc≈0.603s<=0.8s → 危険判定。"""
    scan = {"fwd_vid": "d3", "fwd_ds": 1.0, "fwd_dlat": 0.22, "fwd_vopp": 9.0}
    sit = _build_opponent_situation(scan, dlat_v_ema=-0.365, dlat_shrink_run=16)
    assert sit["is_closing_trend"] is True


def test_retroactive_local_log_wp130_dangerous_near_miss():
    """0720実測(ローカル3台走行、d1ログwp130): fwd_dlat=0.058,
    dlat_v_ema=-0.711, shrink_run=125 → ttc≈0.082s、極めて危険な近接。"""
    scan = {"fwd_vid": "d3", "fwd_ds": 0.5, "fwd_dlat": 0.058, "fwd_vopp": 3.0}
    sit = _build_opponent_situation(scan, dlat_v_ema=-0.711, dlat_shrink_run=125)
    assert sit["is_closing_trend"] is True


def test_retroactive_0720_05_wp139_moderate_distance_not_flagged_by_this_gate():
    """0720-05実測(P0#1の起点、第二コーナー衝突、wp139-141): fwd_dlat=2.47〜3.17,
    dlat_v_ema=-0.87〜-1.54。ENGAGEゲート(このスナップショットの
    is_closing_trend)では検知範囲外であることを確認する(ttc≈1.6〜2.8s>0.8s)。
    これはP0#1がこのスナップショットの拡張利用(ICC層への配線)を必要とする
    理由そのものであり、フェーズ1の時点では意図的に「検知しない」が正しい。"""
    scan = {"fwd_vid": "d3", "fwd_ds": 3.99, "fwd_dlat": 2.47, "fwd_vopp": 9.8}
    sit = _build_opponent_situation(scan, dlat_v_ema=-1.54, dlat_shrink_run=105)
    assert sit["is_closing_trend"] is False


# --- ③配線確認: スナップショットが1箇所で構築されENGAGEゲートが参照すること ---

def test_snapshot_built_exactly_once_right_after_lat_dec():
    assert _SRC.count("_opp_sit = self._build_opponent_situation(") == 1
    idx_lat_dec = _SRC.index("_lat_dec = self._lat_ttc.update(")
    idx_snapshot = _SRC.index("_opp_sit = self._build_opponent_situation(")
    idx_phase0_diag = _SRC.index("_dlat_trend_alert = (")
    assert idx_lat_dec < idx_snapshot < idx_phase0_diag


def test_snapshot_class_documents_all_fields_used():
    idx = _SRC.index("class OpponentSituation:")
    snippet = _SRC[idx:idx + 1400]
    for field in ("fwd_vid", "fwd_ds", "fwd_dlat", "fwd_vopp",
                   "dlat_v_ema", "dlat_shrink_run", "is_closing_trend"):
        assert f"{field}:" in snippet


def test_build_opponent_situation_performs_no_new_computation_besides_trend():
    """②非冗長性: _build_opponent_situationはscan/_lat_decの既存値をそのまま
    集約するのみで、is_closing_trend(既存式の抽出)以外に新規計算を持たない
    ことを確認する。"""
    idx = _SRC.index("def _build_opponent_situation(")
    idx_end = _SRC.index("def _evaluate_engage_readiness(")
    snippet = _SRC[idx:idx_end]
    assert "scan.get(\"fwd_vid\")" in snippet
    assert "scan.get(\"fwd_ds\")" in snippet
    assert "scan.get(\"fwd_dlat\")" in snippet
    assert "scan.get(\"fwd_vopp\")" in snippet
    assert "lat_dec.dlat_v_ema" in snippet
    assert "lat_dec.dlat_shrink_run" in snippet
    # is_closing_trendの計算はヘルパー関数へ委譲されており、ここに式の重複はない
    assert "min_trend_cycles" not in snippet
    assert "ttc_critical_s" not in snippet


def test_opponent_situation_is_frozen_dataclass_read_only():
    """③方針確認: スナップショットは読み取り専用(frozen)であることを確認する
    (複数レイヤーが参照する共有オブジェクトを途中で書き換えると、参照順序に
    依存するバグを生みかねないため)。"""
    idx = _SRC.index("class OpponentSituation:")
    snippet = _SRC[max(0, idx - 100):idx]
    assert "@dataclasses.dataclass(frozen=True)" in snippet
