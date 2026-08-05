"""候補④(追い越し地点推定によるENGAGE遅延、最短距離オーバーテイク)の実装
(2026-08-05、task#293、design_docs predictive_control_overtake_development_plan_20260805.md
6-9節参照)。

背景: ユーザー提案「自分の車速と相手の車速を意識した追い越しアルゴリズムを設計」
「できるだけ最短距離でオーバーテイクを遂行する」「相手を追い越すコーナーを推定し、
そこまでは既定経路を走行する」。外部AI(Gemini・別Claudeインスタンス)への相談を
経て、「ENGAGE自体(_close_enough判定)を遅らせる」設計(icc_stopには一切触れない)
で確定した。Phase 0のオフライン実測(scripts/analyze_engage_vs_icc_ordering.py)で
margin=1.0〜1.5秒が実用バランス点と判明、1.5を既定値とした。

新関数_predicted_catch_up_time(既存_predicted_time_to_wpの2体問題版、相手が
走行中の場合にも対応)を追加し、_evaluate_engage_readiness内の_close_enough計算に
config gate(既定OFF)付きで組み込んだ。既存の停止相手向け分岐は無変更。

mpc_controller.pyはrclpy依存で直接importできないため、①ロジックをミラー実装した
数値検証、②ソーステキスト構造検証、の両方を組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _predicted_catch_up_time_mirror(segments, ref_v_kmh, opp_ds_now, opp_v, pass_clear,
                                     max_dist, kmh_to_mps=lambda k: k / 3.6):
    """_predicted_catch_up_timeのミラー実装。segmentsは弧長のリスト(各区間長)、
    ref_v_kmhは対応する区間ごとの自車計画速度[km/h]のリスト。"""
    gap = opp_ds_now - pass_clear
    if gap <= 0.0:
        return 0.0
    t = 0.0
    total_d = 0.0
    for seg, v_kmh in zip(segments, ref_v_kmh):
        v_ego = max(kmh_to_mps(v_kmh), 0.1)
        seg_time = seg / v_ego
        v_rel = v_ego - opp_v
        if v_rel > 1e-6 and gap <= v_rel * seg_time:
            return t + gap / v_rel
        gap -= v_rel * seg_time
        t += seg_time
        total_d += seg
        if total_d > max_dist:
            return None
    return None


# ---------------------------------------------------------------------------
# ①数値検証: ミラー実装で閉形式の追いつき計算が正しいことを確認
# ---------------------------------------------------------------------------

def test_single_segment_catch_up_closed_form():
    """単一区間・一定速度: 相手が自車より5m/s遅く、ギャップ20m(pass_clear込みで
    17m)なら、17m / 5m/s = 3.4秒で追いつく(高校物理の相対速度そのもの)。"""
    # v_ego=36km/h=10m/s、opp_v=5m/s、v_rel=5m/s
    t = _predicted_catch_up_time_mirror(
        segments=[1000.0], ref_v_kmh=[36.0], opp_ds_now=20.0, opp_v=5.0,
        pass_clear=3.0, max_dist=1000.0)
    assert t == 3.4


def test_already_within_pass_clear_returns_zero():
    """既に抜き切りクリアランス以内(ギャップ<=pass_clear)なら即座に0.0を返す。"""
    t = _predicted_catch_up_time_mirror(
        segments=[1000.0], ref_v_kmh=[36.0], opp_ds_now=2.0, opp_v=5.0,
        pass_clear=3.0, max_dist=1000.0)
    assert t == 0.0


def test_negative_relative_speed_never_catches_up():
    """自車の方が遅い(v_rel<=0)場合、ギャップは詰まらず(むしろ開き)、
    max_dist超過でNoneを返す(_engage_dist_dynamicへのフォールバックを促す)。"""
    t = _predicted_catch_up_time_mirror(
        segments=[10.0] * 200, ref_v_kmh=[18.0] * 200,  # v_ego=5m/s
        opp_ds_now=20.0, opp_v=8.0,  # opp_vの方が速い
        pass_clear=3.0, max_dist=100.0)
    assert t is None


def test_multi_segment_catch_up_crosses_boundary():
    """2区間にまたがるケース: 1区間目では追いつかず、2区間目で閉形式に解ける
    ことを確認する(区間境界をまたぐ走査ロジックの検証)。"""
    # 区間1: 5m, v_ego=10m/s, v_rel=10-6=4m/s, gap開始=20-3=17
    #   区間1通過時間=0.5s、この間にgapが4*0.5=2詰まる → gap=15 (17>4*0.5なので未到達)
    # 区間2: 100m, v_ego=10m/s, v_rel=4m/s, 残りgap=15
    #   15 <= 4*(100/10)=40 なので区間2内で解ける: t = 0.5 + 15/4 = 4.25
    t = _predicted_catch_up_time_mirror(
        segments=[5.0, 100.0], ref_v_kmh=[36.0, 36.0], opp_ds_now=20.0, opp_v=6.0,
        pass_clear=3.0, max_dist=200.0)
    assert t == 4.25


def test_max_dist_exceeded_returns_none():
    """相手に永遠に追いつかない訳ではないが、max_dist(既存fwd_max_considerと
    同じ走査上限)を超える場合はNoneを返す(安全側フォールバック)。"""
    t = _predicted_catch_up_time_mirror(
        segments=[10.0] * 3, ref_v_kmh=[36.0] * 3, opp_ds_now=1000.0, opp_v=6.0,
        pass_clear=3.0, max_dist=25.0)
    assert t is None


# ---------------------------------------------------------------------------
# ②ソーステキスト構造検証: 実装が設計通りの配線になっていること
# ---------------------------------------------------------------------------

def test_predicted_catch_up_time_function_exists():
    assert "def _predicted_catch_up_time(self, from_wp: int, opp_ds_now: float," in _SRC


def test_predicted_catch_up_time_falls_back_to_none_when_no_ref_vel_configulator():
    idx = _SRC.index("def _predicted_catch_up_time(")
    idx_end = _SRC.index("def _switchback_curvature_veto(", idx)
    snippet = _SRC[idx:idx_end]
    assert "if self._ref_vel_configulator is None:" in snippet
    assert "return None" in snippet


def test_close_enough_gated_by_catchup_predict_enable_flag():
    """相手走行中の新分岐がconfig gate(既定OFF)で保護されていること。"""
    idx = _SRC.index("_t_reach_profile = None")
    idx_end = _SRC.index("if _t_reach_profile is not None:", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._ot_catchup_predict_enable" in snippet
    assert "not _is_stopped_for_profile" in snippet
    assert "self._predicted_catch_up_time(" in snippet
    assert "self._ot_catchup_engage_margin_s" in snippet


def test_stopped_opponent_branch_unchanged():
    """既存の停止相手向け分岐(_predicted_time_to_wp呼び出し)が無変更のまま
    残っていることを確認する(退行防止)。"""
    idx = _SRC.index("_t_reach_profile = None")
    idx_end = _SRC.index("if _t_reach_profile is not None:", idx)
    snippet = _SRC[idx:idx_end]
    assert "if _is_stopped_for_profile and scan.get(\"fwd_wp\") is not None:" in snippet
    assert "self._predicted_time_to_wp(" in snippet


def test_t_reach_margin_defaults_to_zero_for_stopped_branch():
    """停止相手向け分岐では_t_reach_marginが0.0のまま(=既存の閾値式と
    ビット等価)であることを確認する。"""
    idx = _SRC.index("_t_reach_profile = None")
    idx_end = _SRC.index("if _t_reach_profile is not None:", idx)
    snippet = _SRC[idx:idx_end]
    assert "_t_reach_margin = 0.0" in snippet


def test_t_reach_thr_formula_includes_margin():
    idx = _SRC.index("if _t_reach_profile is not None:")
    idx_end = _SRC.index("else:", idx)
    snippet = _SRC[idx:idx_end]
    assert "_t_reach_thr = (self._ot_t_lateral + _t_reach_margin" in snippet
    assert "self._ot_pass_clear / _closing_est)" in snippet


def test_config_keys_exist_with_safe_defaults():
    assert 'self._ot_catchup_predict_enable = bool(\n                _otget("catchup_predict_enable", False))' in _SRC
    assert 'self._ot_catchup_engage_margin_s = float(\n                _otget("catchup_engage_margin_s", 1.5))' in _SRC
