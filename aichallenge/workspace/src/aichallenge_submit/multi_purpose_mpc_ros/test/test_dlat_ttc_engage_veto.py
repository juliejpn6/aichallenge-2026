"""Unit tests for the dlat-trend TTC ENGAGE gate (131-6節①、Gap①Phase1,
2026-07-20).

Background: 0720-04予選ログ実測(wp93、第2コーナー)で、ENGAGE時点で
fwd_dlat=0.22m・dlat_v_ema=-0.365m/s・dlat_shrink_run=16と既に急速に縮小中
だった対象車へ仕掛け、0.22秒後のカーブ起因switchback→さらに0.22秒後に
footprint_riskで強制giveup→衝突、という一連の流れを実測で確認した。

131節の第1案(fwd_dlatの絶対値のみで拒否)は「相手が正面にいる、ごく普通の
engage直前」の正常ケースまで誤ってブロックし撤回した。今回はPhase0(133節)
が既に毎周期計算しているdlat_v_ema/dlat_shrink_run(トレンド)を使い、
「このまま縮み続けたら何秒で接触するか」というTTC概念を、LAT-TTCが既に
switchback/C2判定で使っているttc_critical_s(0.8秒)・min_trend_cycles(3周期)
へそのまま当てはめる。新規パラメータ0個。

このゲートはmpc_controller.pyの_control()内部(rclpy依存で直接importできない
巨大メソッド)にあるため、ロジックをミラー実装した上で、①実測値を使った
遡及検証、②境界値・fail-open系の検証、③ソーステキストによる配線確認、の
3種類で検証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

TTC_CRITICAL_S = 0.8   # 既存LateralTTCMonitorの既定値(ttc_critical_s)
MIN_TREND_CYCLES = 3   # 既存LateralTTCMonitorの既定値(min_trend_cycles)


def _dlat_ttc_veto(dlat_shrink_run, dlat_v_ema, fwd_dlat,
                    min_trend_cycles=MIN_TREND_CYCLES, ttc_critical_s=TTC_CRITICAL_S):
    """mpc_controller.pyの_dlat_ttc_veto計算のミラー実装。"""
    if dlat_shrink_run < min_trend_cycles:
        return False
    if dlat_v_ema >= 0.0:
        return False
    if fwd_dlat is None:
        return False
    return (fwd_dlat / max(abs(dlat_v_ema), 1e-6)) <= ttc_critical_s


# --- 遡及検証: 実測値 ---

def test_retroactive_0720_04_wp93_second_corner_collision_now_vetoed():
    """0720-04実測wp93(第2コーナー衝突の起点): fwd_dlat=0.22, dlat_v_ema=-0.365,
    shrink_run=16 は、ttc=0.22/0.365≈0.60秒<=0.8秒のため、修正後はvetoされる
    ことを確認する。"""
    assert _dlat_ttc_veto(dlat_shrink_run=16, dlat_v_ema=-0.365, fwd_dlat=0.219) is True


def test_retroactive_0720_04_wp59_engage_still_allowed():
    """遡及検証(退行なし): 0720-04実測wp59はshrink_run=121と大きいが
    dlat_v_ema=-1.398と急速接近中、fwd_dlat推定2.0m前後(実測OK)であれば
    ttc=2.0/1.398≈1.43秒>0.8秒でvetoされないことを確認する
    (「長時間・遠距離でのゆっくりした収束」を「近距離での急接近」と
    誤判定しない設計であることの確認)。"""
    assert _dlat_ttc_veto(dlat_shrink_run=121, dlat_v_ema=-1.398, fwd_dlat=2.0) is False


def test_retroactive_0720_02_wp264_fresh_engage_not_vetoed():
    """遡及検証(131節当時の教訓の再確認): fwd_dlat=0.275(相手がほぼ正面)でも
    dlat_v_ema>=0(縮小トレンドが確立していない、fresh engage)であれば
    vetoされないことを確認する(131節第1案が誤ってブロックしていた
    正常ケースが、今回の設計では正しく通過することの確認)。"""
    assert _dlat_ttc_veto(dlat_shrink_run=0, dlat_v_ema=0.0, fwd_dlat=0.275) is False
    assert _dlat_ttc_veto(dlat_shrink_run=0, dlat_v_ema=0.176, fwd_dlat=0.275) is False


# --- 境界値・fail-open ---

def test_shrink_run_below_min_trend_cycles_does_not_veto():
    """min_trend_cycles未満(単発の縮小)ではvetoしない(既存デバウンス方針の再利用)。"""
    assert _dlat_ttc_veto(dlat_shrink_run=2, dlat_v_ema=-5.0, fwd_dlat=0.01) is False


def test_shrink_run_exactly_at_min_trend_cycles_boundary():
    assert _dlat_ttc_veto(dlat_shrink_run=3, dlat_v_ema=-1.0, fwd_dlat=0.5) is True  # ttc=0.5<=0.8
    assert _dlat_ttc_veto(dlat_shrink_run=3, dlat_v_ema=-1.0, fwd_dlat=1.0) is False  # ttc=1.0>0.8


def test_ttc_exactly_at_critical_threshold_boundary():
    assert _dlat_ttc_veto(dlat_shrink_run=10, dlat_v_ema=-1.0, fwd_dlat=0.8) is True  # ttc==0.8
    assert _dlat_ttc_veto(dlat_shrink_run=10, dlat_v_ema=-1.0, fwd_dlat=0.8001) is False


def test_growing_dlat_never_vetoes_regardless_of_shrink_run():
    """dlat_v_ema>=0(縮小していない)なら、shrink_runが大きくてもvetoしない
    (shrink_runは過去のトレンドの残存値であり得るため、現在の符号を優先する)。"""
    assert _dlat_ttc_veto(dlat_shrink_run=50, dlat_v_ema=0.001, fwd_dlat=0.01) is False


def test_fwd_dlat_none_fails_open():
    assert _dlat_ttc_veto(dlat_shrink_run=10, dlat_v_ema=-2.0, fwd_dlat=None) is False


def test_far_but_shrinking_target_not_vetoed_when_ttc_generous():
    """遠距離(fwd_dlat大)なら縮小中でもTTCに余裕がありvetoされない
    (絶対距離ではなくTTCで判断する設計であることの確認)。"""
    assert _dlat_ttc_veto(dlat_shrink_run=10, dlat_v_ema=-0.5, fwd_dlat=5.0) is False


# --- 配線・②非冗長性の構造検証 ---

def test_veto_reuses_existing_lat_ttc_thresholds_no_new_parameter():
    """②非冗長性: 新規閾値を追加せず、既存self._lat_ttc.min_trend_cycles/
    ttc_critical_sを再利用していることを確認する。
    2026-07-20修正(141節、フェーズ1): 判定式は_dlat_closing_trend()へ抽出され、
    ENGAGEゲート側は共有スナップショット(opp_sit.is_closing_trend)を参照する
    形に変わった。式そのものの置き場所を追って検証する。"""
    idx = _SRC.index("def _dlat_closing_trend(")
    snippet = _SRC[idx:idx + 1700]
    assert "self._lat_ttc.min_trend_cycles" in snippet
    assert "self._lat_ttc.ttc_critical_s" in snippet


def test_engage_gate_reads_from_shared_opponent_situation():
    """141節フェーズ1: ENGAGEゲートは判定式を自前で再計算せず、_control()内で
    1回だけ構築された共有スナップショット(opp_sit.is_closing_trend)を
    参照することを確認する(同じ相手について層ごとに別々に再計算しない、
    というユーザー指摘への対処)。
    2026-07-21修正(148節): 判定式自体は_evaluate_engage_readiness()へ抽出され
    メソッド定義順(_build_opponent_situationより前)に置かれたため、文字位置の
    前後関係ではなく_control()の実行順(_build_opponent_situation呼び出し→
    _evaluate_engage_readiness呼び出し)で検証する。"""
    idx = _SRC.index("_dlat_ttc_veto = opp_sit.is_closing_trend")
    assert idx > 0
    idx_build_call = _SRC.index("_opp_sit = self._build_opponent_situation(")
    idx_eval_call = _SRC.index("_eval = self._evaluate_engage_readiness(")
    assert idx_build_call < idx_eval_call


def test_veto_applied_once_at_call_site_not_inside_plan_pass():
    """①非矛盾性: ゲートが_plan_pass内部(障害物分岐/走行中の相手分岐)を
    変更するのではなく、呼び出し元1箇所でどちらの経路にも一律適用される
    ことを確認する(131節の第1案が2箇所に別々の実装を持ち、revertが
    片方漏れた反省を踏まえた設計)。"""
    assert _SRC.count("_dlat_ttc_veto = opp_sit.is_closing_trend") == 1
    idx_call = _SRC.index("_plan_ok, _plan_side, _plan_req = self._plan_pass(")
    idx_veto = _SRC.index("_dlat_ttc_veto = opp_sit.is_closing_trend")
    assert idx_call < idx_veto


def test_can_engage_includes_veto_condition():
    idx = _SRC.index("_can_engage = _cheap_ok and _plan_ok and not _dlat_ttc_veto")
    assert idx > 0


def test_dbg_plan_reason_set_only_when_veto_is_the_actual_blocker():
    """plan_ok=True(_plan_pass自体は成立)だがdlat_ttc_vetoで最終的に
    ブロックされた場合のみ、診断理由が"dlat_ttc"に上書きされることを確認する
    (plan_ok=False側の既存理由タグを不要に上書きしない)。"""
    idx = _SRC.index('if _plan_ok and _dlat_ttc_veto:')
    snippet = _SRC[idx:idx + 100]
    assert '_dbg_plan_reason = "dlat_ttc"' in snippet


def test_veto_log_is_edge_triggered():
    idx = _SRC.index('f"[DLAT-TTC-VETO]')
    snippet = _SRC[max(0, idx - 400):idx]
    assert "_dlat_ttc_veto_effective and not self._dlat_ttc_veto_active" in snippet


def test_veto_state_initialized():
    assert "self._dlat_ttc_veto_active = False" in _SRC
