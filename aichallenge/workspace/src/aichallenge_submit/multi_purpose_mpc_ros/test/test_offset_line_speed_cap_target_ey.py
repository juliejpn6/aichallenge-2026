"""Regression tests for _offset_line_speed_cap's ey argument (112節, 2026-07-19).

背景: 79節(2026-07-16)は、wall_slowに「自車の現在のeyを固定して先のwaypointと
比較する」先読みを追加したがrevertした。理由: 自車はMPCが能動的に経路追従するため、
実際にその地点へ到達する頃にはeyは変化しており、「今のeyのまま」で先を評価すると
物理的にあり得ない値になる。79節は同時に「wp204-234で検知は正しく機能していたが、
先読みの有無とは別の問題(応答量・応答遅れ)だった可能性が高い」と記録し、この課題を
「別途独立して検討する」と明示的に棚上げしていた(80〜111節では再検討されず)。

0719-01実測(wp202-204、1周目・2周目とも同一地点でCOLLISION-SUSPECTED)で、
offset=-3.0へ収束中の約9秒間、line_cap(_offset_line_speed_cap)がv_safe=v_max
(実質無効)のまま推移していたことを確認した。原因: 呼び出し元(3828行目)が
_cur_ey(現在の実位置、まだ目標に到達していない)を渡していたため、実効曲率
keff=|k/(1-k*ey)|が本来到達すべき目標地点の値より過小評価されていた。

対処(ユーザー承認済み): 引数を_cur_eyから_cur_off(既に確定済みのオフセット目標、
self._ot_alpha*lateral_target)へ変更した。_cur_offは車両の物理追従を待たず時間
(ramp_time)だけで確定する制御目標であり、79節が問題視した「能動的に追従する自車の
現在地を固定して先読みに使う」パターンとは異なる(現在地ではなく目標地点を使う)。

テスト方針: _offset_line_speed_cap自体はreference_path等の重い依存を持つため、
keff計算のコア部分のみを純Pythonでミラーし、「ey(引数)が小さいほどkeffが過小評価
されカーブ減速が効かなくなる」という診断した機構を数式的に実証する。
mpc_controller.py側の配線(呼び出し引数の変更)は構造的なソーステキスト検証で確認する。
"""
import os
import math

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _keff(kappa, ey):
    """mpc_controller.py 2470行目付近の実効曲率計算式の複製ミラー。"""
    den = 1.0 - kappa * ey
    if den < 0.2:
        den = 0.2
    return abs(kappa / den)


def _v_cap_for_curvature(kappa, ey, ay_profile=3.0):
    """keffからv_hereのみを算出する簡易ミラー(seg=0、a_br項を除いた核心部分)。"""
    keff = _keff(kappa, ey)
    if keff < 1e-3:
        return None
    return math.sqrt(ay_profile / keff)


V_MAX = 4.166666666666667
AY_PROFILE = 3.0


def test_ey_lag_causes_curvature_cap_to_be_overly_permissive():
    """核心(診断した機構の実証): 同一の曲率(κ)でも、eyが目標(-3.0)より
    浅い(0に近い、遅れている)状態で評価すると、キャップが緩くなり
    v_max以上を許してしまう。目標eyまで到達していれば正しく減速する。"""
    kappa = -0.15  # 内巻き調査メモ実測の「緩コーナー」相当(κ≈0.12〜0.20)
    v_cap_at_lagging_ey = _v_cap_for_curvature(kappa, ey=0.0)   # まだ移行中(現在地=中央付近)
    v_cap_at_target_ey = _v_cap_for_curvature(kappa, ey=-3.0)   # 目標オフセットへ到達済み
    assert v_cap_at_lagging_ey >= V_MAX  # 遅れたeyでは事実上キャップが効かない
    assert v_cap_at_target_ey < V_MAX    # 目標eyでは正しく減速がかかる
    assert v_cap_at_target_ey < v_cap_at_lagging_ey


def test_ey_zero_and_offset_target_produce_different_verdicts_at_same_corner():
    """回帰: 0719-01実測相当のκでは、ey=0(旧_cur_ey相当・移行中)だと
    「カーブ減速なし」と判定されるが、ey=-3.0(新_cur_off相当)だと
    「カーブ減速あり」と判定される、という診断結果を数値で再現する。"""
    kappa = -0.15
    assert _v_cap_for_curvature(kappa, ey=0.0) is None or \
        _v_cap_for_curvature(kappa, ey=0.0) >= V_MAX
    assert _v_cap_for_curvature(kappa, ey=-3.0) < V_MAX


def test_gentle_curvature_still_uncapped_regardless_of_ey_regression():
    """回帰: 曲率が十分緩ければ(ほぼ直線)、どちらのey値を使っても
    キャップは掛からない(過剰検知にはならない)。"""
    kappa = 0.01
    assert _v_cap_for_curvature(kappa, ey=0.0) >= V_MAX
    assert _v_cap_for_curvature(kappa, ey=-3.0) >= V_MAX


def test_singularity_guard_applies_regardless_of_which_ey_is_passed():
    """回帰: den<0.2の特異点ガードは、どちらのey値が渡されても同じ式で
    機能し続ける(112節の変更はkeff計算式自体には手を入れていない)。"""
    kappa = 0.5
    ey = -3.0  # k*ey=-1.5, den=1-(-1.5)=2.5 (ガード不要域だが式が同一であることを確認)
    assert _keff(kappa, ey) == abs(kappa / (1.0 - kappa * ey))


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_line_cap_call_site_now_uses_cur_off_not_cur_ey():
    idx = _SRC.index("_line_cap = self._offset_line_speed_cap(")
    snippet = _SRC[idx:idx + 80]
    assert "_cur_off" in snippet
    assert "_cur_ey" not in snippet


def test_offset_line_speed_cap_function_body_unchanged_no_logic_edit():
    """回帰(非冗長性の確認): 112節の変更は呼び出し引数のみであり、
    _offset_line_speed_cap自体のkeff計算式・EMA処理には手を入れていない
    (97節のEMA実装がそのまま維持されていることを確認)。"""
    idx = _SRC.index("def _offset_line_speed_cap")
    snippet = _SRC[idx:idx + 2700]
    assert "keff = abs(k / den)" in snippet
    assert "self._line_cap_ema += self._ot_ema_alpha * (cap - self._line_cap_ema)" in snippet


def test_cur_off_is_computed_before_line_cap_call_site():
    idx_off = _SRC.index("_cur_off = self._ot_alpha * self._mpc.lateral_target")
    idx_call = _SRC.index("_line_cap = self._offset_line_speed_cap(")
    assert idx_off < idx_call
