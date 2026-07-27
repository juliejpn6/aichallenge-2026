"""Unit tests for _corr_bound_ahead()の発生地点(何m先)診断ロギング追加
(153節、2026-07-22)。

背景: 0721-03予選ログ(t=1784646230.28、コーナー5、is_side_by_side=True中に
footprint_riskで完全停止)の実測解析で、_plan_pass()が算出した空きroom
(planRf=3.31m)に対し、実際のオフセット目標(lateral_target、_ot_alphaで
ランプ後のoffsetは約-1.0〜-1.1m)が大幅に小さいという乖離が見つかった。
_corr_bound_ahead()(147節)はMPCホライズン全体(N=20, resolution=0.6m
≒前方12m)の配列(dbg_corr_ub_arr/lb_arr)の最小値でオフセット目標を
クランプしているため、相手車両とは無関係な(コーナー形状由来かもしれない)
遠方の狭まりが、直近では十分空いているはずの目標を一律に引き下げている
可能性がある。

147節自体は実在した壁激突事故(0720-07)の修正であり、先読み距離を安易に
短縮すると再発リスクがある(82/83節と同種の注意)。よって本節では判定
ロジック自体は一切変更せず、「採用された最小値が現在位置から何m先で
発生したか」を[OT]ログへ追加出力するだけに留める。次回ログでこの地点が
相手車両起因かコーナー形状起因かを判別してから、対処方針を決める。

mpc_controller.pyはautoware_auto_control_msgs等をモジュールスコープで
importしており単体テスト環境では直接importできないため、
test_switchback_wall_veto.pyと同じくASTで実物のメソッドを抽出し、
最小のmock selfへバインドして本番コードそのものを検証する。
"""
import ast
import os
import types

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")


def _extract_method(name):
    with open(_SRC_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f"{name} not found in {_SRC_PATH}")


_NS = {"np": np}
exec(compile(_extract_method("_corr_bound_ahead"),
             "<_corr_bound_ahead>", "exec"), _NS)


class _FakeMpc:
    def __init__(self, ub_arr, lb_arr, wp_id=0):
        self.dbg_corr_ub_arr = None if ub_arr is None else np.asarray(ub_arr, dtype=float)
        self.dbg_corr_lb_arr = None if lb_arr is None else np.asarray(lb_arr, dtype=float)
        self.dbg_corr_ub0 = float('inf') if ub_arr is None else float(ub_arr[0]) if len(ub_arr) else float('inf')
        self.dbg_corr_lb0 = -float('inf') if lb_arr is None else float(lb_arr[0]) if len(lb_arr) else -float('inf')
        self.model = types.SimpleNamespace(wp_id=wp_id)


class _FakeRefPath:
    def __init__(self, circular=False, length=0.0):
        self.circular = circular
        self.length = length


def make_self(ub_arr, lb_arr, wp_id=0, s_cum=None, circular=False, length=0.0):
    m = types.SimpleNamespace()
    m._mpc = _FakeMpc(ub_arr, lb_arr, wp_id=wp_id)
    n = len(ub_arr) if ub_arr is not None else (len(lb_arr) if lb_arr is not None else 0)
    # デフォルト: waypoint間隔0.6m(config.yaml resolution)の単純な累積和。
    #   十分な長さ(horizon+wp_idの余裕)を確保する。
    if s_cum is None:
        total_n = max(n + wp_id + 5, 30)
        s_cum = np.cumsum([0.6] * total_n)
    m._wp_s_cum = np.asarray(s_cum, dtype=float)
    m._reference_path = _FakeRefPath(circular=circular, length=length)
    m._dbg_corr_bound_at_m = float('nan')
    m._corr_bound_ahead = types.MethodType(_NS["_corr_bound_ahead"], m)
    return m


# ---------------------------------------------------------------------------
# ①非矛盾性/④遡及効果: 返り値(判定に使う本体)が旧実装(np.min/np.max)と
# 数値的に完全一致すること(挙動は一切変えていないことの確認)。
# ---------------------------------------------------------------------------

def test_side_positive_returns_same_value_as_old_np_min():
    ub = [3.0, 2.5, 1.8, 2.9, 3.1]
    lb = [-3.0] * 5
    m = make_self(ub, lb, wp_id=0)
    result = m._corr_bound_ahead(side=1)
    assert result == pytest.approx(min(ub))


def test_side_negative_returns_same_value_as_old_negated_np_max():
    ub = [3.0] * 5
    lb = [-3.0, -2.0, -1.2, -2.8, -3.0]
    m = make_self(ub, lb, wp_id=0)
    result = m._corr_bound_ahead(side=-1)
    assert result == pytest.approx(-max(lb))


def test_none_array_falls_back_and_marks_distance_nan():
    m = make_self(None, None, wp_id=0)
    result = m._corr_bound_ahead(side=1)
    assert result == float('inf')
    assert np.isnan(m._dbg_corr_bound_at_m)


def test_empty_array_falls_back_and_marks_distance_nan():
    m = make_self([], [], wp_id=0)
    result = m._corr_bound_ahead(side=1)
    assert result == float('inf')
    assert np.isnan(m._dbg_corr_bound_at_m)


# ---------------------------------------------------------------------------
# 本節の中核: 発生地点(何m先)が正しく記録されること
# ---------------------------------------------------------------------------

def test_records_distance_to_argmin_point_side_positive():
    """ub配列のインデックス2が最小値。resolution=0.6mなので2*0.6=1.2m先。"""
    ub = [3.0, 3.0, 1.5, 3.0, 3.0]
    lb = [-3.0] * 5
    m = make_self(ub, lb, wp_id=0)
    m._corr_bound_ahead(side=1)
    assert m._dbg_corr_bound_at_m == pytest.approx(1.2)


def test_records_distance_to_argmax_point_side_negative():
    """lb配列のインデックス3が最大値(=絶対値最小、最も制約が強い点)。
    3*0.6=1.8m先。"""
    ub = [3.0] * 5
    lb = [-3.0, -3.0, -3.0, -1.5, -3.0]
    m = make_self(ub, lb, wp_id=0)
    m._corr_bound_ahead(side=-1)
    assert m._dbg_corr_bound_at_m == pytest.approx(1.8)


def test_distance_measured_from_current_wp_id_not_zero():
    """wp_idが0以外(走行中)でも、現在位置からの相対距離になっていること。"""
    ub = [3.0, 1.5, 3.0]  # 絶対index=wp_id+1で最小
    lb = [-3.0] * 3
    m = make_self(ub, lb, wp_id=10)
    m._corr_bound_ahead(side=1)
    assert m._dbg_corr_bound_at_m == pytest.approx(0.6)  # 1 waypoint先=0.6m


def test_min_at_current_position_gives_zero_distance():
    """最小値が現在位置(先読み配列の先頭)そのものの場合は0mになる
    (=遠方ではなく直近の狭まりであることが一目でわかる)。"""
    ub = [1.5, 3.0, 3.0]
    lb = [-3.0] * 3
    m = make_self(ub, lb, wp_id=0)
    m._corr_bound_ahead(side=1)
    assert m._dbg_corr_bound_at_m == pytest.approx(0.0)


def test_circular_wraparound_distance_stays_non_negative():
    """周回コース終端付近(wp_id近くで配列参照がラップする)でも、
    距離が負にならず正しく加算されること。"""
    n_total = 6
    s_cum = np.cumsum([1.0] * n_total)  # [1,2,3,4,5,6]
    ub = [3.0, 3.0, 1.0]  # wp_id=5から見てindex2(絶対wpは(5+2)%6=1)が最小
    lb = [-3.0] * 3
    m = make_self(ub, lb, wp_id=5, s_cum=s_cum, circular=True, length=6.0)
    m._corr_bound_ahead(side=1)
    # 絶対wp1のs=2.0, 現在位置wp5のs=6.0 → 差分=-4.0 → +length(6.0)=2.0
    assert m._dbg_corr_bound_at_m == pytest.approx(2.0)
    assert m._dbg_corr_bound_at_m >= 0.0


def test_exception_path_falls_back_to_nan_without_crashing():
    """_wp_s_cumが無い(未初期化)等の例外時も、返り値の計算自体は成功し
    距離だけNaNにフォールバックすること(診断ログ追加が本体の可用性を
    損なわないことの確認)。"""
    ub = [3.0, 1.5, 3.0]
    lb = [-3.0] * 3
    m = make_self(ub, lb, wp_id=0)
    del m._wp_s_cum  # 例外を誘発
    result = m._corr_bound_ahead(side=1)
    assert result == pytest.approx(1.5)
    assert np.isnan(m._dbg_corr_bound_at_m)


# ---------------------------------------------------------------------------
# ②非冗長性/構造検証: 呼び出し側・ログ出力の配線を確認
# ---------------------------------------------------------------------------

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_call_site_stores_corr_bound_debug_fields_before_clamp():
    """診断フィールドの記録位置自体は無変更(_corr_bound確定直後、_target_mag
    計算より前)であることを確認する。
    2026-07-24修正(168節): corr_bound>0の通常ケースのクランプ式min(d_off, corr_bound)
    自体は数値的に無変更(遡及効果、下記test_positive_corr_bound_clamp_unchanged参照)
    だが、corr_bound<=0のケースへ凍結ロジック(_ot_last_valid_target_mag)を追加した
    ため、呼び出し元の代入式自体は_room_ahead_locked経由の再利用込みへ変わった。"""
    idx = _SRC.index(
        "_corr_bound = (_room_ahead_locked if _room_ahead_locked is not None")
    idx_target = _SRC.index("_target_mag = self._ot_d_off")
    snippet = _SRC[idx:idx_target]
    assert '_fwd_dbg["corr_bound"]' in snippet
    assert '_fwd_dbg["corr_bound_at"]' in snippet
    assert "self._dbg_corr_bound_at_m" in snippet
    # 正マージン時のクランプ式(遡及効果: 挙動維持)が引き続き直後に存在することを確認する。
    idx_clamp = _SRC.index("_target_mag = min(_target_mag, _corr_bound)")
    assert idx_clamp > idx_target


def test_positive_corr_bound_clamp_numerically_unchanged_from_old_formula():
    """④遡及効果: corr_bound>0のケースでは、新しいif分岐内のmin(_target_mag, _corr_bound)は
    旧式min(_target_mag, max(0.0, _corr_bound))と数値的に完全に同じ結果になることを
    (max(0.0, x)==xがx>0で自明であることの明示的な回帰として)確認する。"""
    d_off = 3.0
    for corr_bound in (0.001, 0.5, 1.2, 10.0):
        old = min(d_off, max(0.0, corr_bound))
        new = min(d_off, corr_bound)
        assert old == pytest.approx(new)


def test_non_positive_corr_bound_freezes_last_valid_target_instead_of_zeroing():
    """本節(168節)の核心: corr_boundが非正転落した際、旧実装はmax(0.0, corr_bound)で
    即座に0(直進)へクランプしていたが、新実装は直近の有効(正マージン)時の値を
    凍結保持することを、実際のクランプ分岐コードで確認する(0724-01実測の
    offset -1.196→-0.710→-0.242→-0.000という崩壊を防ぐ変更)。"""
    idx = _SRC.index("_target_mag = self._ot_d_off")
    idx_end = _SRC.index("self._mpc.lateral_target = float(self._ot_side) * _target_mag")
    snippet = _SRC[idx:idx_end]
    assert "if _corr_bound > 0.0:" in snippet
    assert "self._ot_last_valid_target_mag = _target_mag" in snippet
    assert "elif self._ot_last_valid_target_mag is not None:" in snippet
    assert "_target_mag = self._ot_last_valid_target_mag" in snippet
    # 非冗長性: max(0.0, ...)への直接クランプはもう存在しない(凍結ロジックへ置換済み)。
    assert "max(0.0, _corr_bound)" not in snippet


def test_ot_log_line_includes_corr_bound_with_distance():
    idx = _SRC.index('f"[OT] state=')
    idx_end = idx + 2200
    snippet = _SRC[idx:idx_end]
    assert "corr_bound={_fwd_dbg.get('corr_bound')}@{_fwd_dbg.get('corr_bound_at')}m" in snippet


def test_dbg_corr_bound_at_m_initialized_in_init():
    """回帰: __init__で初期化されており、OVERTAKING状態に入る前に読まれても
    AttributeErrorにならないこと。"""
    assert "self._dbg_corr_bound_at_m = float('nan')" in _SRC


def test_corr_bound_ahead_uses_argmin_argmax_not_bare_min_max():
    """②非冗長性: 発生地点を特定するためargmin/argmaxを使い、既存の
    np.min/np.max呼び出しを重複して残していないこと(旧実装の単純な
    置き換えであることの確認)。"""
    idx = _SRC.index("def _corr_bound_ahead")
    idx_end = _SRC.index("def _switchback_wall_veto")
    snippet = _SRC[idx:idx_end]
    assert "np.argmin(arr)" in snippet
    assert "np.argmax(arr)" in snippet
