"""Unit tests for 2026-07-26(徹底解析): _scan_traffic()のalong_lat(並走ねばり)候補選定が
自分の後方にいるだけの車まで拾っていたバグの修正。

背景: 予選ログ0726-02(動画42秒、レース開始+37秒)で、自車が約5.5秒間完全停止し続けた
直後、真横にいた後方車(d2)が後退(バック)を終えて前進を再開した瞬間に自車も動き出す、
という強い相関を実測した。コード側を確認すると、`along_lat`候補の選定
(_scan_traffic()内、mpc_controller.py)が`abs(ds)`(前後を区別しない絶対値)を使っており、
自分の後方3m以内・横1〜3mにいるだけの車も「並走中の相手」として扱われ、その相手の
縦速度に自車速度を合わせる(`_v_yield = max(0, 相手速度-0.5)`)ため、後方車が停止/後退
していると自車も引きずられて停止していた。

同じ関数内のbeing_overtaken判定(2528行目)は既に「後方は自分より速い場合のみ」正しく
限定しており、along_lat側だけがこの非対称性を欠いていた。修正は`abs(ds)<=dist`を
`0.0<=ds<=dist`(前方〜真横のみ)へ変更するのみで、新規パラメータ・新規安全弁は
追加していない。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装した上で
ソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

DEF_ALONGSIDE_DIST = 3.0
DEF_ALONGSIDE_LAT = 1.0


def mirror_along_lat_eligible(ds, dlat, dist=DEF_ALONGSIDE_DIST, lat_min=DEF_ALONGSIDE_LAT):
    """修正後のalong_lat候補選定条件のミラー(mpc_controller.py:2516-2517)。"""
    return (0.0 <= ds <= dist) and (lat_min <= dlat <= 3.0)


# ---------------------------------------------------------------------------
# ①非矛盾性: 前後の区別
# ---------------------------------------------------------------------------

def test_rear_car_no_longer_eligible():
    """回帰の核心: 真後ろ2m・横2mの車は、静止/後退していても対象から除外される。"""
    assert mirror_along_lat_eligible(ds=-2.0, dlat=2.0) is False


def test_rear_car_just_outside_old_abs_boundary_also_excluded():
    """旧実装(abs(ds)<=3.0)では対象だった真後ろ2.9mの車も、新実装では除外される。"""
    assert mirror_along_lat_eligible(ds=-2.9, dlat=2.0) is False


def test_forward_car_still_eligible():
    """前方の車(ds>0)は従来通り対象のまま(挙動不変)。"""
    assert mirror_along_lat_eligible(ds=2.0, dlat=2.0) is True


def test_exactly_side_by_side_ds_zero_still_eligible():
    """真横(ds=0)は境界として対象に含める(前方〜真横、という設計意図)。"""
    assert mirror_along_lat_eligible(ds=0.0, dlat=2.0) is True


def test_dlat_gate_unaffected_by_fix():
    """dlat条件(1.0〜3.0m)自体はこの修正と無関係、従来通り機能する。"""
    assert mirror_along_lat_eligible(ds=1.0, dlat=0.5) is False   # 真正面追従、既存ガード
    assert mirror_along_lat_eligible(ds=1.0, dlat=3.5) is False   # 離れすぎ
    assert mirror_along_lat_eligible(ds=1.0, dlat=1.0) is True    # 下限ちょうど


def test_forward_distance_gate_unaffected_by_fix():
    """前方側の距離条件(<=3.0m)も従来通り機能する。"""
    assert mirror_along_lat_eligible(ds=3.0, dlat=2.0) is True
    assert mirror_along_lat_eligible(ds=3.1, dlat=2.0) is False


# ---------------------------------------------------------------------------
# ②非冗長性・③検証: ソーステキストで実装箇所とbeing_overtaken非破壊を確認
# ---------------------------------------------------------------------------

def test_along_lat_condition_uses_directional_ds():
    idx = _SRC.index('if (0.0 <= ds <= self._def_alongside_dist')
    idx_end = idx + 200
    snippet = _SRC[idx:idx_end]
    assert 'self._def_alongside_lat <= dlat <= 3.0' in snippet
    assert 'abs(ds)' not in snippet


def test_being_overtaken_directional_logic_untouched():
    """同じ関数内のbeing_overtaken判定(既に前後を正しく区別している既存ロジック)は
    今回の修正で変更されていないことを確認する(④遡及効果: 意図しない副作用が無いこと)。"""
    idx = _SRC.index('if v_long >= self._opp_obstacle_speed:')
    idx_end = idx + 400
    snippet = _SRC[idx:idx_end]
    assert 'abs(ds) <= self._def_alongside_dist and dlat >= self._def_alongside_lat' in snippet
    assert '-self._def_rear_dist <= ds < 0.0 and v_long > v_ego + self._def_rear_faster' in snippet


def test_no_new_config_parameters_introduced():
    """②非冗長性: 修正はds比較の演算子変更のみで、新規パラメータを増やしていない。"""
    assert '_def_alongside_dist = float(_otget("def_alongside_dist"' in _SRC
    assert '_def_alongside_lat = float(_otget("def_alongside_lat"' in _SRC
    # 新規のdef_*系パラメータが増えていないことの簡易確認
    assert _SRC.count('self._def_alongside_dist = float(') == 1
    assert _SRC.count('self._def_alongside_lat = float(') == 1
