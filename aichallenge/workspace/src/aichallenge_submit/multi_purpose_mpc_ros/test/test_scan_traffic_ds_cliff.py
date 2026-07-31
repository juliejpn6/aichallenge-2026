"""Regression tests for the "_scan_traffic ds>0 cliff" relaxation and its
2026-07-31(254節)撤回。

背景(旧仕様、105節発見→110節で2環境目再確認→2026-07-19実装):
`_scan_traffic`(mpc_controller.py)の前方車判定は当初`0.0 < ds`という猶予無しの
厳密な正値要求だった。静止/低速の相手にごく至近距離まで詰めた場合、自車の弧長
位置がわずかに相手を跨いだ瞬間にfwd_ds=Noneへ落ち、OFFSET-RETURNが「通過完了」
と誤判定して全開加速する事象(105/110節)への対処として、前方車判定の下限を
車両全長(along_min_length)分だけ後方へ緩和し、`-along_min_length < ds`を
forward判定に含めていた。

2026-07-31(254節)で撤回: この負のds許容窓が、2026-07-20(129節続報)で追加
された対象車選択のフォールバック(前方候補が無ければ後方along_min_length以内で
代替する)と組み合わさることで、icc_stop本体(_follow_speed_limit)・n_fwd
(_ot_state遷移の起点判定)・NO-VSAFEブリッジの3箇所が意図せず後方車を「前方の
対象車」として扱ってしまい、前方が完全に空いていても後方2m以内で減速/停止した
相手にegoが同期して停止し続けるバグ(0731-03 wp243実測、design_docs 254節)を
引き起こしていた。ユーザー確認: 後方車両を追従する必要は無い。

当初の懸念(OFFSET-RETURNのcliff問題)は、131-6節で当該消費箇所自身が既に
`fwd_ds_now > 0.0`を明示チェックする独立した安全策を導入済みであることを
確認した上で、この緩和なしでも再発しないと判断し撤回した。

本ファイルは「撤回後の新しい前方判定式(0.0<=ds)」を検証する回帰テスト群へ
全面的に書き換えた(旧仕様を検証していたテストをそのまま残すと、撤回済みの
挙動を正として固定してしまうため)。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。分類条件
自体は単純な不等式のため、同一の式を複製したミラー関数で数式的性質を検証する。
mpc_controller.py側の配線(条件式そのもの・along_min_lengthが他消費箇所では
引き続き使われていること)は末尾の構造的ソーステキスト検証で確認する。
"""
import os


def _is_forward(ds, lat, max_consider=25.0, lat_band=3.0):
    """mpc_controller.py _scan_trafficの前方車判定式(254節で0.0<=dsへ復帰)の
    複製ミラー。"""
    return 0.0 <= ds <= max_consider and abs(lat) <= lat_band


def test_ds_exactly_zero_classified_as_forward():
    assert _is_forward(ds=0.0, lat=0.0) is True


def test_ds_slightly_negative_no_longer_classified_as_forward_254():
    """核心(254節での変更点): 105/110節時代はdsがわずかに負でも車両全長以内なら
    forward扱いだったが、撤回後はds<0は一律forward対象外に戻る(このds<0側の
    車が後方車追従バグの直接の入力元だったため)。"""
    assert _is_forward(ds=-0.5, lat=0.0) is False
    assert _is_forward(ds=-0.05, lat=0.0) is False


def test_ds_far_behind_still_excluded_regression():
    assert _is_forward(ds=-5.0, lat=0.0) is False


def test_ds_positive_small_still_classified_as_forward_regression():
    """回帰: ds>0の通常ケースは105/110節時代・254節後のいずれでも変更されない。"""
    assert _is_forward(ds=0.5, lat=0.0) is True


def test_lat_band_still_enforced_regression():
    """回帰: lat_band(コリドー帯)の制約は無変更のまま引き続き効く。"""
    assert _is_forward(ds=0.5, lat=10.0) is False


def test_retroactive_0731_03_wp243_rear_car_no_longer_classified_as_forward():
    """遡及検証(254節、0731-03 wp243実測): 後方2m前後(ds=-1.95〜-1.97)に
    居続けた対戦車(d3)は、105/110節時代の判定式ならforward=Trueとなり
    icc_stop等の「前方対象」に選ばれ得た。撤回後の判定式ではforward=Falseと
    なり、この経路自体が塞がれることを確認する。"""
    for ds in (-1.95, -1.97, -1.01, -0.98):
        assert _is_forward(ds=ds, lat=0.3) is False


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_forward_classification_uses_ds_ge_zero_not_negative_window():
    """254節の核心配線確認: cars構築条件が0.0<=dsであり、旧-along_min_length<ds
    という負の許容窓がもう使われていないことを確認する。"""
    idx = _SRC.index("if 0.0 <= ds <= self._ot_max_consider")
    snippet = _SRC[idx:idx + 120]
    assert "self._along_min_length" not in snippet
    assert "-self._along_min_length < ds" not in _SRC


def test_no_negative_ds_lower_bound_pattern_remains_in_cars_construction():
    """水平展開の確認: cars構築条件に負のds許容窓パターンが他に残っていない
    ことを確認する(同型の緩和が別途復活していないことの回帰防止)。"""
    assert "-self._along_min_length < ds <=" not in _SRC
    assert "-self._along_min_length<ds<=" not in _SRC


def test_along_min_length_still_used_by_other_consumers():
    """②非冗長性・非退行の確認: along_min_length自体はfootprint_risk等の
    他の既存消費箇所で引き続き使われている定数であり、254節の変更で
    削除・無効化されたわけではないことを確認する(cars構築条件からの
    参照を外しただけ)。"""
    assert _SRC.count("self._along_min_length") >= 3


def test_offset_return_still_references_fwd_ds_unchanged_by_254():
    """回帰確認: OFFSET-RETURN判定(131-6節で既にds>0の明示チェックを持つ)は
    254節による変更を受けず、引き続き同じ取得元(_scan.get("fwd_ds"))を
    参照していることを確認する(下流コードへの変更は不要だったことの確認)。"""
    idx = _SRC.index('_fwd_ds_now = _scan.get("fwd_ds")')
    assert idx > 0
    idx2 = _SRC.index('_offset_return_ok = self._ot_cleared and not _still_ahead')
    assert idx2 > idx
    idx3 = _SRC.index("_fwd_ds_now is not None and _fwd_ds_now > 0.0")
    assert idx > 0 and idx3 > idx


def test_n_fwd_gate_now_only_counts_genuine_forward_cars():
    """254節の波及確認: _n_fwd(_ot_state遷移のマスターゲート)がscan["cars"]の
    要素数であり、carsが0.0<=ds限定になったことで後方車だけでは真にならない
    ことを、配線(該当コメント)の存在で確認する。"""
    idx = _SRC.index('_n_fwd = len(_scan["cars"])')
    snippet = _SRC[max(0, idx - 350):idx]
    assert "254節" in snippet
