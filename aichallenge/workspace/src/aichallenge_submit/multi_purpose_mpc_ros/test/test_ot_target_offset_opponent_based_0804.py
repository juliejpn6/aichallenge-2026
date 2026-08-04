"""Unit tests for opponent-position-based minimum overtake offset (2026-08-04).

背景: dev3 A/B検証中(D2-D3)、wp340-8(緩いコーナー、|kappa|≈0.02-0.13)でD2が
D3へoffset=2.294m(コリドー上限とほぼ一致)まで幅寄せする事象が2周連続で再現した。
調査の結果、`_target_mag = self._ot_d_off`(固定3.0m)を`_corr_bound_ahead()`で
クランプするだけの従来設計では、「対象車を安全にクリアするのに必要な最小
オフセット」を計算する項が存在せず、コリドーが許す限り常に上限まで幅寄せする
ことが判明した。ユーザー方針:「オーバーテイクが成功すればいい、幅寄せする
必要はない」。

対処: 対象車(`self._ot_target_vid`)の現在の横位置(`_scan["cars"]`のlat、
`e_y`と同一フレーム)から、`side*lat + クリアランス(自車半幅+block_half)`を
「これだけ離れれば十分」という必要最小量として計算し、従来のd_off(3.0m)固定値
との小さい方を採用する(corr_boundによる上限クランプは維持)。対象車が今周期の
cars候補から外れている場合は旧来通りd_off固定へフォールバックする(退行防止)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装した
上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _min_needed_mag(ot_side, opp_lat, d_off, clear_needed):
    """_target_mag計算(対象車ベースの枝)のミラー実装。"""
    if opp_lat is None:
        return d_off
    return max(0.0, min(d_off, ot_side * opp_lat + clear_needed))


# --- ①非矛盾性: 対象車がほぼ中央にいる通常ケース ---

def test_opponent_near_centerline_needs_less_than_full_d_off():
    """相手がほぼレース基準線上(lat=0.1)にいる場合、必要量はクリアランス
    (自車半幅1.15+block_half0.9=2.05)程度で足り、従来の固定d_off(3.0m)より
    小さくなる(=不要な幅寄せをしなくなる、本節の核心)。"""
    mag = _min_needed_mag(ot_side=1, opp_lat=0.1, d_off=3.0, clear_needed=2.05)
    assert mag == 2.15
    assert mag < 3.0


def test_opponent_already_far_on_opposite_side_needs_even_less():
    """相手が既に反対側(ego進行方向と逆)に寄っている場合、必要量はさらに
    小さくなる(相手が自ら空けている側へ最小限だけ動けばよい)。"""
    mag = _min_needed_mag(ot_side=1, opp_lat=-1.0, d_off=3.0, clear_needed=2.05)
    assert mag == pytest.approx(1.05)


def test_right_side_symmetric():
    """side=-1(右)でも符号が対称に効く(wp340-8はside反転を伴う事例だった)。"""
    mag = _min_needed_mag(ot_side=-1, opp_lat=-0.2, d_off=3.0, clear_needed=2.05)
    assert mag == 2.25


# --- ②境界: 必要量がd_offを超える/負になるケース ---

def test_needed_magnitude_capped_at_d_off():
    """相手が目標側へ大きく寄っている場合、必要量がd_off(3.0m)を超えても
    従来通りd_offで頭打ちする(コリドー上限クランプは別途下流で適用)。"""
    mag = _min_needed_mag(ot_side=1, opp_lat=5.0, d_off=3.0, clear_needed=2.05)
    assert mag == 3.0


def test_needed_magnitude_floored_at_zero():
    """相手が既に目標側の奥まで離れている場合、必要量は負にならず0で
    floorされる(逆方向への目標は出さない)。"""
    mag = _min_needed_mag(ot_side=1, opp_lat=-4.0, d_off=3.0, clear_needed=2.05)
    assert mag == 0.0


# --- ③退行防止: 対象車が見つからない周期は旧来のd_off固定へフォールバック ---

def test_falls_back_to_fixed_d_off_when_opponent_not_in_view():
    mag = _min_needed_mag(ot_side=1, opp_lat=None, d_off=3.0, clear_needed=2.05)
    assert mag == 3.0


# --- ④配線確認: 実装が対象車位置ベースの計算をcorr_boundクランプより前に持つこと ---

def test_target_mag_computed_from_target_vid_lookup_before_corr_bound_clamp():
    idx_corr_at = _SRC.index('_fwd_dbg["corr_bound_at"] = round(self._dbg_corr_bound_at_m, 2)')
    idx_clamp = _SRC.index("if np.isfinite(_corr_bound):")
    snippet = _SRC[idx_corr_at:idx_clamp]
    assert "_opp_lat_now" in snippet
    assert 'if _c_vid == self._ot_target_vid:' in snippet
    assert "_opp_lat_now = _c_lat" in snippet
    # 必要量計算式: side*lat + クリアランス、d_offとの小さい方、0でfloor
    assert "float(self._ot_side) * _opp_lat_now + _clear_needed" in snippet
    assert "_target_mag = max(0.0, min(" in snippet
    # 対象車が見つからない場合は旧来のd_off固定へフォールバック
    assert "else:" in snippet
    assert "_target_mag = self._ot_d_off" in snippet


def test_clearance_uses_own_half_width_plus_block_half_no_double_margin():
    """②非冗長性: クリアランスは自車半幅+block_half(=相手半幅+余裕)のみで、
    corr_bound側(ub0/lb0)に既に織り込み済みのsafety_margin_overtakeは
    二重に加算しない(6269行目付近の既存コメントの設計判断を踏襲)。"""
    idx = _SRC.index("_clear_needed = self._mpc.model.width / 2.0 + self._ot_block_half")
    idx_end = _SRC.index("if np.isfinite(_corr_bound):")
    snippet = _SRC[idx:idx_end]
    assert "self._ot_safety_margin" not in snippet


def test_opponent_lookup_uses_scan_cars_not_fwd_dlat():
    """①非矛盾性: fwd_dlat(自車現在位置基準の絶対横間隔)ではなく、対象車の
    絶対横位置(lat、lateral_targetと同一フレーム)を使う。fwd_dlatは目標e_y
    との直接比較ができないため使えない(調査時に確認済みの誤用パターン)。"""
    idx = _SRC.index('_fwd_dbg["corr_bound_at"] = round(self._dbg_corr_bound_at_m, 2)')
    idx_end = _SRC.index("if np.isfinite(_corr_bound):")
    snippet = _SRC[idx:idx_end]
    assert 'for _c_ds, _c_lat, _c_vlong, _c_dlat, _c_vid, _c_wp in _scan["cars"]:' in snippet
    assert "_opp_lat_now = _c_lat" in snippet
