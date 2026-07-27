"""Unit tests for the OVERTAKING v_safe candidate slimdown (143節続報、
2026-07-20)。

背景: ユーザー指摘(「v_safe/追従/footprint_risk/LAT-TTCまわりが追記の積み重ねで
複雑」)を受け、フェーズ2(ICCのnear_sepゲートをOpponentSituationへ配線)に
着手する前段として、_control()内のOVERTAKING v_safe候補選択(①前車なし
②G2-RELEASE解放③F3クリープ床)から、G2-RELEASE判定(_g2_release_ready)と
F3-taper床計算(_f3_taper_speed)をそれぞれ専用メソッドへ抽出した。ロジック・
デバウンス状態・ログとも完全に無変更の純粋リファクタ(挙動は一切変更しない)。

既存のtest_g2_release_debounce.py/test_f3_taper_gap.pyがロジック内容自体は
既に検証済みのため、本ファイルは①抽出の構造そのもの(3択の優先順位が
_control()側から一望できる形になっているか)②_control()が実質的に短くなった
ことの2点に絞って検証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _overtaking_v_safe_snippet():
    idx = _SRC.index("_eff_v_cap = max(self._ot_v_cap,")
    idx_end = _SRC.index("# B: 内側ライン減速")
    return _SRC[idx:idx_end]


def test_three_way_selection_visible_at_a_glance():
    """①非矛盾性: OVERTAKING中のv_safe候補選択が、if/elif/elseの3分岐で
    優先順位ごと一望できる形になっていることを確認する(以前は2番目の分岐が
    100行超のインライン処理で埋もれていた)。"""
    snippet = _overtaking_v_safe_snippet()
    assert "if _vlim is None:" in snippet
    assert "elif self._g2_release_ready(" in snippet
    assert "else:" in snippet
    assert "self._f3_taper_speed(" in snippet
    idx_if = snippet.index("if _vlim is None:")
    idx_elif = snippet.index("elif self._g2_release_ready(")
    idx_else = snippet.index("else:")
    assert idx_if < idx_elif < idx_else


def test_control_block_no_longer_contains_inline_g2_or_f3_logic():
    """②非冗長性: _control()側のOVERTAKING分岐から、G2-RELEASEのデバウンス
    変数計算(_side_clear_raw等)・F3-taperのゾーン計算(_est_gap等)が完全に
    除去され、メソッド呼び出しのみが残っていることを確認する。"""
    snippet = _overtaking_v_safe_snippet()
    for removed in ("_side_clear_raw", "_stopped_opponent", "_side_room_ok_now",
                     "_actual_lat_clear_now", "_offset_committed", "_est_gap",
                     "_f3_zone"):
        assert removed not in snippet, f"{removed} should have been extracted"


def test_slimmed_block_is_dramatically_shorter():
    """スリム化の効果を定量的に確認する(以前は同じ範囲が約180行あった)。"""
    snippet = _overtaking_v_safe_snippet()
    n_lines = snippet.count("\n")
    assert n_lines < 25, f"expected a short 3-way selection, got {n_lines} lines"


def test_g2_release_ready_method_exists_with_expected_signature():
    idx = _SRC.index("def _g2_release_ready(self, scan, fwd_vopp, vtgt, left_free, right_free,")
    assert idx > 0


def test_f3_taper_speed_method_exists_with_expected_signature():
    idx = _SRC.index("def _f3_taper_speed(self, vtgt, eff_v_cap: float, vlim: float)")
    assert idx > 0


def test_g2_release_ready_still_mutates_same_debounce_state():
    """④過去ログへの遡及効果に相当する健全性チェック: 抽出後もデバウンス状態
    (self._g2_clear_on_count / self._g2_release_debounced / self._g2_release_prev)
    が全て同じ属性名のまま維持されており、既存ログ(0718実測の7回反転抑止など)
    との整合が保たれていることを確認する。"""
    idx = _SRC.index("def _g2_release_ready(")
    idx_end = _SRC.index("def _f3_taper_speed(")
    snippet = _SRC[idx:idx_end]
    for attr in ("self._g2_clear_on_count", "self._g2_release_debounced",
                 "self._g2_release_prev", "self._ot_engage_debounce"):
        assert attr in snippet


def test_f3_taper_speed_still_uses_same_gap_constants():
    idx = _SRC.index("def _f3_taper_speed(")
    idx_end = _SRC.index("def _control(self):")
    snippet = _SRC[idx:idx_end]
    assert "self._ot_f3_taper_gap" in snippet
    assert "self._ot_hard_stop_gap" in snippet
    assert "self._ot_v_creep" in snippet
    assert "self._f3_taper_zone_prev" in snippet
