"""壁基準の左右空き幅計算の共通化(230節続報3、2026-07-29)。

背景: 横方向重複調査(230節続報、STUCK WAIT_REVERSE統合[230節続報2]に続く2件目)
で、「壁位置(wp.ub/wp.lb)から相手車の該当側端(lat ± self._ot_block_half)までの
空き幅」を計算する同一の2行が、以下7箇所に手作業で複製されていることが判明した:
  1) _scan_traffic (lf/rf、常にclamp)
  2) _side_blocked_by_other_car / [K-CHECK] (c_room、sideでゲート、常にclamp)
  3) _opponent_room_ahead (room、sideでゲート、意図的にNOT clamp)
  4) _plan_pass の lf0/rf0 (常にclamp)
  5) _plan_pass の窓内ループ lf_i/rf_i (常にclamp)
  6) _plan_pass の steps.append 蓄積 (意図的にNOT clamp)
  7) _control の along_lat(並走ねばり)処理 (_opp_rightでゲート、意図的にNOT clamp)

7箇所中4箇所はclamp=True(負値を0に丸め「瞬時の空きゼロ」として扱う)だったが、
残り3箇所(_opponent_room_ahead・_plan_passのsteps蓄積・along_lat)は意図的に
clamp=False(負値そのものが「既に食い込んでいる」量として後続のroom_exhausted
判定・最小値追跡に使われる)だった。全箇所を機械的にclamp=Trueへ統一すると
後続ロジックが壊れるため、共通ヘルパー_room_to_wall(wp, lat, want_left, clamp)は
clamp引数を必須パラメータとして残し、各呼び出し元の既存挙動を1件も変えずに
そのまま集約した。

mpc_controller.pyはrclpy非依存のため直接importできず、他の巨大メソッド関連
テスト群と同じ方針(ソーステキストによる構造的検証、および純Python版ヘルパーを
複製した数式的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


class _WP:
    def __init__(self, ub, lb):
        self.ub = ub
        self.lb = lb


def _room_to_wall_mirror(wp, lat, want_left, clamp=True):
    """_room_to_wallの純Python複製(数式の健全性そのものを検証するため)。"""
    val = (float(wp.ub) - (lat + 0.4) if want_left
           else (lat - 0.4) - float(wp.lb))
    return max(0.0, val) if clamp else val


# ---------------------------------------------------------------------------
# ①非矛盾性: ヘルパー自体の数式が正しいこと(左右対称・clamp有無)
# ---------------------------------------------------------------------------

def test_want_left_true_matches_ub_minus_lat_plus_block_half():
    wp = _WP(ub=3.0, lb=-3.0)
    assert _room_to_wall_mirror(wp, lat=0.5, want_left=True) == 3.0 - (0.5 + 0.4)


def test_want_left_false_matches_lat_minus_block_half_minus_lb():
    wp = _WP(ub=3.0, lb=-3.0)
    assert _room_to_wall_mirror(wp, lat=0.5, want_left=False) == (0.5 - 0.4) - (-3.0)


def test_clamp_true_floors_negative_at_zero():
    wp = _WP(ub=1.0, lb=-1.0)
    # lat=2.0はwp.ub=1.0を大きく超えており、want_left側は本来大幅な負値になる
    assert _room_to_wall_mirror(wp, lat=2.0, want_left=True, clamp=True) == 0.0


def test_clamp_false_preserves_negative_value():
    wp = _WP(ub=1.0, lb=-1.0)
    val = _room_to_wall_mirror(wp, lat=2.0, want_left=True, clamp=False)
    assert val < 0.0
    assert val == 1.0 - (2.0 + 0.4)


def test_helper_signature_has_clamp_default_true():
    idx = _SRC.index("def _room_to_wall(self, wp, lat: float, want_left: bool, clamp: bool = True)")
    assert idx > 0


# ---------------------------------------------------------------------------
# ④遡及効果: 7箇所全ての呼び出し元が、検証済みのclamp挙動を保ったまま
#   ヘルパー経由に置き換わっていること
# ---------------------------------------------------------------------------

def test_scan_traffic_uses_helper_clamped_both_sides():
    idx = _SRC.index("lf = self._room_to_wall(wp, lat, want_left=True, clamp=True)")
    snippet = _SRC[idx:idx + 150]
    assert "rf = self._room_to_wall(wp, lat, want_left=False, clamp=True)" in snippet


def test_k_check_uses_helper_side_gated_clamped():
    idx = _SRC.index(
        "c_room = self._room_to_wall(wp_t, c_lat, want_left=(side > 0), clamp=True)")
    assert idx > 0


def test_opponent_room_ahead_uses_helper_side_gated_not_clamped():
    idx = _SRC.index("def _opponent_room_ahead")
    snippet = _SRC[idx:idx + 2200]
    assert "self._room_to_wall(wps[i], lat_o, want_left=(side > 0), clamp=False)" in snippet


def test_plan_pass_lf0_rf0_uses_helper_clamped():
    idx = _SRC.index("lf0 = self._room_to_wall(wp_t, fwd_lat, want_left=True, clamp=True)")
    snippet = _SRC[idx:idx + 150]
    assert "rf0 = self._room_to_wall(wp_t, fwd_lat, want_left=False, clamp=True)" in snippet


def test_plan_pass_lookahead_loop_uses_helper_clamped():
    idx = _SRC.index("lf_i = self._room_to_wall(wps[i], fwd_lat, want_left=True, clamp=True)")
    snippet = _SRC[idx:idx + 200]
    assert "rf_i = self._room_to_wall(wps[i], fwd_lat, want_left=False, clamp=True)" in snippet


def test_plan_pass_steps_append_uses_helper_not_clamped():
    idx = _SRC.index("steps.append((seg,")
    snippet = _SRC[idx:idx + 250]
    assert "self._room_to_wall(wps[i], lat_o, want_left=True, clamp=False)" in snippet
    assert "self._room_to_wall(wps[i], lat_o, want_left=False, clamp=False)" in snippet


def test_along_lat_uses_helper_opp_right_gated_not_clamped():
    idx = _SRC.index(
        "_lane = self._room_to_wall(_w, _a_lat, want_left=_opp_right, clamp=False)")
    assert idx > 0


# ---------------------------------------------------------------------------
# ②非冗長性: 旧来の手作業複製(壁基準の2行フォーミュラのインライン展開)が
#   ヘルパー本体以外のどこにも残っていないこと
# ---------------------------------------------------------------------------

def test_no_hand_duplicated_inline_formula_remains():
    idx_helper = _SRC.index("def _room_to_wall")
    idx_helper_end = _SRC.index("\n    def ", idx_helper + 10)
    before = _SRC[:idx_helper]
    after = _SRC[idx_helper_end:]
    for outside_snippet, label in ((before, "helper定義より前"), (after, "helper定義より後")):
        assert "self._ot_block_half))" not in outside_snippet, (
            f"{label}に旧来のインライン展開が残っている")


def test_total_call_count_matches_seven_known_sites():
    """新しい壁基準空き幅計算箇所が追加/削除された場合はこのテスト自体の更新も必要。
    7箇所×呼び出し数(_scan_traffic:2, K-CHECK:1, _opponent_room_ahead:1,
    _plan_pass lf0/rf0:2, _plan_pass 窓内ループ:2, _plan_pass steps.append:2,
    along_lat:1)=合計11回。"""
    n_calls = _SRC.count("self._room_to_wall(")
    assert n_calls == 11, (
        f"想定していた7箇所・計11回の呼び出しから数が変わっている(現在{n_calls}回)。"
        "新しい壁基準空き幅計算箇所が追加/削除された場合はこのテスト自体の更新も必要。")
