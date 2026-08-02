"""Regression tests for _opponent_room_ahead (103節 Phase 0, 2026-07-18).

Background: 0718-03実測で、A_rescue_relaxed(最終救済の側反転)が同一地点
(wp≈333〜335)で2回発火し、2回とも発火の約0.8秒後に反転先でも
curvature_blocked=Trueのまま強制giveupに至った(102節)。根本原因は、反転の
可否判定が対象車両ID(fwd_vid)の位置を一切見ておらず、静的トラック曲率
(_switchback_curvature_veto)と瞬時のopp_space比較のみで決めていたこと。

一方、エンゲージ時の側選択(_plan_pass)は既にOpponentSpeedMap.lat_mean(vid, i)
で対象車両IDごとに学習済みの走行ラインを使ってroomを算出しており(2205〜2235行目
付近)、並走中の反転判定だけがこの資産を使っていない、という非対称性が
103節のFDで判明した。

Phase 0(本ファイルの対象): _plan_passのroom計算パターンを軽量に切り出した
_opponent_room_aheadを新設し、A_rescue_relaxed発火時にのみ診断ログ
[REVERSE-ROOM-CHECK]として出力する(判定ロジックには一切使わない、既存挙動は
不変)。次回ログで、この新しい先読み値が0718-03のような破綻を事前検知
できていたかを確認する。

テスト方針: mpc_controller.pyはautoware_auto_control_msgs等のROSメッセージ型を
モジュールレベルでimportしており、単体テスト環境では直接importできない
(test_switchback_token_wiring.py等、既存の同種テストと同じ制約)。
_opponent_room_ahead自体はself._opp_map/self._reference_path/self._wp_s_cum/
self._ot_block_halfの4属性のみを参照する自己完結した関数のため、そのロジックを
そのまま複製したミラー関数を用意し、実物のOpponentSpeedMap(ROS非依存)を使って
数式的性質を検証する。mpc_controller.py側の配線(呼び出し箇所・ログ出力)は
末尾の構造的ソーステキスト検証で確認する。
"""
import os

import numpy as np
import pytest

from multi_purpose_mpc_ros.opponent_speed_map import OpponentSpeedMap


class _FakeWp:
    def __init__(self, ub, lb):
        self.ub = ub
        self.lb = lb


class _FakeReferencePath:
    def __init__(self, n, circular=True):
        # 単純化: 全waypointが同じub/lb(壁境界)を持つ直線的なコースを模す。
        self.waypoints = [_FakeWp(ub=3.0, lb=-3.0) for _ in range(n)]
        self.length = float(n)
        self.circular = circular


def _opponent_room_ahead_mirror(opp_map, rp, wp_s_cum, block_half, vid, wp_id, side, n_ahead):
    """mpc_controller.py の _opponent_room_ahead(103節Phase 0)の複製ミラー。
    アルゴリズムはソース側と完全に同一(ub/lb・lat_mean・先読み窓の扱い)。"""
    om = opp_map
    if om is None or vid is None:
        return None, None, 0
    try:
        wps = rp.waypoints
        n = len(wps)
        i0 = int(wp_id) % n
        s0 = float(wp_s_cum[i0])
        total = rp.length
    except Exception:
        return None, None, 0
    room_min = None
    wp_at_min = None
    n_sampled = 0
    for d in range(1, n):
        i = (i0 + d) % n
        seg = float(wp_s_cum[i]) - s0
        if rp.circular and seg < 0.0:
            seg += total
        if seg > n_ahead:
            break
        lat_o = om.lat_mean(vid, i)
        if lat_o is None:
            continue
        n_sampled += 1
        if side > 0:
            room = float(wps[i].ub) - (lat_o + block_half)
        else:
            room = (lat_o - block_half) - float(wps[i].lb)
        if room_min is None or room < room_min:
            room_min = room
            wp_at_min = i
    return room_min, wp_at_min, n_sampled


def _room_ahead(inst, vid, wp_id, side, n_ahead):
    return _opponent_room_ahead_mirror(
        inst["opp_map"], inst["rp"], inst["wp_s_cum"], inst["block_half"],
        vid, wp_id, side, n_ahead)


def _make_env(n=20, block_half=0.9):
    s_cum = np.arange(n, dtype=float)  # 1m間隔
    rp = _FakeReferencePath(n)
    om = OpponentSpeedMap(n_wp=n, s_cum=s_cum)
    inst = {"opp_map": om, "rp": rp, "wp_s_cum": s_cum, "block_half": block_half}
    return inst, om


def test_returns_none_when_no_opponent_map():
    inst = {"opp_map": None, "rp": _FakeReferencePath(10),
            "wp_s_cum": np.arange(10, dtype=float), "block_half": 0.9}
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 10.0)
    assert (room_min, room_wp, n) == (None, None, 0)


def test_returns_none_when_vid_is_none():
    inst, om = _make_env()
    room_min, room_wp, n = _room_ahead(inst, None, 0, 1, 10.0)
    assert (room_min, room_wp, n) == (None, None, 0)


def test_returns_none_with_no_sampled_data_when_unlearned():
    """学習済みwaypointが窓内に1つも無ければ(None, None, 0)を返す
    (フォールバックしない、Phase 0は判定に使わない計装のみのため)。"""
    inst, om = _make_env()
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 10.0)
    assert (room_min, room_wp, n) == (None, None, 0)


def test_computes_room_using_learned_lat_mean_for_left_side():
    """核心: 学習済みwaypointについて、_plan_passと同一の式
    (wp.ub - (lat_o + block_half))でroomを算出する(side=+1、左)。"""
    inst, om = _make_env(block_half=0.9)
    om.update("car1", wp_id=3, v_long=5.0, settled=True, lat=1.0)
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 10.0)
    assert n == 1
    assert room_wp == 3
    # ub=3.0, lat_o=1.0, block_half=0.9 -> room = 3.0 - (1.0+0.9) = 1.1
    assert room_min == pytest.approx(1.1, abs=1e-6)


def test_computes_room_using_learned_lat_mean_for_right_side():
    """核心: side=-1(右)は(lat_o - block_half) - wp.lbの式を使う。"""
    inst, om = _make_env(block_half=0.9)
    om.update("car1", wp_id=3, v_long=5.0, settled=True, lat=-1.0)
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, -1, 10.0)
    assert n == 1
    # lat_o=-1.0, block_half=0.9, lb=-3.0 -> room = (-1.0-0.9) - (-3.0) = 1.1
    assert room_min == pytest.approx(1.1, abs=1e-6)


def test_returns_minimum_room_across_multiple_learned_waypoints():
    """核心: 窓内に複数の学習済みwaypointがあれば、最も厳しい(狭い)roomを返す。"""
    inst, om = _make_env(block_half=0.9)
    om.update("car1", wp_id=2, v_long=5.0, settled=True, lat=0.5)   # room=3.0-(0.5+0.9)=1.6
    om.update("car1", wp_id=5, v_long=5.0, settled=True, lat=1.5)   # room=3.0-(1.5+0.9)=0.6 (最小)
    om.update("car1", wp_id=8, v_long=5.0, settled=True, lat=0.2)   # room=3.0-(0.2+0.9)=1.9
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 10.0)
    assert n == 3
    assert room_wp == 5
    assert room_min == pytest.approx(0.6, abs=1e-6)


def test_respects_n_ahead_window_cutoff():
    """回帰: n_ahead(先読み距離)を超えるwaypointは対象外とする
    (_switchback_curvature_vetoの_fwd_max_considerと同じ扱い)。"""
    inst, om = _make_env(block_half=0.9)
    om.update("car1", wp_id=3, v_long=5.0, settled=True, lat=1.0)   # 窓内(3m)
    om.update("car1", wp_id=15, v_long=5.0, settled=True, lat=1.0)  # 窓外(15m > n_ahead=5.0)
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 5.0)
    assert n == 1
    assert room_wp == 3


def test_unlearned_waypoints_are_skipped_not_fallback():
    """回帰: 未学習のwaypointはフォールバック値を使わず、単純にスキップする
    (Phase 0は判定に使わない計装のみのため、フォールバックによる汚染を避ける
    設計、103節参照)。"""
    inst, om = _make_env(block_half=0.9)
    om.update("car1", wp_id=1, v_long=5.0, settled=True, lat=2.9)  # 極端値(room=3.0-3.8=-0.8)
    # wp_id=2,3,4は未学習のまま(latを渡さない/settled=Falseなど)。
    om.update("car1", wp_id=1, v_long=5.0, settled=False)  # settled=Falseは学習しない(無視)
    room_min, room_wp, n = _room_ahead(inst, "car1", 0, 1, 10.0)
    assert n == 1  # wp_id=1の1件のみが学習済み
    assert room_wp == 1


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_opponent_room_ahead_reuses_ot_block_half_no_new_magic_number():
    """230節続報3(2026-07-29): 壁基準の空き幅計算は_room_to_wall()へ集約された
    (_scan_traffic等、他6箇所の重複と合わせて共通ヘルパー化)。self._ot_block_half/
    wps[i].ub/lbは_room_to_wall内部で参照される形になったため、ここでは
    _opponent_room_aheadが同ヘルパーをclamp=False(このサイトは意図的に非クランプ)で
    正しく呼び出していることを確認する。"""
    idx = _SRC.index("def _opponent_room_ahead")
    snippet = _SRC[idx:idx + 2200]
    assert "self._room_to_wall(wps[i], lat_o, want_left=(side > 0), clamp=False)" in snippet

    idx_helper = _SRC.index("def _room_to_wall")
    helper_snippet = _SRC[idx_helper:idx_helper + 900]
    assert "self._ot_block_half" in helper_snippet
    assert "wp.ub" in helper_snippet
    assert "wp.lb" in helper_snippet


def test_reverse_room_check_log_only_fires_for_a_rescue_relaxed_branch():
    """107節案C更新: room先読みの計算自体はupdate()呼び出し前に移動したため、
    このブロックは事前計算済みの_room_min等をログするだけになった
    (二重呼び出しを避けるため、ここでは_opponent_room_ahead()を再度呼ばない)。"""
    idx = _SRC.index('if _lat_dec.branch == "A_rescue_relaxed":')
    snippet = _SRC[idx:idx + 500]
    assert "self._opponent_room_ahead(" not in snippet
    assert "_room_min" in snippet
    assert '"[REVERSE-ROOM-CHECK]' in snippet


def test_reverse_room_check_log_does_not_recompute_room_regression():
    """回帰防止(107節案C、非冗長性): _opponent_room_ahead()の呼び出しは
    1周期につき1回のみ(update()呼び出し前)であり、A_rescue_relaxed確定後の
    ログ地点では再計算しない。"""
    assert _SRC.count("self._opponent_room_ahead(") == 1


def test_reverse_room_check_does_not_affect_locked_side_regression():
    """回帰防止: [REVERSE-ROOM-CHECK]ログ自体は_lockedの決定
    (_lat_dec.side_overrideの採用)より後に出力され、ログ出力そのものは
    判定へ影響しないことをソース上の出現順で確認する(root causeの
    room先読み計算自体は107節案Cでupdate()呼び出し前へ移動済み、
    下記test_new_side_room_blocked_computed_before_updateで別途検証)。"""
    idx_locked = _SRC.index("_locked = _lat_dec.side_override")
    idx_room_check = _SRC.index('if _lat_dec.branch == "A_rescue_relaxed":')
    assert idx_locked < idx_room_check


# ---------------------------------------------------------------------------
# 107節案C(103節Phase 1): new_side_room_blockedの配線を構造的に検証
# ---------------------------------------------------------------------------

def test_new_side_room_blocked_computed_before_update():
    """核心: _opponent_room_ahead()の呼び出しは_lat_ttc.update()より前に
    実行され、その結果がveto入力として渡されることをソース上の出現順で
    確認する(76/79節のfrozen-ey lookahead問題とは異なり、これは_ot_side
    という現在値ベースの反転先を毎周期再計算するだけで、先の位置における
    自車自身の状態を仮定しない)。"""
    idx_room = _SRC.index("self._opponent_room_ahead(")
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_room < idx_update


def test_new_side_room_blocked_passed_to_update_call():
    idx_update = _SRC.index("_lat_dec = self._lat_ttc.update(")
    # 2026-07-22追加(157節): new_side_curvature_override引数が1行増えたため
    #   窓を700→800へ拡大(検証対象そのものは無変更)。
    snippet = _SRC[idx_update:idx_update + 800]
    assert "new_side_room_blocked=_new_side_room_blocked" in snippet


def test_new_side_room_blocked_reuses_along_min_width_no_new_magic_number():
    """②非冗長性: 閾値は既存のself._along_min_width(along_min_width、既定1.45m)
    を再利用し、新規の数値を導入しない。"""
    idx = _SRC.index("_new_side_room_blocked = (")
    snippet = _SRC[idx:idx + 200]
    assert "self._along_min_width" in snippet


def test_new_side_room_blocked_fails_open_when_ot_side_is_zero():
    """回帰: self._ot_side==0(まだOVERTAKING側が確定していない)の場合、
    -self._ot_sideは意味を持たないため、room先読み自体を計算せず
    (None, None, 0)・new_side_room_blocked=Falseとなる(fail-open)。"""
    idx = _SRC.index("_room_min, _room_wp, _room_n = (")
    snippet = _SRC[idx:idx + 400]
    assert "if self._ot_side != 0 else (None, None, 0)" in snippet
