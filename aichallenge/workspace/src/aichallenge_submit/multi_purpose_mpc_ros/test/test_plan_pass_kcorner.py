"""Unit tests for _plan_pass's k_corner veto width-awareness fix (47節, 2026-07-14).

mpc_controller.py imports rclpy/autoware message types at module scope, which are
not installed in this test environment. We therefore extract just the methods
under test via AST from the real source file and bind them to a minimal mock
`self`, so the ACTUAL production code (not a hand-written mirror) is exercised.
"""
import ast
import os
import types

import numpy as np
import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")


def _extract_methods(names):
    with open(_SRC_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    found[item.name] = ast.get_source_segment(src, item)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"methods not found in {_SRC_PATH}: {missing}")
    return found


_METHOD_NAMES = ["_plan_pass", "_side_blocked_by_other_car", "_plan_obs_log",
                 "_room_debounce_ok", "_room_to_wall"]
_METHODS_SRC = _extract_methods(_METHOD_NAMES)
_NS = {"np": np}
for _name, _src in _METHODS_SRC.items():
    exec(compile(_src, f"<{_name}>", "exec"), _NS)


class WP:
    def __init__(self, ub, lb, kappa=0.0):
        self.ub = ub
        self.lb = lb
        self.kappa = kappa


class _RefPath:
    def __init__(self, waypoints, circular=True):
        self.waypoints = waypoints
        self.circular = circular
        self.length = 1000.0


class _Model:
    def __init__(self, wp_id):
        self.wp_id = wp_id
        # 2026-07-17追加(86/87節検証ロギング用): _plan_obs_log/[PLAN-VETO]が
        # self._mpc.model.safety_marginを参照するため、テスト用ダミー値を用意する
        # (幅判定ロジック自体には影響しない、ログ出力のみで使われる値)。
        self.safety_margin = 1.626


class _MPCStub:
    def __init__(self, wp_id):
        self.model = _Model(wp_id)
        # 2026-07-17追加(86/87節): safety_margin_overrideも同様にログ出力でのみ
        # 参照される(Noneならmodel.safety_marginへフォールバック、本番と同じ優先順位)。
        self.safety_margin_override = None


def make_self(waypoints, wp_id=0, engage_debounce=1):
    """engage_debounce既定値1: 55節で追加したmin-width vetoのデバウンス(事象C対策、
    2026-07-14)を1回の評価で即確定させ、既存テストが検証している「幅の閾値判定」自体を
    デバウンスのノイズなしで単発比較できるようにする。デバウンス自体の挙動は
    test_room_debounce_*で別途検証する。"""
    m = types.SimpleNamespace()
    m._reference_path = _RefPath(waypoints)
    m._mpc = _MPCStub(wp_id)
    n = len(waypoints)
    m._wp_s_cum = np.arange(n, dtype=float)  # 1m/wp間隔の単純モデル
    m._opp_obstacle_speed = 1.67  # 6km/h
    m._opp_min_closing = 0.7
    m._v_pot = 4.17
    m._ot_block_half = 0.4
    m._ot_pass_clear = 3.0
    m._ot_t_lateral = 3.0
    m._ot_pass_block_kappa = 0.3
    m._along_lane_need = 1.85
    m._along_min_width = 1.45  # 2026-07-14追加: k_corner veto/min-width vetoが参照(0714-03修正)
    m._ot_engage_debounce = engage_debounce
    m._plan_obs_prev_result = None
    m._plan_obs_log_count = 0
    m._plan_moving_log_count = 0
    m._plan_fail_prev_reason = None  # 2026-07-14追加: [PLAN-VETO] MIN-WIDTH FAIL間引き用
    m._plan_fail_log_count = 0
    m._plan_room_ok_count = 0        # 2026-07-14追加: min-width vetoデバウンス状態
    m._plan_room_prev_vid = None
    m._plan_room_prev_side = None
    # 190-7節(2026-07-26追加): 反対側フォールバック専用の独立デバウンス状態。
    m._plan_room_ok_count_by_key = {}
    m._plan_room_prev_vid_by_key = {}
    m._plan_room_prev_side_by_key = {}
    m.get_logger = lambda: types.SimpleNamespace(info=lambda *a, **k: None,
                                                  warn=lambda *a, **k: None)
    m._plan_pass = types.MethodType(_NS["_plan_pass"], m)
    m._side_blocked_by_other_car = types.MethodType(_NS["_side_blocked_by_other_car"], m)
    m._plan_obs_log = types.MethodType(_NS["_plan_obs_log"], m)
    m._room_debounce_ok = types.MethodType(_NS["_room_debounce_ok"], m)
    m._room_to_wall = types.MethodType(_NS["_room_to_wall"], m)
    return m


def make_scan(fwd_ds, fwd_lat, vopp, fwd_vid="d3", fwd_wp=5, cars=None):
    return {"fwd_ds": fwd_ds, "fwd_vopp": vopp, "fwd_vid": fwd_vid, "fwd_wp": fwd_wp,
            "fwd_lat": fwd_lat, "fwd_dlat": abs(fwd_lat), "cars": cars or []}


def test_wide_track_opponent_offset_does_not_veto_the_open_side():
    """0713-06 wp168再現: 相手が右寄り(fwd_lat=-2.0)、コーナーは広いtrack上(ub/lb=4.0m)。
    左(lf=5.6m)は本来広く空いているのに、旧実装(kappa閾値のみの一律veto)ならk_corner>0で
    問答無用にvetoしていた(Rfree>Lfreeなのに狭い右へ誤って追い込まれていた実測と一致)。
    修正後は実測空き幅が十分あるためvetoされず、正しく左が選ばれる。"""
    wps = [WP(ub=4.0, lb=-4.0, kappa=(0.5 if i == 1 else 0.0)) for i in range(20)]
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=-2.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == 1  # 左(lf=5.6m)


def test_genuinely_narrow_corner_still_vetoes_that_side():
    """回帰確認: track自体がコーナーで本当に狭い(ub=1.0m<along_min_width)場合は
    従来通りvetoし、右を選ぶ。"""
    wps = [WP(ub=(1.0 if i == 1 else 4.0), lb=-4.0, kappa=(0.5 if i == 1 else 0.0))
           for i in range(20)]
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == -1  # 右


def test_no_corner_ahead_no_veto_applies():
    """コーナー(高kappa)が窓内に存在しない場合、k_corner vetoは一切発動しない(回帰)。"""
    wps = [WP(ub=4.0, lb=-4.0, kappa=0.0) for _ in range(20)]  # kappa=0のみ
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=-2.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == 1  # lf=5.6 > rf=1.6、veto無しでも自然にこちらが選ばれる


def test_veto_boundary_exactly_at_along_min_width_does_not_veto():
    """境界値: コーナー地点の実測幅(左右対称セットアップで比較)がalong_min_width
    (1.45m、2026-07-14修正: k_corner vetoの閾値をalong_lane_needから変更。障害物分岐
    は低相対速度ですり抜ける場面のため、並走継続用の余裕ではなく物理下限を見るのが
    適切、0714-03実測に基づく)ちょうどの場合はveto対象外(`<`厳密比較なので、ちょうど
    の値は「まだ十分」として扱う)。左右対称(ub=-lb, fwd_lat=0)にして、幅の絶対値では
    なくveto発動の有無だけが結果を左右するように分離する。"""
    # lf_i = rf_i = ub-0.4 = 1.45 となるよう ub=1.85(対称)
    wps = [WP(ub=(1.85 if i == 1 else 4.0), lb=(-1.85 if i == 1 else -4.0),
              kappa=(0.5 if i == 1 else 0.0)) for i in range(20)]
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == 1  # lf==rf(タイ)でvetoされなければ既定のlf>=rfでleftが選ばれる


def test_veto_boundary_just_below_along_min_width_vetoes():
    """境界値: along_min_width(1.45m)をわずかに下回るとk_corner vetoが発動する
    (同じ左右対称セットアップ)。この対称セットアップでは左右とも実測1.44m(<1.45m)
    しかなく、k_corner vetoで左が-1e9になった後も「相対的に大きい」というだけで
    右(1.44m)を選んでいた(0714-02実測で確認した①のバグそのもの)。新設のmin-width
    vetoにより、残された側も絶対的に狭ければ engage 自体をしない(ok=False)のが
    正しい挙動になった。"""
    wps = [WP(ub=(1.84 if i == 1 else 4.0), lb=(-1.84 if i == 1 else -4.0),
              kappa=(0.5 if i == 1 else 0.0)) for i in range(20)]
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert ok is False
    assert side == 0
    assert m._dbg_plan_reason == "narrow"


def test_0714_02_wp166_style_scenario_engages_with_along_min_width_threshold():
    """0714-02実測再現(wp166): planLf=1.6・planRf=-1e9(k_corner veto)。ub=2.0固定
    (→lf=1.6が全区間で一定)、lbはindex1のみ-2.0(→そこだけrf=1.6<1.85でk_corner veto
    発動、rf_min=-1e9)、他は-4.4(rf=4.0、広い)。
    2026-07-14再修正(0714-03実測、ユーザー指摘): 障害物分岐(低相対速度ですり抜ける
    場面)の閾値をalong_lane_need(1.85m、並走継続用)からalong_min_width(1.45m、
    物理下限)へ変更した結果、lf=1.6(>=1.45)は物理的に十分な幅とみなされ、正しく
    engageするようになった(0714-03でイン/アウト両方が締め出され完全停止していた
    問題への対処)。"""
    wps = [WP(ub=2.0, lb=(-1.7 if i == 1 else -4.4), kappa=(-0.5 if i == 1 else 0.0))
           for i in range(20)]
    m = make_self(wps, wp_id=0)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m._plan_pass(scan)
    assert m._dbg_plan_lf == pytest.approx(1.6)
    assert m._dbg_plan_rf == -1e9
    assert ok is True
    assert side == 1


def test_min_width_veto_boundary_independent_of_kcorner():
    """min-width vetoの境界値をk_corner非発動(kappa=0、窓内にコーナーなし)の単純な
    非対称コリドーで検証する。右(rf=0.6固定、狭い)より広い左(lf)が自然に勝つ設定で、
    lfがちょうどalong_min_width(1.45m)のケースはengageし、わずかに下回るケースは
    vetoされる。"""
    # lf = ub-0.4 = 1.45 となるよう ub=1.85、rf = (0-0.4)-lb = 0.6 となるよう lb=-1.0(常に狭い)
    wps_ok = [WP(ub=1.85, lb=-1.0, kappa=0.0) for _ in range(20)]
    m_ok = make_self(wps_ok, wp_id=0)
    scan_ok = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m_ok._plan_pass(scan_ok)
    assert m_ok._dbg_plan_lf == pytest.approx(1.45)
    assert ok is True
    assert side == 1

    wps_fail = [WP(ub=1.849, lb=-1.0, kappa=0.0) for _ in range(20)]
    m_fail = make_self(wps_fail, wp_id=0)
    scan_fail = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0)
    ok, side, _req = m_fail._plan_pass(scan_fail)
    assert ok is False
    assert side == 0
    assert m_fail._dbg_plan_reason == "narrow"


def test_room_debounce_requires_consecutive_ok_readings_before_engaging():
    """事象C対策(2026-07-14): roomがちょうど閾値付近でも、engage_debounce回連続で
    条件を満たすまでengageしない。同一対象車・同一側であることを前提にカウントする。"""
    wps = [WP(ub=2.25, lb=-1.0, kappa=0.0) for _ in range(20)]  # lf=1.85(境界ぴったり)
    m = make_self(wps, wp_id=0, engage_debounce=3)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3")
    for _ in range(2):  # debounce未満の間はengageしない
        ok, side, _req = m._plan_pass(scan)
        assert ok is False
        assert side == 0
    ok, side, _req = m._plan_pass(scan)  # 3回目でようやく成立
    assert ok is True
    assert side == 1


def test_room_debounce_resets_when_vehicle_id_changes():
    """回帰: 対象車(vid)が変わればデバウンスカウントは0からやり直す(案Bと同じ考え方)。"""
    wps = [WP(ub=2.25, lb=-1.0, kappa=0.0) for _ in range(20)]
    m = make_self(wps, wp_id=0, engage_debounce=3)
    scan_a = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3")
    m._plan_pass(scan_a)
    m._plan_pass(scan_a)
    scan_b = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d4")  # 対象車が切り替わった
    ok, side, _req = m._plan_pass(scan_b)
    assert ok is False  # カウントがリセットされ、1回目扱いになる


def test_room_debounce_resets_on_a_single_narrow_reading():
    """回帰: 途中で1回でも幅不足になれば、以降どれだけ広くてもカウントは0から再スタート
    (単発の広い読みでチャーンを再発火させない)。"""
    wps_ok = [WP(ub=2.25, lb=-1.0, kappa=0.0) for _ in range(20)]   # lf=1.85(OK, >=1.45)
    wps_bad = [WP(ub=1.7, lb=-1.0, kappa=0.0) for _ in range(20)]   # lf=1.3(NG, <1.45)
    m_ok = make_self(wps_ok, wp_id=0, engage_debounce=3)
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3")
    m_ok._plan_pass(scan); m_ok._plan_pass(scan)  # count=2まで積む
    assert m_ok._plan_room_ok_count == 2
    m_bad = make_self(wps_bad, wp_id=0, engage_debounce=3)
    m_bad._plan_room_ok_count = 2  # 直前まで広かった状態を人為的に再現
    m_bad._plan_room_prev_vid = "d3"; m_bad._plan_room_prev_side = 1
    ok, side, _req = m_bad._plan_pass(scan)
    assert ok is False
    assert m_bad._plan_room_ok_count == 0  # 1回の幅不足でカウントがリセットされる


def test_0714_03_corner_replay_engages_3s_earlier_than_historical_baseline():
    """0714-03実測の遡及リプレイ(第3/第5コーナー相当、wp266-281): 実ログに記録された
    (side, room)系列をそのまま_room_debounce_okへ通し、along_min_width(1.45m)閾値の
    下では実際の履歴(t=810.66、系列末尾のindex29)より約3.7秒早い時点(index9、
    t=806.96相当)でengageが成立することを確認する。旧along_lane_need(1.85m)閾値
    ではindex9時点のroom(1.51-1.60m)は不十分でengageできず、実際に記録された通り
    index29までblockされていたはずである(対比として検証)。"""
    m_new = make_self([WP(ub=4.0, lb=-4.0)], wp_id=0, engage_debounce=8)
    # 実測(0714-03, t=805.67〜810.59)の(side, room)系列。room=lf/rf(勝った側)の実測値。
    seq = [(-1, 1.63), (-1, 1.56), (1, 1.51), (1, 1.54), (1, 1.51), (1, 1.54),
           (1, 1.56), (1, 1.59), (1, 1.60), (1, 1.57), (1, 1.60), (1, 1.61),
           (1, 1.56), (1, 1.59), (1, 1.61), (1, 1.62), (1, 1.62), (1, 1.58),
           (1, 1.58), (1, 1.58), (-1, 0.81), (-1, 0.83), (-1, 0.85), (-1, 0.88),
           (-1, 0.89), (-1, 0.92), (-1, 1.02), (1, 1.85), (1, 1.83), (1, 1.88)]
    engage_idx_new = next(i for i, (s, r) in enumerate(seq)
                           if m_new._room_debounce_ok("d3", s, r, need=1.45))
    assert engage_idx_new == 9  # t=806.96相当、実際の履歴(index29)より20周期早い

    m_old = make_self([WP(ub=4.0, lb=-4.0)], wp_id=0, engage_debounce=8)
    engage_idx_old = next(
        (i for i, (s, r) in enumerate(seq)
         if m_old._room_debounce_ok("d3", s, r, need=1.85)), None)
    # 旧閾値(1.85m)では、この系列内では一度もroomが1.85以上に届かないため
    # (最後のindex29のroom=1.88のみ届くが、直前でside反転しておりcount不足)、
    # 実際の履歴通り系列の最後まで一度もengageできない。
    assert engage_idx_old is None


def test_no_position_information_declines_to_engage():
    """回帰確認: fwd_latがNoneの場合は安全側に倒してengageしない。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0)
    scan = {"fwd_ds": 6.0, "fwd_vopp": 0.0, "fwd_vid": "d3", "fwd_wp": None,
            "fwd_lat": None, "fwd_dlat": None, "cars": []}
    ok, side, _req = m._plan_pass(scan)
    assert ok is False
    assert side == 0


# ---------------------------------------------------------------------------
# K-check(_side_blocked_by_other_car)のneed引数(フローチャートギャップ③、2026-07-14)
# ---------------------------------------------------------------------------

def test_k_check_need_defaults_to_along_lane_need_regression():
    """回帰: need省略時は従来通りalong_lane_need(1.85m)が既定値のまま。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0)
    # side=1(左)、room=1.6、2台目(c2)はcombined=1.6になるよう配置(c_room=1.6)
    scan = {"cars": [(3.0, 2.0, 0.0, 2.0, "c2", 0)]}
    blocked = m._side_blocked_by_other_car(scan, side=1, target_vid="d3",
                                            ds_end=6.0, room=1.6, wp_o=0)
    assert blocked is True   # 1.6 < along_lane_need(1.85) → ブロックされる


def test_k_check_need_along_min_width_no_longer_blocks_same_scenario():
    """59節と同じalong_min_width(1.45m)をneedへ明示的に渡すと、同じ状況
    (combined=1.6m)でもブロックされなくなる。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0)
    scan = {"cars": [(3.0, 2.0, 0.0, 2.0, "c2", 0)]}
    blocked = m._side_blocked_by_other_car(scan, side=1, target_vid="d3",
                                            ds_end=6.0, room=1.6, wp_o=0,
                                            need=1.45)
    assert blocked is False  # 1.6 >= along_min_width(1.45) → ブロックされない


def test_obstacle_branch_plan_pass_uses_along_min_width_for_k_check():
    """統合確認: _plan_passの障害物分岐から呼ばれる際、K-checkにはalong_min_width
    (1.45m)が渡され、59節のmin-width veto緩和と一貫することを確認する。
    2台目が室=1.6m相当でしか塞いでいない場合、min-width veto(1.45m基準)は通過し、
    K-checkも同じ1.45m基準のため、engageが成立する。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0)
    # 対象車(d3)はlat=0(側選択に影響しない)、2台目(c2)はlat=2.0で左側room=1.6相当を塞ぐ
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=0.0, fwd_vid="d3",
                      cars=[(3.0, 2.0, 0.0, 2.0, "c2", 0)])
    ok, side, _req = m._plan_pass(scan)
    assert ok is True
    assert side == 1


# ---------------------------------------------------------------------------
# 「走行中の相手」分岐(moving-opponent branch)のk_corner veto幅盲点
# (水平展開スキャン第4弾、2026-07-14)
#
# 障害物分岐(vopp<obstacle_speed)のk_corner vetoは既に「コーナー地点の実測空き幅
# (相手の実際の位置を反映)を見てから初めてvetoする」よう修正済みだったが、兄弟分岐
# である「走行中の相手」分岐(vopp>=obstacle_speedかつclosingが小さい)のk_corner veto
# は、曲率+タイミングのみで一律vetoする旧ロジックのまま取り残されていた。
#
# 正直な開示: _plan_pass自身のコード(1982-2009行目付近)には既に「2026-07-13の
# ブランチ統合により、この分岐は理論上到達不能なはず」というコメントと、到達時に
# 必ず出る[PLAN-MOVING-ENTER] WARNログが存在する。以下のテストで数式的に確認する
# 通り、moving-opponent分岐はclosing<=opp_min_closingが常に成立してしまうため、
# _plan_pass自身の内部ロジックによりk_corner veto行へは現状決して到達しない(外部の
# 呼び出しゲートに頼らずとも、_plan_pass単体で自己完結的に不到達)。したがってこの
# 修正は現在のライブ挙動を一切変えない。将来ブランチ統合ロジックが変わり到達可能に
# なった場合に備えた防御的修正であり、コード自体がまだ削除されていない(削除是非は
# 実走ログでの再確認待ちと明記されている)ため、内部の閾値盲点だけ先に塞いでおく。
# ---------------------------------------------------------------------------

def test_moving_branch_is_unreachable_via_plan_pass_closing_check_regression():
    """回帰確認(dead-code状態の固定化): vopp>=obstacle_speedかつ(v_pot-vopp)<=
    opp_min_closingで「走行中の相手」分岐へ入っても、closing=v_pot-max(vopp,v_seg)は
    数式上必ずopp_min_closing以下になるため、k_corner veto行へ到達する前に
    reason=closing_raw/closing_segでFalseが返る。このテストが失敗した場合、
    分岐選択ロジックか閾値の関係が変わり、moving分岐のk_corner vetoが実際に
    到達可能になった(=下のveto修正が初めて実効を持つようになった)ことを意味する。"""
    wps = [WP(ub=4.0, lb=-4.0) for _ in range(20)]
    m = make_self(wps, wp_id=0)
    # v_pot=4.17, opp_min_closing=0.7 -> vopp>=3.47で分岐条件(v_pot-vopp<=0.7)を満たす
    scan = make_scan(fwd_ds=6.0, fwd_lat=0.0, vopp=3.5)
    ok, side, _req = m._plan_pass(scan)
    assert ok is False
    assert side == 0
    assert m._dbg_plan_reason in ("closing_raw", "closing_seg")


def moving_branch_kcorner_veto(k_corner, d_corner, req_clear, lf_at_corner, rf_at_corner,
                                along_lane_need, left_room, right_room):
    """mpc_controller.py _plan_pass の「走行中の相手」分岐 k_corner veto(2026-07-14修正)
    の純粋ミラー。_plan_pass自体がclosingチェックにより現状到達不能なため(上のテスト
    参照)、実メソッドをend-to-endで通す経路が無く、この verbatim転記で検証する。"""
    if k_corner is not None and req_clear > d_corner:
        if (k_corner > 0.0 and lf_at_corner is not None
                and lf_at_corner < along_lane_need):
            left_room = -1e9
        elif (k_corner < 0.0 and rf_at_corner is not None
                and rf_at_corner < along_lane_need):
            right_room = -1e9
    return left_room, right_room


ALONG_LANE_NEED = 1.85


def test_moving_branch_mirror_wide_corner_does_not_veto():
    """相手が既に避けていて実測lf_at_cornerが十分広い場合はvetoしない
    (障害物分岐0713-06 wp168修正と同型の確認)。"""
    lr, rr = moving_branch_kcorner_veto(
        k_corner=0.5, d_corner=5.0, req_clear=10.0,
        lf_at_corner=5.6, rf_at_corner=1.6,
        along_lane_need=ALONG_LANE_NEED, left_room=5.6, right_room=1.6)
    assert lr == 5.6  # vetoされない


def test_moving_branch_mirror_narrow_corner_still_vetoes():
    """実測が本当に狭ければ従来通りveto。"""
    lr, rr = moving_branch_kcorner_veto(
        k_corner=0.5, d_corner=5.0, req_clear=10.0,
        lf_at_corner=1.0, rf_at_corner=4.0,
        along_lane_need=ALONG_LANE_NEED, left_room=1.0, right_room=4.0)
    assert lr == -1e9


def test_moving_branch_mirror_boundary_exactly_at_along_lane_need():
    """境界値: along_lane_needちょうどはveto対象外(`<`厳密比較)。"""
    lr, rr = moving_branch_kcorner_veto(
        k_corner=0.5, d_corner=5.0, req_clear=10.0,
        lf_at_corner=ALONG_LANE_NEED, rf_at_corner=4.0,
        along_lane_need=ALONG_LANE_NEED, left_room=ALONG_LANE_NEED, right_room=4.0)
    assert lr == ALONG_LANE_NEED


def test_moving_branch_mirror_boundary_just_below_along_lane_need_vetoes():
    """境界値: along_lane_needをわずかに下回るとveto。"""
    lr, rr = moving_branch_kcorner_veto(
        k_corner=0.5, d_corner=5.0, req_clear=10.0,
        lf_at_corner=ALONG_LANE_NEED - 0.01, rf_at_corner=4.0,
        along_lane_need=ALONG_LANE_NEED, left_room=ALONG_LANE_NEED - 0.01, right_room=4.0)
    assert lr == -1e9


def test_moving_branch_mirror_req_clear_within_reach_no_veto():
    """コーナー前に抜き切れる(req_clear<=d_corner)場合はvetoそのものが発動しない(回帰)。"""
    lr, rr = moving_branch_kcorner_veto(
        k_corner=0.5, d_corner=100.0, req_clear=10.0,
        lf_at_corner=0.5, rf_at_corner=4.0,
        along_lane_need=ALONG_LANE_NEED, left_room=0.5, right_room=4.0)
    assert lr == 0.5  # 実測は狭いがreq_clear<=d_cornerのためvetoされない
