"""room_exhausted giveup直前の反対側最終救済(247節、2026-07-30)。

背景: ユーザーが0730_03予選ログのwp172付近(1周目d3・2回目通過d2)で、
OVERTAKING中にside=1の先読みroomが尽きて一時STOPPING(cdゲート不成立で
約1〜3秒の完全停止)→再エンゲージ→switchbackで側反転、という一連の
「一時停止を挟んだオーバーテイク」を確認し、「反対サイドの状態を確認して
継続できるなら継続すべき」と指摘した。

根本原因: `_room_exhausted`(既存側の先読みroomが_ot_giveup_cycles連続で
非正)が成立すると、反対側を一切確認せず即座に`_side_blocked=True`→
`state=STOPPING`へ落ちていた(190-7節で`_plan_pass`の初回エンゲージ向けに
追加した「SIDE-FALLBACK」と同じ発想が、OVERTAKING継続中のこの離脱経路には
適用されていなかった)。

対処: giveup合流の直前に、反対側の安全性を確認する「最終救済」チェックを
追加した。ハンチング防止のため、LateralTTCMonitorが既に持つ`has_switched`
ラッチ(1エンゲージにつき1回のみ反転)をそのまま共有し、新しいラッチは
一切追加していない。安全条件(is_side_by_side/new_side_wall_blocked/
new_side_room_blocked/new_side_offset_blocked/fwd_ds_overlap_risk)は
通常のswitchback(branch=A/A_dlat)と共通のまま緩和していない。

ユーザー指示による方針転換: 実測(0730_03 wp172、t=1410.18)でopp_space=
2.207mがswitchback_space_m(2.35m)をわずかに下回り不発になることを確認した
上で、「一旦オーバーテイクを試みた後の反対サイド判定は緩めてよい」
「通常のオーバーテイク判定の条件とは分けてほしい」との指示を受け、この
救済経路専用の閾値としてgiveup_space_m(1.85m、既存定数、新規の数値は
導入しない)を使うことにした。通常のswitchback_space_m判定(branch=A/
A_dlat等、lateral_ttc_monitor.py側)には一切触れていない。

mpc_controller.pyはrclpy非依存のため直接importできず、他の巨大メソッド
関連テスト群(test_ot_room_exhausted_giveup.py等)と同じ方針(ソーステキスト
による構造的検証)を用いる。lateral_ttc_monitor.pyのforce_rescue_switch()は
ROS非依存の純粋なクラスメソッドのため、直接importして実際に実行し動作を
検証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

from multi_purpose_mpc_ros.lateral_ttc_monitor import LateralTTCMonitor


# ---------------------------------------------------------------------------
# force_rescue_switch(): 実際にexecして動作検証
# ---------------------------------------------------------------------------

def test_force_rescue_switch_sets_has_switched():
    mon = LateralTTCMonitor()
    assert mon.has_switched is False
    mon.force_rescue_switch()
    assert mon.has_switched is True


def test_force_rescue_switch_resets_space_and_curvature_trend():
    mon = LateralTTCMonitor()
    mon._prev_space = 1.23
    mon._space_ema = 1.23
    mon._v_corridor_ema = -0.5
    mon._shrink_run = 7
    mon._critical_curvature_run = 3
    mon.force_rescue_switch()
    assert mon._prev_space is None
    assert mon._space_ema is None
    assert mon._v_corridor_ema == 0.0
    assert mon._shrink_run == 0
    assert mon._critical_curvature_run == 0


def test_force_rescue_switch_resets_dlat_trend():
    """反転先で基準が変わるため、dlatトレンドもbranch=A/A_dlatの成立時と
    同様にリセットされることを確認する(古い縮小方向を残すと反転直後に
    同じトレンドで誤って再評価されるおそれがあるため)。"""
    mon = LateralTTCMonitor()
    mon._dlat_ema = 0.5
    mon._prev_dlat_ema = 0.6
    mon._v_dlat_ema = -0.3
    mon._dlat_shrink_run = 5
    mon.force_rescue_switch()
    assert mon._dlat_ema is None
    assert mon._prev_dlat_ema is None
    assert mon._v_dlat_ema == 0.0
    assert mon._dlat_shrink_run == 0


def test_force_rescue_switch_reset_set_matches_branch_a_dlat_success_path():
    """①非矛盾性: force_rescue_switch()がリセットするフィールド集合が、
    branch=A_dlat成立時(既存コード)のリセット集合と完全に一致することを
    ソーステキストで確認する(経路が違うだけで後始末は1種類のみ)。"""
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "lateral_ttc_monitor.py")
    with open(src_path) as f:
        src = f.read()
    idx_dlat_success = src.index('return self._decision(side_override=(-side), ttc_lat=ttc_dlat,\n'
                                  '                                           branch="A_dlat", **_diag)')
    idx_dlat_block_start = src.rindex("self.has_switched = True", 0, idx_dlat_success)
    dlat_snippet = src[idx_dlat_block_start:idx_dlat_success]

    idx_method = src.index("def force_rescue_switch(self) -> None:")
    idx_method_end = src.index("\n    def ", idx_method + 10)
    rescue_snippet = src[idx_method:idx_method_end]

    for field in ("self.has_switched = True", "self._prev_space = None",
                  "self._space_ema = None", "self._v_corridor_ema = 0.0",
                  "self._shrink_run = 0", "self._critical_curvature_run = 0",
                  "self._dlat_ema = None", "self._prev_dlat_ema = None",
                  "self._v_dlat_ema = 0.0", "self._dlat_shrink_run = 0"):
        assert field in dlat_snippet, f"branch=A_dlat側に{field}が見当たらない(前提が崩れている)"
        assert field in rescue_snippet, f"force_rescue_switch()に{field}が無い(非矛盾性違反)"


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線: 構造的ソーステキスト検証
# ---------------------------------------------------------------------------

def _rescue_block_snippet():
    """2026-08-09改訂(§45.3): 終端アンカーが`_side_blocked = _lat_dec.force_giveup
    or _room_exhausted`から`_side_blocked = ((_lat_dec.force_giveup and not
    _selflock_escape_override) or _room_exhausted)`へ変わった(自己ロック解除
    エスケープ、configゲート既定OFF時はビット等価)。"""
    idx = _SRC.index("if _room_exhausted and self._ot_room_exhausted_count == self._ot_giveup_cycles:")
    idx_end = _SRC.index("_side_blocked = ((_lat_dec.force_giveup and not _selflock_escape_override)")
    return _SRC[idx:idx_end]


def test_rescue_uses_giveup_space_m_not_switchback_space_m():
    """②非冗長性・ユーザー指示: この救済経路は通常のswitchback判定
    (switchback_space_m)とは明示的に分離された専用の緩和閾値
    (giveup_space_m、既存定数)を使うことを確認する。"""
    snippet = _rescue_block_snippet()
    assert "self._lat_ttc.giveup_space_m" in snippet
    assert "self._lat_ttc.switchback_space_m" not in snippet


def test_normal_switchback_thresholds_unaffected_in_lateral_ttc_monitor():
    """通常のswitchback判定(branch=A/A_dlat/A_rescue)がswitchback_space_mを
    引き続き使っており、247節の変更で緩められていないことを回帰確認する。"""
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "lateral_ttc_monitor.py")
    with open(src_path) as f:
        src = f.read()
    assert src.count("self.switchback_space_m") >= 3  # 通常branch=A/A_dlat/A_rescue分の参照


def test_rescue_requires_not_has_switched_and_not_side_by_side():
    """ハンチング防止: has_switched(1エンゲージ1回のラッチ)とis_side_by_sideは
    通常のswitchbackと共有し、この経路のためだけに緩和していないことを確認する。"""
    snippet = _rescue_block_snippet()
    assert "not _lat_dec.has_switched" in snippet
    assert "not _lat_dec.is_side_by_side" in snippet


def test_rescue_requires_all_new_side_blocked_checks_unrelaxed():
    """反対側の安全条件(壁/room/offset/縦距離)は通常のswitchbackと同一のまま
    緩和していないことを確認する。"""
    snippet = _rescue_block_snippet()
    for cond in ("not _new_side_wall_blocked", "not _new_side_room_blocked",
                 "not _new_side_offset_blocked", "not _fwd_ds_overlap_risk"):
        assert cond in snippet


def test_rescue_uses_corr_bound_ahead_for_opposite_side():
    """反対側の先読みroomをcorr_bound_ahead(既存関数、_lockedと対称に-_lockedへ
    適用するだけ)で確認していることを確認する(新規の指標を導入しない)。"""
    snippet = _rescue_block_snippet()
    assert "_opp_locked = -_locked" in snippet
    assert "self._corr_bound_ahead(_opp_locked)" in snippet


def test_rescue_success_resets_same_state_as_normal_switchback_side_flip():
    """救済成立時のmpc_controller.py側リセットが、通常のswitchback側反転
    (side_override成立時)と同一の状態変数集合であることを確認する
    (①非矛盾性: 側が変わったという事実への後始末は1種類のみ)。
    2026-08-07改訂(Fix B、design_docs...20260806.md §4): 個別の
    self._ot_last_valid_target_mag = None行は共通ヘルパー
    _reset_ot_episode_tracking_state()呼び出しへ統合された。"""
    snippet = _rescue_block_snippet()
    for field in ("self._ot_side_locked = _locked", "self._ot_alpha = 0.0",
                  "self._ot_room_exhausted_count = 0",
                  "self._reset_ot_episode_tracking_state()",
                  "self._ot_cleared = False", "self._ot_reacquire_count = 0"):
        assert field in snippet


def test_rescue_success_clears_room_exhausted_flag_to_skip_giveup():
    """救済成立時は_room_exhaustedをFalseへ戻し、直後の
    _side_blocked = _lat_dec.force_giveup or _room_exhausted へ影響しない
    (giveupへ合流しない)ことを確認する。"""
    snippet = _rescue_block_snippet()
    idx_rescued = snippet.index("if _room_rescued:")
    tail = snippet[idx_rescued:idx_rescued + 100]
    assert "_room_exhausted = False" in tail


def test_rescue_log_and_giveup_log_are_mutually_exclusive():
    """[OT-ROOM-EXHAUSTED-RESCUE](成功)と[OT-ROOM-EXHAUSTED](失敗、giveup合流)
    が同一if/elseブロックにあり、同じ周期には片方しか出ないことを確認する。"""
    snippet = _rescue_block_snippet()
    idx_rescued_true = snippet.index("_room_rescued = True")
    idx_if_rescued = snippet.index("if _room_rescued:")
    idx_else = snippet.index("else:", idx_if_rescued)
    assert "[OT-ROOM-EXHAUSTED-RESCUE]" in snippet[:idx_if_rescued]
    assert "[OT-ROOM-EXHAUSTED]" in snippet[idx_else:]
    assert idx_rescued_true < idx_if_rescued


def test_rescue_calls_force_rescue_switch_on_lat_ttc_instance():
    snippet = _rescue_block_snippet()
    assert "self._lat_ttc.force_rescue_switch()" in snippet
