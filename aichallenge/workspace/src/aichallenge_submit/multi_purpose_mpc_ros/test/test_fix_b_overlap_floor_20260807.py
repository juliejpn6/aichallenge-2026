"""Fix B(並走中オフセット床、2026-08-07)。

背景: design_docs/opp_lat_pred_overlap_guard_design_20260806.md §2。
Fix A'(opp_lat_pred根本修正)だけでは解決しない残存ノイズ・正当な相手の
急な横移動等に対する物理的な安全網として、並走中(縦方向オーバーラップ中)
はtarget_magを縮小させないオフセット床を追加する。ただしコリドー実測(壁)
は常に優先する(床がコリドーを突き破ることは原理的に発生しない)。

外部AIレビュー(6.9節・§2.2)で発見されたmust-fix 1(床は専用の新規状態
_ot_overlap_floor_magを使う、168節の既存フリーズ値_ot_last_valid_target_mag
と混同しない)を反映済みの設計をそのまま実装した。

configゲート`overtake.overlap_floor_enabled`(既定false)でON/OFF切替、
OFF時は_update_overlap_state()の呼び出しも含め早期returnし現行と完全に
ビット等価(段階導入、Fix Aとは独立ゲート)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと
同じ「ソーステキスト構造検証」+「ロジックのミラー実装による数値検証」の
方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG_SRC = _f.read()


# ---------------------------------------------------------------------------
# ①config/状態変数: 新規ゲート・パラメータが既定値で宣言されていること
# ---------------------------------------------------------------------------

def test_config_yaml_has_overlap_floor_keys():
    assert "overlap_floor_enabled:" in _CFG_SRC
    assert "overlap_margin_m:" in _CFG_SRC
    assert "overlap_corr_margin_m:" in _CFG_SRC


def test_state_vars_declared_with_safe_defaults():
    assert 'self._ot_overlap_floor_enabled = bool(\n' in _SRC
    assert '_otget("overlap_floor_enabled", False))' in _SRC
    assert "self._ot_overlapping = False" in _SRC
    assert "self._ot_overlap_floor_mag = None" in _SRC


# ---------------------------------------------------------------------------
# ②_update_overlap_state: ヒステリシス判定、データ欠損時は直前状態を維持
# ---------------------------------------------------------------------------

def _overlap_state_block():
    idx = _SRC.index("def _update_overlap_state(")
    idx_end = _SRC.index("def _apply_overlap_floor(", idx)
    return _SRC[idx:idx_end]


def test_update_overlap_state_uses_hysteresis():
    snippet = _overlap_state_block()
    assert "enter_thr = self._along_min_length + self._ot_overlap_margin_m" in snippet
    assert "exit_thr = self._along_min_length + self._ot_overlap_margin_m * 2.0" in snippet


def test_update_overlap_state_reuses_along_min_length_no_new_distance_constant():
    """footprint_risk判定と同じ物理的下限(along_min_length)を再利用し、
    新規の距離定数を導入しないことを確認する。"""
    snippet = _overlap_state_block()
    assert "self._along_min_length" in snippet


def test_update_overlap_state_none_input_preserves_prior_state():
    snippet = _overlap_state_block()
    assert "if opp_ds_now is None:" in snippet
    idx_if = snippet.index("if opp_ds_now is None:")
    idx_return = snippet.index("return", idx_if)
    body = snippet[idx_if:idx_return + 40]
    assert "return self._ot_overlapping" in body


def _update_overlap_state_mirror(overlapping, opp_ds_now, along_min_length=2.0, margin=0.5):
    if opp_ds_now is None:
        return overlapping
    enter_thr = along_min_length + margin
    exit_thr = along_min_length + margin * 2.0
    d = abs(opp_ds_now)
    return (d < exit_thr) if overlapping else (d < enter_thr)


def test_hysteresis_mirror_enter_narrower_than_exit():
    """侵入判定(enter)より解除判定(exit)を広く取り、境界でのチャタリングを
    防ぐことを数値的に確認する(along_min_length=2.0, margin=0.5前提:
    enter_thr=2.5m, exit_thr=3.0m)。"""
    # 未オーバーラップ状態でenter_thr未満に入るとTrueへ
    assert _update_overlap_state_mirror(False, 2.4) is True
    # 未オーバーラップ状態でenter_thr以上exit_thr未満(境界帯)はまだFalseのまま
    assert _update_overlap_state_mirror(False, 2.7) is False
    # 一度オーバーラップに入った後は、同じ2.7mでもexit_thr未満なのでTrue継続
    assert _update_overlap_state_mirror(True, 2.7) is True
    # exit_thr以上になって初めて解除される
    assert _update_overlap_state_mirror(True, 3.1) is False


def test_hysteresis_mirror_missing_data_preserves_state():
    assert _update_overlap_state_mirror(True, None) is True
    assert _update_overlap_state_mirror(False, None) is False


# ---------------------------------------------------------------------------
# ③_apply_overlap_floor: ゲートOFF時はビット等価、専用状態変数を使用、
#   corr_boundで再キャップ
# ---------------------------------------------------------------------------

def _apply_overlap_floor_block():
    idx = _SRC.index("def _apply_overlap_floor(")
    idx_end = _SRC.index("def _stuck_update_shuffle_cycle(", idx)
    return _SRC[idx:idx_end]


def test_gate_off_early_return_unchanged():
    snippet = _apply_overlap_floor_block()
    idx = snippet.index("if not self._ot_overlap_floor_enabled:")
    idx_end = snippet.index("\n", idx + 60)
    body = snippet[idx:idx_end]
    assert "return target_mag" in body


def test_uses_dedicated_floor_variable_not_last_valid_target_mag():
    """must-fix 1: 168節の既存フリーズ値(_ot_last_valid_target_mag)と混同
    せず、専用の新規状態(_ot_overlap_floor_mag)を使うことを確認する
    (docstring内の説明的言及は許容し、実際の変数参照/代入のみを検査する)。"""
    snippet = _apply_overlap_floor_block()
    assert "self._ot_overlap_floor_mag = max(" in snippet
    assert "self._ot_last_valid_target_mag" not in snippet


def test_floor_recapped_against_current_corr_bound():
    snippet = _apply_overlap_floor_block()
    assert "floor = min(floor, corr_bound - self._ot_overlap_corr_margin_m)" in snippet


def test_calls_update_overlap_state_internally():
    snippet = _apply_overlap_floor_block()
    assert "overlapping = self._update_overlap_state(opp_ds_now)" in snippet


def _apply_overlap_floor_mirror(target_mag, overlapping, floor_state, corr_bound,
                                 corr_margin=0.1):
    """_apply_overlap_floor()のミラー実装(ゲートON・overlapping=True前提の
    コア計算部分のみ、数値検証用)。"""
    floor_state = max(floor_state or 0.0, target_mag)
    floor = floor_state
    if corr_bound == corr_bound and corr_bound > 0.0:  # not NaN
        floor = min(floor, corr_bound - corr_margin)
    return max(target_mag, floor), floor_state


def test_floor_monotonic_non_decreasing_within_episode():
    """不変条件①: 床は同一並走エピソード内で単調非減少(下がらない)。"""
    floor_state = None
    result1, floor_state = _apply_overlap_floor_mirror(1.5, True, floor_state, 10.0)
    result2, floor_state = _apply_overlap_floor_mirror(0.8, True, floor_state, 10.0)
    # target_magが1.5→0.8へ下がっても、床が1.5を記憶しているため結果は1.5のまま
    assert result1 == 1.5
    assert result2 == 1.5


def test_floor_never_punches_through_current_corridor_wall():
    """不変条件②: 床適用後のtarget_magは常に「現在のcorr_bound - マージン」
    以下(床がコリドーの壁を突き破ることは原理的に発生しない)。"""
    floor_state = 2.5  # 過去の広いコリドーで積み上がった高い床
    corr_bound_now = 1.0  # 今周期、コリドーが実際に狭まった
    result, _ = _apply_overlap_floor_mirror(0.3, True, floor_state, corr_bound_now,
                                             corr_margin=0.1)
    assert result <= corr_bound_now - 0.1 + 1e-9


def test_floor_only_raises_never_lowers_target_mag():
    """床は「増やす」方向にのみ効く(target_magを本来の値未満へは絶対に
    下げない)ことを確認する。"""
    floor_state = 0.5
    result, _ = _apply_overlap_floor_mirror(1.2, True, floor_state, 10.0)
    assert result >= 1.2


# ---------------------------------------------------------------------------
# ④呼び出し箇所: OVERTAKING分岐・STOPPING/proactive-bias分岐の2箇所で、
#   lateral_target確定の直前に呼ばれていること
# ---------------------------------------------------------------------------

def test_called_before_overtaking_lateral_target_assignment():
    idx_call = _SRC.index(
        "_target_mag = self._apply_overlap_floor(\n"
        "                    _target_mag, _opp_ds_now, _corr_bound)")
    idx_assign = _SRC.index(
        "self._mpc.lateral_target = float(self._ot_side) * _target_mag", idx_call)
    assert 0 < idx_assign - idx_call < 200


def test_called_before_stopping_lateral_target_assignment():
    idx_call = _SRC.index(
        "_target_mag = self._apply_overlap_floor(\n"
        "                    _target_mag, _fwd_ds, _corr_bound)")
    idx_assign = _SRC.index(
        "self._mpc.lateral_target = float(_eval.plan_side) * _target_mag", idx_call)
    assert 0 < idx_assign - idx_call < 200


def test_total_call_count_matches_two_known_sites():
    n_calls = _SRC.count("self._apply_overlap_floor(")
    assert n_calls == 2, (
        f"想定していた2箇所(OVERTAKING分岐・STOPPING分岐)から数が変わっている"
        f"(現在{n_calls}箇所)。新しい適用箇所が追加/削除された場合はこの"
        "テスト自体の更新も必要。")


# ---------------------------------------------------------------------------
# ⑤診断ログ: [OVERLAP-FLOOR]がエッジトリガー(床が実際に効いた時のみ)で
#   発火すること
# ---------------------------------------------------------------------------

def test_overlap_floor_log_is_edge_triggered():
    idx = _SRC.index('f"[OVERLAP-FLOOR] side=')
    snippet = _SRC[idx - 300:idx]
    assert "if target_mag > _before:" in snippet


def test_overlap_floor_log_fields():
    idx = _SRC.index('f"[OVERLAP-FLOOR] side=')
    snippet = _SRC[idx:idx + 400]
    assert "floor=" in snippet
    assert "target_mag_before=" in snippet
    assert "target_mag_after=" in snippet
    assert "corr_bound=" in snippet
    assert "wp=" in snippet


# ---------------------------------------------------------------------------
# ⑥リセット統合: _reset_ot_episode_tracking_state()がFix Bの状態も
#   リセットし、既存4箇所の重複実装から呼ばれていること
# ---------------------------------------------------------------------------

def _reset_helper_body():
    idx = _SRC.index("def _reset_ot_episode_tracking_state(self)")
    idx_end = _SRC.index("\n    def ", idx + 10)
    return _SRC[idx:idx_end]


def test_reset_helper_resets_overlap_state_and_floor():
    snippet = _reset_helper_body()
    assert "self._ot_overlapping = False" in snippet
    assert "self._ot_overlap_floor_mag = None" in snippet


def test_reset_helper_does_not_touch_side_commit_state():
    """責務分離: 本ヘルパーは横速度推定・並走関連の状態のみに責務を限定し、
    _ot_side/_ot_side_locked/_ot_giveup_count/_ot_room_exhausted_count等
    (呼び出し元で既に個別に正しくリセット済み)には一切触れない。"""
    snippet = _reset_helper_body()
    assert "self._ot_side " not in snippet
    assert "self._ot_side_locked" not in snippet
    assert "self._ot_giveup_count" not in snippet
    assert "self._ot_room_exhausted_count" not in snippet


def test_reset_helper_called_from_four_known_sites():
    """側反転・rescue側反転・新規エンゲージ・STUCK復帰(_reset_ot_side_for_
    fresh_replan経由)の計4箇所から呼ばれていることを確認する。"""
    n_calls = _SRC.count("self._reset_ot_episode_tracking_state()")
    assert n_calls == 4, (
        f"想定していた4箇所から数が変わっている(現在{n_calls}箇所)。"
        "新しい離脱経路が追加/削除された場合はこのテスト自体の更新も必要。")
