"""Fix B(OVERTAKING中オフセット床、2026-08-07、§14で再設計)。

背景: design_docs/opp_lat_pred_overlap_guard_design_20260806.md §2/§14。
Fix A'(opp_lat_pred根本修正)だけでは解決しない残存ノイズ・正当な相手の
急な横移動等に対する物理的な安全網として、target_magを縮小させない
オフセット床を追加する。ただしコリドー実測(壁)は常に優先する(床が
コリドーを突き破ることは原理的に発生しない)。

**§14での再設計(2026-08-07)**: 当初はds(対象車との縦距離)ベースの
ヒステリシス判定(_update_overlap_state())で「並走中」を検知していたが、
CLAUDE.md §1.4準拠のオフライン反実仮想検証で、Fix Bを正当化した2つの
動機事例(18節衝突事例・0805-07慢性未達事例、計45サンプル)全てで
dsが3〜13m台に分布し、footprint_risk由来の閾値(2.5〜3.0m)には一度も
到達しないことが判明した(スコープ取り違え、外部AI[Gemini]レビューで
確認・裏付け済み)。実際の不具合はENGAGE直後の「接近しながらオフセットを
広げている」段階全体で起きていたため、判定をself._ot_state==
"OVERTAKING"全体(側が確定してから離脱するまでの全期間)へ広げた。
dsベース判定(_update_overlap_state())は撤去せずFix C(未実装、離脱保留、
「本当に真横にいる」という物理的近接性が本質)専用として温存する。

configゲート`overtake.overlap_floor_enabled`(既定false)でON/OFF切替、
OFF時は状態チェックも含め早期returnし現行と完全にビット等価
(段階導入、Fix Aとは独立ゲート)。

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
# ②_update_overlap_state: Fix C専用として温存(dsベースヒステリシス判定)
#   本体はFix Bからはもう呼ばれないが、ロジック自体の健全性は維持確認する
# ---------------------------------------------------------------------------

def _overlap_state_block():
    idx = _SRC.index("def _update_overlap_state(")
    idx_end = _SRC.index("def _apply_overlap_floor(", idx)
    return _SRC[idx:idx_end]


def test_update_overlap_state_reserved_for_fix_c():
    """§14改訂: Fix Bはこの判定を使わなくなった旨がdocstringに明記されて
    いることを確認する(将来の実装者が誤ってFix Bへ再結線しないため)。"""
    snippet = _overlap_state_block()
    assert "Fix C専用として温存" in snippet


def test_update_overlap_state_uses_hysteresis():
    snippet = _overlap_state_block()
    assert "enter_thr = self._along_min_length + self._ot_overlap_margin_m" in snippet
    assert "exit_thr = self._along_min_length + self._ot_overlap_margin_m * 2.0" in snippet


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
    enter_thr=2.5m, exit_thr=3.0m)。Fix C実装時に再利用される想定。"""
    assert _update_overlap_state_mirror(False, 2.4) is True
    assert _update_overlap_state_mirror(False, 2.7) is False
    assert _update_overlap_state_mirror(True, 2.7) is True
    assert _update_overlap_state_mirror(True, 3.1) is False


def test_hysteresis_mirror_missing_data_preserves_state():
    assert _update_overlap_state_mirror(True, None) is True
    assert _update_overlap_state_mirror(False, None) is False


# ---------------------------------------------------------------------------
# ③_apply_overlap_floor: ゲートOFF時はビット等価、専用状態変数を使用、
#   corr_boundで再キャップ、判定はself._ot_state=="OVERTAKING"
# ---------------------------------------------------------------------------

def _apply_overlap_floor_block():
    idx = _SRC.index("def _apply_overlap_floor(")
    idx_end = _SRC.index("def _stuck_update_shuffle_cycle(", idx)
    return _SRC[idx:idx_end]


def test_signature_no_longer_takes_opp_ds_now():
    """§14改訂: dsベース判定を撤去したため、引数からopp_ds_nowが消えて
    いることを確認する(回帰防止、旧シグネチャの復活を検知する)。"""
    assert "def _apply_overlap_floor(self, target_mag: float, corr_bound: float) -> float:" in _SRC


def test_gate_off_early_return_unchanged():
    snippet = _apply_overlap_floor_block()
    idx = snippet.index("if not self._ot_overlap_floor_enabled:")
    idx_end = snippet.index("\n", idx + 60)
    body = snippet[idx:idx_end]
    assert "return target_mag" in body


def test_scope_is_overtaking_state_not_ds_based():
    """§14の核心的な変更: 判定がself._ot_state=="OVERTAKING"であること。
    dsベースの_update_overlap_state()はもう呼ばれない。"""
    snippet = _apply_overlap_floor_block()
    assert 'if self._ot_state != "OVERTAKING":' in snippet
    assert "self._update_overlap_state(" not in snippet


def test_floor_reset_when_leaving_overtaking():
    """OVERTAKING以外の状態では床(_ot_overlap_floor_mag)をNoneへ戻し、
    次回エンゲージ時に前回エピソードの高い床値を持ち越さないことを
    確認する。"""
    snippet = _apply_overlap_floor_block()
    idx = snippet.index('if self._ot_state != "OVERTAKING":')
    idx_end = snippet.index("return target_mag", idx)
    body = snippet[idx:idx_end]
    assert "self._ot_overlap_floor_mag = None" in body


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


def _apply_overlap_floor_mirror(target_mag, ot_state, floor_state, corr_bound,
                                 corr_margin=0.1):
    """_apply_overlap_floor()の1:1ミラー実装(ゲートON前提、数値検証用)。
    §14改訂後: 判定はot_state=="OVERTAKING"かどうかのみ。"""
    if ot_state != "OVERTAKING":
        return target_mag, None
    floor_state = max(floor_state or 0.0, target_mag)
    floor = floor_state
    if corr_bound == corr_bound and corr_bound > 0.0:  # not NaN
        floor = min(floor, corr_bound - corr_margin)
    return max(target_mag, floor), floor_state


def test_floor_monotonic_non_decreasing_within_overtaking_episode():
    """不変条件①: 床は同一OVERTAKINGエピソード内で単調非減少(下がらない)。"""
    floor_state = None
    result1, floor_state = _apply_overlap_floor_mirror(1.5, "OVERTAKING", floor_state, 10.0)
    result2, floor_state = _apply_overlap_floor_mirror(0.8, "OVERTAKING", floor_state, 10.0)
    assert result1 == 1.5
    assert result2 == 1.5  # target_magが1.5→0.8へ下がっても床が記憶している


def test_floor_never_punches_through_current_corridor_wall():
    """不変条件②: 床適用後のtarget_magは常に「現在のcorr_bound - マージン」
    以下(床がコリドーの壁を突き破ることは原理的に発生しない)。"""
    floor_state = 2.5  # 過去の広いコリドーで積み上がった高い床
    corr_bound_now = 1.0  # 今周期、コリドーが実際に狭まった
    result, _ = _apply_overlap_floor_mirror(0.3, "OVERTAKING", floor_state, corr_bound_now,
                                             corr_margin=0.1)
    assert result <= corr_bound_now - 0.1 + 1e-9


def test_floor_only_raises_never_lowers_target_mag():
    """床は「増やす」方向にのみ効く(target_magを本来の値未満へは絶対に
    下げない)ことを確認する。"""
    floor_state = 0.5
    result, _ = _apply_overlap_floor_mirror(1.2, "OVERTAKING", floor_state, 10.0)
    assert result >= 1.2


def test_floor_inactive_outside_overtaking_state_mirror():
    """不変条件③: NORMAL/STOPPING等OVERTAKING以外の状態では床が一切
    働かず、target_magは無変更で返り、床状態もリセットされる。"""
    for state in ("NORMAL", "STOPPING"):
        result, floor_state = _apply_overlap_floor_mirror(0.3, state, 5.0, 10.0)
        assert result == 0.3
        assert floor_state is None


# ---------------------------------------------------------------------------
# ④呼び出し箇所: OVERTAKING分岐の1箇所のみ(STOPPING分岐からは撤去、
#   §14: opp_lat_predを参照しないためFix Bのノイズ対策が不要と判明)
# ---------------------------------------------------------------------------

def test_called_before_overtaking_lateral_target_assignment():
    idx_call = _SRC.index(
        "_target_mag = self._apply_overlap_floor(_target_mag, _corr_bound)")
    idx_assign = _SRC.index(
        "self._mpc.lateral_target = float(self._ot_side) * _target_mag", idx_call)
    assert 0 < idx_assign - idx_call < 200


def test_not_called_from_stopping_proactive_bias_branch():
    """§14改訂: STOPPING/proactive-bias分岐のtarget_magはopp_lat_predを
    一切参照しない固定小値+corr_boundクランプのみで構成され、Fix Bが
    対処すべきノイズ源が存在しないため、この分岐からは呼ばれなくなった
    ことを確認する(外部AI[Gemini]レビューで指摘・確認済み)。"""
    idx_stopping = _SRC.index('elif (self._ot_state == "STOPPING" and _eval is not None')
    idx_end = _SRC.index(
        "self._mpc.lateral_target = float(_eval.plan_side) * _target_mag", idx_stopping)
    snippet = _SRC[idx_stopping:idx_end]
    assert "self._apply_overlap_floor(" not in snippet


def test_total_call_count_matches_single_known_site():
    n_calls = _SRC.count("self._apply_overlap_floor(")
    assert n_calls == 1, (
        f"想定していた1箇所(OVERTAKING分岐のみ)から数が変わっている"
        f"(現在{n_calls}箇所)。新しい適用箇所が追加/削除された場合は"
        "このテスト自体の更新も必要。")


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


def test_reset_helper_resets_invalid_corr_count():
    """§14.3追加分: corr_bound無効連続カウンタもリセットされること。"""
    snippet = _reset_helper_body()
    assert "self._ot_overlap_floor_invalid_corr_count = 0" in snippet


def test_reset_helper_called_from_five_known_sites():
    """側反転・rescue側反転・新規エンゲージ・STUCK復帰(_reset_ot_side_for_
    fresh_replan経由)・STUCK突入時(_stuck_enter_wait_reverse、Fix C §3.4
    外部AIレビュー推奨4、2026-08-07追加)の計5箇所から呼ばれていることを
    確認する。"""
    n_calls = _SRC.count("self._reset_ot_episode_tracking_state()")
    assert n_calls == 5, (
        f"想定していた5箇所から数が変わっている(現在{n_calls}箇所)。"
        "新しい離脱経路が追加/削除された場合はこのテスト自体の更新も必要。")


# ---------------------------------------------------------------------------
# ⑦corr_bound無効タイムアウト(2026-08-07、§14.3、外部AI[別Claude]レビュー):
#   corr_bound無効(負転落/非有限)が既存のunlock_inf_cycles(H4-lite、
#   80周期≈2秒)を超えて連続したら床の適用自体を止める
# ---------------------------------------------------------------------------

def test_invalid_corr_count_state_declared_with_safe_default():
    assert "self._ot_overlap_floor_invalid_corr_count = 0" in _SRC


def test_timeout_reuses_existing_unlock_after_no_new_magic_number():
    """新規マジックナンバーを増やさず、既存のH4-lite上限(self._unlock_after
    = unlock_inf_cycles由来)を流用していることを確認する。"""
    snippet = _apply_overlap_floor_block()
    assert "self._ot_overlap_floor_invalid_corr_count > self._unlock_after" in snippet


def test_invalid_count_resets_to_zero_when_corr_becomes_valid():
    snippet = _apply_overlap_floor_block()
    idx_valid = snippet.index("if _corr_valid:")
    idx_else = snippet.index("else:", idx_valid)
    body = snippet[idx_valid:idx_else]
    assert "self._ot_overlap_floor_invalid_corr_count = 0" in body


def test_invalid_count_increments_only_in_invalid_branch():
    snippet = _apply_overlap_floor_block()
    idx_else = snippet.index("else:\n            self._ot_overlap_floor_invalid_corr_count += 1")
    assert idx_else >= 0  # インデックスが見つかること自体がアサーション


def test_floor_application_skipped_after_timeout():
    """タイムアウト超過時、床の適用(target_mag = max(target_mag, floor))を
    経由せずtarget_magをそのまま返すことを確認する。"""
    snippet = _apply_overlap_floor_block()
    idx_over = snippet.index(
        "if self._ot_overlap_floor_invalid_corr_count > self._unlock_after:")
    idx_return = snippet.index("return target_mag", idx_over)
    idx_next_target_mag_max = snippet.find("target_mag = max(target_mag, floor)", idx_over)
    # target_mag = max(target_mag, floor)より前でreturnしていること
    assert idx_return < idx_next_target_mag_max


def test_timeout_log_is_one_shot():
    """[OVERLAP-FLOOR-TIMEOUT]ログが発火した最初の周期だけの
    ワンショットであること(境界値+1での等価比較)。"""
    snippet = _apply_overlap_floor_block()
    assert (
        "if self._ot_overlap_floor_invalid_corr_count == self._unlock_after + 1:"
        in snippet)
    assert '"[OVERLAP-FLOOR-TIMEOUT]' in snippet


def _apply_overlap_floor_v3_mirror(target_mag, ot_state, corr_bound, floor_mag,
                                    invalid_count, unlock_after=80, corr_margin=0.1):
    """§14.3改訂後の_apply_overlap_floor()の1:1ミラー(数値検証用)。
    戻り値: (target_mag_after, floor_mag_after, invalid_count_after)"""
    if ot_state != "OVERTAKING":
        return target_mag, None, 0
    floor_mag = max(floor_mag or 0.0, target_mag)
    floor = floor_mag
    corr_valid = corr_bound == corr_bound and corr_bound is not None and corr_bound > 0.0
    if corr_valid:
        floor = min(floor, corr_bound - corr_margin)
        invalid_count = 0
    else:
        invalid_count += 1
        if invalid_count > unlock_after:
            return target_mag, floor_mag, invalid_count
    return max(target_mag, floor), floor_mag, invalid_count


def test_mirror_floor_disabled_after_prolonged_invalid_corr_bound():
    """corr_bound無効がunlock_after周期を超えて続くと、床の適用が止まる
    (target_magが持ち上げられなくなる)ことを数値的に確認する。"""
    floor_mag = 3.0  # 高いピークが既に記録されている
    invalid_count = 0
    target_mag = 0.1
    unlock_after = 80
    # unlock_after周期ちょうどまでは床が適用され続ける(target_mag=floor_mag)
    for _ in range(unlock_after):
        target_mag_out, floor_mag, invalid_count = _apply_overlap_floor_v3_mirror(
            target_mag, "OVERTAKING", -1.0, floor_mag, invalid_count,
            unlock_after=unlock_after)
        assert target_mag_out == floor_mag
    # unlock_afterを超えた次の周期からは床が外れ、target_magがそのまま返る
    target_mag_out, floor_mag, invalid_count = _apply_overlap_floor_v3_mirror(
        target_mag, "OVERTAKING", -1.0, floor_mag, invalid_count,
        unlock_after=unlock_after)
    assert target_mag_out == target_mag


def test_mirror_invalid_count_resets_and_floor_resumes_when_corr_recovers():
    """corr_boundが有効な値へ戻れば、カウンタが即座に0へ戻り床の適用が
    再開することを確認する(床のピーク値自体はリセットされない)。"""
    floor_mag = 3.0
    invalid_count = 90  # 既にタイムアウト超過している状態
    target_mag_out, floor_mag_out, invalid_count_out = _apply_overlap_floor_v3_mirror(
        0.1, "OVERTAKING", 5.0, floor_mag, invalid_count)
    assert invalid_count_out == 0
    assert target_mag_out == min(floor_mag, 5.0 - 0.1)  # 床が再適用される
