"""Structural regression guard for the switchback-token-waste fix (79節, 2026-07-16).

mpc_controller.py imports rclpy/autoware message types at module scope, and the
wiring under test spans a ~50-line region deep inside the 600+ line `_control()`
method with many ROS-typed free variables, so full AST extraction / execution is
impractical (same situation as test_ot_offset_ramp.py and test_wall_slow_lookahead
before it). Instead this file does a structural source-text check: it asserts the
NEW wiring pattern is present and the OLD (buggy) post-hoc veto pattern is gone.
This is intentionally coarse (regex/substring, not exec), but it directly guards
against the single most likely regression: someone reintroducing the post-hoc
veto-after-consumption pattern that caused the original bug.

Background: 77節's `_switchback_curvature_veto()` was called AFTER
`LateralTTCMonitor.update()` already returned side_override (and had already
mutated has_switched/has_rescued internally). Vetoing at that point discarded the
ACTION but not the internal state consumption, wasting the one-shot token
(confirmed via 0715-08 wp61->wp73->wp75 real log replay, see
test_lateral_ttc_monitor.py's new_side_blocked tests). 79節 moves the curvature
check to BEFORE update() is called, passing it in as `new_side_blocked` so the
Monitor's own eligibility conditions absorb it (no token consumed if blocked).
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_new_side_blocked_is_computed_before_the_update_call():
    """新配線: _new_side_blockedが_lat_ttc.update()呼び出しより前に計算され、
    update()へ引数として渡されている(ソース中の出現順で確認)。"""
    idx_compute = _SRC.index("_new_side_blocked = (self._switchback_curvature_veto(")
    idx_update_call = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_compute < idx_update_call
    # update()呼び出しブロック内でnew_side_blockedが渡されていることを確認する。
    update_call_region = _SRC[idx_update_call:idx_update_call + 400]
    assert "new_side_blocked=_new_side_blocked" in update_call_region


def test_new_side_blocked_computation_mirrors_expected_formula():
    """新配線: 反転先(-self._ot_side)を渡し、side==0の場合はFalseにフォールバック
    する(既存のreset_episode()呼び出し条件と対をなす安全な既定値)。"""
    idx = _SRC.index("_new_side_blocked = (self._switchback_curvature_veto(")
    snippet = _SRC[idx:idx + 200]
    assert "self._switchback_curvature_veto(-self._ot_side)" in snippet
    assert "if self._ot_side != 0 else False" in snippet


def test_post_hoc_veto_after_consumption_pattern_is_removed_regression():
    """回帰防止(最重要): 77節にあった「update()の戻り値を受け取った後に
    _switchback_curvature_vetoを呼んで実行のみを止める」という、トークン浪費の
    直接原因だったパターンが再導入されていないことを確認する。"""
    assert "_curvature_vetoed = self._switchback_curvature_veto(\n" \
           "                            _lat_dec.side_override)" not in _SRC
    assert "switchback_vetoed reason=k_corner" not in _SRC
    # _switchback_curvature_veto自体は(呼び出しタイミングを変えただけで)引き続き
    # 存在し、new_side_blockedの計算に使われていることを確認する。84節で
    # lookahead_favor_switch用に同じ関数を現在側へも適用する呼び出しが1つ
    # 追加されたため、合計呼び出し回数は2(反対側1回+現在側1回、新規スキャン
    # 処理の追加ではなく既存関数の再利用)。
    assert _SRC.count("self._switchback_curvature_veto(") == 2


def test_switchback_suppressed_reason_unified_for_both_margin_and_curvature():
    """新配線: switchback_suppressedのログreasonが、margin/k_corner の両方を
    _lat_dec.switchback_curvature_blockedの値で単一のログタグから区別できる
    ように一本化されている(旧: 別タグswitchback_vetoedとの二重体制だった)。"""
    idx = _SRC.index('_reason = ("cleared_margin" if _lat_dec.switchback_cleared_margin_blocked')
    # 2026-07-20追加(125節、A-1): "wall"分岐挿入でログ行までのオフセットが伸びた
    # ため、500→600へ拡幅(113節で確立した同種の固定長ウィンドウ対処と同じ方針)。
    # 2026-07-22追加(159節): "offset"分岐挿入(コメント込み)で900へさらに拡幅
    # (検証対象は無変更)。
    # 2026-07-26追加(191節): "ds"分岐挿入(コメント込み)で1200へさらに拡幅
    # (検証対象は無変更)。
    snippet = _SRC[idx:idx + 1200]
    assert 'if _lat_dec.switchback_curvature_blocked' in snippet
    assert 'else "margin")' in snippet
    assert "switchback_suppressed reason={_reason}" in snippet


def test_switchback_suppressed_reason_cleared_margin_takes_priority():
    """回帰防止(84節、2026-07-17に82節分のcleared判定を削除): reasonの優先順位は
    cleared_margin > k_corner > marginであることをソース上のif/elif順序で確認する
    (各ブロックは互いに独立の理由であり、同時に真になり得ても診断ログ上で
    取り違えないようにするため)。"""
    idx = _SRC.index('_reason = ("cleared_margin" if _lat_dec.switchback_cleared_margin_blocked')
    # 2026-07-22追加(159節): "offset"分岐挿入で400→700へ拡幅(検証対象は無変更)。
    # 2026-07-26追加(191節): "ds"分岐挿入で700→1000へさらに拡幅(検証対象は無変更)。
    snippet = _SRC[idx:idx + 1000]
    cleared_margin_pos = snippet.index('"cleared_margin"')
    k_corner_pos = snippet.index('"k_corner"')
    margin_pos = snippet.index('"margin"')
    assert cleared_margin_pos < k_corner_pos < margin_pos


def test_lookahead_favor_switch_computed_from_both_side_directions():
    """新配線(84節②): lookahead_favor_switchは_switchback_curvature_veto()を
    現在側(self._ot_side)と反対側(-self._ot_side、既存のnew_side_blocked)の
    両方に適用して算出されており、新規スキャン処理が追加されていないことを
    確認する(既存関数の2回呼び出しのみ、新規パラメータ0個)。"""
    assert "_current_side_closing_ahead = (self._switchback_curvature_veto(self._ot_side)" in _SRC
    # 2026-07-22修正(157節): _new_side_curvature_override成立時もlookahead経路を
    #   許可するよう更新された(通常経路と挙動を一致させるための意図的な変更)。
    assert ("_lookahead_favor_switch = _current_side_closing_ahead and (\n"
            "                not _new_side_blocked or _new_side_curvature_override)") in _SRC
    # 新規スキャン処理を増やしていないことの確認: 呼び出し回数は現在側・反対側の2回のみ
    assert _SRC.count("self._switchback_curvature_veto(") == 2


def test_lookahead_favor_switch_passed_into_update_call():
    """新配線(84節②): 算出したlookahead_favor_switchが実際にLateralTTCMonitor.
    update()の引数として渡されていることを確認する。"""
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    # 2026-07-22追加(157節): new_side_curvature_override引数が1行増えたため
    #   窓を400→500へ拡大(検証対象そのものは無変更)。
    snippet = _SRC[idx:idx + 500]
    assert "lookahead_favor_switch=_lookahead_favor_switch" in snippet


def test_switchback_and_suppressed_logs_include_lookahead_field():
    """検証ロギング用(84節): switchback発火ログ・抑制ログの両方に
    lookahead=(_lat_dec.lookahead_favor_switch)が含まれ、先回り切り替え条件が
    その周期に成立していたかをログ単体で確認できる。"""
    assert _SRC.count("lookahead={_lat_dec.lookahead_favor_switch}") == 2


def test_side_override_commit_no_longer_conditionally_skips_via_curvature_check():
    """回帰防止: side_override is not None のブロックが、以前のif/elseによる
    curvatureチェックを経由せず、直接既存の反転処理(_locked = ...)へ進むこと
    (update()が既にcurvature-safeな結果のみside_overrideを返すため)を確認する。"""
    idx = _SRC.index("if _lat_dec.side_override is not None:")
    snippet = _SRC[idx:idx + 800]
    assert "_locked = _lat_dec.side_override" in snippet
    # このブロックの直後(数行以内)にside_overrideの中身を判定するif/elseが
    # 存在しない(旧: if _curvature_vetoed: ... else: ...という二重分岐だった)。
    assert "_curvature_vetoed" not in snippet


def test_current_side_closing_ahead_passed_into_update_call():
    """新配線(92節①): 既存の_current_side_closing_ahead(84節②でlookahead用に
    算出済み、新規スキャン処理0個)がupdate()呼び出しへそのまま渡されている
    ことを確認する。"""
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx:idx + 700]
    assert "current_side_closing_ahead=_current_side_closing_ahead" in snippet


def test_lat_ttc_log_includes_curvature_run_field():
    """検証ロギング用(92節①): [LAT-TTC]ログにcurvature_run=
    (_lat_dec.critical_curvature_run)が含まれ、C1_deferredが何周期継続中かを
    ログ単体で確認できる。"""
    idx = _SRC.index('f"[LAT-TTC] branch=')
    snippet = _SRC[idx:idx + 1200]
    assert "curvature_run={_lat_dec.critical_curvature_run}" in snippet


def test_fwd_is_obstacle_class_no_longer_passed_into_update_call_regression():
    """回帰防止(100節、Tier1裁定の外出し): fwd_is_obstacle_classはLAT-TTC.update()の
    引数ではなくなった(旧C1_obstacle_yield分岐がlateral_ttc_monitor.py内から削除され、
    呼び出し元のv_safe候補集約側へ移設されたため)。update()呼び出しの引数一覧には
    もう含まれていないことを確認する。"""
    idx = _SRC.index("_lat_dec = self._lat_ttc.update(")
    snippet = _SRC[idx:idx + 900]
    assert "fwd_is_obstacle_class=" not in snippet


def test_fwd_is_obstacle_class_computed_as_local_before_update_call():
    """新配線(100節): 既存のopp_obstacle_speed閾値をそのまま再利用して算出した
    _fwd_is_obstacle_classが、update()呼び出しより前にローカル変数として計算されて
    いることを確認する(新規スキャン処理0個、値の計算式自体は不変)。"""
    idx_local = _SRC.index(
        "_fwd_is_obstacle_class = (_fwd_vopp is not None\n"
        "                                       and _fwd_vopp < self._opp_obstacle_speed)")
    idx_update_call = _SRC.index("_lat_dec = self._lat_ttc.update(")
    assert idx_local < idx_update_call


def test_lat_c1_yield_wired_at_v_safe_candidate_site():
    """新配線(100節): 旧C1_obstacle_yield分岐(lateral_ttc_monitor.py内)の判定が
    v_safe候補集約側(_lat_dec.branch=="C1" かつ _fwd_is_obstacle_class の場合のみ
    候補から除外)へ移設されていることを確認する。"""
    idx = _SRC.index('_v_safe_cand.append((_lat_dec.v_safe_cap_label, _lat_dec.v_safe_cap))')
    snippet = _SRC[max(0, idx - 1000):idx]
    assert '_lat_dec.branch == "C1" and _fwd_is_obstacle_class' in snippet
    assert "not _lat_c1_yielded" in snippet


def test_tier1_c1_yield_edge_triggered_log_present():
    """検証ロギング用(100節): [TIER1-C1-YIELD]ログが、LAT-TTCのC1候補を実際に
    破棄した瞬間(遷移時)にのみ出力されることを確認する(既存の[V-SAFE-SRC-CHANGE]
    と同じedge-triggeredパターン)。"""
    assert "_lat_c1_yielded != self._lat_ttc_c1_yield_prev" in _SRC
    assert '"[TIER1-C1-YIELD]' in _SRC
    assert "self._lat_ttc_c1_yield_prev = _lat_c1_yielded" in _SRC


def test_lat_ttc_c1_yield_prev_state_initialized():
    assert "self._lat_ttc_c1_yield_prev = False" in _SRC
