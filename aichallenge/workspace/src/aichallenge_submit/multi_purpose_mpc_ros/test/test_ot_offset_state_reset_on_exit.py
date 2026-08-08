"""横オフセット状態(lateral_target/_ot_alpha)のOVERTAKING離脱時リセット(230節続報、2026-07-29)。

背景: 実ログ(0728-04 wp243)の調査で、MPCソルバーのinfeasibility_counterが
17→124まで拡大し続ける事象を発見した。根本原因を遡ると、self._mpc.lateral_target
(横オフセット目標)とself._ot_alpha(0..1ブレンド係数)が、OVERTAKINGへ入る際
(5158/5179行目付近)にのみ書き込まれる一方、離脱側の3経路
(giveup系/通常exit_clear/infeasibility強制)のいずれも_ot_side・_ot_side_locked・
_ot_giveup_count等は個別にリセットしていたのに、この2変数だけ触れていなかった
ことが判明した。結果、離脱後も既存の_ot_ramp_time(2.5秒)ランプで緩やかにしか
_ot_alphaが0へ減衰しないため、最大2.5秒間「もう無効なはずの横オフセット目標」を
保持し続け、これが離脱直後のQP解にも影響してinfeasibilityの拡大を助長していた。

対処: 側フリップ(4778行目)・側変更を伴う再エンゲージ(5024行目)で既に
self._ot_alphaのみ即時リセットする前例があったため、その適用範囲をOVERTAKING
離脱全般へ一般化する専用ヘルパー_reset_ot_offset_state()を新設し、
STUCK復帰用の既存_reset_ot_side_for_fresh_replan()と、OVERTAKING離脱3経路
(giveup系・通常exit_clear・infeasibility強制)の計4箇所全てから呼び出す形に
統一した。特定地点(wp243)への対処ではなく、状態遷移の一貫性を保つ汎用修正。

mpc_controller.pyはrclpy非依存のため直接importできず、他の巨大メソッド関連
テスト群(test_stuck_recovery_side_reselection.py等)と同じ方針(純Pythonミラー
ではなくソーステキストによる構造的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: ヘルパー自体が正しい3変数をリセットすること
# ---------------------------------------------------------------------------

def _helper_body():
    idx = _SRC.index("def _reset_ot_offset_state(self)")
    idx_end = _SRC.index("\n    def ", idx + 10)
    return _SRC[idx:idx_end]


def test_reset_offset_state_helper_resets_alpha_lateral_target_and_blend():
    snippet = _helper_body()
    assert "self._ot_alpha = 0.0" in snippet
    assert "self._mpc.lateral_target = 0.0" in snippet
    assert "self._mpc.lateral_blend = 0.0" in snippet


def test_reset_offset_state_helper_does_not_touch_side_commit_state():
    """②非冗長性: 本ヘルパーは横オフセット関連の3変数のみに責務を限定し、
    _ot_side/_ot_side_locked/_ot_giveup_count等(既に各離脱箇所で個別に
    正しくリセットされている)には一切触れない。責務を分離することで、
    将来どちらかの変数セットだけを変更する際に他方を巻き込まない。"""
    snippet = _helper_body()
    assert "self._ot_side" not in snippet
    assert "self._ot_giveup_count" not in snippet
    assert "self._ot_state" not in snippet


# ---------------------------------------------------------------------------
# ④遡及効果: OVERTAKING離脱の全経路(giveup系・exit_clear・infeasibility強制)
#   および STUCK復帰経路のいずれからも呼ばれていること
# ---------------------------------------------------------------------------

def test_called_from_giveup_family_exit():
    """giveup系離脱(force_giveup/room_exhausted/側消失、4935行目付近)で
    呼ばれていることを確認する。"""
    idx = _SRC.index('self._ot_footprint_risk_gated = _lat_dec.footprint_risk_triggered')
    # 2026-08-05追加(engage_cooldown早期解除①③): speed_gated/room_gated設定+
    #   説明コメントが間に挿入されたため、窓を400→1600へ拡大(検証対象そのものは
    #   無変更)。
    idx_end = idx + 1600
    snippet = _SRC[idx:idx_end]
    assert 'self._ot_state = "STOPPING"' not in snippet  # 直前の代入なので後方にはもう無いはず
    assert "self._reset_ot_offset_state()" in snippet


def test_called_from_ordinary_exit_clear():
    """通常のexit_clear離脱(前方クリア連続、5033行目付近)で呼ばれていることを
    確認する。2026-08-06修正(重複ログバグ修正)でアンカーを
    `self._fwd_clear_count += 1`直後の状態ガード付きifへ更新。"""
    idx = _SRC.index("self._fwd_clear_count += 1")
    idx_end = idx + 1300
    snippet = _SRC[idx:idx_end]
    assert 'self._ot_state != "NORMAL"' in snippet
    assert 'self._ot_state = "NORMAL"' in snippet
    assert "self._reset_ot_offset_state()" in snippet


def test_called_from_infeasibility_forced_exit():
    """infeasibility強制STOPPING(5053行目付近、0728-04 wp243の実事例)で
    呼ばれていることを確認する。2026-08-09追加(§45.3)のselflock_escape_active
    リセット行の分だけ窓を600→700へ拡大(検証内容自体は変更なし)。"""
    idx = _SRC.index("if self._mpc.infeasibility_counter == self._ot_infeasible_stop:")
    idx_end = idx + 700
    snippet = _SRC[idx:idx_end]
    assert 'self._ot_state = "STOPPING"' in snippet
    assert "self._reset_ot_offset_state()" in snippet


def test_called_from_stuck_recovery_fresh_replan_helper():
    """STUCK復帰(_reset_ot_side_for_fresh_replan経由、4箇所)でも同様に
    横オフセットが持ち越されないよう、composition(委譲呼び出し)されている
    ことを確認する。"""
    idx = _SRC.index("def _reset_ot_side_for_fresh_replan(self)")
    idx_end = _SRC.index("def _reset_ot_offset_state(self)", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._reset_ot_offset_state()" in snippet


def test_total_call_count_matches_four_known_sites():
    """②非冗長性: 呼び出し箇所が想定通り4箇所(giveup系・exit_clear・
    infeasibility強制・_reset_ot_side_for_fresh_replan内)であることを確認する。
    新しい離脱経路が追加/削除された場合はこのテスト自体の更新も必要。"""
    n_calls = _SRC.count("self._reset_ot_offset_state()")
    assert n_calls == 4, (
        f"想定していた4箇所から数が変わっている(現在{n_calls}箇所)。"
        "新しい離脱経路が追加/削除された場合はこのテスト自体の更新も必要。")


# ---------------------------------------------------------------------------
# 非干渉性: OVERTAKING継続中の通常パス(側フリップ・再エンゲージ)からは
#   呼ばれていないこと(それらは既にlateral_targetを同一周期内で再計算するため、
#   このヘルパーによる0クリアを挟むと不要な瞬間ゼロ落ちを生む)
# ---------------------------------------------------------------------------

def test_not_called_from_side_flip_or_fresh_engage_paths():
    """switchback側フリップ(4778行目付近)・新規/側変更を伴う再エンゲージ
    (5024行目付近、self._ot_alpha=0.0のみ既存の即時リセットを持つ箇所)は、
    どちらも同一周期内でlateral_targetを直後に再計算するOVERTAKING継続経路の
    ため、本ヘルパーを呼ぶ必要が無い(呼ぶと不要な中間ゼロ経由が発生しうる)。"""
    idx_side_flip = _SRC.index("self._ot_alpha = 0.0  # 既存H3ガード2と同じ再ランプ(急ハンドル防止)")
    snippet_side_flip = _SRC[idx_side_flip - 50:idx_side_flip + 200]
    assert "_reset_ot_offset_state" not in snippet_side_flip

    idx_reengage = _SRC.index('self._ot_state = "OVERTAKING"')
    idx_alpha0 = _SRC.index("self._ot_alpha = 0.0\n", idx_reengage)
    snippet_reengage = _SRC[idx_alpha0 - 50:idx_alpha0 + 200]
    assert "_reset_ot_offset_state" not in snippet_reengage
