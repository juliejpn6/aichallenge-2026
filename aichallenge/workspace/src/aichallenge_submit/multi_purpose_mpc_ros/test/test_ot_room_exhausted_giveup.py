"""Unit tests for OT-ROOM-EXHAUSTED giveup integration (168節, 2026-07-24)。

背景: 0724-01予選ログ(wp160-163)実測で、side=-1でOVERTAKING継続中のまま、
2台のほぼ停止した相手車を同時に回避しようとした結果、先読みコリドー
(_corr_bound_ahead(self._ot_side))が非正へ転落(=先読み内に正の隙間が皆無、
幾何学的に不可能)した。従来の実装は `_target_mag = min(d_off, max(0.0, corr_bound))`
でこれを即座に0(直進)へクランプしており、_ot_side/_ot_stateは「OVERTAKING継続中」
を主張したまま、実際の指令(lateral_target)だけが3周期(約1秒)でオフセット
-1.196→-0.710→-0.242→-0.000へ崩壊し、その直後にGHOST-BLOCK/STUCKが再発した。
ユーザー指摘:「バックした後の再発進時には眼の前の状況を把握し、障害物を
避けるような動作が必要」——実際には、そもそもオフセット目標が直進化して
いたこと自体が二次STUCKの直接原因だった。

対処は2段構成:
1. 決定層(本ファイルの主対象): _corr_bound_ahead(委託側)が非正のまま
   _ot_giveup_cycles(既存の断念デバウンス、≈1s、新規定数0個)連続した場合、
   既存のlat_ttc系force_giveupと同じ_side_blocked合流点へ折り込み、
   STOPPINGへ正しく離脱させる(_ot_room_exhausted_count、新規カウンタ1個)。
2. 症状層(test_corr_bound_ahead_diagnostic.pyでカバー): 上記giveupが実際に
   合流するまでのデバウンス窓の間、目標をmax(0.0,...)で0へ落とさず、直近の
   有効(正マージン)時の値を凍結保持する(_ot_last_valid_target_mag)。

mpc_controller.pyはautoware_auto_control_msgs等をモジュールスコープでimportして
おり単体テスト環境では直接importできず、対象の状態機械は数千行の巨大な1メソッド
深くに埋め込まれているため、test_stuck_recovery_side_reselection.py等と同じく
実物のソーステキストに対する構造的検証を行う(フル実行は非実用的、という同一の
既存判断を踏襲)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 状態変数の初期化・リセット箇所
# ---------------------------------------------------------------------------

def test_new_state_vars_declared_in_init():
    """新規状態は_initialize()内(_ot_giveup_count等の既存OT状態と同じ箇所)で
    宣言されていることを確認する(__init__自体は_initialize()を呼ぶだけの薄いラッパ)。"""
    idx = _SRC.index("    def _initialize(self) -> None:")
    idx_next_method = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_next_method]
    assert "self._ot_room_exhausted_count = 0" in snippet
    assert "self._ot_room_exhausted_prev_side = 0" in snippet
    assert "self._ot_last_valid_target_mag = None" in snippet


def test_reset_helper_also_clears_room_exhausted_state():
    """STUCK復帰の側リセット(_reset_ot_side_for_fresh_replan)でも、旧側の
    room_exhausted計数・凍結オフセットを持ち越さないことを確認する
    (持ち越すと、STUCK復帰直後の新規側選択が古い計数のせいで不当に早く
    giveupしてしまう非矛盾性違反になる)。"""
    idx = _SRC.index("def _reset_ot_side_for_fresh_replan(self)")
    idx_end = _SRC.index("def _stuck_recovery_complete(", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._ot_room_exhausted_count = 0" in snippet
    assert "self._ot_last_valid_target_mag = None" in snippet


def test_switchback_side_flip_resets_room_exhausted_state():
    """side_override(電撃switchback)で側が反転した際も、旧側の計数・凍結値を
    新側へ持ち越さないことを確認する(alpha=0.0の再ランプと同一箇所)。
    2026-08-07改訂(Fix B、design_docs...20260806.md §4): 個別の
    self._ot_last_valid_target_mag = None行は共通ヘルパー
    _reset_ot_episode_tracking_state()呼び出しへ統合された。"""
    idx = _SRC.index("_locked = _lat_dec.side_override")
    idx_end = idx + 700
    snippet = _SRC[idx:idx_end]
    assert 'self._ot_alpha = 0.0' in snippet
    assert "self._ot_room_exhausted_count = 0" in snippet
    assert "self._reset_ot_episode_tracking_state()" in snippet


def test_fresh_engage_resets_room_exhausted_state():
    """新規エンゲージ(state=OVERTAKINGへの遷移)でも、前回エピソード
    (別側・別相手)の計数・凍結値を持ち越さないことを確認する。
    2026-08-07改訂(Fix B、design_docs...20260806.md §4): 個別の
    self._ot_last_valid_target_mag = None行は共通ヘルパー
    _reset_ot_episode_tracking_state()呼び出しへ統合された。"""
    idx = _SRC.index('self._ot_state = "OVERTAKING"')
    idx_end = idx + 700
    snippet = _SRC[idx:idx_end]
    assert "self._ot_giveup_count = 0" in snippet
    assert "self._ot_room_exhausted_count = 0" in snippet
    assert "self._reset_ot_episode_tracking_state()" in snippet


# ---------------------------------------------------------------------------
# 決定層: room_exhaustedの計算・_side_blockedへの合流
# ---------------------------------------------------------------------------

def test_room_ahead_computed_for_locked_side_after_switchback_resolution():
    """_room_ahead_lockedは、switchback(側反転)の判定・実行(if/elifブロック)が
    終わった後の_lockedを使って計算されていることを出現順で確認する
    (反転前の古い側で判定すると、反転直後の1周期だけ誤った側のroomを見る)。"""
    idx_switchback_start = _SRC.index("if _lat_dec.side_override is not None:")
    idx_suppressed_end = _SRC.index(
        "# 案3(2026-07-12)→2026-07-17に旧EMA判定(_ot_side_block_ema)を削除し完全統一:")
    idx_room = _SRC.index(
        "_room_ahead_locked = (\n"
        "                        self._corr_bound_ahead(_locked) if _locked != 0 else float('inf'))")
    assert idx_switchback_start < idx_suppressed_end < idx_room


def test_room_exhausted_count_increments_only_when_non_finite_excluded_and_non_positive():
    idx = _SRC.index("if np.isfinite(_room_ahead_locked) and _room_ahead_locked <= 0.0:")
    snippet = _SRC[idx:idx + 260]
    assert "self._ot_room_exhausted_count += 1" in snippet
    assert "self._ot_room_exhausted_count = 0" in snippet


def test_room_exhausted_count_resets_on_side_change():
    """委託側が変わった周期は計数を仕切り直すことを確認する(別の側の履歴を
    引き継いで誤発火しないため)。"""
    idx = _SRC.index("if _locked != self._ot_room_exhausted_prev_side:")
    snippet = _SRC[idx:idx + 200]
    assert "self._ot_room_exhausted_count = 0" in snippet
    assert "self._ot_room_exhausted_prev_side = _locked" in snippet


def test_room_exhausted_reuses_existing_giveup_cycles_debounce():
    """②非冗長性: room_exhaustedの発火デバウンスは新規の閾値ではなく、既存の
    _ot_giveup_cycles(≈1s、lat_ttc系giveupと共通)をそのまま再利用している
    ことを確認する。"""
    idx = _SRC.index("_room_exhausted = self._ot_room_exhausted_count >= self._ot_giveup_cycles")
    assert idx > 0


def test_side_blocked_folds_in_room_exhausted_alongside_force_giveup():
    idx = _SRC.index("_side_blocked = _lat_dec.force_giveup or _room_exhausted")
    assert idx > 0


def test_giveup_trigger_label_distinguishes_room_exhausted_from_lat_ttc():
    idx = _SRC.index('_giveup_trigger = ("room_exhausted"')
    snippet = _SRC[idx:idx + 300]
    assert "not _lat_dec.force_giveup" in snippet
    assert 'f"lat_ttc_{_lat_dec.branch}"' in snippet
    idx_log = _SRC.index('f"[LAT-TTC-ACT] giveup trigger={_giveup_trigger}')
    assert idx_log > idx


def test_ot_room_exhausted_verification_log_present():
    """③検証ロギング: giveup合流の瞬間(閾値到達エッジ)を専用ログで一度だけ記録し、
    次回ログで発火有無を直接確認できること。"""
    idx = _SRC.index('"[OT-ROOM-EXHAUSTED] side={_locked} "')
    snippet = _SRC[idx:idx + 300]
    assert "corr_bound={_room_ahead_locked:.3f}" in snippet
    assert "count={self._ot_room_exhausted_count}" in snippet
    # エッジ検知(==giveup_cycles到達周期のみ)であり、continued cyclesで連打しないこと。
    # 247節(2026-07-30)でこのif節内に反対側の最終救済チェック(_room_rescued)が
    # 追加されたため、guard直後ではなく「else節(救済失敗時)の中」に位置するように
    # なった——window自体は広げたが、依然として同一のif節内であることを
    # _side_blocked合流行より前であることで確認する(guardとの直接近接は保証しない)。
    idx_guard = _SRC.index("if _room_exhausted and self._ot_room_exhausted_count == self._ot_giveup_cycles:")
    idx_log = _SRC.index('"[OT-ROOM-EXHAUSTED] side={_locked} "')
    idx_side_blocked = _SRC.index("_side_blocked = _lat_dec.force_giveup or _room_exhausted")
    assert idx_guard < idx_log < idx_side_blocked
    assert "if _room_rescued:" in _SRC[idx_guard:idx_side_blocked]


# ---------------------------------------------------------------------------
# ①非矛盾性: room_exhausted判定が_side_blocked経由の既存離脱経路を
#   そのまま再利用しており、独自の状態遷移を新設していないこと
# ---------------------------------------------------------------------------

def test_room_exhausted_does_not_introduce_a_new_state_transition_path():
    """_room_exhaustedはブール値として_side_blockedへ折り込まれるだけで、
    その後のSTOPPING遷移・cooldown設定・prev_side記録等は既存の
    if (giveup_count>=cycles or locked==0 or side_blocked) 分岐を
    無変更のまま通ることを確認する(新しい状態遷移コードパスを増やしていない)。"""
    idx = _SRC.index("_side_blocked = _lat_dec.force_giveup or _room_exhausted")
    idx_end = idx + 400
    snippet = _SRC[idx:idx_end]
    assert ("if (self._ot_giveup_count >= self._ot_giveup_cycles\n"
            "                            or _locked == 0 or _side_blocked):") in snippet
