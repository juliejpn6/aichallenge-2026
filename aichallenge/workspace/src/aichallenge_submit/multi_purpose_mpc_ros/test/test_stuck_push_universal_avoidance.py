"""Regression tests for the "PUSH for every STUCK path" redesign (171節, 2026-07-24).

背景: 168節でwp161スタック再発の根本原因(corr_bound非正転落によるオフセット崩壊)を
修正したが、その後の予選ログ(0724-02)で経路1/2(通常のSTUCK/infeasibility検知)は
BACKUP後もステア0固定の直進で復帰し、目の前に残っている障害物・壁へ再度向かって
しまうことが分かった(経路3=完全停止デッドロックだけは既存のPUSH機構で操舵回避
していたが、上限6°という保守的な小角に留まっていた)。

ユーザー指示:「スタックを検出し、バックしたらそこからはgate2の課題を解くのと同じ
です。眼の前には必ず障害物がある(もしくは壁のほうを向いている)ので、その障害物を
避けてレースラインに復帰する処理が必要です。低速でステアリング舵角を最大まで使って
障害物を回避しましょう。回避できたら通常処理に復帰してください」。

対処: 経路1/2/3の全てでBACKUP後は必ずWAIT_DRIVE_PUSH→PUSHを経由するよう統一し、
①側選択はgate2のENGAGE判定と同じ_plan_pass(_compute_stuck_push_steer経由、新規
側選択ロジックは作らない)、②舵角上限はMPC自体のハード制約delta_max_degをそのまま
使う(独自の最大値を持たない)、③完了条件を固定距離/タイムアウトだけでなく
「実際に回避できたか」(_corr_bound_ahead再評価)ベースに拡張、の3点で実装した。

mpc_controller.pyはautoware_auto_control_msgs等をモジュールスコープでimportしており
単体テスト環境では直接importできないため、他の巨大メソッド関連テストと同じく実物の
ソーステキストに対する構造的検証を行う。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")

with open(_SRC_PATH) as _f:
    _SRC = _f.read()
with open(_YAML_PATH) as _f:
    _YAML_SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: 経路によらず必ずPUSHを経由する
# ---------------------------------------------------------------------------

def test_backup_completion_always_routes_to_wait_drive_push():
    """旧実装は_stuck_trigger_path==3のときのみWAIT_DRIVE_PUSHへ分岐し、それ以外は
    WAIT_DRIVE(ステア0固定)へ分岐していた。新実装は経路によらず常に
    WAIT_DRIVE_PUSHへ遷移することを確認する。
    2026-07-26更新(184節): 判定閾値がself._stuck_backup_dist(固定2.0m)から
    self._stuck_backup_dist_eff(後方の相手車を考慮した実効後退距離)へ変わった
    ため、検索対象の文字列を更新した(閾値の意味が変わっただけで、経路によらず
    必ずWAIT_DRIVE_PUSHへ進むという本テストの主張自体は変わらない)。"""
    idx = _SRC.index("if dist >= self._stuck_backup_dist_eff:")
    snippet = _SRC[idx:idx + 600]
    assert '_next = "WAIT_DRIVE_PUSH"' in snippet
    # 旧来の経路3限定の三項演算子が残っていないこと
    assert "if self._stuck_trigger_path == 3 else" not in snippet


def test_wait_drive_state_no_longer_reachable():
    """②非冗長性: 到達不能になったWAIT_DRIVE状態への遷移代入がソース中に
    残っていないこと(死んだ状態遷移コードを残さない)。"""
    assert '"WAIT_DRIVE"' not in _SRC or _SRC.count('"WAIT_DRIVE"') == 0


def test_push_state_handles_stuck_state_variable_comment_updated():
    """_stuck_stateが取りうる値のコメントからWAIT_DRIVEが除かれていること
    (実装と乖離したドキュメントコメントを残さない)。"""
    idx = _SRC.index('self._stuck_state = "NORMAL"   # NORMAL/')
    line_end = _SRC.index("\n", idx)
    line = _SRC[idx:line_end]
    assert "WAIT_DRIVE_PUSH" in line
    assert "WAIT_DRIVE/" not in line  # "WAIT_DRIVE/WAIT_DRIVE_PUSH"の形で残っていないこと


# ---------------------------------------------------------------------------
# 壁マージンフォールバック(172節続報、2026-07-24): 相手車ベースの側判定が失敗
#   (STUCKの原因が相手車でなく壁そのもの)しても直進のまま再突入しないための対処。
#   0724-03予選ログでは41/41回のPUSHが全てside不明→直進のまま同じ壁へ再突入していた。
# ---------------------------------------------------------------------------

def test_wall_fallback_only_activates_when_plan_pass_fails():
    idx = _SRC.index("def _compute_stuck_push_steer(self, pose) -> float:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    idx_if = snippet.index("if not _plan_ok or _plan_side == 0:")
    idx_fallback = snippet.index("_room_left = self._corr_bound_ahead(1)")
    assert idx_if < idx_fallback  # フォールバックはplan_pass失敗ブロックの内側


def test_wall_fallback_reuses_existing_corr_bound_ahead_both_directions():
    """②非冗長性: 新しい空き幅計算式を作らず、既存のcorr_bound_ahead()を
    左右(+1/-1)それぞれに呼ぶだけであることを確認する。"""
    idx = _SRC.index("_room_left = self._corr_bound_ahead(1)")
    snippet = _SRC[idx:idx + 200]
    assert "_room_right = self._corr_bound_ahead(-1)" in snippet


def test_wall_fallback_requires_meaningful_asymmetry_before_picking_a_side():
    """①非矛盾性: 左右差が僅少な場合まで無理に方向を決めると誤判定リスクの方が
    大きいため、有意な差がある場合のみ採用することを確認する。"""
    idx = _SRC.index("_room_left = self._corr_bound_ahead(1)")
    snippet = _SRC[idx:idx + 500]
    assert "np.isfinite(_room_left)" in snippet
    assert "np.isfinite(_room_right)" in snippet
    assert "abs(_room_left - _room_right) > self._along_min_width * 0.1" in snippet


def test_wall_fallback_sets_stuck_push_side_so_cleared_check_becomes_live():
    """168/171節の「実際に回避できたか」判定(_stuck_push_side!=0が前提)が、
    壁フォールバック採用時にも機能するようself._stuck_push_sideを設定すること
    を確認する(0724-03ログで41/41回とも発火していなかった問題への対処)。"""
    idx = _SRC.index("_wall_side = 1 if _room_left > _room_right else -1")
    snippet = _SRC[idx:idx + 200]
    assert "self._stuck_push_side = _wall_side" in snippet


def test_wall_fallback_verification_log_present():
    """③検証ロギング: 壁フォールバックが発火したことと、採用したside・左右の
    マージン値を次回ログで直接確認できること。"""
    idx = _SRC.index('"[STUCK-PUSH-WALL-FALLBACK]')
    snippet = _SRC[idx:idx + 200]
    assert "left={_room_left:.2f}" in snippet
    assert "right={_room_right:.2f}" in snippet
    assert "side={_wall_side}" in snippet


def test_wall_fallback_scales_magnitude_same_way_as_opponent_based_push():
    """②非冗長性: 舵角の room-based スケーリング式(room/room_ref、delta_max_deg
    上限)は相手車ベースの経路と同一の計算式を再利用しており、別のスケール式を
    増やしていないことを確認する。"""
    idx = _SRC.index("_wall_side = 1 if _room_left > _room_right else -1")
    snippet = _SRC[idx:idx + 600]
    assert "_scale = min(1.0, _room / self._stuck_push_steer_room_ref)" in snippet
    assert "_mag = np.deg2rad(self._stuck_push_steer_max_deg) * _scale" in snippet


def test_wall_fallback_falls_through_to_straight_when_both_sides_invalid_or_symmetric():
    """安全側フォールバック: 左右差が僅少、またはcorr_bound_aheadが非有限
    (先読み配列が無い等)の場合は、従来通りside=0/直進(0.0)のままであることを
    構造的に確認する(壁フォールバックのif分岐の後に元のside=0/return 0.0が
    そのまま残っていること)。"""
    idx = _SRC.index("_room_left = self._corr_bound_ahead(1)")
    idx_end = _SRC.index("self._stuck_push_side = _plan_side", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_push_side = 0" in snippet
    assert "return 0.0" in snippet


# ---------------------------------------------------------------------------
# ②非冗長性: 舵角上限はMPC自体のdelta_max_degを一元的な出典とする
# ---------------------------------------------------------------------------

def test_push_steer_max_defaults_to_mpc_delta_max_deg():
    idx = _SRC.index("self._stuck_push_steer_max_deg = float(")
    snippet = _SRC[idx:idx + 200]
    assert "self._cfg.mpc.delta_max_deg" in snippet


def test_yaml_no_longer_hardcodes_an_independent_push_steer_max_deg():
    """config.yamlに独自のpush_steer_max_deg値を残さない(mpc.delta_max_degとの
    ドリフトを防ぐ、非冗長性)。"""
    assert "push_steer_max_deg:" not in _YAML_SRC


# ---------------------------------------------------------------------------
# 側選択(gate2再利用)の保存 + PUSH中の再評価
# ---------------------------------------------------------------------------

def test_compute_stuck_push_steer_stores_side_for_later_reuse():
    idx = _SRC.index("def _compute_stuck_push_steer(self, pose) -> float:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_push_side = 0" in snippet  # plan_ok=False/side不明の経路
    assert "self._stuck_push_side = _plan_side" in snippet  # 通常決定の経路
    # side不明の分岐がside決定の分岐より前に現れる(早期returnであることの確認)
    assert snippet.index("self._stuck_push_side = 0") < snippet.index(
        "self._stuck_push_side = _plan_side")


def test_push_state_initial_value_declared_in_init():
    idx = _SRC.index("    def _initialize(self) -> None:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_push_side = 0" in snippet


# ---------------------------------------------------------------------------
# ③検証ロギング/非矛盾性: 「実際に回避できたか」ベースの早期終了
# ---------------------------------------------------------------------------

def test_push_completion_checks_corr_bound_ahead_not_just_dist_or_timeout():
    idx = _SRC.index('elif self._stuck_state == "PUSH":')
    idx_end = _SRC.index("self._publish_control_command(now, u, acc, False)", idx)
    snippet = _SRC[idx:idx_end]
    # 2026-07-26追加(186節続報): 最小移動量要求(_dist_ok)が追加されたため、
    #   先頭が"_dist_ok and"に変わった(corr_bound_ahead判定自体は無変更)。
    assert "_cleared = (_dist_ok and self._stuck_push_side != 0" in snippet
    assert "self._corr_bound_ahead(self._stuck_push_side) > self._along_min_width" in snippet
    assert "or _cleared" in snippet


def test_push_completion_reason_distinguishes_cleared_from_backstop():
    """③検証ロギング: 実際に回避完了で抜けたのか、安全側バックストップ
    (距離到達/タイムアウト)で強制終了したのかを次回ログで区別できること。"""
    idx = _SRC.index('elif self._stuck_state == "PUSH":')
    idx_end = _SRC.index("self._publish_control_command(now, u, acc, False)", idx)
    snippet = _SRC[idx:idx_end]
    assert '"cleared" if _cleared' in snippet
    assert '"dist" if dist >= self._stuck_push_dist else "timeout"' in snippet
    assert "reason={_reason}" in snippet


def test_push_completion_is_a_single_call_site_not_duplicated():
    """②非冗長性: cleared/dist/timeoutの3条件はORで1つのif文にまとめられており、
    _stuck_recovery_complete()の呼び出し自体は重複していないこと。"""
    idx = _SRC.index('elif self._stuck_state == "PUSH":')
    idx_end = _SRC.index("self._publish_control_command(now, u, acc, False)", idx)
    snippet = _SRC[idx:idx_end]
    assert snippet.count("self._stuck_recovery_complete(") == 1


def test_cleared_check_reuses_existing_corr_bound_ahead_helper_no_new_room_calc():
    """②非冗長性: 168節で新設したcorr_bound_ahead()をそのまま再利用しており、
    新規の「空き幅」計算式を増やしていないこと。"""
    idx = _SRC.index('elif self._stuck_state == "PUSH":')
    idx_end = _SRC.index("self._publish_control_command(now, u, acc, False)", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._corr_bound_ahead(" in snippet
    assert snippet.count("self._corr_bound_ahead(") == 1
