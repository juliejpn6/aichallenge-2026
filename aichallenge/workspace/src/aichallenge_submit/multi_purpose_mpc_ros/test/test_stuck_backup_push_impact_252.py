"""BACKUP/PUSH中の衝突疑い(1周期急減速)検知(252節、2026-07-31)。

背景: v_max=20km/h予選ログ(0730_04/06)実測で、STUCK復帰のBACKUP中に後方の壁へ
衝突したとみられる急減速(v: -1.38->-0.51 m/s、1周期0.24秒)が発生した後も、
「後退不能検知」(実速度がblocked_v_thr=0.05m/s未満のまま1.5秒継続)がv≈-0.05
付近の境界値に張り付いて確定に時間がかかり、さらに実時間ウォッチドッグ
(backup_timeout_s=5.0秒)も同じ方向への再試行を2回繰り返してから初めて
シャッフル(方向転換)へ切り替わった。結果、1件のSTUCKエピソードで約27秒
(うち約14秒が同一方向への無駄な再試行)を要していた。

またユーザー指摘により、後退後の再発進(PUSH、前進での回避走行)には
BACKUPの「後退不能検知」に相当する早期離脱機構が元々無く、push_timeout_s
(短縮後2.5秒)まで無反応に前進し続ける盲点があることも判明した。

対処: 正面衝突検知に既にチューニング済みの`collision_suspect_dv`(0.8m/s、
1周期での速度急落閾値)を再利用し、「1周期での速度の大きさ(絶対値)の急減」を
BACKUP・PUSH双方で検知する。新規パラメータは追加しない。検知したら、BACKUPは
既存の後退不能ブロック処理(シャッフル/断念)へ即座に合流し、PUSHは即座に終了
してNORMALへ委譲する(PUSHにはBACKUPのようなシャッフル再試行機構が元々無い
ため、単純に既存のPUSH終了経路へ合流するのみ)。

さらに、0730_06実測(v: -1.38->-0.64->-0.12、drop=0.74m/s/周期で単発閾値
0.8m/s未満)で、単発の急落検知だけでは2〜3周期に分散した緩やかな減速を
捕捉できないと判明したため、正面衝突検知が既に持つ累積版
(collision_suspect_cum_dv=1.0m/s/collision_cum_window_cycles=5周期、新規
パラメータ0個)と全く同じ考え方を追加した(単発↔累積の2段構えは、正面衝突
検知(COLLISION-SUSPECTED/-CUM)と完全に同一のパターンの水平展開)。

合わせて、実時間ウォッチドッグ(backup_timeout_s/push_timeout_s)を5.0秒から
2.5秒へ短縮した(ユーザー指摘「タイムアウト5秒はもったいない」。今回の
即時衝突検知により、衝突相当の事態はこのタイムアウトより先に検知・離脱
できる想定のため、残りは緩やかな停滞のみを対象とする安全網として妥当な水準
に短縮する)。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装
した上でソーステキスト検証と組み合わせる(既存テストと同じ方針、
test_stuck_giveup_streak_186.py参照)。
"""
import os
from collections import deque

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG_SRC = _f.read()

BLOCKED_V_THR = 0.05
COLLISION_SUSPECT_DV = 0.8
COLLISION_SUSPECT_CUM_DV = 1.0
COLLISION_CUM_WINDOW_CYCLES = 5


def mirror_impact_detected(v_prev, v_now, blocked_v_thr=BLOCKED_V_THR,
                            collision_suspect_dv=COLLISION_SUSPECT_DV):
    """mpc_controller.pyの単発(1周期)急落判定式の純Pythonミラー
    (速度の符号に依らず「大きさの急減」を見る共通式)。"""
    if v_prev is None:
        return False
    if not (abs(v_prev) > blocked_v_thr):
        return False
    return (abs(v_prev) - abs(v_now)) >= collision_suspect_dv


def mirror_impact_over_sequence(v_sequence, blocked_v_thr=BLOCKED_V_THR,
                                 collision_suspect_dv=COLLISION_SUSPECT_DV,
                                 collision_suspect_cum_dv=COLLISION_SUSPECT_CUM_DV,
                                 window_cycles=COLLISION_CUM_WINDOW_CYCLES):
    """単発+累積(正面衝突検知のCOLLISION-SUSPECTED/-CUMと同一の2段構え)を
    毎周期のv系列に順次適用する純Pythonミラー。最初に検知が成立した
    周期のインデックス(0始まり)を返す。1件も検知しなければNoneを返す。"""
    v_prev = None
    window = deque(maxlen=window_cycles)
    for i, v_now in enumerate(v_sequence):
        impact = mirror_impact_detected(v_prev, v_now, blocked_v_thr, collision_suspect_dv)
        v_prev = v_now
        window.append(abs(v_now))
        if not impact and len(window) == window.maxlen:
            cum_drop = max(window) - abs(v_now)
            if cum_drop >= collision_suspect_cum_dv:
                impact = True
        if impact:
            return i
    return None


# ---------------------------------------------------------------------------
# ①非矛盾性: 検知式そのもの(数式の健全性、実測値による遡及検証を含む)
# ---------------------------------------------------------------------------

def test_first_cycle_no_prior_value_never_triggers():
    """v_prev=None(BACKUP/PUSH開始直後の初回周期)では誤検知しない。"""
    assert mirror_impact_detected(v_prev=None, v_now=0.0) is False


def test_not_yet_moving_meaningfully_does_not_trigger():
    """後退/前進の立ち上がり中(直前速度がまだblocked_v_thr以下)は、
    速度がさらに小さくなっても衝突ではなく単なる加速待ちのため誤検知しない。"""
    assert mirror_impact_detected(v_prev=0.02, v_now=0.0) is False


def test_retroactive_0730_04_backup_wall_impact_now_detected():
    """遡及検証: 0730_04ログ実測(t=1785414687.618->687.862、v: -1.38->-0.51、
    dist=1.52->1.75m)で観測された壁接触疑いの急減速が、この式で検知できる
    ことを確認する。"""
    assert mirror_impact_detected(v_prev=-1.38, v_now=-0.51) is True


def test_retroactive_0730_06_single_cycle_drop_alone_is_not_enough():
    """0730_06実測の単発落差(-1.38->-0.64、drop=0.74m/s)単独では単発閾値
    (0.8m/s)未満のため、単発判定だけでは検知できないことを確認する
    (だからこそ累積版が必要、というテスト設計)。"""
    assert mirror_impact_detected(v_prev=-1.38, v_now=-0.64) is False


def test_retroactive_0730_06_gradual_decel_detected_via_cumulative():
    """遡及検証: 0730_06ログ実測(v: -1.38->-0.64->-0.12->-0.07->-0.05、
    2〜3周期に分散した緩やかな減速)を、単発+累積の2段構えミラーへ通すと
    検知できることを確認する(実際に壁で足止めされる直前、通常巡航中の
    値も含めた最小5周期のウィンドウで検証)。"""
    v_seq = [-1.2, -1.3, -1.38, -0.64, -0.12, -0.07, -0.05]
    hit_idx = mirror_impact_over_sequence(v_seq)
    assert hit_idx is not None
    # 累積判定が成立するのは窓が5周期分埋まった後(インデックス4以降)。
    assert hit_idx >= 4


def test_cumulative_detection_needs_full_window_not_just_two_cycles():
    """累積版は窓(5周期)が埋まって初めて評価される。3周期分の緩やかな
    減速だけではまだ検知しない(実装がwindow満杯まで待つ設計であることの確認、
    誤検知を避けるための意図的な遅延)。"""
    v_seq = [-1.38, -0.64, -0.12]  # 3周期のみ、窓(5)未満
    assert mirror_impact_over_sequence(v_seq) is None


def test_cumulative_drop_below_threshold_does_not_trigger():
    """5周期そろっても、窓内最大値からの下落幅がcollision_suspect_cum_dv
    (1.0m/s)未満なら検知しない。"""
    v_seq = [1.0, 1.0, 1.0, 1.0, 0.5]  # 下落幅0.5m/s < 1.0
    assert mirror_impact_over_sequence(v_seq) is None


def test_forward_push_direction_symmetric_detection():
    """PUSH(前進、v>0)方向でも同じ式で対称に検知できることを確認する
    (符号反転の特別扱いを式自体に持たせていないことの確認)。"""
    assert mirror_impact_detected(v_prev=1.2, v_now=0.3) is True


def test_drop_just_below_threshold_does_not_trigger():
    assert mirror_impact_detected(v_prev=-1.38, v_now=-0.60) is False  # drop=0.78


def test_drop_exactly_at_threshold_triggers():
    """境界値ちょうど(0.8m/s)で確実に検知することを確認する。
    値の組み合わせによってはPython浮動小数点演算の丸め誤差
    (例: abs(-1.38)-abs(-0.58) == 0.7999999999999999 < 0.8)で
    意図せず閾値未満になり得るため、厳密に0.8を再現する値
    (-1.30, -0.50)を選んでいる(境界判定の>=自体は無変更)。"""
    assert abs(-1.30) - abs(-0.50) == 0.8  # 前提: 丸め誤差なしで厳密境界
    assert mirror_impact_detected(v_prev=-1.30, v_now=-0.50) is True  # drop=0.80


def test_normal_accel_ramp_up_does_not_trigger():
    """加速中(速度の大きさが増加していく通常の立ち上がり)は誤検知しない
    (dropが負値になり、閾値以上という条件を満たさない)。"""
    assert mirror_impact_detected(v_prev=-0.5, v_now=-0.9) is False


def test_small_cycle_to_cycle_noise_does_not_trigger():
    assert mirror_impact_detected(v_prev=-1.38, v_now=-1.30) is False


# ---------------------------------------------------------------------------
# ②非冗長性: 新規パラメータを追加せず既存のcollision_suspect_dv/
#   backup_blocked_v_thrを再利用していること
# ---------------------------------------------------------------------------

def test_backup_impact_reuses_existing_collision_suspect_dv_no_new_constant():
    idx = _SRC.index("_stuck_backup_impact = (")
    snippet = _SRC[idx:idx + 400]
    assert "self._collision_suspect_dv" in snippet
    assert "self._stuck_backup_blocked_v_thr" in snippet


def test_push_impact_reuses_same_thresholds_as_backup():
    idx = _SRC.index("_stuck_push_impact = (")
    snippet = _SRC[idx:idx + 400]
    assert "self._collision_suspect_dv" in snippet
    # PUSH専用の閾値は新設せず、BACKUP用のblocked_v_thrをそのまま流用する。
    assert "self._stuck_backup_blocked_v_thr" in snippet


# ---------------------------------------------------------------------------
# ③検証ロギング + ④配線確認: BACKUP側
# ---------------------------------------------------------------------------

def test_backup_impact_log_present():
    idx = _SRC.index('"[STUCK-BACKUP-IMPACT] v drop')
    assert idx > 0


def test_backup_impact_ored_into_blocked_elif():
    idx = _SRC.index(
        "elif _zero_v_elapsed >= self._stuck_backup_blocked_confirm_s or _stuck_backup_impact:")
    assert idx > 0


def test_backup_v_prev_updated_after_impact_check_each_cycle():
    """v_prevの更新(次周期比較用)が、今周期の判定に使った直後であることを
    確認する(先に更新してしまうと自分自身との比較になり常にFalseになる回帰を防ぐ)。"""
    idx_calc = _SRC.index("_stuck_backup_impact = (")
    idx_update = _SRC.index("self._stuck_backup_v_prev = _v_now")
    idx_zero_v_block = _SRC.index("if abs(_v_now) < self._stuck_backup_blocked_v_thr:")
    assert idx_calc < idx_update < idx_zero_v_block


def test_backup_v_prev_reset_on_fresh_backup_entry():
    """WAIT_REVERSE->BACKUP遷移(新規のBACKUP開始)のたびに、前回エピソードの
    v_prevを持ち越さないことを確認する(古い値が残ると初回周期で誤検知しうる)。"""
    idx = _SRC.index('self._stuck_state = "BACKUP"')
    snippet = _SRC[idx:idx + 700]
    assert "self._stuck_backup_v_prev = None" in snippet
    assert "self._stuck_backup_zero_v_since = None" in snippet


def test_backup_v_prev_declared_in_initialize():
    idx = _SRC.index("def _initialize(self) -> None:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "self._stuck_backup_v_prev = None" in snippet
    assert "self._stuck_push_v_prev = None" in snippet


def test_shuffle_log_includes_trigger_reason():
    """252節追加: impact起因の合流時はzero_v_elapsedが0.0のままのことがあるため、
    ログのトリガー種別(impact/sustained_zero_v)を明示することを確認する。"""
    idx = _SRC.index('_trigger_reason = "impact" if _stuck_backup_impact else "sustained_zero_v"')
    snippet = _SRC[idx:idx + 400]
    assert "reason={_trigger_reason}" in snippet


def test_backup_v_window_declared_with_shared_maxlen_in_initialize():
    """新規のウィンドウ長パラメータを増やさず、既存の正面衝突検知用
    collision_cum_window_cyclesをそのまま流用していることを確認する。"""
    idx = _SRC.index("self._collision_v_window = deque(maxlen=self._collision_cum_window_cycles)")
    snippet = _SRC[idx:idx + 900]
    assert "self._stuck_backup_v_window = deque(maxlen=self._collision_cum_window_cycles)" in snippet
    assert "self._stuck_push_v_window = deque(maxlen=self._collision_cum_window_cycles)" in snippet


def test_backup_v_window_reset_on_fresh_backup_entry():
    """新規BACKUP開始のたびに、前回エピソードの窓を持ち越さないことを確認する
    (古い値が残ると新規エピソード最初の数周期で誤検知しうる)。"""
    idx = _SRC.index('self._stuck_state = "BACKUP"')
    snippet = _SRC[idx:idx + 700]
    assert "self._stuck_backup_v_window.clear()" in snippet


def test_backup_cum_check_only_runs_when_single_cycle_check_did_not_already_fire():
    """単発判定が既に成立している周期では、累積判定を重複実行しない
    (ログ二重出力・②非冗長性の確認)。"""
    idx = _SRC.index("self._stuck_backup_v_window.append(abs(_v_now))")
    snippet = _SRC[idx:idx + 400]
    assert "if (not _stuck_backup_impact" in snippet


def test_backup_impact_cum_log_present():
    idx = _SRC.index('"[STUCK-BACKUP-IMPACT-CUM] v drop')
    assert idx > 0


def test_backup_cum_uses_max_of_window_minus_current_as_drop():
    """累積下落幅は「窓内最大値-現在値」で計算されることを確認する
    (単純な先頭-末尾の差ではなく、途中でピークを迎えるケースも捕捉するため)。"""
    idx = _SRC.index('"[STUCK-BACKUP-IMPACT-CUM] v drop')
    preceding = _SRC[max(0, idx - 300):idx]
    assert "_backup_cum_drop = max(self._stuck_backup_v_window) - abs(_v_now)" in preceding
    assert "_backup_cum_drop >= self._collision_suspect_cum_dv" in preceding


# ---------------------------------------------------------------------------
# ③検証ロギング + ④配線確認: PUSH側
# ---------------------------------------------------------------------------

def test_push_impact_log_present():
    idx = _SRC.index('"[STUCK-PUSH-IMPACT] v drop')
    assert idx > 0


def test_push_impact_included_in_exit_condition_with_dedicated_reason():
    idx = _SRC.index("if (dist >= self._stuck_push_dist or elapsed >= self._stuck_push_timeout_s")
    snippet = _SRC[idx:idx + 400]
    assert "_stuck_push_impact" in snippet
    assert '_reason = ("impact" if _stuck_push_impact' in snippet


def test_push_v_prev_updated_after_impact_check_each_cycle():
    idx_calc = _SRC.index("_stuck_push_impact = (")
    idx_update = _SRC.index("self._stuck_push_v_prev = v_now")
    idx_gap_chk = _SRC.index("_dist_ok = dist >= self._stuck_push_min_dist_for_cleared")
    assert idx_calc < idx_update < idx_gap_chk


def test_push_v_prev_reset_on_fresh_push_entry():
    idx = _SRC.index('self._stuck_state = "PUSH"')
    snippet = _SRC[idx:idx + 400]
    assert "self._stuck_push_v_prev = None" in snippet
    assert "self._stuck_push_start = (pose.x, pose.y)" in snippet


def test_push_v_window_reset_on_fresh_push_entry():
    idx = _SRC.index('self._stuck_state = "PUSH"')
    snippet = _SRC[idx:idx + 400]
    assert "self._stuck_push_v_window.clear()" in snippet


def test_push_cum_check_only_runs_when_single_cycle_check_did_not_already_fire():
    idx = _SRC.index("self._stuck_push_v_window.append(abs(v_now))")
    snippet = _SRC[idx:idx + 400]
    assert "if (not _stuck_push_impact" in snippet


def test_push_impact_cum_log_present():
    idx = _SRC.index('"[STUCK-PUSH-IMPACT-CUM] v drop')
    assert idx > 0


def test_push_cum_uses_max_of_window_minus_current_as_drop():
    idx = _SRC.index('"[STUCK-PUSH-IMPACT-CUM] v drop')
    preceding = _SRC[max(0, idx - 300):idx]
    assert "_push_cum_drop = max(self._stuck_push_v_window) - abs(v_now)" in preceding
    assert "_push_cum_drop >= self._collision_suspect_cum_dv" in preceding


# ---------------------------------------------------------------------------
# config.yaml: タイムアウト短縮(5.0->2.5秒)
# ---------------------------------------------------------------------------

def test_backup_and_push_timeout_shortened_to_2_5s():
    assert "backup_timeout_s: 2.5" in _CFG_SRC
    assert "push_timeout_s: 2.5" in _CFG_SRC
    assert "backup_timeout_s: 5.0" not in _CFG_SRC
    assert "push_timeout_s: 5.0" not in _CFG_SRC


def test_code_side_fallback_default_unaffected_by_config_change():
    """config.yamlが優先されるため、_stkgetのPython側フォールバック既定値
    (5.0)自体は変更不要であることを確認する(config.yaml不在時の後方互換、
    既存の設計方針を踏襲)。"""
    assert 'float(_stkget("backup_timeout_s", 5.0))' in _SRC
    assert 'float(_stkget("push_timeout_s", 5.0))' in _SRC


# ---------------------------------------------------------------------------
# ④遡及効果: 既存の後退不能検知(sustained zero-v)自体は無変更
# ---------------------------------------------------------------------------

def test_existing_sustained_zero_v_detection_untouched():
    idx = _SRC.index("if abs(_v_now) < self._stuck_backup_blocked_v_thr:")
    snippet = _SRC[idx:idx + 260]
    assert "self._stuck_backup_zero_v_since = now" in snippet
    assert "_zero_v_elapsed = (now - self._stuck_backup_zero_v_since).nanoseconds / 1e9" in snippet


def test_existing_push_cleared_check_untouched():
    idx = _SRC.index("_dist_ok = dist >= self._stuck_push_min_dist_for_cleared")
    snippet = _SRC[idx:idx + 700]
    assert "_gap_width_chk > self._along_min_width" in snippet
    assert "_psi_err_chk < self._stuck_push_heading_tol_rad" in snippet
