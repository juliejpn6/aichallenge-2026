"""Regression tests for G2-RELEASE(_side_clear)の非対称デバウンス(104節, 2026-07-18).

背景: 0718実測(ローカル3台走行)で、wp≈269〜270(相手までfwd_ds≈1.94mという近距離)
において、[G2-RELEASE]がわずか0.5秒間に7回ON/OFFを繰り返し、その都度
eff_v_cap(≈4.17m/s)⇔icc_f3(≈1.5m/s、F3-TAPERクリープ床)という全く別の計算式を
切り替えていた。これと同時刻に[COLLISION-SUSPECTED-CUM]が3.30m/sの速度低下
(4.00→2.09→1.13→0.70→0.70)を記録している。

根本原因: `_actual_lat_clear_now`(fwd_dlat>=along_min_width)と`_side_room_ok_now`
(side_room>=along_min_width)がいずれも単一閾値でデバウンス無しのままORで
`_side_clear`を決めており、fwd_dlatが閾値付近でノイズにより往復するたびに
即座に反映されていた。

対処(ユーザー承認済み設計、3案中の案1採用): 同じファイル内の_def_active
(2994〜3005行目)と同じ「ON方向は連続確認・OFF方向は即時反映」という非対称
デバウンスを適用した。ON(解放)方向のみ既存のengage_debounce(8周期、新規
パラメータ0個)分の連続確認を要求し、OFF(安全側=再度キャップをかける方向)は
瞬時値がFalseになった時点で即座に反映する。

テスト方針: デバウンスの状態機械自体は自己完結したロジックのため、実物と
同一のアルゴリズムを複製したミラークラスで数式的性質を検証する。
mpc_controller.py側の配線(_ot_engage_debounceの再利用・OFF即時反映)は
末尾の構造的ソーステキスト検証で確認する。
"""
import os


class _G2DebounceMirror:
    """mpc_controller.py の _side_clear デバウンスロジック(104節)の複製ミラー。"""

    def __init__(self, enter_cycles=8):
        self.enter_cycles = enter_cycles
        self.on_count = 0
        self.debounced = False

    def update(self, raw):
        if raw:
            self.on_count += 1
        else:
            self.on_count = 0
            self.debounced = False
        if not self.debounced and self.on_count >= self.enter_cycles:
            self.debounced = True
        return self.debounced


def test_stays_off_while_raw_is_false():
    m = _G2DebounceMirror(enter_cycles=8)
    for _ in range(20):
        assert m.update(False) is False


def test_does_not_latch_on_before_enter_cycles_reached():
    """核心: ON方向はenter_cycles未満の連続確認では成立しない。"""
    m = _G2DebounceMirror(enter_cycles=8)
    for _ in range(7):
        assert m.update(True) is False


def test_latches_on_exactly_at_enter_cycles():
    """核心: ちょうどenter_cycles回連続でTrueになった周期でONへ確定する。"""
    m = _G2DebounceMirror(enter_cycles=8)
    for _ in range(7):
        m.update(True)
    assert m.update(True) is True


def test_off_direction_is_instantaneous_regression():
    """核心(安全方向): 一度ONへ確定した後でも、瞬時値が1回でもFalseになれば
    即座にOFFへ戻る(デバウンスしない、安全側の反応速度を犠牲にしない)。"""
    m = _G2DebounceMirror(enter_cycles=8)
    for _ in range(8):
        m.update(True)
    assert m.debounced is True
    assert m.update(False) is False
    assert m.on_count == 0


def test_retroactive_0718_chatter_pattern_now_stays_off():
    """遡及検証(0718実測、wp≈269〜270): fwd_dlatが閾値付近で往復し、旧実装では
    0.5秒間(約20周期@40Hz)に7回ON/OFFが切り替わっていた(=連続True区間の
    平均長は3周期未満)ことを模した合成系列。enter_cycles=8では、このような
    短い連続True区間ではどれも8周期に到達できず、デバウンス後はOFFのまま
    安定し、eff_v_cap/icc_f3間の切り替えが起きない。"""
    m = _G2DebounceMirror(enter_cycles=8)
    # 3周期True→2周期False、を4セット(=20周期でON/OFFが多数回入れ替わる
    # 0718実測相当のパターン)。
    pattern = ([True, True, True, False, False] * 4)
    results = [m.update(v) for v in pattern]
    assert all(r is False for r in results)


def test_sustained_clear_still_releases_eventually():
    """回帰: 本当に十分な時間(enter_cycles以上)クリアが持続すれば、従来通り
    解放される(安全に倒しすぎて永久にONにならない、ということはない)。"""
    m = _G2DebounceMirror(enter_cycles=8)
    results = [m.update(True) for _ in range(15)]
    assert results[7:] == [True] * 8


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_g2_debounce_reuses_existing_engage_debounce_no_new_parameter():
    idx = _SRC.index("_side_clear_raw = (")
    snippet = _SRC[idx:idx + 900]
    assert "self._ot_engage_debounce" in snippet


def test_g2_debounce_off_direction_resets_immediately():
    idx = _SRC.index("_side_clear_raw = (")
    snippet = _SRC[idx:idx + 900]
    assert "self._g2_clear_on_count = 0" in snippet
    assert "self._g2_release_debounced = False" in snippet


def test_g2_clear_state_initialized():
    assert "self._g2_clear_on_count = 0" in _SRC
    assert "self._g2_release_debounced = False" in _SRC


def test_side_clear_downstream_uses_debounced_value_regression():
    """回帰防止: G2-RELEASE判定は`_side_clear = self._g2_release_debounced`
    (デバウンス後の値、生値_side_clear_rawではない)をそのままreturnし、
    _control()側の消費箇所(`elif self._g2_release_ready(...)`)は戻り値を
    直接ifへ渡すのみで生値へアクセスする経路を持たないことを確認する。
    2026-07-20修正(143節続報、スリム化): _g2_release_readyへ抽出後、
    生値(_side_clear_raw)はメソッドの外から一切参照できない構造になった
    (以前より強い保証)。"""
    idx = _SRC.index("_side_clear = self._g2_release_debounced")
    snippet = _SRC[idx:idx + 1200]
    assert "return _side_clear" in snippet
    idx_call = _SRC.index("elif self._g2_release_ready(")
    call_snippet = _SRC[idx_call:idx_call + 400]
    assert '_v_safe_cand.append(("eff_v_cap' in call_snippet
    # 生値は_g2_release_readyの外(_control()側)からは一切見えない
    assert "_side_clear_raw" not in _SRC[idx_call:idx_call + 5000]


# ---------------------------------------------------------------------------
# 108節: stopped_bypassへのalpha経路追加(67節「自己参照ループ」の部分的再発への対処)
#
# 背景: 67節が特定した「floorが低い→前進できない→dlatが育たない→floorが低い
# まま」という自己参照ループが、68節の緊急修正(0715-01の実追突事故を防ぐための
# _actual_lat_clear_now追加)により部分的に再発することを0718-06実測(予選)で
# 確認した(1周目コーナー3でfwd_dlatが35秒間along_min_width未満に張り付いた)。
#
# 対処: fwd_dlat(速度に依存し、ループの内側にある信号)に加え、
# self._ot_alpha>=1.0-1e-3(ramp_timeと時間だけで決まり、ループの外側にある信号)
# を代替条件としてOR追加する。0714-02節で全く同じ閾値・同じ考え方
# (「実際に新側への横移動が完了して初めて真横到達を認める」)が既に導入されて
# おり、新規パラメータは0個。0715-01の危険な瞬間(エンゲージ直後、alpha≈0)は
# H3ガード2(switchback/再エンゲージ時のalpha=0リセット)により引き続きブロック
# される。
# ---------------------------------------------------------------------------

def _stopped_bypass_mirror(stopped_opponent, side_room_ok_now, actual_lat_clear_now,
                            alpha):
    """mpc_controller.py の stopped_bypass 判定式(108節)の複製ミラー。"""
    offset_committed = alpha >= 1.0 - 1e-3
    return stopped_opponent and side_room_ok_now and (
        actual_lat_clear_now or offset_committed)


def test_offset_committed_releases_even_when_dlat_still_small():
    """核心: fwd_dlatがalong_min_width未満でも、alpha≈1(オフセットランプ完了)
    ならstopped_bypassが成立する。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=True,
        actual_lat_clear_now=False, alpha=1.0) is True


def test_dlat_path_still_works_regression():
    """回帰: 従来通り、alphaが低くてもfwd_dlatが十分ならstopped_bypassは成立する
    (68節の経路は無変更)。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=True,
        actual_lat_clear_now=True, alpha=0.0) is True


def test_retroactive_0715_01_fresh_engage_still_blocked():
    """遡及検証(0715-01実測、追突事故発生時): エンゲージ直後でalpha≈0・
    fwd_dlat≈0.24(along_min_width未満)だった実測条件を再現する。alpha経路を
    追加した後も、この危険な瞬間は引き続きブロックされる。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=True,
        actual_lat_clear_now=False, alpha=0.0) is False


def test_retroactive_0718_06_lap1_corner3_now_releases():
    """遡及検証(0718-06実測、1周目コーナー3): offset=-3.0(alpha=1.0相当)が
    30秒以上持続したにもかかわらずfwd_dlatが0.6m前後(along_min_width未満)に
    張り付いていた実測条件を再現する。alpha経路の追加により、この張り付きは
    解消される。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=True,
        actual_lat_clear_now=False, alpha=1.0) is True


def test_offset_committed_alone_does_not_bypass_side_room_check_regression():
    """回帰(安全性): alphaが1でも、side_room_ok_now(壁基準の物理的な空き)が
    Falseならstopped_bypassは成立しない(コリドーが実際に狭い場面の保護は不変)。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=False,
        actual_lat_clear_now=False, alpha=1.0) is False


def test_offset_committed_alone_does_not_bypass_stopped_opponent_check_regression():
    """回帰(安全性): alphaが1でも、相手が停止/低速でなければ(stopped_opponent=False)
    stopped_bypassは成立しない(走行中の相手への保護は不変)。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=False, side_room_ok_now=True,
        actual_lat_clear_now=False, alpha=1.0) is False


def test_offset_committed_just_below_threshold_does_not_trigger():
    """境界値: alphaがわずかに1.0-1e-3を下回る場合はoffset_committedとして
    扱われない(0714-02節と同一の境界)。"""
    assert _stopped_bypass_mirror(
        stopped_opponent=True, side_room_ok_now=True,
        actual_lat_clear_now=False, alpha=1.0 - 1e-3 - 1e-6) is False


def test_offset_committed_reuses_same_threshold_as_0714_02_ot_cleared_gate():
    """②非冗長性の確認: 108節の新条件が、0714-02節で_ot_clearedゲートに
    既に使われている閾値(1.0 - 1e-3)と完全に同一の式であることをソース上で
    確認する(新規チューニング値ではなく既存パターンの再利用)。"""
    idx_new = _SRC.index("_offset_committed = self._ot_alpha >= 1.0 - 1e-3")
    idx_existing = _SRC.index("if self._ot_alpha >= 1.0 - 1e-3:")
    assert idx_new > 0
    assert idx_existing > 0


def test_offset_committed_computed_before_side_clear_raw():
    idx_offset = _SRC.index("_offset_committed = self._ot_alpha >= 1.0 - 1e-3")
    idx_raw = _SRC.index("_side_clear_raw = (")
    assert idx_offset < idx_raw


def test_offset_committed_ored_with_actual_lat_clear_now_in_stopped_bypass():
    idx = _SRC.index("_side_clear_raw = (")
    snippet = _SRC[idx:idx + 900]
    assert "(_actual_lat_clear_now or _offset_committed)" in snippet


def test_g2_release_log_includes_offset_committed_fields():
    idx = _SRC.index('f"[G2-RELEASE]')
    snippet = _SRC[idx:idx + 1200]
    assert "offset_committed=" in snippet
    assert "alpha=" in snippet
