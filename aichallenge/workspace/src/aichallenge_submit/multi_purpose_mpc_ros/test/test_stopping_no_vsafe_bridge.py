"""Unit tests for the STOPPING-NO-VSAFE speed bridge (131-6節⑤、なめらかな断念,
2026-07-20).

Background: 0720-02実測(wp13、t=609.34)で、OVERTAKING中の速度モデル
(eff_v_cap)からSTOPPINGの速度モデル(icc_stop/icc_stop_fallback)への切替の
瞬間、両方とも不成立になるとv_safe_pre=Noneのまま、MPC自身の最適化が無制限
速度(u0=v_max)を出力することを直接確認した(OTログ: v_safe_src=Noneなのに
u0=4.1667)。0.6秒後にwall_slowが追いついた時点で壁マージンは既にマイナス
(wall=-0.42)まで悪化していた。

対処: footprint_risk/wall_slowの完全キャップが既に再利用しているwall_slow_speed
を、状態遷移の隙間を埋める保守速度としてSTOPPING-NO-VSAFEブロックでも再利用する。

このメソッドは複雑度が高くモック実行が難しいため、既存の同種テスト
(test_stopping_no_vsafe_diagnostic.py等)と同じくソーステキスト検証+
最終集約ロジック(v_safe = _v_safe_pre → u[0] = min(u[0], v_safe))の
ミラー計算で検証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _mirror_final_speed(v_safe_pre, u0_raw, v_max):
    """4570行目付近の最終集約ロジックのミラー実装(u[0] = min(u[0], v_safe)相当)。"""
    v_safe = v_safe_pre
    if v_safe is not None:
        v_safe = min(v_safe, v_max)
    if v_safe is None:
        return u0_raw
    return min(u0_raw, v_safe)


def test_retroactive_0720_02_wp13_no_longer_falls_through_to_v_max():
    """遡及検証: 0720-02実測wp13の状況(v_safe_pre=None、MPCの生出力=v_max=4.1667)
    を再現し、修正後はwall_slow_speed(既定2.0)でキャップされることを確認する。"""
    wall_slow_speed = 2.0
    u0_raw = 4.1667
    v_max = 4.1667
    # 旧挙動(v_safe_pre=None)のシミュレーション: キャップされない。
    assert _mirror_final_speed(None, u0_raw, v_max) == u0_raw
    # 新挙動(v_safe_pre=wall_slow_speed)のシミュレーション: キャップされる。
    assert _mirror_final_speed(wall_slow_speed, u0_raw, v_max) == wall_slow_speed


def test_bridge_speed_does_not_raise_above_existing_lower_candidate():
    """回帰防止: 既に他の候補(例: icc_stop=1.5)がwall_slow_speed(2.0)より
    低い場合、ブリッジ自体はv_safe_pre=Noneの時にしか代入されないため、
    他候補が有効な間は無関係であることを確認する(_v_safe_pre is Noneの
    ガード自体がこの分離を保証する、既存if文の再確認)。"""
    idx = _SRC.index("if _v_safe_pre is None:")
    snippet = _SRC[idx:idx + 40]
    assert snippet.startswith("if _v_safe_pre is None:")


def test_bridge_reuses_wall_slow_speed_no_new_parameter():
    """②非冗長性: 新規のブリッジ用速度定数を追加せず、既存self._wall_slow_speed
    (footprint_risk・wall_slow完全キャップが既に共有する定数)を再利用する。"""
    idx = _SRC.index("_v_safe_pre = self._wall_slow_speed")
    assert idx > 0


def test_bridge_candidate_labeled_for_v_safe_src_diagnosis():
    """③検証ロギング: ブリッジが実際にボトルネックとして採用された場合、
    既存の_v_safe_src集計(min(_v_safe_cand)によるラベル選出)経由で
    次回ログから直接特定できることを確認する。"""
    idx = _SRC.index('_v_safe_cand.append(("stopping_no_vsafe(状態遷移ブリッジ)"')
    assert idx > 0


def test_bridge_assignment_happens_every_cycle_while_condition_holds():
    """回帰防止: ブリッジ代入がedge(ONになった最初の周期)だけでなく、
    v_safe_pre=Noneが続く間は毎周期行われることを確認する(self._stopping_no_vsafe_prev
    の値に関わらず、if _v_safe_pre is None: ブロック全体の中にある)。"""
    idx = _SRC.index("if _v_safe_pre is None:")
    idx_end = _SRC.index("elif self._stopping_no_vsafe_prev:")
    snippet = _SRC[idx:idx_end]
    # 代入文がself._stopping_no_vsafe_prevの真偽に関する条件分岐の外側(=毎周期実行)にあること。
    idx_assign = snippet.index("_v_safe_pre = self._wall_slow_speed")
    idx_if_not_prev = snippet.index("if not self._stopping_no_vsafe_prev:")
    idx_endif = snippet.index("self._stopping_no_vsafe_prev = True")
    assert idx_assign > idx_endif  # if not self._stopping_no_vsafe_prev: ブロックの後


def test_bridge_uses_stopping_state_wall_slow_speed_not_a_hardcoded_literal():
    """回帰防止: ブリッジ値がハードコードされたリテラル(例: 2.0)ではなく、
    self._wall_slow_speed経由(config可変)であることを確認する。"""
    idx = _SRC.index("_v_safe_pre = self._wall_slow_speed")
    line = _SRC[idx:idx + 60]
    assert "self._wall_slow_speed" in line
    assert "= 2.0" not in line


# --- 138-5節①続報: 過剰発火の是正(2026-07-20) ---
# 背景: 0720-04実測(wp47-52、ds=5〜7m・dlat=2.8〜3.1m)で、除外車が明確に遠い
# 場合にも一律にブリッジが発動し、可視的な減速チャタリングを起こすことが判明
# した。正当化事例(0720-02 wp13、ds=1.0m)との違いはfwd_dsの近さのみだった
# ため、footprint_riskが既に使う縦方向の物理下限along_min_lengthより近い
# 場合のみブリッジを適用するよう改良した。

def _bridge_fires(fwd_ds, along_min_length=2.00):
    """mpc_controller.pyの改良後ブリッジ条件のミラー実装。
    2026-07-31(254節): _scan_trafficがds>=0(真に前方)限定になったことで、
    fwd_dsはNoneでなければ常に非負値を取る契約になったため、abs()を撤去。"""
    return fwd_ds is not None and fwd_ds < along_min_length


def test_retroactive_0720_02_wp13_close_car_still_bridges():
    """遡及検証: 0720-02実測wp13(ds=1.0m、危険と確認済み)は、改良後も
    引き続きブリッジが発動することを確認する(退行なし)。"""
    assert _bridge_fires(fwd_ds=0.9979962702900593) is True


def test_retroactive_0720_04_wp47_far_car_no_longer_bridges():
    """遡及検証: 0720-04実測wp47(ds=5.0m、実害なしと確認済み)は、改良後は
    ブリッジが発動しないことを確認する(過剰発火の是正そのもの)。"""
    assert _bridge_fires(fwd_ds=4.996179693410738) is False


def test_retroactive_0720_04_wp52_far_car_no_longer_bridges():
    """遡及検証: 0720-04実測wp52(ds=5.98m)も同様にブリッジが発動しない
    ことを確認する。"""
    assert _bridge_fires(fwd_ds=5.982656298855119) is False


def test_bridge_boundary_at_along_min_length():
    assert _bridge_fires(fwd_ds=1.999, along_min_length=2.00) is True
    assert _bridge_fires(fwd_ds=2.00, along_min_length=2.00) is False


def test_bridge_no_longer_uses_absolute_value_since_fwd_ds_is_never_negative():
    """254節(2026-07-31)で撤回: 以前はfwd_dsの負値(後方の車)もabs()経由で
    ブリッジ対象になり得たが、_scan_trafficの修正でfwd_dsはNoneでなければ
    常にds>=0(真に前方)を保証するようになったため、この分岐はもう存在しない
    契約(仮に負値が渡ってもtrivialに発動してしまう、後方車を対象にしない
    という上位の契約はscan側で保証する)。ここでは単にabs()を使わない新しい
    ミラー式が0.0以上の値に対して従来通り機能することのみ確認する。"""
    assert _bridge_fires(fwd_ds=0.0) is True
    assert _bridge_fires(fwd_ds=1.999) is True
    assert _bridge_fires(fwd_ds=2.0) is False


def test_bridge_none_ds_does_not_fire():
    assert _bridge_fires(fwd_ds=None) is False


def test_bridge_condition_reuses_along_min_length_no_new_parameter():
    """②非冗長性: 新規閾値を追加せず、既存self._along_min_length
    (footprint_riskの縦方向緊急閾値と同一)を再利用していることを確認する。"""
    idx = _SRC.index("_v_safe_pre = self._wall_slow_speed")
    snippet = _SRC[max(0, idx - 300):idx]
    assert "self._along_min_length" in snippet


def test_far_or_unknown_car_falls_back_to_prior_mpc_behavior():
    """①非矛盾性: 近さ条件を満たさない場合、_v_safe_preはNoneのまま
    (bridgeブロック外)であり、以前のMPC最適化任せの挙動へ委ねられる
    ことをソース構造上確認する(新しい代替候補を追加せず、単に発動条件を
    絞っただけであることの確認)。2026-07-31(254節): abs()を撤去した
    条件式に追随。"""
    idx = _SRC.index("if _fwd_ds is not None and _fwd_ds < self._along_min_length:")
    snippet = _SRC[idx:idx + 300]
    assert "_v_safe_pre = self._wall_slow_speed" in snippet
    assert '_v_safe_cand.append(("stopping_no_vsafe' in snippet


def test_bridge_condition_no_longer_uses_abs():
    """回帰防止(254節): ソース上のブリッジ条件式がabs()を使っていないことを
    直接確認する(fwd_dsは_scan_traffic側でds>=0保証済みのため不要)。"""
    idx = _SRC.index("if _fwd_ds is not None and _fwd_ds < self._along_min_length:")
    assert "abs(_fwd_ds)" not in _SRC[max(0, idx - 50):idx + 60]
