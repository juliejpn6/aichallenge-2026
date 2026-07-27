"""Unit tests for 190-3節(2026-07-26): コリドー境界ラチェットの拡大方向レートリミット。

背景: `core/reference_path.py`の`update_path_constraints()`内`add_constraint()`は、
毎周期新しく計算したub_sm/lb_smを、前回の`wp.ub_sm`/`wp.lb_sm`より広げない(縮小方向
にのみ即座にラチェットする)。この拡大方向の解除は`reset_dynamic_constraints()`が
呼ばれた時のみで、それは`mpc_controller.py`の`_rebuild_deadband`(0.3m以上の障害物
移動)にのみ連動している。80節(2026-07-16)はこのラチェットが「STUCK復帰完了時にも
解除する」形で部分対処されたが、通常走行中(rebuild_deadband解除の瞬間)に蓄積された
狭まり分が一気に解消される不連続ジャンプ自体は未対処のままだった。

実測(0720-02/0726-02/0726-04、直線+単独走行区間、同一手法): corr_ub0/lb0の
1サンプルあたりのジャンプが平均0.14〜0.32m・最大0.94mに達し、QPコリドー(実軌道)・
wall_slow・switchback(new_side_wall_blocked)・オフセット目標クランプ
(_corr_bound_ahead)へ同時に伝播していた。

対処: 縮小方向(安全側)は無変更のまま即座に反映し、拡大方向のみ`corridor_widen_step_m`
(1周期あたりの最大回復量[m])でレートリミットする。新規パラメータは
`corridor_widen_rate_mps`(config.yaml、既定1.0 m/s)の1個のみ。`ReferencePath`側の
既定値はfloat('inf')(=無制限、従来と完全に同一の挙動)とし、`mpc_controller.py`が
`__init__`と`_set_active_path()`の両方で実値を注入する。

`add_constraint`は`update_path_constraints`内のクロージャで直接呼び出せないため、
①ラチェット比較式そのものをミラー実装で検証し、②ソーステキスト検証で実際の修正箇所
(80節と同様の既存手法)を確認する。
"""
import os

import yaml

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "reference_path.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CTRL_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_CTRL_SRC_PATH) as _f:
    _CTRL_SRC = _f.read()

_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def mirror_ub_sm(prev_ub_sm, fresh_ub_sm, widen_step_m):
    """add_constraint内のub_sm側ラチェット比較のミラー。"""
    if prev_ub_sm < fresh_ub_sm:
        return min(fresh_ub_sm, prev_ub_sm + widen_step_m)
    return fresh_ub_sm


def mirror_lb_sm(prev_lb_sm, fresh_lb_sm, widen_step_m):
    """add_constraint内のlb_sm側ラチェット比較のミラー(符号反転)。"""
    if prev_lb_sm > fresh_lb_sm:
        return max(fresh_lb_sm, prev_lb_sm - widen_step_m)
    return fresh_lb_sm


# ---------------------------------------------------------------------------
# ①非矛盾性: 縮小方向は無制限・即座、拡大方向のみレートリミット
# ---------------------------------------------------------------------------

def test_narrowing_ub_is_instant_regardless_of_widen_step():
    """縮小(新値がprevより狭い)は widen_step が小さくても無条件で即座に反映される。"""
    result = mirror_ub_sm(prev_ub_sm=3.0, fresh_ub_sm=0.5, widen_step_m=0.01)
    assert result == 0.5


def test_narrowing_lb_is_instant_regardless_of_widen_step():
    result = mirror_lb_sm(prev_lb_sm=-3.0, fresh_lb_sm=-0.5, widen_step_m=0.01)
    assert result == -0.5


def test_widening_ub_is_capped_by_widen_step():
    """拡大(新値がprevより広い)は1周期でwiden_step分しか進めない。"""
    result = mirror_ub_sm(prev_ub_sm=0.5, fresh_ub_sm=3.0, widen_step_m=0.3)
    assert result == 0.5 + 0.3


def test_widening_lb_is_capped_by_widen_step():
    result = mirror_lb_sm(prev_lb_sm=-0.5, fresh_lb_sm=-3.0, widen_step_m=0.3)
    assert result == -0.5 - 0.3


def test_widening_within_budget_reaches_target_in_one_cycle():
    """拡大幅がwiden_step以下なら、その周期内で目標値まで完全到達する(不要な遅延なし)。"""
    result = mirror_ub_sm(prev_ub_sm=0.5, fresh_ub_sm=0.6, widen_step_m=0.3)
    assert result == 0.6


# ---------------------------------------------------------------------------
# ②非冗長性: 既定値(inf)では従来と完全に同一の挙動(後方互換性)
# ---------------------------------------------------------------------------

def test_default_inf_step_reproduces_old_unconditional_override_behavior():
    """corridor_widen_step_m既定値(inf)なら、拡大方向も従来通り新値がそのまま通る
    (このテストがinfの場合に限りcappingが一切効かないことを保証する)。"""
    assert mirror_ub_sm(prev_ub_sm=0.1, fresh_ub_sm=999.0, widen_step_m=float("inf")) == 999.0
    assert mirror_lb_sm(prev_lb_sm=-0.1, fresh_lb_sm=-999.0, widen_step_m=float("inf")) == -999.0


# ---------------------------------------------------------------------------
# 複数周期での収束(実測ジャンプ規模での妥当性確認)
# ---------------------------------------------------------------------------

def test_multi_cycle_convergence_matches_reasoned_recovery_time():
    """実測典型ジャンプ(0.3m)を、提案初期値(corridor_widen_rate_mps=1.0, control_rate=40Hz
    ⇒ widen_step_m=0.025)でシミュレートし、約0.3秒(12周期)で完全収束することを確認する。"""
    control_rate_hz = 40.0
    widen_step_m = 1.0 / control_rate_hz
    prev = 0.5
    target = 0.8  # 0.3mの典型ジャンプ
    cycles = 0
    while prev < target and cycles < 100:
        prev = mirror_ub_sm(prev, target, widen_step_m)
        cycles += 1
    assert prev == target
    assert cycles == 12  # 0.3 / 0.025 = 12周期 ≈ 0.3秒


def test_new_narrowing_interrupts_widen_ramp_immediately_no_hunting():
    """③ハンチング防止: 拡大レートリミット中に新たな縮小イベントが来ても、
    縮小は即座に優先される(拡大リミットと綱引きにならない)。"""
    control_rate_hz = 40.0
    widen_step_m = 1.0 / control_rate_hz
    prev = mirror_ub_sm(0.5, 0.8, widen_step_m)  # 拡大リミット中(0.525)
    assert prev < 0.8
    # 途中で相手が接近し縮小イベントが来る
    prev = mirror_ub_sm(prev, 0.3, widen_step_m)
    assert prev == 0.3  # 即座に反映、拡大リミットの残り分は関係ない


# ---------------------------------------------------------------------------
# ソーステキスト検証: 実際の修正箇所(core/reference_path.py)
# ---------------------------------------------------------------------------

def test_reference_path_default_widen_step_is_inf():
    assert "self.corridor_widen_step_m = float('inf')" in _SRC


def test_add_constraint_ub_ratchet_rate_limited():
    idx = _SRC.index("ub_sm = ub - safety_margin")
    snippet = _SRC[idx:idx + 800]
    assert "ub_sm = min(ub_sm, wp.ub_sm + self.corridor_widen_step_m)" in snippet
    # 旧: 無条件上書き(wp.ub_smのみ)の形が残っていないこと
    assert "ub_sm = wp.ub_sm\n" not in snippet


def test_add_constraint_lb_ratchet_rate_limited():
    idx = _SRC.index("ub_sm = ub - safety_margin")
    snippet = _SRC[idx:idx + 800]
    assert "lb_sm = max(lb_sm, wp.lb_sm - self.corridor_widen_step_m)" in snippet
    assert "lb_sm = wp.lb_sm\n" not in snippet


def test_reset_dynamic_constraints_itself_unchanged_80節回帰防止():
    """④遡及効果: 80節のreset_dynamic_constraints()自体(縮小方向の解除ロジック)は
    本修正の対象外であり、無変更のまま残っていることを確認する。"""
    idx = _SRC.index("def reset_dynamic_constraints(")
    idx_end = _SRC.index("def set_v_ref(")
    snippet = _SRC[idx:idx_end]
    assert "wp.ub_sm = copy.deepcopy(wp.ub)" in snippet
    assert "wp.lb_sm = copy.deepcopy(wp.lb)" in snippet


# ---------------------------------------------------------------------------
# ソーステキスト検証: mpc_controller.py側の配線
# ---------------------------------------------------------------------------

def test_controller_computes_widen_step_from_config_before_ref_path_construction():
    idx = _CTRL_SRC.index("self._corridor_widen_step_m = (")
    idx_ref = _CTRL_SRC.index("self._reference_path = create_ref_path(self._map)")
    assert idx < idx_ref  # widen_step確定がreference_path構築より先
    snippet = _CTRL_SRC[idx:idx + 300]
    assert '"corridor_widen_rate_mps", 1.0' in snippet
    assert "self._cfg.mpc.control_rate" in snippet


def test_controller_sets_widen_step_at_initial_construction():
    idx = _CTRL_SRC.index("self._reference_path = create_ref_path(self._map)")
    snippet = _CTRL_SRC[idx:idx + 200]
    assert "self._reference_path.corridor_widen_step_m = self._corridor_widen_step_m" in snippet


def test_controller_sets_widen_step_at_path_swap():
    """④遡及効果: _set_active_path()での経路差し替え(ピット等)後も同じ拡大レート
    リミットが適用されること(新規ReferencePathインスタンスは既定値infに戻るため、
    ここで設定し忘れると経路差し替え後だけ本対処が無効化される)。"""
    idx = _CTRL_SRC.index("def _set_active_path(")
    idx_end = idx + 600
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._reference_path = path" in snippet
    assert "self._reference_path.corridor_widen_step_m = self._corridor_widen_step_m" in snippet


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------

def test_config_yaml_has_corridor_widen_rate_mps():
    with open(_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["v2x_obstacle_avoidance"]["corridor_widen_rate_mps"] == 1.0
