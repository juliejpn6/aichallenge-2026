"""Unit tests for the OT shadow-QP evaluation (Part B, 2026-08-05, design_docs 283/285節).

背景: OT判断のMPC化検討(依頼書)のPart B。「候補ごとにQPを解いてコストで比較する」
診断機能で、本線制御(_ot_side・コリドー・状態機械)には一切影響しない設計を要求
されている。実装調査(Explore agent、design_docs参照)により、実際の側選択は
コリドー(ub/lb)自体を左右で変えるのではなく、1本の共有コリドー内でlateral_target
(ソフトオフセット目標)を切り替えているだけと判明したため、シャドー版もこの設計を
踏襲する(共有コリドー+lateral_target差し替え版)。

安全上の核心的な要求は「本線が読む状態を一切変更しないこと」——特に
`_stage_data()`はreuse無しで呼ぶたびに`self.model.wp_id`を加算する副作用を持つため
(core/MPC.py、262節guard関連テスト参照)、シャドー評価は`_stage_data()`/`_corridor()`
/`update_path_constraints()`を再呼び出しせず、本線の`get_control()`が既に組んだ
stage dict(`self.last_stage_data`)を再利用する設計とした。P(コスト構造)・
A(力学+レート制約)・l/u(コリドー境界含む)はxr(lateral_target由来)に依存しないため
候補間で共通、xrのみ差し替えてqだけ再計算する。

MPC.pyはrclpy非依存のため直接import。既存の262節ガードテストと同じ「ダックタイプの
fake self」パターン(実際の値は物理的な意味を持たず、次元整合性とAPI契約のみ検証)
に倣う。
"""
import copy
import types

import numpy as np
from scipy import sparse

from multi_purpose_mpc_ros.core.MPC import MPC


def _make_fake_mpc(nx=3, nu=2, N=2, ub0=None, lb0=None):
    """solve_shadow_candidates()の依存を満たす最小限のfake self。
    _assemble_legacy/_vectors/_rate_matrices/_cost_diagは実物のMPCメソッドを
    そのままfakeへ束縛する(数値ロジックはMPC.py本体を検証、fakeは配線のみ)。"""
    fake = types.SimpleNamespace()
    fake.nx = nx
    fake.nu = nu
    fake.Q = sparse.diags([1.0] * nx)
    fake.R = sparse.diags([1.0] * nu)
    fake.QN = sparse.diags([2.0] * nx)
    fake.r_drate = 0.0
    fake.input_constraints = {
        'umin': np.array([0.0, -1.0]), 'umax': np.array([10.0, 1.0])}
    fake._rate_cache = {}

    if ub0 is None:
        ub0 = np.full(N, 3.0)
    if lb0 is None:
        lb0 = np.full(N, -3.0)

    d = {
        'N': N,
        'A_blk': np.stack([0.1 * np.eye(nx) for _ in range(N)]),
        'B_blk': np.stack([np.vstack([np.full(nu, 0.1)] * nx) for _ in range(N)]),
        'uq': np.zeros(N * nx),
        'ur': np.zeros(N * nu),
        'umax_dyn': np.kron(np.ones(N), fake.input_constraints['umax']),
        'rate_hi': np.full(max(N - 1, 0), 1.0),
        'x0': np.zeros(nx),
        'xmin_dyn': np.kron(np.ones(N + 1), np.full(nx, -5.0)),
        'xmax_dyn': np.kron(np.ones(N + 1), np.full(nx, 5.0)),
        'ub0': ub0,
        'lb0': lb0,
        'xr': np.zeros(nx * (N + 1)),
    }
    fake.last_stage_data = d

    # 実物のMPCメソッドをfakeへ束縛(第一引数selfへfakeを渡すだけ、ロジックは共有)
    fake._assemble_legacy = lambda N, d: MPC._assemble_legacy(fake, N, d)
    fake._vectors = lambda N, d: MPC._vectors(fake, N, d)
    fake._rate_matrices = lambda N: MPC._rate_matrices(fake, N)
    fake._cost_diag = lambda N: MPC._cost_diag(fake, N)
    return fake


LEFT_RIGHT = [("left", 1.0, 1.0, 0.0), ("right", -1.0, 1.0, 0.0)]


# ---------------------------------------------------------------------------
# ①非矛盾性: 空/未設定時は安全に空リストを返す(呼び出し元のnullチェック省略を許す契約)
# ---------------------------------------------------------------------------

def test_returns_empty_list_when_no_stage_data():
    fake = _make_fake_mpc()
    fake.last_stage_data = None
    out = MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    assert out == []


def test_returns_empty_list_when_no_candidates():
    fake = _make_fake_mpc()
    out = MPC.solve_shadow_candidates(fake, [])
    assert out == []


# ---------------------------------------------------------------------------
# ②基本機能: 候補ごとに解けてobj_valが返る
# ---------------------------------------------------------------------------

def test_solves_both_candidates_successfully():
    fake = _make_fake_mpc()
    out = MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    assert len(out) == 2
    labels = {r['label'] for r in out}
    assert labels == {'left', 'right'}
    for r in out:
        assert r['solved'] is True
        assert np.isfinite(r['obj_val'])
        assert r['solve_time'] >= 0.0


def test_left_and_right_costs_differ_for_asymmetric_target():
    """左右非対称な目標(lateral_target=+1.0 vs -1.0)なら、対称なコリドー・
    コスト構造の下で目的関数値が(符号はどうあれ)異なることを確認する
    —— 完全に無意味な計算(常に同一値)になっていないことの最低限の保証。"""
    fake = _make_fake_mpc()
    out = MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    by_label = {r['label']: r['obj_val'] for r in out}
    assert by_label['left'] != by_label['right']


def test_lateral_psi_bias_candidate_field_is_applied():
    fake = _make_fake_mpc()
    out = MPC.solve_shadow_candidates(
        fake, [("straight", 0.0, 0.0, 0.0), ("biased_psi", 0.0, 0.0, 0.3)])
    assert len(out) == 2
    assert all(r['solved'] for r in out)


# ---------------------------------------------------------------------------
# ③非侵襲性(最重要): 本線が読む状態を一切変更しない
# ---------------------------------------------------------------------------

def test_does_not_call_stage_data_or_corridor_again():
    """_stage_data()はreuse無しで呼ぶと self.model.wp_id を加算する副作用を持つ
    (core/MPC.py、262節ガードテスト参照)。シャドー評価がこれを再呼び出しすると
    本線のwp追従が二重加算で壊れるため、fakeに_stage_data/_corridorを一切
    定義しない状態で呼び出し、AttributeErrorにならず(=呼ばれていない)正常終了
    することを確認する。"""
    fake = _make_fake_mpc()
    assert not hasattr(fake, '_stage_data')
    assert not hasattr(fake, '_corridor')
    out = MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    assert len(out) == 2  # AttributeErrorなく完走


def test_last_stage_data_dict_is_not_mutated():
    """d_shadow = dict(d)の浅いコピーでxrのみ差し替える設計のため、元のd
    (self.last_stage_data、本線が次周期以降参照する可能性のある値)自体は
    書き換わらないことを確認する。"""
    fake = _make_fake_mpc()
    d_before = copy.deepcopy(fake.last_stage_data)
    MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    d_after = fake.last_stage_data
    assert set(d_before.keys()) == set(d_after.keys())
    for k in d_before:
        np.testing.assert_array_equal(np.asarray(d_before[k]), np.asarray(d_after[k]))


def test_does_not_introduce_new_attributes_on_self_other_than_declared():
    """solve_shadow_candidates()がself(fake)へ新規属性を書き込まないことを確認する。
    (last_shadow_solve_timeはMPC.__init__側で事前宣言される想定のため、ここでは
    そのフィールドを含め「呼び出し前後で属性集合が不変」であることまでは要求せず、
    既知の非制御系フィールド以外が増えないことのみ確認する。)"""
    fake = _make_fake_mpc()
    attrs_before = set(vars(fake).keys())
    MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    attrs_after = set(vars(fake).keys())
    # last_shadow_solve_timeのみ新規追加を許容(実装がself.last_shadow_solve_timeを
    # 更新する設計のため)。それ以外の新規属性が増えていないことを確認する。
    new_attrs = attrs_after - attrs_before
    assert new_attrs <= {'last_shadow_solve_time'}


# ---------------------------------------------------------------------------
# ④例外安全性: 解けない/壊れた候補があっても例外を外へ漏らさない
# ---------------------------------------------------------------------------

def test_survives_assemble_legacy_exception():
    fake = _make_fake_mpc()

    def _boom(N, d):
        raise RuntimeError("boom")
    fake._assemble_legacy = _boom
    out = MPC.solve_shadow_candidates(fake, LEFT_RIGHT)
    assert len(out) == 1
    assert out[0]['solved'] is False
    assert 'error' in out[0]


# ---------------------------------------------------------------------------
# 遡及効果: ソーステキスト構造検証(呼び出し元・configゲート・非侵襲性の配線確認)
# ---------------------------------------------------------------------------

def _read(path_parts):
    import os
    path = os.path.join(os.path.dirname(__file__), "..", *path_parts)
    with open(path) as f:
        return f.read()


_MPC_SRC = _read(["multi_purpose_mpc_ros", "core", "MPC.py"])
_CTRL_SRC = _read(["multi_purpose_mpc_ros", "mpc_controller.py"])


def test_get_control_caches_stage_data_after_init_problem():
    idx = _MPC_SRC.index("    def get_control(self)")
    idx_solve = _MPC_SRC.index("dec = self._active.solve()", idx)
    snippet = _MPC_SRC[idx:idx_solve]
    assert "d = self._init_problem(N, sm)" in snippet
    assert "self.last_stage_data = d" in snippet
    # last_stage_data代入がd確定「後」であること(未確定dを保持しないため)
    assert snippet.index("d = self._init_problem(N, sm)") < snippet.index(
        "self.last_stage_data = d")


def test_shadow_qp_config_declares_valid_bool_enable():
    """287節: dev3実走行検証のためON/OFFを行き来する運用対象(CLAUDE.md §1.1同様の
    race value)であり、どちらの値でも正当なため固定値ではなくtrue/falseいずれかで
    あることのみを検査する(test_enable_diag_log_bypass_214.pyと同じ方針)。"""
    import os
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_src = f.read()
    idx = cfg_src.index("ot_shadow_qp:")
    snippet = cfg_src[idx:idx + 300]
    assert "enable: true" in snippet or "enable: false" in snippet


def test_controller_gates_shadow_call_behind_enable_flag():
    idx = _CTRL_SRC.index("self._run_ot_shadow_qp()")
    before = _CTRL_SRC[max(0, idx - 300):idx]
    after = _CTRL_SRC[idx:idx + 200]
    assert "if self._ot_shadow_qp_enable:" in before
    assert "try:" in before
    assert "except Exception" in after


def test_shadow_call_happens_after_get_control_not_before():
    idx_get_control = _CTRL_SRC.index("u, max_delta = self._mpc.get_control()")
    idx_shadow_call = _CTRL_SRC.index("self._run_ot_shadow_qp()")
    assert idx_get_control < idx_shadow_call


def test_run_ot_shadow_qp_never_writes_ot_side_or_lateral_target():
    """_run_ot_shadow_qp()の本体が、本線の側選択(_ot_side)やMPCのlateral_target/
    lateral_blendへ一切代入していないことをソーステキストで確認する
    (依頼書の「本線には一切書き戻さない」という設計要求の直接検証)。"""
    idx = _CTRL_SRC.index("def _run_ot_shadow_qp(self):")
    idx_end = _CTRL_SRC.index("\n    def _control(self):", idx)
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._ot_side =" not in snippet
    assert "self._mpc.lateral_target =" not in snippet
    assert "self._mpc.lateral_blend =" not in snippet


def test_shadow_qp_thinning_counter_and_period_declared():
    idx = _CTRL_SRC.index("self._ot_shadow_qp_enable = bool(_shqget(")
    idx_end = idx + 400
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._ot_shadow_qp_period" in snippet


# ---------------------------------------------------------------------------
# 288節(2026-08-05): 追従(follow)候補+OT結果の成功/失敗記録
# ---------------------------------------------------------------------------

def test_run_ot_shadow_qp_includes_follow_candidate():
    """ユーザー指摘: left/rightのみでは「オーバーテイクすべきか追従すべきか」に
    答えられない。3択目のfollow候補(lateral_blend=0.0)が実際に送られていることを
    ソーステキストで確認する。"""
    idx = _CTRL_SRC.index("def _run_ot_shadow_qp(self):")
    idx_end = _CTRL_SRC.index("def _log_ot_outcome(self", idx)
    snippet = _CTRL_SRC[idx:idx_end]
    assert '("follow", 0.0, 0.0, 0.0)' in snippet
    assert "obj_follow" in snippet
    assert "shadow_choice" in snippet


def test_log_ot_outcome_called_at_all_three_exit_points():
    """OT離脱の3経路(giveup・exit_clear・infeasibility強制)すべてで
    _log_ot_outcome()が呼ばれ、成功/失敗が記録されることを確認する。"""
    assert _CTRL_SRC.count("self._log_ot_outcome(") == 3
    assert 'self._log_ot_outcome(\n                            "giveup"' in _CTRL_SRC
    assert 'self._log_ot_outcome("success", self._ot_side, reason="exit_clear")' in _CTRL_SRC
    assert 'self._log_ot_outcome("failure", self._ot_side, reason="infeasible")' in _CTRL_SRC


def test_log_ot_outcome_called_before_side_reset_at_each_site():
    """side_usedとして渡すself._ot_sideが、直後の`self._ot_side = 0`より前に
    評価されること(呼び出し元でresetする前に読む設計)をソーステキストで確認する。"""
    for marker in ['"giveup"', '"success"', '"failure"']:
        idx = _CTRL_SRC.index(f"self._log_ot_outcome(\n" if marker == '"giveup"'
                               else f"self._log_ot_outcome({marker}")
        idx_reset = _CTRL_SRC.index("self._ot_side = 0", idx)
        assert idx < idx_reset


def test_log_ot_outcome_method_reads_but_does_not_write_ot_side():
    idx = _CTRL_SRC.index("def _log_ot_outcome(self")
    idx_end = _CTRL_SRC.index("\n    def _control(self):", idx)
    snippet = _CTRL_SRC[idx:idx_end]
    assert "self._ot_side =" not in snippet
    assert "self._ot_shadow_last" in snippet
