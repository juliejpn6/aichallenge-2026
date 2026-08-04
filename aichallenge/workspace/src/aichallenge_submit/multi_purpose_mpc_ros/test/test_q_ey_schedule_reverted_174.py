"""Unit tests confirming the Q[e_y] curvature schedule (v1-v5+quantization gate) is
disabled again (174節, 2026-07-24) as an isolated A/B test.

背景: 171節でQ[e_y]ベース値(3M)確定の上、スケジュール自体の効果を再検証するため
再導入していたが、予選環境で観測される「全体的な蛇行」の原因切り分けのため、
Q[e_y]ベース値(3M、不変)は維持したままスケジュールの有無だけを単独変数として
一時的にOFFにする。170節の撤去手順・検証項目と同一(mpc_controller.py 3箇所+
config.yaml)。ベース値自体は一切変更しない。

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
# ①非矛盾性: スケジュール由来の状態変数・計算・設定値がソースに存在しないこと
# ---------------------------------------------------------------------------

def test_schedule_state_vars_not_declared():
    for tok in ["_q_ey_corner_boost", "_q_ey_kappa_lo", "_q_ey_kappa_hi",
                "_q_ey_lookahead_wp", "_q_ey_ema_beta", "_q_ey_kappa_ema",
                "_q_ey_applied_value"]:
        assert f"self.{tok}" not in _SRC, f"{tok} が撤去されず残っている"


def test_schedule_params_not_in_yaml():
    for tok in ["q_ey_corner_boost", "q_ey_kappa_lo", "q_ey_kappa_hi",
                "q_ey_lookahead_wp", "q_ey_ema_beta"]:
        assert tok not in _YAML_SRC, f"{tok} が撤去されず残っている"


def test_kappa_ahead_computation_removed():
    for tok in ["_kappa_ahead_raw", "_kappa_ahead", "kappa_ahead"]:
        assert tok not in _SRC, f"{tok} が撤去されず残っている"


# ---------------------------------------------------------------------------
# ②非冗長性: NORMAL分岐はOVERTAKING/pitと同型の「変化した時だけupdate_Q」に統一
# ---------------------------------------------------------------------------

def test_normal_branch_uses_static_config_q_quantized_on_state_change():
    idx = _SRC.index('elif self._ot_state == "OVERTAKING":')
    idx_else = _SRC.index("\n            else:\n", idx)
    idx_end = _SRC.index("self._pf_mark('state_v_safe')", idx_else)
    snippet = _SRC[idx_else:idx_end]
    assert 'if self._ot_q_applied != "normal":' in snippet
    assert "self._mpc.update_Q(sparse.diags(list(self._cfg.mpc.Q)))" in snippet
    assert 'self._ot_q_applied = "normal"' in snippet


def test_overtake_branch_no_longer_resyncs_removed_state():
    idx = _SRC.index('elif self._ot_state == "OVERTAKING":')
    idx_end = _SRC.index("\n            else:\n", idx)
    snippet = _SRC[idx:idx_end]
    assert "_q_ey_applied_value" not in snippet


def test_pit_branch_no_longer_resyncs_removed_state():
    idx = _SRC.index('if getattr(self, "_ot_q_applied", None) != "pit":')
    snippet = _SRC[idx:idx + 500]
    assert "_q_ey_applied_value" not in snippet


# ---------------------------------------------------------------------------
# ③検証ロギング: [OT]ログからq_ey/kappa_ahead欄が除去されていること(存在しない値を
#   出力し続ける方が誤解を招くため)
# ---------------------------------------------------------------------------

def test_ot_log_line_no_longer_prints_removed_fields():
    idx = _SRC.index('f"[OT] state=')
    idx_end = idx + 2600
    snippet = _SRC[idx:idx_end]
    assert "q_ey=" not in snippet
    assert "kappa_ahead=" not in snippet
    assert 'f"gate={_fwd_dbg.get(\'gate\')}")' in snippet


# ---------------------------------------------------------------------------
# ④遡及効果: Q[e_y]ベース値(3M)はこのA/Bテストで変更していないことを確認する
#   (170→171節の教訓=ベース値変更とスケジュール有無を混同しない)
#   Q[e_psi](第2要素)は別件(201節、AXIS06のQ/R再チューニング実験)で1e8→7e7へ一時変更
#   したが、実測で改善が見られず1e8へ復元した(201節続報)。第4要素は201節続報の
#   delta_actual状態拡張(nx=3→4)再実装に伴う追加(コストなし、0.0)。
# ---------------------------------------------------------------------------

def test_q_ey_base_value_untouched_at_3m():
    """225-226節(2026-07-28): 一定曲率区間で指令舵角が理論要求値の最大2.3倍まで
    振動するリミットサイクルを発見し、Q[e_y](R[delta]の約6000倍という強い追従重み)
    が過制御の一因という仮説のもと、Q[e_y]自体を意図的に能動的スイープ中(ローカル
    3.0M→1.2M→0.8M→1.0M、予選1.2M等、debug_extra_actuator_delay_sと同じ「頻繁に
    切り替えるライブ実験値」の扱い)。本テストが元々ガードしていた「170→171節の
    Q[e_y]スケジュールA/Bで意図せず変更されていないこと」という趣旨を保ちつつ、
    Q[e_y]自体の値ではなくQ[t]/Q[delta_actual](第3〜4要素、このスイープの
    対象外)が不変であることを確認する形へ更新した(弱体化ではない)。
    2026-08-04: Q[e_psi](第2要素)はstage15 273-275節のハード制約先行判定により
    100000000.0→1000000.0へ確定変更されたため、本テストの対象から外した(CLAUDE.md
    §3参照、確定結論の変更であり退行ではない)。"""
    import re
    m = re.search(r"Q:\s*\[[\d.]+,\s*[\d.]+,\s*200000\.0,\s*0\.0\]", _YAML_SRC)
    assert m is not None, "Q[t]/Q[delta_actual]が想定値から変化している"


def test_overtake_and_pit_q_ey_overrides_still_intact():
    """OVERTAKING/pit専用のQ[e_y]上書き(q_ey_overtake/q_ey_pit)はスケジュールとは
    無関係の既存機能であり、このA/Bテストで意図せず巻き込まれていないことを確認する。"""
    assert "q_ey_overtake: 5000000.0" in _YAML_SRC
    assert "q_ey_pit: 50000000.0" in _YAML_SRC
    assert "self._ot_q_ey = float(_otget(\"q_ey_overtake\"" in _SRC
    assert "self._pit_q_ey = float(getattr(_pit, \"q_ey_pit\"" in _SRC
