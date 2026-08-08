"""Unit tests for the R[delta] curvature-swing schedule (176節続報, 2026-07-24).

背景: 176節の実測分析で、予選環境の「全体的な蛇行」が曲率の絶対値(r=0.346)ではなく
前後17m窓内での曲率の変動幅=swing(r=0.807)と強く相関すること、局所化層(EKF/GNSS)の
ノイズは全区間で一定(=無罪)であることが判明した。これを受け、シケイン(短距離での
曲率符号反転)を検知した時だけR[delta](舵角変化ペナルティ)を引き上げる機構を実装した。
Q[e_y]ではなくR[delta]を触るのは、v1-v5のQ[e_y]曲率スケジュールと同じ「目標を動かす」
失敗パターンを避けるため。R[delta]の常時・一律な引き上げ(500→1000)は既に撤回済みだが、
今回はシケイン区間のみへの限定的な引き上げである点が異なる。

167節で「毎周期無条件update_Qが予選環境の処理落ちを悪化させた」教訓を踏まえ、量子化
ゲート(目標値がboostの1%以上動いた時のみ実際にupdate_Rを呼ぶ)を最初から組み込んでいる。

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
# 状態変数の宣言・config既定値
# ---------------------------------------------------------------------------

def test_state_vars_declared():
    idx = _SRC.index("    def _initialize(self) -> None:")
    idx_end = _SRC.index("\n    def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    for tok in ["_r_delta_swing_boost", "_r_delta_swing_kappa_lo", "_r_delta_swing_kappa_hi",
                "_r_delta_swing_lookahead_wp", "_r_delta_swing_ema_beta", "_r_delta_swing_ema",
                "_r_delta_applied_value", "_r_delta_swing_update_count", "_r_delta_swing_dbg_loop"]:
        assert f"self.{tok} = " in snippet, f"{tok} が宣言されていない"


def test_ema_and_applied_value_initialized_to_none():
    assert "self._r_delta_swing_ema = None" in _SRC
    assert "self._r_delta_applied_value = None" in _SRC


def test_counters_initialized_to_zero():
    assert "self._r_delta_swing_update_count = 0" in _SRC
    assert "self._r_delta_swing_dbg_loop = 0" in _SRC


def test_params_declared_in_yaml_with_correct_defaults():
    """2026-08-05(task#295、306節続報3)にboost値400.0→800.0へ改定。dev3 A/Bで
    wp340-40帯の周回間std改善(1周目暖機を除きほぼ0.00-0.03m、旧400.0時は
    0.19-0.85m)を確認し、他ホットスポット・対照区への悪影響なしを確認した上で
    採用した(config.yamlのr_delta_swing_boost行コメント参照)。
    2026-08-08(タスク#308、design_docs §38.7): gain=0.6/tau=0.05モデル向けに
    0/400/800(旧)/1200/1600/2000を再スイープ、1600へ更新。"""
    assert "r_delta_swing_boost: 1600.0" in _YAML_SRC
    assert "r_delta_swing_kappa_lo: 0.12" in _YAML_SRC
    assert "r_delta_swing_kappa_hi: 0.30" in _YAML_SRC
    assert "r_delta_swing_lookahead_wp: 16" in _YAML_SRC
    assert "r_delta_swing_ema_beta: 0.15" in _YAML_SRC


def test_boost_restored_after_ab_test_exonerated_it_178():
    """178節: dev3実走行で超過率29.5-52.75%を観測しboost=0.0のA/Bテストを行ったが、
    boost=0.0でも超過率31.75-45%とほぼ同水準(r_delta_swing_total avg=0.1ms、updates=1で
    固定)であり、本機構は処理落ちの原因ではないと判定された。400.0へ復帰した。
    2026-08-05(306節続報3): 別の実験(wp340-40帯対策)により400.0→800.0へ再改定。
    2026-08-08(タスク#308): gain=0.6/tau=0.05モデル向け再スイープにより800.0→1600.0へ
    改定。テスト名は歴史的経緯を示す旧名のまま維持し、値のみ現行に追従する。"""
    assert "r_delta_swing_boost: 1600.0" in _YAML_SRC


def test_base_r_array_unchanged_500():
    """④遡及効果: R[delta]の常時一律な引き上げ(500→1000)は既に撤回済みという過去の
    教訓を踏まえ、base値そのものは触っていないことを確認する。206節(2026-07-27、
    AXIS06delta_actual導入後)でr_drate=3e6を堅持したまま500→800への小幅な引き上げを
    再検証したが、直線std=3.86→4.29°へ悪化し棄却、500へ復元済み。"""
    assert "R: [100000.0, 500.0]" in _YAML_SRC


# ---------------------------------------------------------------------------
# ①非矛盾性: OT/pit状態と無関係に毎周期無条件で計算されること
# ---------------------------------------------------------------------------

def test_swing_computation_is_unconditional_not_nested_in_ot_or_pit_branch():
    idx_active = _SRC.index('_ot_active = bool(self.USE_OBSTACLE_AVOIDANCE)')
    idx_kappas = _SRC.index("_kappas_fwd = [")
    idx_get_control = _SRC.index("u, max_delta = self._mpc.get_control()")
    assert idx_active < idx_kappas < idx_get_control
    # インデント幅が_ot_activeと同一(状態分岐の外側)であることを確認
    line_active = _SRC[_SRC.rfind("\n", 0, idx_active) + 1:idx_active]
    line_kappas = _SRC[_SRC.rfind("\n", 0, idx_kappas) + 1:idx_kappas]
    assert line_active == line_kappas


def test_swing_uses_forward_only_lookahead_not_centered_window():
    idx = _SRC.index("_kappas_fwd = [")
    idx_end = _SRC.index("_swing_raw = max", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._mpc.model.wp_id + _i" in snippet
    assert "range(self._r_delta_swing_lookahead_wp + 1)" in snippet


def test_swing_is_max_minus_min_not_magnitude():
    """176節の発見(|kappa|よりswing=max-minの方が強く相関)をそのまま反映しているか。"""
    assert "_swing_raw = max(_kappas_fwd) - min(_kappas_fwd)" in _SRC


def test_ema_smoothing_applied_before_smoothstep_in_source():
    idx_ema = _SRC.index("self._r_delta_swing_ema_beta * _swing_raw")
    idx_smoothstep = _SRC.index("_smooth_sw = _t_sw * _t_sw * (3 - 2 * _t_sw)")
    assert idx_ema < idx_smoothstep


def test_smoothstep_formula_matches_cubic_hermite_in_source():
    assert "_smooth_sw = _t_sw * _t_sw * (3 - 2 * _t_sw)" in _SRC


def test_only_r_delta_index_modified_not_r_v():
    idx = _SRC.index("_r = list(self._cfg.mpc.R)")
    idx_end = _SRC.index("if (self._r_delta_applied_value is None", idx)
    snippet = _SRC[idx:idx_end]
    assert "_r[1] = _r[1] + self._r_delta_swing_boost * _smooth_sw" in snippet
    assert "_r[0]" not in snippet


# ---------------------------------------------------------------------------
# ②非冗長性・②処理落ち防止: 量子化ゲートが167節と同じ形で存在すること
# ---------------------------------------------------------------------------

def test_update_r_is_quantization_gated_not_unconditional():
    idx = _SRC.index("if (self._r_delta_applied_value is None")
    idx_end = _SRC.index("self._r_delta_swing_dbg_loop += 1", idx)
    snippet = _SRC[idx:idx_end]
    assert "abs(_r[1] - self._r_delta_applied_value) > self._r_delta_swing_boost * 0.01" in snippet
    assert "self._mpc.update_R(sparse.diags(_r))" in snippet
    assert "self._r_delta_applied_value = _r[1]" in snippet
    assert "self._r_delta_swing_update_count += 1" in snippet


def test_perf_bucket_measures_update_r_cost_exactly_once():
    assert _SRC.count("self._pf_add('r_delta_swing_update'") == 1


def test_perf_bucket_measures_full_unconditional_block_cost_exactly_once():
    """2026-07-25追加(177節続報): r_delta_swing_updateはupdate_R呼び出し自体しか計測して
    いなかった(量子化ゲートで弾かれた周期のkappa取得ループ+EMA+smoothstepは未計測)。
    ユーザー指摘を受け、ブロック全体を1本のタイマーで漏れなく計測できているか確認する。"""
    assert _SRC.count("self._pf_add('r_delta_swing_total'") == 1
    idx_t0 = _SRC.index("_t0_total = _time.perf_counter()")
    idx_kappas = _SRC.index("_kappas_fwd = [")
    idx_total_add = _SRC.index("self._pf_add('r_delta_swing_total'")
    idx_update_gate = _SRC.index("if (self._r_delta_applied_value is None")
    # タイマー開始がkappa取得ループより前、計測終了(pf_add)が量子化ゲート判定より後にあり、
    # ゲートで弾かれる(update_Rを呼ばない)周期も含めて計測対象になっていることを構造的に確認
    assert idx_t0 < idx_kappas < idx_update_gate < idx_total_add


# ---------------------------------------------------------------------------
# ③検証ロギング: [R-DELTA-SWING]ログが実際に出力され、量子化ゲートの発火回数を
#   直接確認できること(167節は事後にしか気付けなかった反省への対処)
# ---------------------------------------------------------------------------

def test_verification_log_present_with_swing_and_update_count():
    idx = _SRC.index('f"[R-DELTA-SWING]')
    idx_end = idx + 500
    snippet = _SRC[idx:idx_end]
    assert "swing={_swing:.3f}" in snippet
    assert "r_delta_target={_r[1]:.1f}" in snippet
    assert "r_delta_applied={self._r_delta_applied_value:.1f}" in snippet
    assert "updates={self._r_delta_swing_update_count}" in snippet


def test_verification_log_throttled_to_1hz_like_existing_ot_log():
    """214節でログ出力のみをenable_diag_logでガードする説明コメントが追加され
    間隔が広がったため、窓を300→600へ拡大した(検証内容自体は変更なし)。"""
    idx = _SRC.index("self._r_delta_swing_dbg_loop += 1")
    idx_end = idx + 600
    snippet = _SRC[idx:idx_end]
    assert "self._r_delta_swing_dbg_loop % int(max(1, self._mpc_cfg.control_rate)) == 0" in snippet
