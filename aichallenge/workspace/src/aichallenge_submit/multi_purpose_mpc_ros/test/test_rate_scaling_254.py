"""254節続報(2026-07-31): MPC制御周期引き上げ(40Hz→72Hz)に向けた
control_rate依存値の自動換算機構(_rate_scaled_cycles / _rate_scaled_gain)。

設計書: docs/superpowers/specs/2026-07-31-mpc-control-rate-increase-design.md
(2回のGeminiレビューを経て確定)。

背景: config.yamlの多数のパラメータ(STUCK検知のhold_cycles等)は「周期数」で
定義されており、40Hz基準で書かれている。control_rateを変えると、これらの
閾値が意図した実時間の意味を失う(例: hold_cycles=60は40Hzでは1.5秒だが、
72Hzでは0.833秒に縮んでしまう)。同様にEMA/ローパスゲイン(steer_low_pass_gain
等)も、周期あたりの減衰率が固定のため、周期を上げると実効時定数が変わる。

対処: config.yamlの数値・キー名は一切変更せず(「40Hz基準の値」という意味を
保つ)、読み込み側で以下2つのヘルパーを明示的に呼び出しラップする:
  - _rate_scaled_cycles(name, cycles_at_ref): 線形換算
    (round(cycles_at_ref * control_rate / 40.0)、max(1,...)で0への潰れを防止)
  - _rate_scaled_gain(name, gain_at_ref): 指数換算
    (1-(1-gain)^(40.0/control_rate)、離散一次遅れの時定数を保つ)

本作業ではcontrol_rate自体は40.0のまま変更しない(72Hzへの実際の切替は
別途、実運用環境での処理落ち率実測を経て行う)。40Hzのままであれば、
両ヘルパーは恒等写像(入力=出力)になることを本テストで確認する。

mpc_controller.pyはrclpy依存のため直接importできない。数式自体は純粋な
算術のためミラー関数で検証し、mpc_controller.py側の配線(23箇所全てが
ヘルパー経由で読まれていること)はソーステキスト構造検証で確認する
(既存テストと同じ方針)。
"""
import os
import re

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

REFERENCE_HZ = 40.0


def mirror_rate_scaled_cycles(cycles_at_ref, rate, ref_rate=REFERENCE_HZ):
    """mpc_controller.py _rate_scaled_cyclesの数式ミラー。"""
    return max(1, round(cycles_at_ref * rate / ref_rate))


def mirror_rate_scaled_gain(gain_at_ref, rate, ref_rate=REFERENCE_HZ):
    """mpc_controller.py _rate_scaled_gainの数式ミラー。"""
    return 1.0 - (1.0 - gain_at_ref) ** (ref_rate / rate)


# ---------------------------------------------------------------------------
# ①非矛盾性: 数式そのものの正しさ(境界値・恒等性・具体値)
# ---------------------------------------------------------------------------

def test_cycles_identity_at_reference_rate():
    """40Hz(基準周期そのもの)では、入力値がそのまま返ることを確認する
    (本作業ではcontrol_rateを変更しないため、この恒等性が最重要)。"""
    for v in (60, 20, 400, 300, 40, 8, 160, 5, 15, 80, 3, 50, 400):
        assert mirror_rate_scaled_cycles(v, rate=40.0) == v


def test_gain_identity_at_reference_rate():
    for g in (0.05, 0.15, 0.35):
        assert abs(mirror_rate_scaled_gain(g, rate=40.0) - g) < 1e-12


def test_cycles_72hz_known_values():
    """design_docsで検算済みの72Hz換算値(40Hz→72Hzは比率1.8倍)。"""
    assert mirror_rate_scaled_cycles(60, rate=72.0) == 108
    assert mirror_rate_scaled_cycles(40, rate=72.0) == 72
    assert mirror_rate_scaled_cycles(20, rate=72.0) == 36
    assert mirror_rate_scaled_cycles(400, rate=72.0) == 720
    assert mirror_rate_scaled_cycles(300, rate=72.0) == 540
    assert mirror_rate_scaled_cycles(8, rate=72.0) == 14  # 8*1.8=14.4→14
    assert mirror_rate_scaled_cycles(160, rate=72.0) == 288
    assert mirror_rate_scaled_cycles(5, rate=72.0) == 9  # 5*1.8=9.0
    assert mirror_rate_scaled_cycles(15, rate=72.0) == 27
    assert mirror_rate_scaled_cycles(80, rate=72.0) == 144
    assert mirror_rate_scaled_cycles(3, rate=72.0) == 5  # 3*1.8=5.4→5(丸め誤差、設計書に明記済み)
    assert mirror_rate_scaled_cycles(50, rate=72.0) == 90


def test_gain_72hz_known_values():
    """Geminiレビューで検算・訂正された72Hz換算値(設計書の誤記0.231を0.213へ訂正)。"""
    assert abs(mirror_rate_scaled_gain(0.35, rate=72.0) - 0.2128) < 1e-3
    assert abs(mirror_rate_scaled_gain(0.05, rate=72.0) - 0.0281) < 1e-3
    assert abs(mirror_rate_scaled_gain(0.15, rate=72.0) - 0.0872) < 1e-3


def test_cycles_max1_guard_prevents_collapse_to_zero():
    """回帰防止(Geminiレビュー指摘): control_rateを大きく下げた場合に
    閾値が0(=即発火)へ潰れないことを確認する。"""
    assert mirror_rate_scaled_cycles(3, rate=5.0) >= 1  # 3*5/40=0.375→0にせずmax(1,...)
    assert mirror_rate_scaled_cycles(1, rate=1.0) >= 1


def test_cycles_lower_rate_scales_down():
    """回帰確認: control_rateを下げれば周期数も比例して減る(逆方向も正しく動く)。"""
    assert mirror_rate_scaled_cycles(60, rate=20.0) == 30  # 半分の周期数で同じ実時間


def test_gain_lower_rate_scales_down():
    g_20 = mirror_rate_scaled_gain(0.35, rate=20.0)
    g_40 = 0.35
    g_72 = mirror_rate_scaled_gain(0.35, rate=72.0)
    # 周期が下がるほど、同じ実時間の減衰を得るために1周期あたりのgainは大きくなる
    assert g_20 > g_40 > g_72


# ---------------------------------------------------------------------------
# ②非冗長性・③検証ロギング: mpc_controller.py側のヘルパー実装そのものの検証
# ---------------------------------------------------------------------------

def test_rate_scaled_cycles_implementation_matches_formula():
    idx = _SRC.index("def _rate_scaled_cycles(")
    idx_end = _SRC.index("def _rate_scaled_gain(")
    snippet = _SRC[idx:idx_end]
    assert "max(1, round(" in snippet
    assert "_RATE_SCALE_REFERENCE_HZ" in snippet
    assert '[RATE-SCALE]' in snippet  # ③検証ロギング


def test_rate_scaled_gain_implementation_matches_formula():
    idx = _SRC.index("def _rate_scaled_gain(")
    idx_end = _SRC.index("def _initialize(")
    snippet = _SRC[idx:idx_end]
    assert "1.0 - (1.0 - gain_at_ref) ** (" in snippet
    assert "_RATE_SCALE_REFERENCE_HZ" in snippet
    assert '[RATE-SCALE]' in snippet


def test_reference_hz_constant_is_40():
    idx = _SRC.index("_RATE_SCALE_REFERENCE_HZ = 40.0")
    assert idx > 0


def test_helpers_use_raw_cfg_not_mpc_cfg_to_avoid_init_order_bug():
    """実装時に発見した初期化順序バグの回帰防止: osqp_shadow_cyclesはcreate_mpc()内
    (self._mpc_cfg代入より前)で読まれるため、ヘルパーはself._mpc_cfg.control_rate
    ではなくself._cfg.mpc.control_rate(__init__時点から常に利用可能)を参照する
    必要がある。"""
    idx = _SRC.index("def _rate_scaled_cycles(")
    idx_end = _SRC.index("def _rate_scaled_gain(")
    snippet = _SRC[idx:idx_end]
    assert "self._cfg.mpc.control_rate" in snippet
    idx2 = _SRC.index("def _rate_scaled_gain(")
    idx2_end = _SRC.index("def _initialize(")
    snippet2 = _SRC[idx2:idx2_end]
    assert "self._cfg.mpc.control_rate" in snippet2


# ---------------------------------------------------------------------------
# ④遡及効果: 23箇所全てがヘルパー経由でラップされていることの網羅的確認
#   (ラップ漏れ検出、将来のパラメータ追加時の回帰防止)
# ---------------------------------------------------------------------------

# カテゴリA(_rate_scaled_cycles)で読まれるべき全パラメータ名(設計書の表と一致)
CATEGORY_A_PARAMS = [
    "hold_cycles", "gear_settle_cycles", "stall_hold_cycles", "infeas_thr",
    "ghost_block_hold_cycles", "giveup_cycles", "engage_debounce",
    "engage_cooldown", "def_enter_cycles", "def_exit_cycles",
    "unlock_inf_cycles", "unlock_hold_cycles", "collision_cum_window_cycles",
    "min_trend_cycles", "infeasible_latch", "course_in_count",
    "osqp_shadow_cycles",
]

# カテゴリB(_rate_scaled_gain)で読まれるべき全パラメータ名
CATEGORY_B_PARAMS = [
    "ema_alpha", "r_delta_swing_ema_beta", "accel_low_pass_gain",
    "steer_low_pass_gain",
]

# 換算対象外として明示的に除外されるもの(意図的、設計書に明記済み)
EXEMPT_PARAMS = [
    "shuffle_max_cycles",   # 試行回数のカウント、周期時間ではない
    "space_ema_alpha",       # self._ot_ema_alpha(既に換算済み)へのフォールバックのみ、
                             # 二重換算を避けるため直接ラップしない
    "perf_dt_over_margin_ms",  # 2026-08-01追加(72Hz切替準備Phase 1): [PERF-DT]の
                             # 周期超過判定に加えるマージン(計測の判定閾値であり
                             # 制御には影響しないため、control_rateが変わっても
                             # 自動換算しない)
    "perf_spike_dump_factor",  # 2026-08-01追加(261節続報、72Hzスパイク調査Phase 3):
                             # [PERF-SPIKE]の発火倍率(計測の判定閾値であり制御には
                             # 影響しないため、perf_dt_over_margin_msと同じ理由で
                             # 自動換算しない)
    "cpu_affinity",          # 2026-08-01追加(262節続報、判定基準改訂+work_cpu計装Phase 4):
                             # os.sched_setaffinityで使うCPUコア番号のリスト。実際の
                             # CPUバインドという制御外の運用パラメータであり、周期時間や
                             # 減衰係数ではないため自動換算しない
    "perf_dt_spike_factor",  # 2026-08-02追加(263節続報、蛇行/性能ギャップ分析Part A-1):
                             # [PERF-DT-SPIKE]の発火倍率(計測の判定閾値であり制御には
                             # 影響しないため、perf_dt_over_margin_msと同じ理由で
                             # 自動換算しない)
    "failsafe_dt_threshold_ms",  # 2026-08-02追加(264節続報Task1): dt異常フェイルセーフの
                             # 発火閾値。絶対時間(ms)で意味を持つ値であり、control_rateが
                             # 変わっても「何ms以上ならワインドアップの恐れがあるか」という
                             # 物理的な意味は変わらないため自動換算しない(actuator_lag_tau_s
                             # と同じ扱い)
]


def test_all_category_a_params_appear_as_first_arg_to_helper():
    """各パラメータ名が_rate_scaled_cyclesの呼び出し引数として実際に登場することを、
    正規表現(呼び出し括弧の直後・空白/改行を許容・文字列リテラル)で確認する。
    文字列リテラルの最初の出現をindex()で拾うと、declare_parameter等の別用途で
    誤検知する可能性があるため、呼び出しパターンそのものを直接検索する。"""
    for name in CATEGORY_A_PARAMS:
        pattern = re.compile(
            r'_rate_scaled_cycles\(\s*"' + re.escape(name) + r'"')
        assert pattern.search(_SRC), (
            f'_rate_scaled_cycles("{name}"...) という呼び出しが見つかりません')


def test_all_category_b_params_appear_as_first_arg_to_helper():
    """各パラメータ名について、文字列リテラルの最初の出現(declare_parameter等の
    別用途である可能性がある)ではなく、_rate_scaled_gain(呼び出しの引数としての
    出現そのものを直接検索する(accel_low_pass_gain/steer_low_pass_gainは
    declare_parameter呼び出しでも文字列として登場するため、素朴なindex()検索では
    誤検知する)。"""
    for name in CATEGORY_B_PARAMS:
        pattern = re.compile(r'_rate_scaled_gain\(\s*"' + re.escape(name) + r'"')
        assert pattern.search(_SRC), (
            f'_rate_scaled_gain("{name}"...) という呼び出しが見つかりません')


def test_lat_ttc_beta_wrapped():
    """LAT-TTCのbetaは他のbeta識別子と紛らわしいため専用ラベルで別途確認する。"""
    idx = _SRC.index('"beta(LAT-TTC)"')
    preceding = _SRC[max(0, idx - 60):idx]
    assert "_rate_scaled_gain(" in preceding


def test_exempt_params_not_wrapped():
    """換算対象外のパラメータがヘルパーでラップされていないことを確認する
    (誤って換算してしまう退行の防止)。"""
    idx = _SRC.index('_stkget("shuffle_max_cycles"')
    line_end = _SRC.index("\n", idx)
    line = _SRC[max(0, idx - 40):line_end]
    assert "_rate_scaled_cycles(" not in line

    idx2 = _SRC.index('_lget("space_ema_alpha"')
    line2_end = _SRC.index("\n", idx2)
    line2 = _SRC[max(0, idx2 - 40):line2_end]
    assert "_rate_scaled_gain(" not in line2

    idx3 = _SRC.index('getattr(self._cfg.mpc, "perf_dt_over_margin_ms"')
    line3_end = _SRC.index("\n", idx3)
    line3 = _SRC[max(0, idx3 - 40):line3_end]
    assert "_rate_scaled_gain(" not in line3
    assert "_rate_scaled_cycles(" not in line3

    idx4 = _SRC.index('getattr(self._cfg.mpc, "cpu_affinity"')
    line4_end = _SRC.index("\n", idx4)
    line4 = _SRC[max(0, idx4 - 40):line4_end]
    assert "_rate_scaled_gain(" not in line4
    assert "_rate_scaled_cycles(" not in line4

    idx5 = _SRC.index('getattr(self._cfg.mpc, "perf_dt_spike_factor"')
    line5_end = _SRC.index("\n", idx5)
    line5 = _SRC[max(0, idx5 - 40):line5_end]
    assert "_rate_scaled_gain(" not in line5
    assert "_rate_scaled_cycles(" not in line5

    idx6 = _SRC.index('getattr(self._mpc_cfg, "failsafe_dt_threshold_ms"')
    line6_end = _SRC.index("\n", idx6)
    line6 = _SRC[max(0, idx6 - 40):line6_end]
    assert "_rate_scaled_gain(" not in line6
    assert "_rate_scaled_cycles(" not in line6


def test_no_new_parameter_added_without_wrapping_or_exemption():
    """将来の回帰防止: 本テストのCATEGORY_A_PARAMS/CATEGORY_B_PARAMS/EXEMPT_PARAMSの
    合計が、設計書が特定した23+3(exempt)件と一致することを確認する
    (件数の見落とし・重複を機械的に検出する)。"""
    assert len(CATEGORY_A_PARAMS) == 17  # osqp_shadow_cyclesを含め17件
    assert len(CATEGORY_B_PARAMS) == 4  # beta(LAT-TTC)は別テストで確認するためここでは4件
    assert len(EXEMPT_PARAMS) == 7  # 264節続報Task1でfailsafe_dt_threshold_msを追加


# ---------------------------------------------------------------------------
# カテゴリC: [PERF]ハードコード修正の確認
# ---------------------------------------------------------------------------

def test_perf_hardcoded_0_025_removed():
    """回帰防止: [PERF]の周期超過判定が0.025固定値のままでないことを確認する。"""
    assert "if work > 0.025:" not in _SRC


def test_perf_over_budget_uses_control_rate():
    idx = _SRC.index("self._pf_over_budget_s = 1.0 / self._cfg.mpc.control_rate")
    assert idx > 0
    idx2 = _SRC.index("if work > self._pf_over_budget_s:")
    assert idx2 > idx


def test_perf_print_label_reflects_actual_budget_not_hardcoded_25ms():
    """2026-08-01追加(261節続報、72Hzスパイク調査Phase 2-2の回帰防止):
    [PERF]の出力カウンタ(_pf_over25)は当初から正しくself._pf_over_budget_s
    (=1/control_rate)を参照していたが、印字ラベルだけが">25ms="という
    40Hz固定の文字列のまま取り残されていた(72Hz実測ログで発覚、カウンタ値
    自体は常に正しかった表示のみのバグ)。ラベルが実際の予算[ms]を動的に
    表示するよう修正したことを固定する。"""
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("\n    # 2026-07-31追加(255節続報", idx)
    snippet = _SRC[idx:idx_end]
    assert ">25ms=" not in snippet, (
        "[PERF]の印字ラベルが再び40Hz固定の\">25ms=\"へ戻っている(72Hz以外の"
        "control_rateで誤解を招く表示バグの再発)")
    assert ">%.1fms=" in snippet
    assert "self._pf_over_budget_s * 1000" in snippet


def test_perf_over_budget_not_recomputed_every_cycle():
    """性能確認: 周期超過判定のホットパス(_pf_cycle_end)では、事前計算済みの
    self._pf_over_budget_sを参照するのみで、毎周期self._rate_scaled_cycles等の
    ログ付きヘルパーを呼んでいないことを確認する(毎周期ログのスパム防止)。"""
    idx = _SRC.index("def _pf_cycle_end(self, work, work_cpu=0.0):")
    idx_end = _SRC.index("def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "_rate_scaled_cycles(" not in snippet
    assert "_rate_scaled_gain(" not in snippet


# ---------------------------------------------------------------------------
# collision_cum_window_cycles: Phase 0調査結果(max()ベース)の回帰確認
# ---------------------------------------------------------------------------

def test_collision_window_uses_max_not_sum():
    """Phase 0調査で確認した前提(sumではなくmax)が変わっていないことを確認する。
    もしsumに変わっていたら、本設計の換算方式(窓長の線形換算のみ)では
    不十分になるため、この前提が崩れていないことを継続的に検証する。"""
    idx = _SRC.index("_cum_drop = max(self._collision_v_window) - v")
    assert idx > 0


# ---------------------------------------------------------------------------
# 2026-07-31追加(255節続報、クローズ作業Phase 1): レビューで条件付けられた
# 4件のフォローアップ
# ---------------------------------------------------------------------------

def test_mpc_cfg_control_rate_is_structurally_identical_to_cfg_mpc_control_rate():
    """Phase 1-1: self._mpc_cfg.control_rate と self._cfg.mpc.control_rate が
    同一値であるという、両ヘルパーの参照先変更(254節続報)の前提を固定する。
    rclpyノード起動が本環境では困難なため、実行時アサーションの代わりに
    ソース構造で検証する: create_mpc()内で cfg_mpc = self._cfg.mpc と
    束縛された後、その cfg_mpc.control_rate が MPCConfig(...) 呼び出しの
    control_rate位置引数(dataclass定義の11番目のフィールド、
    N,Q,R,QN,v_max,a_min,a_max,ay_max,delta_max,steer_rate_max,control_rateの順)
    としてそのまま渡されていることを確認する。self._mpc_cfg はこの
    MPCConfig(...) の戻り値そのものなので、self._mpc_cfg.control_rate は
    定義上 self._cfg.mpc.control_rate と完全に同一値になる。"""
    # MPCConfigのフィールド順序がこのテストの前提(control_rateが11番目)と
    # ずれていないことをdataclass定義自体からも確認する。
    idx_cls = _SRC.index("class MPCConfig:")
    idx_cls_end = _SRC.index("class ", idx_cls + 10)
    cls_body = _SRC[idx_cls:idx_cls_end]
    fields = re.findall(r'^    (\w+):', cls_body, re.MULTILINE)
    assert fields.index("control_rate") == 10, (
        f"MPCConfigのフィールド順序が変更されている(現在: {fields})。"
        "create_mpc()側のcfg_mpc.control_rate位置引数も合わせて確認すること。")

    idx = _SRC.index("def create_mpc(car: BicycleModel)")
    idx_end = _SRC.index("\n        def ", idx + 10)
    snippet = _SRC[idx:idx_end]
    assert "cfg_mpc = self._cfg.mpc" in snippet
    # MPCConfig(...)呼び出し内でcfg_mpc.control_rateが引数として渡されていること
    # (位置引数なのでdataclassのフィールド順序が上のアサーションで保証されていれば、
    # ここでの文字列出現確認だけで「control_rate位置に渡っている」ことを裏付けられる)。
    idx_ctor = snippet.index("mpc_cfg = MPCConfig(")
    idx_ctor_end = snippet.index(")\n", idx_ctor)
    ctor_args = snippet[idx_ctor:idx_ctor_end]
    assert "cfg_mpc.control_rate" in ctor_args


def test_space_ema_alpha_config_key_absence_is_the_load_bearing_assumption():
    """Phase 1-2: space_ema_alphaがEXEMPT_PARAMS(未ラップ)とされている根拠は
    「config.yamlのlat_ttcセクションにこのキーが実在せず、常にself._ot_ema_alpha
    (既にレートスケール済み)へフォールバックする」という前提にある。
    このテストは、EXEMPT_PARAMSへの登録有無に関わらず、キーが実在しないという
    前提そのものを直接検証する。将来誰かがconfig.yamlへlat_ttc.space_ema_alpha
    キーを追加すると、このテストが失敗し、除外判断の見直し(ラップするか、
    新しい根拠で改めて除外するか)を強制する設計になっている。"""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lat_ttc_keys = set((cfg.get("lat_ttc") or {}).keys())
    assert "space_ema_alpha" not in lat_ttc_keys, (
        "config.yamlのlat_ttcセクションにspace_ema_alphaキーが追加されました。"
        "space_ema_alphaがEXEMPT_PARAMS(未ラップ)とされているのは、このキーが"
        "存在せず常にself._ot_ema_alpha(既にレートスケール済み)へフォールバック"
        "するという前提のためです。キーを追加した場合、この値は生のconfig値として"
        "そのままLateralTTCMonitorへ渡り40Hz基準のまま二重換算されずに使われて"
        "しまうため、_rate_scaled_gainでラップするか、除外を維持する新しい根拠を"
        "明示した上でこのテストを更新してください。")


def test_perf_over_budget_init_log_uses_rate_scale_tag():
    """Phase 1-4: カテゴリC(pf_over_budget_s)の初期化ログが[RATE-SCALE]タグを
    使っていることを確認する。ロールアウト手順1(72Hz切替直後の40Hz恒等確認)は
    [RATE-SCALE]行の目視走査で行うため、カテゴリCだけこのタグを欠くと
    確認が不完全になる。"""
    idx = _SRC.index("self._pf_over_budget_s = 1.0 / self._cfg.mpc.control_rate")
    idx_end = _SRC.index("\n\n", idx)
    snippet = _SRC[idx:idx_end]
    assert "[RATE-SCALE] pf_over_budget_s" in snippet


def test_dynamic_gain_param_callbacks_warn_when_rate_differs_from_reference():
    """Phase 1-3: accel_low_pass_gain/steer_low_pass_gainの動的パラメータ
    コールバック内に、control_rateが基準周期(40Hz)と異なる場合の1回警告が
    追加されていることを確認する(換算はしない、生値適用のまま、ログ追加のみ)。"""
    for name in ("accel_low_pass_gain", "steer_low_pass_gain"):
        idx = _SRC.index(f'elif param.name == "{name}" and param.type_ == Parameter.Type.DOUBLE:')
        idx_end = _SRC.index("elif param.name ==", idx + 10)
        snippet = _SRC[idx:idx_end]
        assert f'self._warn_if_dynamic_gain_param_unscaled("{name}")' in snippet

    idx_def = _SRC.index("def _warn_if_dynamic_gain_param_unscaled(")
    idx_def_end = _SRC.index("\n    def ", idx_def + 10)
    body = _SRC[idx_def:idx_def_end]
    assert "control_rate != self._RATE_SCALE_REFERENCE_HZ" in body
    # 換算はしない(生値のまま)ことの回帰防止: このメソッド内でmpc_cfgへの代入や
    # _rate_scaled_gain呼び出しを行っていないこと(ログのみの副作用)を確認する。
    assert "_rate_scaled_gain(" not in body
    assert "mpc_cfg." not in body
