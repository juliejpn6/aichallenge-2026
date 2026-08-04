"""Unit tests for the diagnostic-logging bypass flag ([CONFIG] enable_diag_log,
214節, 2026-07-28)。

背景: 予選環境スペックが3vCPU/12GiBに確定した。制御ループ(_control())が
create_timerではなく「while + create_rate(40Hz).sleep()」の単一同期関数
(約2000行)で実装されており、STEER-XCORR/GNSS-EKF-XCORRの1Hzクロス相関計算が
_publish_control_command呼び出しより前にインラインで実行される構造であるため、
これらの計算が重いとその周期のpublish自体を直接遅延させ得ることが判明した
(Gemini提案のCallback Group分離は、そもそも分離対象のタイマーコールバックが
存在しないため前提が誤りと判断し見送った)。

対処: 診断計算(STEER-XCORR/GNSS-EKF-XCORR)自体をバイパスする
`enable_diag_log`フラグ(既定true=現状維持)を追加した。

重要な訂正: Geminiの当初案は「R-DELTA-SWING等の診断計算およびログ出力ブロック
全体をバイパス」するよう指示していたが、R-DELTA-SWINGブロックはログだけでなく
実際のR[delta]動的引き上げ機構(176節、curvature swing対策の制御ロジック本体、
update_R呼び出しを含む)を含んでいる。ブロック全体をenable_diag_logでバイパス
すると、falseにしたとき制御機能そのものが無効化されてしまうため、ログ出力文の
みをバイパス対象とし、swing計算/update_R呼び出しは無条件で常に実行されるよう
修正した。

同様にSTEER-XCORRの計算軽量化についても、Gemini当初案(ラグ探索範囲を
150-250msへ縮小)は188節で実測遅延が110msだった前例と矛盾するリスクがあるため
採用せず、探索範囲(-0.05〜0.4s)は維持したままstep幅のみ0.01→0.025へ粗くする
(46点→19点)代替案を採用した。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。配線
(dataclassフィールド・パラメータ宣言・コールバック配線・呼び出し箇所の
ガード有無)は構造的なソーステキスト検証で確認する。
"""
import os
import re

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG = _f.read()


# ---------------------------------------------------------------------------
# 1) MPCConfig dataclass / config.yaml / create_mpc()配線
# ---------------------------------------------------------------------------

def test_mpc_config_has_enable_diag_log_field_default_true():
    assert "enable_diag_log: bool = True" in _SRC


def test_config_yaml_declares_enable_diag_log_as_valid_bool():
    """2026-08-04: 値そのもの(true/false)は「ローカル調査中はtrue・予選提出時は
    false」という意図的な運用切り替え対象(CLAUDE.md §1.2)であり、どちらの値でも
    正当なため固定値ではなくtrue/falseいずれかであることのみを検査する(退行防止の
    趣旨は「キーが存在し有効なbool値である」ことの確認に絞る、弱体化ではない)。"""
    assert re.search(r"^\s*enable_diag_log:\s*(true|false)\s*(#.*)?$", _CFG, re.MULTILINE)


def test_create_mpc_passes_enable_diag_log_with_default_true_fallback():
    assert 'bool(getattr(cfg_mpc, "enable_diag_log", True))' in _SRC


# ---------------------------------------------------------------------------
# 2) ROS2パラメータ宣言・ライブ更新・起動時ログ
# ---------------------------------------------------------------------------

def test_declare_parameter_registers_enable_diag_log():
    assert 'self.declare_parameter("enable_diag_log", mpc_cfg.enable_diag_log)' in _SRC


def test_param_callback_handles_enable_diag_log_bool_update():
    assert 'param.name == "enable_diag_log" and param.type_ == Parameter.Type.BOOL' in _SRC
    assert "mpc_cfg.enable_diag_log = bool(param.value)" in _SRC


def test_startup_highlight_log_present():
    """Task3: 予選提出時の切替忘れ防止。起動時に必ず現在値をログ出力する。"""
    assert '"[CONFIG] enable_diag_log: "' in _SRC


# ---------------------------------------------------------------------------
# 3) 呼び出し箇所のガード: STEER-XCORR/GNSS-EKF-XCORRは丸ごとバイパス対象
# ---------------------------------------------------------------------------

def test_steer_xcorr_and_gnss_ekf_xcorr_call_site_gated_by_flag():
    m = re.search(
        r"if \(self\._loop % int\(max\(1, self\._mpc_cfg\.control_rate\)\) == 0\s*"
        r"\n\s*and self\._mpc_cfg\.enable_diag_log\):\s*\n"
        r"\s*self\._maybe_log_gnss_ekf_xcorr\(\)\s*\n"
        r"\s*self\._maybe_log_steer_xcorr\(\)",
        _SRC)
    assert m is not None, "STEER-XCORR/GNSS-EKF-XCORR呼び出しがenable_diag_logでガードされていません"


# ---------------------------------------------------------------------------
# 4) R-DELTA-SWING: ログ文のみガード、swing計算/update_Rは無条件実行のまま
# ---------------------------------------------------------------------------

def test_r_delta_swing_log_statement_gated_by_flag():
    m = re.search(
        r"if \(self\._r_delta_swing_dbg_loop % int\(max\(1, self\._mpc_cfg\.control_rate\)\) == 0\s*"
        r"\n\s*and self\._mpc_cfg\.enable_diag_log\):\s*\n"
        r"\s*self\.get_logger\(\)\.info\(\s*\n"
        r"\s*f\"\[R-DELTA-SWING\]",
        _SRC)
    assert m is not None, "R-DELTA-SWINGログ文がenable_diag_logでガードされていません"


def test_r_delta_swing_update_r_call_not_gated_by_flag():
    """swing計算/update_R呼び出しは制御ロジック本体(176節)であり、診断ではない。
    enable_diag_log=falseでも常に実行されなければならない(バイパス対象外)。"""
    update_r_idx = _SRC.index("self._mpc.update_R(sparse.diags(_r))")
    # このupdate_R呼び出しの直前100文字以内に「if ... enable_diag_log」というガードが
    # 無いこと(=無条件実行であること)を確認する。
    preceding = _SRC[max(0, update_r_idx - 300):update_r_idx]
    assert "enable_diag_log" not in preceding, (
        "update_R呼び出しがenable_diag_logでガードされてしまっています"
        "(falseにするとR[delta]swing制御機能自体が無効化されるバグ)")


def test_r_delta_swing_ema_and_smoothstep_calc_not_gated_by_flag():
    """_swing/_smooth_swの算出自体(EMA平滑化含む)も制御ロジックの一部であり、
    ログ用ガードの外側(無条件実行)になければならない。"""
    swing_calc_idx = _SRC.index("_swing_raw = max(_kappas_fwd) - min(_kappas_fwd)")
    preceding = _SRC[max(0, swing_calc_idx - 300):swing_calc_idx]
    assert "enable_diag_log" not in preceding


# ---------------------------------------------------------------------------
# 5) STEER-XCORR軽量化: 範囲は維持、stepのみ粗くする(範囲縮小はしない)
# ---------------------------------------------------------------------------

def test_steer_xcorr_lag_range_still_covers_original_span():
    """188節で実測遅延110msが観測された前例があるため、探索範囲を
    150-250ms等へ狭めていないことを確認する(範囲は-0.05〜0.4sのまま)。"""
    assert "_lag_lo, _lag_hi = -0.05, 0.4" in _SRC


def test_steer_xcorr_lag_step_widened_for_lightweight_search():
    assert "_lag_step = 0.025" in _SRC
    assert "_lag_step = 0.01  # [s]" not in _SRC


def test_steer_xcorr_lag_point_count_reduced_by_at_least_half():
    import numpy as np
    lag_lo, lag_hi, lag_step = -0.05, 0.4, 0.025
    lags = np.arange(lag_lo, lag_hi + 1e-9, lag_step)
    old_lags = np.arange(lag_lo, lag_hi + 1e-9, 0.01)
    assert len(lags) <= len(old_lags) * 0.5
    assert len(lags) >= 15  # Gemini提案の「15〜20点程度」を満たす


# ---------------------------------------------------------------------------
# 6) スコープ外の診断(HOTSPOT-DEVIATION/LOC-XCHECK)はバイパス対象にしていない
# ---------------------------------------------------------------------------

def test_hotspot_deviation_call_site_not_gated_by_flag():
    """HOTSPOT-DEVIATIONはピーク追跡のため毎周期呼ばれ計算量も軽いこと、
    まだコーナー立ち上がりリンギング調査で使用中であることから、今回の
    バイパス対象(重い1Hzクロス相関×2種)の範囲外とした。直前の行(コメント含む)に
    enable_diag_logによるガードが無く、呼び出しがトップレベルの文であることを
    行単位で確認する(charベースの窓は別ブロックの残骸を拾い誤検知するため使わない)。"""
    lines = _SRC.splitlines()
    call_line_idx = next(
        i for i, l in enumerate(lines) if "self._maybe_log_hotspot_deviation()" in l)
    call_line = lines[call_line_idx]
    assert call_line.startswith("        self.")  # 8スペース=ifの中ではないトップレベル文
    # 直前の非空行(コメント除く)にif ... enable_diag_log: のような開き括弧が無いこと
    prev_code_lines = [
        l for l in lines[max(0, call_line_idx - 3):call_line_idx]
        if l.strip() and not l.strip().startswith("#")]
    assert not any("enable_diag_log" in l for l in prev_code_lines)


def test_loc_xcheck_log_not_gated_by_flag():
    """LOC-XCHECKは相関分析(213節続報)で偶然のベースラインとほぼ同水準(倍率1.14倍)と
    判明しており、重い計算でもないため今回のバイパス対象外とした。行単位で、
    このブロックを開くif文自体にenable_diag_logが含まれていないことを確認する。"""
    lines = _SRC.splitlines()
    log_line_idx = next(
        i for i, l in enumerate(lines) if 'f"[LOC-XCHECK] wp={_idx}' in l)
    # LOC-XCHECKのif開始行(control_rate // 4間引き)を直近8行以内から探す
    if_line = next(
        l for l in reversed(lines[max(0, log_line_idx - 8):log_line_idx])
        if l.strip().startswith("if "))
    assert "enable_diag_log" not in if_line
