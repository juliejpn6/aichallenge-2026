"""対象車横方向速度推定のノイズ耐性強化(301節続報、2026-08-05)。

背景: task#293候補①(300節、min_needed_mag速度考慮)のdev3実地検証中、d1がd3を
追い越し中(state=OVERTAKING side=1)に衝突する事象を発見した。ログを精査すると、
衝突直前に`opp_lat_pred=-2.352 t_reach=0.678`という値が出力されており、これを
逆算すると推定横方向速度は約-4.7m/sという物理的にあり得ない値だった。これにより
`min_needed_mag`が実際には不要な0.0まで縮み、コリドー側には2.69mの余裕
(corr_bound)があったにもかかわらずオフセットを一切取らずに直進し衝突した。

原因は2つ:
1. 微分(位置差分/dt)は元々ノイズを増幅する演算だが、生の差分値へのクランプが
   なかった(EKFジッタ・対象車切替直後の1周期ジャンプがそのままEMAへ混入)。
2. EMA初期値がフィルタなしの生の初回サンプルだった
   (`if ema is None: ema = raw_lat_vel`)。コールドスタート直後のノイズが
   無平滑で採用される欠陥。

対処(ユーザー指摘「異常値を弾いた後の平均値で計算すべき」): ①生の差分値を
物理的に妥当な範囲(既定±2.0m/s)へクランプしてからEMAへ入れる、②EMA初期値を
0.0(静止と仮定、安全側)にする、③十分なサンプル数(既定20周期≈0.5s@40Hz)が
貯まるまでは予測を使わず現在値のまま(300節導入前と同一の安全な挙動)とする、
の3点を追加した。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _lat_vel_ema_block():
    idx = _SRC.index("_raw_lat_vel = (_opp_lat_now - self._ot_opp_lat_prev) / _dt")
    idx_end = _SRC.index("self._ot_opp_lat_prev = _opp_lat_now", idx)
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①クランプ: 生の差分値を物理的に妥当な範囲へクランプしてからEMAへ入れる
# ---------------------------------------------------------------------------

def test_raw_lat_vel_is_clamped_before_feeding_ema():
    snippet = _lat_vel_ema_block()
    idx_clamp = snippet.index(
        "_raw_lat_vel = max(\n"
        "                            -self._ot_min_needed_lat_vel_clamp,")
    idx_ema_update = snippet.index("self._ot_opp_lat_vel_ema += self._ot_ema_alpha")
    # クランプ代入がEMA更新より前にあること
    assert idx_clamp < idx_ema_update


def test_clamp_config_key_exists_with_default():
    assert "min_needed_lat_vel_clamp" in _SRC
    assert "self._ot_min_needed_lat_vel_clamp = float(" in _SRC


# ---------------------------------------------------------------------------
# ②EMA初期値: 生の初回サンプルではなく0.0(静止と仮定)から開始する
# ---------------------------------------------------------------------------

def test_ema_seeds_at_zero_not_raw_first_sample():
    snippet = _lat_vel_ema_block()
    assert "if self._ot_opp_lat_vel_ema is None:" in snippet
    idx_if = snippet.index("if self._ot_opp_lat_vel_ema is None:")
    idx_next = snippet.index("self._ot_opp_lat_vel_ema += self._ot_ema_alpha", idx_if)
    seed_body = snippet[idx_if:idx_next]
    assert "self._ot_opp_lat_vel_ema = 0.0" in seed_body
    # 退行防止: 旧実装(生の初回サンプルをそのまま採用)が復活していないこと
    assert "self._ot_opp_lat_vel_ema = _raw_lat_vel" not in snippet


# ---------------------------------------------------------------------------
# ③ウォームアップ: 十分なサンプル数が貯まるまでは予測を使わず現在値のまま
# ---------------------------------------------------------------------------

def test_warmup_counter_increments_on_valid_sample():
    snippet = _lat_vel_ema_block()
    assert "self._ot_opp_lat_warmup_count += 1" in snippet


def test_warmup_counter_resets_on_target_switch():
    snippet = _lat_vel_ema_block()
    idx_else = snippet.rindex("else:")
    tail = snippet[idx_else:]
    assert "self._ot_opp_lat_warmup_count = 0" in tail


def test_prediction_gated_by_warmup_count():
    idx = _SRC.index(
        "if (self._ot_opp_lat_vel_ema is not None\n"
        "                            and self._ot_opp_lat_warmup_count")
    idx_end = _SRC.index("_fwd_dbg[\"opp_lat_pred\"]", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._ot_min_needed_lat_vel_warmup_cycles" in snippet
    assert "_opp_lat_pred = _opp_lat_now + self._ot_opp_lat_vel_ema * _t_reach" in snippet
    assert "_opp_lat_pred = _opp_lat_now" in snippet


def test_warmup_config_key_exists_with_default():
    assert "min_needed_lat_vel_warmup_cycles" in _SRC
    assert "self._ot_min_needed_lat_vel_warmup_cycles = self._rate_scaled_cycles(" in _SRC


# ---------------------------------------------------------------------------
# ④退行防止: 新規変数のリセット箇所数が既存のきょうだい変数と一致すること
# ---------------------------------------------------------------------------

def test_warmup_count_reset_site_count_matches_sibling_variable():
    """_ot_opp_lat_warmup_countのリセット箇所数が、既存のきょうだい変数
    _ot_opp_lat_vel_emaのリセット箇所数と一致すること。

    2026-08-07改訂(Fix B、design_docs opp_lat_pred_overlap_guard_design_
    20260806.md §4): 従来個別に4リセット箇所(側反転/rescue反転/新規
    エンゲージ/STUCK復帰)に重複実装されていたブロックを、共通ヘルパー
    _reset_ot_episode_tracking_state()へ統合した。そのため実際のソース上の
    出現数は「__init__(1) + ヘルパー定義内(1) + 本体ロジック内の
    コールドスタートリセット1箇所」の3箇所に減る(4リセット箇所は
    ヘルパー呼び出し1行に置き換わったため、個別カウントには現れない)。"""
    n_new = _SRC.count("self._ot_opp_lat_warmup_count = 0")
    n_sibling = _SRC.count("self._ot_opp_lat_vel_ema = None")
    assert n_new == n_sibling == 3, (
        f"想定は両方3箇所(__init__+ヘルパー定義+本体コールドスタート)だが "
        f"warmup_count={n_new}, vel_ema={n_sibling}")
