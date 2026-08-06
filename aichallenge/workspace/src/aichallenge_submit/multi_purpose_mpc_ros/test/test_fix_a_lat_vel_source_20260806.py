"""Fix A'(opp_lat_pred根本修正+片側利用+変位物理拘束、2026-08-06)。

背景: design_docs/opp_lat_pred_overlap_guard_design_20260806.md。対象車
横方向速度の自前差分(40Hz固定dtで、実測≈15.2HzのV2X位置を単純差分するため
13Hzエイリアシングにより最大26.7m/s級の物理的にあり得ない値を生む、19節・
6.8節で実データ確認済み)を、既存のV2Xトラッカー速度推定(窓端点差分、
v2x_vehicle_tracker.py、既に本番投入済みでv_long計算にも使われている資産)
の再利用へ置き換える。

あわせて、Phase 1実データ検証(6.8節)で判明した「t_reach外挿(最大1.5秒)は
速度推定がどれだけ正確でも本質的に脆く、min_neededが縮小方向(=衝突リスク
増)へ逆転しうる」という問題(wp85事例)に対し、外部AI(Gemini・別Claude)
レビュー(6.9節)を踏まえ以下2点をFix A自体のスコープへ統合する(Fix A'):
1. 予測の片側利用: 予測は必要クリアランスを増やす方向にのみ使い、投機的な
   縮小は許さない(need_from_pred/need_from_nowのmax)。
2. 変位の物理拘束: 既存の速度クランプ×t_reach上限という導出値で外挿量の
   絶対値を上限クランプする(新規マジックナンバー0個)。

configゲート`overtake.lat_vel_source_tracker`(既定false)でON/OFF切替、
OFF時は現行の自前差分パイプラインと完全にビット等価(段階導入)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと
同じ「ソーステキスト構造検証」+「ロジックのミラー実装による数値検証」の
方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG_SRC = _f.read()


# ---------------------------------------------------------------------------
# ①config/状態変数: 新規ゲートが既定falseで宣言されていること
# ---------------------------------------------------------------------------

def test_config_yaml_has_lat_vel_source_tracker_key():
    assert "lat_vel_source_tracker:" in _CFG_SRC


def test_state_var_declared_with_safe_default_false():
    assert ('self._ot_lat_vel_source_tracker = bool(\n'
            '                _otget("lat_vel_source_tracker", False))') in _SRC


# ---------------------------------------------------------------------------
# ②新規ヘルパー_estimate_opp_lateral_velocity: tracker.velocity()を再利用し
#   新規の速度源を追加しないこと、is_settled()未充足時はNoneを返すこと
# ---------------------------------------------------------------------------

def _helper_block():
    idx = _SRC.index("def _estimate_opp_lateral_velocity(")
    idx_end = _SRC.index("def _scan_traffic(", idx)
    return _SRC[idx:idx_end]


def test_helper_reuses_tracker_velocity_no_new_source():
    snippet = _helper_block()
    assert "tracker.velocity(vid)" in snippet
    assert '"_v2x_tracker"' in snippet


def test_helper_returns_none_when_not_settled():
    """未観測vid/速度窓未充足時、「相手速度0」と「データなし」を混同しない
    (must-fix 3、外部AIレビュー2026-08-06)。"""
    snippet = _helper_block()
    assert "tracker.is_settled(vid)" in snippet
    assert "return None" in snippet


def test_helper_projects_onto_lateral_axis_same_rotation_as_v_long():
    """_scan_traffic()のv_long計算(cos(wp.psi)*vx+sin(wp.psi)*vy)と対になる
    直交成分(-sin(wp.psi)*vx+cos(wp.psi)*vy)であることを確認する(新規の
    回転行列を作らない)。"""
    snippet = _helper_block()
    assert "_math.sin(wp.psi) * vx" in snippet
    assert "_math.cos(wp.psi) * vy" in snippet


def _estimate_opp_lateral_velocity_mirror(vx, vy, psi):
    """helper関数のミラー実装(数値検証用)。"""
    import math
    return -math.sin(psi) * vx + math.cos(psi) * vy


def test_lateral_projection_orthogonal_to_forward_projection():
    """前方射影(v_long)と横射影(本ヘルパー)が直交する(内積0)ことを、
    ミラー実装で数値的に確認する。"""
    import math
    vx, vy, psi = 5.0, 2.0, 0.7
    lat = _estimate_opp_lateral_velocity_mirror(vx, vy, psi)
    fwd = math.cos(psi) * vx + math.sin(psi) * vy
    # 元のベクトル(vx,vy)は fwd*forward_dir + lat*lateral_dir に分解できるはず
    # (forward_dir, lateral_dirは直交単位ベクトル)ので、大きさの二乗が保存する。
    assert abs((fwd ** 2 + lat ** 2) - (vx ** 2 + vy ** 2)) < 1e-9


# ---------------------------------------------------------------------------
# ③呼び出し側: ゲートOFF時は旧実装がそのまま実行され(elif化のみ)、
#   ゲートON時は新実装が使われること
# ---------------------------------------------------------------------------

def test_gate_off_path_preserves_old_ema_logic_unchanged():
    idx = _SRC.index("if self._ot_lat_vel_source_tracker:")
    idx_end = _SRC.index("self._ot_opp_lat_prev = _opp_lat_now", idx)
    snippet = _SRC[idx:idx_end]
    assert "elif (self._ot_opp_lat_prev is not None" in snippet
    assert "and self._ot_opp_lat_prev_vid == self._ot_target_vid):" in snippet
    # 旧実装のクランプ+EMA本体が無変更で残っていること
    assert "_raw_lat_vel = (_opp_lat_now - self._ot_opp_lat_prev) / _dt" in snippet
    assert "self._ot_opp_lat_vel_ema += self._ot_ema_alpha * (" in snippet


def test_gate_on_path_calls_new_helper():
    idx = _SRC.index("if self._ot_lat_vel_source_tracker:")
    idx_end = _SRC.index("elif (self._ot_opp_lat_prev is not None", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._estimate_opp_lateral_velocity(" in snippet
    assert "self._ot_min_needed_lat_vel_clamp" in snippet  # 既存クランプを流用


# ---------------------------------------------------------------------------
# ④変位物理拘束: 新規マジックナンバーを使わず既存定数の積であること
# ---------------------------------------------------------------------------

def test_displacement_bound_derived_from_existing_constants_no_new_magic_number():
    idx = _SRC.index("_max_disp = (self._ot_min_needed_lat_vel_clamp")
    idx_end = idx + 400
    snippet = _SRC[idx:idx_end]
    assert "self._ot_min_needed_horizon_cap_s" in snippet
    assert "_disp = max(-_max_disp, min(" in snippet


def test_displacement_bound_mirror_numeric():
    """変位物理拘束のミラー数値検証: |pred - lat_now|が
    clamp×horizon_cap以下に収まること。"""
    clamp = 2.0
    horizon_cap = 1.5
    lat_now = -0.7
    lat_vel = 5.0  # クランプ前提として明らかに過大な値を投入
    t_reach = 1.5
    pred_raw = lat_now + lat_vel * t_reach
    max_disp = clamp * horizon_cap
    disp = max(-max_disp, min(max_disp, pred_raw - lat_now))
    pred = lat_now + disp
    assert abs(pred - lat_now) <= max_disp + 1e-9


# ---------------------------------------------------------------------------
# ⑤片側利用: 予測がクリアランスを縮める方向には効かないこと
# ---------------------------------------------------------------------------

def test_one_sided_use_present_in_source():
    idx = _SRC.index("if self._ot_lat_vel_source_tracker:")
    idx2 = _SRC.index("if self._ot_lat_vel_source_tracker:", idx + 1)
    idx_end = _SRC.index("self._ot_last_valid_min_needed_mag = _target_mag", idx2)
    snippet = _SRC[idx2:idx_end]
    assert "_need_from_pred = max(0.0, min(" in snippet
    assert "_need_from_now = max(0.0, min(" in snippet
    assert "_target_mag = max(_need_from_pred, _need_from_now)" in snippet


def _one_sided_target_mag_mirror(side, opp_lat_pred, opp_lat_now, clear_needed, d_off):
    need_from_pred = max(0.0, min(d_off, side * opp_lat_pred + clear_needed))
    need_from_now = max(0.0, min(d_off, side * opp_lat_now + clear_needed))
    return max(need_from_pred, need_from_now)


def test_one_sided_use_mirror_wp85_style_regression():
    """名前付きテストケース(6.9節、wp85): 予測が「相手が離れていく」方向
    (=必要クリアランスを縮める方向)に大きく振れても、min_neededが
    現在位置ベースの値を下回らないことを確認する。"""
    side = -1
    opp_lat_now = -0.708
    opp_lat_pred = 3.286  # 実データ検証(6.8節)で観測された投機的な予測値
    clear_needed = 2.05
    d_off = 3.0
    target_mag = _one_sided_target_mag_mirror(
        side, opp_lat_pred, opp_lat_now, clear_needed, d_off)
    need_from_now = max(0.0, min(d_off, side * opp_lat_now + clear_needed))
    assert target_mag >= need_from_now
    assert target_mag > 0.0  # 6.8節で観測された0.000への逆転が起きないこと


def test_one_sided_use_still_allows_increase_when_opponent_approaches():
    """片側利用は「増やす」方向は妨げない(相手接近の先読み機能は維持される)
    ことを確認する。"""
    side = 1
    opp_lat_now = 0.5
    opp_lat_pred = 1.5  # 相手がこちらへ寄ってくる(必要クリアランス増)予測
    clear_needed = 1.0
    d_off = 3.0
    target_mag = _one_sided_target_mag_mirror(
        side, opp_lat_pred, opp_lat_now, clear_needed, d_off)
    need_from_now = max(0.0, min(d_off, side * opp_lat_now + clear_needed))
    assert target_mag > need_from_now  # predの方が大きく採用されている


# ---------------------------------------------------------------------------
# ⑥診断ログ: lat_vel_srcマーカーが[OT]ログへ追加されていること
# ---------------------------------------------------------------------------

def test_ot_log_includes_lat_vel_src_marker():
    idx = _SRC.index('f"[OT] state=')
    snippet = _SRC[idx:idx + 13000]
    assert "lat_vel_src={_fwd_dbg.get('lat_vel_src')}" in snippet


def test_lat_vel_src_field_reflects_gate_value():
    idx = _SRC.index('_fwd_dbg["lat_vel_src"] = (')
    snippet = _SRC[idx:idx + 150]
    assert '"tracker" if self._ot_lat_vel_source_tracker else "diff"' in snippet
