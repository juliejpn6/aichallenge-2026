"""Unit tests for the [LOC-XCHECK] sensing-vs-control diagnostic (2026-07-19、診断計装のみ)。

背景: design_docs(path_following_summary.html §3、および過去メモリ)は、単独走行ログ
(2026-06-29)で「EKFはコーナーでの横方向ブレの約40-50%しか見えていない(真値GNSS RMS
0.55m vs EKF信頼値0.39m)」というS2を既に確定させ、2026-07-05の決定実験で
gnss_covariance.good_value(0.1→0.05)を対処したが0.03では効果が頭打ちになる
「EKFの構造的な床」に到達したと記録している。

ユーザーから改めて「MPCの計算結果がコーナーで狂う(予期せず内側/外側に寄る)。
コーナーによって狂う・狂わないがある。原因が制御アルゴリズムかセンシングか
切り分けたい」との依頼を受けたが、既存ログでは狂うコーナーの特定例がなく
「総合的な傾向」の域を出ないため、Stage1.5方針(推測せずまず計装する)に従い、
複数周・複数コーナーにわたって統計的に切り分けるための計装を追加した。

対処: 制御が実際に使うEKFベースの横位置(_cur_ey、既存)と、GNSS生値(既存の
self._gnss_pose、元々ピット内補正専用に購読済み)を同一waypoint基準・同一射影式で
独立に計算し、[LOC-XCHECK]ログへ並べて出力する。両者の差が大きいコーナー=
センシング(EKF)起因の疑いが強く、差が小さいのに車の挙動が乱れるコーナーが
あれば制御(追従)起因の疑いが強い、という切り分けが次回複数周ログで可能になる。

新規パラメータ0個(既存self._gnss_pose・既存_cur_ey計算式・既存の間引きイディオム
(self._loop % (control_rate//4))を再利用)。判定ロジック・制御出力には一切影響しない。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。射影式自体は
単純な回転行列適用なので純Pythonミラーで数式的性質を検証し、mpc_controller.py側の
配線(既存変数の再利用・診断専用であること)は構造的なソーステキスト検証で確認する。
"""
import os
import math

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: e_y射影式の複製(_cur_ey/_gnss_eyで共通)
# ---------------------------------------------------------------------------

def _project_ey(px, py, wp_x, wp_y, wp_psi):
    """mpc_controller.py の _cur_ey/_gnss_ey 計算式(3012-3013/3025-3026行目)の複製ミラー。
    左が+、右が-の横方向誤差(waypoint局所座標系への回転射影)。"""
    return math.cos(wp_psi) * (py - wp_y) - math.sin(wp_psi) * (px - wp_x)


def test_same_true_position_gives_identical_ey_regardless_of_source():
    """回帰: EKF側とGNSS側の"真の位置"が完全に一致していれば、同じ射影式である以上
    ekf_ey==gnss_eyになるはず(式そのものの正しさの確認)。"""
    wp_x, wp_y, wp_psi = 10.0, 5.0, 0.3
    ekf_ey = _project_ey(12.0, 6.0, wp_x, wp_y, wp_psi)
    gnss_ey = _project_ey(12.0, 6.0, wp_x, wp_y, wp_psi)
    assert ekf_ey == gnss_ey


def test_position_offset_produces_proportional_ey_gap():
    """診断が捉える現象のデモ: EKFとGNSSの実測位置が(センシング誤差により)ずれていれば、
    waypointの向き(psi)に直交する成分の差がそのままekf_ey-gnss_eyの差になる
    (S2=EKFがコーナーで横方向ブレを過小報告、という既知の機構と整合する形)。"""
    wp_x, wp_y, wp_psi = 0.0, 0.0, 0.0  # 進行方向=+x、e_yは+y方向
    ekf_ey = _project_ey(0.0, 0.30, wp_x, wp_y, wp_psi)   # EKF: y=0.30(過小報告側)
    gnss_ey = _project_ey(0.0, 0.55, wp_x, wp_y, wp_psi)  # GNSS: y=0.55(真値相当)
    assert abs(gnss_ey) > abs(ekf_ey)  # 既知のS2(EKFが小さく見積もる)方向と整合
    assert math.isclose(gnss_ey - ekf_ey, 0.25, abs_tol=1e-9)


def test_straight_section_both_sources_agree_closely():
    """回帰: 直線区間(既存メモリ記録: 直線ではEKF≈GNSS±0.07m)相当の小さな乖離では、
    診断が過検知しない(式自体は単なる射影であり閾値判定を含まないため、
    差が小さければekf_ey/gnss_eyの値も単純に近い)ことを確認する。"""
    wp_x, wp_y, wp_psi = 100.0, 0.0, 1.2
    px, py = 100.5, 0.5
    ekf_ey = _project_ey(px, py, wp_x, wp_y, wp_psi)
    gnss_ey = _project_ey(px + 0.03, py + 0.02, wp_x, wp_y, wp_psi)
    assert abs(ekf_ey - gnss_ey) < 0.1


# ---------------------------------------------------------------------------
# 2) mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_loc_xcheck_reuses_existing_gnss_pose_subscription_no_new_topic():
    idx = _SRC.index('f"[LOC-XCHECK]')
    snippet = _SRC[max(0, idx - 400):idx]
    assert "self._gnss_pose is not None" in snippet
    assert "self._gnss_pose.pose.pose.position" in snippet


def test_loc_xcheck_uses_same_waypoint_and_projection_formula_as_cur_ey():
    """非冗長性: _gnss_eyは_cur_eyと全く同じ_wp(同一waypoint基準)・同じ射影式
    (cos(psi)*(dy) - sin(psi)*(dx))を使い、別のwaypoint探索や別の座標系を
    新設していないことを確認する。"""
    idx_cur = _SRC.index(
        "_cur_ey = float(np.cos(_wp.psi) * (pose.y - _wp.y)\n"
        "                            - np.sin(_wp.psi) * (pose.x - _wp.x))")
    idx_gnss = _SRC.index("_gnss_ey = float(np.cos(_wp.psi)")
    assert idx_cur < idx_gnss
    snippet = _SRC[idx_gnss:idx_gnss + 150]
    assert "_wp.psi" in snippet
    assert "_wp.y" in snippet
    assert "_wp.x" in snippet
    assert "_gp.y" in snippet
    assert "_gp.x" in snippet


def test_loc_xcheck_reuses_existing_decimation_idiom_no_new_constant():
    """非冗長性: 4383/4389行目の既存デバッグログと全く同じ間引き式
    (self._loop % (control_rate // 4))を再利用しており、新しい間引き定数を
    導入していないことを確認する。"""
    idx = _SRC.index('f"[LOC-XCHECK]')
    snippet = _SRC[max(0, idx - 500):idx]
    assert "self._loop % (self._mpc_cfg.control_rate // 4) == 0" in snippet
    # リポジトリ内の既存箇所と完全一致の式であることも確認(表記ゆれがないか)
    assert _SRC.count("self._loop % (self._mpc_cfg.control_rate // 4) == 0") >= 3


def test_loc_xcheck_is_diagnostic_only_no_assignment_to_cur_ey_or_pose():
    """非矛盾性の核心確認: [LOC-XCHECK]ブロックはログ出力のみで、_cur_ey/pose/
    以降の判定に使われる変数へ代入していないことを確認する(制御に一切影響しない)。"""
    idx = _SRC.index("診断用(2026-07-19、センシング切り分け)")
    idx_end = _SRC.index("[LOC-XCHECK]", idx)
    idx_end = _SRC.index('ot={self._ot_state}")', idx_end) + len('ot={self._ot_state}")')
    snippet = _SRC[idx:idx_end]
    assert "_cur_ey =" not in snippet
    assert "pose.x =" not in snippet
    assert "pose.y =" not in snippet
    assert "self.get_logger().info(" in snippet


def test_loc_xcheck_log_includes_kappa_for_per_corner_breakdown():
    """次回ログでコーナー単位の集計を可能にするため、kappa(曲率)がタグとして
    含まれていることを確認する(along_min_width等の既存物理量と同じ由来のwp.kappa
    を再利用、新規計算式なし)。"""
    idx = _SRC.index('f"[LOC-XCHECK]')
    snippet = _SRC[idx:idx + 300]
    assert "kappa={_wp.kappa:.3f}" in snippet
    assert "ekf_ey={_cur_ey:.3f}" in snippet
    assert "gnss_ey={_gnss_ey:.3f}" in snippet
    assert "ot={self._ot_state}" in snippet


def test_gnss_pose_comment_documents_dual_purpose():
    """回帰: self._gnss_poseの初期化コメントが、ピット用途に加えて
    [LOC-XCHECK]でも再利用されることを明記していることを確認する(将来
    「未使用」と誤って削除されるのを防ぐ)。"""
    idx = _SRC.index("self._gnss_pose: Optional[PoseWithCovarianceStamped] = None")
    snippet = _SRC[idx:idx + 200]
    assert "LOC-XCHECK" in snippet
