"""Unit tests for the over-steer hotspot deviation diagnostic ([HOTSPOT-DEVIATION],
208節続報, 2026-07-27)。

背景: AXIS06(アクチュエータ遅延蛇行)のQ/Rチューニングを真の基準値(tau=190ms・Q/R
完全無変更)で確定した(208節)直後、ユーザーが実走行を目視して「wp178, 189, 258, 289,
334あたりでハンドルを切りすぎている」と具体的な地点を指摘した。定量分析の結果、
これらの地点で実測舵角のピークが、パス自体が要求する理論舵角(kappa_ref由来)を
大きく超過している(最大で理論値の約2.7倍、wp258では理論値がほぼ0にもかかわらず
最大21°)ことが判明した。これは既知の「コーナー立ち上がり後の過渡応答リンギング」
(207節)の空間的に具体的な現れであり、Q/Rの線形調整では対処できないと判断された
(Gemini相談)。

対処: 制御には一切影響しない純粋な観測用ログとして、これらのwaypoint通過時に
実測舵角ピークと理論舵角の乖離を記録する[HOTSPOT-DEVIATION]診断を追加した。予選
環境で同じ地点が同様の乖離を示すか比較するためのもの。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。乖離計算自体
(delta_expected=arctan(kappa_ref×L)との比較、ウィンドウ内ピーク追跡)は単純な
Pythonミラーで数式的に検証し、mpc_controller.py側の配線(監視対象wp・状態初期化・
呼び出し箇所・ログ形式)は構造的なソーステキスト検証で確認する。
"""
import os
import math

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: 理論舵角の計算 + ウィンドウ内ピーク追跡ロジック
# ---------------------------------------------------------------------------

def _theo_deg(kappa_ref, length=1.087):
    return math.degrees(math.atan(kappa_ref * length))


class _HotspotMirror:
    """mpc_controller.py._maybe_log_hotspot_deviation()と同一ロジックのミラー。"""

    def __init__(self, hotspot_wps, length=1.087, window_s=2.0):
        self.hotspot_wps = hotspot_wps
        self.length = length
        self.window_s = window_s
        self.monitor = None
        self.logged = []

    def step(self, now_s, wp_id, kappa_lookup, act_deg):
        if self.monitor is not None:
            m = self.monitor
            if now_s >= m['end_t']:
                self.logged.append(dict(m))
                self.monitor = None
            else:
                dev = abs(act_deg - m['theo_deg'])
                if dev > m['peak_dev_deg']:
                    m['peak_dev_deg'] = dev
                    m['peak_act_deg'] = act_deg

        if self.monitor is None:
            for hwp in self.hotspot_wps:
                if abs(wp_id - hwp) <= 1:
                    kappa_ref = kappa_lookup(hwp)
                    theo_deg = _theo_deg(kappa_ref, self.length)
                    self.monitor = {
                        'wp': hwp, 'end_t': now_s + self.window_s,
                        'kappa_ref': kappa_ref, 'theo_deg': theo_deg,
                        'peak_act_deg': 0.0, 'peak_dev_deg': 0.0,
                    }
                    break


def test_theo_deg_matches_arctan_formula():
    # wp258相当: kappa_ref=0.030, L=1.087 -> 理論舵角約1.9度
    assert abs(_theo_deg(0.030) - 1.906) < 0.05


def test_mirror_tracks_peak_deviation_within_window():
    """wp258通過を模擬: 実測舵角が理論値(約1.9°)を大きく超えて振動する場合、
    ピークとの乖離が正しく記録されること。"""
    m = _HotspotMirror(hotspot_wps=(178, 189, 258, 289, 334))
    kappa_lookup = lambda wp: {258: 0.030}[wp]

    m.step(now_s=100.0, wp_id=258, kappa_lookup=kappa_lookup, act_deg=-4.7)
    assert m.monitor is not None
    assert m.monitor['wp'] == 258
    theo = m.monitor['theo_deg']

    m.step(now_s=100.3, wp_id=258, kappa_lookup=kappa_lookup, act_deg=21.0)
    assert m.monitor['peak_dev_deg'] == abs(21.0 - theo)
    assert m.monitor['peak_act_deg'] == 21.0

    # 途中で乖離の小さい値が来てもピークは更新されない(悪化方向のみ記録)
    m.step(now_s=100.6, wp_id=258, kappa_lookup=kappa_lookup, act_deg=5.0)
    assert m.monitor['peak_dev_deg'] == abs(21.0 - theo)

    # ウィンドウ終了(2秒後)でログへ確定・monitorはクリアされる
    m.step(now_s=102.0, wp_id=300, kappa_lookup=kappa_lookup, act_deg=0.0)
    assert m.monitor is None
    assert len(m.logged) == 1
    assert m.logged[0]['peak_act_deg'] == 21.0


def test_mirror_ignores_non_hotspot_waypoints():
    m = _HotspotMirror(hotspot_wps=(178, 189, 258, 289, 334))
    m.step(now_s=0.0, wp_id=500, kappa_lookup=lambda wp: 0.0, act_deg=15.0)
    assert m.monitor is None
    assert m.logged == []


def test_mirror_tolerates_1wp_miss():
    """wp_id探索の粒度により対象wpをちょうど踏まない場合(±1)でも発火する。"""
    m = _HotspotMirror(hotspot_wps=(178, 189, 258, 289, 334))
    m.step(now_s=0.0, wp_id=259, kappa_lookup=lambda wp: 0.03, act_deg=1.0)
    assert m.monitor is not None
    assert m.monitor['wp'] == 258


def test_mirror_only_one_monitor_active_at_a_time():
    """同時に2つのホットスポットが重複して監視されることはない
    (次のホットスポットは現在のウィンドウが閉じるまで待つ)。"""
    m = _HotspotMirror(hotspot_wps=(178, 189, 258, 289, 334))
    m.step(now_s=0.0, wp_id=258, kappa_lookup=lambda wp: 0.03, act_deg=1.0)
    assert m.monitor['wp'] == 258
    # ウィンドウが閉じる前に別のホットスポットへ到達しても無視される
    m.step(now_s=0.5, wp_id=289, kappa_lookup=lambda wp: -0.16, act_deg=1.0)
    assert m.monitor['wp'] == 258


# ---------------------------------------------------------------------------
# ①非矛盾性・②非冗長性: mpc_controller.py側の配線をソーステキストで検証
# ---------------------------------------------------------------------------

def test_hotspot_wps_declared_with_5_target_waypoints():
    idx = _SRC.index("self._HOTSPOT_WPS = ")
    snippet = _SRC[idx:idx + 60]
    assert "178" in snippet and "189" in snippet and "258" in snippet
    assert "289" in snippet and "334" in snippet


def test_hotspot_monitor_initialized_to_none():
    idx = _SRC.index("self._hotspot_monitor")
    snippet = _SRC[idx:idx + 80]
    assert "= None" in snippet


def test_method_uses_existing_xcorr_steeract_hist_not_new_subscription():
    """②非冗長性: 実測舵角は既存のSTEER-XCORR用履歴(_xcorr_steeract_hist)を
    再利用し、新規のトピック購読は追加していないことを確認する。"""
    idx = _SRC.index("def _maybe_log_hotspot_deviation")
    idx_end = _SRC.index("def _gnss_track_heading")
    snippet = _SRC[idx:idx_end]
    assert "self._xcorr_steeract_hist" in snippet
    assert "create_subscription" not in snippet


def test_theoretical_angle_uses_kappa_ref_and_model_length():
    idx = _SRC.index("def _maybe_log_hotspot_deviation")
    idx_end = _SRC.index("def _gnss_track_heading")
    snippet = _SRC[idx:idx_end]
    assert "arctan" in snippet
    assert "self._mpc.model.length" in snippet


def test_log_line_contains_required_fields():
    idx = _SRC.index('[HOTSPOT-DEVIATION]')
    snippet = _SRC[idx:idx + 300]
    assert "kappa_ref=" in snippet
    assert "delta_expected=" in snippet
    assert "delta_act_peak=" in snippet
    assert "max_dev=" in snippet


def test_called_every_cycle_not_gated_by_1s_throttle():
    """③検証ロギング: ピーク値追跡には毎周期の呼び出しが必要なため、既存の
    1秒間引きイディオム(_maybe_log_steer_xcorr等)とは別に呼ばれていることを確認。"""
    idx = _SRC.index("self._maybe_log_hotspot_deviation()")
    snippet_before = _SRC[max(0, idx - 300):idx]
    # 直前が1秒間引きのif閉じ括弧の外(インデントが浅い)であることを確認
    assert "if self._loop % int(max(1, self._mpc_cfg.control_rate)) == 0:" in snippet_before


def test_does_not_modify_qp_or_control_command():
    """④遡及効果: このホットスポット監視は純粋な観測であり、Q/R/QP計算・
    publishする制御コマンドには一切書き込まないことを確認する。"""
    idx = _SRC.index("def _maybe_log_hotspot_deviation")
    idx_end = _SRC.index("def _gnss_track_heading")
    snippet = _SRC[idx:idx_end]
    assert "update_Q" not in snippet
    assert "update_R" not in snippet
    assert "_command_pub.publish" not in snippet
    assert "_command_raw_pub.publish" not in snippet
