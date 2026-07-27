"""Regression guard for the 94節 token-consistency audit (2026-07-17).

Background: 93節でLateralTTCMonitorのcritical_curvature_runが側反転時にリセット
されていない矛盾を修正した際、ユーザー指示で「対象車IDが変わったらトレンド
追跡状態を仕切り直す」という設計原則をソースコード全体に照らして再監査した。
_scan_traffic()の fwd_vid/along_vid はいずれも毎周期「その時点で最も近い車」を
選び直す実装(対象車IDに固定されない)であるにも関わらず、以下3箇所が対象車
IDの変化を検知せずにトレンド/デバウンスカウンタを積算し続けていた(既に正しい
参照実装である_plan_room_ok_count/_room_debounce_ok、mpc_controller.py:1850-1858、
の"if vid != prev_vid or side != prev_side: reset"パターンとの不整合):
  - _ot_worth_count(エンゲージ判定のデバウンス)
  - _ot_giveup_count(OVERTAKING中の断念デバウンス)
  - _along_lane_ema(並走レーン幅の平滑化。along_vid自体が存在しなかった)

mpc_controller.py(rclpy依存のため直接importできない)に対する構造的なソース
テキスト検証で、3箇所全てに同一パターンの対象車ID変化検知が追加されている
ことを確認する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_reference_pattern_plan_room_ok_count_unchanged():
    """前提確認: 正しい参照実装(_plan_room_ok_count)が変更されていないこと。"""
    idx = _SRC.index("if vid != self._plan_room_prev_vid or side != self._plan_room_prev_side:")
    snippet = _SRC[idx:idx + 150]
    assert "self._plan_room_ok_count = 0" in snippet


def test_ot_worth_count_resets_on_fwd_vid_change():
    """①_ot_worth_countが、_pass_worth判定の前にfwd_vidの変化を検知して
    リセットしていることを確認する。"""
    idx = _SRC.index("self._ot_worth_count = self._ot_worth_count + 1 if pass_worth else 0")
    snippet = _SRC[max(0, idx - 400):idx]
    # 2026-07-22修正(00節監査): scan.get("fwd_vid")からopp_sit.fwd_vid経由へ変更(値は同一)。
    assert "_fwd_vid_worth = opp_sit.fwd_vid" in snippet
    assert "if _fwd_vid_worth != self._ot_worth_prev_vid:" in snippet
    assert "self._ot_worth_count = 0" in snippet
    assert "self._ot_worth_prev_vid = _fwd_vid_worth" in snippet


def test_ot_giveup_count_resets_on_fwd_vid_change():
    """②_ot_giveup_countが、closing判定の前にfwd_vidの変化を検知して
    リセットしていることを確認する。"""
    # 2026-07-22修正(00節監査): _scan.get("fwd_vid")からopp_sit経由へ変更(値は同一)。
    idx = _SRC.index("_fwd_vid_giveup = _opp_sit.fwd_vid")
    snippet = _SRC[idx:idx + 600]
    assert "if _fwd_vid_giveup != self._ot_giveup_prev_vid:" in snippet
    assert "self._ot_giveup_count = 0" in snippet
    assert "self._ot_giveup_prev_vid = _fwd_vid_giveup" in snippet
    # このリセットが実際のclosing判定(self._ot_giveup_count += 1)より前に
    # 位置していることを確認する(順序が逆だと今周期の判定に間に合わない)。
    assert snippet.index("self._ot_giveup_prev_vid = _fwd_vid_giveup") < snippet.index(
        "self._ot_giveup_count += 1")


def test_scan_traffic_tracks_along_vid():
    """③_scan_traffic()がalong車選択時にalong_vidを記録し、出力dictの
    初期値にも含まれていることを確認する(修正前は対象車ID自体が
    存在しなかった)。"""
    assert '"along_vid": None' in _SRC
    idx = _SRC.index('out["along_lat"] = lat')
    snippet = _SRC[idx:idx + 200]
    assert 'out["along_vid"] = vid' in snippet


def test_along_lane_ema_resets_on_along_vid_change():
    """③_along_lane_emaが、along_vidの変化を検知してリセットしている
    ことを確認する。非適用状態への遷移時(else節)でも
    _along_lane_prev_vidが_along_lane_emaと共にリセットされる。"""
    idx = _SRC.index("_along_vid_now = _scan.get(\"along_vid\")")
    snippet = _SRC[idx:idx + 300]
    assert "_along_vid_now != self._along_lane_prev_vid" in snippet
    assert "self._along_lane_ema = None" in snippet
    assert "self._along_lane_prev_vid = _along_vid_now" in snippet

    idx2 = _SRC.index("# 非適用状態への遷移時はリセット")
    snippet2 = _SRC[idx2:idx2 + 150]
    assert "self._along_lane_prev_vid = None" in snippet2


def test_all_three_prev_vid_trackers_initialized():
    """回帰: 3箇所の対象車IDトラッカーが全て__init__で初期化されている
    ことを確認する(初期化漏れがあるとAttributeErrorで即座に落ちる)。"""
    assert "self._ot_worth_prev_vid = None" in _SRC
    assert "self._ot_giveup_prev_vid = None" in _SRC
    assert "self._along_lane_prev_vid = None" in _SRC
