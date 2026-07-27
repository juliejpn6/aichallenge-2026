"""Regression tests for the line_cap(_offset_line_speed_cap) EMA smoothing (97節, 2026-07-17).

Background: 0717-02実測で、icc_f3とline_capのv_safe_srcが2区間・合計24回チャタリング
していることを発見した(97節監査)。原因は_offset_line_speed_cap自体の内部不安定性
(自車wp_id=離散インデックスが1つ進むだけで15m先読み窓全体がシフトし、コーナー頂点
付近の曲率スパイクが窓の出入りを起こす)であり、icc_f3との「縄張りの重複」ではない
ことを確認済み(96節のC1×icc_f3とは異なる根本原因)。

対処(ユーザー承認済み設計、案A): 既存のalong_lane_ema/v_corridor_emaと同じ考え方・
同じ時定数(_ot_ema_alpha)で、_offset_line_speed_capの出力値自体を平滑化する。

_offset_line_speed_cap自体はself._reference_path/self._wp_s_cum等の重い依存を持つ
ため、EMA部分のロジックのみを純Pythonでミラーして数式的性質(初回パススルー・
ステップ変化への漸近収束・Noneでのリセット)を検証し、mpc_controller.py側の配線
(リセット箇所)は構造的なソーステキスト検証で確認する。
"""
import os

import pytest

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

EMA_ALPHA = 0.05  # 既定の_ot_ema_alpha(config.yaml既定値と同一)


class LineCapEmaMirror:
    """_offset_line_speed_cap内のEMA部分のみを抽出したミラー。"""

    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.ema = None

    def update(self, cap):
        if cap is None:
            self.ema = None
            return None
        if self.ema is None:
            self.ema = cap
        else:
            self.ema += self.alpha * (cap - self.ema)
        return self.ema


def test_first_sample_passes_through_unsmoothed():
    """初回サンプルはEMA状態が無いため、生の値がそのまま採用される
    (over-cautiousな初期遅延を発生させない)。"""
    mirror = LineCapEmaMirror()
    assert mirror.update(3.0) == 3.0


def test_step_change_is_smoothed_not_instantaneous():
    """回帰の核心: 生の値が4.17→2.0のように急変しても、EMAは即座には
    追従せず緩やかに近づく(0717-02実測のようなチャタリングを吸収する)。"""
    mirror = LineCapEmaMirror()
    mirror.update(4.17)
    smoothed = mirror.update(2.0)
    assert 2.0 < smoothed < 4.17
    # alpha=0.05の1周期分だけ動く: 4.17 + 0.05*(2.0-4.17) ≈ 4.06
    assert smoothed == pytest.approx(4.17 + EMA_ALPHA * (2.0 - 4.17), abs=1e-6)


def test_repeated_chatter_converges_toward_the_lower_value_over_time():
    """0717-02実測のような、生の値が4.17と2.0付近を往復し続ける場面でも、
    EMAは往復に完全追従せず、相対的に安定した値へ収束していく
    (往復のたびに大きくジャンプしない)。"""
    mirror = LineCapEmaMirror()
    raw_sequence = [4.13, 3.06, 4.13, 2.67, 4.17, 2.09, 4.17, 2.04, 4.17, 1.90]
    smoothed_values = [mirror.update(v) for v in raw_sequence]
    # 生の値の振れ幅(4.17-1.90=2.27)に比べ、EMAの振れ幅は大幅に小さい。
    smoothed_range = max(smoothed_values) - min(smoothed_values)
    raw_range = max(raw_sequence) - min(raw_sequence)
    assert smoothed_range < raw_range * 0.2


def test_none_resets_ema_state():
    """cap=None(先読み対象なし/例外)の周期はEMA状態自体をリセットする。
    次に有効な値が来た際、古い平滑化値を引きずらず初回パススルーとなる。"""
    mirror = LineCapEmaMirror()
    mirror.update(2.0)
    assert mirror.update(None) is None
    assert mirror.ema is None
    assert mirror.update(5.0) == 5.0  # リセット後の初回は生の値そのまま


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証(ソーステキスト検証)
# ---------------------------------------------------------------------------

def test_line_cap_ema_state_initialized():
    assert "self._line_cap_ema = None" in _SRC


def test_offset_line_speed_cap_resets_ema_on_exception_path():
    idx = _SRC.index("def _offset_line_speed_cap")
    snippet = _SRC[idx:idx + 1400]
    assert "except Exception:" in snippet
    assert "self._line_cap_ema = None\n            return None" in snippet


def test_offset_line_speed_cap_resets_ema_when_cap_is_none():
    idx = _SRC.index("def _offset_line_speed_cap")
    snippet = _SRC[idx:idx + 2400]
    assert "if cap is None:\n            self._line_cap_ema = None\n            return None" in snippet


def test_offset_line_speed_cap_applies_ema_and_returns_smoothed_value():
    idx = _SRC.index("def _offset_line_speed_cap")
    snippet = _SRC[idx:idx + 2700]
    assert "self._line_cap_ema += self._ot_ema_alpha * (cap - self._line_cap_ema)" in snippet
    assert "return self._line_cap_ema" in snippet


def test_line_cap_ema_reset_on_new_engage():
    """回帰: 新規エンゲージ毎にline_cap_emaも仕切り直される(前回のオーバーテイクの
    平滑化値を持ち越さない、93/94節で確立した原則の踏襲)。"""
    idx = _SRC.index('self._ot_state = "OVERTAKING"\n                        self._ot_giveup_count = 0')
    # 2026-07-24追加(168節): room_exhausted状態のリセット代入2行が
    #   間に挿入されたため、窓を500→750へ拡大(検証対象そのものは無変更)。
    snippet = _SRC[idx:idx + 750]
    assert "self._line_cap_ema = None" in snippet
