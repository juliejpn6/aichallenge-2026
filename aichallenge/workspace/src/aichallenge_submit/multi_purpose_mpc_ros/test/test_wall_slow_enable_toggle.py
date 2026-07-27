"""Unit tests for the wall_slow_enable diagnostic toggle (2026-07-23、166節続報)。

背景: wp257(コース最急コーナー直後)でユーザーが体感した揺れの原因切り分け実験として、
wall_slow(壁近接減速、現在wp1点のみ評価・先読み無し、79節で確定済みの設計)を実験的に
完全無効化できるスイッチを追加した。実測(0722系ログ)で5周中4周、wp250-252(コース
最急コーナー kappa=0.245)にてwall_slowが発動し、margin=0.01〜0.09mというほぼ壁に
接触寸前の値で介入していたことを確認済み(加速指令が+1.37/-1.37を往復するバンバン
挙動を伴う)。既定値はtrue(従来通り有効)で、恒久的な無効化ではなく実験専用。

mpc_controller.pyはrclpy依存のため直接importできないため、既存テストと同じ方針
(ソーステキストによる構造的検証)を用いる。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_wall_slow_enable_declared_default_true():
    idx = _SRC.index('self._wall_slow_enable = bool(_otget("wall_slow_enable", True))')
    assert idx > 0


def test_wall_slow_block_gated_by_enable_flag():
    """wall_slowの判定ブロック(_corr_ub0/_corr_lb0を使ったmargin計算)自体が
    self._wall_slow_enableでガードされていることを確認する(無効化時はマージン計算
    もwall_slow候補への追加も一切行われない、従来の判定条件・数式は無変更)。"""
    idx = _SRC.index("_corr_ub0 = self._mpc.dbg_corr_ub0")
    snippet = _SRC[max(0, idx - 200):idx + 1000]
    assert "if self._wall_slow_enable and np.isfinite(_corr_ub0)" in snippet
    # 既存のテーパー計算式(124節)は無変更であることも確認
    assert "_wall_cap = self._wall_slow_speed + _frac * (_umax - self._wall_slow_speed)" in snippet


def test_wall_slow_enable_key_present_in_yaml():
    """config.yamlの値そのもの(true/false)はユーザーが実験中に切り替える可変状態のため
    固定値をアサートしない。キー自体が存在し、Pythonコード側のデフォルト(True、上記
    テストで確認済み)へフォールバックする`_otget`経由で読まれることのみ確認する。"""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_src = f.read()
    assert "wall_slow_enable:" in cfg_src
