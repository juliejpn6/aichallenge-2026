"""Unit tests for 190-2節(2026-07-26): shuffle_episode_gap_sをpath=3の検知遅延に
合わせて6.0秒→15.0秒へ引き上げ。

背景: ローカル3台走行(run_dev3_20260726_171301)で、d2がpath=3(完全停止検知、
count=400≒10.0秒@40Hz)経由のSTUCKを同一地点で6回以上繰り返したが、毎回の
PUSH終了→次回検知までの間隔が実測10.2〜10.5秒と、旧既定値6.0秒を常に上回って
いたため、`_stuck_update_shuffle_cycle()`の同一エピソード判定
(`_gap_s <= shuffle_episode_gap_s`)が一度も成立せず、187節で実装した反転リトライ
(giveup_streak)が一度も発動しないまま同じ失敗を繰り返していた。

対処はconfig.yaml・mpc_controller.py双方のデフォルト値を15.0秒へ引き上げるのみ
(path=3検知の最低所要時間10.0秒に約50%の余裕を持たせた値)。同一エピソード判定の
ロジック自体・radius_mによる位置一致判定は無変更。

mpc_controller.pyはrclpy依存で直接importできないため、ソーステキスト検証で
実際の変更箇所を確認する(既存テストと同じ方針)。"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

_CFG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")
with open(_CFG_PATH) as _f:
    _CFG_SRC = _f.read()

CONTROL_RATE_HZ = 40.0
PATH3_DETECT_COUNT = 400


def test_path3_detect_time_exceeds_old_default_gap():
    """①非矛盾性: 変更の必要性そのものを数値で裏付ける。path=3の検知だけで
    最低10.0秒かかり、旧既定値6.0秒を上回っていたことを確認する。"""
    path3_detect_s = PATH3_DETECT_COUNT / CONTROL_RATE_HZ
    old_default_gap_s = 6.0
    assert path3_detect_s > old_default_gap_s


def test_new_default_gap_comfortably_covers_path3_detect_time():
    path3_detect_s = PATH3_DETECT_COUNT / CONTROL_RATE_HZ
    new_default_gap_s = 15.0
    assert new_default_gap_s > path3_detect_s
    # 実測ギャップ(10.2〜10.5秒)にも十分な余裕があること
    assert new_default_gap_s > 10.5


def test_code_default_updated_to_15():
    idx = _SRC.index('self._stuck_shuffle_episode_gap_s = float(_stkget("shuffle_episode_gap_s"')
    idx_end = idx + 100
    snippet = _SRC[idx:idx_end]
    assert "15.0" in snippet
    assert '"shuffle_episode_gap_s", 6.0' not in snippet


def test_config_yaml_updated_to_15():
    idx = _CFG_SRC.index("shuffle_episode_gap_s:")
    line = _CFG_SRC[idx:idx + 40]
    assert "15.0" in line


def test_episode_continuity_check_logic_untouched():
    """④遡及効果: 同一エピソード判定の条件式自体(gap_s以下 AND radius_m以内)は
    変更していない。radius_mの既定値(3.0)も無変更のまま。"""
    idx = _SRC.index("def _stuck_update_shuffle_cycle(")
    idx_end = _SRC.index("def _stuck_recovery_complete(")
    snippet = _SRC[idx:idx_end]
    assert "_gap_s <= self._stuck_shuffle_episode_gap_s" in snippet
    assert "_dist <= self._stuck_shuffle_episode_radius_m" in snippet
    idx2 = _SRC.index('self._stuck_shuffle_episode_radius_m = float(_stkget("shuffle_episode_radius_m"')
    assert '"shuffle_episode_radius_m", 3.0' in _SRC[idx2:idx2 + 100]


def test_giveup_streak_escalation_logic_itself_untouched():
    """④遡及効果: 187節の反転リトライ本体(giveup_streak判定式)は今回変更していない
    (このgap_s修正だけで、同一エピソードとして正しく認識されればそのまま機能する
    設計であることの確認)。"""
    idx = _SRC.index("self._stuck_giveup_streak += 1")
    idx_end = idx + 300
    snippet = _SRC[idx:idx_end]
    assert "if self._stuck_giveup_streak <= self._stuck_max_giveup_streak:" in snippet
