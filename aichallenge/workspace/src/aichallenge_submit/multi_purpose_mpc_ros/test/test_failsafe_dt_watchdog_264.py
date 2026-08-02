"""264節続報(2026-08-02、Task1): dt異常フェイルセーフ(watchdog)の単体テスト。

背景: 予選ログ(2026-08-01/02)から、約1秒間・全ノードが同時に完全停止する
ストールを2回発見した(263/266節、[PERF-DT]のp999/max=764.37ms)。この種の
ストールから復帰した直後、MPCは最大1秒弱前の古い状態量に基づいて算出した
指令をそのまま発行してしまう——これは「ワインドアップ」に相当し、レース中
なら致命的なリスクになりうる。本フェイルセーフは、このサイクルの壁時計dt
(既存の[PERF-DT]計装が使う_wall_dt、263節でsteal時間等の外部要因と
相関づけた計測基盤をそのまま再利用)が閾値を超えていた場合のみ、実発行
コマンドを安全側(操舵=直前発行値を保持、速度=0・加速度=a_minでブレーキ)
へ上書きする。MPC本体の計算・内部状態には一切触れない(このサイクルの
発行値のみを上書きし、次サイクル以降は通常通りMPCが実際の現在状態から
再計画する)。

mpc_controller.pyはrclpy依存のため直接importできない。ロジック自体は
純粋な条件分岐のためミラー関数で検証し、mpc_controller.py側の実装
(呼び出し配線・ログタグ・無効化規約)はソーステキスト構造検証で確認する
(既存のtest_dt_spike_and_steal_263.py等と同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①非矛盾性: 発火条件・オーバーライド値算出のミラー検証
# ---------------------------------------------------------------------------

def mirror_failsafe_decision(wall_dt_s, threshold_s, last_steer, a_min):
    """_publish_control_command()冒頭のフェイルセーフ判定+オーバーライド値
    算出部分のミラー。発火する場合は(u, acc)を、しない場合はNoneを返す。"""
    if threshold_s <= 0.0:
        return None
    if wall_dt_s <= threshold_s:
        return None
    return ([0.0, last_steer], a_min)


def test_failsafe_fires_above_threshold():
    result = mirror_failsafe_decision(
        wall_dt_s=0.300, threshold_s=0.200, last_steer=0.15, a_min=-1.37)
    assert result is not None
    u, acc = result
    assert u == [0.0, 0.15]
    assert acc == -1.37


def test_failsafe_does_not_fire_at_or_below_threshold():
    assert mirror_failsafe_decision(0.200, 0.200, 0.15, -1.37) is None
    assert mirror_failsafe_decision(0.050, 0.200, 0.15, -1.37) is None


def test_failsafe_fires_for_observed_764ms_stall():
    """実際に発見した764msストール(既定閾値200ms)は確実に発火する。"""
    result = mirror_failsafe_decision(0.764, 0.200, 0.0, -1.37)
    assert result is not None


def test_failsafe_disabled_when_threshold_non_positive():
    assert mirror_failsafe_decision(10.0, 0.0, 0.15, -1.37) is None
    assert mirror_failsafe_decision(10.0, -1.0, 0.15, -1.37) is None


def test_failsafe_holds_previous_steering_not_zero():
    """操舵は直前値を保持する(0.0への急変ではない)ことの確認。"""
    result = mirror_failsafe_decision(0.500, 0.200, last_steer=-0.42, a_min=-1.37)
    _, steer = result[0]
    assert steer == -0.42


def test_failsafe_commands_zero_speed_and_a_min_accel():
    result = mirror_failsafe_decision(0.500, 0.200, 0.0, a_min=-1.37)
    u, acc = result
    assert u[0] == 0.0
    assert acc == -1.37


# ---------------------------------------------------------------------------
# ④遡及効果: mpc_controller.py側の実装配線をソーステキストで確認
# ---------------------------------------------------------------------------

def test_control_stores_wall_dt_as_instance_attribute():
    idx = _SRC.index("def _control(self):")
    idx_end = _SRC.index("self._last_t = now", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._last_wall_dt_s = _wall_dt" in snippet


def test_publish_control_command_reads_threshold_from_config_with_default():
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("self._last_u[0] = float(u[0])", idx)
    snippet = _SRC[idx:idx_end]
    assert 'getattr(self._mpc_cfg, "failsafe_dt_threshold_ms", 200.0)' in snippet


def test_publish_control_command_disabled_when_threshold_non_positive():
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("self._last_u[0] = float(u[0])", idx)
    snippet = _SRC[idx:idx_end]
    assert "_failsafe_threshold_s > 0.0 and _wall_dt_s > _failsafe_threshold_s" in snippet


def test_publish_control_command_reads_wall_dt_defensively():
    """self._last_wall_dt_sが未設定(初回呼び出し等)でもクラッシュしない
    よう、getattrでデフォルト値付きで読むこと。"""
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("self._last_u[0] = float(u[0])", idx)
    snippet = _SRC[idx:idx_end]
    assert "getattr(self, '_last_wall_dt_s', 0.0)" in snippet


def test_publish_control_command_overrides_u_and_acc_on_failsafe():
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("self._last_u[0] = float(u[0])", idx)
    snippet = _SRC[idx:idx_end]
    assert "u = [0.0, self._last_u[1]]" in snippet
    assert "acc = self._mpc_cfg.a_min" in snippet


def test_publish_control_command_logs_failsafe_tag():
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("self._last_u[0] = float(u[0])", idx)
    snippet = _SRC[idx:idx_end]
    assert "[FAILSAFE]" in snippet
    assert "self.get_logger().warn(" in snippet


def test_failsafe_override_happens_before_last_u_bookkeeping():
    """オーバーライドされたu/accが、その後の self._last_u/_last_acc 更新
    (「実際に発行した値」を記録する既存の唯一の場所、issue④③参照)より前に
    確定していること——そうでないと_last_uがフェイルセーフ発行値と食い違う。"""
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("_XCORR_WINDOW_S = 3.5", idx)
    snippet = _SRC[idx:idx_end]
    failsafe_idx = snippet.index("u = [0.0, self._last_u[1]]")
    bookkeeping_idx = snippet.index("self._last_u[0] = float(u[0])")
    assert failsafe_idx < bookkeeping_idx


def test_failsafe_does_not_touch_mpc_solve_or_internal_state():
    """フェイルセーフの発火判定・オーバーライドが_publish_control_command
    内(発行の直前)に閉じており、MPC本体(_control()の大部分)には一切
    触れていないことの構造確認。次サイクル以降は通常通りMPCが再計画する
    設計であることの裏付け。"""
    idx = _SRC.index("def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):")
    idx_end = _SRC.index("_XCORR_WINDOW_S = 3.5", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._mpc.solve" not in snippet
    assert "osqp" not in snippet.lower()


def test_config_default_registered():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg_src = f.read()
    assert "failsafe_dt_threshold_ms: 200.0" in cfg_src
