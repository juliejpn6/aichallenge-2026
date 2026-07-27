"""Unit tests for the steering actuator-delay diagnostic ([STEER-XCORR], 192節続報,
2026-07-27)。

背景: AXIS06(先頭走行時の約1.3秒周期リミットサイクル、185/186節)の根本原因候補である
アクチュエータ遅延(200ms)を実測で特定するには、指令操舵角と実測操舵角(/vehicle/status/
steering_status)の時系列比較が必要。予選環境のbagレコーダー(aichallenge/utils/
record_rosbag.bash)にsteering_statusを追加する対処は192節で一度実装したが、この
スクリプトは公開リポジトリ(aichallenge/utils/)側に属し、予選環境のDockerイメージは
このリポジトリをGitHubから直接git cloneしてビルドするため、ユーザーのpush権限が無い
公開リポジトリでは反映できないと判明した(deployment gap、design_docs 196節)。

対処: 自分たちの提出物(mpc_controller.py)側から直接steering_statusを購読し、既存の
[GNSS-EKF-XCORR](118節続報)と全く同じ相互相関パターンを操舵角1軸へ適用して、
既存のautoware.logテキストログへ[STEER-XCORR]として記録する。bagレコーダーを一切
経由しないため、公開リポジトリの変更なしに予選環境でも実測できる。

符号規約: lag_ms>0 は actual(t) ≈ cmd(t-lag) を意味し、実測操舵角が指令より遅延して
いることを示す(実アクチュエータが指令に追従する向きなので通常は正の値が出るはず)。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。相互相関探索
アルゴリズム自体は単純な数式(np.interpによるリサンプル+RMS残差の最小化)のため、
同一のロジックを複製したPythonミラーで、既知の遅延を注入した合成信号から正しく
復元できることを数式的に実証する。mpc_controller.py側の配線(新規購読・publish時の
履歴記録・呼び出し箇所)は構造的なソーステキスト検証で確認する。
"""
import os
import math
import numpy as np

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# 1) 純Pythonミラー: ラグ探索アルゴリズム(1軸・非対称探索範囲)
# ---------------------------------------------------------------------------

def _xcorr_best_lag(cmd_t, cmd_v, act_t, act_v, lag_lo=-0.05, lag_hi=0.4, lag_step=0.01):
    """mpc_controller.py の _maybe_log_steer_xcorr() 内のラグ探索の複製ミラー。"""
    cmd_t = np.asarray(cmd_t); cmd_v = np.asarray(cmd_v)
    act_t = np.asarray(act_t); act_v = np.asarray(act_v)
    t_lo = cmd_t[0] + lag_hi
    t_hi = cmd_t[-1] - max(lag_lo, 0.0)
    mask = (act_t >= t_lo) & (act_t <= t_hi)
    act_t_m, act_v_m = act_t[mask], act_v[mask]

    lags = np.arange(lag_lo, lag_hi + 1e-9, lag_step)
    best_lag, best_resid, resid_at_zero = 0.0, None, None
    for lag in lags:
        cmd_i = np.interp(act_t_m - lag, cmd_t, cmd_v)
        resid = float(np.sqrt(np.mean((act_v_m - cmd_i) ** 2)))
        if abs(lag) < 1e-9:
            resid_at_zero = resid
        if best_resid is None or resid < best_resid:
            best_resid, best_lag = resid, float(lag)
    return best_lag, best_resid, resid_at_zero


def _make_synthetic_series(true_lag_s, n=300, dt=0.025, noise=0.0, seed=0):
    """cmd(t)=sin波(操舵の往復)、actual(t)=cmd(t - true_lag)(実アクチュエータが
    true_lag秒だけ遅延して追従)を合成する。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    cmd_v = 15.0 * np.sin(0.7 * t)  # [deg]、往復操舵を模擬
    act_t = t
    act_v = 15.0 * np.sin(0.7 * (t - true_lag_s)) + rng.normal(0, noise, n)
    return t, cmd_v, act_t, act_v


def test_recovers_known_actuator_delay():
    """核心: 実測操舵角が指令に対して0.2s(既知のアクチュエータ遅延相当)遅延している
    合成信号から、正のlag(遅延)がおおよそ復元できることを確認する。"""
    cmd_t, cmd_v, act_t, act_v = _make_synthetic_series(true_lag_s=0.2)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(cmd_t, cmd_v, act_t, act_v)
    assert abs(best_lag - 0.2) <= 0.01 + 1e-9  # 探索刻み(0.01s)以内で一致
    assert best_resid < resid_zero  # 真のラグでの残差はゼロラグ仮定より必ず小さい


def test_recovers_small_delay():
    """核心: 遅延が小さい(0.05s)場合でも正しく復元できることを確認する。"""
    cmd_t, cmd_v, act_t, act_v = _make_synthetic_series(true_lag_s=0.05)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(cmd_t, cmd_v, act_t, act_v)
    assert abs(best_lag - 0.05) <= 0.01 + 1e-9
    assert best_resid < resid_zero


def test_zero_lag_when_series_identical():
    """回帰: 遅延が無い(actual=cmd)場合、最良ラグは0付近で、resid_at_bestと
    resid_at_zeroがほぼ一致する。"""
    cmd_t, cmd_v, act_t, act_v = _make_synthetic_series(true_lag_s=0.0)
    best_lag, best_resid, resid_zero = _xcorr_best_lag(cmd_t, cmd_v, act_t, act_v)
    assert abs(best_lag) <= 0.01 + 1e-9
    assert math.isclose(best_resid, resid_zero, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 2) mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

def test_xcorr_buffers_declared():
    """非矛盾性: 相互相関用の時刻付きバッファが宣言されていることを確認する。"""
    assert "self._xcorr_steercmd_hist: List[Tuple[float, float]] = []" in _SRC
    assert "self._xcorr_steeract_hist: List[Tuple[float, float]] = []" in _SRC


def test_steering_status_subscription_wired():
    """核心: /vehicle/status/steering_statusへの新規subscriptionが配線されていること
    を確認する(公開リポジトリのbagレコーダーに依存しない独立経路)。"""
    assert 'SteeringReport, "/vehicle/status/steering_status", self._steering_status_callback, 1)' in _SRC


def test_steering_status_callback_appends_history():
    idx = _SRC.index("def _steering_status_callback(")
    snippet = _SRC[idx:idx + 500]
    assert "self._xcorr_steeract_hist.append((t, float(msg.steering_tire_angle)))" in snippet
    assert "_XCORR_WINDOW_S" in snippet


def test_publish_control_command_appends_cmd_history_after_gain():
    """核心: 履歴に記録する指令値は、gain適用前のraw値ではなく、実際にアクチュエータへ
    渡るgain適用後の値(cmd_gained.lateral.steering_tire_angle *= gain の後)であることを
    確認する。gain適用前を記録すると実測とのスケールが不一致になり遅延推定が崩れる。
    2026-07-27追加(196節続報、遅延再現実験のためcmd_gainedを別オブジェクト化): 即時経路
    (_extra_delay_s<=0.0)ではcmd_gainedがそのままself._command_pub.publish()へ渡ることを
    確認する(遅延経路では_delayed_cmd_queue経由になるため、この即時経路が既定の挙動)。"""
    idx_gain = _SRC.index("cmd_gained.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var")
    idx_append = _SRC.index("self._xcorr_steercmd_hist.append(")
    assert idx_gain < idx_append
    snippet = _SRC[idx_gain:idx_append + 900]
    assert "self._command_pub.publish(cmd_gained)" in snippet  # 既定(即時)経路でgain適用後の実発行と同じcmd_gainedを参照


def test_steer_xcorr_evaluated_alongside_gnss_ekf_xcorr():
    """非冗長性: 既存の[GNSS-EKF-XCORR]と同じ間引き・呼び出し箇所を再利用しており、
    新しい間引き定数を導入していないことを確認する。"""
    idx_gnss = _SRC.index("self._maybe_log_gnss_ekf_xcorr()")
    idx_steer = _SRC.index("self._maybe_log_steer_xcorr()")
    assert idx_steer > idx_gnss
    assert idx_steer - idx_gnss < 100  # 直後の行に配置
    snippet = _SRC[max(0, idx_gnss - 300):idx_gnss]
    assert "int(max(1, self._mpc_cfg.control_rate))" in snippet


def test_steer_xcorr_log_has_expected_fields():
    idx = _SRC.index('"[STEER-XCORR]')
    snippet = _SRC[idx:idx + 250]
    assert "lag_ms=" in snippet
    assert "resid_at_best_deg=" in snippet
    assert "resid_at_zero_deg=" in snippet
    assert "n_cmd=" in snippet
    assert "n_act=" in snippet


def test_steer_xcorr_method_guards_against_insufficient_data():
    """回帰: バッファが十分に溜まっていない(起動直後等)場合、例外を投げず
    早期returnすることを確認する(存在確認のみ、実行はrclpy依存のため不可)。"""
    idx = _SRC.index("def _maybe_log_steer_xcorr(self)")
    snippet = _SRC[idx:idx + 1500]
    assert "if len(cmd_h) < 20 or len(act_h) < 20:" in snippet
    assert "return" in snippet


def test_publish_control_command_uses_rclpy_time_nanoseconds_not_msg_sec():
    """回帰(2026-07-27緊急修正): _publish_control_command()の`stamp`引数は
    rclpy.time.Time型(self.get_clock().now()、_create_ackerman_control_command内の
    stamp.to_msg()と同じ型)であり、builtin_interfaces/Timeメッセージとは異なり
    .sec/.nanosec属性を持たない。初回実装時にこれを誤って`stamp.sec + stamp.nanosec`
    としてしまい、ノード起動直後の最初の発行サイクルでAttributeErrorが発生して
    プロセスが即死し、車両が全く発進しない致命的な回帰を引き起こしていた
    (ローカル3台走行run_dev3_20260727_004123・単独走行の両方で再現)。
    正しくは`stamp.nanoseconds`(ROS epochからのナノ秒、int)を使う。"""
    idx = _SRC.index("self._xcorr_steercmd_hist.append(")
    snippet = _SRC[max(0, idx - 400):idx]
    assert "_t_cmd = stamp.nanoseconds * 1e-9" in snippet
    assert "stamp.sec" not in snippet


# ---------------------------------------------------------------------------
# 3) 遅延再現実験(debug_extra_actuator_delay_s、196節続報)の配線検証
# ---------------------------------------------------------------------------

def test_debug_extra_actuator_delay_defaults_to_zero():
    """安全性: 競技提出時に誤って有効化されたままにならないよう、既定値は0.0
    (完全に無効・従来と同一の即時publish経路)であることを確認する。"""
    assert "debug_extra_actuator_delay_s: float = 0.0" in _SRC
    assert "debug_extra_actuator_delay_s: 0.0" in open(
        os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")).read()


def test_immediate_path_taken_when_delay_is_zero_or_less():
    """核心: _extra_delay_s<=0.0の分岐が即時publish(従来と同一)であることを確認する。"""
    idx = _SRC.index("_extra_delay_s = float(getattr(")
    snippet = _SRC[idx:idx + 500]
    assert "if _extra_delay_s <= 0.0:" in snippet
    assert "self._command_raw_pub.publish(cmd)" in snippet
    assert "self._command_pub.publish(cmd_gained)" in snippet


def test_delayed_path_uses_fifo_queue_and_drains_due_items():
    """核心: 遅延経路は_delayed_cmd_queueへ(due_t, raw_cmd, gained_cmd)をFIFOで積み、
    到達時刻を過ぎたものだけをpop(0)で順に発行することを確認する
    (途中で順序が入れ替わらない、非同期な即時発行にはならない)。"""
    idx = _SRC.index("else:\n            _due = _t_cmd + _extra_delay_s")
    snippet = _SRC[idx:idx + 500]
    assert "self._delayed_cmd_queue.append((_due, cmd, cmd_gained))" in snippet
    assert "while self._delayed_cmd_queue and self._delayed_cmd_queue[0][0] <= _now_s:" in snippet
    assert "self._delayed_cmd_queue.pop(0)" in snippet


def test_delayed_cmd_queue_declared_empty_by_default():
    assert "self._delayed_cmd_queue: List[Tuple[float, object, object]] = []" in _SRC


def test_steer_xcorr_uses_list_snapshot_to_avoid_race_condition():
    """非矛盾性: 190節で発見・修正したGNSS-EKF-XCORRの競合状態クラッシュ
    (購読コールバック側でappend/pop(0)により毎周期変化する可変リストへの
    エイリアスから複数の内包表記を構築し配列長が食い違う)と同じパターンを
    最初から回避していることを確認する(list()で1回だけ独立コピーを取る)。"""
    idx = _SRC.index("def _maybe_log_steer_xcorr(self)")
    snippet = _SRC[idx:idx + 800]
    assert "cmd_h = list(self._xcorr_steercmd_hist)" in snippet
    assert "act_h = list(self._xcorr_steeract_hist)" in snippet
