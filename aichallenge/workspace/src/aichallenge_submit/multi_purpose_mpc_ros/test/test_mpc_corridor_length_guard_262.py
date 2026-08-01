"""262節続報(2026-08-02、判定基準改訂+work_cpu計装Part A): MPC.py初期化クラッシュ防御。

背景: C4実験中(72Hz調査、単一車両、約10回の実行中1回のみ)に
`core/MPC.py`の`_stage_data()`で`xmin_dyn[nx::nx] = lb`が
`ValueError: could not broadcast input array from shape (20,) into shape (1,)`
でクラッシュし、ノードが落ちた(ROS2が自動再起動して復帰)。

コード解析による原因の特定(高確度の仮説): `get_control()`は`N`を
「非循環経路なら残りwaypoint数」に基づいて確定させるが、直後に呼ばれる
`_stage_data()`冒頭の`self.model.wp_id += self.wp_id_offset`(制御遅れ補償の
先読み)がこの**N確定より後**に効くため、非循環経路(ピット等)の終端付近では
`wp_id + N`が`n_waypoints`を超えうる。この場合`_corridor()`が返す`lb`/`ub`の
長さが呼び出し時の`N`と一致しないことがある(`update_path_constraints`内部の
全分岐までは完全にはトレースし切れていないため「仮説」と明記)。

起動シーケンス(wp_id_offset加算タイミング)自体の修正は本ガードのスコープ外
(起動シーケンスに触れる変更は別途慎重に行う)とし、`_stage_data()`に長さ不一致
検出時の安全側整形(`_resize_to_length`)を追加してクラッシュを防いだ。
通常時(長さ一致、全周期の大半)はこの分岐に入らず既存の数値・挙動に一切影響
しない——これはコリドー等価性回帰1514件+回帰スイート全体のPASSで別途確認する。

MPC.pyはrclpy非依存(numpy/scipy/osqpのみ)のため直接importでき、
`_stage_data()`をダックタイピングした最小限のfake self経由で実際に
呼び出して検証する(mpc_controller.pyのようなソーステキスト構造検証ではなく、
実際にガードを動作させる振る舞いテスト)。
"""
import types

import numpy as np

from multi_purpose_mpc_ros.core.MPC import MPC, _resize_to_length


def _make_fake_mpc(nx=3, nu=2, wp_id=998, e_y=0.1, use_obstacle_avoidance=True,
                    use_path_constraints_topic=False, lateral_blend=0.0,
                    lateral_target=0.0, lateral_psi_bias=0.0):
    fake = types.SimpleNamespace()
    fake.nx = nx
    fake.nu = nu
    fake.state_constraints = {
        'xmin': np.full(nx, -5.0), 'xmax': np.full(nx, 5.0)}
    fake.input_constraints = {
        'umin': np.array([0.0, -1.0]), 'umax': np.array([10.0, 1.0])}
    fake.model = types.SimpleNamespace()
    fake.model.wp_id = wp_id
    fake.model.spatial_state = types.SimpleNamespace(e_y=e_y)
    fake.lateral_blend = lateral_blend
    fake.lateral_target = lateral_target
    fake.lateral_psi_bias = lateral_psi_bias
    fake.use_obstacle_avoidance = use_obstacle_avoidance
    fake.use_path_constraints_topic = use_path_constraints_topic
    return fake


# ---------------------------------------------------------------------------
# ①非矛盾性: _resize_to_length()そのものの正しさ
# ---------------------------------------------------------------------------

def test_resize_no_op_when_length_already_matches():
    arr = np.array([1.0, 2.0, 3.0])
    out = _resize_to_length(arr, 3)
    assert np.array_equal(out, arr)


def test_resize_truncates_when_too_long():
    arr = np.arange(20, dtype=float)
    out = _resize_to_length(arr, 5)
    assert len(out) == 5
    assert np.array_equal(out, arr[:5])


def test_resize_pads_with_last_value_when_too_short():
    arr = np.array([1.0, 2.0, 3.0])
    out = _resize_to_length(arr, 6)
    assert len(out) == 6
    assert np.array_equal(out[:3], arr)
    assert np.all(out[3:] == 3.0)


def test_resize_zero_fills_when_empty():
    out = _resize_to_length(np.array([]), 4)
    assert len(out) == 4
    assert np.all(out == 0.0)


def test_resize_noop_when_n_is_zero():
    arr = np.array([1.0, 2.0])
    out = _resize_to_length(arr, 0)
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# ④遡及効果: _stage_data()内のガードを実際に動作させて検証
# ---------------------------------------------------------------------------

def test_stage_data_survives_length_mismatch_reuse_path():
    """262節続報で実際に観測されたクラッシュ(N=1・lb長20)を再現し、
    例外を投げずにxmin_dyn/xmax_dynが正しい長さで返ることを確認する。"""
    fake = _make_fake_mpc(nx=3, wp_id=998)
    reuse = {'ub0': np.full(20, 2.0), 'lb0': np.full(20, -2.0), 'margin0': 0.3}
    d = MPC._stage_data(fake, N=1, safety_margin=0.2, reuse=reuse)
    assert d['xmin_dyn'].shape == (fake.nx * 2,)
    assert d['xmax_dyn'].shape == (fake.nx * 2,)
    assert d['xr'].shape == (fake.nx * 2,)


def test_stage_data_clamped_corridor_uses_first_element_when_too_long():
    """長すぎるlb/ubは先頭(直近waypoint)を優先して使うことを確認する。"""
    fake = _make_fake_mpc(nx=3, wp_id=998)
    reuse = {'ub0': np.array([9.0] + [2.0] * 19), 'lb0': np.array([-9.0] + [-2.0] * 19),
              'margin0': 0.3}
    d = MPC._stage_data(fake, N=1, safety_margin=0.3, reuse=reuse)
    # N=1なのでxmin_dyn/xmax_dynはインデックスnxのみがコリドー値(先頭要素=9.0/-9.0系)
    assert d['xmax_dyn'][fake.nx] == 9.0
    assert d['xmin_dyn'][fake.nx] == -9.0


def test_stage_data_no_op_when_lengths_already_match():
    """長さが一致する通常ケースでは、ガードが値を一切変更しないことを確認する
    (既存の数値・挙動への無影響)。"""
    fake = _make_fake_mpc(nx=3, wp_id=100)
    reuse = {'ub0': np.array([1.5, 1.6, 1.7]), 'lb0': np.array([-1.5, -1.6, -1.7]),
              'margin0': 0.3}
    d = MPC._stage_data(fake, N=3, safety_margin=0.3, reuse=reuse)
    assert np.allclose(d['xmax_dyn'][fake.nx::fake.nx], [1.5, 1.6, 1.7])
    assert np.allclose(d['xmin_dyn'][fake.nx::fake.nx], [-1.5, -1.6, -1.7])


def test_stage_data_prints_guard_warning_on_mismatch(capsys):
    fake = _make_fake_mpc(nx=3, wp_id=998)
    reuse = {'ub0': np.full(20, 2.0), 'lb0': np.full(20, -2.0), 'margin0': 0.3}
    MPC._stage_data(fake, N=1, safety_margin=0.2, reuse=reuse)
    captured = capsys.readouterr()
    assert '[MPC-GUARD]' in captured.out
    assert 'N=1' in captured.out
    assert 'wp_id=998' in captured.out


def test_stage_data_silent_when_lengths_match(capsys):
    fake = _make_fake_mpc(nx=3, wp_id=100)
    reuse = {'ub0': np.array([1.5, 1.6, 1.7]), 'lb0': np.array([-1.5, -1.6, -1.7]),
              'margin0': 0.3}
    MPC._stage_data(fake, N=3, safety_margin=0.3, reuse=reuse)
    captured = capsys.readouterr()
    assert '[MPC-GUARD]' not in captured.out


# ---------------------------------------------------------------------------
# 遡及効果: ソーステキスト構造検証(ガードの配置・ロジック確認)
# ---------------------------------------------------------------------------

def test_guard_placed_before_xmin_dyn_assignment():
    import os
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "core", "MPC.py")
    with open(src_path) as f:
        src = f.read()
    idx_guard = src.index("if len(lb) != N or len(ub) != N:")
    idx_assign = src.index("xmin_dyn[nx::nx] = lb")
    assert idx_guard < idx_assign
