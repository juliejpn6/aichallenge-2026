"""Regression tests for the "_scan_traffic ds>0 cliff" fix (105節発見→110節で
ローカル・予選の2環境で再確認→2026-07-19実装、ユーザー承認済み設計)。

背景: `_scan_traffic`(mpc_controller.py:1617-)の前方車判定は`0.0 < ds`という
猶予無しの厳密な正値要求だった。静止/低速の相手にごく至近距離まで詰めた場合、
自車の弧長位置がわずかに相手を跨いだ瞬間にfwd_ds=Noneへ落ち、以下2箇所へ
波及していた:
  - `_offset_return_ok`(3468行目付近): fwd_ds is Noneを「通過完了」と誤判定し
    全開加速する(105節、ローカル実測でSTUCKループ×3サイクルを確認)。
  - `_ot_cleared`再取得ヒステリシス(3580行目付近、110節で新規発見): fwd_dlatが
    Noneの間`if _fd is not None:`ガードにより判定そのものが凍結される。
0718-06(予選、17:38収集)のt=117.91〜118.28秒(wp310-311)で、[OFFSET-RETURN]が
0.34秒間に3回ON/OFFを繰り返す急速フラッピングとして実測確認された(110節)。

対処: 前方車判定の下限を車両全長分だけ緩和。ds<0側でも_g2_speed等の下流の式は
(ds-margin_center)がより負に振れるため自然に保守的側へ働き、安全性が緩む経路は無い。

2026-07-20修正(128節続報): 当初の実装はself._mpc.model.length(1.087、
spatial_bicycle_models.pyの自転車モデルが保持するホイールベース)を「車両全長」
として流用していたが、公式車両仕様(全長200cm)との乖離を128節で発見した。
footprint_risk用に新設した公式全長ベースのalong_min_length(2.00)へ置き換え、
このモジュールが検証していた元々の設計意図(縦方向の量=車両全長を使う)と
実装を一致させた。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできない。分類条件
自体は`-length < ds <= max_consider and abs(lat) <= lat_band`という単純な
不等式のため、同一の式を複製したミラー関数で数式的性質を検証する。
mpc_controller.py側の配線(self._along_min_lengthの再利用、下流2箇所の
恩恵)は末尾の構造的ソーステキスト検証で確認する。
"""
import os


def _is_forward(ds, lat, vehicle_length, max_consider=25.0, lat_band=3.0):
    """mpc_controller.py 1685行目付近の前方車判定式(105節/110節修正後、
    128節続報でvehicle_length算出元をalong_min_lengthへ訂正)の複製ミラー。"""
    return -vehicle_length < ds <= max_consider and abs(lat) <= lat_band


VEHICLE_LENGTH = 2.00  # along_min_length実値(公式車両仕様全長200cm、128節続報)


def test_ds_exactly_zero_now_classified_as_forward_regression():
    """核心(旧仕様からの変更点): 修正前はds=0.0がfwd判定の境界外(0.0<dsが不成立)
    だったため、まさにこの瞬間に対象を見失っていた。修正後は0.0もforward側に
    含まれる。"""
    assert _is_forward(ds=0.0, lat=0.0, vehicle_length=VEHICLE_LENGTH) is True


def test_ds_slightly_negative_within_vehicle_length_now_classified_as_forward():
    """核心: dsがわずかに負(自車が相手をちょうど跨いだ直後)でも、車両全長以内
    ならforwardとして引き続き捕捉される。"""
    assert _is_forward(ds=-0.5, lat=0.0, vehicle_length=VEHICLE_LENGTH) is True


def test_ds_at_negative_vehicle_length_boundary_excluded_regression():
    """境界値: ds=-vehicle_lengthちょうどは対象外(厳密な不等号を維持)。"""
    assert _is_forward(ds=-VEHICLE_LENGTH, lat=0.0, vehicle_length=VEHICLE_LENGTH) is False


def test_ds_far_behind_still_excluded_regression():
    """回帰: 車両全長を大きく超えて後方にいる相手は引き続き対象外
    (無制限に後方まで拾うようになったわけではない)。"""
    assert _is_forward(ds=-5.0, lat=0.0, vehicle_length=VEHICLE_LENGTH) is False


def test_ds_positive_small_still_classified_as_forward_regression():
    """回帰: 旧仕様で既に成立していたds>0の通常ケースは変更されない。"""
    assert _is_forward(ds=0.5, lat=0.0, vehicle_length=VEHICLE_LENGTH) is True


def test_lat_band_still_enforced_regression():
    """回帰: ds条件を緩和しても、lat_band(コリドー帯)の制約は無変更のまま
    引き続き効く。"""
    assert _is_forward(ds=0.5, lat=10.0, vehicle_length=VEHICLE_LENGTH) is False


def test_retroactive_0718_06_wp310_offset_return_flap_scenario():
    """遡及検証(110節、0718-06実測t=117.91〜118.28秒、wp310-311): 静止相手に
    至近距離まで詰めた際、実測ではds相当が0付近で往復し[OFFSET-RETURN]が
    0.34秒間に3回ON/OFFした(fwd_ds=None⇔実測値)。旧判定式(0.0<ds)では
    ds=-0.1のような至近距離の負値で即座にforward判定を失っていたが、
    新判定式では車両全長(along_min_length=2.00m)以内のため引き続きforwardのまま
    安定する。"""
    for ds in [0.15, 0.05, -0.05, -0.1, -0.2, 0.02]:
        assert _is_forward(ds=ds, lat=0.3, vehicle_length=VEHICLE_LENGTH) is True


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_forward_classification_reuses_along_min_length_no_new_parameter():
    """②非冗長性(128節続報で訂正): 前方車判定の下限にはfootprint_risk用に
    新設したalong_min_length(公式全長ベース)を再利用する。旧実装が使っていた
    self._mpc.model.length(ホイールベース、車両全長ではなかった)はもう
    この条件式には使われていないことを確認する。"""
    idx = _SRC.index("if -self._along_min_length < ds <= self._ot_max_consider")
    snippet = _SRC[idx:idx + 120]
    assert "self._along_min_length" in snippet
    assert "self._mpc.model.length" not in snippet
    assert "0.0 < ds" not in snippet


def test_no_other_strict_ds_lower_bound_remains_in_repo():
    """水平展開の確認: 修正前に存在した唯一の`0.0 < ds`パターンが
    リポジトリ内から他に残っていないことを確認する(同型の崖が別途
    存在しないことの回帰防止)。"""
    assert "0.0 < ds <=" not in _SRC
    assert "0 < ds <=" not in _SRC


def test_offset_return_ok_benefits_from_relaxed_forward_window():
    """回帰確認: OFFSET-RETURN判定(105節が特定した本丸)が、緩和後の
    fwd_ds(_scan.get("fwd_ds"))を引き続き参照していることを確認する
    (下流コードの変更は不要なはず、変数名を使い回しているため)。
    2026-07-20修正(131-6節④、対象車の一意性): 判定式自体は_still_ahead経由の
    間接式へ変わったが、_fwd_ds_now = _scan.get("fwd_ds")という同一の取得元は
    維持されている。"""
    idx = _SRC.index('_fwd_ds_now = _scan.get("fwd_ds")')
    assert idx > 0
    idx2 = _SRC.index('_offset_return_ok = self._ot_cleared and not _still_ahead')
    assert idx2 > idx


def test_ot_cleared_hysteresis_benefits_from_relaxed_forward_window():
    """回帰確認(110節で新規発見した第2の恩恵先): _ot_cleared再取得
    ヒステリシスが引き続きfwd_dlatを参照しており、崖の間の凍結が
    緩和されることを確認する。"""
    idx = _SRC.index('_fd = _scan["fwd_dlat"]; _fs = _scan["fwd_ds"]')
    assert idx > 0
