"""Unit tests for the footprint_risk adaptive cooldown clearing (148節②、2026-07-21)。

背景: ローカル実測ログ(20260721-150016/172416)でgiveup直後のfwd_dlat推移を
waypoint単位で追跡したところ、footprint_risk起因のgiveup8件中3件は実際の間隔が
わずか1.3〜5.0秒で回復していたにもかかわらず、既存の固定8秒cooldown(139節)に
よりその後3〜7秒を無駄に待っていたことが判明した。逆に残り5件は8秒経過時点でも
まだ間隔が狭いままで、固定8秒が実際に必要だった。

「固定秒数」ではなく「footprint_risk条件自体(_footprint_risk、毎周期計算済み)が
実際に解消したか」で再エンゲージを解除するよう変更した。footprint_risk起因の
giveupの場合のみ_ot_engage_cooldownの固定タイマーの代わりにこの解除方式を使う。
他のgiveup理由(相手が速すぎる等)は従来通りの固定cooldownのまま(139節の元の
設計意図を維持)。デバウンス(単発の1周期回復でチャーンしないための連続性要求)は
既存のengage_debounce(フリッカー防止、8周期≈0.2秒)をそのまま再利用し、新規
パラメータは追加していない。

mpc_controller.pyはrclpy依存で直接importできないため、ロジックをミラー実装した
上でソーステキスト検証と組み合わせる(既存テストと同じ方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _mirror_cd_clear(footprint_risk_gated, clear_count, engage_debounce,
                      engage_cooldown):
    """_cd_clear計算式のミラー実装。"""
    if footprint_risk_gated:
        return clear_count >= engage_debounce
    return engage_cooldown == 0


# --- ①非矛盾性: footprint_risk起因の場合とそれ以外で経路が分かれること ---

def test_non_footprint_risk_giveup_uses_fixed_timer_unchanged():
    """footprint_risk以外のgiveup理由(例: 相手が速すぎる)は、従来通り
    固定cooldownタイマーのみで解除されることを確認する(139節の意図を維持)。"""
    assert _mirror_cd_clear(footprint_risk_gated=False, clear_count=0,
                             engage_debounce=8, engage_cooldown=5) is False
    assert _mirror_cd_clear(footprint_risk_gated=False, clear_count=0,
                             engage_debounce=8, engage_cooldown=0) is True


def test_footprint_risk_giveup_ignores_fixed_timer_uses_clear_count():
    """footprint_risk起因の場合、固定タイマーがまだ残っていても
    (engage_cooldown>0)、footprint_risk条件がengage_debounce周期連続で
    解消していれば再エンゲージ可能になることを確認する(早期回復の実証)。"""
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=8,
                             engage_debounce=8, engage_cooldown=200) is True


def test_footprint_risk_giveup_still_blocked_if_condition_persists():
    """footprint_risk起因で、固定タイマーが既に0になっていても、footprint_risk
    条件がまだ解消していなければ(clear_count不足)引き続きブロックされることを
    確認する(8秒を超えて必要なら待ち続ける、安全性は損なわない)。"""
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=3,
                             engage_debounce=8, engage_cooldown=0) is False


def test_debounce_boundary_exact():
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=7,
                             engage_debounce=8, engage_cooldown=0) is False
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=8,
                             engage_debounce=8, engage_cooldown=0) is True


# --- ②非冗長性: 新規パラメータを追加していないこと ---

def test_reuses_existing_engage_debounce_no_new_parameter():
    idx = _SRC.index("_cd_clear = (")
    idx_end = idx + 300
    snippet = _SRC[idx:idx_end]
    assert "self._ot_engage_debounce" in snippet
    assert "self._ot_footprint_risk_clear_count" in snippet
    # 新規debounce定数(例: footprint_risk_clear_debounce等)を追加していないこと。
    assert "clear_debounce" not in snippet.replace("_ot_footprint_risk_clear_count", "")


def test_non_footprint_risk_cooldown_value_unchanged():
    """footprint_risk以外のgiveup理由では、_ot_engage_cooldownの設定値
    (self._ot_engage_cooldown_cycles、倍化なし)が従来通り使われることを確認する。"""
    idx = _SRC.index("self._ot_engage_cooldown = (")
    snippet = _SRC[idx:idx + 250]
    assert "self._ot_engage_cooldown_cycles * 2" in snippet
    assert "if _lat_dec.footprint_risk_triggered" in snippet
    assert "else self._ot_engage_cooldown_cycles" in snippet


# --- ③検証ロギング ---

def test_fp_cooldown_clear_log_fires_once_per_episode():
    idx = _SRC.index('f"[FP-COOLDOWN-CLEAR] footprint_risk条件が')
    snippet = _SRC[max(0, idx - 400):idx]
    assert "not self._ot_fp_clear_logged" in snippet
    assert "self._ot_fp_clear_logged = True" in snippet


def test_fp_clear_logged_reset_at_new_giveup_episode():
    idx = _SRC.index("self._ot_footprint_risk_gated = _lat_dec.footprint_risk_triggered")
    snippet = _SRC[idx:idx + 200]
    assert "self._ot_footprint_risk_clear_count = 0" in snippet
    assert "self._ot_fp_clear_logged = False" in snippet


# --- ④過去ログへの遡及効果 ---

def test_retroactive_20260721_150016_d2_case1_would_clear_early():
    """0721実測(150016/d2、giveup@772.59): fwd_dlatが+5.04秒でalong_min_width
    (1.45m)を超え、以降+11秒まで持続的に1.45m以上を維持していた。engage_debounce
    (8周期≈0.2秒)分の連続クリアは+5.04秒台には満たされていたはずで、旧来の固定
    8.05秒cooldownより早く(実測では約3秒早く)解除できていたことを確認する。"""
    # footprint_risk条件が5.04秒時点で不成立に転じ、以降11秒までずっと不成立
    # (=毎周期clear_count加算)だったと仮定すると、0.2秒のデバウンスは即座に満たされる。
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=8,
                             engage_debounce=8, engage_cooldown=120) is True
    # 旧方式(固定8.05秒=322周期@40Hz)ならこの時点(cooldown=120残り)ではまだFalseだった。
    assert _mirror_cd_clear(footprint_risk_gated=False, clear_count=8,
                             engage_debounce=8, engage_cooldown=120) is False


def test_retroactive_20260721_150016_d1_wp800_still_needs_full_wait():
    """0721実測(150016/d1、giveup@800.91): fwd_dlatは8秒経過時点でも0.02m程度
    まで縮小したままで、一度も1.45mを超えなかった。この場合、footprint_risk条件が
    解消しないままclear_countが0に張り付き続け、新方式でも旧方式と同様に
    ブロックされ続けることを確認する(安全性が損なわれないことの裏付け)。"""
    assert _mirror_cd_clear(footprint_risk_gated=True, clear_count=0,
                             engage_debounce=8, engage_cooldown=0) is False
