"""Unit tests for 190-4節(2026-07-26): footprint_risk起因cooldownの
デッドロック解消(タイムアウトフォールバックの追加)。

背景: 5日分18ログの機械的横断調査(190節)で、「相手車が停止しているとき
自車が相手の後方で完全停止し、相手が発進するまで再発進できない」という
ユーザー報告症状の主要因の1つとして、footprint_risk起因cooldownの
自己ロックが5ログ(0722-03/0724-01/0724-02/0725-02/0726-02)で確認された
(最長383秒未解決)。

根本原因: `_evaluate_engage_readiness()`の`_cd_clear`計算式(148節②)は、
footprint_risk起因のcooldown中は`_ot_footprint_risk_clear_count`
(危険域=`_fp_near_zone`から連続8周期抜けたか)のみで解除判定しており、
固定タイマー`self._ot_engage_cooldown`を一切参照していなかった。しかし
icc_stop追従は自車を相手と同一ライン上へ収束させるため、`fwd_dlat`が
物理的に0近傍へ張り付き続け、`_fp_near_zone`が恒常的に真になり
`_ot_footprint_risk_clear_count`が0からリセットされ続ける——ENGAGE
(このcooldownで塞がれている当のもの)なしには絶対に解消しない構造的な
デッドロックだった。

一方`self._ot_engage_cooldown`自体は、footprint_risk起因かどうかに
関わらず毎周期無条件でデクリメントされており(139節でfootprint_risk
起因時は2倍≈8秒@40Hzに設定済み)、単に本判定式が参照していなかった
だけだった。

対処: `_cd_clear`のfootprint_risk起因分岐に
`or self._ot_engage_cooldown == 0`を追加するのみ。新規パラメータ0個、
新規状態変数0個。デバウンス方式(高速解除、通常0.2秒)はそのまま維持し、
それが機能しない場合の上限(既存の8秒タイマー)を安全弁として追加する。

mpc_controller.pyはrclpy依存で直接importできないため、`_cd_clear`計算式
をミラー実装した上でソーステキスト検証と組み合わせる(既存テストと同じ
方針)。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

ENGAGE_DEBOUNCE = 8


def mirror_cd_clear(clear_count, cooldown_remaining, gated, debounce=ENGAGE_DEBOUNCE):
    """`_evaluate_engage_readiness`内`_cd_clear`計算式(190-4節修正後)のミラー。"""
    if gated:
        return clear_count >= debounce or cooldown_remaining == 0
    return cooldown_remaining == 0


# ---------------------------------------------------------------------------
# ①非矛盾性: デバウンス方式(高速解除)は無変更のまま維持される
# ---------------------------------------------------------------------------

def test_debounce_path_still_clears_fast_when_zone_actually_exits():
    """危険域を実際に抜けて8周期連続すれば、タイマーが残っていても即座に
    解除される(通常ケース、従来通り高速)。"""
    assert mirror_cd_clear(clear_count=8, cooldown_remaining=250, gated=True) is True


def test_debounce_path_blocks_while_count_below_debounce_and_timer_alive():
    """危険域内にい続け、かつタイマーもまだ残っている間はブロックされ続ける。"""
    assert mirror_cd_clear(clear_count=0, cooldown_remaining=100, gated=True) is False
    assert mirror_cd_clear(clear_count=7, cooldown_remaining=1, gated=True) is False


# ---------------------------------------------------------------------------
# 核心: タイマーによるデッドロック脱出(本節の追加分)
# ---------------------------------------------------------------------------

def test_timer_fallback_clears_deadlock_when_count_never_resets():
    """本修正の核心: _fp_near_zoneが恒常的に真でclear_countが0に張り付いた
    ままでも、タイマーが0に達すれば解除される(旧実装ではFalseのまま
    永久に固定されていた=デッドロック)。"""
    assert mirror_cd_clear(clear_count=0, cooldown_remaining=0, gated=True) is True


def test_timer_counts_down_to_bound_the_worst_case_wait():
    """タイマーの初期値(139節、footprint_risk起因は2倍≈320周期@40Hz≈8秒)を
    使って、最悪ケースでも8秒で確実に解除されることをシミュレートする。"""
    remaining = 320  # engage_cooldown_cycles(160) * 2, footprint_risk起因
    clear_count = 0  # _fp_near_zoneが一度も抜けない最悪ケース
    cleared_at_cycle = None
    for cycle in range(400):
        if mirror_cd_clear(clear_count, remaining, gated=True):
            cleared_at_cycle = cycle
            break
        remaining = max(0, remaining - 1)
    assert cleared_at_cycle == 320  # 8秒@40Hzでちょうど解除される


# ---------------------------------------------------------------------------
# ②非冗長性・回帰: gated=Falseの経路(149節の固定タイマー方式)は無変更
# ---------------------------------------------------------------------------

def test_not_gated_path_unaffected_by_this_change():
    assert mirror_cd_clear(clear_count=0, cooldown_remaining=1, gated=False) is False
    assert mirror_cd_clear(clear_count=0, cooldown_remaining=0, gated=False) is True
    # not-gated分岐はclear_countを一切参照しない(引数を変えても結果不変)
    assert (mirror_cd_clear(clear_count=999, cooldown_remaining=0, gated=False)
            == mirror_cd_clear(clear_count=0, cooldown_remaining=0, gated=False))


# ---------------------------------------------------------------------------
# 遡及検証: 5ログで観測された自己ロックパターンの再現
# ---------------------------------------------------------------------------

def test_retroactive_icc_stop_following_deadlock_now_bounded():
    """遡及検証: icc_stop追従でfwd_dlat≈0(_fp_near_zone恒常的に真)に張り付き
    続けるシナリオ(0722-03/0724-01/0724-02/0725-02/0726-02で実測)を模擬し、
    旧実装なら解除されなかったケースが、新実装ではタイマー到達で解除される
    ことを確認する。"""
    clear_count = 0
    remaining = 320
    old_impl_cleared = False  # 旧実装: clear_countのみで判定
    for _ in range(320):
        # _fp_near_zoneが恒常的に真 → clear_countは常に0のまま
        clear_count = 0
        remaining = max(0, remaining - 1)
        if clear_count >= ENGAGE_DEBOUNCE:
            old_impl_cleared = True
    assert old_impl_cleared is False  # 旧実装: 320周期経っても一度も解除されない
    assert mirror_cd_clear(clear_count=0, cooldown_remaining=0, gated=True) is True  # 新実装: 解除される


# ---------------------------------------------------------------------------
# ソーステキスト検証: 実際の修正箇所
# ---------------------------------------------------------------------------

def test_source_cd_clear_has_timer_fallback():
    idx = _SRC.index("_cd_clear = (")
    snippet = _SRC[idx:idx + 300]
    assert "self._ot_footprint_risk_clear_count >= self._ot_engage_debounce" in snippet
    assert "or self._ot_engage_cooldown == 0" in snippet
    assert "if self._ot_footprint_risk_gated" in snippet
    assert "else self._ot_engage_cooldown == 0)" in snippet


def test_source_cooldown_timer_still_decrements_unconditionally():
    """④遡及効果: この修正が成立する前提(タイマーがgatedかどうかに関わらず
    毎周期デクリメントされ続けること、139節)自体は無変更であることを確認する。"""
    idx = _SRC.index("if self._ot_engage_cooldown > 0:")
    snippet = _SRC[idx:idx + 100]
    assert "self._ot_engage_cooldown -= 1" in snippet


def test_source_footprint_risk_doubling_still_present():
    """④遡及効果: 139節のfootprint_risk起因2倍化ロジック自体は無変更のまま
    残っており、本修正のタイマー上限がその値(≈8秒)を引き続き使うことを
    確認する。"""
    idx = _SRC.index("self._ot_engage_cooldown = (")
    snippet = _SRC[idx:idx + 200]
    assert "self._ot_engage_cooldown_cycles * 2" in snippet
    assert "if _lat_dec.footprint_risk_triggered" in snippet
