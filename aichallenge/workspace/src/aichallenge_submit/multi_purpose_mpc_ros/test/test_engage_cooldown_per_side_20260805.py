"""task#265: side別engage_cooldown解除の実装(2026-08-05)。

背景: wp211-224実測(2026-08-04)で、右側からのOTがgiveupした直後、相手が
(giveupとは無関係に)左側を大きく開けたにもかかわらず、左側への即座の再エンゲージが
グローバルクールダウン(約4秒)の間ブロックされ続けた事象を確認した。外部AI
(Gemini・別Claudeインスタンス)への相談を経て、以下の設計で確定した:

- giveup原因のうち空間的失敗(_side_blockedかつfootprint_risk起因でない、
  主にroom_exhausted)のみ該当側だけをブロックする。
- footprint_risk・相手が速すぎる等「側と無関係な相手の属性」による断念は
  従来通り両側グローバルブロック(既存のself._ot_engage_cooldown)のまま維持する。
- 既存のグローバルクールダウン・_cd_clear(footprint_risk専用の実測解除経路)には
  一切触れず、left_ok/right_okへ個別にANDする形で追加する(config gate
  cooldown_per_side、既定OFF時は現行とビット等価)。
- _plan_passは独自の地形判定でsideを選ぶためside別クールダウンを知らない。
  選ばれた側がクールダウン中ならplan_okを事後に打ち消すガードを追加する。
- 対象車ID別(vid)への拡張は見送り(side単独キー、3台密集時の保護効果を
  A/B検証してから再検討)。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


# ---------------------------------------------------------------------------
# ①状態変数・config: 新規変数が既定値付きで宣言されていること
# ---------------------------------------------------------------------------

def test_new_state_variables_declared_with_safe_defaults():
    assert "self._ot_engage_cooldown_l = 0" in _SRC
    assert "self._ot_engage_cooldown_r = 0" in _SRC
    assert ('self._ot_cooldown_per_side = bool(\n'
            '                _otget("cooldown_per_side", False))') in _SRC


def test_config_yaml_has_cooldown_per_side_key():
    """2026-08-05修正: 当初はconfig.yamlの現在の運用値(false)を直接検証して
    いたが、これは実地検証・予選投入のために意図的にtrueへ変更されうる
    (現に本日中に変更された)運用値であり、テストとして不適切だった。
    「既定値がfalseであること」の検証は上記test_new_state_variables_
    declared_with_safe_defaultsが_otget()のPython側デフォルト引数を通して
    既に行っているため、ここではconfig.yamlにキー自体が存在することのみ
    確認する(既存のcatchup_predict_enable系テストと同じ設計に統一)。"""
    _cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(_cfg_path) as f:
        cfg = f.read()
    assert "cooldown_per_side:" in cfg


# ---------------------------------------------------------------------------
# ②デクリメント: side別カウンタも(gate値に関わらず)常に消化されること
# ---------------------------------------------------------------------------

def test_per_side_cooldown_always_decremented():
    idx = _SRC.index("if self._ot_engage_cooldown > 0:\n                self._ot_engage_cooldown -= 1")
    snippet = _SRC[idx:idx + 600]
    assert "if self._ot_engage_cooldown_l > 0:" in snippet
    assert "self._ot_engage_cooldown_l -= 1" in snippet
    assert "if self._ot_engage_cooldown_r > 0:" in snippet
    assert "self._ot_engage_cooldown_r -= 1" in snippet


# ---------------------------------------------------------------------------
# ③giveup時のセット: footprint_risk起因でない空間的失敗のみ該当側をセット
# ---------------------------------------------------------------------------

def test_per_side_cooldown_set_only_for_non_footprint_side_blocked():
    idx = _SRC.index("self._ot_engage_cooldown = (\n"
                      "                            self._ot_engage_cooldown_cycles * 2")
    idx_end = _SRC.index("self._ot_footprint_risk_gated = _lat_dec.footprint_risk_triggered", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._ot_cooldown_per_side and _side_blocked" in snippet
    assert "not _lat_dec.footprint_risk_triggered" in snippet
    assert "_locked != 0" in snippet
    assert "self._ot_engage_cooldown_l = self._ot_engage_cooldown_cycles" in snippet
    assert "self._ot_engage_cooldown_r = self._ot_engage_cooldown_cycles" in snippet


def test_global_cooldown_still_set_unconditionally():
    """footprint_risk起因・相手が速すぎる等でも、既存のグローバル
    self._ot_engage_cooldown設定は無変更のまま維持されていること(退行防止)。"""
    idx = _SRC.index("self._ot_engage_cooldown = (\n"
                      "                            self._ot_engage_cooldown_cycles * 2")
    snippet = _SRC[idx:idx + 300]
    assert "if _lat_dec.footprint_risk_triggered" in snippet
    assert "else self._ot_engage_cooldown_cycles)" in snippet


def test_global_cooldown_reset_to_zero_when_per_side_mode_blocks_side():
    """2026-08-05訂正で発見・修正した欠陥の再発防止テスト: gate ONかつ
    room_exhausted系(_side_blockedかつfootprint_risk起因でない)の場合、
    グローバルself._ot_engage_cooldownを0へリセットしないと、_cd_clear
    (グローバル判定、無変更)がグローバル値のみを見るため、side別クール
    ダウンをいくら分離してもグローバルが残っている間は_cheap_ok全体が
    ブロックされ続け、side分離の効果が事実上ゼロになる。"""
    idx = _SRC.index("if (self._ot_cooldown_per_side and _side_blocked\n"
                      "                                and not _lat_dec.footprint_risk_triggered\n"
                      "                                and _locked != 0):")
    idx_end = _SRC.index("self.get_logger().warn(", idx)
    snippet = _SRC[idx:idx_end]
    assert "self._ot_engage_cooldown = 0" in snippet
    # 上記のリセット代入が、side別カウンタへのセットより前にあること
    #   (同じifブロック内での実行順序確認)
    idx_reset = snippet.index("self._ot_engage_cooldown = 0")
    idx_l = snippet.index("self._ot_engage_cooldown_l = self._ot_engage_cooldown_cycles")
    assert idx_reset < idx_l


# ---------------------------------------------------------------------------
# ④cheap_ok呼び出し元: gate ON時のみleft_ok/right_okへ側別クールダウンをAND
# ---------------------------------------------------------------------------

def test_left_right_ok_anded_with_per_side_cooldown_at_call_site():
    idx = _SRC.index("_left_ok_cd = _left_ok")
    idx_end = _SRC.index("_eval = self._evaluate_engage_readiness(", idx)
    snippet = _SRC[idx:idx_end]
    assert "_right_ok_cd = _right_ok" in snippet
    assert "if self._ot_cooldown_per_side:" in snippet
    assert "_left_ok_cd = _left_ok and self._ot_engage_cooldown_l == 0" in snippet
    assert "_right_ok_cd = _right_ok and self._ot_engage_cooldown_r == 0" in snippet


def test_evaluate_engage_readiness_receives_cd_variants():
    idx = _SRC.index("_eval = self._evaluate_engage_readiness(\n"
                      "                        _scan, _pass_worth, _v_odom, _left_ok_cd, _right_ok_cd,")
    assert idx > 0  # インデックスが見つかること自体が配線確認


def test_gate_off_is_bit_equivalent_structurally():
    """cooldown_per_side=False(既定)の場合、_left_ok_cd/_right_ok_cdは
    _left_ok/_right_okそのままとなり(if文の外で初期値として代入済み)、
    以降のロジックへ現行と同一の値が渡ることを、ソース構造から確認する。"""
    idx = _SRC.index("_left_ok_cd = _left_ok")
    idx_end = _SRC.index("if self._ot_cooldown_per_side:", idx)
    snippet = _SRC[idx:idx_end]
    assert "_right_ok_cd = _right_ok" in snippet


# ---------------------------------------------------------------------------
# ⑤_plan_pass後の事後チェック: _plan_passが独自にsideを選ぶため、
#   クールダウン中の側が選ばれた場合はplan_okを打ち消すガードが必要
# ---------------------------------------------------------------------------

def test_plan_side_checked_against_left_right_ok_after_plan_pass():
    idx = _SRC.index("_plan_ok, _plan_side, _plan_req = self._plan_pass(scan, _prefer_side)")
    idx_end = _SRC.index("else:\n            _plan_ok, _plan_side = False, 0", idx)
    snippet = _SRC[idx:idx_end]
    assert "if _plan_ok and _plan_side != 0:" in snippet
    assert "_plan_side > 0 and not left_ok" in snippet
    assert "_plan_side < 0 and not right_ok" in snippet
    assert "_plan_ok = False" in snippet
    assert 'self._dbg_plan_reason = "side_cooldown_blocked"' in snippet


# ---------------------------------------------------------------------------
# ⑥退行防止: 既存のグローバル_cd_clear計算・footprint_risk実測解除経路は無変更
# ---------------------------------------------------------------------------

def test_global_cd_clear_unchanged():
    assert "_cd_clear = (" in _SRC
    assert "self._ot_footprint_risk_clear_count >= self._ot_engage_debounce" in _SRC
    assert "or self._ot_engage_cooldown == 0)" in _SRC
