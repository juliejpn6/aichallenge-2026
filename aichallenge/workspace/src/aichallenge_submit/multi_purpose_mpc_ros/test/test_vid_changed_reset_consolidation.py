"""対象車ID変化検知の共通化(233節続報、2026-07-29、監査結果④)。

背景: コード重複の全面監査(230節続報)で、STUCK WAIT_REVERSE統合(232節)・
壁基準空き幅計算の共通化(233節)に続く3件目の対処として、「対象車IDが前回と
変わった周期は関連状態を仕切り直す」という慣用句(94節で導入、
test_scan_traffic_target_id_consistency.py参照)が、以下3箇所に
read→比較→prev_attr更新の3行として手作業で複製されていることが判明した:
  - _ot_worth_count(エンゲージ判定デバウンス、リセット先=カウンタ0)
  - _ot_giveup_count(OVERTAKING中の断念デバウンス、リセット先=カウンタ0)
  - _along_lane_ema(並走レーン幅の平滑化、リセット先=None)

231節(lateral_target/_ot_alpha)・232節(STUCK WAIT_REVERSE)と同型の
「複数箇所に手書きされた同一処理」パターンであり、片方だけ将来修正されて
挙動が乖離するリスクがあった。

対処: 新規ヘルパー_vid_changed_reset(current_vid, prev_attr)を追加し、
「比較→prev_attr更新→変化有無をboolで返す」部分のみを共通化した。実際の
リセット対象(カウンタを0にする/EMAをNoneにする)はサイトごとに意味が
異なるため、_room_to_wallのclamp引数と同じ考え方で、呼び出し元の責務として
明示的に残した(機械的に統一すると個々の意味が壊れるため)。

mpc_controller.pyはrclpy非依存のため直接importできないが、本ヘルパー自体は
self.get_logger()等ROS依存を一切持たない自己完結した関数のため、ソース
テキストによる構造的検証に加えて、実際にexecして動作(比較・更新・戻り値)を
直接検証する。
"""
import os
import types

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _extract_method(name):
    idx = _SRC.index(f"def {name}(self")
    idx_end = _SRC.index("\n    def ", idx + 10)
    return _SRC[idx:idx_end]


_NS = {}
exec(compile(_extract_method("_vid_changed_reset"), "<_vid_changed_reset>", "exec"), _NS)


def make_self(**prev_attrs):
    m = types.SimpleNamespace(**prev_attrs)
    m._vid_changed_reset = types.MethodType(_NS["_vid_changed_reset"], m)
    return m


# ---------------------------------------------------------------------------
# ①非矛盾性: ヘルパー自体の比較・更新・戻り値が正しいこと
# ---------------------------------------------------------------------------

def test_returns_true_and_updates_prev_when_vid_changes():
    m = make_self(_x_prev_vid="d1")
    changed = m._vid_changed_reset("d2", "_x_prev_vid")
    assert changed is True
    assert m._x_prev_vid == "d2"


def test_returns_false_and_leaves_prev_unchanged_when_vid_same():
    m = make_self(_x_prev_vid="d1")
    changed = m._vid_changed_reset("d1", "_x_prev_vid")
    assert changed is False
    assert m._x_prev_vid == "d1"


def test_none_to_none_is_not_a_change():
    """初期状態(未エンゲージ、対象車なし)がNone→Noneの間、誤って
    毎周期「変化した」扱いされないことを確認する。"""
    m = make_self(_x_prev_vid=None)
    changed = m._vid_changed_reset(None, "_x_prev_vid")
    assert changed is False


def test_none_to_vid_is_a_change():
    """初回エンゲージ(対象車なし→初検知)は正しく「変化」として扱われる
    (94節の元の意図: 初回もリセットが必要な場面がある)。"""
    m = make_self(_x_prev_vid=None)
    changed = m._vid_changed_reset("d3", "_x_prev_vid")
    assert changed is True
    assert m._x_prev_vid == "d3"


def test_vid_to_none_is_a_change():
    """対象車が消失した周期(相手がいなくなった)も正しく「変化」扱いされる。"""
    m = make_self(_x_prev_vid="d3")
    changed = m._vid_changed_reset(None, "_x_prev_vid")
    assert changed is True
    assert m._x_prev_vid is None


def test_prev_attr_is_generic_works_for_any_attribute_name():
    """②非冗長性の裏返し: 1つの実装が_ot_worth_prev_vid/_ot_giveup_prev_vid/
    _along_lane_prev_vidのいずれの属性名に対しても同じロジックで動作する
    (attr名をパラメータ化しているため、3箇所分の重複コードが不要)。"""
    m = make_self(_ot_worth_prev_vid="d1", _ot_giveup_prev_vid="d1",
                  _along_lane_prev_vid="d1")
    assert m._vid_changed_reset("d2", "_ot_worth_prev_vid") is True
    assert m._vid_changed_reset("d1", "_ot_giveup_prev_vid") is False
    assert m._vid_changed_reset("d9", "_along_lane_prev_vid") is True
    assert m._ot_worth_prev_vid == "d2"
    assert m._ot_giveup_prev_vid == "d1"
    assert m._along_lane_prev_vid == "d9"


# ---------------------------------------------------------------------------
# ④遡及効果: 3箇所全ての呼び出し元がヘルパー経由になっていること
# ---------------------------------------------------------------------------

def test_ot_worth_call_site_uses_helper():
    idx = _SRC.index("_fwd_vid_worth = opp_sit.fwd_vid")
    snippet = _SRC[idx:idx + 200]
    assert 'if self._vid_changed_reset(_fwd_vid_worth, "_ot_worth_prev_vid"):' in snippet
    assert "self._ot_worth_count = 0" in snippet


def test_ot_giveup_call_site_uses_helper():
    idx = _SRC.index("_fwd_vid_giveup = _opp_sit.fwd_vid")
    snippet = _SRC[idx:idx + 200]
    assert 'if self._vid_changed_reset(_fwd_vid_giveup, "_ot_giveup_prev_vid"):' in snippet
    assert "self._ot_giveup_count = 0" in snippet


def test_along_lane_call_site_uses_helper():
    idx = _SRC.index('_along_vid_now = _scan.get("along_vid")')
    snippet = _SRC[idx:idx + 200]
    assert 'if self._vid_changed_reset(_along_vid_now, "_along_lane_prev_vid"):' in snippet
    assert "self._along_lane_ema = None" in snippet


def test_total_call_count_matches_three_known_sites():
    """新しい「対象車ID変化検知」箇所が追加/削除された場合はこのテスト自体の
    更新も必要。room-debounce(_plan_room_ok_count系、3134/3143行目付近)は
    既に別関数_room_debounce_ok内で意図的に別実装(counter_key分岐)のため
    対象外——本ヘルパーへ統合すると案B由来の独立設計(190-7節)の意図を壊す。"""
    n_calls = _SRC.count("self._vid_changed_reset(")
    assert n_calls == 3, (
        f"想定していた3箇所から数が変わっている(現在{n_calls}箇所)。"
        "新しい「対象車ID変化検知」箇所が追加/削除された場合はこのテスト自体の更新も必要。")


# ---------------------------------------------------------------------------
# ②非冗長性: 旧来の手作業複製(read→比較→prev更新の3行インライン展開)が
#   ヘルパー本体以外のどこにも残っていないこと
# ---------------------------------------------------------------------------

def test_no_hand_duplicated_inline_pattern_remains():
    idx_helper = _SRC.index("def _vid_changed_reset")
    idx_helper_end = _SRC.index("\n    def ", idx_helper + 10)
    before = _SRC[:idx_helper]
    after = _SRC[idx_helper_end:]
    for outside_snippet, label in ((before, "helper定義より前"), (after, "helper定義より後")):
        assert "!= self._ot_worth_prev_vid:" not in outside_snippet, (
            f"{label}に_ot_worth_prev_vidの旧来インライン比較が残っている")
        assert "!= self._ot_giveup_prev_vid:" not in outside_snippet, (
            f"{label}に_ot_giveup_prev_vidの旧来インライン比較が残っている")
        assert "!= self._along_lane_prev_vid" not in outside_snippet, (
            f"{label}に_along_lane_prev_vidの旧来インライン比較が残っている")
