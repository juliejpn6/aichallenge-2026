"""OT必要最小オフセット(min_needed_mag)の一時ロスト時凍結保持(298節続報、2026-08-05)。

背景: ユーザー指摘「コーナーアウトからの追い越しで大回りしないようにしたい」を受け、
dev3実測ログ([OT]の`min_needed`フィールド)を確認したところ、対象車が現在周期の
`_scan["cars"]`候補から一時的に外れた(コーナーでの視野角変化・一時的な視認ロス等)
瞬間に、必要最小オフセット計算(0804節で実装済み、対象車の現在横位置ベース)が
`None`になり、即座に固定値`self._ot_d_off`(3.0m、コリドーが許す限りの最大幅寄せ)へ
切り替わっていることを確認した。これが「大回り」の一因と考えられる。

対処: すぐ下にある既存のcorr_bound凍結ロジック(168節、非正転落時に直近の有効値を
凍結保持する設計)と同じ考え方で、対象車ロスト時は直近に計算できていた必要最小
オフセットを凍結保持し、固定値へは(一度も計算できていない場合のみ)フォールバック
する。

mpc_controller.pyはrclpy依存で直接importできないため、既存の同種テストと同じ
「ソーステキスト構造検証」の方針を踏襲する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _min_needed_block():
    idx = _SRC.index("_opp_lat_now = None\n")
    idx_end = _SRC.index("_fwd_dbg[\"min_needed_mag\"]", idx)
    return _SRC[idx:idx_end]


# ---------------------------------------------------------------------------
# ①ソーステキスト構造検証: 3分岐(計算成功/凍結保持/初回フォールバック)の存在
# ---------------------------------------------------------------------------

def test_computes_and_caches_when_opponent_lat_available():
    snippet = _min_needed_block()
    assert "if _opp_lat_now is not None:" in snippet
    assert "self._ot_last_valid_min_needed_mag = _target_mag" in snippet


def test_freezes_last_valid_value_when_opponent_lost():
    """対象車ロスト時、固定値へ即座に切り替わらず直近の有効値を優先すること。"""
    snippet = _min_needed_block()
    assert "elif self._ot_last_valid_min_needed_mag is not None:" in snippet
    idx_elif = snippet.index("elif self._ot_last_valid_min_needed_mag is not None:")
    idx_else = snippet.index("else:", idx_elif)
    freeze_body = snippet[idx_elif:idx_else]
    assert "_target_mag = self._ot_last_valid_min_needed_mag" in freeze_body


def test_falls_back_to_fixed_d_off_only_when_never_computed():
    """一度も対象車の位置が取れていない(=凍結値も無い)場合のみ、
    従来通りの固定値d_offへフォールバックすること(退行防止)。"""
    snippet = _min_needed_block()
    idx_else = snippet.rindex("else:")
    tail = snippet[idx_else:]
    assert "_target_mag = self._ot_d_off" in tail


# ---------------------------------------------------------------------------
# ②退行防止: 既存のcorr_bound凍結ロジック(168節、_ot_last_valid_target_mag)と
#   混同・上書きしていないこと(別軸の値として独立管理)
# ---------------------------------------------------------------------------

def test_new_freeze_variable_is_independent_from_corr_bound_freeze_variable():
    assert "self._ot_last_valid_min_needed_mag" in _SRC
    assert "self._ot_last_valid_target_mag" in _SRC
    # 新変数は既存のcorr_bound凍結ロジック(_target_mag = self._ot_last_valid_target_mag)
    #   の行を書き換えていないこと
    assert "_target_mag = self._ot_last_valid_target_mag" in _SRC


# ---------------------------------------------------------------------------
# ③初期化・全リセット箇所の網羅性(既存の_ot_last_valid_target_magと同数)
# ---------------------------------------------------------------------------

def test_reset_site_count_matches_sibling_variable():
    """新変数のリセット箇所数が、既存のきょうだい変数(_ot_last_valid_target_mag)の
    リセット箇所数と一致すること(新規エピソード開始点を漏れなく網羅しているかの
    網羅性チェック、将来どちらかだけ増減した際の検知用)。

    2026-08-07改訂(Fix B、design_docs opp_lat_pred_overlap_guard_design_
    20260806.md §4): 従来個別に4リセット箇所に重複実装されていたブロックを
    共通ヘルパー_reset_ot_episode_tracking_state()へ統合したため、実際の
    ソース上の出現数は「__init__(1) + ヘルパー定義内(1)」の2箇所に減る。"""
    n_new = _SRC.count("self._ot_last_valid_min_needed_mag = None")
    n_sibling = _SRC.count("self._ot_last_valid_target_mag = None")
    assert n_new == n_sibling == 2, (
        f"想定は両方2箇所(__init__+ヘルパー定義)だが "
        f"min_needed_mag={n_new}, target_mag={n_sibling}")
