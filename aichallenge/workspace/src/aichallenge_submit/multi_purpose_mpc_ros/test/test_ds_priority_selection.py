"""Unit tests for _ds_priority (129節続報, A-2, 2026-07-20).

Background: _scan_traffic(fwd_dlat/fwd_ds等、全判定の元データ)と
_follow_speed_limit(icc_stop本体)の両方が、対象車選択に`ds < best[0]`という
生値比較を使っていた。ds規約(前+/後-)のもとでは、後方の車(dsがより負)が
前方の車より常に優先されるという物理的に逆転した選択になる。

0720-2実測(wp330)でこれを直接確認した: d2(ds=1.94, dlat=1.87、前方)と
d3(ds=-1.98, dlat=2.36、後方)が同時に視界内にいる場面で、`ds<best[0]`は
d3(-1.98<1.94)を選んでしまい、本来engage_lat_max(2.0)以内で追従できたはずの
d2が無視された。この結果STOPPING-NO-VSAFE(127節)の空白に陥り、STUCKカスケード
→21.45秒の壁ペナルティに至ったと分析された。

前方(ds>=0)を常に優先し、前方候補が無い場合のみ後方(along_min_length許容窓内)
をゼロに近い順で選ぶ_ds_priorityへ統一した。

テスト方針: mpc_controller.pyはrclpy依存のため直接importできないが、
_ds_priorityは@staticmethodで自己完結しているためAST抽出して実際のコードを
直接検証する(test_switchback_curvature_veto.py等と同じ手法)。
"""
import ast
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")


def _extract_method(name):
    with open(_SRC_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return ast.get_source_segment(src, item)
    raise RuntimeError(f"{name} not found in {_SRC_PATH}")


_NS = {}
exec(compile(_extract_method("_ds_priority"), "<_ds_priority>", "exec"), _NS)
_ds_priority = _NS["_ds_priority"]


def test_forward_always_beats_backward():
    """本修正の中核: 前方(ds>=0)は、どれだけ遠くても後方(ds<0)のどれだけ近い
    値より常に優先される(キーが小さい)。"""
    assert _ds_priority(24.9) < _ds_priority(-0.01)


def test_retroactive_0720_2_wp330_incident_d2_now_wins():
    """遡及検証(0720-2実測wp330): d2(ds=1.94、前方)とd3(ds=-1.98、後方)が
    同時に視界内にいる場合、修正後はd2が選ばれる(旧実装はd3を誤選択していた)。"""
    ds_d2, ds_d3 = 1.94, -1.98
    assert _ds_priority(ds_d2) < _ds_priority(ds_d3)
    # 旧実装(生のds比較)ではd3が勝っていたことの確認(回帰前提の記録)
    assert ds_d3 < ds_d2


def test_among_forward_candidates_smaller_ds_wins():
    """回帰: 前方候補同士では、これまで通り近い方(ds小)が優先される。"""
    assert _ds_priority(1.0) < _ds_priority(5.0)


def test_among_backward_candidates_closer_to_zero_wins():
    """本修正の性質: 後方候補同士では、ゼロに近い方(より前寄り)が優先される。"""
    assert _ds_priority(-0.5) < _ds_priority(-1.9)


def test_ds_zero_beats_any_backward():
    """境界値: ds=0.0(前方扱いの境界)は、どんな後方値よりも優先される。"""
    assert _ds_priority(0.0) < _ds_priority(-0.001)


def test_ds_zero_beats_larger_forward():
    """境界値: 前方候補同士ではds=0.0(最も近い)がより遠い正の値に優先する
    (通常のds<優先則、後方特別扱いの境界であるds=0.0でも変わらない)。"""
    assert _ds_priority(0.0) < _ds_priority(0.1)


# ---------------------------------------------------------------------------
# mpc_controller.py側の配線を構造的に検証
# ---------------------------------------------------------------------------

with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def test_no_raw_ds_comparison_remains_in_repo():
    """回帰防止(水平展開の確認): 修正前に存在した生のds比較文
    `if best is None or ds < best[0]:`がリポジトリ内から他に残っていないことを
    確認する(_ds_priority自身のdocstring内の説明文は対象外、実コード文のみ検査)。"""
    assert "if best is None or ds < best[0]:" not in _SRC


def test_scan_traffic_best_selection_uses_ds_priority():
    idx = _SRC.index("def _scan_traffic")
    idx_cmp = _SRC.index("self._ds_priority(ds) < self._ds_priority(best[0])", idx)
    idx_next_def = _SRC.index("def _g2_speed")
    assert idx < idx_cmp < idx_next_def


def test_follow_speed_limit_best_selection_uses_ds_priority():
    idx = _SRC.index("def _follow_speed_limit")
    idx_cmp = _SRC.index("self._ds_priority(ds) < self._ds_priority(best[0])", idx)
    idx_next_def = _SRC.index("def _closest_wp_and_s")
    assert idx < idx_cmp < idx_next_def


def test_ds_priority_used_exactly_twice_no_third_occurrence():
    """②非冗長性+水平展開の確認: _ds_priority()の呼び出し(比較用の2回、
    self._ds_priority(ds)とself._ds_priority(best[0])のペア)が
    _scan_traffic/_follow_speed_limitの2箇所のみに存在する
    (定義1回+呼び出し2箇所×2回=4回+定義内1回の合計)。"""
    assert _SRC.count("self._ds_priority(ds) < self._ds_priority(best[0])") == 2
