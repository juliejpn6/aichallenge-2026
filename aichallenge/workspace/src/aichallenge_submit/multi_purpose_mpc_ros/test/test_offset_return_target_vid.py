"""Unit tests for the target-vehicle-aware OFFSET-RETURN gate (131-6節④、
対象車の一意性, 2026-07-20).

Background: 131-3節の実測(0720-02 wp338→339)で、_offset_return_okが
`self._ot_cleared and _scan.get("fwd_ds") is None` という式で、
「前方40m以内の任意の1台」の有無だけを見ていたことが判明した。129節が
footprint_risk向けに後方許容窓(-along_min_lengthまで)を拡張したことで、
71節の「_scan_trafficは0<dsの車のみをcarsに含める」という前提が静かに
崩れ、追い越し直後の相手(ds=-1.99、既に自車の後方)がfwd_ds is not Noneを
成立させ続け、オフセット復帰開始0.83秒後に誤って再拡大していた。

対処: fwd_ds>0(実際に前方にいる)かつfwd_vidが今回のエンゲージ対象
(_ot_target_vid、_plan_passが実際に計画したscan["fwd_vid"]をエンゲージの
たびに記録)と一致する場合のみ「まだクリアしていない」とみなす。3台以上の
レースで無関係な別車が窓内に入った場合の阻害も同時に解消する。対象車ID
不明時は従来の"fwd_ds is not None"へフォールバックする(安全側)。

このメソッドは複雑度が高くモック実行が難しいため、既存の同種テスト
(test_scan_traffic_ds_cliff.py等)と同じくソーステキスト検証+ミラー関数
方式で検証する。
"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


def _still_ahead(target_vid, fwd_ds, fwd_vid):
    """mpc_controller.pyの該当ロジックのミラー実装(構造テストの遡及値検証用)。"""
    if target_vid is not None:
        return fwd_ds is not None and fwd_ds > 0.0 and fwd_vid == target_vid
    return fwd_ds is not None


# --- 振る舞い検証(ミラー関数、実測値の遡及確認) ---

def test_retroactive_0720_02_wp338_339_negative_ds_no_longer_blocks():
    """遡及検証: 0720-02実測wp339のfwd_ds=-1.99(追い越した相手が既に後方)は、
    修正後は「まだ前方にいる」と判定されない(offset-returnを不当にブロックしない)。"""
    assert _still_ahead(target_vid="d3", fwd_ds=-1.99, fwd_vid="d3") is False


def test_same_target_vid_positive_ds_still_blocks_return():
    """対象車が本当にまだ前方にいる(ds>0)間は、引き続き復帰をブロックする
    (回帰防止: 71/105節が対処した「幅寄せ」危険パターンを再現させない)。"""
    assert _still_ahead(target_vid="d3", fwd_ds=1.5, fwd_vid="d3") is True


def test_different_vid_does_not_block_return_uniqueness_fix():
    """131-6節④の核心: 無関係な別車(3台目等)がfwd_ds>0で窓内にいても、
    今回エンゲージした対象車(target_vid)と一致しなければブロックしない。"""
    assert _still_ahead(target_vid="d3", fwd_ds=1.5, fwd_vid="d5") is False


def test_target_vid_none_fwd_ds_positive_still_blocks_fallback():
    """対象車ID不明時(target_vid=None)は、fwd_ds is not Noneのみでブロックする
    従来挙動へフェイルオープンする(安全側、退行なし)。"""
    assert _still_ahead(target_vid=None, fwd_ds=1.5, fwd_vid="d3") is True


def test_target_vid_none_negative_ds_still_blocks_old_behavior_preserved():
    """target_vid=Noneの場合は旧実装同様、ds符号を見ずにfwd_ds is not Noneで
    判定する(フォールバック時は旧挙動をそのまま踏襲する設計)。"""
    assert _still_ahead(target_vid=None, fwd_ds=-1.99, fwd_vid="d3") is True


def test_fwd_ds_none_never_blocks_regardless_of_vid():
    assert _still_ahead(target_vid="d3", fwd_ds=None, fwd_vid=None) is False
    assert _still_ahead(target_vid=None, fwd_ds=None, fwd_vid=None) is False


def test_exact_zero_ds_does_not_block():
    """境界値: ds=0.0(自車と完全に横並び)は">0"を満たさずブロックしない
    (side-by-side区間は_ot_cleared/is_side_by_side等の別機構が担当する領分)。"""
    assert _still_ahead(target_vid="d3", fwd_ds=0.0, fwd_vid="d3") is False


# --- 構造テスト: 配線・②非冗長性・ログの確認 ---

def test_target_vid_captured_at_engage_commit_reusing_existing_scan():
    """②非冗長性: エンゲージ確定時にscan["fwd_vid"]をそのまま_ot_target_vidへ
    複製するだけで、新規スキャン処理を追加していないことを確認する。"""
    idx = _SRC.index("self._ot_side = _eval.plan_side")
    snippet = _SRC[idx:idx + 700]
    assert 'self._ot_target_vid = _scan.get("fwd_vid")' in snippet


def test_target_vid_initialized_in_init():
    assert "self._ot_target_vid = None" in _SRC


def test_offset_return_gate_reuses_ot_cleared_no_new_state():
    """①非矛盾性: _offset_return_okが引き続きself._ot_clearedを起点とし、
    G-2/G-3・LAT-TTC B_clearedバイパスと同じラッチを共有し続けていることを
    確認する(3箇所目の再利用という既存設計方針を維持)。"""
    idx = _SRC.index("_offset_return_ok = self._ot_cleared and not _still_ahead")
    assert idx > 0


def test_offset_return_log_includes_target_vid_for_diagnosis():
    """[OFFSET-RETURN]ログにfwd_vid/target_vidが追加され、次回ログで
    「別車による誤ブロックが解消したか」を直接確認できることを確認する。"""
    idx = _SRC.index("[OFFSET-RETURN]")
    snippet = _SRC[idx:idx + 400]
    assert "fwd_vid={_fwd_vid_now}" in snippet
    assert "target_vid={self._ot_target_vid}" in snippet


def test_fallback_branch_present_for_unknown_target_vid():
    """target_vid不明時のフォールバック分岐がソース上に存在することを確認する
    (安全側、既存挙動からの退行なし)。"""
    idx = _SRC.index("if self._ot_target_vid is not None:")
    snippet = _SRC[idx:idx + 400]
    assert "_still_ahead = _fwd_ds_now is not None" in snippet
