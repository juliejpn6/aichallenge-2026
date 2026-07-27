"""Unit tests for 2026-07-26: `_maybe_log_gnss_ekf_xcorr()`の競合状態クラッシュ修正。

背景: ローカル3台走行(run_dev3_20260726_171301)でd1のmpc_controllerノードが走行
開始からわずか42秒でクラッシュした。

    IndexError: boolean index did not match indexed array along axis 0;
    size of axis is 167 but size of corresponding boolean axis is 168

原因: `self._xcorr_ekf_hist`/`self._xcorr_gnss_hist`は購読コールバック側で
append/pop(0)により毎周期変化する可変リストである。`_maybe_log_gnss_ekf_xcorr()`
はこれらのエイリアス(`ekf_h = self._xcorr_ekf_hist`)から`ekf_t`/`ekf_x`/`ekf_y`を
別々のリスト内包表記で構築していたため、内包表記の間にコールバックが要素を
追加/削除すると配列長が食い違い、`ekf_x[mask]`(maskは先に作ったekf_tベース)で
IndexErrorが発生していた。修正は`list(...)`で1回だけ独立スナップショットを取り、
以降の全ての内包表記がこの不変コピーのみを参照するようにするだけ(新規ロック機構・
新規パラメータは追加しない)。

mpc_controller.pyはrclpy依存で直接importできないため、①競合状態そのものを
モデル化したミラー実装で「修正前パターンは長さ不一致を起こし得るが、修正後パターンは
起こさない」ことを再現し、②ソーステキスト検証で実際の修正箇所を確認する(既存テストと
同じ方針)。"""
import os

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()


class MutatingList(list):
    """既存の可変履歴リスト(self._xcorr_ekf_hist等)を模す。イテレーション中に
    要素が追加/削除されうる実環境を再現するため、__iter__を1回消費するたびに
    自分自身へ副作用(追加)を起こすテスト専用のリスト。"""
    def __init__(self, *args, mutate_after_n_iters=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mutate_after_n_iters = mutate_after_n_iters
        self._iter_count = 0

    def __iter__(self):
        self._iter_count += 1
        if self._mutate_after_n_iters is not None and self._iter_count == self._mutate_after_n_iters:
            # 2回目のイテレーション(=2つ目のリスト内包表記)の直前にコールバックが
            # 新しい要素をappendしたのと同じ効果。
            self.append((999.0, 999.0, 999.0))
        return super().__iter__()


def old_pattern_alias(shared_list):
    """修正前(バグ再現用): エイリアスから複数の内包表記を個別に構築する。"""
    h = shared_list  # バグ: 参照をそのまま使う(独立コピーではない)
    t = [r[0] for r in h]
    x = [r[1] for r in h]  # ここでshared_listへの追加が起きていると長さが食い違う
    return t, x


def new_pattern_snapshot(shared_list):
    """修正後: list()で1回だけ独立スナップショットを取ってから使う。"""
    h = list(shared_list)  # 修正: 独立コピー
    t = [r[0] for r in h]
    x = [r[1] for r in h]
    return t, x


# ---------------------------------------------------------------------------
# ①非矛盾性: 修正前パターンは競合で長さ不一致を起こし得るが、修正後は起こさない
# ---------------------------------------------------------------------------

def test_old_alias_pattern_can_produce_mismatched_lengths_under_mutation():
    """バグ再現: 2つ目の内包表記の直前に要素が追加されると、1つ目より長い配列が
    出来上がる(実際のクラッシュの直接的な原因を再現)。"""
    shared = MutatingList([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)], mutate_after_n_iters=2)
    t, x = old_pattern_alias(shared)
    assert len(t) != len(x)  # バグ: 長さが食い違う
    assert len(x) == len(t) + 1


def test_new_snapshot_pattern_immune_to_mutation_during_construction():
    """修正確認の核心: 同じ「2つ目の内包表記の直前に追加が起きる」状況でも、
    list()で取った独立スナップショットを使えば長さは常に一致する。
    さらに、list()による1回のコピー後は以降の内包表記が全てそのコピー(h、
    通常のlist)側を辿るため、可変な共有リスト(shared)自体は1回しか
    イテレーションされない——mutate_after_n_iters=2の条件(2回目の
    イテレーション時に要素追加)が一度も成立せず、sharedは元の長さ2のまま
    残る。旧パターン(3回以上イテレーションが発生しうる)より競合の露出
    機会自体が減っていることの副次的な確認。"""
    shared = MutatingList([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)], mutate_after_n_iters=2)
    t, x = new_pattern_snapshot(shared)
    assert len(t) == len(x)  # 修正後: 常に一致
    assert len(shared) == 2  # sharedは1回しかイテレーションされず、追加は発生しない


def test_snapshot_independent_of_later_mutation_to_original():
    """スナップショット取得"後"に元のリストが変化しても、既に取得済みのローカル変数
    (h)には一切影響しないことを確認する(独立コピーであることの直接確認)。"""
    shared = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)]
    h = list(shared)
    shared.append((4.0, 4.0, 4.0))
    shared.pop(0)
    assert len(h) == 3
    assert h[0] == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# ②非冗長性・③検証: ソーステキストで実際の修正箇所を確認
# ---------------------------------------------------------------------------

def test_maybe_log_gnss_ekf_xcorr_uses_list_snapshot_for_both_histories():
    idx = _SRC.index("def _maybe_log_gnss_ekf_xcorr(")
    idx_end = idx + 1500
    snippet = _SRC[idx:idx_end]
    assert "ekf_h = list(self._xcorr_ekf_hist)" in snippet
    assert "gnss_h = list(self._xcorr_gnss_hist)" in snippet
    # 修正前の「参照をそのまま使う」パターンが残っていないことを確認する。
    assert "ekf_h = self._xcorr_ekf_hist" not in snippet
    assert "gnss_h = self._xcorr_gnss_hist" not in snippet


def test_no_new_locking_mechanism_introduced():
    """②非冗長性: ロックやミューテックス等の新規排他制御機構は追加していない
    (list()による1回だけの独立コピーのみで十分、という設計判断の裏付け)。"""
    idx = _SRC.index("def _maybe_log_gnss_ekf_xcorr(")
    idx_end = idx + 1500
    snippet = _SRC[idx:idx_end]
    assert "Lock(" not in snippet
    assert "threading" not in snippet
