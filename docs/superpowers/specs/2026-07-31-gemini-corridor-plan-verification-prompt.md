あなたは自動運転レーシングカート(ROS2 Humble / Autoware、spatial MPCで制御)のソフトウェアアーキテクトです。以前、コリドー計算(`update_path_constraints`)のパフォーマンス最適化について相談し、あなたを含む複数のAIから「Phase 0(等価性回帰ハーネス)→Phase 1(計測分解)→Phase 2(境界安全修正)→Phase 3(NumPyベクトル化+DP化)」という詳細な実装計画を受け取りました。この計画を実装に移す前に、コード側で2点を独自に検証したところ、計画の前提に関わる発見がありました。実装コードを書く前の「計画修正」段階として、この2点を踏まえて計画をどう調整すべきか助言してください。

# 受け取った計画の要点(実装はまだしていません)

- Phase 0: `update_path_constraints`の入出力を記録・再生する等価性回帰ハーネスを作成(障害物なし/1点2分断/複数点2〜4分断/全segmentゼロのフォールバック/マップ端、の5パターンをゴールデン化)
- Phase 1: `_compute_free_segments`(ピクセル走査)と組み合わせ探索(`itertools.product`+`has_collision_in_line`)を別々に計測する`[PERF-CORRIDOR]`ログを追加
- Phase 2: `map.data[y, x]`参照における境界外インデックス(負のインデックスがnumpyで配列の反対側を静かに読んでしまう問題)を修正する「境界安全の修正」
- Phase 3-1: 静的な`line_aa`結果を初期化時に事前計算し、毎周期のピクセル走査をNumPyのgather(`map_data[y_arr, x_arr]`)+ラン検出(`np.diff`/`np.flatnonzero`)に置き換える(既存のPythonループのセマンティクスを厳密に再現)
- Phase 3-2: 現行の`itertools.product`による全組み合わせ列挙(最悪 segment数^5通り、各組み合わせで最大5回`has_collision_in_line`)を、レイヤーグラフ上の動的計画法(DP、O(D×S²))へ置き換える。目的関数(各層で選んだsegmentの幅の合計最大化、隣接層間の中点線が衝突する遷移は禁止)は現行と厳密に同一とする

## 制約(元の依頼から)
- コリドー計算の出力(lb/ub)を1ビットも変えないこと(Phase 2の境界クリップによる差分は意図的な例外)
- 前回周期のキャッシュ・差分更新はしない、numba/Cython等の依存追加はしない、探索深さ5・N・目的関数は変更しない

# 私が独自に検証した2つの発見

## 発見1: Phase 2の前提(境界外インデックスのバグ)が誤りである可能性

`core/map.py`の`w2m()`を確認したところ、既に`np.clip`で範囲内へクランプされていました:

```python
def w2m(self, x, y):
    """World2Map. Transform coordinates from global coordinate system to map coordinates."""
    dx = int((x - self.origin[0]) / self.resolution + 0.5)
    dy = int((self.height - 1) - (y - self.origin[1]) / self.resolution + 0.5)
    dx = np.clip(dx, 0, self.width - 1)
    dy = np.clip(dy, 0, self.height - 1)
    return dx, dy
```

`_compute_free_segments`・`has_collision_in_line`のいずれも、`map.data[y, x]`へアクセスする前に必ず`self.map.w2m(...)`を経由しています:

```python
# _compute_free_segments内
ub_p = self.map.w2m(wp.static_border_cells[0][0], wp.static_border_cells[0][1])
lb_p = self.map.w2m(wp.static_border_cells[1][0], wp.static_border_cells[1][1])
x_list, y_list, _ = line_aa(ub_p[0], ub_p[1], lb_p[0], lb_p[1])  # 両端点は既にクランプ済み

# has_collision_in_line内
def has_collision_in_line(map, p0, p1):
    p0m = map.w2m(p0[0], p0[1])  # ここでもクランプされる
    p1m = map.w2m(p1[0], p1[1])
    x_list, y_list, _ = line_aa(p0m[0], p0m[1], p1m[0], p1m[1])
    occupied_indices = map.data[y_list, x_list] == 0
    return bool(np.any(occupied_indices))
```

`line_aa`は既にクランプされた2点間を補間するだけなので、中間ピクセルも両端点のバウンディングボックス内(=範囲内)に収まるはずです。つまり**「境界外インデックスが配列の反対側を静かに読む」という問題は、現状のコードでは発生し得ないのではないか**、というのが私の見立てです。

## 発見2: Phase 3-2(DP化)のタイブレーク一致という最大のリスク

現行の`calculate_combination_total_segment_length`は、衝突を検知すると即座に`-1000000.0`を返します:

```python
def calculate_combination_total_segment_length(index_combination, ub_pw, lb_pw):
    total_segment_length = 0.0
    for i, segment_index in enumerate(index_combination):
        ub_fs, lb_fs = free_segments_hor[n+i][segment_index]
        mean_prev = (np.array(ub_pw) + np.array(lb_pw)) / 2.
        mean_fs = (np.array(ub_fs) + np.array(lb_fs)) / 2.
        if has_collision_in_line(self.map, mean_prev, mean_fs):
            return -1000000.0  # 衝突した組み合わせは全て同点(-1000000.0)になる
        total_segment_length += dist(ub_fs[0], ub_fs[1], lb_fs[0], lb_fs[1])
        ub_pw, lb_pw = ub_fs, lb_fs
    return total_segment_length

# 呼び出し側: 全組み合わせのスコアからPythonのmax()で最良を選ぶ
```

**全ての組み合わせが衝突する(=完全に道が塞がれている)場合**、全候補が`-1000000.0`で同点になり、`itertools.product`の列挙順(=各層のsegmentインデックスの辞書式順序)に依存して「どれが選ばれるか」が決まります(Pythonの`max()`は最初に見つかった最大値を保持し、同値では更新しないため、実質的に列挙順の先頭が勝つ)。

このプロジェクトはこれまでの分析で、渋滞により道が完全に塞がれるシーン(既知のwp279-285クラスタ等)でのSTUCK復帰・衝突判定に多くの時間を費やしてきました。**DP化がこの「全滅時のタイブレーク」を厳密に再現しそこなうと、まさに一番危険な「完全封鎖」の場面でのコリドー選択が(気づきにくい形で)変わってしまうリスクがある**と考えています。

# 相談したいこと

1. 発見1について、私の検証(w2m()が既にクランプ済み)は正しいでしょうか。見落としている経路(例えば`static_border_cells`自体が`w2m`を経由しない別の計算式で境界外の値を持ち得る可能性等)はないか、改めて確認してほしいです。もし発見1が正しければ、Phase 2は「境界外バグの修正」ではなく「この不変条件(w2mで必ずクランプされること)を保証する回帰テストの追加」へ縮小すべきだと考えていますが、この判断は妥当でしょうか。

2. 発見2(タイブレーク一致リスク)について、DP実装がこの「全滅時の列挙順依存タイブレーク」を厳密に再現するための、より具体的な実装方針(疑似コードレベル)を提案してほしいです。例えば「Python `max()`は同値のとき最初の要素を保持する」という性質を、DPの遷移選択でどう模倣すべきか(遷移候補を`itertools.product`と同じ列挙順で構築し、同値のときは番号が若い方を優先する、等)。

3. 発見2に関連して、回帰ハーネス(Phase 0)に追加すべき最重要テストケースとして「全レイヤーで衝突する(完全封鎖)」パターンを具体的にどう構成すべきか(合成occupancy grid・waypoint配置の設計)、提案してほしいです。

4. その他、この2つの発見を踏まえて計画全体で見直すべき点があれば指摘してください。

# 出力形式

各質問に対して具体的な技術的見解を述べてください。最後に、Phase 2・Phase 3-2の修正版の方針を簡潔にまとめてください。
