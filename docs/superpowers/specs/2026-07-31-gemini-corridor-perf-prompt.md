あなたは自動運転レーシングカート(ROS2 Humble / Autoware、spatial MPCで制御、OSQPソルバー)のパフォーマンスエンジニアです。以下の状況について、コード最適化の必要性・方向性を相談したいです。まだ実装前の「検討段階」であり、あなたにコードを書いてもらう必要はありません。分析と提案をお願いします。

# 背景

現在、制御周期を40Hz→72Hzへ引き上げる設計を進めています(理由: 車速を20km/h→36km/hへ引き上げる計画に伴い、対戦車への反応速度[1周期あたりの走行距離]を維持するため)。この変更により、MPCの1回あたりのソルバー計算コストが仮に変わらないとしても、**1秒あたりの総計算回数が1.8倍**になります。

実運用環境(予選環境)での実測(過去のログ分析結果):
- 1周期(現行25ms予算)あたりの平均処理時間: 10.2ms(予算の約40%使用)
- 内訳: コリドー計算(`mpc_corridor`)平均4.01ms(**最重量**) > linearize平均1.32ms > solve平均0.97ms(残り約4msはMPC外の対戦車判断ロジック等)
- 72Hzでは1周期の予算が13.9msとなり、処理時間がほぼ変わらないと仮定すると使用率は約72%まで上昇する見込み

つまり、**現状で既に最も重い「コリドー計算」が、72Hz化によってさらにボトルネックとして重要になる**可能性があります。この計算の最適化が必要か、必要ならどう進めるべきかを相談したいです。

なお、以前(2026-07-07、Stage1.7/R1)、QP求解が実行不可能(infeasible)だった場合の緩和リトライ処理において、リトライのたびにこのコリドー計算を再実行していた(5回×4ms≈20msの追加コスト)バグは既に修正済みで、現在はリトライ時にはマージン差分を算術的に広げるだけで再計算を避けています。**今回相談したいのは、リトライではない通常の1周期あたり1回のコリドー計算そのものの重さ**についてです。

# 該当コード(実際のPython実装、抜粋)

## コリドー計算のエントリポイント(`core/MPC.py`)

```python
def _corridor(self, N, safety_margin):
    """e_yコリドー(lb, ub)。回避ON=動的(占有格子+現姿勢) / OFF=静的テーブル。"""
    rp = self.model.reference_path
    if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
        ub, lb, _ = rp.update_path_constraints(
            self.model.wp_id + 1,
            [self.model.temporal_state.x, self.model.temporal_state.y,
             self.model.temporal_state.psi],
            N, self.model.length, self.model.width, safety_margin)
    else:
        # (静的テーブル参照、軽量)
        ...
    return np.asarray(ub, dtype=float), np.asarray(lb, dtype=float)
```

`use_obstacle_avoidance=True`(通常走行中は常時True)の場合、毎周期1回`update_path_constraints`が呼ばれます。

## `update_path_constraints`(`core/reference_path.py`、抜粋・要約)

このメソッドは、MPCホライズン内のN個(現行N=20程度)のwaypointそれぞれについて:

1. **`_compute_free_segments(wp, min_width)`を呼ぶ**——占有格子(occupancy grid)上で、そのwaypointの左右境界を結ぶ線分を`line_aa`(アンチエイリアス線分ラスタライズ、`skimage.draw`由来)でピクセル単位に走査し、障害物で区切られた「走行可能な区間(free segment)」を検出する:

```python
def _compute_free_segments(self, wp, min_width):
    free_segments = []
    all_segments = []
    ub_p = self.map.w2m(wp.static_border_cells[0][0], wp.static_border_cells[0][1])
    lb_p = self.map.w2m(wp.static_border_cells[1][0], wp.static_border_cells[1][1])
    x_list, y_list, _ = line_aa(ub_p[0], ub_p[1], lb_p[0], lb_p[1])
    ub_o, lb_o = ub_p, ub_p
    free_cells = False
    map_data = self.map.data
    for x, y in zip(x_list[1:], y_list[1:]):  # ← Pythonループでピクセル単位に走査
        cell_value = map_data[y, x]
        if cell_value == 1:
            free_cells = True
            lb_o = (x, y)
        if (cell_value == 0 or (x, y) == lb_p) and free_cells:
            # セグメント確定、world座標へ変換、リストへ追加
            ...
        elif cell_value == 0 and not free_cells:
            ub_o = (x, y)
            lb_o = (x, y)
    # (free_segmentsが空ならall_segmentsから最大幅を採用するフォールバックあり)
    return free_segments
```

2. **複数のfree segmentが見つかった場合(=障害物で道が複数レーンに分断されている場合)**、現在のwaypointから最大5waypoint先まで、各waypointのfree segment候補の**全組み合わせ**(`itertools.product`)を列挙し、それぞれの組み合わせについて隣接waypoint間で`has_collision_in_line`(2点間のピクセル走査による衝突判定)を呼びながら合計距離を計算し、最良の組み合わせを選ぶ:

```python
# n番目のwaypointでfree_segmentsが2つ以上ある場合
free_segments_indices = [[idx for idx in range(len(free_segments))]]
for i in range(n+1, n+5):  # 最大5waypoint先まで
    if i >= N:
        break
    free_segments = free_segments_hor[i]
    if len(free_segments) == 0:
        break
    free_segments_indices.append([idx for idx in range(len(free_segments))])

free_segments_indices_combinations = itertools.product(*free_segments_indices)

def calculate_combination_total_segment_length(index_combination, ub_pw, lb_pw):
    total_segment_length = 0.0
    for i, segment_index in enumerate(index_combination):
        ub_fs, lb_fs = free_segments_hor[n+i][segment_index]
        mean_prev = (np.array(ub_pw) + np.array(lb_pw)) / 2.
        mean_fs = (np.array(ub_fs) + np.array(lb_fs)) / 2.
        if has_collision_in_line(self.map, mean_prev, mean_fs):  # ← ここでも線分ラスタライズ
            return -1000000.0
        total_segment_length += dist(ub_fs[0], ub_fs[1], lb_fs[0], lb_fs[1])
        ub_pw, lb_pw = ub_fs, lb_fs
    return total_segment_length

# 全組み合わせについてcalculate_combination_total_segment_lengthを呼び、最良を選ぶ
```

`has_collision_in_line`自体もピクセル単位のラスタライズです:

```python
def has_collision_in_line(map, p0, p1):
    p0m = map.w2m(p0[0], p0[1])
    p1m = map.w2m(p1[0], p1[1])
    x_list, y_list, _ = line_aa(p0m[0], p0m[1], p1m[0], p1m[1])
    occupied_indices = map.data[y_list, x_list] == 0
    return bool(np.any(occupied_indices))
```

# 分かっていること・分かっていないこと

- 対戦車が多く道が複数に分断されるほど、free segmentの候補数が増え、組み合わせ数(最大 segment数^5程度)と`has_collision_in_line`呼び出し回数が増える構造になっている。渋滞シーンでPERFが悪化しやすいという過去の実測傾向とも整合する
- N(ホライズン長、現行20程度)全waypointについて`_compute_free_segments`が毎周期呼ばれる。ピクセル単位のPythonループ(`line_aa`の結果をfor文で走査)であり、numpyでのベクトル化はされていない
- 平均4.01msという実測値が、上記のどの部分(単純ケース vs 分断ケース)にどれだけ配分されているかの詳細な内訳は取れていない(対戦車が少ない場面では組み合わせ探索自体が発生しないため、平均値は「軽いケースが大半、稀に重いケースがある」分布の可能性が高い)
- MPCのホライズン(N=20程度、waypoint間隔=resolution[m])やこの計算のNは、control_rateとは独立(空間ベース)であり、72Hz化そのものはこの1回あたりの計算コストを変えない。変わるのは「1秒あたりに何回呼ばれるか」のみ

# 相談したいこと

1. **このアルゴリズム自体の計算量評価**: `_compute_free_segments`(ピクセル単位のPythonループ)と組み合わせ探索(最大5waypoint先までの`itertools.product`+ペアワイズ`has_collision_in_line`)について、最悪計算量(free segment数・N・探索深さに対する)を整理してほしい。5waypoint先までという探索深さの制限は妥当か、もっと危険な指数的爆発のリスクはないか。

2. **最も費用対効果の高い最適化はどこか**: (a) `_compute_free_segments`のピクセル走査をnumpyベクトル化する、(b) 組み合わせ探索の枝刈り(悪い部分解を早期に切る)、(c) 前回周期の結果をキャッシュして差分更新する、(d) その他——それぞれの実装難易度とリスク(挙動が変わってしまう可能性)も含めて優先順位をつけてほしい。

3. **72Hz化との兼ね合いでの緊急度判断**: 「まず72Hzを実装して実測し、処理落ちが実際に問題になってから最適化に着手する」のと「先に最適化してから72Hz化に進む」のどちらが合理的か、一般的なパフォーマンス改善の進め方として意見がほしい。

4. **見落としているリスク**: この設計(占有格子+ピクセルラスタライズによる走行可能領域の検出+組み合わせ最適化)自体に、パフォーマンス以外の観点(数値的な脆弱性、境界条件でのバグの入りやすさ等)で気になる点があれば指摘してほしい。

# 出力形式

各項目について「問題なし/軽微な懸念/重大な懸念」ではなく、率直な技術的見解と具体的な提案(疑似コードレベルで構いません)をお願いします。最後に、次にやるべきアクションを1〜3個、優先順位付きで挙げてください。
