# opp_lat_pred根本修正 + 並走ガード + 離脱意味論 設計書(task#295/306統合対応)

**ステータス**: 設計完了、実装前。外部AIレビュー待ち。
**根拠**: `predictive_control_overtake_development_plan_20260805.md` 14-19節、
`stage15_perf_20260707.html` 306節続報3。

## 0. 対象とする3つの不具合、および設計方針

1. **Fix A(根本原因)**: `opp_lat_pred`(対象車横方向速度の外挿予測)が、
   V2X(~13Hz)によって階段状に据え置かれる位置を、40Hz固定dtで単純差分
   しているため、V2Xパケット着弾周期に一致して真の速度の約40/13≈3.1倍の
   スパイクを生む(19節で確定、コード直読で検証済み)。
2. **Fix B(症状対策・空間優先)**: 並走中(縦方向オーバーラップ中)は、
   `opp_lat_pred`のノイズに関わらずoffsetを必要クリアランス未満へ縮小
   させない。ただしコリドー実測(壁)は常に優先する。
3. **Fix C(症状対策・離脱の意味論)**: giveup判定が成立した瞬間、
   `_ot_side`を即座に0へスナップして`_reset_ot_offset_state()`で
   `lateral_target=0.0`まで即時ゼロ化する現行実装は、並走中に発火すると
   相手へ向けて横に引き戻す形になり衝突を招く(18節の事例)。並走中の
   非緊急giveupは、縦方向の車間が空くまで離脱を保留する。

**設計方針(CLAUDE.md §1.3の慎重領域を踏まえて)**:
- switchback/rescue判定・`cleared`判定そのものには一切触れない(82/83節の
  教訓、既存のガード追加=重大リグレッション前例を厳格に踏まえる)。
- footprint_risk・LAT-TTC強制giveup(`force_giveup`)等の**緊急系は
  Fix B/Cの対象外**とし、現行の即時挙動を完全に維持する。
- 保留(Fix C)には必ず上限周期数のフェイルセーフを設け、無期限の保留を
  禁止する。
- 同じ動作を複数箇所で書く場合は共通ヘルパーへ集約する(下記2.3/2.4)。

---

## 1. Fix A: opp_lat_predの根本修正

### 1.1 現状(問題箇所)

```python
# _c_lat = Frenet変換された対象車横位置(_scan_traffic内で計算、変更なし)
if (self._ot_opp_lat_prev is not None
        and self._ot_opp_lat_prev_vid == self._ot_target_vid):
    dt = 1.0 / self._cfg.mpc.control_rate  # 常に25ms固定
    raw_lat_vel = (opp_lat_now - self._ot_opp_lat_prev) / dt
    raw_lat_vel = clamp(raw_lat_vel, -2.0, 2.0)
    ema += 0.05 * (raw_lat_vel - ema)  # EMA平滑化
self._ot_opp_lat_prev = opp_lat_now
self._ot_opp_lat_prev_vid = self._ot_target_vid
opp_lat_pred = opp_lat_now + ema * t_reach
```

`opp_lat_now`(=`_c_lat`)の元データ`cx, cy`は
`tracker.predict_positions(vid, [0.0])`(`v2x_vehicle_tracker.py:124`)から
取得されるが、これは`x_last + vx*0 = x_last`——**V2Xの直近受信位置を
そのまま返すだけで、補間・外挿を一切行わない**。`x_last`は`update()`
(V2Xコールバック、~13Hz)が呼ばれた時だけ変わるため、40Hzで呼ばれる
`_scan_traffic()`から見ると`cx, cy`(ひいては`opp_lat_now`)は約3周期に
1回だけ変化する階段状の値になる。差分計算が常時25ms固定dtを使うため、
値が変化した周期だけ「本来77ms(13Hz周期)かけて動いた分」を25msで割り、
速度が約3.1倍に水増しされる。

### 1.2 修正方針

対象車位置を**自前で微分しない**。既に`v2x_vehicle_tracker.py`が窓端点差分
(既存の`speed_window`平滑化、`update()`内、コメント「2サンプル差分は
27km/h級のスパイクを出す(v2x ~13Hz)。窓端点差で平滑化」)で頑健に速度
推定済みであり、`_scan_traffic()`自身も`v_long`計算で既に
`tracker.velocity(vid)`を使っている(mpc_controller.py:3416-3417)。
`opp_lat_pred`もこの既存の信頼できる速度推定を再利用し、横方向へ射影する
だけにする。

### 1.3 新規ヘルパー

```python
def _estimate_opp_lateral_velocity(self, vid, wp) -> Optional[float]:
    """対象車の横方向速度を、V2Xトラッカーの既存の平滑速度推定(窓端点差分、
    v2x_vehicle_tracker.py)から、対象車自身の最近傍waypoint方位角基準で
    横方向へ射影して求める。自前の位置差分(旧実装、19節で確定した13Hz
    階段状データの誤差分)は行わない。_scan_traffic()のv_long計算
    (mpc_controller.py:3416-3417)と同一のtracker.velocity()・同一の
    回転行列を再利用する(新規の速度源を追加しない)。"""
    tracker = getattr(self, "_v2x_tracker", None)
    if tracker is None:
        return None
    vx, vy = tracker.velocity(vid)
    return -math.sin(wp.psi) * vx + math.cos(wp.psi) * vy
```

### 1.4 呼び出し側の変更

現行の`opp_lat_pred`計算ブロック(mpc_controller.py:6864-6929付近)を、
「`_c_lat`(現在横位置)+`_estimate_opp_lateral_velocity()`(横速度)×
`t_reach`(外挿時間、既存の予測ホライズン計算は無変更)」へ置き換える。
`self._ot_opp_lat_prev`/`_ot_opp_lat_prev_vid`/`_ot_opp_lat_vel_ema`/
`_ot_opp_lat_warmup_count`(EMA関連の状態変数4個)は**不要になり削除**
する(自前差分・EMA・ウォームアップ・クランプが丸ごと不要になるため)。
既存の対象車vid切替時リセット処理(4箇所、下記2.4で統合)からも、この4個の
リセット行が不要になり削除できる(副産物として重複コード削減)。

**安全のための残置**: `tracker.velocity()`自体は`v2x_vehicle_tracker.py`の
`_v_max_safety`クランプで既に物理的に妥当な範囲に収まる設計だが、念のため
既存の`self._ot_min_needed_lat_vel_clamp`(既定2.0m/s)を最終防御として
射影後の値にも適用する(縮退防止・保険、新規パラメータは増やさない)。

### 1.5 診断ログへの影響

`opp_wp`/`opp_raw_lat_vel`(2026-08-05追加の診断フィールド)は、旧実装の
自前差分の生値を見るためのものだったため、Fix A実装後は意味が変わる。
`opp_raw_lat_vel`は新しい`_estimate_opp_lateral_velocity()`の返り値
(クランプ前)を出力するよう用途を継続する(フィールド名・位置は維持、
中身が「V2Xトラッカー速度の射影」に置き換わる)。

---

## 2. Fix B: 並走中オフセット床ガード

### 2.1 縦オーバーラップ判定(共有ヘルパー、新規)

```python
def _update_overlap_state(self, opp_ds_now: Optional[float]) -> bool:
    """対象車と縦方向にオーバーラップ中(=並走中)かをヒステリシス付きで判定する。
    Fix B(オフセット床)・Fix C(離脱保留)の両方から呼ばれる共通判定
    (同一動作を2箇所に重複実装しない)。footprint_risk判定
    (mpc_controller.py:6071、abs(fwd_ds)<along_min_length)と同じ物理的
    下限(along_min_length=2.00m、カート全長)を再利用し、新規の距離
    定数は導入しない。侵入判定(enter)より解除判定(exit)を広く取り、
    境界での毎周期チャタリングを防ぐ(Gemini提案のヒステリシス)。
    データ欠損時(opp_ds_now is None、対象車が一時的に視野外)は、
    直前の状態を維持する(保守側、Claude提案の"鮮度切れ=継続中とみなす")。"""
    if opp_ds_now is None:
        return self._ot_overlapping
    enter_thr = self._along_min_length + self._ot_overlap_margin_m
    exit_thr = self._along_min_length + self._ot_overlap_margin_m * 2.0
    d = abs(opp_ds_now)
    self._ot_overlapping = (d < exit_thr) if self._ot_overlapping else (d < enter_thr)
    return self._ot_overlapping
```

新規config: `overtake.overlap_margin_m`(既定0.5m程度を想定、要チューニング)。
新規状態: `self._ot_overlapping = False`。

### 2.2 オフセット床の適用(共有ヘルパー、新規)

```python
def _apply_overlap_floor(self, target_mag: float, opp_ds_now: Optional[float]) -> float:
    """並走中はtarget_magを縮小させない。既存の168節フリーズ値
    (self._ot_last_valid_target_mag、corr_bound>0だった直近周期のmin(min_needed,
    corr_bound))を床として再利用する(新規の床専用変数を増やさない=非冗長性)。
    コリドー実測(corr_bound)は既にこのフリーズ値へ織り込み済みのため、壁を
    優先する既存の優先順位(168節)はそのまま維持される——本ガードはコリドーの
    上限を上書きしない。"""
    if (self._update_overlap_state(opp_ds_now)
            and self._ot_last_valid_target_mag is not None):
        target_mag = max(target_mag, self._ot_last_valid_target_mag)
    return target_mag
```

### 2.3 呼び出し箇所(2箇所、同一ヘルパーで重複排除)

1. OVERTAKING分岐の`_target_mag`確定直後(mpc_controller.py:6968直前):
   `_target_mag = self._apply_overlap_floor(_target_mag, _opp_ds_now)`
2. STOPPING/proactive-bias分岐(mpc_controller.py:6989直前): 同様に適用。
   ただしこちらは`_stopped_opp`(停止・低速車)向けの小さいバイアスであり、
   対象車のds値が同じ形で手に入るか要実装時確認(手に入らない場合はこの
   分岐は対象外とし、その理由をコメントで明記する)。

---

## 3. Fix C: 並走中の離脱保留(worth=0/giveup時の意味論)

### 3.1 現状(問題箇所)

```python
# mpc_controller.py:6519-6521
_side_blocked = _lat_dec.force_giveup or _room_exhausted
if (self._ot_giveup_count >= self._ot_giveup_cycles or _locked == 0 or _side_blocked):
    ...
    self._ot_state = "STOPPING"
    self._ot_side = 0              # 即座に0へスナップ
    ...
    self._reset_ot_offset_state()  # lateral_target=0.0も即座にゼロ化
```

`_reset_ot_offset_state()`は230節続報で「stale offsetがinfeasibility
カスケードを招く」問題への対処として導入された経緯があり、**単純に
呼び出しを削除・遅延させることはできない**(元の不具合を再発させる)。

### 3.2 修正方針

giveup条件が成立した瞬間、(a)緊急系(footprint_risk・force_giveup由来)
なら**現行どおり即座に**実行する。(b)非緊急(room_exhausted・
opponent_too_fast由来)かつ並走中なら、離脱そのものを**有限時間だけ**
保留し、並走が解消してから通常のgiveup処理(状態遷移・オフセットゼロ化
を含め無変更)を実行する。速度制御は一切新設せず、既存のicc_stop/
lat_ttc系の減速に委ねる(offsetを床(Fix B)で維持したまま自然に車間が
開くのを待つのみ)。

### 3.3 実装(giveup分岐の先頭に追加)

```python
_side_blocked = _lat_dec.force_giveup or _room_exhausted
_giveup_now = (self._ot_giveup_count >= self._ot_giveup_cycles
               or _locked == 0 or _side_blocked)
if _giveup_now and not _lat_dec.footprint_risk_triggered:
    # Fix C: 並走中の非緊急giveupは、車間が空くまで離脱を保留する。
    #   footprint_risk(緊急反応的トリガー)はここで除外し、現行どおり
    #   即座に処理する(82/83節の教訓に基づき、安全反応系の遅延は厳禁)。
    if self._update_overlap_state(_opp_sit.fwd_ds):
        self._ot_pending_disengage_count += 1
        if self._ot_pending_disengage_count < self._ot_pending_disengage_max_cycles:
            _giveup_now = False  # 今回は保留、OVERTAKING継続
    else:
        self._ot_pending_disengage_count = 0
if _giveup_now:
    self._ot_pending_disengage_count = 0
    # ↓ 既存のgiveup処理、無変更
    ...
```

**安全弁(必須)**: `_ot_pending_disengage_count`が
`self._ot_pending_disengage_max_cycles`(新規config、既存の
`_ot_giveup_cycles`の整数倍を既定値とし新規の大きさ感覚を持ち込まない、
例: `_ot_giveup_cycles * 2`≈2秒)に達したら、並走が解消していなくても
**強制的に通常のgiveup処理へ合流する**(無期限保留を禁止)。これは
82/83節の教訓(`cleared`判定周りへの安易なガード追加が重大リグレッション
を招いた)を踏まえた必須のフェイルセーフであり、実装時に省略しない。

### 3.4 状態機械への影響確認

- `_ot_state`は保留中も`"OVERTAKING"`のまま(新規state値を追加しない、
  既存の3値(NORMAL/STOPPING/OVERTAKING)を維持)。
- STUCK検知(`_handle_stuck_recovery`)は`_ot_state`と独立にv≈0を監視する
  既存設計のため、保留中にegoが実際に停止すれば通常どおりSTUCKへ移行
  する(相互干渉なし、要実装時に単体テストで確認)。
- footprint_risk・LAT-TTC C2等の安全系トリガーは保留の影響を受けない
  (3.3で明示的に除外)。
- switchback(側反転、`_lat_dec.side_override`)は保留とは独立の分岐
  (6336行目、giveup分岐より前で評価済み)であり、影響なし。

---

## 4. リセット処理の統合(既存重複の解消+新規状態の追加)

### 4.1 現状の重複

`_ot_last_valid_target_mag = None; _ot_last_valid_min_needed_mag = None;
_ot_opp_lat_prev = None; _ot_opp_lat_prev_vid = None;
_ot_opp_lat_vel_ema = None; _ot_opp_lat_warmup_count = 0`という同一の
6行ブロックが、側反転(6348-6355)・rescue反転(6496-6503)・新規エンゲージ
(6684-6692)・NORMAL復帰(1916-1923)の**4箇所に重複実装**されている。

### 4.2 統合方針

Fix A採用により`_ot_opp_lat_prev`系4変数は削除されるため、重複ブロックは
`_ot_last_valid_target_mag`/`_ot_last_valid_min_needed_mag`の2行のみに
縮小する。ここへFix B/Cの新規状態(`_ot_overlapping`、
`_ot_pending_disengage_count`)を統合し、共有ヘルパーへ切り出す:

```python
def _reset_ot_episode_tracking_state(self) -> None:
    """側変更・新規エンゲージ・OVERTAKING離脱の全ての契機で共通に呼ぶ、
    エピソード単位の追跡状態リセット(4箇所の重複実装を統合)。"""
    self._ot_last_valid_target_mag = None
    self._ot_last_valid_min_needed_mag = None
    self._ot_overlapping = False
    self._ot_pending_disengage_count = 0
```

4箇所の該当行をこの1行呼び出しへ置換する。

---

## 5. config.yaml 新規パラメータ(2個、既定値は要チューニング)

```yaml
overtake:
  overlap_margin_m: 0.5          # 並走(縦オーバーラップ)判定のヒステリシス幅
  pending_disengage_max_cycles: 80  # 離脱保留の上限周期数(既定=giveup_cycles*2目安)
```

---

## 6. 検証計画(CLAUDE.md §1.4準拠)

1. **反実仮想リプレイ(最優先)**: 18節の衝突事例(dev3
   `output/20260805-222055`d1、wp215→233)の入力列(min_needed・
   corr_bound・opp_ds・worth・footprint_risk_triggered)へ、Fix A/B/Cを
   机上で再適用し、wp233時点でoffsetが必要クリアランス以上に保持される
   ことを確認する。**この事例を名前付きの回帰テストフィクスチャとして
   永続化する**(再発防止)。
2. **全OTエピソードへの横展開**: 既存ログ全体(本日の3本のdev3+3本の
   予選ログ)の全OVERTAKINGエピソードへ同じ反実仮想を適用し、
   (a)並走ガード・離脱保留の発動頻度、(b)正常完了エピソードでの
   「余計な保留」件数とその時間コスト(離脱遅延=ラップ損失見込み)を
   集計する。偽陽性コストが小さいことを実装採否の判断材料とする。
3. **単体テスト**: ヒステリシス境界(enter/exit)・footprint_risk免除・
   保留上限フェイルセーフ・STUCK非干渉・4箇所→1関数への統合の等価性。
4. **回帰スイート全件PASS**(既存3156件+新規)。
5. **dev3実地検証(2本以上)**: ガード発動ログ・衝突/STUCK/OT成功率の
   非悪化、Fix Aによる`opp_lat_pred`分布のクランプ張り付き率改善
   (14節の24%基準と比較)。
6. 上記が全てPASSして初めて予選環境投入を検討する(CLAUDE.md §2ルール7、
   予選ログのn=1評価を過信しない)。

---

## 7. 未解決・要レビュー事項

- Fix Bの2箇所目(STOPPING/proactive-bias分岐)へ同一ガードを適用できるか
  (対象車dsの入手可否)は実装時に要確認。
- `overlap_margin_m`/`pending_disengage_max_cycles`の具体的な既定値は
  未チューニング(設計段階の暫定値)。
- Fix C実装方式(新規フラグ+カウンタ vs 既存`_ot_giveup_count`の転用)は、
  外部AI(別Claude)が提案した「低侵襲な方を選ぶ」方針に従い、実装時に
  再検討の余地を残す。
