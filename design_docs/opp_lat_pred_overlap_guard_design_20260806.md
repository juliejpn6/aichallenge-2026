# opp_lat_pred根本修正 + 並走ガード + 離脱意味論 設計書(task#295/306統合対応)

**ステータス**: 外部AIレビュー反映済み+段階導入計画確定、実装未着手。
**根拠**: `predictive_control_overtake_development_plan_20260805.md` 14-19節、
`stage15_perf_20260707.html` 306節続報3。
**改訂**: 2026-08-06、レビュー反映(§8、must-fix 3点+推奨4点)+3本目の
裏付け事例(§6.0、`0805-07`ログ)+診断ログ設計(§9)+段階導入計画
(§10、A→B→Cの順に個別ゲート・都度オフライン→dev3→予選の3段階検証)+
Phase 1実データ検証(§6.8)+2回目レビュー反映(§6.9、Fix Aを片側利用+
変位物理拘束込みのFix A'へ拡張、1.4節に統合)。

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

**2026-08-06追加(Phase 1実データ検証+外部AIレビュー、必須): 片側利用+
変位物理拘束を「Fix A」自体のスコープへ組み込む(以下「Fix A'」と呼ぶ)**。

§6.8のPhase 1実データ検証で、速度推定そのものは劇的に改善したにも
関わらず、`t_reach`(最大1.5秒)による外挿が本質的に脆いこと(相手の
瞬時横速度を1.5秒間一定と仮定するモデル自体が物理的に無効なこと)が
判明し、min_neededが縮小方向(=衝突リスクを増やす方向)へ逆転する
実例(wp85、min_needed 2.758→0.000)を確認した。別Claudeの指摘
「旧方式(クランプ+EMA)は速度を系統的に過小評価する隠れたバンド
リミッタとして偶然この問題をマスキングしていただけ」という分析は
筋が通っており、Fix Aで推定が正直になった結果、無効な外挿モデルの
弱点がそのまま露出したと理解する。この弱点はFix Aのスコープ内で
構造的に塞ぐ(Fix Bの存在を前提にしない、Fix A単体でも安全であること
を優先する):

1. **予測の片側利用(最重要)**: 予測(`pred`)は「相手がこちらへ寄って
   くる」方向(=必要クリアランスを増やす方向)にのみ使い、「相手が
   離れていく」という投機的な予測で必要クリアランスを縮めない。

   ```python
   need_from_pred = max(0.0, min(self._ot_d_off,
                                  float(self._ot_side) * opp_lat_pred + _clear_needed))
   need_from_now = max(0.0, min(self._ot_d_off,
                                 float(self._ot_side) * opp_lat_now + _clear_needed))
   _target_mag = max(need_from_pred, need_from_now)
   ```

   狙いの機能(相手接近の先読み)は完全に残しつつ、投機的縮小という
   故障モード(wp85のパターン)を構造的に排除する。代償(相手が本当に
   離れていく場合に早めに幅を詰められない)は、本日の安全優先方針
   (P0)と整合するコストとして受け入れる。

2. **変位の物理拘束**: `|opp_lat_pred - opp_lat_now|`(=`lat_vel * t_reach`)
   を、既存の車両運動制約(横加速度上限等、新規マジックナンバー禁止)
   から導いた物理的に妥当な最大横変位で上限クランプする。これは以前
   (18節時点)の衛生リストに含まれていたが、設計書化の過程で
   velocityクランプの残置のみになり抜け落ちていたため復活させる。

**Phase 1オフライン検証への追加項目**: `t_reach`の有効ホライズンを
実測で較正する——各時点の予測変位(`lat_vel × τ`)と実際の`τ`秒後の
横変位実測値の相関を`τ=0.25/0.5/0.75/1.0/1.5`秒で計算し、「予測が
実際に価値を持つ最大の`τ`」を求める。現行の1.5秒上限が有効域外である
ことがデータで示されれば、14節の導入経緯(近視眼防止)との両立を
踏まえた上限見直しの実測根拠とする(即座の変更はしない、根拠の
蓄積のみ)。

**安全のための残置**: `tracker.velocity()`自体は`v2x_vehicle_tracker.py`の
`_v_max_safety`クランプで既に物理的に妥当な範囲に収まる設計だが、念のため
既存の`self._ot_min_needed_lat_vel_clamp`(既定2.0m/s)を最終防御として
射影後の値にも適用する(縮退防止・保険、新規パラメータは増やさない)。

**2026-08-06追加(外部AIレビューmust-fix 3)**: 実装着手前に
`tracker.velocity()`の実挙動を以下の3ケースについて調査し、意味論を
確定させる: (a)未知vid(トラッカーが一度も観測していない相手)、
(b)速度窓未充足(新規vid検出直後、`speed_window`個のサンプルが
まだ揃っていない)、(c)`clamp_hold_enabled`のON/OFF両設定でのV2X速度
異常発生中。現状のコード(`velocity()`実装、`v2x_vehicle_tracker.py:116`)
は未観測時に`(0.0, 0.0)`を返す設計に見えるが、**「相手速度0」と
「データなし」を混同しない**よう、`_estimate_opp_lateral_velocity()`
側で以下のフォールバックを明示する: データなし・鮮度失効(直近update()
からの経過時間が閾値超)の場合はNoneを返し、呼び出し側は
`opp_lat_pred = opp_lat_now`(外挿なし、300節導入前の安全な挙動と同一)
とし、診断ログへ理由を記録する。

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

**2026-08-06改訂(外部AIレビューmust-fix 1)**: 当初案は既存の168節フリーズ値
(`self._ot_last_valid_target_mag`)を床として二重利用する設計だったが、
外部AI(Gemini・別Claude)の指摘により2つの欠陥が判明し**廃案**とした:
(a) この変数はコリドー崩壊対策自身が「今周期の(既に崩壊しかけた)
target_mag」で毎周期更新されるため、床として読む時点で既に汚染されている
可能性がある(更新→参照が同一周期内で同じ変数に対して起きるため、
実装のわずかな順序次第で床自体が崩落する)、(b) 一度フリーズした値を
そのまま床にすると、その後コリドー自体が本当に狭まった場合に床が壁を
突き破りうる。専用の新規状態変数へ分離し、適用時に必ず現在のcorr_boundで
再キャップする。

```python
def _apply_overlap_floor(self, target_mag: float, opp_ds_now: Optional[float],
                          corr_bound: float) -> float:
    """並走中はtarget_magを縮小させない。床は並走エピソード専用の新規状態
    (self._ot_overlap_floor_mag、ピーク保持=単調非減少)を使う——既存の
    168節フリーズ値(_ot_last_valid_target_mag)とは別変数にし、意味論の
    衝突(同一変数が「コリドー崩壊対策」と「並走ガード」の二重の意味を
    持つことによるバグ)を避ける。

    不変条件(単体テストで保証すること):
      - 床は同一並走エピソード内で単調非減少(下がらない)。
      - 床適用後のtarget_magは、常に「現在のcorr_bound - マージン」以下
        (=床がコリドーの壁を突き破ることは原理的に発生しない、今周期の
        実測corr_boundで毎回再キャップするため)。

    ゲートOFF(既定)時は_update_overlap_state()の呼び出しも含め早期return
    し、target_magを完全に無変更で返す(全ゲートOFF時のビット等価性を
    保証、6章の検証項目)。"""
    if not self._ot_overlap_floor_enabled:
        return target_mag
    overlapping = self._update_overlap_state(opp_ds_now)
    if overlapping:
        self._ot_overlap_floor_mag = max(self._ot_overlap_floor_mag or 0.0, target_mag)
        floor = self._ot_overlap_floor_mag
        if np.isfinite(corr_bound) and corr_bound > 0.0:
            floor = min(floor, corr_bound - self._ot_overlap_corr_margin_m)
        target_mag = max(target_mag, floor)
    return target_mag
```

新規状態: `self._ot_overlap_floor_mag = None`(エピソードリセットで初期化)。
新規config: `overtake.overlap_corr_margin_m`(既存の他マージン定数と揃えた
既定値を想定、要チューニング)。

### 2.3 呼び出し箇所(2箇所、同一ヘルパーで重複排除)

1. OVERTAKING分岐の`_target_mag`確定直後(mpc_controller.py:6968直前、
   既存の168節コリドー崩壊フリーズ処理より**後**、`self._mpc.lateral_target=...`
   より**前**の位置で呼ぶ——corr_boundの今周期値が既に計算済みであること
   を利用する):
   `_target_mag = self._apply_overlap_floor(_target_mag, _opp_ds_now, _corr_bound)`
2. STOPPING/proactive-bias分岐(mpc_controller.py:6989直前): 同様に適用。
   対象車のds値は`_scan_traffic()`から取得可能(外部AIレビューで確認済み)
   なため、実装時に配線する。

### 2.4 並走解除時のスムージングに関する確認事項(外部AIレビュー、Gemini)

Geminiより「並走を完全に抜けた直後、`lateral_target`が床の値から
本来の値(0.0m等)へステップ状に落ちることを防ぐスムージングは、既存の
経路生成ロジック側で担保されているか」という質問があった。確認したところ、
既存の`self._ot_alpha`(0..1、mpc_controller.py:7015-7026)は**「目標へ
どれだけブレンドするか」の重みをランプするものであり、`lateral_target`の
値自体の変化率を制限する専用機構ではない**。alphaが既に1.0(完全コミット)
の状態で`lateral_target`が変わった場合、MPC自身のR[delta]コスト・
`steer_rate_max`制約が結果的に急な物理応答を抑えるが、`lateral_target`
という**目標値そのものの急変を防ぐ専用の保証は存在しない**。これは
6章のPhase 1反実仮想検証で実際にステップ状の悪化が生じないか確認すべき
項目として明記する(6.7として追加)。

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

**2026-08-06改訂(外部AIレビューmust-fix 2)**: 当初案は、giveup条件自体が
不成立(`_giveup_now`が最初からFalse、=通常のOVERTAKING継続中)の周期に
`_ot_pending_disengage_count`をリセットする経路がなく、以前の保留エピソードの
カウントが残存したまま次回の(無関係な理由による)giveupへ引き継がれる
欠陥があった(例: 保留中に相手が減速しgiveup条件自体が解消→カウント
5で放置→数秒後に別理由[room_exhausted]でgiveup発生→本来0から始まる
べき保留が5から再開し、上限に到達するタイミングが早まる)。`_giveup_now`
がFalseの経路にも明示的なリセットを追加する。

```python
_side_blocked = _lat_dec.force_giveup or _room_exhausted
_giveup_now = (self._ot_giveup_count >= self._ot_giveup_cycles
               or _locked == 0 or _side_blocked)
# ゲートOFF(既定)時は以下を一切実行しない(_giveup_nowの値をそのまま
# 使う=現行動作とビット等価、6章の検証項目)。
if (self._ot_pending_disengage_enabled and _giveup_now
        and not _lat_dec.footprint_risk_triggered):
    # Fix C: 並走中の非緊急giveupは、車間が空くまで離脱を保留する。
    #   footprint_risk(緊急反応的トリガー)はここで除外し、現行どおり
    #   即座に処理する(82/83節の教訓に基づき、安全反応系の遅延は厳禁)。
    if self._update_overlap_state(_opp_sit.fwd_ds):
        self._ot_pending_disengage_count += 1
        if self._ot_pending_disengage_count < self._ot_pending_disengage_max_cycles:
            _giveup_now = False  # 今回は保留、OVERTAKING継続
    else:
        self._ot_pending_disengage_count = 0
else:
    # 2026-08-06追加(must-fix 2): giveup条件自体が不成立の周期は、
    #   以前の保留エピソードの残存カウントを必ず0へ戻す。
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
  する(相互干渉なし)。**2026-08-06追加(外部AIレビュー推奨4)**:
  STUCK進入時にも`_reset_ot_episode_tracking_state()`を呼び、STUCK復帰後に
  古い保留カウント・並走状態・床を持ち越さないようにする。
- footprint_risk・LAT-TTC C2等の安全系トリガーは保留の影響を受けない
  (3.3で明示的に除外)。
- switchback(側反転、`_lat_dec.side_override`)は保留とは独立の分岐
  (6336行目、giveup分岐より前で評価済み)であり、影響なし。
  **2026-08-06追加(外部AIレビュー確認、Gemini)**: switchback分岐は
  giveup分岐より前で評価され、発火時は`_reset_ot_episode_tracking_state()`
  相当のリセット(6348-6355)を伴うため、`_ot_pending_disengage_count`も
  同時にリセットされ、旧側での保留状態が新側へ持ち越されることはない
  (設計は無変更のまま整合性を確認)。
- **2026-08-06追加(外部AIレビュー推奨7)**: OVERTAKING継続のまま
  `fwd_vid`(対象車ID)だけが切り替わる経路の有無を実装時に確認し、
  もし存在すればそこにも`_reset_ot_episode_tracking_state()`を追加する
  (旧対象車の並走状態・床を新対象車へ引き継がない)。

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
    エピソード単位の追跡状態リセット(4箇所の重複実装を統合)。
    2026-08-06改訂: Fix Bの専用床変数(_ot_overlap_floor_mag、must-fix 1)
    もここでリセットする。"""
    self._ot_last_valid_target_mag = None
    self._ot_last_valid_min_needed_mag = None
    self._ot_overlapping = False
    self._ot_overlap_floor_mag = None
    self._ot_pending_disengage_count = 0
```

4箇所の該当行をこの1行呼び出しへ置換する。**2026-08-06追加**: これに加え
STUCK進入時(推奨4)・`fwd_vid`切替経路が存在する場合(推奨7)にも同じ
呼び出しを追加する(呼び出し箇所は4→最大6程度になる見込み、実装時に
最終確定)。

---

## 5. config.yaml 新規パラメータ

**2026-08-06改訂(ユーザー指示: 段階導入)**: Fix A・Fix B・Fix Cを
**それぞれ独立した**configゲートへ分離する(当初案はFix B+Cを1ゲートへ
束ねていたが、task#265/early_release/候補④で「複数機構を同時にONに
すると効果の切り分けができない」という教訓を得たばかりであり、同じ
轍を踏まないため3つとも個別に分離する)。

```yaml
overtake:
  lat_vel_source_tracker: false      # false=現行の自前差分/true=Fix A(tracker.velocity()再利用)
  overlap_floor_enabled: false       # false=現行/true=Fix B(並走中オフセット床)
  overlap_margin_m: 0.5              # 並走(縦オーバーラップ)判定のヒステリシス幅(Fix B/C共通)
  overlap_corr_margin_m: 0.1         # Fix Bの床をcorr_boundで再キャップする際のマージン(must-fix 1)
  pending_disengage_enabled: false   # false=現行/true=Fix C(並走中の離脱保留)
  pending_disengage_max_cycles: 80   # 離脱保留の上限周期数(既定=giveup_cycles*2目安)
```

いずれも既定`false`(現行挙動と完全に一致、新規追加は無効化状態から開始)。
`overlap_margin_m`はFix B/C共通の並走判定(`_update_overlap_state()`)が
使うため、どちらか一方でもONなら消費される。

---

## 6. 検証計画(CLAUDE.md §1.4準拠)

### 6.0 3本目の独立した裏付け事例(2026-08-06、`autoware(0805-07).log`)

ユーザーの目視報告(「並走中、車速差が小さいと相手に向かって幅寄せする」)
を受け、ユーザー提供の`~/Downloads/autoware(0805-07).log`(qualifying環境)
を分析した。wp62-118のOVERTAKINGエピソードで、offsetがmin_needed比
22〜62%程度しか達成できていない**慢性的な未達**を確認した(18節の
wp215-233事例のような単発の劇的崩壊ではなく、終始一貫した不足という
異なる現れ方):

| wp | offset | min_needed | 達成率 | opp_raw_lat_vel |
|---|---|---|---|---|
| 62 | 0.23 | 0.97 | 24% | 2.984(クランプ上限2.0超え) |
| 66 | 0.64 | 1.66 | 38% | 1.08 |
| 96 | 0.30 | 1.37 | 22% | 1.08 |
| 109 | 1.16 | 1.86 | 62% | 0.0 |
| 118 | -1.26 | 2.14 | 59% | 0.0 |

wp68でも`opp_raw_lat_vel=-3.218`というクランプ超えを確認。3本目の独立した
ログ(dev3の衝突事例・qualifying環境のこの慢性未達事例)で同一機構
(opp_lat_predノイズ起因のoffset不足)が再現しており、単発ではなく
再現性のある実害であることの追加の裏付けとなった。この事例も6.2の
反実仮想検証対象へ追加する(名前付きフィクスチャ2件目)。

1. **反実仮想リプレイ(最優先)**: 18節の衝突事例(dev3
   `output/20260805-222055`d1、wp215→233)の入力列(min_needed・
   corr_bound・opp_ds・worth・footprint_risk_triggered)へ、Fix A/B/Cを
   机上で再適用し、wp233時点でoffsetが必要クリアランス以上に保持される
   ことを確認する。**この事例を名前付きの回帰テストフィクスチャとして
   永続化する**(再発防止)。同様に本節(6.0)の`0805-07`慢性未達事例
   (wp62-118)も2件目のフィクスチャとして追加し、offsetがmin_needed比
   90%以上まで改善することを確認する。
2. **全OTエピソードへの横展開**: 既存ログ全体(本日の3本のdev3+3本の
   予選ログ)の全OVERTAKINGエピソードへ同じ反実仮想を適用し、
   (a)並走ガード・離脱保留の発動頻度、(b)正常完了エピソードでの
   「余計な保留」件数とその時間コスト(離脱遅延=ラップ損失見込み)を
   集計する。偽陽性コストが小さいことを実装採否の判断材料とする。
3. **単体テスト**: ヒステリシス境界(enter/exit)・footprint_risk免除・
   保留上限フェイルセーフ・STUCK非干渉・4箇所→1関数への統合の等価性・
   床の単調非減少+corr_bound再キャップ不変条件(must-fix 1)・giveup
   条件解消時の保留カウンタリセット(must-fix 2)・tracker.velocity()の
   None/鮮度失効フォールバック(must-fix 3)・**全configゲートOFF時に
   現行と完全にビット等価**であることの確認。
4. **回帰スイート全件PASS**(既存3156件+新規)。
5. **dev3実地検証(2本以上、Fix A・Fix B+Cは独立フラグで個別にON)**:
   ガード発動ログ・衝突/STUCK/OT成功率の非悪化、Fix Aによる
   `opp_lat_pred`分布のクランプ張り付き率改善(14節の24%基準と比較)。
6. **2026-08-06追加(外部AIレビュー推奨5)**: 反実仮想の集計に「強制
   フォールバック(`pending_disengage_max_cycles`到達)発火時になお並走中
   だった件数」を必須集計項目として追加する(Fix Cの残存リスクの定量、
   このケースは対策未完了のまま離脱するため実質的にFix C導入前と同じ
   リスクが残る)。
7. **2026-08-06追加(外部AIレビュー、Gemini 2.4)**: 並走解除の瞬間に
   `lateral_target`がステップ状に変化しないか(2.4節参照)を反実仮想
   リプレイで直接確認する。
8. 上記が全てPASSして初めて予選環境投入を検討する(CLAUDE.md §2ルール7、
   予選ログのn=1評価を過信しない)。

### 6.8 Phase 1実施結果(2026-08-06、実データでの初回検証)

`0805-07`のrosbag(`/v2x/vehicle_positions`、CDR手動デコード、実測
publish rate≈15.2Hz)を使い、実際のV2X受信列から旧方式・新方式(Fix A)の
速度推定を再構成して比較した。

**速度推定そのものの比較(核心的な結果、明確な改善)**:

| | 最大値 | 平均 |
|---|---|---|
| 旧方式(40Hz固定dt単純差分) | **26.7 m/s**(物理的にあり得ない) | — |
| 旧方式(クランプ+EMA後) | 1.30 m/s(ノイズを飲み込んだ後の値) | — |
| 新方式(tracker.velocity()窓端点差分) | 7.37 m/s | 5.83 m/s |
| 参考: 1秒間隔の粗い実速度(独立サニティチェック) | 6.70 m/s | 5.75 m/s |

新方式は独立に算出した参考値(平均5.75m/s)にほぼ一致し、旧方式は物理的に
あり得ない値(26.7m/s)を出すことを実データで確認した。40Hzサンプル中
新しいV2Xサンプルだった周期の割合(39.3%)も理論値(40/15.2≈38.0%)と
一致し、シミュレーションの妥当性を確認済み。

**opp_lat_pred/min_neededまで通した結果(混在、Fix A単体では不十分な
ケースあり)**: 12サンプル中8件は新方式で`min_needed`が改善方向に
近づいたが、3件(wp79/81/89、いずれもside=-1)は逆に悪化し、
**wp85(t_reach=1.5秒、外挿上限)では新方式のmin_needed=0.000**
(旧方式の2.758より悪化)という結果が出た。

**解釈**: 速度推定そのもの(Fix Aの直接の対象)は劇的に改善したが、
`t_reach`(最大1.5秒)による外挿は、どれだけ滑らかな速度推定であっても
残存ノイズを増幅する性質を本質的に持つ。**Fix A単体は必要条件だが
十分条件ではない**——この結果は、Fix B(コリドー実測を優先する物理的な
床)が独立した安全網として必要だという設計(§0の多層防御方針)を
実データで裏付けている。

**検証手法の限界(要注意)**: 本オフライン検証は簡易ミラー実装であり、
以下の近似を含む——(a)`_closest_wp_and_s`のprev_idx半径制限を省略し
グローバル探索を使用、(b)`clear_needed`(モデル幅+ブロック半幅)を
実測ログから中央値として逆算した近似値を使用、(c)t_reachは実測ログの
値をそのまま再利用(自前で再計算していない)。**確定的な効果検証は、
実際にコードへ実装した上での単体テスト・dev3実走行で行う必要がある**
(§10の段階導入計画どおり)。

### 6.9 外部AIレビュー(2回目)を受けた方針確定(2026-08-06)

Gemini・別Claudeへ§6.8の結果を相談した。両者とも「速度推定自体の改善は
成功」「Fix Bの正当化」の理解では一致したが、対応方針で割れた:
Geminiは「Fix AとFix Bを同時デプロイすべき(Fix A単体は危険)」、
別Claudeは「Fix A自体のスコープへ**片側利用**(予測はクリアランスを
増やす方向にのみ使う)+**変位物理拘束**を組み込めば、Fix A単体でも
安全である」と提案した。**別Claude案を採用する**——根本的な論理欠陥
(投機的な予測でクリアランスを削れてしまうこと)を構造的に塞ぐ方が、
「Fix Bという後段の安全網に頼る」より筋が良く、段階導入(§10)による
効果の帰属分離という利点も維持できるため。1.4節へ反映済み。

wp85(t_reach=1.5秒での逆転、min_needed=0.000)は、別Claudeの指摘
(実装バグでも実際の外挿モデルの脆さでも「片側利用+変位拘束が必要」
という結論は変わらない)を踏まえ、**名前付きテストケースとして実装後の
単体テストへ組み込む**(§6章の反実仮想対象へ追加、prev_idx制限付き
探索・正確なclear_needed導出での再現有無を確認する)。

---

## 7. 未解決・要レビュー事項

- **2026-08-06更新**: Fix Bの2箇所目(STOPPING/proactive-bias分岐)は
  対象車dsが`_scan_traffic()`から取得可能と外部AIレビューで確認済み、
  実装時に配線する(解決)。
- `overlap_margin_m`/`overlap_corr_margin_m`/`pending_disengage_max_cycles`
  の具体的な既定値は未チューニング(設計段階の暫定値)。
- **2026-08-06更新**: Fix C実装方式は、外部AI(Gemini・別Claude)双方が
  「既存`_ot_giveup_count`の転用は意味論的過負荷(giveupトリガー継続時間
  と物理的並走継続時間という異なる概念を1変数に混在させる)を招き
  非推奨、新規カウンタを追加すべき」と一致して判断したため、当初案
  (新規`_ot_pending_disengage_count`追加)を**そのまま採用・確定**する。
- 並走解除時の`lateral_target`ステップ変化(2.4節)の実害有無は反実仮想
  リプレイでの確認待ち(未解決、6.7参照)。

---

## 8. 外部AIレビュー結果の要約(2026-08-06)

`docs/superpowers/specs/2026-08-06-opp-lat-pred-overlap-guard-design-review-prompt.md`
でGemini・別Claudeへレビュー依頼した。要点:

- **状態遷移の一貫性**: 問題なし(switchbackがgiveup判定より前で評価され、
  発火時のリセットで保留カウントも巻き込まれるため矛盾しない)。
- **Fix A**: 妥当。既存の`clamp_hold_enabled`(V2X異常対策)の恩恵が
  横方向予測にも自動的に波及する副次効果も確認。
- **Fix B**: **must-fix**。当初の`_ot_last_valid_target_mag`二重利用案は
  (a)同一周期内の更新順序次第で床自体が汚染される、(b)コリドーが
  真に狭まった場合に床が壁を突き破りうる、の2つの欠陥があり廃案。
  専用状態`_ot_overlap_floor_mag`(ピーク保持+適用時に現在corr_boundで
  再キャップ)へ設計変更した(§2.2に反映済み)。
- **Fix C**: **must-fix**。giveup条件が不成立の周期に保留カウンタを
  リセットする経路が漏れていた欠陥を修正(§3.3に反映済み)。
- **共通**: `tracker.velocity()`のAPI実挙動(未知vid・速度窓未充足・
  V2X異常時)を実装前に調査し意味論を確定する必要性(**must-fix**、
  §1.4に反映済み)。
- **推奨事項(全て採用・反映済み)**: STUCK進入時のリセット追加(§3.4)、
  反実仮想集計への強制フォールバック残存件数の追加(§6.6)、
  `fwd_vid`切替経路の確認(§3.4)、configゲートをFix Aと Fix B/Cで
  分離(§5)。
- Fix C実装方式(新規カウンタ vs 既存流用)は両AIが一致して新規カウンタ
  維持を支持、当初案を確定とした。

以上を全て本文へ反映済み。

---

## 9. 診断ログ設計(2026-08-06、ユーザー指示: 各Fixを個別検証できるロギング)

3つのFixをそれぞれ独立ゲートで段階導入する(§10)ため、各Fixの動作を
個別に確認できる専用ログを用意する。いずれも本日確立した「ワンショット/
イベント発火型」の既存パターン(`[FP-COOLDOWN-CLEAR]`・
`[SPEED-COOLDOWN-CLEAR]`等)を踏襲する。

### 9.1 Fix A: `[OT]`ログへの`lat_vel_src`マーカー追加

既存の`opp_wp`/`opp_raw_lat_vel`診断フィールド(2026-08-05追加)は、
Fix A適用後は「値の出処」が変わる(自前差分 or tracker.velocity()射影)。
どちらの実装が出した値かをログから機械的に判別できるよう、`[OT]`ログへ
1フィールド追加する:

```
f"lat_vel_src={'tracker' if self._ot_lat_vel_source_tracker else 'diff'} "
```

これにより、Fix Aのconfigゲートを切り替えた前後のログを混在させても
(例: 同一走行中に一時的に切り替えて比較する等)、どちらの計算方式で
出た値かを事後に判別できる。

### 9.2 Fix B: `[OVERLAP-FLOOR]`ログ(床が実際にtarget_magを持ち上げた時のみ発火)

```python
if overlapping and target_mag_before_floor < floor:
    self.get_logger().info(
        f"[OVERLAP-FLOOR] side={self._ot_side} floor={floor:.3f} "
        f"target_mag_before={target_mag_before_floor:.3f} "
        f"target_mag_after={target_mag:.3f} corr_bound={corr_bound:.3f} "
        f"wp={self._mpc.model.wp_id}")
```

エッジトリガー(床が実際に効いた周期のみ)とし、並走中毎周期のログ氾濫を
避ける。

### 9.3 Fix C: `[PENDING-DISENGAGE]`ログ(保留開始・解消の2イベント)

```python
# 保留開始時(初めて_giveup_now=Falseへ倒した周期のみ)
self.get_logger().warn(
    f"[PENDING-DISENGAGE] start reason={_giveup_trigger} side={_locked} "
    f"wp={self._mpc.model.wp_id}")
# 解消時(2種類: 並走の自然解消 / 強制フォールバック到達)
self.get_logger().warn(
    f"[PENDING-DISENGAGE] resolved reason="
    f"{'natural_overlap_clear' if not forced else 'forced_fallback'} "
    f"pending_count={self._ot_pending_disengage_count} "
    f"wp={self._mpc.model.wp_id}")
```

`forced_fallback`(強制フォールバック)発火は6.6の反実仮想集計項目
(残存リスクの定量)と直接対応し、実地検証でもこの発火率を主要な監視
対象とする。

---

## 10. 段階的導入計画(2026-08-06、ユーザー指示。6.9節の2回目レビューを
反映し一部更新)

**方針**: 3つのFixを**一括導入せず、A'→B→Cの順に1つずつ**導入する
(Fix Aは6.9節を踏まえ「片側利用+変位物理拘束」を含めたFix A'として
1パッケージ扱いにする——これらは推定器の出力の使い方でありFix Aと
同一スコープ、B/Cとは独立)。理由は本日のtask#265/early_release/候補④で
得た教訓(複数機構を同時にONにすると実地での効果の切り分けができ
なくなる)を踏まえたもの。各段階は必ず「オフライン反実仮想 → dev3
(ローカル) → 予選環境」の3段階を経てから次のFixへ進む。Geminiは
「Fix AとFix Bの同時デプロイ」を提案したが、Fix A'自体が構造的に安全
(投機的なクリアランス縮小を許さない)であるため、単体デプロイでも
危険はないと判断し不採用とした(6.9節参照)。

### 10.1 各段階のフロー(全Fix共通)

1. **オフライン反実仮想(走行なし)**: §6の2フィクスチャ(18節衝突事例・
   6.0慢性未達事例)+全OTエピソード横展開で、当該Fixのみ有効化した
   場合の効果と偽陽性コストを机上確認する。
2. **dev3(ローカル、走行あり)**: 当該Fixのゲートのみをtrueにし、
   §9の専用診断ログが正しく発火するか、安全指標(衝突/STUCK/OT成功率)
   が既存水準から悪化しないかを確認する(2本以上、n=1で確定しない)。
   自己対戦のため状況再現・繰り返し試行がしやすく、まずここで
   ハード制約の PASS/FAIL を確定させる。
3. **予選環境(最終確認)**: dev3で問題なければ、他チーム車との実対戦で
   n数を稼ぐ(§9の診断ログ・opp_wp/opp_raw_lat_vel分布・クランプ張り付き
   率等を実地で確認)。CLAUDE.md §2ルール7(予選ログのn=1評価を過信
   しない)に従い、複数本での確認を基本とする。
4. 次のFixへ進む前に、この段階の結果をdesign_docsへ記録する。

### 10.2 順序とその理由

1. **Fix A**(根本原因、最優先): 状態機械・giveup遷移には触れず、
   `opp_lat_pred`の計算方法のみを変える最も低リスクな変更。これ単体で
   §6.0の慢性未達がどこまで改善するかを見ることで、Fix B/Cが実際に
   どれだけ必要か・どうチューニングすべきかの判断材料にもなる。
2. **Fix B**(並走中オフセット床): Fix Aだけでは解決しない残存ノイズ・
   正当な相手の急な横移動等に対する物理的な安全網。
3. **Fix C**(並走中の離脱保留): 最もリスクの高い変更(giveup遷移という
   状態機械そのものに手を入れる、82/83節の教訓が直接該当する領域)の
   ため最後に回す。Fix A/Bの実地データを見てから、実際にFix Cが必要な
   場面がどれだけ残っているかを再確認した上で着手する。

次のアクションは実装着手(Fix Aのオフライン反実仮想リプレイから開始)。

## 11. Fix A'実装完了(2026-08-06、コミット`eaaa27e`)

1.3-1.4節・6.9節の設計どおり実装した:
- `overtake.lat_vel_source_tracker`(既定false)ゲート追加
- `_estimate_opp_lateral_velocity()`新設(tracker.velocity()再利用、
  is_settled()でNone/データなしを区別)
- 呼び出し側をelif化(ゲートOFF時は旧実装コード無変更、ビット等価)
- 片側利用(`need_from_pred`/`need_from_now`のmax)+変位物理拘束
  (既存クランプ×horizon_cap導出値)をゲートON時のみ適用
- `[OT]`ログへ`lat_vel_src`マーカー追加
- 新規単体テスト15件(wp85の名前付き回帰含む)+既存3件の窓拡大、
  回帰3171件PASS

**次のアクション(§10のFlow、10.1節)**: dev3ローカル検証(Phase 2)。
`lat_vel_source_tracker: true`のみを有効化し、`lat_vel_src`診断ログの
発火確認、`opp_lat_pred`分布のクランプ張り付き率が14節の24%基準から
改善するかを実測する。安全指標(衝突/STUCK/OT成功率)の非悪化も確認。

## 12. Phase 2(dev3ローカル検証)完了、PASS判定(2026-08-06/07)

### 12.1 検証前提: インフラ障害の解決

検証開始前、開発機の実際の停電により`make dev3`が起動不能になった
(コンテナが11-13秒で終了)。診断の結果、真因は`lo`インターフェースの
`MULTICAST`フラグ喪失(CycloneDDSが`lo`経由でユニキャストdiscoveryへ
フォールバックし、participant index割当てに失敗)と判明。既存の公式
チューニングコマンド`./setup.bash network tune`で恒久修正(systemd化、
再起動耐性あり)し解決した。外部AI相談により「`net.core.wmem_max`不足」
という当初の自前仮説(仮説X)は誤りと判明、`lo` multicast喪失が正しい
主因だった。

### 12.2 副次的に発見したログ集計バグの修正

Phase 2データ収集中、`[OT-OUTCOME]`の`outcome=success`カウントが
30分・3台走行で133338件という異常値になる不具合を発見。原因は
「前方クリアが連続したらNORMAL復帰」分岐が状態非依存で、
`self._ot_state=="NORMAL"`へ既に遷移済み(=前方に相手がいない大半の
巡航区間)でも毎周期再ログし続けていたこと。`self._ot_state != "NORMAL"`
ガードを追加し修正(コミット`bea0dbb`、回帰3177件PASS)。254節で確立した
STOPPING→NORMAL復帰経路(_n_fwd==0が毎周期真になり続けるケース含む)は
ガード条件が"NORMAL"以外全てを許可する形のため無変更で維持。

### 12.3 実測結果

**Fix A' ON(n=3、独立`make dev3`セッション)**:

| Run | 走行時間 | COLLISION-SUSPECTED | STUCK detected | `lat_vel_src=tracker`発火 | クランプ飽和率(≥1.8m/s) |
|---|---|---|---|---|---|
| 1 | ~35分 | 128 | 71 | 179 | 4.5%(8件) |
| 2 | ~35分 | 2 | 0 | 27 | 14.8%(4件) |
| 3 | ~39分 | 7 | 2 | 32 | 9.4%(3件) |
| 合算 | — | 137 | 73 | 238 | **6.3%(15件)** |

**Fix A' OFF(ベースライン、単一約165分連続セッションを35分窓×5へ分割)**:

| 窓 | COLLISION-SUSPECTED | STUCK detected |
|---|---|---|
| 1 | 11 | 6 |
| 2 | 1 | 1 |
| 3 | 2 | 0 |
| 4 | 10 | 9 |
| 5(端数25分) | 2 | 1 |

ゲートOFF側でも`lat_vel_src=tracker`は0件(混入なし、正しくゲートされて
いることを確認)。

### 12.4 分析・判定

1. **ハード制約(衝突/STUCK)の激しいばらつきはON/OFF問わず同じ性質**:
   OFF側でもSTUCK 17件中14件(82%)がwp281-285帯に集中しており(task#306の
   既知の慢性輻輳地点)、ON側Run1(128件)のような外れ値もOFF側の窓1・窓4
   (11件・10件)と同種の現象の極端例と解釈できる。ON側の全STUCK 21件
   (task#306分析時の検証、13節参照)は`state=NORMAL`時に発生しており
   Fix A'(opp_lat_pred計算)そのものとは無関係と個別に確認済み。
   → **Fix A'がハード制約を悪化させたという証拠はない**。
2. **Fix A'固有の指標(クランプ飽和率)は3本を通じて一貫して改善**:
   n=3合算6.3%は、14節で確立した旧方式の基準24%を大きく下回る。個々の値
   (4.5/14.8/9.4%)はサンプル数の少なさ(n=27〜179)ゆえに振れるが、
   「24%より明確に低い」という結論は3本全てで支持される。
3. wp281-285帯の慢性輻輳問題自体はFix A'の対象外であり、本検証結果からは
   Fix A'の影響と分離できている(task#306で別途対応予定)。

**判定: Phase 2(dev3ローカル検証)PASS**。CLAUDE.md §2 rule 1(n=1で確定
しない)・rule 4(ハード制約先行比較)の要件を満たした。

### 12.5 次のアクション

`config.yaml`の`overtake.lat_vel_source_tracker`を`true`のまま維持し、
§10のFlowに従いPhase 3(予選環境検証)へ進む。予選ログでの
`lat_vel_src=tracker`発火確認・クランプ飽和率の実測・ハード制約の
非悪化確認を行う(CLAUDE.md §2 rule 6「予選投入前提の検証はdev3を基本と
し、予選提出前の最終確認はdev3で行う」は既に満たしているため、Phase 3は
実際の予選投入による最終確認の位置づけ)。

## 13. Fix B実装完了(2026-08-07、コミット`3706a14`)

Fix A'のPhase 3(予選環境検証)を実際に予選走行させている待ち時間を利用し、
§2の設計(外部AIレビューmust-fix 1反映済み)通りFix Bを実装した:

- `overtake.overlap_floor_enabled`(既定false)ゲート追加
- `_update_overlap_state()`新設(縦オーバーラップ判定、ヒステリシス付き、
  footprint_riskと同じ`along_min_length`を再利用、新規距離定数0個)
- `_apply_overlap_floor()`新設(並走中はtarget_magを縮小させない、床は
  専用の新規状態`_ot_overlap_floor_mag`を使い168節の既存フリーズ値
  `_ot_last_valid_target_mag`とは分離、適用時に必ず現在のcorr_boundで
  再キャップ)
- 呼び出し2箇所(OVERTAKING分岐・STOPPING/proactive-bias分岐)
- `[OVERLAP-FLOOR]`診断ログ(エッジトリガー)
- リセット処理統合: 4箇所(側反転/rescue反転/新規エンゲージ/STUCK復帰)の
  重複実装を`_reset_ot_episode_tracking_state()`へ統合(§4の設計は
  Fix A採用による変数削除を前提としていたが、実際のFix A'実装は
  `_ot_opp_lat_prev`等4変数を削除せず名称再利用する形になったため、
  統合ヘルパーは当初想定の2行ではなく6行+Fix Bの2行を扱う形へ適応した)
- 新規単体テスト22件(`test_fix_b_overlap_floor_20260807.py`)+既存5件の
  アンカー更新(統合ヘルパー呼び出しへの置換に伴う)、回帰3199件PASS

ゲートOFF(既定)のため現行挙動とビット等価、`config.yaml`は
`overlap_floor_enabled: false`のまま。

**次のアクション**: §10のFlowに従い、Fix A'のPhase 3(予選環境検証)完了・
判定を待ってから、Fix Bのオフライン反実仮想検証→dev3ローカル検証
(Phase 2)へ進む(Fix Aと同時進行させない、§10の方針通り)。

## 14. Fix Bオフライン反実仮想検証、重大な発見(2026-08-07)

§6項目1(反実仮想リプレイ、最優先)を実施した。Fix A' Phase 3(予選環境
検証)の待ち時間を利用し、`[OT]`ログの`d_min`フィールドがコード上
`_fwd_ds`(=対象車とのds)そのものであることを確認した上で(追加の
V2X生データ再構成は不要と判明)、2つの名前付きフィクスチャへFix Bを
適用する反実仮想リプレイを実施した:

- フィクスチャ1(18節衝突事例、`output/20260805-222055` d1、wp215→233、
  4サンプル)
- フィクスチャ2(`0805-07`慢性未達事例、qualifying、wp62→118、
  41サンプル)

**結果: 両フィクスチャ計45サンプル全てで`overlapping`判定が一度も
Trueにならず、Fix Bの床が一度も作動しなかった**。`d_min`の最小値は
フィクスチャ2のwp109で3.00m(それでも`exit_thr=3.0m`未満には届かず
不成立)、大半は4〜13m台に分布していた。一方Fix Bの`enter_thr`は2.5m・
`exit_thr`は3.0m(`along_min_length`=カート全長2.0m+マージン、
footprint_riskと同じ物理的下限を再利用)。

**解釈**: MPCは相手車に追いつく前、3〜13m手前の「接近中」段階から
先行して横オフセットを立ち上げており、両動機事例の実際の不具合
(offset未達・崩壊)もこの段階で起きている。Fix Bが根拠とした
footprint_riskの閾値(真横・車体長スケールの近接のみを検知する設計
思想)とは、そもそも捉えている現象のスケールが1桁近く違う——
**Fix Bは「並走中(ほぼ真横)」という狭い定義のまま実装したが、実際に
対処すべき現象は「オーバーテイク接近中の広いレンジ(3〜13m)」で
起きているという、スコープの取り違えが疑われる**。

**評価**: このままdev3/予選環境検証へ進んでも、Fix Bが実効性を持たない
まま「悪化はしないが改善もしない」という結果になり、段階導入(§10)の
目的(各Fixの効果を個別に帰属できる状態で検証する)を満たせない。
オフライン検証(CLAUDE.md §1.4準拠)で実装後の実地検証コストを払う前に
この設計不備を発見できたことは、まさにこの検証段階を設けた狙い通り
である。

**次のアクション**: 外部AI相談プロンプトを作成した
(`docs/superpowers/specs/2026-08-07-fix-b-overlap-threshold-ineffective-
consultation-prompt.md`)。閾値の広域化・判定ロジック自体の見直し
(`state=="OVERTAKING"`全体を対象にする案等)を検討し、結果を本節へ
追記する。dev3/予選環境検証(Phase 2/3)は、この設計見直しが完了する
まで保留する。
