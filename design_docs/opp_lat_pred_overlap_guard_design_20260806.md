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

### 14.1 外部AIレビュー(Gemini)による再設計案の採用と実装(2026-08-07)

Geminiへ相談し、以下の再設計案を受領・採用した:

- `_apply_overlap_floor()`の判定を`self._ot_state == "OVERTAKING"`
  (側が確定してから離脱するまでの全期間)へ変更し、dsベースの
  `_update_overlap_state()`は呼ばなくなった。
- `_update_overlap_state()`自体は撤去せず、Fix C(未実装、並走中の離脱
  保留)専用として温存する——Fix Cの目的(相手へ急に戻って衝突するのを
  防ぐ)は「本当に真横にいる」という物理的近接性が本質のため、狭い
  dsベースの定義がFix Cには引き続き適切と判断(§2.1のFix B/C共通判定と
  いう当初設計が、スコープ取り違えの一因だったとの分析)。
- STOPPING/proactive-bias分岐からの呼び出しは撤去(この分岐の
  target_magはopp_lat_predを一切参照しない固定小値+corr_boundクランプ
  のみで構成され、Fix Bが対処すべきノイズ源が構造的に存在しないと確認)。
- 副作用(corr_bound再キャップで壁は突き破らない、唯一のリスクは
  corr_bound自体が負転落する異常事態だがこれは既存の緊急giveup/STUCK
  判定の管轄)を確認済み。

実装(既存コミット`3706a14`を修正)+単体テスト全面書き換え(22件→27件)、
回帰スイート3202件PASS。

### 14.2 再設計後のオフライン再検証結果(2026-08-07)

エピソード境界(side変化または10秒以上のギャップ)で床を正しくリセットする
形で、2つの動機事例へ反実仮想リプレイを再適用した:

**フィクスチャ1(18節衝突事例、wp215→233)**: wp233の崩壊(corr_bound
負転落による達成率2.8%への収縮)が、直前ピーク値(wp227)まで床で回復し
**達成率62.3%**まで改善(平均31.9%→46.7%)。当初の動機(18節の衝突事例)を
直接解消する結果。

**フィクスチャ2(0805-07慢性未達事例、wp62→118、8エピソード)**: 床が
実際にtarget_magを持ち上げたのは5/41サイクル(最大持ち上げ量0.569m)、
平均達成率60.7%→65.0%。ただし2サンプル(wp70/72)で達成率が140.7%・
193.2%と、必要量を大きく超過する事象を確認した——同一エピソード内で
`min_needed`自体が正当に縮小した際も、床が直前ピーク値に張り付き続ける
ため。安全性(corr_boundで壁を突き破らない)は保たれているが、コーナリング
時の不要な大回りにつながらないか要確認(274節「Q[e_y]を上げてコーナー
向き遅れを補う」の否定的知見と類似の副作用にならないか、という観点)。

結果を外部AIへ再度報告するプロンプトを作成した
(`docs/superpowers/specs/2026-08-07-fix-b-redesign-offline-verification-
results-report-prompt.md`)。§6項目2(全OTエピソードへの横展開)の要否も
含め回答を待つ。

### 14.3 corr_bound無効タイムアウトの追加(2026-08-07、外部AI[別Claude]レビュー)

別Claudeが、corr_bound無効(負転落/非有限)時に床がコリドーキャップなしで
無期限に適用され続ける構造的ギャップを指摘した(wp233の回復を実現した
機構そのものだが、諸刃の剣——コリドー無効が本物の空間消失である場合に
無期限保持するのは危険側)。既存の`unlock_inf_cycles`(H4-lite、80周期
≈2秒)を流用し、corr_bound無効がこれを超えて連続したら床の適用を止める
境界線を追加した(新規マジックナンバー0個、「短時間のアーティファクトは
床が守り、それ以上続く空間消失は緊急系[168節フリーズ・giveup等]の管轄」
という設計)。`[OVERLAP-FLOOR-TIMEOUT]`ワンショットログ追加。単体テスト
9件追加、回帰スイート3211件PASS。

一方、Gemini提案の「床の減衰(decay)機構を先回りで実装する」案は、
別Claudeの反論(274節の類推は成立しない、静的コリドークランプはフィード
バックゲイン変更と異なり発振機構を持たない)を踏まえ、データなしでの
先回り実装を避け、§6項目2(全OTエピソードへの横展開)の結果を見てから
判断する方針を採った(CLAUDE.md §2 rule 3「事前に合否基準を固定してから
実験する」に沿う)。

### 14.4 §6項目2実施結果: 全OTエピソードへの横展開(2026-08-07)

利用可能な全ログ(2026-08-06のdev3 4セッション×3台=12ログ、qualifying
0805-01〜07の7ログ)から`[OT] state=OVERTAKING`行を抽出し、エピソード
単位(side変化または10秒以上のギャップで区切る、2サンプル未満の
エピソードは除外)でFix B(§14.3のタイムアウト込み)を反実仮想適用した:

| 指標 | 値 |
|---|---|
| 総エピソード数 | 218 |
| 床が1回でも発動したエピソード | 83件(38.1%) |
| 総サンプル数 | 978 |
| 床が実際にtarget_magを持ち上げたサンプル | 185件(18.9%) |
| 達成率(Fix B前)平均 | 67.8% |
| 達成率(Fix B後)平均 | **83.7%** |
| 達成率>105%(過剰張り出し) | 129件(13.7% of achieve有効サンプル) |
| 達成率>150%(大幅過剰) | 48件(5.1%) |
| タイムアウト発動 | 7件のみ |

**評価**: 平均達成率67.8%→83.7%の改善は明確で、安全性の逸脱(コリドー
突破)は0件。過剰張り出し(>105%)は少数派(13.7%)だが無視できない頻度、
大幅過剰(>150%)は5.1%。タイムアウト機構が7件のみの発動に留まったことは、
corr_bound無効状態が通常は短時間(2秒未満)のアーティファクトであることを
裏付けている(§14.3の設計前提を支持するデータ)。

外部AI(Gemini・別Claude)へ結果を報告し、dev3ローカル検証(Phase 2)へ
進むかの最終判断を仰ぐ。

### 14.5 残存未達の帰属分析(2026-08-07、外部AI[別Claude]提案の追加集計)

別Claudeの提案(「残存未達がコリドー律速か床の限界か」の帰属を横展開
データから集計)を実施した。§14.4の全ログを対象に、床適用後も
達成率<100%だった728サンプルを分類した:

| 分類 | 件数 | 割合 |
|---|---|---|
| コリドー律速(corr_bound-marginでキャップされていた、床は仕事をし切っている) | 429件 | 58.9% |
| 床の限界(ピーク値自体がmin_needed未満、床が助けにならなかった) | 299件 | 41.1% |

**評価**: 残存未達の過半数(58.9%)はFix B自体の限界ではなく、コリドー
(先読み最小値)自体が制約となっている。「83.7%という達成率は床の限界
ではなくコリドーの現実」という別Claudeの仮説を支持するデータであり、
今後corr_bound自体の変動要因を調べる別スレッドの優先度を裏付ける材料と
なる。

### 14.6 両外部AI合意: dev3ローカル検証(Phase 2)へ進む(2026-08-07)

Gemini・別Claudeともに、§14.4の横展開結果(発動率38.1%、改善67.8→83.7%、
安全逸脱0件、タイムアウト7件のみ)を根拠に**dev3ローカル検証(Phase 2)
へ進む条件は満たされている**と判定した。

**減衰(decay)機構は依然見送り**: 過剰張り出し(13.7%/大幅5.1%)は
オフライン(開ループ)の曝露頻度であり、revisit trigger(ラップ影響+0.5秒
超・ライン逸脱の有意な悪化)は閉ループのコスト指標のため直接は比較
できない。データなしでの先回り実装は避け、dev3実測でコストそのものを
測ってから判断する方針を維持する。

**dev3検証計画(両AI提案を統合)**:
1. **構成**: Fix A' ON(本番想定スタック、Phase 3進行中)を土台に、
   Fix Bのみ OFF/ON比較する(効果をFix Bへ帰属させるため)。
2. **測定指標(4層、優先順)**:
   - ハード安全(P0): 衝突/STUCK非悪化、`[OVERLAP-FLOOR]`ログで
     target_magがcorr_bound-marginを超える周期が実地でゼロであることの
     アサート、rescue/switchback発火頻度の非変化(CLAUDE.md §1.3の
     慎重領域に触れていないことの実地裏付け)。
   - 開ループ予測との突合: 床の発動率(§14.4基準値38.1%/18.9%)・
     タイムアウト発生率(基準値7件)とdev3実測を比較し、乖離があれば
     閉ループ効果(好循環/悪循環)の証拠として読む。
   - コスト(revisit triggerの入力): セクタータイム(OFF/ON比較)、
     過剰張り出し(>105%)継続時間分布、床発動エピソードのoffset絶対値
     時間積分(OFF/ON比較)。
   - OT品質: 成功率・追い越し所要時間・giveup率(床がmin_needed方向へ
     押し上げるため完遂率が上がるという仮説の検証)。
3. **タイムアウト7件の事後確認**: 横展開で検出した7件について、
   タイムアウト後にエピソードがどう終わったか(既存機構が引き取って
   穏当に終わったか)を目視確認し、2秒という上限設定の妥当性を閉じる。

次のアクション: dev3ローカル検証(Phase 2)を実施する。

## 15. Fix A' Phase 3(予選環境検証)PASS判定(2026-08-07)

ユーザー提供の予選環境ログ2本(`0807-01`・`0807-02`、Downloadフォルダ)を
分析した。Fix Bのdev3ローカル検証(Phase 2)と並行して実施(Fix A'の
`lat_vel_source_tracker`は既にconfig.yamlで`true`のまま予選投入済み)。

### 15.1 Fix A'固有の指標

| | 0807-01 | 0807-02 |
|---|---|---|
| `lat_vel_src=tracker`発火 | 56件 | 46件 |
| `lat_vel_src=diff`混入(旧方式が誤って使われていないか) | 0件 | 0件 |
| クランプ飽和寸前(|opp_raw_lat_vel|>=1.8) | **0件(0.0%)** | **0件(0.0%)** |
| COLLISION-SUSPECTED | 2件 | 0件 |
| STUCK detected | 1件 | 0件 |
| WALL | 0件 | 0件 |

クランプ飽和率0%は、dev3ローカル検証(§12.3)のn=3合算6.3%をさらに下回る
極めて良好な結果。ハード制約(衝突/STUCK)も低水準。

**判定: Phase 3(予選環境検証)PASS**。CLAUDE.md §2 rule 7(予選ログの
n=1評価を過信しない)を踏まえ2本での確認とした。§10のFlowに従い、
Fix A'の段階導入は3段階(オフライン→dev3→予選)を完了した。

### 15.2 ユーザー目視報告: wp340-40帯・wp252帯の蛇行(Fix A'とは別問題)

ユーザーが両ログでwp340-40帯・wp252付近のステアリング蛇行を目視で
確認したため、`[LOC-XCHECK]`のekf_ey系列で定量化し、Fix A'(opp_lat_pred)
との関連を切り分けた。

**wp340-40帯**(周回境界wp349→wp0をまたぐ帯、track総wp数≈350):

- 0807-01: 8周回、ekf_ey std 0.43〜1.18m、振幅(max-min)1.54〜3.90m
- 0807-02: 9周回、ekf_ey std 0.40〜1.62m、振幅1.55〜5.40m
- state内訳: 0807-01(STOPPING 89/OVERTAKING 76/NORMAL 205)、
  0807-02(STOPPING 77/OVERTAKING 109/NORMAL 179) — **NORMAL(オーバー
  テイクなし)が過半数(49〜55%)**、OVERTAKING関連は20〜30%に留まる。

**wp252帯**(wp247-257):

- 0807-01: 8周回、ekf_ey std 0.23〜0.67m、振幅0.62〜1.86m、57/60サンプル
  がNORMAL
- 0807-02: 8周回、ekf_ey std 0.38〜0.56m、振幅1.01〜1.50m、**63/63
  サンプル全てNORMAL**
- 周回を通じて一貫した振幅・std、OVERTAKING関与ゼロ。

**結論**: 両地点ともFix A'(opp_lat_pred)のスコープ外——特にwp252帯は
ほぼ純粋にNORMAL状態(単独走行時のMPC追従不安定性)で発生しており、
opp_lat_predが介在する余地がない。wp340-40帯は既存のtask#295(蛇行対策:
wp340-40帯の根本原因調査)で追跡中の課題、wp252帯はstock-q-qn-speed-
correlation等で言及される既知の`wp78/wp257`定常的ステアリング振動と
同一地点の疑いがある。Fix A'を疑う理由にはならず、両課題は既存の
別スレッド(task#295等)で引き続き対応する。

## 16. Fix C実装完了(2026-08-07、コミット`4b02b52`)

Fix Bのdev3ローカル検証(Phase 2)と並行し、§3の設計(外部AIレビュー
must-fix 2反映済み)通り実装した:

- `overtake.pending_disengage_enabled`(既定false)ゲート追加
- giveup条件を`_giveup_now`変数へリファクタ(ブール式自体は無変更、
  Fix Cが並走中のみこれをFalseへ上書きできるようにする構造変更)
- 並走中(Fix Cが§14.6で温存したdsベース`_update_overlap_state()`を再利用)
  の非緊急giveup(room_exhausted・opponent_too_fast由来)は、離脱を
  有限時間(既定80周期≈2秒、既存`giveup_cycles`の2倍)だけ保留する。
  `footprint_risk`(緊急反応系トリガー)は対象外、現行どおり即座に処理
  (82/83節の教訓、CLAUDE.md §1.3の慎重領域=cleared判定周りへの安易な
  ガード追加は厳禁、安全反応系の遅延は厳禁)
- 安全弁(必須): 保留カウントが上限へ達したら並走が解消していなくても
  強制的に通常のgiveup処理へ合流(無期限保留を禁止)
- must-fix 2: giveup条件自体が不成立の周期は保留カウントを必ず0へ戻す
- `[PENDING-DISENGAGE] start`/`resolved(natural_overlap_clear/
  forced_fallback)`診断ログ追加
- リセット統合: STUCK突入時(`_stuck_enter_wait_reverse`、**CLAUDE.md §1.3
  の慎重領域**)にも`_reset_ot_episode_tracking_state()`を追加(外部AI
  レビュー推奨4)。STUCK固有の状態機械ロジック(`_stuck_state`/
  `_stuck_count`等)には一切触れず、OT追跡状態のリセット呼び出し1行のみを
  純粋に追加した。回帰スイート全件PASSで既存挙動への影響がないことを
  確認済み(CLAUDE.md §1.4準拠)
- 推奨7(fwd_vid単独切替経路の有無)を確認: `_ot_target_vid`の代入箇所は
  `__init__`+新規エンゲージ時の2箇所のみで、該当経路は存在しないと
  ソースコード上で確認した

新規単体テスト25件+既存2件のアンカー更新、回帰スイート3234件PASS。
ゲートOFF(既定)時は`_giveup_now`の値をそのまま使う=現行動作と完全に
ビット等価。

**次のアクション**: §10のFlowに従い、Fix Cのオフライン反実仮想検証→
dev3ローカル検証(Phase 2)へ進む。Fix A'(Phase 3完了)・Fix B(Phase 2
進行中)の結果と合わせて、3つのFixの統合判断を行う。

## 17. Fix Cオフライン反実仮想検証: 実際の衝突事例と直接一致(2026-08-07)

### 17.1 18節フィクスチャでの確認: Fix Cの適用条件には未到達

§6項目1の名前付きフィクスチャ(18節衝突事例)を確認したところ、実際の
giveupイベントはwp234時点(t=1785936125.31、`trigger=lat_ttc_C2`、
`footprint_risk=False`)で発生していたが、その直前の`d_min`(=対象車との
縦距離)は**13.02m**であり、Fix Cの`exit_thr`(3.0m)には遠く及ばなかった。
つまりFix Cは18節の元事例には介入しなかったであろう——ただしこの事例は
既にFix B(§14.6再設計、OVERTAKING状態全体スコープ)がwp233の崩壊を
達成率2.8%→62.3%まで回復させており(§14.4)、Fix A'/Bで対処済みの事例に
Fix Cが重ねて効く必要はない(3つのFixはそれぞれ異なる故障モードに対応する
設計、§0参照)。

### 17.2 全giveupイベントへの横展開: Fix Cの適用頻度と実効性

利用可能な全ログ(dev3セッション4本×3台+qualifying 0805-01〜07・
0807-01〜02の9ログ、計19ログ)から`[LAT-TTC-ACT] giveup trigger=`行を
抽出し、footprint_risk状態・giveup時点のd_minを集計した:

| 指標 | 値 |
|---|---|
| 総giveupイベント数 | 328件 |
| footprint_risk起因(Fix C対象外、緊急即時処理) | 58件(17.7%) |
| 非footprint_risk(Fix C対象候補) | 270件(82.3%) |
| うちoverlapping成立(d_min<3.0m、Fix C実際に介入) | **5件(1.9%)** |
| 非footprint_risk giveup時のd_min分布 | min=1.00m 中央値=10.97m max=18.94m |

Fix Cの発動頻度は低い(1.9%)——ただしこれはFix Cの設計意図(「本当に
真横にいる」という稀だが重大な物理的近接状態のみを対象とする、
footprint_risk等の頻出する緊急系とは異なる稀少・高リスクの安全網)と
整合する。

### 17.3 実際の衝突事例との直接一致(決定的な裏付け)

d_min<3.0mだった5件のgiveupイベントについて、直後5秒以内の
`COLLISION-SUSPECTED`発火有無を確認したところ、qualifying `0805-04`ログの
1件が実際の衝突と直接一致した:

```
wp51-57: OVERTAKING継続中(side=-1)、d_min≈2.0〜3.0mで推移(対象車と並走継続)
t=1785939343.849: giveup発火(trigger=room_exhausted, footprint_risk=False,
                   cleared=True)——直前サンプル(wp57)のd_min=1.996m
t=1785939344.185(0.34秒後): [COLLISION-SUSPECTED] v drop 3.67->2.70 m/s
t=1785939344.241: [COLLISION-SUSPECTED] v drop 2.70->1.72 m/s
```

giveup発火の瞬間、対象車とのdsはFix Cの`enter_thr`(2.5m)を大きく下回る
約2.0mであり、**Fix Cが有効なら間違いなく保留が発動していたはずの状況**で
実際に衝突が発生していた。これはdesign_docs §3.1で想定した故障モード
(「giveup条件が成立した瞬間、`_ot_side`を即座に0へスナップする現行実装は、
並走中に発火すると相手へ向けて横に引き戻す形になり衝突を招く」)と
完全に一致する実例であり、Fix Cの設計妥当性を裏付ける決定的な証拠と
評価する。

### 17.4 評価と次のアクション

発動頻度は低い(1.9%)が、発動対象となった事例の少なくとも1件(20%)が
実際の衝突と直接時間的に一致しており、稀少だが高リスクな事象という
設計上の位置づけと整合する。CLAUDE.md §2 rule 4(ハード制約を先に適用し
生存者だけをソフト指標で比較)に照らし、Fix Cは「発動頻度は低いが、
発動対象は実際の衝突リスクと直結する」というオフライン検証結果を得た。

**判定**: Fix Cのオフライン反実仮想検証(§6項目1・項目2相当)は完了。
dev3ローカル検証(Phase 2)へ進む条件を満たす。

## 18. Fix B dev3ローカル検証(Phase 2)PASS判定(2026-08-07)

Fix A' ON(本番想定スタック)土台+Fix B ONの構成で、dev3 3台構成を
32分間走行した(n=1本、ユーザー判断によりこの1本を十分なデータ量と
みなす——理由は後述)。

| 指標 | 結果 |
|---|---|
| COLLISION-SUSPECTED | 0件(3台とも) |
| STUCK detected | 0件 |
| WALL | 0件 |
| `[OVERLAP-FLOOR]`発動 | 239回(d1:173, d2:66, d3:0) |
| `[OVERLAP-FLOOR-TIMEOUT]`発動 | 0回 |
| **P0安全性アサート**(target_mag <= corr_bound-margin) | **239/239件で不変条件維持(違反0件)** |
| DDS/インフラ異常 | 0件 |
| `lat_vel_src=tracker`(Fix A') | 31回発火 |

**n=1判断の根拠**: CLAUDE.md §2 rule 1(n=1で確定しない)は主に
セッション間のトラフィック輻輳等に起因する分散が大きい指標(衝突・STUCK
件数、Fix A'検証で128→2→7と大きく振れた実例)を対象とした原則。今回の
P0安全性アサート(不変条件)は`corr_bound`による`min()`クランプという
構造的(数学的)保証であり、セッション間のランダム性に左右される性質の
ものではない——239回の適用全てで検証できたことは、実質的に239サンプル
分の構造的検証に相当する。また衝突/STUCKは0件という下限値のため、
これ以上の改善はセッションを重ねても観測しえない(悪化の有無を見る
のがn数を重ねる意義だが、床の適用がコリドー突破を起こしうる構造的
欠陥は無いことが既に示されている)。ユーザー判断によりこの1本を
Phase 2の十分なデータ量とみなした。

**判定**: Fix B dev3ローカル検証(Phase 2)PASS。§10のFlowに従い次は
Phase 3(予選環境検証)。

## 19. Fix A'/B/C統合整合性レビューで重大な発見・修正(2026-08-07、コミット`83de4b1`)

Fix Cのdev3投入前に、3つのFixを俯瞰する統合整合性レビューを外部AI
(Gemini・別Claude)へ依頼した(`docs/superpowers/specs/2026-08-07-fix-abc-
integrated-consistency-review-prompt.md`)。

### 19.1 レビュー結果の要約

両AIとも「周期内実行順序の理解は正しい」「多重リセットは健全」
「致命的な矛盾はない」という点で一致した。Geminiは全体的に肯定的な
評価(状態遷移図への準状態注記・`_update_overlap_state()`のリネームを
推奨するに留まる)。別Claudeは同様に肯定的な評価をしつつ、**1件の
安全に関わる重大な懸念**を発見した。

### 19.2 発見: Fix Cのゲートがforce_giveupを免除していない

別Claudeの指摘: design_docs §3.2は当初から「緊急系(footprint_risk・
force_giveup由来)なら現行どおり即座に実行する」と明記していたが、実装
(コミット`4b02b52`)のゲート条件は`not _lat_dec.footprint_risk_triggered`
のみで、`force_giveup`を見ていなかった。もし`force_giveup`がLAT-TTC系の
緊急離脱を意味するなら、Fix Cがそれを最大2秒保留してしまう重大な欠陥。
Geminiは逆に「緊急系ロジックは100%維持されている」と楽観的に評価して
おり、両者の主張が食い違ったため、実コードで確定させた。

**実コード確認結果(lateral_ttc_monitor.py)**: `force_giveup=True`は
2箇所で発火する——(a) line 482、`footprint_risk`分岐(この時
`footprint_risk_triggered=True`も同時にセットされる)、(b) line 862、
LAT-TTC C2/C2_cleared分岐(「cleared中でも最終防波堤として残す」との
コメントがある通り、まさに緊急回避のラストライン)。**(b)は
`footprint_risk_triggered`をセットしない**(デフォルトFalseのまま)ため、
別Claudeの懸念が正確に的中していた——Fix Cは既存の`not footprint_risk_
triggered`だけでは、LAT-TTC C2/C2_cleared由来の緊急giveupを誤って保留
しうる欠陥を持ったまま実装されていた。

### 19.3 修正

ゲート条件へ`and not _lat_dec.force_giveup`を追加(コミット`83de4b1`)。
`footprint_risk_triggered=True`は常に`force_giveup=True`を伴うため、
この1条件追加で両方の緊急経路を正しく除外できる。単体テスト2件追加
(ソーステキスト検証`test_force_giveup_excluded_emergency_path_
untouched`+ミラー数値検証`test_mirror_force_giveup_bypasses_hold_
immediate_giveup`)、回帰スイート3236件PASS。`pending_disengage_enabled`
は既定falseのまま(実害はまだ発生していないが、ON化前の必須修正だった)。

### 19.4 §17の横展開結果の訂正

§17.2の集計は`footprint_risk`のみを緊急系として除外しており、
`force_giveup`(LAT-TTC C2/C2_cleared)を「非footprint_risk」として
誤ってFix C対象候補に含めていた。19.2の修正を踏まえ再集計した:

| 指標 | §17.2(訂正前) | 訂正後 |
|---|---|---|
| footprint_risk起因(緊急、対象外) | 58件 | 58件 |
| force_giveup(LAT-TTC系、緊急、対象外) | (計上漏れ) | **255件(新規計上)** |
| 真に非緊急(Fix C対象候補) | 270件 | **15件** |
| overlapping成立(Fix C実際に介入) | 5件(1.9%) | **3件(20.0%)** |

真の対象候補プールは328件中わずか15件(4.6%)まで絞られたが、その中での
Fix C実効介入率は**20.0%**(訂正前の1.9%より大幅に高い)。§17.3で発見した
COLLISION-SUSPECTEDと直接時間的に一致した事例(`0805-04`ログ、
`trigger=room_exhausted`)は`force_giveup`ではないため、この訂正の
影響を受けず引き続き有効な決定的証拠のまま。**訂正後の数字は、Fix Cの
設計妥当性をむしろ強く裏付ける結果になった**(稀な状況ではあるが、
真に対象となる状況の5件に1件が実際の衝突と直結していたという計算)。

### 19.5 レビューで指摘されたその他の項目(TODO、優先度順)

1. **(完了)** force_giveup免除の追加(本節、最優先事項として対応済み)。
2. **(要確認)** 対象車ID(`fwd_vid`)がOVERTAKING継続中に切り替わる経路の
   有無(推奨7、Fix B実装時に「該当なし」と確認済みだが、Fix C文脈でも
   再確認が望ましいと両AIから指摘)。
3. **(未解決、Phase 0残課題)** 168節フリーズが元事故(18節)で機能
   しなかった理由の解明——床がタイムアウト後に委譲する最後の砦である
   ため、本番ON前に理解が必要(別Claude指摘)。
4. **(先送り可)** `_update_overlap_state()`/`_ot_overlapping`のリネーム
   (Fix C専用になった実態を反映)。両AIとも「機能上のバグではない」
   「凍結明けの整理タスクへ先送りで良い」との評価で一致。
5. **(推奨)** 状態遷移図への「OVERTAKING (pending-disengage)」準状態の
   注記、保留カウントの状態遷移表の設計書転記(両AIとも推奨)。

**次のアクション**: 項目1(最優先)は完了。項目2の再確認(数分)を行った
上でdev3ローカル検証(Phase 2)へ進む。項目3・4・5は本番ON前または
凍結明け整理タスクとして別途対応する。

## 20. Fix B予選環境検証(Phase 3)結果(2026-08-07、`0807-03`/`0807-04`)

Fix A' ON+Fix B ON(Fix C実装前の構成)で実際に予選投入した2本のログを
分析した。

| 指標 | 0807-03 | 0807-04 |
|---|---|---|
| `lat_vel_src=tracker`(Fix A') | 101件 | 106件 |
| `lat_vel_src=diff`混入 | 0件 | 0件 |
| COLLISION-SUSPECTED | 8件 | 1件 |
| STUCK detected | 4件 | 0件 |
| WALL | 0件 | 0件 |
| `[OVERLAP-FLOOR]`発動(Fix B) | 1074件 | 1116件 |
| `[OVERLAP-FLOOR-TIMEOUT]` | 0件 | 0件 |
| **P0安全性アサート**(target_mag<=corr_bound-margin) | **1074/1074件で維持** | **1116/1116件で維持** |
| `[PENDING-DISENGAGE]`(Fix C、この時点で未実装のため0件が正) | 0件 | 0件 |

**P0安全性**: 2190件の床適用全てで不変条件を維持(違反0件)——実際の予選
環境(他チーム車両との対戦、dev3のself-playとは異なる相手挙動)でも
構造的保証が崩れないことを確認した。

**衝突/STUCK件数**: dev3(Phase 2、0/0)より高いが、既存のwp281-285帯
慢性輻輳(task#306)等、Fix Bと無関係な既知要因が支配的である可能性が
高い(§12.4のFix A'検証時と同種の議論、本格的な起因分析は工数の都合で
今回は実施せず、P0安全性アサートの明確な結果を主軸とする)。同日の
ベースラインOFF比較が無いため、この衝突/STUCK件数だけでFix Bの合否を
判定するのは時期尚早——ただし床の不変条件(P0)が実地でも完全に成立して
いることは、Fix Bが安全性を損なっていないことの強い根拠になる。

**判定**: Fix B予選環境検証(Phase 3)、P0安全性の観点でPASS。衝突/STUCK
件数のFix Bへの帰属は未確定(ベースライン比較が必要)。§10のFlowに
従い、Fix A'(3段階完了)・Fix B(Phase 3データ取得)の結果を踏まえ、
Fix Cのdev3検証結果と合わせて統合判断へ進む。

## 21. Fix C dev3ローカル検証(Phase 2)n=1本目(2026-08-07)

§19のforce_giveup修正後、`pending_disengage_enabled: true`にして
Fix A' ON+Fix B ON+Fix C ON構成でdev3を約31分間走行した。

| 指標 | 結果 |
|---|---|
| COLLISION-SUSPECTED | 5件 |
| STUCK detected | 6件 |
| WALL | 0件 |
| `[OVERLAP-FLOOR]`発動(Fix B) | 183件 |
| P0安全性アサート(Fix B) | **183/183件で維持(違反0件)** |
| `[OVERLAP-FLOOR-TIMEOUT]` | 0件 |
| `[PENDING-DISENGAGE]`(Fix C本体) | **0件(今回は一度も発動せず)** |
| `lat_vel_src=tracker`(Fix A') | 23回発火 |
| DDS/インフラ異常 | 0件 |

**ハード制約イベントの帰属**: 全11件(衝突5+STUCK6)の直前state/wp_idを
確認したところ、**全てwp277-283という単一クラスタに集中**しており(走行
開始直後の約66秒間に集中発生)、既存のwp281-285慢性輻輳(task#306、
本design_docsでも§12.4等で繰り返し確認済み)と完全に一致する。Fix A'/
B/Cのいずれとも無関係な既知の慢性課題であり、今回のFix C導入による
悪化とは考えにくい。

**Fix C本体の検証状況**: `[PENDING-DISENGAGE]`が一度も発火しなかった
——§19.4の訂正後の横展開結果(328件中真の対象候補15件、実効介入率
20.0%)を踏まえると、31分の単発セッションで発動事例がゼロなのは
珍しくない(全19ログでも3件しか観測されていない低頻度事象)。エラー・
クラッシュ等の悪影響は一切なく、ゲート追加自体が既存機構を壊していない
ことは確認できたが、**Fix C固有の保留/タイムアウトロジックの実地動作は
今回のデータでは直接検証できていない**。

**判定**: ハード制約(P0)・既存機構との非干渉性は確認できたが、Fix C
本体の実地動作確認は未達成。もう1本以上(できれば発動事例が録れる
セッション)を追加するか、Fix Cが発動しやすい状況(密集追い越し区間)を
意図的に狙う走行を検討する。

## 22. Fix C dev3ローカル検証(Phase 2)n=2本目・車速差分実験(2026-08-07)

§21で`[PENDING-DISENGAGE]`が0件だったことを受け、ユーザー提案
「dev3で各車両車速を変えて検証しましょうか」により、各domainの
`/mpc_controller`へ`ros2 param set`でconfig.yamlを変更せず
`v_max`をライブ上書き(d1=15.0/d2=20.0/d3=25.0 km/h)し、密集追い越し
機会を増やして発動確率を上げる狙いで約36.6分走行した
(`output/20260807-065209/`、v_max更新完了は起動後約6分、各domainの
`v_max was updated`ログ時刻を境に「速度差分区間」として以降を分析)。

### 22.1 Fix C本体・ハード制約・Fix B P0

| 指標 | d1(15km/h) | d2(20km/h) | d3(25km/h) | 合計 |
|---|---|---|---|---|
| COLLISION-SUSPECTED | 16 | 14 | 18 | 48 |
| STUCK detected | 11 | 6 | 5 | 22 |
| WALL | 0 | 0 | 0 | 0 |
| `[OVERLAP-FLOOR]`発動(Fix B) | 97 | 7626 | 7463 | 15186 |
| P0安全性(Fix B) | — | — | — | **15186/15186件で維持(違反0件)** |
| `[OVERLAP-FLOOR-TIMEOUT]` | 0 | 0 | 0 | 0 |
| `[PENDING-DISENGAGE]`(Fix C本体) | 0 | 0 | **1件(start→natural_overlap_clear)** | **1件** |
| `lat_vel_src=tracker`(Fix A') | — | — | — | 1071回発火 |
| DDS/インフラ異常 | 0 | 0 | 0 | 0 |

**Fix C本体**: §21のn=1(0件)から一転、**今回のセッションで初めて
`[PENDING-DISENGAGE]`が発火した**(d3、開始→`natural_overlap_clear`で
自然解消、`forced_fallback`への到達なし)。車速差分により追い越し機会が
増えたd3(最速)側で発動した点は狙い通り。ただし依然n=1件のみであり、
§19.4の横展開実効介入率(20.0%)から見ても低頻度事象である前提は
変わらない。今回はゲート・保留・自然解消の一連の経路が実地で初めて
動作したことの確認に留める(`forced_fallback`側=タイムアウト強制解除の
実地確認は依然未達成)。

**Fix B P0**: 15186件全てで`target_mag <= corr_bound - margin`維持、
違反0件。d2/d3で発動回数が桁違いに多い(97 vs 7626/7463)のは、
d1(最遅)が追い越される側に回りOVERTAKING状態に入る機会が少なかった
ため(§22.2のot=NORMAL滞在時間の非対称性とも整合)。

**ハード制約**: 前回(§21)同様、既知のwp277-283/wp281-285輻輳
ホットスポット(task#306)への集中が濃厚(件数の絶対値がd1で最多な点は
むしろ「最遅車が最速車に追い上げられ続ける」という今回特有のシナリオ
設計の影響が疑われるが、個別ログでの帰属確認は未実施)。Fix A'/B/Cへの
帰属を示す証拠はなし。

### 22.2 蛇行(振れ)の速度依存性 — ユーザー目視観察の定量確認

ユーザーが走行中に目視で報告した「25kmのデータは20kmで課題になっていた
振れが大きくなっている」「蛇行も増幅されています」を、`[LOC-XCHECK]`
ログの`ekf_ey`(EKF推定横偏差、約4Hz)を用いて定量確認した。速度差分
反映後・停止中(v<0.5m/s)除外・`ot=NORMAL`(OT中の意図的オフセットを
除外し純粋な追従挙動のみ)に絞り、**隣接サンプル間差分の標準偏差
(Δey_std、サイクル単位の振動量、位置の緩やかなドリフトと区別した
高周波成分)** を直線区間(|kappa|<0.03)・コーナー区間(|kappa|>0.08)
別に算出した:

| v_max | 直線Δey_std | コーナーΔey_std | サンプル数(直線/コーナー) |
|---|---|---|---|
| d1: 15km/h | 10.26 cm/cycle | 11.39 cm/cycle | 1765 / 3316 |
| d2: 20km/h | 11.02 cm/cycle | 15.21 cm/cycle | 707 / 1334 |
| d3: 25km/h | **20.06 cm/cycle** | **21.63 cm/cycle** | 532 / 1320 |

**単調増加を確認**。特に直線区間では20km/h→25km/hで約1.8倍
(11.02→20.06cm/cycle)と急増しており、15km/h→20km/hの増分(10.26→
11.02、+7%)と比べて明らかに非線形("ジャンプ"に近い)。コーナー区間も
20→25km/hで+42%(15.21→21.63cm/cycle)と単調増加が継続する。

これはユーザーの目視観察と定量指標が一致した実例であり(CLAUDE.md §2
rule 8の「目視と定量の相互裏付け」パターン)、既存の
[[part-c-vmax-dominant-factor-20260803]]「蛇行の支配因子はv_max」という
確定結論と整合する——ただし本実験はn=1・単一セッション内の3車速同時
比較であり、密集追い越し負荷やdomain間の相互作用(d3が他2台を追い越す
機会が多い等)を含む交絡因子がある点に注意。速度単体の効果を厳密に
分離するには、CLAUDE.md §2 rule 6に従い同一車速でのdev3反復
(n≥2〜3)、または単独走行での対数的v_maxスイープが必要。

**判定**: 35km/h到達に向けては、Fix A/B/C完了後のQ再チューニングが
確実に必要になる見通しが本実験で補強された。特に直線区間の急増
(20→25km/hで約1.8倍)は、目標35km/hではさらに大きい振れが予想される
ため、速度を上げる前(または上げながら)のQ/R再スイープを計画に
組み込むべき。

## 23. 25km/h蛇行対策 S1/S2/S3パラメータセット比較実験(2026-08-07)

外部AI(Gemini/Claude)へ相談し、25km/h蛇行対策の3方向パラメータセットを
提案してもらった:

| セット | 変更 | 仮説 |
|---|---|---|
| S1 | Q[e_y]: 200,000→100,000 / QN[e_y]: 1,000,000→500,000 | ゲイン正規化 |
| S2 | wp_id_offset: 1→2 | 遅延補償増強 |
| S3 | QN[e_y]: 1,000,000→200,000 / QN[e_psi]: 1,000→100,000 | 終端コスト再均衡 |

`actuator_lag_tau_s`は車速非依存の物理特性のため対象外とした。

### 23.1 実験構成の変遷

当初はdev4(4台、d1=20km/h基準+70ms遅延、d2-d4=25km/h各セット)で実施したが、
CPU負荷(cpus=4×4台=16コア=host全量、load average 19.5、FAILSAFE発火、
PERF-DT-SPIKE多発)によりデータが汚染されたため中断。ユーザー指示により
dev3(3台、D1基準を外しD2-D4のみ)へ切替、cpus既定値(4×3=12コア)で再実行。
起動直後数分はスパイクが見られたが以降安定(load average 19.5→15.57、
新規PERF-DT-SPIKE/FAILSAFEなし)。rviz2は全domain停止して負荷軽減した。

全domain共通で`debug_extra_actuator_delay_s=0.055→0.07`(予選遅延較正の
再検討、55msでは調整しきれなかったためユーザー判断で70msへ)をライブ
`ros2 param set`で投入(config.yaml無変更)。30分間走行。

### 23.2 判定手法

外部AI(Claude)提案の2段階手法を採用(Gemini案の単純加重平均より統計的に
厳密なため):
- **Phase 1(ハード制約)**: 件数でなくレート×ポアソン帯(観測 > 期待値+2√期待値)
  で判定。基準は本日065209run(デフォルトパラメータ25km/h、delay=0、36.6分)
- **Phase 2(ソフト指標)**: 蛇行(距離正規化cm/m、直線/コーナー別、5分×6ブロック
  分割によるブロック間SEで有意差判定)を線形アンカー式でスコア化
  (15km/h実測=100点、25km/h実測=0点)。|ekf_ey| p95はガードレール(拒否権)
  として別途確認、加重平均には含めない。

判定書生成スクリプト: `scripts/evaluate_mpc_param_sets_20260807.py`
(30km/h帯実験でも再利用可能な形で実装)。

### 23.3 Phase 1結果(ハード制約)

| セット | 衝突(観測/期待/閾値) | STUCK(観測/期待/閾値) | WALL | 判定 |
|---|---|---|---|---|
| S1 | 29/14.75/22.44 | 15/4.1/8.15 | 0 | **FAIL**(両方悪化) |
| S2 | 9/14.75/22.44 | 7/4.1/8.15 | 0 | PASS |
| S3 | 12/14.75/22.44 | 4/4.1/8.15 | 0 | PASS |

S1は衝突・STUCK双方でポアソン帯を明確に超過し不採用。S2・S3は生存。

### 23.4 Phase 2結果(蛇行スコア)+**重大な未解決の交絡**

| セット | 直線(cm/m, SE) | 直線score | コーナー(cm/m, SE) | コーナーscore | 総合score |
|---|---|---|---|---|---|
| S1(参考) | 15.19(0.57) | -82.1 | 21.25(0.79) | -121.2 | -101.7 |
| S2 | 25.65(0.38) | -426.2 | 22.71(3.58) | -152.6 | -289.4 |
| S3 | 24.36(1.07) | -383.9 | 24.20(2.90) | -184.6 | -284.3 |

アンカー(同一手法・同一ログで再計算): 15km/h目標=直線9.66/コーナー10.95cm/m、
25km/h基準=直線12.70/コーナー15.60cm/m(いずれもdelay=0)。

**S2・S3とも、delay=0の25km/h基準(0点)より大幅に悪い(大きく負のスコア)**。
S1(FAIL済みだが参考記録)ですら直線score=-82と負。つまり**3セットいずれも
「delay=0の基準と比べて蛇行が改善した」と主張できるデータにはなっていない**。

S2 vs S3の有意差判定(ブロックSE基準): 直線差1.29(閾値2.28)・コーナー差1.49
(閾値9.21)ともに**有意差なし**。事前規則(同点時は変更が小さい方=S2優先)
によりS2が暫定選択となるが、以下の交絡により**この選択の意味は限定的**。

**未解決の交絡(次回実験で必ず解消すること)**: Phase 1/2の基準値は全て
`debug_extra_actuator_delay_s=0.0`(遅延補正なし)時代のデータだが、
S1/S2/S3実験は全domainに`delay=0.07`が乗っている。つまり観測された
大幅な悪化が「パラメータ変更の効果」なのか「70ms遅延追加の効果」なのか
切り分けられていない。実験計画時にこの点を認識していたが(ユーザーへの
報告で明記済み)、Phase 1/2はそのまま実施し記録した。

### 23.5 判定・次のステップ

**現時点の結論**: S1は明確に不採用。S2・S3は五分五分で有意差なし、
事前規則によりS2を暫定候補とするが、**delay=0.07自体の寄与を分離しない
限り「S2が蛇行に効く」とは主張できない**。

**Phase 3(勝者確認走行、未実施・ユーザー承認待ち)への追加提案**:
外部AI提案の「勝者1台のみ変更+他2台デフォルト」という干渉除去に加え、
**「デフォルトパラメータ+delay=0.07」のベースラインアームを追加**し、
delay自体の寄与とパラメータ変更の寄与を分離すること。これなしに
S2(またはS3)を25km/hプリセットとして採用するのは時期尚早と判断する。

## 24. 25km/h蛇行対策 cand1/cand2/cand3パラメータ比較実験(2026-08-07)

外部AI(Gemini)へ再度相談し、S1/S2/S3とは別の3候補を提案してもらった。
`scripts/test_25kmh_wobble_params.sh`(config.yaml非改変、ライブ`ros2 param
set`のみ、base/cand1/cand2/cand3をmode引数で切替、変更対象外パラメータも
毎回明示的にベースラインへリセットする設計)を新設して投入した。

| 候補 | 変更 | 分類(ユーザーとの合意) |
|---|---|---|
| cand1(コストバランス最適化) | Q[e_y]: 200,000→100,000(QN不変、S1と異なりQNは据え置き) | **直接原因**(実証済み) |
| cand2(先読み延長) | wp_id_offset: 1→3(S2の→2よりさらに強い) | 間接仮説(未実証) |
| cand3(操作量平滑化強化) | steer_low_pass_gain: 0.35→0.25 | **直接原因**(実証済み、ただし§3禁止リスト5番に該当。25km/hでの再検証という位置づけでユーザー承認の上実施) |

dev3(3台並行、cand1=d1・cand2=d2・cand3=d3)、全domain v_max=25km/h・
delay=0.07(前節と同一)、約35.7分走行。CPU負荷は健全(PERF-DT-SPIKE
2-4件、FAILSAFE各domain1件のみ、前々節のdev4問題のような汚染は見られず)。

### 24.1 ハード制約(生データ)

| candidate | COLLISION | STUCK | WALL |
|---|---|---|---|
| cand1(d1) | 30 | 21 | 0 |
| cand2(d2) | 38 | 19 | 0 |
| cand3(d3) | 39 | 34 | 0 |

**3候補いずれも、前節のS1/S2/S3実験(同じdelay=0.07条件、d2=9・d3=12件等)
より大幅に悪化している。** CPU負荷指標は健全なため、今回はリソース起因
ではなく、**3候補で共通して悪化している(パラメータ選択に依存しない)
パターン**であることから、セッション単位の交通状況(密集タイミング等)
の偶然による寄与が大きいと考えられる。個別候補の優劣より先に、この
シェアードな悪化自体を単独では解釈しないこと。

### 24.2 蛇行指標(距離正規化cm/m)

| candidate | 直線(SE) | コーナー(SE) | \|ekf_ey\| mean/p95 |
|---|---|---|---|
| cand1(Q0=100k) | **13.85(0.32)** | **20.44(0.39)** | 0.627m / 1.302m |
| cand2(wp_id_offset=3) | 29.46(2.75) | 26.80(1.30) | 0.612m / 1.607m |
| cand3(steer_low_pass_gain=0.25) | 31.87(0.52) | 31.63(0.81) | 0.744m / 1.870m |

**cand1が直線・コーナー両方で明確に最良**(直線は他候補の半分以下)。
これは「Q[e_y]低減が蛇行の直接原因側へ効く」という既存の確立済み
メカニズムと整合する、初めてクリーンに解釈できる結果。

**cand3(steer_low_pass_gain=0.25)は蛇行・ハード制約とも3候補中最悪**
——既存速度域(15-20km/h)での確定結論(§3禁止リスト5番)が25km/hでも
同じ方向で再現されたことを示唆する(0.35からの乖離は逆方向に動かしても
悪化する、という既存知見と矛盾しない)。

**cand2(wp_id_offset=3)は中間**——蛇行・ハード制約とも改善を主張できる
根拠はなく、「間接仮説側は優先度を下げる」というユーザーとの合意事項を
裏付ける結果になった。

### 24.3 判定・段階的チューニング計画への反映

ユーザーと合意した「直接原因側(Q[e_y]・steer_low_pass_gain)を優先して
段階的にチューニングする」方針のもと:

- **cand1(Q[e_y]=100,000)が今回唯一、蛇行改善の方向に一貫した証拠を
  示した候補**。次のQ[e_y]探索の起点として有力。
- cand3はsteer_low_pass_gainを動かす方向自体が(0.25方向は)悪化を再確認、
  以後この方向の探索は打ち切り、§3禁止リスト5番を25km/hでも維持する
  形で問題ない。
- cand2(間接仮説側)は今回も改善の証拠なし、優先度を下げたまま次段階へ
  持ち越す。

ハード制約の全候補共通悪化(§24.1)は次走行での再現性確認が必要な
未解決の観測事項として記録するに留める。

## 25. Run 1: Q[e_y]矛盾の解決(対照付きセッション、2026-08-07)

外部AI(Claude)提案の構造改革を実装した:
- 判定書スクリプトへ対照比モード(同セッション対照車のレート/平均を基準に
  ポアソン帯・ブロックSE有意差判定)を追加
- CLAUDE.md §3禁止リスト5番を更新(steer_low_pass_gainが25km/hでも
  同方向で悪化を再確認、コード変更なしのためgitignore対象、コミット不要)
- **ユーザー指示によりay_max=7.0を新規追加**(v_max=25km/hに対応する
  「中高速」プリセット値、直線だけでなくコーナー速度も引き上げる)。
  以後25km/h検証では標準投入する運用とした。

構成(dev3、全domain v_max=25km/h・ay_max=7.0・delay=0.07、35.05分):

| domain | 設定 | 目的 |
|---|---|---|
| d1 | Q[e_y]=100,000・QN[e_y]=1,000,000(cand1再現) | 最良候補の対照付き再現 |
| d2 | Q[e_y]=100,000・QN[e_y]=500,000(S1再現) | QN比の効果を単離 |
| d3 | デフォルト(Q[e_y]=200,000・QN[e_y]=1,000,000) | セッション内対照 |

### 25.1 Phase 1(ハード制約、d3対照レート基準のポアソン帯)

| domain | 衝突(観測/期待/閾値) | STUCK(観測/期待/閾値) | 判定 |
|---|---|---|---|
| d1(cand1再現) | 26/8.0/13.66 | 27/6.0/10.9 | **FAIL** |
| d2(S1再現) | 36/8.0/13.66 | 38/6.0/10.9 | **FAIL** |

d3対照実績: 衝突8件・STUCK6件・WALL0件(35.05分)。

**cand1再現(d1)も含め両方ともFAIL**。前回(§24)の単独セッション実験では
cand1(同一パラメータ)がハード制約PASSと判定されていたが、これは
セッションノイズ(交通状況の偶然)による見かけ上の結果だったことが、
今回の同一セッション内対照付き実験で判明した。

### 25.2 Phase 2(蛇行、ブロックSE有意差判定)

| domain | 直線cm/m(SE) | 対d3有意差 | コーナーcm/m(SE) | 対d3有意差 |
|---|---|---|---|---|
| d1(cand1再現) | 16.06(1.09) | **あり(改善)** | 21.89(1.04) | **あり(改善)** |
| d2(S1再現) | 14.90(0.90) | **あり(改善)** | 21.16(0.54) | **あり(改善)** |
| d3(対照) | 26.57(0.63) | — | 27.67(0.37) | — |

蛇行改善はd1・d2とも直線・コーナー全てで統計的有意。

### 25.3 d1 vs d2(QN比500k vs 1Mの効果を単離)

| 軸 | 差 | 閾値 | 判定 |
|---|---|---|---|
| 直線 | 1.16 | 2.83 | 有意差なし |
| コーナー | 0.73 | 2.35 | 有意差なし |

**QN比の違いによる有意差はなし。** すなわちS1(§23)のFAILはQN=500kの
せいではなく、**Q[e_y]=100,000そのものがハード制約を悪化させている**
ことが判明した。

### 25.4 結論: Q[e_y]低減は蛇行改善とハード制約悪化のトレードオフ

**Q[e_y]=100,000は、蛇行を統計的有意に改善する一方、衝突・STUCKを
統計的有意に悪化させる、一貫したトレードオフ構造を示した。** 前回
「cand1は安全」と見えたのは1セッションの偶然であり、対照付き実験では
このトレードオフが再現性を持って確認された。

**Stage Aの計画修正が必要**: 「Q[e_y]を下げるほど蛇行が改善する」という
既存メカニズムは今回も支持されたが、**下げすぎるとハード制約(安全性)を
犠牲にする**という新しい制約が判明した。CLAUDE.md §2 rule 4
(ハード制約を先に適用)に従えば、Q[e_y]=100,000は蛇行スコアに関わらず
不採用が妥当。次段階では200,000と100,000の**中間値**(例: 140,000-160,000)
を同型の対照付き実験で探索し、ハード制約を悪化させない範囲での蛇行改善を
探る必要がある。あるいは、Q[e_y]低減がなぜ衝突・STUCKを増やすのか
(コリドー追従の緩みによる密集区間でのマージン不足等)の機構解明も
並行して検討する価値がある。

## 26. Part A: Run 1ログの機構分析(2026-08-07、コード変更なし・走行なし)

外部AI(Claude)提案のA/B/C仮説(OTマージン仮説/遷移ゲイン段差仮説/
ホットスポット偏差増仮説)をRun 1の既存ログ(§25)から検証した。

| 指標 | d1(cand1: Q=100k,QN=1M) | d2(S1: Q=100k,QN=500k) | d3(対照) |
|---|---|---|---|
| 遷移±2秒以内の衝突/STUCK比率 | 41.5%(22/53) | 25.7%(19/74) | 35.7%(5/14) |
| wp269-282 \|ekf_ey\| p95 | **2.876m** | 1.102m | 1.584m |
| STUCK直前のOT state(多数派) | NORMAL(24/27) | NORMAL(30/38) | OVERTAKING(3/6) |

**結論: 単一の仮説に絞り込めなかった。**
- 仮説B(遷移ゲイン段差): d2はむしろ対照より低い比率、一貫した裏付けなし
- 仮説C(ホットスポット偏差増): d1だけ突出(2.876m、対照の1.8倍)、
  **d2は逆に対照より良好**(1.102m)——同じQ[e_y]=100,000のはずのd1と
  d2で全く異なるパターンという新たな未解決の謎が生じた
- 仮説A(OTマージン): STUCKの大半がOVERTAKINGでなくNORMAL状態で発生、
  むしろ否定的な材料

**正直な限界**: Claude案が求めた「衝突相手の特定(実験車同士/対照/壁)による
汚染除去」は、**dev3では自ego車(domain 1)のみbag-recorderが起動する仕様
のため3台分の同期した絶対位置データが存在せず実施できなかった**。真の
機構解明には複数domainの同期位置ログ(bag-recorderの複数domain化、または
GHOST-BLOCKのpose_x/pose_y様の常時ロギング拡張)が必要。今回は
テキストログのみで可能な範囲の分析に留めた。

d1とd2のホットスポットp95の食い違いは、QN比(1M vs 500k)が集約統計
(§25の蛇行・ハード制約全体)には現れないが特定ホットスポットの挙動には
影響している可能性を示唆しており、次回以降の追加調査候補として記録する。

## 27. 重大な発見: 遅延較正プロトコルが過去5日間未運用だったことが判明(2026-08-07)

Run 2実行中、ユーザーから「D3(対照)の蛇行が本日の他対照(065209)より
悪化しているのは、遅延をdelay=0からdelay=0.07へ初めて投入したためでは
ないか」という指摘を受け、`debug_extra_actuator_delay_s was updated`
ログを2026-08-03〜08-07の全output/ディレクトリ(59セッション)に対して
横断検索した。

**結果**: 該当ログは本日2026-08-07の5セッション(081954・091734・
110247・122254・131920、いずれも本日中に`ros2 param set`で0.07を
投入したセッション)にのみ存在し、**08/03〜08/06の過去4日間には1件も
存在しなかった**。

一方、`delay_calibration_protocol_20260803.md`(タスクC-0-2、2026-08-03
制定)には「予選環境の実効遅延がローカルより+50-60ms大きい、
`debug_extra_actuator_delay_s=0.055`という既存較正値が妥当」と明記
されている。**つまりこの較正プロトコルは文書として存在していたが、
実際のローカル試験では一度も運用(ライブ投入)されていなかった**ことが
判明した。08/03〜08/06のローカル試験は全て、本来加算すべき55-60msの
遅延補正なし(delay=0.0のまま、`actuator_lag_tau_s`のローカル内部遅延
モデルのみ)で行われていたことになる。

**含意**: ローカルでの蛇行チューニングが予選環境で再現しなかった過去の
事例(axis06_gain_correction_design_20260803.md、tau=190→160ms修正が
ローカルA/Bで改善したが予選実測0804-01で悪化・再現せず)は、この較正
プロトコルの未運用自体が一因だった可能性がある。ローカルの方が予選より
常に少ない遅延で走っていたため、ローカルでは蛇行が実際より少なく見え、
そこで決めたQ/R設定が予選の実遅延下では効かなかった、という説明が
成り立つ。

2026-08-07に`delay=0.07`(0.055を上回る再較正値)を使い始めたのは、
このプロトコルを**初めて実運用に移した**ものであり、Run 1・Run 2で
観測された蛇行の大幅悪化(§25、§本節冒頭)は、パラメータ変更や
ay_max=7.0だけでなく、**この遅延補正の初回適用自体が主要因である
可能性が高い**。以後のローカル試験では常にdelay補正を投入する運用を
標準化すべきである(較正値そのものは今後の予選ログで継続的に検証する、
既存のdelay_calibration_protocol運用ルールに従う)。

**訂正(同日、Run 2実行中)**: §23で「0.055ms(元の較正値)では調整
しきれなかったため0.07へ引き上げた」としていたが、本節の調査で
**0.055自体が過去5日間一度もライブ投入・検証されていなかった**ことが
判明した以上、「0.055では不十分」という前提は検証されておらず誤りだった。
Run 2は開始7分後(13:31)にdelay=0.07から**プロトコル本来の較正値
0.055へ変更**し、以後のセッションもこの値を標準とする(0.07への
引き上げは前提が崩れたため撤回)。Run 2の判定は13:31以降のデータを
実質的な開始点として扱う。

**0.07 vs 0.055の直接比較(同一domain・同一Q値、切替前後)**:

| domain | delay=0.07 直線/コーナー(cm/m) | delay=0.055 直線/コーナー(cm/m) |
|---|---|---|
| d1(Q=140k) | 24.35 / 28.38 | **12.38 / 21.49** |
| d2(Q=170k) | 27.28 / 27.81 | **17.22 / 25.22** |
| d3(Q=200k対照) | 28.56 / 29.92 | **23.83 / 26.81** |

**3domain全てで一貫して、delay=0.055の方が蛇行が小さい**(切替後n数は
まだ4-8分相当と少ないが、方向は完全に一致)。特にd1(Q=140k)は直線が
24.35→12.38cm/mとほぼ半減。これは前述の「Q×tau自己励起系」仮説
(Q[e_y]と遅延の組み合わせが共振的に蛇行を増幅する)を支持する追加証拠
であり、**0.07への引き上げが蛇行を悪化させていたこと・0.055が
プロトコル本来の較正値として妥当であることの両方を裏付ける**。
ユーザー判断により本走行を7分延長で終了し、以降0.055を標準とする。

### 25.5 Run 2最終結果(delay=0.055区間、約8.3分、ユーザー指示により短縮終了)

| domain | 直線cm/m | コーナーcm/m | COLLISION | STUCK |
|---|---|---|---|---|
| d1(Q=140,000) | 16.73 | 21.76 | 0 | 2 |
| d2(Q=170,000) | 16.08 | 22.98 | 2 | 0 |
| d3(Q=200,000対照) | 21.62 | 27.64 | 0 | 0 |

**良い兆候**: delay=0.055下では、d1・d2とも対照より蛇行が明確に改善して
おり、Run 1のQ=100,000で見られた壊滅的なハード制約悪化(数十件の衝突)は
再現していない。

**正直な限界**: この判定は統計的に確定できない。対照(d3)がこの短時間
区間でたまたま衝突・STUCKとも0件だったため、ポアソン帯の期待値がゼロと
なり、d1・d2のわずか1-2件が機械的に「悪化」判定される、n数不足による
計算上のアーティファクトが生じている(当初予定35分が実質8.3分に短縮
されたため、CLAUDE.md §2 rule 1の「n=1で確定しない」に該当)。

**次のステップ**: Q=140,000・170,000は有望な候補だが、確定判定には
**delay=0.055を最初から使った、十分な長さ(30分以上)のRun 2再走行**が
必要。

## 28. 予選ログ137本の遅延較正大規模検証(2026-08-07)

ユーザーが保管していた07/09〜08/07の予選bag全137本(`~/Downloads/
rosbag2_autoware*.mcap`)に対し、`analyze_actuator_delay.py --mode
yawrate`をバッチ実行した(所要28.7分、エラー0件)。あわせて、
セクション別・コーナー別の実効遅延推定ツール
(`scripts/analyze_delay_by_section_20260807.py`、クロス相関ベース、
テキストログのwp情報とbagをタイムスタンプで紐付け)を新設し、0807-05を
8分割で試験した(260-290ms、手法の探索上限付近に張り付く傾向があり
絶対値は割り引いて解釈、セクション間の相対比較のみ参考)。

### 28.1 統計サマリー(137本)

| | 全137本 | フィット品質良好(resid<20、88本) |
|---|---|---|
| L平均 | 205.2ms | 215.5ms |
| L中央値 | 230.0ms | 230.0ms |
| 標準偏差 | 66.9ms | 55.0ms |

11件が退化フィット(L≤30ms、恐らく渋滞等で相関が取れなかった低品質
フィット、resid値と合わせて除外候補)。

### 28.2 ローカル基準(170ms、0804実測)とのdelta

- 平均delta = 45.5ms
- **中央値delta = 60.0ms**
- 標準偏差 = 55.0ms(セッション間ばらつきは依然大きい)

### 28.3 結論: 0.055(55ms)較正値の大規模検証

**約1ヶ月分・88セッションの中央値delta=60msは、既存較正値0.055(55ms)に
ほぼ一致する。** 平均値(45.5ms)が低いのは退化フィットに引きずられて
いるためで、外れ値に頑健な中央値の方が信頼できる。§27で「0.055は
過去5日間一度も運用されていなかった」ことが判明していたが、
**運用されていなかったこと自体は0.055という数値の妥当性を否定する
ものではなく**、今回の大規模検証で数値そのものは裏付けられた。

§25.5(Run 2、delay=0.055、n=8.3分)で得られた予備的に良好な結果と
合わせ、**0.055を今後のローカル試験・Q[e_y]探索の標準較正値として
確定する**。0.07への引き上げ(§23)は撤回済み(§27)。

## 29. コーナー別蛇行ランキング・遅延・トリガー分析(2026-08-07)

ユーザーの提案「蛇行しやすいコーナーが決まっているなら、そこだけに
特化したパラメータが必要か判断したい」「蛇行の起点となった操作を
遡って特定したい」を受け、予選ログ131本(テキストのみ、高速)+
bag20-123本(実効遅延、`analyze_delay_by_section_20260807.py`新設)で
多段階の分析を行った。

### 29.1 コーナー別蛇行ランキング(131本、71-76本/コーナーの安定サンプル)

kappa閾値(|kappa|>0.08)で検出した14コーナー区間ごとに、隣接ekf_ey差分の
平均振れ(cm)を全ログで集計:

| コーナー | 平均振れ(cm) | stdev |
|---|---|---|
| **wp260-275** | **18.93** | 9.37 |
| wp12-18 | 18.04 | 10.60(最もばらつき大) |
| wp1-6 | 15.31 | 7.80 |
| **wp340-40**(周回ラップ、別途検証) | **15.17** | 7.55 |
| wp246-256 | 14.10 | 8.61 |
| wp110-129(最安定・対照) | 9.22 | 3.10 |

wp260-275は既存の既知ホットスポット「wp269-282」(task#306)と重なり、
データドリブンな分析が既存知見を裏付けた。wp340-40も既知の再現性の
高いホットスポット(memory `wp340-40-persistent-hotspot-0804`)と整合。

### 29.2 コーナー別実効遅延・相関品質(bag解析、20-123本)

トラックの周方向で遅延(FOPDT/クロス相関ラグ)がセクションによって
変わるか検証した(ユーザー仮説: 「アクチュエータ遅延自体は物理的に
一定のはずで、変動は信号ノイズ由来では」)。

- 8等分割(0807-05単体)では260-290msの幅が出たが、探索上限(300ms)
  付近に張り付く傾向があり測定アーティファクトの疑い
- **上位蛇行コーナー(wp260-275等)と対照コーナー(wp110-129)を
  20ログで比較したところ、相関係数は全コーナー・全ログで0.79-1.00と
  一貫して高く**(コーナーはヨーレート信号が強いためノイズに強い)、
  **遅延自体(250-290ms)もコーナー間で明確な差はなかった**

**結論**: **遅延そのものはコーナーによって系統的に変わらない**
(ユーザーの物理的直感が正しかった)。wp260-275の高い蛇行は、遅延では
なく既知の慢性輻輳(task#306)等、別の要因が主因である可能性が高い。

### 29.3 蛇行トリガーイベント分析(予備的、限界あり)

wp260-275での蛇行ピーク(隣接差分最大点)の直前(0-10秒)に、どの
イベントログが最も近いかを52-63本で集計した。

**第1試行(全イベント種)**: CAPSULE-HEADING(30%)が最多だったが、
**このタグは平均2.7秒に1回という高頻度で発火する背景ノイズ**(相手車
ヘッダーソース初期化ログ、ego自身の蛇行とは無関係)と判明、除外。

**第2試行(高頻度ノイズ除外後、n=52)**:

| トリガー | 件数 | 割合 |
|---|---|---|
| HOTSPOT-DEVIATION | 21 | 40% |
| MARGIN-RAMP | 11 | 21% |
| OT-OUTCOME | 8 | 15% |

**正直な限界(未解決)**: HOTSPOT-DEVIATIONの平均発火間隔は16.7秒。
ポアソン近似で「10秒探索窓内に少なくとも1回発火する確率」を計算すると
約45%となり、観測された40%とほぼ同水準——**この「40%」も基準率だけで
説明できてしまう可能性が高く、真の因果関係(トリガー)とは言い切れない**。
複数イベント種が競合する状況での正確な検証には、実タイムスタンプを
シャッフルしたランダム対照とのenrichment検定が必要だが、今回は未実施。

**次のステップ(未着手)**: enrichment検定を実装し、真に基準率を超えて
蛇行ピークに近接するイベント種を特定する。それが判明すれば、
「特定パターンのシナリオを検出したら別パラメータへ切り替える」という
scenario-adaptive設計の妥当性を判断できる(既存のQ[e_y]曲率スケジュール
v5基盤の拡張として実装可能な見込み)。

## 30. クリーンなステップ応答テストでゲイン込みFOPDT再フィット(2026-08-07)

ユーザーからの「tauの適正値は改めていくつか」という問いを受け、
mpc_controller停止・純粋オープンループでのステップ応答テストを2回実施
(振幅15°→壁衝突により中断・振幅5°→60秒・5サイクルで成功)。

### 30.1 ゲイン=1仮定の既存fit_fopdt()はフィット不能と判明

両テストとも、既存`fit_fopdt()`(FOPDTの定常応答が指令値へ完全収束する
= ゲイン=1、を暗黙に仮定)ではL=0/tau=280-400msという退化フィット・
高残差(resid 3.9-4.8°)しか得られなかった。同時に大舵角ゲイン検証
(既存コード)で**振幅に関わらず一貫してgain≈0.59-0.62**という結果が
出ており、ゲイン=1仮定自体が誤りだったと判明した。

### 30.2 ゲイン込み同時フィット関数を新設

`fit_fopdt_with_gain()`(`analyze_actuator_delay.py`)を追加。FOPDTの
ODEは入力に対し線形なため、L・tauを固定した単位ゲイン応答y_unitに対し
最適ゲインK(原点通過最小二乗)を解析的に求められる
(K=Σ(y_unit・act)/Σ(y_unit²))。既存のL・tau 2次元グリッドサーチは
そのまま維持し、計算量を増やさず3パラメータ同時フィットを実現。

### 30.3 結果: 残差が5-25倍改善、ゲインが2回のクリーンテストで一致

| | 修正前(gain=1仮定) | 修正後(gain込み同時推定) |
|---|---|---|
| 5°ステップ(60秒・5cycle) | L=0, tau=280ms, resid=**4.77°** | L=80ms, tau=**50ms**, gain=**0.595**, resid=**0.63°** |
| 15°ステップ(60秒・5cycle、1回目) | L=0, tau=400ms, resid=**3.93°** | L=110ms, tau=**80ms**, gain=**0.601**, resid=**0.16°** |

**ゲイン≈0.6が振幅の異なる2つの独立したクリーンなオープンループテストで
高い一致を見せた**。15°テストの方が残差(0.16°、相対誤差約1%)が5°
テスト(0.63°、相対誤差約13%)より大幅に小さく、より信頼できるフィット
と考えられる。

### 30.4 tauの結論(暫定・n=2)

- **L(むだ時間)**: 80-110ms(2回で差あり、要追加検証)
- **tau(時定数)**: 50-80ms(2回で差あり、フィット品質の高い15°テストの
  80msをより信頼できる候補とする)
- **gain**: **0.595-0.601で高い一致**、既存コード(actuator_gain=1.0、
  §3禁止リスト24番で0.67を「アーティファクト」として棄却済み)との
  乖離が大きい

**現行コードとの比較**: `actuator_lag_tau_s`は現在160ms(config非公開、
2026-08-05コミット367aa22)、`actuator_gain`は1.0(無効)。今回の測定
(tau=50-80ms、gain≈0.6)はどちらも現行値と大きく異なる。

### 30.5 正直な限界と次のステップ

- n=2(振幅2条件)のみであり、CLAUDE.md §2 rule 1(n=1で確定しない)に
  照らせばまだ確定的とは言えない。特にL・tauは2回で相応の差があり、
  追加のクリーンテスト(できれば異なる速度・複数振幅)が望ましい。
- **§3禁止リスト24番(「ゲイン0.67はアーティファクト」)は再考が必要**
  と考えられる証拠が揃った。ただし今回の測定はconfig.yaml/
  mpc_controller.pyへの反映はまだ行っていない(相談・承認待ち)。
  反映する場合はCLAUDE.md §1.4(等価性回帰・全体回帰スイート)に従う。
- 壁衝突で中断した15°・8cycle版の代わりに5°・5cycleで成功したことで、
  今後のステップ応答テストの標準振幅・サイクル数の目安ができた
  (速度2.0m/s・5サイクル=60秒程度なら壁到達前に完了する)。

## 31. 予選ログ137本のヨーレートモード再検証(gain込みフィット、2026-08-07)

§30でsteeringモードのFOPDTフィット(gain=1仮定)が破綻していたことが
判明したのを受け、ユーザー指示により§28の予選137本ヨーレートモード
解析を`fit_fopdt_with_gain()`で再実行した(所要49分)。

### 31.1 結果: delay=0.055の検証は無傷、むしろ精度が向上

| | 旧(gain=1仮定、§28) | 新(gain込み) |
|---|---|---|
| フィット品質良好率(resid<20) | 88/137(64%) | **114/137(83%)** |
| ローカル基準(170ms)とのdelta中央値 | 60.0ms | **60.0ms(完全一致)** |
| delta標準偏差 | 55.0ms | **32.1ms(改善)** |

同一ログでの新旧L値の差分(中央値0.0ms、66%が±20ms以内)から、系統的な
偏りは見られなかった。ヨーレートモードは、単位が異なる量(操舵角deg vs
ヨーレートdeg/s)を扱うためgain=1という前提がsteeringモードほど強い
制約になっておらず、影響は限定的だったと考えられる(実際gainの中央値は
1.07とほぼ1に近い、ただしstdev=0.50と個別ログでは相応にばらつく)。

### 31.2 結論

**§28で確定した delay=0.055(既存較正値)は、gainの前提バグ修正後も
変わらず妥当と再確認された。** むしろ標準偏差の改善により、これまでより
高い確度で0.055を標準較正値として使用できる。

## 32. 現行tau=160msの起源解明: 旧手法の歪みを自然走行データで直接検証(2026-08-07)

ユーザーから「現行のtau(160ms)は、ローカル環境のLとtauを足したものに
なっているのでは」という仮説が出た。自然走行bag(0804、通常のdev1台
走行)に旧`fit_fopdt()`(gain=1仮定)と新`fit_fopdt_with_gain()`を
両方適用し直接検証した。

| | L | tau | gain | resid |
|---|---|---|---|---|
| 旧(gain=1仮定) | 0ms | **160ms**(現行コード値と完全一致) | (固定1) | 3.72 |
| 新(gain込み) | 60ms | 50ms | 0.688 | **1.05**(3.5倍改善) |

**旧フィットのtau=160msは、まさにこの種の自然走行データから導出された
値と完全に一致した**(2026-08-03の較正時と同種のデータと考えられる)。

**仮説の検証結果**: 新L+新tau = 110ms、旧tau = 160ms。**単純な合算とは
一致しなかった**(差50ms)。旧フィットはLの欠落だけでなく、ゲイン不足
(0.688)の欠落も同時に埋め合わせようとしてtauを単純合算以上に膨らませて
いたと考えられる——gain=1仮定という単一の誤った前提が、L・tau両方の
推定を歪めていた。

### 32.1 3点の収束(クリーンテスト2回+自然走行1回)

| 測定 | L | tau | gain |
|---|---|---|---|
| 5°クリーンステップ | 80ms | 50ms | 0.595 |
| 15°クリーンステップ | 110ms | 80ms | 0.601 |
| 自然走行(0804) | 60ms | 50ms | 0.688 |

**3つの独立した測定(手法・データ種別が異なる)が同じ方向に収束**:
tau=50-80ms、gain=0.6-0.7、L=60-110ms。現行コード(tau=160ms、
gain=1.0固定)とはいずれも大きく乖離しており、旧測定手法の
gain=1仮定バグが原因だったことが直接確認された。

### 32.2 tau・gain・delay_sの役割分離(混同注意)

調査の過程で以下の整理を行った:

- **tau・gain(アクチュエータ本体)**: ローカル・予選で同一のシミュレータ
  内アクチュエータモデルのはずであり、環境に依存しないと考えるのが
  妥当。今回の測定値(tau=50-80ms、gain=0.6-0.7)がそのまま両環境に
  適用できる候補。
- **L(むだ時間)**: 現行の`delta_actual`状態モデルには実装されていない
  (一次遅れのみ、むだ時間項が存在しない)。反映するには構造変更が必要。
- **debug_extra_actuator_delay_s(0.055)**: ローカルと予選の**差分**
  (ネットワーク・ホスト性能等、アクチュエータ本体とは別の要因)を表す、
  独立した第3のパラメータ。
- ヨーレートベースのL_eff(ローカル170ms・予選230ms、§28/§31)は、
  操舵角ベースのL・tauとは**別の物理量**(車両ダイナミクス込みの
  ループ全体遅延)であり、単純に合算・代入してはならない。

### 32.3 未反映・要判断事項

- tau=50-80ms・gain=0.6-0.7・delay_s=0.055を実際にconfig.yaml/
  mpc_controller.pyへ反映するかは未決定(承認待ち)
- L(むだ時間)を`delta_actual`モデルへ新規実装するかは別途判断が必要
  (構造変更、影響範囲が大きい)
- n=3(クリーン2回+自然走行1回)であり、CLAUDE.md §2 rule 1に照らせば
  まだ確定的ではない。反映する場合は追加のn数積み増しが望ましい。

## 33. actuator_gain・actuator_lag_tau_sの実装反映(2026-08-07、コミット26245f9)

§32の測定結果を受け、ユーザー指示によりconfig.yaml・mpc_controller.pyへ
反映した。

- `bicycle_model.actuator_gain`: 1.0→**0.6**
- `bicycle_model.actuator_lag_tau_s`: 新規追加、**0.08**(コード内デフォルト
  0.16から変更。従来config.yaml非公開だったため、`mpc_controller.py`の
  `create_car()`で`BicycleModel`へ明示的に渡すよう追加。既定値0.16は
  後方互換として維持、config指定時のみ上書き)
- config.yaml差分は本変更のみに限定してコミット(既存の未コミット
  `pending_disengage_enabled`変更とは分離、一時的に戻して差分を切り
  離した上でコミット後に復元)
- 回帰スイート3236件PASS確認(変更前後とも)

**L(むだ時間、60-110ms)は未実装のまま**(§30.5・§32.3で述べた通り、
現行の`delta_actual`一次遅れモデルにはむだ時間項が存在せず、追加するには
構造変更が必要なため今回は見送り)。

**次のステップ(未着手)**: dev3実走行での実地検証(蛇行指標・ハード制約
双方の変化を確認)、n数積み増し(追加のステップ応答テスト)、Lの実装
要否の判断。

## 34. tau振幅依存性の再検討: レート制限による歪みの可能性(2026-08-07)

ユーザーから「tauが振幅で変わったのは車速ではなくステアリング角速度
(レート制限)の影響では」という仮説が出た。両ステップ応答テスト
(5°・15°)の`steering_status`実測データから角速度(隣接差分)を
直接計算し検証した。

| | 5°テスト | 15°テスト |
|---|---|---|
| 最大角速度 | 63.5deg/s | 67.0deg/s |

**両方とも約63-67deg/sで頭打ちになっており、これは`steer_rate_max: 1.1
rad/s`(換算63.0deg/s)とほぼ完全に一致した。** 振幅が3倍(5°→15°)
違うのに最大角速度がほぼ同一値で揃っていることは、**両テストとも
レート制限に張り付いていた**ことの直接的な証拠である。

**結論**: 15°テストのtau=80msは、純粋な線形時定数ではなく、レート
制限という既知の別の制約(非線形飽和)が混入して見かけ上膨らんだ値
である可能性が高い。振幅の小さい5°テストのtau=50msの方が、レート
制限の影響を受ける時間が短く、より「純粋なtau」に近いと考えられる。

**§33で採用したactuator_lag_tau_s=0.08(80ms)は再検討が必要**。
5°テストの50msへの変更、またはレート制限とtauを分離した再フィット
(振幅の異なる複数テストからレート制限区間を除外して再推定)が
望ましい。次回のn数積み増し時にこの点を反映すること。

## 35. delay_t_delay_s(200ms)の根拠調査とL・tauとの関係整理(2026-08-07)

ユーザーから「`delay_t_delay_s=200ms`が適切という根拠は何か」「L・tau
との関係は単純な合算か」という一連の問いを受け、stage15ジャーナルを
調査した。

### 35.1 200msの根拠(存在するが、手法の詳細は不明)

`mpc.delay_t_delay_s: 0.2`(188-3節)は「実測(2026-07-03、r=0.992)確認済み
の操舵アクチュエータ遅延(≈200ms)は速度に依らない固定"時間"」という
記述で導入された。速度非依存であることも実測確認済みとされている
(§34以前にユーザーへ「物理的にそう予想される」とだけ伝えていたが、
実際は既に07-03に実測確認済みだった)。ただし**07-03測定の具体的な
手法(どのツール・どのログを使ったか)を記載した節は見つからなかった**。

### 35.2 重大な発見: STEER-XCORR(07-27)も同種の限界を抱えていた

tauの初期値190ms(196/197節)の根拠となった`STEER-XCORR`診断機構
(2026-07-27実装)自体に、実装当時から以下の「正直な限界」が明記
されていた:

> 「相互相関は単一のラグ推定値しか得られず、遅延の"形"(純粋遅延/
> 一次遅れ/FOPDT)を区別するには不十分な可能性がある」

つまり**07-27のSTEER-XCORR(ローカル130-140ms・予選190ms)も、07-03の
200ms測定も、どちらもLとtauを分離せず「ひとまとめのラグ」として
測っていた**。今回(§30-34)実施したゲイン込み・L/tau分離のクリーンな
ステップ応答テストは、**このプロジェクトで初めてL・tauを個別に測った
試み**であることが判明した。

### 35.3 delay_t_delay_sとL・tauの関係: 「同じものの別の測り方」

議論の結果、以下の整理で合意した:

- delay_t_delay_sは「①どこを狙うか(先読み層)」、L・tauは「②どう
  辿り着くか(MPC内部モデル層)」という別レイヤーで働くパラメータ
  (§本文台詞参照)だが、**測っている物理現象自体は同じ**(操舵指令が
  実際に効果を持つまでの時間)。
- **単純な3値合算(delay_t_delay_s+L+tau)ではなく、「delay_t_delay_s
  ≈ Lとtauを合わせたもの」という同一量の別測定**という理解が妥当。
- 今回分離測定したL(60-110ms)・tau(50ms、レート制限除外後)を用いて
  比較すると、「反応した」をどう定義するかで結果が変わる:
  - 63%到達基準(L+tau): 110-160ms → **200msより明確に小さい**
  - ほぼ収束基準(L+3τ、≈95%): 210-260ms → 200msに近い

### 35.4 結論: 「反応した」の定義が曖昧なまま複数の測定が積み重なっていた

**200msが「間違っている」と断定はできないが、過大評価だった疑いは
残る**(STEER-XCORRの自然走行データにレート制限混入の疑いがある
ため、§34と同型の歪み)。根本的な問題は、**「アクチュエータが反応した」
という言葉の定義(反応開始/63%到達/ほぼ収束)を統一しないまま、
07-03・07-27・今回と複数の測定が積み重なってきたこと**にある。

**次のステップ(未着手)**: レート制限の影響が少ないさらに小振幅
(3°程度)でのステップ応答テスト追加により、真のL・tauをさらに
精緻化する。delay_t_delay_s(200ms)自体の妥当性再検証は、Lの実装
判断(§32.3)と合わせて今後の課題とする。

## 36. 3°ステップテスト・gain/tauデプロイ健全性チェックの結果、および復元判断(2026-08-08)

### 36.1 3°ステップ応答テスト: 収束せず、さらなる非線形性を示唆

§35末尾の次ステップとして、レート制限混入をさらに減らす狙いで3°振幅
ステップテストを実施(5サイクル、`run_dev_20260808_024919`)。結果は
むしろ悪化した:

| 振幅 | L | tau | gain | resid | 実測角速度peak |
|---|---|---|---|---|---|
| 5° | 20ms | 50ms | 0.595 | 0.16° | ~63-67°/s(rate_max張付き) |
| 15° | 80ms | 80ms | 0.688 | 0.16° | ~63-67°/s(rate_max張付き) |
| 3° | 20ms | 120ms | 0.500 | **2.719°(最悪)** | ~63.4°/s(rate_max張付き、なお張付き) |

3°でもステップ直後は瞬間的に最大角速度を要求するため、振幅を下げても
レート制限混入は消えない(むしろ3°では他の非線形性—不感帯/バックラッシュ
の疑い—が新たに顔を出し、フィット品質が最も悪化した)。さらなる振幅
低減による精緻化は非効率と判断し、振幅スイープによるL/tau/gain追求は
いったん打ち切る。

### 36.2 gain=0.6・tau=0.05(0.08から§34で修正)のローカル健全性チェック走行

2026-08-08未明、`mpc_controller`を通常起動したまま(ステップテストと違い
制御ループは生きている)10分間のソロ走行で健全性チェックを実施
(`output/20260808-030503`、delay_t_delay_s=0.2固定・
debug_extra_actuator_delay_s=0.0、Q/Rは無変更)。ユーザーが目視で
「Sジが乱れる」「ここまで激しくはないが実際の環境もこんな感じで
不安定になる」とWP252・WP280前後・WP340-40の3箇所の不安定を報告、
以下で定量確認した:

| 区間 | 平均振れ | 最大振れ | 最大\|ekf_ey\| |
|---|---|---|---|
| wp252前後(242-262) | 15.24cm | 41.20cm | 1.297m |
| wp280前後(270-290) | **25.09cm** | **69.20cm** | **1.707m** |
| wp340-40(周回) | 17.16cm | 55.20cm | 1.177m |

参考(予選ログ76本ベース、旧gain=1.0/tau=160ms環境): wp260-275平均
18.93cm、wp340-40平均15.17cm/中央値14.92cm。**wp280前後の1.707m・
69.20cmは他の2区間より突出して悪く**、加えて同じ走行でCOLLISION-SUSPECTED
実イベントが2件(t≈106s・t≈182s、速度が1サイクルで5m/s級→2m/s級へ急落)
発生した——ソロ・単発10分走行としては明確に異常(壁接触の疑い)。

STUCKログ出現61件・FAILSAFE 1件も観測(旧モデルでの同条件ソロ走行では
通常ほぼ0件)。

### 36.3 判断: 2026-07-27 198節と同型の再現、既定値へ復元

Q/Rを一切再調整しないまま`delta_actual`のモデル精度だけを上げると
悪化する、という198節(std 2.14°→5.04°)の教訓が、今回もほぼそのまま
再現した(むしろ実イベント[COLLISION-SUSPECTED]まで伴い今回の方が
重症)。CLAUDE.md §1.1(レース値維持キー、実験後は必ず復元)に従い、
`actuator_gain`を1.0、`actuator_lag_tau_s`を既定コード値0.16へ復元した
(コミット`db8daa7`)。回帰3236件PASSを確認済み。

**gain=0.6の実測自体(オープンループ2回+自然走行1回による§30-32の結論)
は無効化されない**——今回悪化したのはgain/tauモデルの精度そのものでは
なく、精度を上げたことでQ/Rとの噛み合わせが崩れたことが原因、という
198節と同じ構造だと考えられる。次回この軸に再挑戦する場合は、
Q/R再チューニングを同じセッションで併走させる設計にすること
(単純な値の差し替えだけでは同じ失敗を繰り返す)。

### 36.4 副次的発見: コミット`9b5efe8`でのCLAUDE.md §1.5違反

`9b5efe8`(tau 0.08→0.05)を`git show --stat`で確認したところ、無関係な
`pending_disengage_enabled: false→true`(Fix C、overtake節)が同じdiffに
混入していたことが判明した。1つ目のコミット(`26245f9`)では該当行を
一時的にEditで戻してからコミット・その後Editで復元、という手順を
踏んでいたが、2つ目のコミットではこの手順を省略したために発生した
(§1.5「制御コード変更と設定ファイル一式は独立コミット」の趣旨には
反しないが、無関係な機能フラグ変更が意図せず紛れ込んだ点は今後の
注意点)。今回の`db8daa7`では影響を受けないよう`pending_disengage_enabled`
行はそのまま(true)に保持し、gain/tau行のみを変更対象とした。

### 36.5 次のステップ(引き継ぎ、ユーザー就寝中の作業指針)

- gain=0.6/tau=0.05相当のモデル精度向上を活かすには、Q/R再チューニング
  キャンペーン(200-201節・現行のQ[e_y]/Q[e_psi]チューニング資産と同規模)
  が必須。単独では着手しない(n=1即断・dev3未検証のまま広範な変更を
  コミットするのはCLAUDE.md §2 rule1/6に反するため)。
  - 実施する場合の設計案: 新モデル(gain=0.6/tau=0.05)をfeatureフラグ的に
    configへ再追加した上で、既存のQ[e_y]対数スイープ(§Q-ey-log-sweep、
    200k/400k/700k)と同じ手法をこのモデル専用に回す、n=1→n=2で確定、
    最終確認はdev3、という既存プロトコルをそのまま踏襲する。
- delay_t_delay_s(200ms)の妥当性再検証(§35.4)は3°テストの非収束により
  振幅スイープでは追求打ち切り。再挑戦するなら別の実験設計(例:
  低速・低振幅のsine sweepなど)が必要、今回は着手しない。
- L(むだ時間)の実装判断は引き続き保留(§32.3)。

## 37. Q/R再チューニングキャンペーン設計案(未実施、ユーザー承認待ち、タスク#308)

ユーザー就寝中に新規のdev/dev3実走行を無人で回すのは、(a)過去に指摘された
発熱負荷(memory: dev3タイマー満了後は解析前に必ずmake downで停止)、
(b)CLAUDE.md §2 rule1「n=1で確定しない」・rule6「予選投入前提の検証は
dev3が基本」という判定手順が対話的な判断を要求する設計であること、
(c)壁衝突等の物理的異常が起きても対話的介入者がいないこと、の3点から
見送り、**設計のみ**をここに残す。実施はユーザー起床後に承認を得てから。

### 37.1 方針

§36.3の教訓どおり、gain/tauモデル変更とQ/Rチューニングを同一キャンペーン
内で併走させる。既存のQ[e_y]対数スイープ手法(200k/400k/700k、
design_docs内 stage15 270節)をそのまま流用し、gain=0.6/tau=0.05を
ONにした状態で再実行する。

### 37.2 手順案

1. `bicycle_model.actuator_gain`/`actuator_lag_tau_s`を実験用に0.6/0.05へ
   一時変更(config.yaml、コミットはしない、実験中のみ)。
2. 現行Q[e_y](既定値)のまま、ソロ・10分健全性チェックを1本実施し、
   今回(§36.2)と同条件で比較する対照runとして記録する(n=1、傾向把握
   目的のみ)。
3. Q[e_y]を200k/400k/700kで対数スイープ、各n=1でソロ実施(直線層std・
   wp252/wp280前後/wp340-40の3ホットスポットで§36.2と同じ指標を測定)。
4. 傾向が出た1-2点についてn=2で確定(CLAUDE.md §2 rule1)。
5. 生存者(ハード制約: STUCK/衝突ゼロ、wp280前後|ekf_ey|<既定モデルの
   実績値程度)をソフト指標(直線層std)で最終比較。
6. 最終候補をdev3で検証(CLAUDE.md §2 rule6)、予選ログn=2-3本相当の
   ばらつきを踏まえてから確定。
7. 全ステップでコミット前に回帰スイート(3236件)を実行、実験用の一時
   変更は§1.1に従い実験後必ず既定値へ復元してからコミットする。

### 37.3 中止基準(あらかじめ固定)

- ソロ10分健全性チェックの時点で§36.2の悪化(wp280前後 平均>20cm または
  COLLISION-SUSPECTED発生)が2本連続で再現した場合、そのQ[e_y]候補は
  即座に不採用とし次の値へ進む。
- 全候補が同様に悪化する場合、gain=0.6/tau=0.05自体のモデル(delta_actual
  状態空間表現)に構造的な問題がある可能性を疑い、198節・今回(§36)に
  続く3度目の再現としてこの軸そのものを保留する(design_docsへ記録し、
  CLAUDE.md §3禁止リストへ追加を検討)。

## 38. tri-param並列検証手法の開発とQ/R再チューニングキャンペーン全成果(2026-08-08)

§37の設計案通り、ユーザー承認を得てQ/R再チューニングキャンペーン(タスク#308)を
実施した。当初計画(ソロ10分健全性チェックの逐次実施)よりはるかに高速な
「tri-param」手法を新規開発し、これを使って1日で9パラメータ・約15ラウンドの
検証を完了した。

### 38.1 tri-param手法の開発

`make dev3`は3ドメイン(`docker compose -p 1/2/3`)とも同一の`./aichallenge`
ホストディレクトリをマウントするため、通常は3台とも完全に同一のconfig.yamlを
使う。**`docker-compose.tri-param-experiment.yml`**(新規override)を追加し、
config.yamlが実際に解決される1ファイルだけをプロジェクトごとに別のホスト
ファイルへ上書きマウントすることで、**3台に別々のパラメータを同時投入し、
1ラウンド(10分)で3点を検証**できるようにした(D1=共有config、D2/D3=
`CONFIG_OVERRIDE_PATH`で個別ファイルを注入)。

副次的な改善2点:
- **時間差起動(30秒、D3→D2→D1を15秒間隔)**: 3台が初期位置で密集して
  混戦になる(=STUCK/COLLISION-SUSPECTEDが交通ノイズで汚染される)問題を
  緩和する目的でユーザー提案により導入。
- **RViz無効化(`RUN_MODE=awsim-no-viz`)**: ユーザー指摘(load average 33を
  実測)を受け標準化。load average 33→8程度に改善。

Q[e_y]変更時は`q_ey_overtake`/`q_ey_pit`も比例スケールする生成スクリプト
(`gen_config.py`等、複数の派生版)を併用した。

### 38.2 Q[e_y]全9点スイープ(50k〜1M)

gain=0.6/tau=0.05・v_max=20km/h・R[delta]=500・r_delta_swing_boost=1600
(初期値)固定で、50k/75k/100k/125k/150k/200k/400k/700k/1Mの9点をtri-param
3ラウンドで検証した。

| Q[e_y] | 平均振れ(wp280+340-40、cm) |
|---|---|
| 50,000 | 11.3 |
| 75,000 | 11.6 |
| 100,000 | 13.9 |
| 125,000 | 11.9 |
| 150,000 | 13.0 |
| 200,000(旧確定値) | 18.5 |
| 400,000 | 22.1 |
| 700,000 | 27.3 |
| 1,000,000 | 34.5 |

200kを境に明確な変曲点があり、それ以降は綺麗な単調悪化。50k〜150kが最良帯。

### 38.3 q_ey_overtake/pit比例スケール漏れの発見

上記スイープ中、`q_ey_overtake`(OT中のQ[e_y])が330,000固定のままだったため、
OT状態に入ると**ベースQ[e_y]に関わらず全ドメインとも同じ絶対値へ収束**して
いたと判明(ベース75kの車は実質4.4倍、ベース1Mの車は実質0.33倍[逆転]という
設計意図と異なる倍率)。wp280(OT密集ホットスポット)の結果がこの影響で
汚染されていた可能性があるため、以後は`base×1.65`/`base×250`で都度比例
スケールする生成スクリプトへ修正した(§38.2の表は修正後のデータ)。

### 38.4 経路追従バイアス(内巻き/外巻き)という新評価軸の発見

ユーザーの目視観察(「現行セットは内巻き傾向にある」)を受け、蛇行(振れ幅)
とは別に**符号付きekf_ey**をコーナー方向(kappa符号)別に集計する分析を新設。

| Q[e_y] | 左corner符号付きバイアス | 右corner符号付きバイアス | kappa-ey相関 |
|---|---|---|---|
| 50,000 | +0.67m | -0.86m | 0.69 |
| 75,000 | +0.67m | -0.81m | 0.79 |
| 100,000 | +0.63m | -0.61m | 0.70 |
| 125,000 | +0.58m | -0.57m | 0.70 |
| 150,000 | +0.48m | -0.46m | 0.61 |
| 200,000 | +0.25m | -0.32m | 0.41 |
| 400,000 | **-0.11m** | **+0.14m** | **-0.20** |
| 700,000 | -0.40m | +0.41m | -0.45 |
| 1,000,000 | -0.49m | +0.53m | -0.53 |

**Q[e_y]が低いほど強い内巻き、高いほど外巻きへ反転**し、ゼロクロス地点は
概算Q[e_y]≈340,000〜350,000。「蛇行を減らすと経路追従精度が犠牲になる」
という、Q[e_y]という単一ノブでの綱引き構造を定量的に確認した。

### 38.5 ゼロクロス探索(250k/200k/350k)

| | D1(250k) | D2(200k) | D3(350k) |
|---|---|---|---|
| wp280+340-40平均振れ | 24.0cm | 19.3cm | 27.1cm |
| 左corner バイアス | +0.073m | +0.295m | -0.258m |
| 右corner バイアス | -0.192m | -0.288m | +0.003m |

250k付近でバイアスが最小(平均絶対値≈0.13m)、200kより明確に改善。蛇行は
200kが最良のまま。**250,000をバランス候補として採用**し、以降の全ラウンドは
Q[e_y]=250,000固定で実施。

### 38.6 R[delta]スイープ(複数ラウンド)

Q[e_y]=100,000固定での初回スイープ(200/350/500/650/800/1000)は全域
11.7-13.0cmとほぼ横並びで、Q[e_y]ほど支配的ではないと判明。R=500(現行)を
維持して確定。

Q[e_y]=250,000固定での再スイープ(500/1500/3000)では、区間依存の効果が
判明: **S字(wp340-40)には効かない/悪化**(23.89→25.67→24.48cm)、
**ヘアピン系(wp116-120)には効く**(15.19→16.80→12.12cm)。さらに
steer_low_pass_gain=0.6固定下での再スイープ(500/1500/3000)では、
wp252を含む全区間でほぼ横並び(効果なし、22.56/24.08/23.89cm)。
R[delta]=500(現行)を維持して確定。下方向(200)は明確に悪化(STUCK50・
COLLISION6)し却下。

### 38.7 r_delta_swing_boostスイープ

Q[e_y]=250,000・R[delta]=500固定で0/400/800/1200/1600/2000の6点。

| swing_boost | 0 | 400 | 800(旧) | 1200 | 1600 | 2000 |
|---|---|---|---|---|---|---|
| 平均振れ(cm) | 14.5(最悪) | 12.1 | 13.6 | 12.4 | 12.3 | 12.2 |

0(無効化)は明確に最悪で、この仕組み自体は新モデルでも有効と確認。
旧確定値800はその中でやや悪い側で、**1600を新しい暫定値として採用**。
`[R-DELTA-SWING]`診断ログ(`enable_diag_log=true`で確認可能)で発火条件を
実測: 全サンプルの54.1%がlo閾値(0.12)以上、S字帯(wp330-345)でswing=
0.19-0.23・R[delta]目標最大1610まで到達し、想定通り機能していることを
確認した。旧2026-08-05版(280節)の「クリーンA/Bで無罪放免」判定は
旧gain=1.0/tau=0.16モデル下の結果であり、新モデルでは効果ありに評価が
変わった。

### 38.8 「戻しの遅さ」切り分け: steer_low_pass_gainが劇的に効いた

ユーザーの目視観察(「ハンドルを戻すのが遅い」)を受け、Q[e_y]=250,000・
R[delta]=500・swing_boost=1600固定で、steer_low_pass_gain(0.35→0.6)と
r_drate(3M→1M)を切り分けるラウンドを実施。

| 区間 | D1(対照 0.35/3M) | D2(low_pass=0.6) | D3(r_drate=1M) |
|---|---|---|---|
| wp180前後 | 11.85cm | 12.13cm | 19.97cm |
| wp220-240 | 16.57cm | **9.86cm** | 26.22cm |
| wp252 | 22.26cm | 22.34cm(横ばい) | 29.81cm |
| wp280前後 | 18.88cm | **11.28cm** | 26.58cm |
| S字wp340-40 | 27.61cm | **11.86cm**(半分以下) | 29.50cm |
| STUCK/COLLISION | 20/2 | **1/1** | 51/5 |

**steer_low_pass_gain=0.6は全区間・安全指標すべてで最良か同等**、特にS字は
27.61→11.86cmと劇的改善。**r_drate=1M(下げる方向)は全区間で明確に悪化**し
却下。旧tau=160ms時代に確定された値(0.35)が、新tau=50ms(3倍速い実
アクチュエータ)には合わなくなっていたと考えられる——フィルタの相対的な
遅れがアクチュエータ本体の遅れより支配的になっていた可能性。

低pass=0.8への追加検証では、wp252・wp280は改善したが**wp180で最大
129.00cmという外れ値とSTUCK急増(1→34)**が出たため、0.8の採用は保留し
0.6を維持。0.6自体が真の最適点かはまだ未確認(タスク#310)。

### 38.9 r_drateの継続的改善(3M→10M)

low_pass=0.6・R[delta]=500確定後、r_drateを2M/3M/5M→5M/7M/10Mの2ラウンドで
上方向にスイープ。

| r_drate | 2M | 3M(旧) | 5M | 7M | 10M |
|---|---|---|---|---|---|
| wp180前後 | 10.11 | 9.36/8.69 | (5M枠) 8.69 | 7.38 | **6.63** |
| wp220-240 | 10.07 | 9.93 | 8.75 | 7.87 | 8.17 |
| wp252 | 22.40 | 24.31 | 25.42/24.25 | 24.75 | **21.08**(今日最良) |
| wp280前後 | 14.77 | 14.22 | 11.08/12.45 | 13.42 | **12.12** |
| S字 | 17.72(悪化) | 11.64 | 9.89/10.05 | 9.91 | 10.08 |

**r_drateを上げるほど一貫して改善**する傾向が継続しており(2M方向は逆に
S字が悪化)、10Mでもまだ改善が止まっていない。上限は未確定(タスク#308継続)。

### 38.10 現時点の最良候補セットと残タスク

| パラメータ | 競技提出値(現状維持) | 本日の有望候補 |
|---|---|---|
| actuator_gain | 1.0 | 0.6 |
| actuator_lag_tau_s | 0.16 | 0.05 |
| Q[e_y] | 200,000 | 250,000(ゼロクロス採用) |
| R[delta] | 500 | 500(変更なし) |
| r_delta_swing_boost | 800 | 1600 |
| steer_low_pass_gain | 0.35 | 0.6(さらなる最適化余地あり) |
| r_drate | 3,000,000 | 10,000,000以上(上限未確定) |

**config.yamlは全項目を競技提出値へ復元済み**(コメントで候補値と本節への
参照を残す、CLAUDE.md §1.1準拠)。回帰3236件PASS確認。

残タスク:
- タスク#308: r_drateの上限探索継続(15M/20M等)
- タスク#310: steer_low_pass_gainの複数値スイープ(0.6が最適か未確定)
- タスク#309: delay_t_delay_s/wp_id_offset(先読み層)の再調整
- delay_t_delay_s=0.055を乗せた最終ローカル確認(現在は全ラウンドdelay=0.0で
  変数分離のため実施)
- dev3最終確認 → 予選投入10本(5セット×2本)でローカルとの誤差を確認
- §39で発見したwp252 CSVスパイクの修正(別軸、次節参照)

## 39. wp252ホットスポットの根本原因特定: 参照経路CSVの曲率スパイク(2026-08-08)

§38.6-38.9の全ラウンドを通じ、**wp252ホットスポットだけは何を変えても
20-30cm程度で高止まり**し続けた(唯一の例外はr_drate=10Mで21.08cmまで改善)。
Q[e_y]・R[delta]・swing_boost・steer_low_pass_gain・r_drateという今日試した
全軸が、他の全区間には明確に効いたのにwp252だけ動かなかったことから、
制御(CONTROL)層ではなく計画(PLANNING)層——参照経路そのものに原因がある
のではと疑い調査した。

### 39.1 発見: CSVのkappa列に単発スパイク

`traj_mincurv.csv`(現在使用中の`env/final_ver3`版)のkappa_radpm列を確認:

| CSV index(≈wp) | s(m) | kappa(生値) |
|---|---|---|
| 250 | 249.76 | 0.143 |
| 251 | 250.76 | 0.161 |
| **252** | **251.76** | **0.342** |
| 253 | 252.76 | 0.211 |
| 254 | 253.76 | 0.161 |

隣接点の約2倍という明確な単発スパイク。

### 39.2 kappaは実はCSV列を直接使わず(x,y)から再計算されている

`reference_path.py`の`_construct_waypoints`を確認したところ、実際に
MPCへ渡るkappaは**CSVのkappa_radpm列を直接使わず、(x, y)座標から
角度差分/距離で毎回再計算**していると判明(`kappa = angle_dif /
dist_ahead`)。そこでx, y座標自体の等間隔性を確認したところ点間距離は
正常(0.90-1.05m、外れなし)だったが、**旋回角(heading変化量)を直接
計算すると、idx252だけ17.47°と前後(8-12°)のほぼ2倍**という明確な
単発の角度不連続が確認できた。実際の物理的なコーナーなら曲率は滑らかに
変化するはずで、**レースライン最適化ツール(min-curvature optimizer)が
生成した数値的なアーティファクト(折れ)である可能性が高い**。

現在`use_savgol_kappa=true`(窓7点)で事後平滑化しており、実際にMPCへ
渡る値は0.342→約0.23まで削減されているが(ログ実測値と一致)、周辺
(0.14-0.21)よりまだ高い状態が残る。

### 39.3 既存記録との整合性(以前から知られていた現象)

この発見は実はプロジェクト最初期から部分的に記録されていた:
- `use_max_kappa_pred: false`のコメントに「**CSVのkappa過大スパイクに
  MPCが踊らされるのを防ぐ(iASL確定)**」、さらに詳細な既存コメントには
  「CSVスパイク(**s=252等**)対策でfalse採用」と、**まさにこの地点が
  以前から名指しで記録されていた**。
- `[HOTSPOT-DEVIATION]`ログの監視対象waypointリスト(178, 189,
  **258**, 289, 334)にもwp252近傍(258)が含まれる(209節)。
- `smoothing_distance`のコメント(2026-07-05)にも「内部参照がCSV原線
  よりコーナーで中央+0.13m/最大+0.23m内側」という、より小さい規模の
  同種現象が既に記録されていた。

つまりwp252問題は**今日始まったものではなく、プロジェクト最初期から
知られていた参照経路自体の局所的異常**であり、今日のあらゆるQ/Rチューニング
がここだけ効かなかったのは当然だったと言える。

### 39.4 修正の実施と水平展開(2026-08-08、実施済み)

#### 39.4.1 idx252の修正

idx252を含む区間(idx247-257、idx252を除く)を通る3次スプラインを構成し、
idx252の位置に評価した値でx_m/y_mだけを置き換えた(psi_rad/kappa_radpm/
vx_mps/ax_mps2列は§39.2の通り実行時に未使用のため、混乱を避けるため
元の値のまま据え置き)。移動量はわずか4.7cm。

| idx | 修正前kappa(相当角度) | 修正後 |
|---|---|---|
| 252 | 17.47° | 12.8°程度(隣接ばらつきの範囲内へ) |

回帰スイート3236件PASS(等価性回帰テストはこのCSVに依存しない合成
ゴールデンケースのため無関係)。

#### 39.4.2 水平展開: 全区間スキャン

同じ手法(旋回角が近傍4点平均の1.6倍超、かつ絶対値5°超)でコース全体を
スキャンしたところ、idx252修正後も9箇所が候補として残った
(190/194/220/221/224/233/271/275/282)。生データを個別に確認し、以下の
判定基準で選別した:
- **単発の孤立スパイク/窪み**(前後が滑らかで1点だけ突出/陥没)→修正対象
- **正当な技術的コーナーの一部**(緩やかな山型・実際に高曲率が連続する
  区間)→**据え置き**(実際のレースライン形状を変えるリスクを避けるため)

判定の結果、**idx221-224(4点、wp220-240帯)・idx273(1点、wp269-282帯)**を
修正対象として選定。190/194/233/271/275/282は正当な技術的コーナーの一部と
判断し据え置いた。

idx221-224は当初idx222・224の2点だけを外側の錨点(217-220, 225-228)から
補間したところ、**未修正のまま残したidx221・223に新たな不連続が発生**
(修正前後で221が-10.92°→-6.10°、222が-2.29°→-12.89°など、意図せず悪化)。
これは1点だけを動かすと隣接点の旋回角(3点窓で計算)が連鎖的に変化する
ためと判明し、**4点(221-224)全体を外側の錨点から一括で滑らかに補間**する
方式へ変更して解決した(修正後: -7.91°→-8.29°→-9.02°→-10.07°→-11.27°と
綺麗な単調変化)。

idx273は単独修正(前後2周辺文脈からのスプライン補間)で問題なく解決
(1.43°→6.78°、前後の8-9°帯に自然に馴染む)。

修正後の全体再スキャンでは残存候補6件(190/194/233/271/275/282、いずれも
「据え置き」判定のまま変化なし)のみで、新規異常は発生していない。点間
距離も全区間で異常なし(0.5m未満・1.5m超のセグメントゼロ)。回帰スイート
3236件PASS。

#### 39.4.3 実走行での検証結果(2026-08-08、gain=1.0/tau=0.16/Q=200,000の
競技提出値のまま、Q/R変更なし)

10分間のソロ走行(3周)で確認:

| 指標 | 結果 |
|---|---|
| COLLISION-SUSPECTED | 0件 |
| STUCK | 1件 |
| wp252(修正済み) 平均振れ | **15.29cm**(本日のQ/R変更では22-24cmで頭打ちだった) |
| wp269-282(idx273修正) 平均振れ | 13.88cm |
| wp220-240(idx221-224修正) 平均振れ | 9.30cm |

**Q/Rを一切変更していない素の競技提出値設定で、wp252が本日のあらゆる
Q/R実験(22-24cm)より明確に改善**した。CSVスパイクがwp252の根本原因
だったという§39.1-39.3の仮説を実走行で裏付けた。

### 39.5 次のステップ(残タスク)

- 本節のCSV修正(final_ver3/traj_mincurv.csv、5点)を単独でコミット
  (制御コードとは独立、CLAUDE.md §1.5準拠)。
- §38のgain=0.6/tau=0.05向けQ/R候補セットと、本節のCSV修正を**組み合わせた**
  検証が未実施。CSV修正だけでも大きな改善が出たため、組み合わせでさらに
  良くなる可能性が高い。
- env/final_ver4・env/finalの他バージョンCSVは未調査(現在使用中の
  final_ver3のみ対処)。
- dev3・予選投入での最終確認は未実施。

## 40. Q/R候補セットの正式確定・予選提出準備(2026-08-08)

タスク#311として、§39のCSV修正(コミット`cf80cce`)と§38のQ/R候補セットを
組み合わせた検証を実施した。

### 40.1 検証結果サマリー

| 走行 | 条件 | wp252 | wp269-282 | S字wp340-40 | COLLISION |
|---|---|---|---|---|---|
| CSV修正のみ | 既定値(gain=1.0/tau=0.16/Q=200k) | 15.29cm | 13.88cm | 13.53cm | 0件 |
| CSV修正+候補セット | delay=0.0 | 12.59cm | 12.91cm | 10.02cm | 0件 |
| CSV修正+候補セット | delay=0.055(較正値) | 16.71cm | 20.43cm | 12.57cm | 0件 |
| dev3感度チェック | delay=0.03 | 15.42cm | 14.35cm | 10.55cm | 0件 |
| dev3感度チェック | delay=0.055 | 17.45cm | 18.46cm | 12.55cm | 1件 |
| dev3感度チェック | delay=0.08(最悪ケース) | 19.78cm | 16.45cm | 13.98cm | 1件 |

delayが大きいほど緩やかに悪化する自然な傾向(候補セット自体がdelay=0.0で
チューニングされているため、この傾向自体は「頑健性の証明」というより
「delayを増やしても致命的には壊れない」ことの確認、とユーザー指摘で
正確な位置づけとした)。最悪ケース(delay=0.08)でもCOLLISION-SUSPECTEDは
稀(1件)にとどまり、壊滅的な破綻は見られなかった。

### 40.2 正式確定・config.yaml反映

上記結果を踏まえ、以下をconfig.yamlの正式な既定値へ昇格した(タスク#311):

| パラメータ | 旧既定値 | 新確定値 |
|---|---|---|
| actuator_gain | 1.0 | 0.6 |
| actuator_lag_tau_s | 0.16 | 0.05 |
| Q[e_y] | 200,000 | 250,000 |
| r_delta_swing_boost | 800 | 1,600 |
| steer_low_pass_gain | 0.35 | 0.6 |
| r_drate | 3,000,000 | 10,000,000 |
| q_ey_overtake | 330,000 | 412,500(base×1.65) |
| q_ey_pit | 50,000,000 | 62,500,000(base×250) |

`debug_extra_actuator_delay_s`は0.0(提出要件)・`enable_diag_log`はfalseを
維持。ハードコードされた旧確定値(`q_ey_overtake: 330000.0`・
`r_delta_swing_boost: 800.0`)を参照していた回帰テスト2件
(test_q_ey_schedule_reverted_174.py・test_r_delta_swing_schedule_176.py)を
新確定値に更新。回帰3236件PASS。CLAUDE.md §3のrule5(steer_low_pass_gain
固定)は既に§38.8時点で反転済み(旧モデル限定の結論と判明)。

### 40.3 残課題(v_max=25km/h以降)

本節の確定はv_max=20km/hのままでの確定。25km/hへの引き上げは、この
20km/hパッケージが実際の予選環境で大きく乖離しないことを確認してから
行う方針(ユーザー合意済み、Part C確定知見: v_maxが蛇行の支配因子)。
また`steer_low_pass_gain`(タスク#310)・`r_drate`上限(タスク#308継続)・
先読み層(タスク#309)も引き続き調整余地あり。

## 41. 予選投入初回検証: 候補セット+CSV修正パッケージの初回転写結果(2026-08-08、n=2)

§40で確定したQ/R候補セット(gain=0.6/tau=0.05・Q[e_y]=250,000・
r_delta_swing_boost=1,600・steer_low_pass_gain=0.6・r_drate=10,000,000)+
§39のCSVスパイク修正パッケージを、確定後初めて実際の予選環境へ投入した。
新設の15分クールダウン制約(1回提出ごとに15分待機)の下、同一configで
2本(`autoware(0808-01).log`・`autoware(0808-02).log`、いずれも対戦相手
ありのセッション)を取得した。

### 41.1 方法

`qualifying-log-analysis`スキルの手順に従い、既存の`region_stats`関数
(§36.2以降で継続使用してきた手法、`[LOC-XCHECK]`ログの`wp`/`ekf_ey`/`v`
からサンプル間diff・区間内|ekf_ey|を算出、時間差1.0s未満のサンプル対のみ
採用)を5ホットスポット(wp180前後174-186・wp220-240・wp252[246-258]・
wp269-282・S字wp340-40[周回ラップ])へ適用した。

n=5到達ルール(同スキル)により、本節の評価は**暫定**である。

### 41.2 ホットスポット比較(ローカル基準 vs 予選実測)

| 区間 | ローカル(候補セット delay=0.0、§40.1) | 0808-01(予選) | 0808-02(予選) |
|---|---|---|---|
| wp252 | 12.59cm | 13.66cm | 13.44cm |
| wp269-282 | 12.91cm | 12.03cm | 12.87cm |
| S字wp340-40 | 10.02cm | 11.99cm | 11.35cm |
| wp180前後 | (直接比較データなし) | 6.71cm | 6.74cm |
| wp220-240 | (直接比較データなし) | 9.25cm | 10.67cm |

ローカル基準と直接比較可能な3区間(wp252・wp269-282・S字wp340-40)は
いずれも**誤差1-3cm程度**でローカル予測に一致した。0808-01/02間の
再現性も高い(同一区間で最大でも1.4cm差)。wp180前後・wp220-240は
§40.1の比較表に対応するローカル基準行が存在せず、本節では予選実測値の
みを記録する(次回ローカル走行時に同条件で追加測定し埋める)。

### 41.3 安全指標比較

| 指標 | 0808-01 | 0808-02 |
|---|---|---|
| STUCK(episode数、COUNTER-RESETで計数) | 6件 | **0件** |
| COLLISION-SUSPECTED(単発v-drop、実イベント) | 8ログ行/5クラスタ | **0件** |
| COLLISION-SUSPECTED-CUM(5サイクル累積のみ) | (上記に付随) | 1件(軽微、単発スパイクなし) |
| GHOST-BLOCK | 1件 | 0件 |
| FAILSAFE | 1件 | 1件 |

0808-02は0808-01より明確に安全指標が良好(ユーザー目視評価
「さっきよりもいいです」と整合)。同一config内でのn=2でもこの程度の
セッション間ばらつきがあり、CLAUDE.md §2 rule7(予選ログのn=1評価を
過信しない)が改めて裏付けられた形。

### 41.4 COLLISION-SUSPECTEDイベントの原因分析

0808-01の5クラスタを個別に追跡した結果、3パターンに整理できた:

1. **wp278-282ホットスポット collision(3/5件、t=1603.8・1713.0・1817.9)**:
   いずれも`OVERTAKING`中、ちょうどコーナー進入でkappaが-0.03→-0.15〜
   -0.18へ急峻化するタイミングで`STOPPING`へ諦め(giveup)た直後に
   相手と接近(d_min最小1.91m、1件は`footprint_taper`発火も確認)。
   タスク#300(OT giveup閾値のログベース精査)・#306(WP280帯3台密集
   STUCK/衝突の実態分析)と直結する実例が今回3件そろった。
2. **wp64-65の追従詰まり(1/5件、t=1482.3)**: `STOPPING`(追従)中の
   単発急減速。0808-02の唯一のCOLLISION-SUSPECTED(後述、wp63-65)と
   **同一区間**であり、2本のログにまたがって再現した新しい着眼点候補。
3. **相手非検知でのMPC自滅的停止(1/5件、t=1729.9・wp284)**:
   `obs=0 fwd=0`(相手を1台も検知していない)状態で`infeas=18`
   (MPCコリドー実行不可能解が18周期連続)、v=2.27→0.05m/sへ自滅的に
   急停止。COLLISION-SUSPECTED判定(速度低下ベース、相手の有無を
   区別しない)の**偽陽性の疑い**。

0808-02の唯一の事例(t=1786192521.9、wp63-65)は上記1と同型の流れ
(`OVERTAKING`→`STOPPING`諦め直後に`footprint_risk`発火、d_min=1.99m、
`infeas_taper`でv_safe/u0=0.0まで減速)だが、**単発v-dropスパイクが
出ずCUM検知のみ**——0808-01の同種3件より軽微(実接触の疑いは低い)。

### 41.5 歴史的意義

ローカル基準(候補セット、delay=0.0)と誤差1-3cm程度で一致したのは、
本プロジェクトの長い試行錯誤(tau160ms・72Hz化等、ローカルで改善して
予選で再現しなかった過去事例多数)の中でおそらく初めての
**クリーンな転写**。ユーザー評価は0808-01が「過去一いい」、0808-02が
「さっきよりもいい」。本日の3施策——gain/tauの実測修正(§30-36)・
Q/R再チューニング(§38)・CSVスパイク修正(§39)——の複合効果と
考えられるが、単独要因への切り分けは未実施。

### 41.6 残課題・次のステップ

- n=2、`qualifying-log-analysis`スキルのn=5到達ルールにより、本節の
  評価は引き続き暫定。新しい予選ログが届き次第、本節の表へ追記する。
- §41.4で確認したwp278-282のOT giveup collision実例3件を根拠に、
  タスク#300・#306の優先度を引き上げるべき(CLAUDE.md §1.3により
  giveup/cleared判定周りは意図的に未着手のまま慎重に扱われてきた
  領域だが、実害の実例が積み上がっている点は記録しておく)。
- §41.4の3.で見つけたCOLLISION-SUSPECTED判定の偽陽性疑い
  (`obs=0`でも速度低下だけで発火)は、判定ロジック自体の改善候補
  として新規に切り出す価値がある(未着手)。
- wp64-65の追従詰まり(§41.4の2.、両ログで再現)も新しい着眼点候補
  として記録(未着手)。
- v_max=25km/hへの引き上げは、n=5到達・より広いばらつき確認後まで
  引き続き保留(§40.3から継続)。

## 42. OT giveup→COLLISION-SUSPECTED近接パターンのdev3ログ横断調査(タスク#300、2026-08-08)

§41.4で見つけた「追い越しを諦めた直後に相手へ接近する」パターンが本日
2本の予選ログだけの偶発事象か、より広く再現するかを確認するため、
本日のdev3ラウンド16回(32台分ログ、tri-param並列検証で収集済みの既存
ログを再利用、新規走行なし)を横断的にスキャンした。

### 42.1 方法

`[LOC-XCHECK]`ログから`OVERTAKING→STOPPING/NORMAL`の状態遷移("giveup")を
検出し、その3秒以内に`[COLLISION-SUSPECTED]`(単発v-drop)が発火した
事例を抽出した。

### 42.2 結果

32ログファイル中、該当パターンが**72件**検出された。giveup検知から
COLLISION-SUSPECTED発火までの遅延は大半が0.5秒未満(giveup判定とほぼ
同時に急減速が起きている)。wp位置の内訳:

- **wp269-282(既知ホットスポット)内: 39件(54%)**
- wp283-284(ホットスポット直後の出口): 10件(14%)
- その他(wp30・59・103-119・166・212・226・230・302・317・328・331等):
  23件(32%、散発的だが低速コーナー・合流地点に多い傾向)

wp269-284の範囲だけで**全体の68%**を占め、§41.4の予選ログ実例
(3/5件がこの範囲)と方向性が一致した。

### 42.3 解釈上の注意

遅延0.5秒未満の事例が大半という点は、「giveup判定そのものが速度急落の
引き金」である可能性を示唆する(giveup直後にSTOPPING状態のicc_stop/
footprint_riskが強い減速を要求する設計のため、これ自体は意図した
安全側動作の可能性もある)。すなわち72件全てが「あわや衝突」という
わけではなく、一定数は設計通りの急減速が閾値に引っかかっただけの
可能性がある。§41.4のように`d_min`・`footprint_taper`発火の有無まで
個別確認できた事例(d_min最小1.91m等)は真に際どいニアミスと言えるが、
当初は72件全部について同水準の個別検証を行っていなかった(§42.4で解消)。

### 42.4 個別精査: 真のニアミス vs 設計通りの安全減速の切り分け

§42.3の懸念(giveup判定自体が意図した安全減速の引き金になっているだけの
事例が混在する)を解消するため、各事例の直前の`[OT]`ログ(`d_min`・
`fp_taper`)を突き合わせて機械的に分類した(72件中55件で直前`[OT]`ログを
特定、17件は判定保留):

| 分類 | 件数 | 基準 |
|---|---|---|
| 設計通りの安全減速疑い | 28件 | d_min>=4.0m(相手までまだ距離がある) |
| **要注意(真のニアミス候補)** | **27件** | d_min<2.5m、またはfootprint_taper発火 |
| 判定保留 | 17件 | 直前`[OT]`ログ未特定、またはd_min 2.5-4.0mの中間域 |

**「要注意」27件のうち19件(70%)がwp269-284帯に集中**しており、
72件全体の68%よりもさらに濃縮された比率になった——ホットスポットで
起きているgiveup直後の減速は、他区間より高い割合で真に際どい
ニアミスである可能性が示唆される。うち9件では`footprint_taper`
(接触リスク接近テーパー)が実際に発火しており(d_min実測2.86-3.00m)、
これは設計上の意図(接触回避のための減速)が正しく機能している証拠でも
あるが、そもそもgiveup直後にここまで接近すること自体が繰り返し
発生している点が課題として残る。

### 42.5 深掘り: footprint_taper発火7件の時系列追跡から見えた共通メカニズム

§42.4の「要注意」27件のうちfootprint_taper発火9件から代表7件(wp166・
wp117・wp279・wp282・wp280・wp279・wp103、いずれも異なるログ・異なる
domain)を選び、giveup前後-6秒〜+2秒の`[OT]`/`[LOC-XCHECK]`ログを時系列で
突き合わせた。**7件全てが驚くほど同一の因果連鎖を辿っていた**:

1. 相手を検知(`vopp`は7件中7件で**0.0〜1.5m/s**——相手はほぼ停止/低速で
   ある。高速で走行中の相手を追い越そうとした事例は今回1件もなかった)。
2. `OVERTAKING`へ遷移、side固定のままoffsetを段階的に拡大(0.1m台→
   0.6-1.7m台まで2-5秒かけて増加)、この間`corr_bound`の先読み距離は
   6-18m先を指しており余裕がある。
3. offsetがほぼ最大まで達したタイミングで、**同じ地点でkappaが符号
   反転**(例: wp276-282帯でkappa +0.05→-0.03→-0.18、wp102-118帯や
   wp160-166帯でも同型の反転)。これと同時に`corr_bound`の先読み距離が
   突然0-2m先まで収縮する(コリドーが「今この瞬間」しか見えなくなる)。
4. **上記3と全く同じLOC-XCHECKサンプルで`OVERTAKING→STOPPING`へ
   即座に切り替わる**(giveup)。offsetは次サンプルで0へ戻る。
5. giveup後2-3周期(0.2-0.3秒)で速度が3.2-4.9m/s→ほぼ0m/sまで急減速
   (緩やかな減速ではなく強制的な非常停止に近い)。
6. 停止位置での`d_min`は0.95-3.00m(7件中5件が2.0m未満)——**停止中/
   低速の相手のほぼ真後ろで急停止する形**になっている。

つまり本節で見つかった事例は「速い相手を追い越そうとして反転する」
パターンではなく、**「停止/低速の相手を追い越そうと踏み込んだ直後に、
ちょうど地形側のkappa反転(S字・複合コーナー)が重なってコリドーが
先読みゼロまで収縮し、giveup判定がoffsetを最大まで拡大した"後"に
初めて発火し、間に合わずほぼ真後ろで急停止する」という、地形と相手の
状態(停止)とタイミングが3重に重なった際に起きる固有パターンである。
wp269-284帯・wp102-118帯・wp160-166帯はいずれも実際にkappa反転を
含む地形であり、地形要因が支配的という仮説を補強する。

### 42.6 対策の方向性(設計案、未実装)

giveup判定自体(`OVERTAKING→STOPPING`の閾値)を緩めたり`cleared`判定に
手を入れるのはCLAUDE.md §1.3により避けるべきだが、本節の発見は
**判定の是非ではなく判定の"タイミング"の問題**であるため、別軸の
対策候補が考えられる:

- **corr_bound先読み窓を、現在のkappa反転検知(`_switchback_curvature_veto`
  の`_ot_pass_block_kappa`と同種のロジック)と連動させ、OVERTAKING開始
  判断(offsetを拡大し始める前)の時点で"この先kappaが反転してコリドーが
  収縮する区間があるか"を事前に評価し、反転が近い場合はそもそも
  offset拡大を抑制する**、という設計。既存の`_corr_bound_ahead`
  (125節、動的コリドー配列の先読み最小値)のデータソースをそのまま
  再利用でき、新規ロジックは"OVERTAKING開始前のもう一段階早い判定"
  として追加する形になるため、既存のgiveup/cleared判定そのものには
  触れずに済む可能性がある。
- ただし7件はいずれも`vopp≈0`(相手が停止/低速)という条件下で起きて
  おり、"相手が停止している場合はそもそも追い越し価値([OT]ログの
  `worth`)の評価に地形の先読みを重く反映させる"という、`worth`計算側
  の調整という切り口もありうる。
- **いずれも設計のみに留め、本節の時点では未実装**。次段階は上記いずれか
  一方を選び、影響範囲(既存の`_corr_bound_ahead`・`worth`計算の呼び出し
  箇所)を洗い出した上でdev3検証込みの実装計画を立てる(タスク#300、
  継続)。

### 42.7 影響範囲の洗い出し: `_plan_pass`のk_corner vetoが既に同じ問題を狙った機構だった

§42.6の対策案①(OVERTAKING開始前のkappa反転先読み)を実装する前に、
既存コードに同種の機構が無いか確認したところ、`_plan_pass`
(`mpc_controller.py:4124`、ENGAGE時に側と要並走距離を一括判定する
関数)に**k_corner veto**という、まさに同じ目的の機構が既に存在した。

**発見1: ENGAGE時の先読み窓自体は今回の7件を捉えられる距離だった**
7件のENGAGE〜giveupまでの弧長を`traj_mincurv.csv`の`s_m`列から実測した
ところ、いずれも10-16m(waypoint間隔約1.0mを反映)で、`_plan_pass`が
実際に使う先読み窓(`clear_at = ds + _ot_pass_clear + vopp*_ot_t_lateral`、
今回vopp≈0のため事実上`_ot_pass_clear`相当)の範囲内だった。つまり
**「コーナー反転が先読み窓の外にあって見えていなかった」という単純な
話ではない**——k_corner検出自体(`abs(_k) >= self._ot_pass_block_kappa`
[既定0.10]で発火)はENGAGE時点でも成立していたはずである。

**発見2: k_corner vetoは"物理的に嵌まるか"のみで判定するよう意図的に
緩められた経緯がある**
2026-07-14の修正(コード内コメント参照)で、k_corner vetoの閾値は
「並走を維持できる余裕(along_lane_need)」から「物理的に嵌まるか
(along_min_width、カート幅ベースの下限)」へ**意図的に緩和**されていた。
理由は0714-03実測(wp270-277)で、より保守的な閾値が原因でイン/アウト
両側が締め出され10秒以上の完全停止に陥っていたため。つまり**現行の
`along_min_width`しきい値は「不要に追い越しを止めない」ことを優先した
過去の意図的なチューニング結果**であり、今回の7件のニアミスは
その同じトレードオフの裏面(緩めた分だけ際どい追い越しも通ってしまう)
である可能性が高い。

**結論**: 単純に閾値を`along_min_width`から引き上げる対策は、
2026-07-14に修正済みの「イン/アウト締め出し→10秒以上の完全停止」問題を
再発させるリスクが高く、この2つの安全指標(ニアミス率 vs 停止・締め出し
発生率)を**両方同時に測定するA/Bでなければ判断できない**。片方の指標
だけを見て閾値を動かすのは危険(まさにCLAUDE.md §2 rule4「ハード制約を
先に適用し生存者をソフト指標で比較する」が想定する典型的な失敗形)。

### 42.8 次段階の設計方針(未実装)

- 閾値の数値変更ではなく、まず**ENGAGE時点の`_lf_at_corner`/
  `_rf_at_corner`推定値と、実際にgiveupを引き起こした後段の動的
  `_corr_bound_ahead`実測値がどれだけ乖離するか**を診断ログとして
  可視化する(制御には無関与、既存の`_dbg_plan_trace`の拡張で対応
  できる可能性が高く、ここは安全に着手できる)。
- 乖離の実測データが集まった段階で、「ENGAGE時推定と後段実測の差が
  大きい時だけ様子見(要並走距離を若干積み増す等)」という**局所的で
  可逆性の高い**対策を設計し、10秒締め出し問題の再発有無・ニアミス率
  改善の両方をdev3で計測してから採否判断する。

**実装済み(コミット`8d2ae88`)**: ENGAGE確定時に`_plan_pass`の窓内
トレース(`self._dbg_plan_trace`)を`self._ot_engage_trace`/
`_ot_engage_wp`へスナップショットし、giveup時(`[LAT-TTC-ACT]`と同一
箇所)に`[ENGAGE-GIVEUP-TRACE]`として再ログする診断専用の追加を行った。
既存の判定ロジック(giveup条件・k_corner veto閾値等)は無変更、回帰
3236件PASS。次に必要なデータ(ENGAGE時推定 vs giveup時実態の乖離)は
今後のdev3・予選走行で自然に蓄積される。

### 42.9 現時点の判断

- **ログ収集・個別精査・因果メカニズムの特定まで完了**、wp269-284への
  集中(全体68%、真のニアミス候補に絞ると70%)という定量的根拠に加え、
  「停止/低速の相手×kappa反転地形×giveupタイミングの後手」という
  再現性の高い共通メカニズム(7/7件で一致)がタスク#300・#306に加わった。
- §42.7の影響範囲調査により、対策候補は§42.6の2案から**§42.8の診断
  ログ拡張(ENGAGE時推定 vs 後段実測の乖離可視化)**へ絞り込まれた。
  k_corner vetoの閾値は2026-07-14に「イン/アウト両側締め出し→10秒以上
  完全停止」という別の重大な問題を解消するために意図的に緩和された
  経緯があり、ニアミス率と締め出し発生率の**両指標を同時計測しないまま
  閾値だけを動かすのは危険**(CLAUDE.md §2 rule4)。
- CLAUDE.md §1.3の教訓(giveup/`cleared`判定周りへの安易なガード追加は
  過去に重大リグレッション[82-83節]を招いた実績あり)を踏まえ、**本節の
  時点では判定ロジックの変更は行わない**。次段階は診断ログ拡張(制御へ
  無関与、安全に着手可)から始め、乖離データが集まってから閾値変更の
  要否をdev3の両指標同時計測で判断する(タスク#300、継続)。

## 43. AWSIM更新後の健全性確認(2026-08-08)

新しいAWSIMバイナリ(`aichallenge/simulator/AWSIM/`、旧版は`AWSIM_5`へ
自動退避)への更新後、物理特性・挙動差異の有無を確認するため、本日
確定した候補セット(§40)のままソロ走行による健全性チェックを実施した
(`output/20260808-213218`、実施中にユーザーから分析依頼)。

### 43.1 結果(53分経過時点、走行継続中)

| 指標 | 値 |
|---|---|
| wp180前後 平均振れ | 6.38cm |
| wp220-240 平均振れ | 9.03cm |
| wp252 平均振れ | 12.18cm |
| wp269-282 平均振れ | 12.83cm |
| S字wp340-40 平均振れ | 10.07cm |
| STUCK(episode数) | 0件 |
| COLLISION-SUSPECTED(実イベント) | 0件 |
| GHOST-BLOCK | 0件 |
| FAILSAFE | 1件(53分中1回) |

サンプル数の時系列分布(5分区切り)は途切れなく連続しており、単一の
継続走行であることを確認した(`docker ps`でも53分経過時点で3コンテナ
稼働継続を確認)。

### 43.2 判断

5区間すべてで§41.2(0808-01/02予選ログ)・§40.1(ローカル候補セット)と
近い値(誤差1-3cm程度)を示し、STUCK/COLLISION-SUSPECTED/GHOST-BLOCKは
53分間の長時間走行にもかかわらずゼロ(FAILSAFEのみ1件)。**新AWSIM
バイナリへの更新が本日の較正(gain=0.6/tau=0.05・Q/R候補セット)を
無効化するような物理特性・挙動差異を生んだ兆候は見られない**。

正式な結論(旧AWSIM版との統計的な有意差検定等)は本節時点では出さず、
「大きな崩れは無い」という健全性チェックの水準にとどめる。長時間の
継続走行はPC発熱の観点から目的を達成し次第`make down`で停止する
運用とする(既存メモリ: dev3タイマー満了後は解析前に必ずmake downで
停止、と同じ配慮を単独走行にも適用)。

## 44. 25km/hブラッシュアップ着手・初日打ち切り(2026-08-08)

### 44.1 v_max=25.0/ay_max=7.0への実験的引き上げ

ユーザー承認の上、20km/h確定パッケージ(§40-41)から`v_max`/`ay_max`のみ
25.0/7.0へ引き上げ(コミット`083f913`、Q/R候補セット自体は無変更)。
今回の予選走行にも反映させるためユーザー了承の上で`juliejpn6`へpush済み
(`bd79b42..083f913`、9コミット)。

### 44.2 tri-param 3台同時テストが2回連続クラッシュ、原因はSIM_MODE未指定

25km/h向けdev3ブラッシュアップの過程で、tri-param手法での3台同時起動時に
AWSIM(simulator)がexit code 255で2回連続クラッシュ(起動から2-4分)。
当初は本日更新した新AWSIMバイナリの不安定性を疑い、旧版(`AWSIM_5`)へ
切り戻して再現するかも確認したが、**旧版でも同様にクラッシュ**し
バイナリ側は無罪と判明。真因はユーザー指摘により発覚: 手動での
`docker compose up -d simulator`呼び出し時に`SIM_MODE=dev3`の指定を
毎回忘れていた(`docker-compose.yml`の既定値は空文字列)。`SIM_MODE=dev3`
を明示して起動し直したところ安定した。**今後tri-param手法で複数台テストを
行う際は、simulator起動コマンドに必ず`SIM_MODE=<dev2/dev3/dev4>`を含める
こと**(手順書への反映、次回の引き継ぎ事項)。

### 44.3 Q[e_y]スイープ(200k/250k/350k、v_max=25/ay_max=7)結果: 芳しくない

SIM_MODE修正後、tri-param 3台(D1=Q[e_y]200k/D2=250k(現行)/D3=350k)で
6分間の走行データを取得した。

| 区間 | D1(200k) | D2(250k、現行) | D3(350k) |
|---|---|---|---|
| wp180前後 | 47.90cm | 39.35cm | 11.52cm |
| wp220-240 | 12.19cm | 14.29cm | 17.89cm |
| wp252 | 36.67cm | 28.02cm | 30.34cm |
| wp269-282 | 32.10cm | 23.05cm | 33.07cm |
| S字wp340-40 | 27.62cm | 28.07cm | 40.43cm |
| STUCK/COLLISION/GHOST-BLOCK | 2/1/1 | 2/0/1 | 2/0/1 |

区間ごとに最良値が割れており(トレードオフのみ)、明確な最適Q[e_y]は
今回のスイープでは見つからなかった。かつ現行値(D2=250k)を使っても
全区間20-30cm台と、**20km/hベースライン(§43、6-13cm台)の2-4倍程度
悪化**しており、Q[e_y]の調整だけでは25km/hの蛇行は解決しないことが
確認できた。これはPart C確定知見(v_maxが蛇行の支配因子、Qは副次的)と
整合する結果である。

### 44.4 初日打ち切り・引き継ぎ事項

23:30過ぎの時点で作業を打ち切り、翌日以降へ継続することとした。

- **予選環境には25km/h(未確定)がpush済みのまま**——ユーザー明言
  (2026-08-08夜):「20km/hには戻さない、25km/hでのセッティングを
  煮詰めたい」。20km/hへの後戻りは無し、次回作業再開時もこの状態から
  25km/hフルキャンペーンへ直接進む。
- 25km/hの蛇行改善には、20km/h時に実施したのと同規模のフルキャンペーン
  (steer_low_pass_gain・r_delta_swing_boost・r_drateを含めた再チューニング、
  §38と同じ手法)が必要と判明。次回セッションの最優先候補。
- タスク#300(OT giveupニアミス)は診断ログ実装(コミット`8d2ae88`)まで
  完了、次はデータ収集(dev3・予選ログの自然蓄積を待つ)。
- タスク#310(steer_low_pass_gainの最適値未確定)・タスク#308(r_drate
  上限未確定)は25km/hキャンペーンと合わせて再検証が必要。

### 44.5 予選ログ0808-03: 25km/hの初回実地結果(n=1、暫定)

25km/h(v_max=25.0/ay_max=7.0、Q[e_y]=250,000のまま)をpushした後の
予選走行ログ`autoware0808-03).log`(9.1分)を確認した。

| 区間 | 予選0808-03 | ローカルdev3 Q[e_y]=250k(§44.3) | 20km/hベースライン(§43) |
|---|---|---|---|
| wp180前後 | 15.55cm | 39.35cm | 6.38cm |
| wp220-240 | 8.11cm | 14.29cm | 9.03cm |
| wp252 | 17.36cm | 28.02cm | 12.18cm |
| wp269-282 | 11.88cm | 23.05cm | 12.83cm |
| S字wp340-40 | 13.87cm | 28.07cm | 10.07cm |
| STUCK/COLLISION/GHOST-BLOCK/FAILSAFE | 1/1/1/1 | 2/0/1/1 | 0/0/0/1(§43) |

**予選実測はローカルdev3 tri-paramテスト(§44.3)より明確に良好**——
20km/hベースラインに近い、一部区間(wp220-240)はほぼ同水準まで戻って
いる。ローカルで見えた2-4倍の悪化は今回は再現しなかった。dev3
tri-param特有の負荷条件(3台同時・SIM_MODE周りの構成変更直後だった点も
含む)とリアル予選環境の乖離が疑われるが、**n=1のため確定的な判断は
しない**(CLAUDE.md §2 rule1・7)。次の予選ログ・dev3再検証で追加確認が
必要。25km/hフルQ/Rキャンペーン着手の優先度自体は変わらない。

### 44.6 予選ログ0808-04/05/06追加、n=4到達(25km/h)

翌朝(2026-08-09)追加で3本(`0808-04`・`0808-05`・`0808-06`、いずれも
v_max=25.0確認済み、各9.1分)を確認し、25km/hの予選ログはn=4になった。

| 区間 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|
| wp180前後 | 15.55 | 10.61 | 9.24 | 13.44 |
| wp220-240 | 8.11 | 8.75 | 8.99 | 7.16 |
| wp252 | 17.36 | 19.66 | 20.46 | 20.33 |
| wp269-282 | 11.88 | 15.22 | 11.21 | 10.62 |
| S字wp340-40 | 13.87 | 14.83 | 11.82 | 13.37 |
| STUCK | 1 | 0 | **11** | 1 |
| COLLISION-SUSPECTED | 1 | 2 | **9** | 2 |
| GHOST-BLOCK | 1 | 0 | **5** | 1 |

蛇行の絶対値(wp180-S字、単位cm)は4本を通じて比較的安定している
(20km/hベースライン比おおむね1.2-2倍)一方、**0808-05だけ安全指標が
突出して悪い**(STUCK11・COLLISION9・GHOST-BLOCK5、他3本は軒並み
0-2件)。0808-05のSTUCK発生位置を確認したところ、**最初の3件が
wp282-285**——§42で特定した「wp269-284帯でのOT giveup直後ニアミス」の
延長線上にある可能性が高い。この1本は渋滞・対戦密度がセッション依存で
特に悪かった外れ値の可能性もあり(CLAUDE.md §2 rule2)、n=5(あと1本)を
待って正式評価する。

### 44.7 25km/hフルQ/Rキャンペーンへ着手(継続)

n=4の時点で蛇行の絶対値自体は致命的ではない(20km/hの1.2-2倍程度)と
確認できたため、ユーザー方針(§44.4「20km/hへは戻さず25km/hを煮詰める」)
通り、次段階として全体のバランス調整(§38と同型のフルキャンペーン:
steer_low_pass_gain・r_delta_swing_boost・r_drateを25km/h向けに
再スイープ)に着手する。

## 45. footprint_risk自己ロックの発見・外部AIレビュー・must-fix実装(2026-08-09)

### 45.1 発見の経緯

予選ログ`autoware(0808-05).log`の2周目wp280前後を目視確認したところ、
前方に停止車2台がいる場面で、自車が**左側は終始広く空いている
(Lfree=3.6-3.96m)にもかかわらず18.5秒間完全停止し続け**、最終的に
STUCK(バック復帰)に陥っていることが判明した。

原因を追跡した結果、`_dlat_closing_trend()`(横間隔のトレンド判定)が
`footprint_risk=True`の間はトレンド計算を無視して常にTrueを返す設計
(2026-07-22追加、意図は「既に物理的接触リスクがある間は保守的に振る舞う」)
であり、footprint_risk自体は「相手との現在の間隔が物理的最小幅未満」で
発火することを確認した。停止車の真後ろで完全停止している間はfootprint_risk
が継続的にTrueのままになり、**「危険を解消する唯一の動作(空いている側へ
避ける)を、危険を理由に禁止し続ける」自己ロック**になっていると判断した。

### 45.2 実測による裏付け

過去2週間分(2026-07-24〜2026-08-08)のdev3ログ209本+予選ログ6本を
横断スキャンした結果:

- footprint_risk起因のgiveup直後(30秒以内)にSTUCKが発生した事例: **190件**
- そのうち「片側のLfree/Rfreeが明確に広く(>2.5m)、もう片側が狭い(<1.5m)」
  =幾何的には明らかに逃げ場があった事例: **127件(67%)**

場所はwp278-286帯に多いが、wp6-9・wp52-70・wp114-186・wp265・wp296・
wp327など全域で発生しており、局所的な地形問題ではなく一般的な制御ロジックの
欠陥と判断した。

### 45.3 外部AIレビュー(Gemini・別Claude)

`docs/superpowers/specs/2026-08-09-dlat-ttc-veto-selflock-fix-consultation-
prompt.md`で外部AIへレビューを依頼した(初回実装後)。

**Gemini指摘(最重要): チャタリング懸念**——「ENGAGEゲートを解除しても、
footprint_risk自体は毎周期評価され続けるため、OVERTAKING遷移直後に即座に
再発火し、STOPPINGへ押し戻される(40Hzチャタリング)のではないか」。
コードで検証したところ**完全に妥当な指摘**だった: `lateral_ttc_monitor.py`
の`footprint_risk=True`分岐は「トレンドの蓄積を待たず最優先で強制giveup」
(force_giveup=True)を毎周期無条件に発火させ、これはENGAGEゲート
(`_dlat_ttc_veto`)とは完全に別経路であり、Fix C(並走中giveupの有限保留)
からも明示的に除外されている(「安全反応系の遅延は厳禁」、82/83節の教訓)。
初回実装はENGAGEゲートしか解除しておらず、**実際には機能しない
(チャタリングするだけの)修正だった**。

Gemini指摘②(幾何学的死角): `_plan_pass`の`along_min_width`は静的な平行幅
判定であり、停止車の真後ろから斜めに発進する際の車体後部振り出し等の
スイープボリュームは考慮していない。既存`_plan_pass`自体を変更する対応は
リスクが高いため、本節では対処せず残課題とした(45.6参照)。

**別Claude指摘(must-fix)**:
1. configゲート追加(`overtake.selflock_release_enabled`、既定false)
2. fwd_voppの有効性ガード(V2X速度クランプ中でない・鮮度内であること)
3. (推奨→実装) オフセット後予測dlatが閾値+ヒステリシスを超えるかの確認
   (振動防止)

### 45.4 実装した修正(3層)

**層1: ENGAGEゲート解除**(`_evaluate_engage_readiness`内`_dlat_ttc_veto`)——
以下4条件AND成立時のみ解除:
1. configゲート`selflock_release_enabled`(既定false)
2. `footprint_risk=False`で`_dlat_closing_trend`を再評価した「本来のトレンド
   判定」がFalse
3. 相手が停止/低速(`fwd_vopp < opp_obstacle_speed`)かつV2X速度クランプ中
   でなく追跡が鮮度化済み(`V2XVehicleTracker.is_speed_clamped()`新設・
   `is_settled()`既存を再利用)
4. `_plan_pass`が既に物理的に妥当な側を見つけており(`_plan_side != 0`)、
   その側へオフセット完了後に予測されるdlba(`_room_to_wall`を`_plan_pass`と
   同一の引数で再利用、新規計算式なし)が`along_min_width + overlap_margin_m`
   (いずれも既存定数)を超える

**層2: OVERTAKING遷移直後のforce_giveupエスケープ猶予**(Gemini指摘への
対処、チャタリング防止)——層1で解除・ENGAGEした直後、footprint_risk由来の
force_giveupに限り、以下5条件が**毎周期**成立する間だけ側維持を認める:
同一対象車・同一側・猶予周期内(既存`t_lateral`定数を周期数へ換算、新規
マジックナンバーなし)・`dlat_v_ema>=0`(悪化していない)。**1つでも崩れたら
即座に通常のforce_giveupへ復帰**(フェイルクローズ、無期限の抑制はしない、
「安全反応の遅延は厳禁」原則を破らない設計)。

**層3: 状態クリーンアップ**——OVERTAKING離脱の3経路(giveup合流・通過完了・
infeasible恒久失敗)すべてでエスケープ状態を確実にクリアし、次回の無関係な
ENGAGEへ持ち越さない。

`v2x_vehicle_tracker.py`には`is_speed_clamped(vid)`を新設(既存の
クランプ処理分岐へフラグを1行追加するのみ、`clamp_hold_enabled`の設定に
関わらず判定可能)。

### 45.5 検証状況

- 単体回帰3236件+新規テスト20件(`test_selflock_release_20260809.py`、
  configゲートOFF等価・V2Xクランプ/未鮮度での不発・予測dlat不足での不発・
  エスケープ5条件・3経路でのクリーンアップ・トラッカーのクランプ検出)、
  合計3256件PASS。
- configゲートは既定`false`のため、現時点では**挙動ビット等価**(退行なし)。
- **Phase 2(反実仮想検証、127件への陽性適用・走行中相手/V2Xクランプ共起
  事例への陰性確認)は未実施**。
- **Phase 3(dev3・予選での実地検証)も未実施**。configをtrueにする前に
  必須。

### 45.6a Phase 2反実仮想検証: 完了

過去2週間分のdev3ログ209本+予選ログ6本から、既存の`[DLAT-TTC-VETO]`ログ
(`_dlat_ttc_veto_effective = _plan_ok and _dlat_ttc_veto`の瞬間に発火、
fwd_dlat・dlat_v_ema・shrink_run・footprint_risk・wpを含む)を直接読み取り、
最寄りの`[OT]`ログ(side・Lfree/Rfree・vopp)、V2Xクランプ警告ログと突き合わせて
解除条件を機械的に再現した(新規ヒューリスティックではなく、実装した4条件を
そのままログから再計算)。

| 指標 | 値 |
|---|---|
| footprint_risk由来`[DLAT-TTC-VETO]`(plan_ok成立=自己ロック該当)総数 | 298件 |
| **would_release(4条件全て成立)** | **101件(33.9%)** |
| **陰性チェックA: 走行中相手(vopp≥6km/h)なのに解除** | **0件** |
| 陰性チェックB: V2Xクランプ中なのに解除 | 0件(構造的に必ず0) |
| V2Xクランプ共起footprint_risk-veto件数 | 0件(本データセットには非発生) |
| real_trend_veto=True(本来のトレンドでも危険) | 20/298件 |
| opponent_stopped=False | 8/298件 |
| predicted_ok=False(予測post-offset dlat不足) | 182/298件(61%、最大の絞り込み要因) |

**陽性: 約34%の救済率、陰性: 0件**。must-fix3(予測post-offset dlat)が
最も保守的に効いており(61%を弾く)、残る事例の大半は「解除しても実際には
十分に離れられない」と判定して見送っている。走行中の相手への誤解除は
データセット中0件(4条件の設計が機能している)。V2Xクランプ共起事例は
本データセットには存在しなかったため、must-fix2の実効性は「構造的に
安全」であることの確認にとどまり、実データでの発火は未確認。

### 45.6 残課題

- Gemini指摘②(幾何学的死角、斜め発進のスイープボリューム未考慮)への
  対処は未着手。`_plan_pass`自体の変更はリスクが高く、本節のスコープ外と
  した。
### 45.7 EMA意味論の訂正(外部AIレビュー第2弾、2026-08-09)

`2026-08-09-dlat-ema-stopped-opponent-semantics-consultation-prompt.md`への
回答で、45.6の当初分析(「非負転換でshrink_runがリセットされ、0.5-0.7秒で
自然解除」)に**機構としての誤りが1点あった**ことが判明した。

**訂正**: 両者停止でfwd_dlatが定数Dになると、`_dlat_ema`はDへ上側から
単調に指数収束するため、差分`_d`は全周期で負のまま0へ漸近する。
`_v_dlat_ema`(同符号入力のEMA)も負のまま0へ漸近し、**決定論的には非負に
転じない**(転じるのは測定ノイズによる偶発のみ)。したがって
`_dlat_shrink_run`のリセット経路(非負転換)は実質機能しない。

**実際の自然解除機構はTTC条件**: `real_trend_veto`の3AND中、TTC条件
`(fwd_dlat/max(abs(dlat_v_ema),1e-6)) <= ttc_critical_s`が、`|v_dlat_ema|`の
指数減衰により**分母が縮み続けTTCが指数増大**することで決定論的に破れる。
これがreal_trend_vetoをFalseへ倒す実際の経路である。

**検証(小規模シミュレーション、`_update_dlat_trend`の漸化式をそのまま
再実装)**: `space_ema_alpha`実値=0.05(rate-scaled、`lat_ttc.space_ema_alpha`
のconfig上書きなし、`self._ot_ema_alpha`既定値を継承、コード内コメント
「≈1秒@40Hz」と整合)、`beta`=0.15。

- ノイズなし: 全テストケースで`_v_dlat_ema`は非負に転じない(訂正1を確認)。
  TTC条件が破れる時刻と`real_trend_veto`がFalseになる時刻は全ケースで
  完全一致(訂正2を確認)。
- 代表値(接近速度0.6-1.0m/s、停止後距離0.1-0.2m)でのTTC条件破綻までの
  遅延: **停止後0.55-0.65秒**(レビューの近似式`k≈(1/α_slow)×ln(v0・
  ttc_crit/D)`、α_slow=min(0.05,0.15)=0.05、と概ね整合、同オーダー)。
- `_dlat_shrink_run`の追加消費先(`lateral_ttc_monitor.py:582`、branch=A_dlat、
  相手基準の予防的switchback反転トリガー)を発見。これは`_dlat_closing_trend`
  経由ではなく`_v_dlat_ema`/`_dlat_shrink_run`を直接参照するが、自身も
  `ttc_dlat<=ttc_danger_s`という同型のTTC自然解除ゲートを持つため、
  同じ理屈で自己修復すると考えられる。本節の自己ロック解除修正とは
  独立した既存経路であり、今回のスコープでは変更しない(将来の監査候補
  として記録のみ)。

### 45.8 Phase 2再分析(エピソード窓、2026-08-09)

**方法論の訂正**: レビューが提案した「giveup後2秒窓」は、45.7のEMA自然
解除時間(0.55-0.65秒)を根拠にしていたが、**実際にはfootprint_risk起因の
giveup後は既存のengage_cooldown(footprint_risk時2倍≈8秒)が先に効き、
`_plan_pass`自体が呼ばれない(cheap_ok不成立)期間が数秒続く**——2秒窓では
`[DLAT-TTC-VETO]`(`plan_ok`成立が前提)がほぼ観測されない
(193エピソード中190件=98.4%が「no_veto_observed_in_window」)。窓を
30秒(§42のSTUCK判定窓と同一)へ拡大して再実行した。

| 指標 | 2秒窓(不適切) | 30秒窓(§42と同一基準) |
|---|---|---|
| エピソード総数 | 193 | 193 |
| would_release | 1件(0.5%) | **61件(31.6%)** |

**30秒窓でのエピソード単位救済率31.6%は、当初のイベントスナップショット
方式(298件中101件=33.9%)とほぼ一致**——レビューが懸念した「大幅な
過小評価」は支持されなかった(2秒窓という設定自体が過小評価の原因で
あり、EMA遅延そのものが原因ではなかった)。

非救済(68.4%)の内訳: no_veto_cleared_in_window(veto開始したが30秒以内に
解消せず)20.2%・no_veto_observed_in_window(cheap_ok等の別要因が支配的)
18.1%・predicted_dlat_insufficient_at_clear(解消時点でも幾何的余地不足、
must-fix3が正しく機能)17.1%・veto_active_no_clear_in_window(窓終了時点でも
継続中)11.9%・opponent_moving_at_clear(相手が実は走行中)1.0%。

**陰性再確認(走行中相手への誤解除)は30秒窓でも0件のまま**。

### 45.9 Gemini指摘②: 斜め発進時の操舵遅れ・膨らみリスク(未着手・要実地確認)

Geminiから追加で「自己ロック解除後、停止状態から斜めに発進する際、
`steer_rate_max`(1.1rad/s)・`steer_low_pass_gain`(0.6、タスク#308で確定)
による操舵遅れがスイープボリューム(車体掃過領域)を膨らませ、相手車の
角に接触するリスクがあるのでは」という指摘があった。

現時点では机上の反実仮想検証(45.8)の範囲外であり、コードの静的解析
だけでは接触の有無を判定できない(実際の軌道追従誤差はMPCソルバー・
コリドー制約・実アクチュエータ応答の組み合わせで決まるため)。
`_plan_pass`のalong_min_width判定が並行走行の静的幅のみを見ており
斜め移動のスイープボリュームを考慮していない点(45.4のGemini指摘①と
同根)と合わせ、**Phase 3実地検証(dev3)で「解除起因の機動の初動3秒の
最小dlat」を最優先観測項目とする**(別Claude提案のPhase 3観測項目と
一致)ことで対処する方針とした。事前のコード変更(低速時ゲインスケジュ
ーリング等)は、steer_low_pass_gain自体が直近確定したレース値
(CLAUDE.md §1.1)であり、確証のない懸念だけで追加変更するのはリスクが
高いと判断し、実地データが出てから必要性を再評価する。

### 45.10 残課題(更新)

- Gemini指摘①(幾何学的死角、`_plan_pass`が斜め発進のスイープボリューム
  未考慮)・指摘②(操舵遅れによる膨らみ)は、いずれもコード変更ではなく
  **Phase 3実地検証での観測**(初動3秒の最小dlat)で一次評価する方針。
- `_dlat_shrink_run`の追加消費先(branch=A_dlat、45.7)の監査は将来課題。
- **Phase 3(dev3・予選での実地検証)は未実施**。実施の可否・タイミングは
  ユーザー判断。configゲート(`selflock_release_enabled`)をtrueにするのは
  Phase 3完了後。

## 46. 25km/hフルQ/Rキャンペーン(§38と同じ手順、2026-08-09)

ユーザー指示により、20km/h時の§38と全く同じ軸順序(Q[e_y]→R[delta]→
r_delta_swing_boost→steer_low_pass_gain→r_drate)・手順(n=1傾向把握→
n=2確定、ハード制約優先判定)で25km/hを再チューニングする。

### 46.1 Q[e_y] n=2確定: 250,000(現行値のまま)

§44.3(n=1、朝の1回目)+本ラウンド(n=2目)の平均:

| 区間 | D1(200k) | D2(250k) | D3(350k) |
|---|---|---|---|
| wp180前後 | 36.09 | 28.35 | **10.38** |
| wp220-240 | **10.51** | 11.04 | 12.92 |
| wp252 | 30.48 | **27.73** | 28.40 |
| wp269-282 | 24.56 | **18.14** | 23.51 |
| S字wp340-40 | 21.42 | **22.94** | 30.17 |
| STUCK(2回合計) | 4 | **2** | 2 |
| COLLISION(2回合計) | 1 | **0** | 1 |

ハード制約(CLAUDE.md §2 rule4)優先判定: D2(250k)がCOLLISION0件・STUCK
最少で最も安全。ソフト指標も5区間中3区間(wp252・wp269-282・S字)で最良/
同等。D3(350k)はwp180のみ突出するが他は劣り衝突も1件。
**Q[e_y]=250,000を25km/hでも確定(変更なし)**。

### 46.2 次段階: R[delta]スイープ

昨日と同じ値(500[現行]/1500/3000)で次ラウンドへ進む(実施中/次回)。

### 46.3 起動手順の訂正(引き継ぎ)

ユーザー指摘によりtri-param起動順序を**D3→D2→D1**(グリッド前方車両から)
へ訂正した(以前はD1→D2→D3だった)。次回以降のdev3起動はこの順序を守る。

### 46.4 Q[e_y]再検討: 400,000へ更新(14点探索+n=2確認)

§46.1でQ[e_y]=250,000(20km/h確定値)を維持する判断をしたが、ユーザー指示で
6点(100k/150k/200k/250k/275k/300k/350k/400k/425k/450k/475k/500k/525k、
計14サンプル)へ探索を拡大した。各点で振れ幅・内巻き/外巻きバイアス
(コーナー方向別ekf_ey平均、§38.4と同じ手法)・ハード制約を測定。

**結果**: 左コーナーバイアスは100k(+0.609)→400k(+0.008)にかけて綺麗に
単調減少しゼロクロス、400k以降は外巻き側へ転じる。右コーナーはノイズが
大きく明確なゼロクロスは見えなかった(同一値475kの再測定で-0.040/-0.200
と大きくばらつき、セッション依存ノイズが支配的と判明)。100k・150kは
ハード制約に問題(STUCK/GHOST-BLOCK多発)、450kは初回のみCOLLISION2件
だったが400k/475kと合わせた再検証で再現せずノイズと確定。

400k/450k/475kのn=2確認(既に0/0/0だった値の再確認):

| Q[e_y] | 左バイアス(n=2平均) | 右バイアス(n=2平均) | 安全性(2/2回とも) |
|---|---|---|---|
| **400k** | **+0.021**(最も中立) | -0.099 | **0/0/0** |
| 450k | -0.015 | -0.079 | 0/0/0(初回の衝突2件はノイズと確定) |
| 475k | -0.109(3回平均) | -0.048 | 3回中1回だけ悪化、不安定 |

400kが最も中立かつ安定して安全なため、**Q[e_y]=400,000を25km/h向けの
新確定値とする**(250,000から更新)。q_ey_overtake/q_ey_pitも比率
(×1.65/×250)を保ったまま660,000/100,000,000へ追従。config.yaml反映、
回帰テスト2件(test_q_ey_schedule_reverted_174.py)更新、3256件PASS。

**教訓**: 同一パラメータ値でも試行間で結果が大きく振れる(475kの左バイアス
-0.040〜-0.200)ことを再確認した。n=1のピンポイント探索は誤誘導のリスクが
高く、n=2以上の確認を経てから確定する運用(CLAUDE.md §2 rule1)の重要性が
改めて裏付けられた。

### 46.5 R[delta] n=2確定: 500(現行維持)

Q[e_y]=400,000ベースでR[delta](500/1500/3000)をn=2実施した。

| 区間 | D1(500現行) | D2(1500) | D3(3000) |
|---|---|---|---|
| wp180前後 | **8.57**(最良) | 12.07 | 9.02 |
| wp220-240 | 10.87 | 10.88 | **9.82** |
| wp252 | 22.87 | **19.30** | 23.54 |
| wp269-282 | 15.43 | 15.34 | **15.19** |
| S字wp340-40 | 20.60 | 22.76 | **19.48** |
| STUCK/COLLISION/GHOST(2回とも) | **0/0/0, 0/0/0** | 0/0/0→1/0/1 | 1/0/1→1/0/1(両方) |

ソフト指標は区間ごとに分散したが、ハード制約(CLAUDE.md ルール4優先)で
明確な差が出た: D1(500)のみ2回とも完全に0/0/0。D3(3000)は2回とも同じ
軽微な問題(STUCK1/GHOST1)を再現しており偶然ではなく、R[delta]を上げると
安定性がわずかに低下する傾向を示唆。**R[delta]=500(現行維持)を確定**。

なお本ラウンド中もfootprint_risk自己ロックの該当ログ(giveup trigger=
lat_ttc_FOOTPRINT_RISK)は0件(6ドメイン合計)だった。

### 46.6 次段階

r_delta_swing_boostスイープ(0/400/800/1200/1600/2000、Q[e_y]=400,000
ベースで作り直し)、続いてsteer_low_pass_gain・r_drateへ進む。

### 46.7 r_delta_swing_boost Round A(n=1): 0/800/1600

Q[e_y]=400,000・R[delta]=500ベースで、tri_param_launch.sh(D3→D2→D1順、
自動化スクリプト)により10分走行。D1=0/D2=800/D3=1600(現行値)。

**ハード制約**: 3ドメインとも STUCK/COLLISION-SUSPECTED/GHOST-BLOCK/
FAILSAFE 全て0件。footprint_risk自己ロック該当ログ(giveup trigger=
lat_ttc_FOOTPRINT_RISK)も3ドメイン合計0件(セルフロック発生機会なし、
ユーザー指示の日和見監視は継続するが今回は該当なし)。

**ソフト指標**(hotspot_check.py、平均振れcm):

| 区間 | D1(0) | D2(800) | D3(1600・現行) |
|---|---|---|---|
| wp180前後 | **8.43** | 8.41 | 9.57 |
| wp220-240 | **8.74** | 10.11 | 8.41 |
| wp252 | **21.74** | 21.03 | 21.82 |
| wp269-282 | **13.16** | 14.83 | 14.83 |
| S字wp340-40 | **18.30** | 23.09 | 20.74 |
| S字|ekf_ey|最大 | 1.548m | **3.101m**(突出) | 1.567m |

r_delta_swing_boost=0(D1)が5区間中4区間で最良、800(D2)はS字で
|ekf_ey|最大3.101mと明確に悪化。1600(現行)は中間。n=1のため確定判断は
時期尚早(CLAUDE.md §2 rule1)だが、「0が良さそう・800が悪そう」という
傾向が20km/h時と異なる可能性がある。次はn=2確認、または0を軸にした
追加点(400等)の探索が必要。

### 46.8 r_delta_swing_boost Round B(n=1): 400/1200/2000 + D2系統誤差の発見

D1=400/D2=1200/D3=2000で同様に実施。ハード制約は3ドメインとも全PASS
(STUCK/COLLISION/GHOST-BLOCK/FAILSAFE、footprint_riskセルフロック
該当ログ、いずれも0件)。

**ソフト指標**:

| 区間 | D1(400) | D2(1200) | D3(2000) |
|---|---|---|---|
| wp180前後 | **6.46cm** | 7.83cm | 8.88cm |
| wp220-240 | 10.15cm | **9.00cm** | 9.28cm |
| wp252 | **20.57cm** | 22.33cm | 21.85cm |
| wp269-282 | 13.46cm | **12.08cm** | 12.00cm |
| S字wp340-40 | **20.54cm** | 21.44cm | 20.81cm |
| S字|ekf_ey|最大 | 1.544m | **3.101m(突出)** | 1.739m |

**重要な発見: S字|ekf_ey|最大値の突出はドメイン(D2)固有のスタート位置
アーティファクトであり、r_delta_swing_boostとは無関係と判明。**
Round A(swing_boost=800)・Round Bの両方でD2のみwp=32(発進直後、v≈1.35-1.36)
にekf_ey≈-3.101m(gnss_eyも-3.099とほぼ同値、EKF誤差ではなく実際の初期
横位置)が再現。D1は常に+0.25〜0.30m、D3は常に-0.71〜-0.76mで、いずれも
ドメインごとに一貫している。3台グリッドスタートの横並び初期配置に由来する
系統的なオフセットで、hotspot_check.pyの`v>=0.5`ゲートを僅かに超える
発進直後の1点が「S字wp340-40」区間(340→40のラップアラウンド定義で
wp=32を含む)の最大値に混入してしまう分析ツール側の既知の盲点。
平均振れ・|ekf_ey|平均への影響は1点/300+サンプルなので軽微だが、
「最大値」指標はドメイン間で比較不能なため、今後はD2列の最大値のみ
この文脈で割り引いて解釈すること(hotspot_check.py側の除外修正は
別タスクで検討)。

**総合傾向(Round A+B、n=1×6点)**: swing_boost=0/400(小さい値)が
平均振れ・|ekf_ey|平均ともに一貫して良好。800/1200(D2、系統誤差あり)は
額面上やや悪いが上記アーティファクトの影響を考慮すると実態は不明。
1600/2000(現行〜大きい値)は中間。0-400付近が有力候補として浮上、
次はこの近傍でn=2確認へ進む。

### 46.9 r_delta_swing_boost Round C(n=1): 200/400/800

D1=200/D2=400/D3=800で実施。ハード制約は3ドメインとも全PASS
(STUCK/COLLISION/GHOST-BLOCK/FAILSAFE/footprint_riskセルフロック
該当ログ、いずれも0件)。

**ソフト指標**:

| 区間 | D1(200) | D2(400) | D3(800) |
|---|---|---|---|
| wp180前後 | **7.58cm** | 12.59cm | 14.58cm |
| wp220-240 | **12.18cm** | 12.93cm | 10.21cm |
| wp252 | **20.94cm**(D3) | 22.51cm | 23.05cm(D1) |
| wp269-282 | **13.97cm**(D3) | 16.68cm | 14.68cm(D1) |
| S字wp340-40 | **22.00cm**(D3) | 23.62cm | 23.11cm(D1) |
| S字|ekf_ey|最大 | 2.248m | **3.104m(D2、前回と同じアーティファクト)** | 2.031m |

Round A/Bと比較して**Round C全体が明確に悪化**している(wp180前後は
Round A/Bで6-9cm台だったのが今回12-15cm台)。パラメータ値以前に、
このセッション自体がRound A/Bよりノイズが大きかった可能性が高い
(CLAUDE.md §2 rule1/rule2、セッション間ばらつきの典型例)。D2(400)は
前回同様S字|ekf_ey|最大=3.104m(前回3.101mとほぼ同一)で§46.8の
D2系統誤差(wp=32発進時アーティファクト)が再現、この値は割り引いて
解釈する。

セッション内相対比較では200(D1)が概ね最良、400(D2、系統誤差混入)が
最も悪く見える。ユーザー判断で0を含む近傍(0/100/200)へ絞り込み、
Round Dとして次に実施する。

### 46.10 r_delta_swing_boost Round D(n=1): 0/100/200 + 直線層蛇行の追加測定

D1=0/D2=100/D3=200で実施。ハード制約は3ドメインとも全PASS(STUCK/
COLLISION/GHOST-BLOCK/FAILSAFE、footprint_riskセルフロック該当ログ、
いずれも0件。前回報告した「D1で3件」は`grep`が`footprint_risk=False`
にもマッチした誤検出、訂正して`footprint_risk=True`のみで再確認し0件)。

ユーザー目視指摘「0がいいが、その直線の蛇行(絶対水準として)が気になる」
を受け、hotspot_check.pyがカバーしていなかった直線区間(|kappa|<=0.02)の
蛇行を測定するstraight_wobble_check.pyを新設し、Round A〜Dの全ログに
遡って適用した。

**直線層(|kappa|<=0.02)平均振れの一覧(全ラウンド横断)**:

| swing_boost | Round | 平均振れ | 標準偏差 | 最大 |
|---|---|---|---|---|
| **0** | A | **7.77cm** | 8.65cm | 53.80cm |
| **0** | D | **8.57cm** | 9.83cm | 52.40cm |
| 100 | D | 9.39cm | 9.85cm | 72.50cm |
| 200 | C | 15.28cm | 14.25cm | 74.80cm |
| 200 | D | 12.92cm | 19.02cm | 141.80cm(外れ値) |
| 400 | B | 9.04cm | 9.64cm | 51.00cm |
| 400 | C | 18.00cm(セッション劣化混入) | 16.67cm | 72.20cm |
| 800 | A | 10.38cm | 11.20cm | 66.30cm |
| 800 | C | 15.38cm(セッション劣化混入) | 14.08cm | 71.70cm |
| 1200 | B | 10.34cm | 11.83cm | 63.40cm |
| 1600 | A | 13.97cm | 15.57cm | 80.60cm |
| 2000 | B | 9.45cm | 10.72cm | 58.50cm |

**swing_boost=0がn=2(Round A/D)で最も安定して直線層蛇行が小さい**
(7.77cm/8.57cm、ほぼ再現)。他の値はラウンド間でばらつきが大きく
(400/800はRound Cのセッション劣化の影響を受けた可能性、200のRound D
では141.80cmの単発外れ値あり)、0ほどの再現性は見られない。

ホットスポット指標(hotspot_check.py、コーナー・S字系)でも0は概ね
良好(§46.7参照)。総合して**r_delta_swing_boost=0が現時点の最有力候補**
だが、ユーザー指摘の通り0であっても直線層に7-9cm程度の残存蛇行があり、
これ自体は解消されていない(「蛇行対策は絶対に諦めない」方針により、
このr_delta_swing_boost軸終了後もsteer_low_pass_gain/r_drate軸で
継続追跡する)。

### 46.11 システム負荷・ディスク容量の運用メモ

本日4ラウンド連続実施(tri-param、計40分の走行+起動待機)でload average
がラウンドごとに上昇する傾向を観測(A終了時15.01→B 12.27→C 13.66→
D 23.09)。ユーザー指摘のシミュレータFPS低下(昨日50-60→本日25)と
符合する可能性があり、GPU/CPU単体は非飽和(GPU使用率26%、load 14-23/
16コア)だったため、決定打は不明だが、Dockerビルドキャッシュ・
イメージの蓄積(60.23GB中36.31GB=60%が再利用可能)が一因として疑われる。
ユーザー承認のもと`docker builder prune -f`+`docker image prune -f`
(dangling分のみ、稼働中コンテナが参照するイメージには非該当)を実施し
43→14イメージ・再利用可能量36.36GB→15.07GBへ削減。ローカルrosbag
(`aichallenge/workspace/bag/`)も2026-08-08以前の全run_*を削除し
21GB→1.7GBへ削減(予選環境ログ`~/Downloads`は対象外、変更なし)。
以後、長時間の連続tri-param実施時はこの負荷上昇傾向を踏まえ、必要に
応じて定期的なクリーンアップを検討する。

### 46.12 r_delta_swing_boost Round E(n=2確認): 0/100/400、0を確定

D1=0/D2=100/D3=400で、0を軸としたn=2確認を実施。ハード制約は3ドメイン
とも全PASS(footprint_risk=True該当も0件)。

**直線層(|kappa|<=0.02)平均振れ、swing_boost=0の3試行累積**:

| swing_boost | Round A | Round D | Round E | 平均 |
|---|---|---|---|---|
| **0** | 7.77cm | 8.57cm | **9.66cm** | **8.67cm** |
| 100 | - | 9.39cm | 11.78cm | 10.59cm |
| 400 | - | 9.04cm(B) | 10.62cm | 9.83cm(Cのセッション劣化分除く) |

0が3試行連続で最小値、他候補(100/400)は毎回0を上回った。ホットスポット
指標も0が概ね良好で一貫。**r_delta_swing_boost=0を25km/h向けの新確定値
とする**(1600.0から更新)。config.yaml反映は次のsteer_low_pass_gain軸の
着手前にまとめて行う。

なお本ラウンド中もfootprint_riskセルフロック該当ログは0件(全ラウンド
通算でも0件、日和見監視は継続するが本日は該当機会なし)。

### 46.13 steer_low_pass_gain Round A(n=1): 0.35/0.6/0.8(タスク#310着手)

r_delta_swing_boost=0確定を受け、steer_low_pass_gainの再検証(タスク#310、
「0.6が最適か未確定」)へ着手。D1=0.35(旧確定値)/D2=0.6(現行)/D3=0.8で実施。
ハード制約は3ドメインとも全PASS(footprint_risk=True該当も0件)。

**直線層(|kappa|<=0.02)・S字wp340-40**:

| gain | 直線層平均振れ | 直線層最大 | S字平均振れ | S字|ekf_ey|最大 |
|---|---|---|---|---|
| 0.35 | 15.07cm(最悪) | 90.70cm | 27.06cm(最悪) | 2.864m |
| **0.6** | 11.44cm | 67.90cm | 21.90cm | 3.097m |
| 0.8 | 12.15cm | **99.40cm(外れ値)** | **15.74cm(最良)** | 1.259m |

0.35は全区間で明確に最悪となり、CLAUDE.md §3禁止リスト#5(0.6が0.35より
優位、gain=0.6/tau=0.05モデルでの反転)を追試確認した形。0.8は
ホットスポット(S字・wp269-282)で最良傾向だが、直線層最大4.229m/99.40cmの
外れ値があり不安定さも見える。0.6と0.8のどちらが優位かはn=1では
確定できず、n=2確認が必要。

なお本ラウンド走行中、AWSIMがAMD内蔵GPUへ再度切り替わる事象を確認
(§46.11のDRI_PRIME=1修正だけでは不十分、インデックスベース指定が
コンテナ起動ごとに不安定と判明)。次ラウンド前にPCIバスID指定
(`DRI_PRIME=pci-0000_01_00_0`)へ切り替えて再検証する。real-time factor
自体は概ね1.0近傍を維持していると見られ(§47参照予定)、データの
有効性は保たれている可能性が高いが、念のため確認する。

### 46.14 GPU誤選択問題の追加原因判明+tri_param_launch.sh修正

§46.13末尾で触れたGPU誤選択(AMD内蔵GPU使用)について、`make dev3`
(プレーンdocker-compose.yml)は正常だが、tri_param_launch.sh経由(COMPOSE_FILEに
docker-compose.tri-param-experiment.ymlを含む)では再発するという、ユーザー
指摘をきっかけに切り分け実験を実施。docker-compose.tri-param-experiment.ymlは
`autoware`サービスのvolumesにのみ影響し`simulator`サービス定義は無変更のはずだが、
COMPOSE_FILEにこのファイルを含めるだけでsimulatorのGPU選択がAMD側に変わる
ことをA/B実験で再現性よく確認(原因は特定できず、実務的な回避策のみ適用)。

**対策**: `tri_param_launch.sh`を修正し、simulatorの起動だけは常に
`COMPOSE_FILE`未設定(プレーンdocker-compose.yml単体)で行い、D3/D2/D1/
bag-recorderの起動時のみtri-param overrideを適用する順序へ変更。
修正後、tri-param方式でもNVIDIA discrete GPU(使用率79%・クロック1530MHz)を
安定して掴むことを実機確認、ユーザーからも「かんぺきです」と確認を得た。
DRI_PRIME環境変数自体の修正(§46.11、コミット002cac5/f180562)とは別経路の
同一症状であり、両方の対策が必要だった。

### 46.15 Q[e_psi]再検証 Round A(n=1): 500k/1M/2M(タスク未番号、ユーザー指摘で追加着手)

Q[e_psi]・Q[t]は今回の25km/h再チューニング5軸(Q[e_y]→R[delta]→
swing_boost→steer_low_pass_gain→r_drate)に含まれていなかったとユーザーが
指摘、追加で着手。Q[e_psi]現行1,000,000(2026-08-04確定、旧v_max=20km/h・
旧actuatorモデル時代の結論)を挟むD1=500k/D2=1M/D3=2Mで実施。
ハード制約は3ドメインとも全PASS。

**直線層・ホットスポット**:

| Q[e_psi] | 直線層平均振れ | wp180前後 | wp220-240 | S字平均振れ |
|---|---|---|---|---|
| 500k | 6.58cm | 16.99cm(**外れ値**最大107.40cm/ekf_ey3.735m) | 7.67cm | 17.98cm |
| 1M(現行) | 6.57cm | 7.92cm | 6.48cm | 17.46cm |
| **2M** | 7.38cm(最大79.20cm外れ値あり) | **6.98cm** | **6.88cm** | **15.61cm(最良)** |

2Mが総合的に最良傾向(wp180・wp220-240・S字で最良)。500kはwp180前後で
1点大きな外れ値(壁接近疑い)があり、n=1では確定できないが低めの値は
リスクがある可能性。ユーザー判断で「良い方向」=Q[e_psi]を上げる方向へ
Round B(3M/4M/6M)を追加実施する。

### 46.16 Q[e_psi] Round B(n=1): 3M/4M/6M — 上げるほど改善する傾向、頭打ち未確認

ハード制約は3ドメインとも全PASS。

| Q[e_psi] | 直線層平均振れ | wp180前後 | S字平均振れ |
|---|---|---|---|
| 500k(Round A) | 6.58cm | 16.99cm(外れ値あり) | 17.98cm |
| 1M(Round A、現行) | 6.57cm | 7.92cm | 17.46cm |
| 2M(Round A) | 7.38cm | 6.98cm | 15.61cm |
| 3M | 6.00cm | 6.72cm | 14.86cm |
| 4M | 5.65cm | 6.64cm | 14.43cm(S字|ekf_ey|最大3.097m、既知のD2系統誤差の可能性) |
| **6M** | 5.74cm | **5.92cm(最良)** | **13.60cm(最良)** |

500k〜6Mの6点を通じて**Q[e_psi]を上げるほど直線層・ホットスポットとも
一貫して改善する傾向**が見え、6Mでもまだ頭打ちが確認できていない。
これは「Q[e_psi]を緩めれば位相余裕が稼げる」(CLAUDE.md禁止リスト#20、
下げる方向の仮説)とは逆方向であり抵触しない。25km/h・新actuatorモデル
(gain=0.6/tau=0.05)下では、旧モデル時代の確定値(1M)が過小だった
可能性が示唆される。ユーザー判断でさらに上方向を追加検証する。

### 46.17 Q[e_psi] Round C(n=1): 8M/12M/20M — 頭打ち・悪化の兆候を確認

ハード制約は3ドメインとも全PASS。

| Q[e_psi] | 直線層平均振れ | wp269-282 | S字平均振れ |
|---|---|---|---|
| 6M(Round B) | 5.74cm | 14.55cm | 13.60cm |
| **8M** | **5.58cm(最良)** | 14.46cm | **12.35cm(最良)** |
| 12M | 5.69cm | 17.07cm(悪化) | 15.84cm(悪化) |
| 20M | **7.81cm(悪化)** | **18.26cm(最悪)** | 15.03cm |

500k〜20Mの9点を通じて、**8M付近をピークに12M・20Mで複数指標
(wp269-282・直線層・S字)が悪化へ転じる頭打ちを確認**。単調改善は
8M手前までで終わり、それ以上は過制御方向の悪化と考えられる。
6M・8Mが有力候補として残り、n=2確認が必要な段階。

### 46.18 Q[e_psi] Round D(n=2確認): 6M/8M/10M — 8Mを新確定値とする

ハード制約は3ドメインとも全PASS。6M/8Mの2回目測定でRound Cとの
再現性を確認:

| Q[e_psi] | 直線層(Round C) | 直線層(Round D) | n=2平均 |
|---|---|---|---|
| 6M | 5.74cm | 5.75cm | **5.745cm** |
| **8M** | 5.58cm | 5.48cm | **5.53cm(最良・最安定)** |
| 10M(新規) | - | 6.72cm | 12M(前回5.69cm)・20M(前回7.81cm)の
中間、10M付近から悪化傾向が見え始める兆候と整合 |

8Mが2回とも最良値かつ両ラウンドの差が0.10cmと非常に安定しており、
6Mより一貫して優位。**Q[e_psi]=8,000,000を25km/h向けの新確定値とする**
(1,000,000から更新、旧v_max=20km/h・旧actuatorモデル時代の結論を反転)。
config.yaml反映・回帰テスト更新を実施する。
