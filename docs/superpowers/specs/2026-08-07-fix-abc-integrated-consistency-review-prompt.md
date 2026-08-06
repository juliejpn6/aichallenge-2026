# 相談プロンプト: Fix A'/B/C 統合整合性レビュー(状態遷移・既存処理との矛盾・考慮漏れ)

## 依頼内容

`opp_lat_pred`(対象車横方向速度の外挿予測)の不安定性に起因するオーバー
テイク時の横オフセット問題へ、3つのFixを段階的に実装した(design_docs/
opp_lat_pred_overlap_guard_design_20260806.md、全17節)。3つとも
オフライン反実仮想検証まで完了し、dev3ローカル検証(Phase 2)へ進む
直前の段階。実装フェーズの目でのレビューは複数回受けているが、**3つを
まとめて俯瞰する統合整合性レビュー**をお願いしたい:

1. Fix A'からFix Cまでが一貫した矛盾のない処理となっているか
2. 既存処理(Fix導入前から存在するOT状態機械)と矛盾は生じていないか
3. 状態遷移に考慮漏れ・バグはないか

## 背景: 3つのFixの概要

- **Fix A'**(既定ON、`overtake.lat_vel_source_tracker`): `opp_lat_pred`
  (対象車横方向速度の外挿予測)の計算方法を根本修正。V2Xトラッカーの
  既存速度推定(`tracker.velocity()`)を再利用し、片側利用(予測は必要
  クリアランスを増やす方向にのみ使う)+変位物理拘束を追加。
- **Fix B**(既定ON、`overtake.overlap_floor_enabled`): `self._ot_state
  == "OVERTAKING"`である全期間、target_mag(横オフセット目標の絶対値)を
  ノイズで縮小させない床。専用状態`_ot_overlap_floor_mag`(ピーク保持)を
  使い、毎周期`corr_bound`で再キャップ(コリドーの壁は絶対に突き破らない)。
  corr_bound無効(負転落/非有限)が続く場合は既存の`unlock_inf_cycles`
  (80周期≈2秒)を上限にタイムアウトし床の適用を止める。
- **Fix C**(既定OFF、`overtake.pending_disengage_enabled`): 並走中
  (縦方向にds<3.0m、既存の`along_min_length`+ヒステリシスマージン)の
  非緊急giveup(room_exhausted・opponent_too_fast由来、footprint_riskは
  対象外)は、離脱を有限時間(既定80周期≈2秒、既存`giveup_cycles`の2倍)
  だけ保留する。上限到達で強制的に通常のgiveup処理へ合流(安全弁)。

3つとも既定`false`から段階的にON化する設計で、現在の設定は
Fix A'=ON・Fix B=ON・Fix C=OFF(dev3検証中)。

## 状態機械の全体像(前提知識)

`self._ot_state`は`"NORMAL"` / `"OVERTAKING"` / `"STOPPING"`の3値のみ
(Fixどれも新規state値を追加していない)。代入箇所は以下の6箇所のみ
(Fix導入前から不変):

```
__init__ x2箇所(初期値"NORMAL")
giveup成立時                          -> "STOPPING"  (mpc_controller.py:6789)
新規エンゲージ成立時                   -> "OVERTAKING"(mpc_controller.py:6902)
エンゲージ非成立時                     -> "STOPPING"  (mpc_controller.py:6936)
前方クリア連続(exit_clear)            -> "NORMAL"    (mpc_controller.py:6954)
infeasibility強制                     -> "STOPPING"  (mpc_controller.py:6976)
```

1周期の処理順序(概略、上から下へ逐次実行):

```
if _n_fwd > 0:
    if self._ot_state == "OVERTAKING":
        [switchback/rescue側反転処理]
        [room_exhausted判定]
        _side_blocked = _lat_dec.force_giveup or _room_exhausted
        _giveup_now = (giveup_count>=cycles or locked==0 or side_blocked)
        [ここにFix Cの介入(_giveup_nowを条件付きでFalseへ上書き)]
        if _giveup_now:
            [既存giveup処理: state="STOPPING", side=0, cooldown設定等]
        else:
            self._ot_side = _locked
    else:
        [エンゲージ判定、can_engageならstate="OVERTAKING"]
else:
    [exit_clear連続でstate="NORMAL"]

[infeasibility強制チェック、OVERTAKING中ならstate="STOPPING"]

if self._ot_state == "OVERTAKING" and self._ot_side != 0:
    [opp_lat_pred計算(Fix A')、target_mag確定]
    _target_mag = self._apply_overlap_floor(_target_mag, _corr_bound)  # Fix B
    self._mpc.lateral_target = float(self._ot_side) * _target_mag
elif (self._ot_state == "STOPPING" and ...):
    [proactive-bias、Fix Bは適用しない(§14で意図的に撤去)]
else:
    ...
```

**重要**: giveup判定ブロック(Fix C介入含む)は、target_mag/Fix B floor
計算ブロックより**同一周期内で先に**実行され、後者は`self._ot_state`の
**その時点の実値**を読む(渡された値のキャッシュではない)。したがって
Fix Cが今回giveupを保留した場合(`_giveup_now=False`のまま状態遷移
ブロックを素通り)、`self._ot_state`は"OVERTAKING"のまま変化せず、
直後のFix Bブロックが正常に(通常のOVERTAKING継続と同じ経路で)実行
される——という理解でいるが、この理解が正しいか含めて確認してほしい。

## 具体的なコード(giveup判定ブロック、Fix C実装箇所)

```python
_side_blocked = _lat_dec.force_giveup or _room_exhausted
_giveup_now = (self._ot_giveup_count >= self._ot_giveup_cycles
                or _locked == 0 or _side_blocked)
if (self._ot_pending_disengage_enabled and _giveup_now
        and not _lat_dec.footprint_risk_triggered):
    if self._update_overlap_state(_opp_sit.fwd_ds):
        self._ot_pending_disengage_count += 1
        if self._ot_pending_disengage_count == 1:
            self.get_logger().warn(
                f"[PENDING-DISENGAGE] start side={_locked} "
                f"wp={self._mpc.model.wp_id}")
        if (self._ot_pending_disengage_count
                < self._ot_pending_disengage_max_cycles):
            _giveup_now = False  # 今回は保留、OVERTAKING継続
        else:
            self.get_logger().warn(
                f"[PENDING-DISENGAGE] resolved reason=forced_fallback "
                f"pending_count={self._ot_pending_disengage_count} "
                f"wp={self._mpc.model.wp_id}")
    else:
        if self._ot_pending_disengage_count > 0:
            self.get_logger().warn(
                f"[PENDING-DISENGAGE] resolved reason=natural_overlap_clear "
                f"pending_count={self._ot_pending_disengage_count} "
                f"wp={self._mpc.model.wp_id}")
        self._ot_pending_disengage_count = 0
else:
    # must-fix 2: giveup条件自体が不成立の周期は保留カウントを必ず0へ戻す
    self._ot_pending_disengage_count = 0
if _giveup_now:
    self._ot_pending_disengage_count = 0
    # ↓ 既存giveup処理(側消失記録・[LAT-TTC-ACT]ログ・state="STOPPING"・
    #    cooldown設定・_reset_ot_offset_state()等)、無変更のまま続く
    ...
else:
    self._ot_side = _locked
```

## 具体的なコード(target_mag/Fix B floor block)

```python
if self._ot_state == "OVERTAKING" and self._ot_side != 0:
    ... [opp_lat_pred計算、Fix A'の片側利用+変位物理拘束、168節フリーズ
         処理含む、既存のcorr_bound再キャップも含めtarget_mag確定] ...
    _target_mag = self._apply_overlap_floor(_target_mag, _corr_bound)  # Fix B
    self._mpc.lateral_target = float(self._ot_side) * _target_mag
    _lat_active_side = self._ot_side
elif (self._ot_state == "STOPPING" and _eval is not None
        and _eval.plan_ok and _eval.plan_side != 0 and _stopped_opp):
    # proactive-bias(STOPPING中の能動的空き確保)。Fix Bは意図的に不適用
    # (このtarget_magはopp_lat_predを一切参照しない固定小値+corr_bound
    #  クランプのみのため、Fix Bのノイズ対策が構造的に不要)。
    _corr_bound = self._corr_bound_ahead(_eval.plan_side)
    _target_mag = self._ot_proactive_bias_max
    if np.isfinite(_corr_bound):
        _target_mag = min(_target_mag, max(0.0, _corr_bound))
    self._mpc.lateral_target = float(_eval.plan_side) * _target_mag
    _a_target = 1.0
    _lat_active_side = _eval.plan_side
else:
    _a_target = 0.0
    _lat_active_side = 0
```

## 具体的なコード(統合リセットヘルパー、5箇所から呼ばれる)

```python
def _reset_ot_episode_tracking_state(self) -> None:
    """側変更・新規エンゲージ・OVERTAKING離脱(STUCK復帰含む)の全ての契機で
    共通に呼ぶ、エピソード単位の追跡状態リセット。_ot_side/_ot_side_locked/
    _ot_giveup_count/_ot_room_exhausted_count等には一切触れない。"""
    self._ot_last_valid_target_mag = None
    self._ot_last_valid_min_needed_mag = None
    self._ot_opp_lat_prev = None
    self._ot_opp_lat_prev_vid = None
    self._ot_opp_lat_vel_ema = None
    self._ot_opp_lat_warmup_count = 0
    self._ot_overlapping = False
    self._ot_overlap_floor_mag = None
    self._ot_overlap_floor_invalid_corr_count = 0
    self._ot_pending_disengage_count = 0
```

呼び出し箇所(5箇所、全て純粋な追加呼び出し、既存の状態遷移ロジック自体は
無変更):
1. 側反転(switchback、`_lat_dec.side_override`成立時)
2. rescue側反転(room_exhausted救済成立時)
3. 新規エンゲージ(`_eval.can_engage`成立時、state="OVERTAKING"へ遷移)
4. STUCK復帰(`_reset_ot_side_for_fresh_replan()`経由)
5. STUCK突入時(`_stuck_enter_wait_reverse()`、2026-08-07追加、
   **CLAUDE.md上の慎重領域関数**——STUCK固有の状態機械ロジック
   [`_stuck_state`/`_stuck_count`等]には一切触れず、この1行のみを追加した)

## `_update_overlap_state()`の二重利用について

```python
def _update_overlap_state(self, opp_ds_now) -> bool:
    """対象車と縦方向にオーバーラップ中(=真横に近接)かをヒステリシス付きで
    判定する。footprint_risk判定と同じ物理的下限(along_min_length)を
    再利用。侵入判定(enter_thr=2.5m)より解除判定(exit_thr=3.0m)を
    広く取る。opp_ds_now is Noneの場合は直前の状態を維持。"""
    if opp_ds_now is None:
        return self._ot_overlapping
    enter_thr = self._along_min_length + self._ot_overlap_margin_m
    exit_thr = self._along_min_length + self._ot_overlap_margin_m * 2.0
    d = abs(opp_ds_now)
    self._ot_overlapping = (
        d < exit_thr if self._ot_overlapping else d < enter_thr)
    return self._ot_overlapping
```

このメソッドは当初Fix B/C共通の「並走中」判定として設計したが、
Fix Bのオフライン検証で「動機事例ではdsが終始3〜13m台でこの閾値
[2.5〜3.0m]に一度も到達しない」というスコープ取り違えが判明し、
Fix Bは`self._ot_state=="OVERTAKING"`ベースへ再設計した(§14)。
**現在このメソッドを呼ぶのはFix Cのみ**(`self._update_overlap_state
(_opp_sit.fwd_ds)`、giveup判定ブロック内)。`self._ot_overlapping`という
状態変数は実質Fix C専用になったが、変数名・リセット箇所(上記5箇所の
ヘルパー内)は「Fix B/C共通」だった設計時の名残のまま。

## 具体的に確認してほしい点

1. **周期内実行順序の理解の妥当性**: 上記「重要」セクションで述べた
   理解(giveup判定→target_mag/Fix B、同一周期内でこの順、`self._ot_state`
   は実値を都度読む)が正しいか。もし誤りがあれば、Fix Cが保留した周期に
   target_mag/Fix Bブロックが正しく実行されない(あるいは逆に、giveupが
   実際に成立した周期でも古い"OVERTAKING"を見てしまう)という重大な
   バグになりうるため、最優先で確認してほしい。
2. **`_update_overlap_state()`の状態変数命名の整合性**: `_ot_overlapping`
   という変数名が実質Fix C専用になった今、この名残が将来の実装者を
   誤解させるリスクをどう評価するか(コメントで明記済みだが、リネーム
   すべきか)。
3. **Fix B(OVERTAKING全期間スコープ)とFix C(dsベース、稀にしか発動
   しない)の並走時の相互作用**: Fix Cがgiveupを保留している間、
   `self._ot_state`は"OVERTAKING"のまま変わらないため、Fix Bの床は
   保留期間中も通常通り作用し続ける。これは意図通りか(=保留中も
   クリアランスを維持し続けようとする)、それとも保留中は別の挙動
   [床を止める等]が必要か。
4. **リセット呼び出し5箇所の網羅性**: 側反転・rescue側反転・新規
   エンゲージ・STUCK復帰・STUCK突入、の5箇所で全ての「OVERTAKING
   エピソードの実質的な仕切り直し」を網羅できているか。見落としている
   契機(例: 対象車IDが同一のまま何らかの理由で床/保留カウントだけ
   クリアすべき瞬間)はないか。
5. **`_ot_pending_disengage_count`のリセットタイミングの厳密性**: 現在
   の実装は(a)ゲート条件不成立時の`else`節、(b)`_giveup_now`最終決定後の
   `if _giveup_now:`内、(c)並走解消時の`else`節内、の3箇所+統合ヘルパー
   経由の5箇所で計8箇所からリセットされる。冗長に見えるこの多重リセットが
   実際には全て異なる文脈(ゲートOFF/giveup不成立/自然解消/エピソード
   境界)をカバーしており、抜け漏れも二重発火の実害もないか、状態遷移表
   形式で確認してほしい。
6. **既存処理(Fix導入前)との非矛盾性**: 168節フリーズ機構
   (`_ot_last_valid_target_mag`)・230節続報の`_reset_ot_offset_state()`
   ・91節のENGAGE可否判定・159節のswitchback整合性修正、といった
   Fix導入前から存在する既存の安全機構が、Fix A'/B/Cのどれによっても
   迂回・弱体化されていないか(特にFix Bの床とFix Cの保留が、既存の
   168節フリーズが対処してきた「corr_bound崩壊」シナリオへどう
   重畳するか)。
7. **状態遷移図の考慮漏れ**: task#279で作成した`_ot_state`遷移図
   (NORMAL/OVERTAKING/STOPPINGの3値、6箇所の代入)に、Fix Cの「保留中の
   OVERTAKING」という準状態(新規state値は追加していないが、通常の
   OVERTAKINGとは"giveup条件は成立しているが保留中"という点で意味論が
   異なる)を明示的に図示すべきか。図示しない場合の実務上のリスクは
   あるか。

## 制約・注意事項

- CLAUDE.md §1.3の慎重領域(`_stuck_enter_wait_reverse`含むSTUCK復帰関連
  関数、switchback/rescue判定、`cleared`判定周り)には、今回いずれも
  「既存ロジックを無変更のまま、追加の1行呼び出しのみ」で対応した
  つもりだが、この方針自体が適切かも含めて評価してほしい。
- 3つのFixとも既定OFFから段階的にON化する設計(現在Fix A'=ON・
  Fix B=ON・Fix C=OFF)。ゲートOFF時は各Fixとも既存動作とビット等価。
- 全体回帰スイート(3234件)PASS済み、オフライン反実仮想検証(2フィクス
  チャ+全giveupイベント横展開)も完了済み。今回は「実装の正しさ」の
  レビューではなく「3つを俯瞰した統合整合性」のレビューをお願いしたい。
