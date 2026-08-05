---
title: engage_cooldown(4秒固定待機)の早期解除設計相談プロンプト
date: 2026-08-05
status: draft
---

# 背景・現在の環境

ROS2/Autoware自律走行レーシングカート(spatial MPC制御、40Hz制御ループ)の
オーバーテイク(OT)処理。giveup(オーバーテイク断念)後、再エンゲージを
一定期間ブロックするクールダウン機構があります。

```python
self._ot_engage_cooldown = (
    self._ot_engage_cooldown_cycles * 2   # footprint_risk起因なら2倍(≈8秒)
    if _lat_dec.footprint_risk_triggered
    else self._ot_engage_cooldown_cycles)  # それ以外は通常(≈4秒、160周期@40Hz)
```

**このクールダウンの本来の目的は「往復ループ(ピンポン現象)の防止」**です。
giveupした直後に同じ条件のまますぐ再試行すると、同じ結果(再giveup)を
高頻度で繰り返すだけの無駄なループになる、という過去の実測(0720-04
wp240-243、完全停止車への3回以上のENGAGE試行が0.5〜1秒以内に断念する
往復を約9秒間繰り返していた)を踏まえて導入されました。

## 既に確立済みの「実測解消」パターン(footprint_risk専用)

giveup理由がfootprint_risk(相手への接近が物理的に危険)の場合に限り、
固定タイマーを待たずに、条件が実際に解消したかを実測して早期解除する
仕組みが既に実装されています(148節②):

```python
_cd_clear = (
    (self._ot_footprint_risk_clear_count >= self._ot_engage_debounce
     or self._ot_engage_cooldown == 0)
    if self._ot_footprint_risk_gated
    else self._ot_engage_cooldown == 0)
```

`self._ot_footprint_risk_clear_count`は、危険域判定(`_fp_near_zone`)が
不成立の周期を連続カウントし、`self._ot_engage_debounce`(8周期≈0.2秒、
既存のフリッカー防止デバウンス値の再利用)に達したら早期解除します。
固定タイマー経路とOR結合されているため、「早く解消すれば早く再挑戦できる、
最悪でも従来通り8秒で必ず解除される」という安全側の設計です。

## 問題: 他のgiveup理由には実測解消経路が無い

giveup理由は実装上3系統あります:

1. **相手が速すぎる**(`_ot_giveup_count >= _ot_giveup_cycles`): 接近速度
   `(self._v_pot - fwd_vopp)`が`self._opp_giveup_closing`(0.2m/s)未満の
   状態が`_ot_giveup_cycles`(40周期≈1秒)連続したら断念。
2. **ロック外れ**(`_locked == 0`): 詳細不明、側の追跡自体が失われたケース。
3. **空間的失敗**(`_side_blocked`、`force_giveup or room_exhausted`):
   `room_exhausted`は`self._corr_bound_ahead(locked_side)`(コリドー先読み
   最小値)が非正の状態が`_ot_giveup_cycles`連続したら断念。

これらは全て**固定タイマー(4秒)のみ**で解除され、footprint_riskのような
実測解消経路がありません。つまり「相手が速すぎてgiveup」した1秒後に相手が
減速して状況が好転しても、4秒間は一切ENGAGE試行できず、これはタイムロスに
直結します。

## 相談したいこと

footprint_riskで確立済みの「実測解消+固定タイマーのOR結合」パターンを、
①(相手が速すぎる)・③(room_exhausted)にも横展開したいと考えています。

**候補の実測解消条件**:
- ①用: `(self._v_pot - fwd_vopp) >= self._opp_giveup_closing`(接近速度が
  giveup閾値を上回った=再び追いつける状態に戻った)が一定周期連続。
- ③用: `self._corr_bound_ahead(locked_side) > 0.0`(該当側のコリドー先読み
  最小値がプラスに回復した)が一定周期連続。

1. 上記2つの実測解消条件の設計は妥当ですか? 見落としているエッジケースは
   ありますか(例えば、①の接近速度が閾値をまたいで細かく振動するケースで
   デバウンスが正しく機能するか等)?
2. ②(ロック外れ)には実測解消条件を設けず、従来通り固定タイマーのみに
   留めるべきだと考えていますが、この判断は妥当ですか?
3. 往復ループ(ピンポン現象)のリスクをどう評価すべきですか? footprint_risk
   では8周期(≈0.2秒)のデバウンスを使っていますが、①③でも同じ値を
   流用してよいか、それとも理由ごとに異なるデバウンス値が必要か?
4. 本日実装したside別engage_cooldown(task#265、giveup理由が空間的失敗
   [_side_blockedかつfootprint_risk起因でない]の場合のみ該当側だけを
   ブロックし、グローバルタイマーは0へリセットする設計)との組み合わせを
   どう設計すべきですか? 側別クールダウンにも同様の実測解消経路を追加する
   必要がありますか、それとも固定タイマー(現状のside別実装)のままで
   十分ですか?
5. CLAUDE.mdの慎重さの原則(giveup判定自体・LAT-TTC系の安全判定への安易な
   変更は禁止、過去に82/83節で「clearedへのガード追加」が衝突4.3倍という
   重大リグレッションを招いた実績がある)に照らし、今回の変更(giveup判定
   自体は変えず、クールダウンの「解除」条件だけを早める)がこの慎重扱い
   領域に抵触するリスクをどう評価しますか?

以上、率直な意見・懸念点・代替案があればいただけると助かります。
