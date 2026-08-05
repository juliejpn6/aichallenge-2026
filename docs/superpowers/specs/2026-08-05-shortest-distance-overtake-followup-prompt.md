---
title: 最短距離オーバーテイク+side別cooldown フォローアップ相談プロンプト
date: 2026-08-05
status: draft
---

# 背景

前回のプロンプト(`2026-08-05-shortest-distance-overtake-consultation-prompt.md`)
に対し、Gemini・別Claudeインスタンスの両方から回答をもらいました。回答内容を
実際のコード(`_evaluate_engage_readiness`関数、`_close_enough`計算、giveup時の
cooldown設定箇所)と突き合わせて検証したところ、いくつか確認・整理したい点が
出てきたので、追加でレビューをお願いします。

## 前回の回答の要旨(参考)

- **Gemini**: 追いつき地点予測はオイラー法の反復シミュレーション。OVERTAKING
  状態に入った後でオフセットランプの開始を遅らせる設計。TTC閾値をランプ待機中
  は早めにシフトさせる案。
- **別Claude**: 追いつき地点予測は区間ごとの閉形式(反復不要)。**ENGAGE自体を
  遅らせる**(`_close_enough`計算の拡張)ことを推奨し、OVERTAKING状態に新しい
  中間状態を作らない設計。TTC閾値は触らずマージン(1〜1.5秒)で吸収。
  Cooldownはgiveup原因を(a)空間的失敗→片側ブロック、(b)footprint_risk→両側
  グローバルの2系統に分けて、`_ot_engage_cooldown_l/_r`2カウンタ+config
  フラグ1個で実装する案。

## 検証① 「ENGAGE自体を遅らせる」設計をコードで裏付け

実際の`_evaluate_engage_readiness`のコードを確認したところ、別Claude案を
支持する構造になっていました:

```python
_t_reach_profile = None
_is_stopped_for_profile = (opp_sit.fwd_vopp is not None
                            and opp_sit.fwd_vopp < self._opp_obstacle_speed)
if _is_stopped_for_profile and scan.get("fwd_wp") is not None:
    _t_reach_profile = self._predicted_time_to_wp(
        int(self._mpc.model.wp_id), int(scan["fwd_wp"]), self._fwd_max_consider)
if _t_reach_profile is not None:
    _t_reach_thr = self._ot_t_lateral + self._ot_pass_clear / _closing_est
    _close_enough = _t_reach_profile <= _t_reach_thr
else:
    _close_enough = (opp_sit.fwd_ds is not None
                      and opp_sit.fwd_ds <= _engage_dist_dynamic)
...
_cheap_ok = (self._ot_enable and (left_ok or right_ok)
             and self._ot_infeasible_latch == 0
             and _cd_clear
             and self._ot_worth_count >= self._ot_engage_debounce
             and _on_path and _ego_ready and _close_enough
             and not being_overtaken)
```

`_close_enough`がFalseの間、`_cheap_ok`全体がFalseになり`_ot_state`は
`"STOPPING"`のまま(既存の前車追従ロジック`icc_stop`が働く)です。つまり
`_is_stopped_for_profile`条件を「相手が走行中の場合」にも拡張し、2体問題版の
`_predicted_time_to_wp`をここに差し込めば、追いつくまでは自然に既存の
STOPPING状態(=レースライン/コリドー中心の走行)が維持されます。新しい中間
状態は不要という別Claude案の主張はコード構造から見て妥当と判断しました。

### 確認したいこと①

`_close_enough`をFalseに保つ(意図的にENGAGEを遅らせる)期間、`_ot_state=
"STOPPING"`中の速度制御は`icc_stop`(前車追従、相手の速度に同期して減速する
ロジック)が担います。ここで懸念があります: 「自車の方が速く、いずれ追いつく
(=ENGAGEすべき)」状況であるにもかかわらず、`icc_stop`が相手の(遅い)速度に
同期して自車を不要に減速させてしまわないでしょうか? もしそうなら、
「追いつくまでレースラインを維持する」ことと「追いつくまで無駄に減速しない」
ことを両立させるには、STOPPING状態の速度制御側にも「追いつき予測が近い将来
成立する見込みなら、相手の速度でなく自車の計画速度で走ってよい」という
条件を追加する必要が出てくる可能性があります。この点についてどう設計すべきか
意見をください。

### 確認したいこと②

`_ot_worth_count`(ENGAGE判定の連続成立カウンタ、`_ot_engage_debounce`
周期以上必要)は`_close_enough`とは独立に、`pass_worth`という別条件で
毎周期更新されています。「追いつき予測でまだ早い」と`_close_enough`が
継続的にFalseを返す間、`_ot_worth_count`は通常通りカウントされ続けて
問題ないという理解で合っていますか(=`_close_enough`だけが「まだ待て」を
表現し、他の判定要素の意味は変えない設計になっているか)?

## 検証② cooldownのgiveup原因は実は3系統

giveup発生箇所の実コードを確認したところ、原因は以下の3系統でした
(前回の別Claude案は(a)(b)の2系統を想定していました):

```python
if (self._ot_giveup_count >= self._ot_giveup_cycles   # ①相手が速すぎる(側と無関係)
        or _locked == 0                                  # ②ロック外れ
        or _side_blocked):                                # ③空間的失敗(force_giveup or room_exhausted)
    if _side_blocked and _locked != 0:
        self._ot_prev_side = _locked   # 側反転ヒステリシス用に記録(①②では記録しない)
        ...
    if _side_blocked:
        _giveup_trigger = ("room_exhausted" if (_room_exhausted and not _lat_dec.force_giveup)
                            else f"lat_ttc_{_lat_dec.branch}")  # footprint_risk等はここに含まれる
```

①(相手が速すぎる)は現状「側と無関係の理由」として扱われており、
`_ot_prev_side`(側反転ヒステリシス用)への記録もスキップされています。

### 確認したいこと③

side別cooldownを導入する場合、①(相手が速すぎてgiveup)は側別クールダウンの
対象にすべきでしょうか、それとも「側と無関係」という現状の扱いを踏襲し、
両側ともブロックしない(クールダウン自体を掛けない)べきでしょうか? 別Claude
案の(a)(b)分類には①の扱いが明示されていませんでした。「相手が速すぎる」は
本質的に相手そのものの属性(footprint_riskと同じくらい「側と無関係」に見える)
なので、②③と同じ両側ブロックにすべきか、それとも①はそもそも再エンゲージを
即座に試みても無駄なので短いクールダウンで十分か、意見をください。

### 確認したいこと④

3台以上(dev3/dev4環境、本プロジェクトの標準検証環境)で複数の対戦車が近傍に
いる場合、side単独キー(対象車を区別しない)で本当に十分でしょうか? 例えば
「d1に対して右側でgiveup」した直後、別の対戦車d2が右側に新たに現れた場合、
side単独クールダウンだと(相手が変わったにもかかわらず)右側への再エンゲージが
引き続きブロックされます。前回の回答では「近傍の対戦車は通常≤2台」という
前提で(side, vid)複合キーを不要と判断されていましたが、本プロジェクトの
dev3実測(2026-08-05当日)ではwp278-282帯で3台が同時に密集する事例を複数回
確認しています。この前提は再考が必要でしょうか?

以上、前回の回答を踏まえた上での追加の確認点です。率直な意見をお願いします。
