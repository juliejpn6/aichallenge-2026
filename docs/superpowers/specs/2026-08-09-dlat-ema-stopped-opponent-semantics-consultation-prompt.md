# 相談プロンプト: 両者停止中のdlat_v_ema/dlat_shrink_run意味論(§45.6残課題)

## 背景

`2026-08-09-dlat-ttc-veto-selflock-fix-consultation-prompt.md`で依頼した
footprint_risk自己ロック解除の実装について、外部レビュー(別Claude)から
Phase 0検証事項として以下が指摘されていた:

> 両者停止中のEMA意味論(レビュー観点3): dlatが不変のとき`dlat_shrink_run`/
> `dlat_v_ema`が正しくリセット/減衰されるか確認。古い値の残留は「解除が
> 効かない」安全側の故障だが、127件の救済率を下げるため実態を報告する。

Phase 1実装・Phase 2反実仮想検証(dev3ログ209本+予選ログ6本)は完了済みで、
298件のfootprint_risk由来`[DLAT-TTC-VETO]`イベント中101件(33.9%)が
解除条件を満たし、走行中相手への誤解除は0件だった。ただしこの検証は
**ログに記録された時点のdlat_v_ema/shrink_run値をそのまま使っており、
それらの値自体が「両者停止後も古い(縮小中の)値を引きずっていないか」は
未検証**のまま反映してしまっている。この点を確認したい。

## 該当コード(`lateral_ttc_monitor.py:_update_dlat_trend`)

```python
def _update_dlat_trend(self, fwd_dlat, fwd_vid, dt) -> None:
    if fwd_dlat is None:
        self._dlat_ema = None
        self._prev_dlat_ema = None
        self._v_dlat_ema = 0.0
        self._dlat_shrink_run = 0
        ...
        return
    _dlat_vid_changed = (...)
    if _dlat_vid_changed:
        self._dlat_ema = fwd_dlat
        self._prev_dlat_ema = None
        self._v_dlat_ema = 0.0
        self._dlat_shrink_run = 0
        ...
        return
    if self._dlat_ema is None:
        self._dlat_ema = fwd_dlat
    else:
        self._dlat_ema += self.space_ema_alpha * (fwd_dlat - self._dlat_ema)
    if self._prev_dlat_ema is None:
        self._prev_dlat_ema = self._dlat_ema
        ...
        return
    _d = self._dlat_ema - self._prev_dlat_ema
    self._prev_dlat_ema = self._dlat_ema
    _v_inst_raw = _d / max(dt, 1e-3)
    _v_inst = max(-self.v_inst_max, min(self.v_inst_max, _v_inst_raw))
    self._v_dlat_ema += self.beta * (_v_inst - self._v_dlat_ema)
    if self._v_dlat_ema < 0.0:
        self._dlat_shrink_run += 1
    else:
        self._dlat_shrink_run = 0
```

`beta`の既定値は0.15(40Hz基準、`_rate_scaled_gain`で制御周波数に追従)。
`fwd_dlat`(自車〜相手の実測横距離)がリセット対象(None/対象車ID変化)以外の
経路では、**一度負に転じた`_v_dlat_ema`は指数減衰でしか0へ戻らない**
(即座のリセット経路が無い)。

## 自分で行った分析(検証してほしい)

両者が完全停止し`fwd_dlat`が定数になった場合:

1. `_dlat_ema`は`space_ema_alpha`のEMAで`fwd_dlat`(定数)へ指数収束する
2. `_d = _dlat_ema - _prev_dlat_ema`は収束につれて0へ近づく
3. `_v_inst_raw = _d/dt`も0へ近づく
4. `_v_dlat_ema`は`beta=0.15`のEMAで`_v_inst`(≈0)へ**さらに**指数収束する
5. `_v_dlat_ema`が負から非負へ転じた最初の周期で`_dlat_shrink_run`が0へ
   リセットされる

つまり「両者停止直前まで急接近していた」場合、停止後も**2段階のEMA遅延
(dlat_ema自体の収束 + v_dlat_emaの収束)**を経てから`shrink_run`がリセット
される。beta=0.15なら時定数は1/0.15≈6.67周期、3-4時定数で実用上収束と
みなすと20-27周期程度(40Hz基準で0.5-0.7秒)——**恒久的な固着ではなく
一時的な遅延**と考えられるが、以下を検証してほしい:

## 検証してほしいこと

1. 上記の2段階EMA遅延の分析が正しいか(コードから見て見落としがないか)。
2. 実際の遅延時間(周期数/秒数)の見積もりが妥当か。特に`space_ema_alpha`
   (既定は`self._ot_ema_alpha`を参照、別途確認要)側の収束速度も
   `beta`と合わせて評価する必要があるのではないか。
3. **この遅延が、Phase 2の反実仮想検証(298件中101件=33.9%の救済率)の
   結果にどの程度影響しているか**——「本来は解除できたはずだが、
   footprint_risk直後のわずかな期間だけEMA遅延で`real_trend_veto`が
   まだTrueのまま検出され、would_release=Falseに数えられてしまった」
   ケースがどの程度混在しているか、追加のログ分析(EMA遅延を考慮した
   再計算)が必要かどうか。
4. これは「解除が効かない」方向の安全側の故障(=リグレッションではなく
   保守性の高さ)ではあるが、もし遅延が長すぎて実用上ほぼ常に
   `real_trend_veto=True`になっているなら、本来の34%という救済率
   見積もり自体が過小評価になっている可能性がある。この点への
   見解も欲しい。

## 参考: 関連ファイル

- `aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/multi_purpose_mpc_ros/lateral_ttc_monitor.py`
  (`_update_dlat_trend`関数、上記コード全文)
- `design_docs/opp_lat_pred_overlap_guard_design_20260806.md` §45(全体設計・
  Phase 0-2結果)
- 前回相談: `docs/superpowers/specs/2026-08-09-dlat-ttc-veto-selflock-fix-consultation-prompt.md`
