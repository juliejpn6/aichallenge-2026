# 相談プロンプト: footprint_risk由来is_closing_trendの自己ロック解除(ENGAGEゲート、実装済み・実地未検証)

## 背景・時系列

1. 予選ログ(`autoware(0808-05).log`、v_max=25km/h、2026-08-08夜投入)の
   2周目wp280前後を目視確認したところ、前方に停止車2台がいる場面で、
   自車が**左側は終始広く空いている(Lfree=3.6-3.96m)にもかかわらず
   18.5秒間完全停止し続け**、最終的にSTUCK(バック復帰)に陥っている
   ことが判明した。
2. ログを追跡した結果、直前に`[LAT-TTC-ACT] giveup trigger=lat_ttc_
   FOOTPRINT_RISK`(接触リスク近接によるOVERTAKING断念)が発火しており、
   その後の再ENGAGE試行が`gate=...plan=1:dlat_ttc`(=`_plan_pass`は
   左側を物理的に妥当と判定したが、`dlat_ttc_veto`で最終的にブロック)
   というログを繰り返していた。
3. コードを読み、`_dlat_closing_trend()`が`footprint_risk=True`の間は
   トレンド計算を無視して常にTrueを返す設計(2026-07-22追加、意図は
   「既に物理的接触リスクがある間は保守的に振る舞う」)であり、
   `footprint_risk`自体は「相手との現在の間隔が物理的最小幅未満」で
   発火することを確認した。停止車の真後ろで完全停止している間は
   `footprint_risk`が継続的にTrueのままになり、**「間隔を広げる方向の
   移動(=追い越し)」自体がその間隔の狭さを理由にブロックされ続ける
   自己ロック**になっていると判断した。
4. 過去2週間分(2026-07-24〜2026-08-08)のdev3ログ209本+予選ログ6本を
   横断スキャンし、footprint_risk起因のgiveup直後(30秒以内)にSTUCKが
   発生した事例190件のうち、**片側が明確に(>2.5m vs <1.5m)空いていた
   =幾何的には明らかに逃げ場があった事例が127件(67%)**であることを
   定量的に確認した。場所はwp278-286帯(既知ホットスポット)に多いが
   wp6-9・wp52-70・wp114-186・wp265・wp296・wp327など全域で発生しており、
   局所的な地形問題ではなく一般的な制御ロジックの欠陥と判断した。
5. ユーザー承認の上、修正を実装し単体回帰(3236件)PASSまで確認した。
   **ただし実地(dev3・予選)での検証はまだ行っていない**。安全判定
   ロジックの変更のため、投入前に外部レビューを希望する。

## 既存の安全機構の設計意図(なぜこのvetoがあるか)

`_dlat_closing_trend()`は「相手との横間隔が縮み続けているか」を判定する
関数で、`footprint_risk`(自車が相手にすでに物理的接触リスクがあるほど
近接している、という別の即時判定)がTrueの間は、トレンド計算によらず
常にTrueを返す:

```python
def _dlat_closing_trend(self, fwd_dlat, dlat_v_ema, dlat_shrink_run,
                         footprint_risk: bool = False) -> bool:
    if footprint_risk:
        return True
    return (dlat_shrink_run >= self._lat_ttc.min_trend_cycles
            and dlat_v_ema < 0.0
            and fwd_dlat is not None
            and (fwd_dlat / max(abs(dlat_v_ema), 1e-6))
                <= self._lat_ttc.ttc_critical_s)
```

この出力(`is_closing_trend`)は3箇所で共有される:
1. ENGAGEゲート(`_dlat_ttc_veto`) — 新規追い越しを許可するか
2. G2-RELEASE — 既存の追い越し継続を解放するか
3. force_include_vid(ICC近接除外)

3箇所とも「既に物理的接触リスクがある間はより保守的に振る舞う」という
同じ意味を持つため、共有元1箇所で拡張されている(消費箇所ごとの
個別条件追加は意図的に避けられている設計)。

## 実装した修正

`_evaluate_engage_readiness()`内、ENGAGEゲートで`_dlat_ttc_veto`を
`_can_engage`へ適用する直前に、以下の条件を全て満たす場合のみ
vetoを解除する:

1. `_plan_pass()`が既に物理的に妥当な側(`_plan_side != 0`、
   `along_min_width`基準の実測空き幅を検証済み)を見つけている
2. `footprint_risk=False`で`_dlat_closing_trend()`を再評価した
   「本来のトレンド判定」がFalse(=純粋なfootprint_risk起因のvetoで
   あり、実際に相手が急接近しているわけではない)
3. 相手が停止/低速(`fwd_vopp < opp_obstacle_speed`) —
   **走行中の相手には適用しない**(自己ロックは主に停止相手への
   接近時に起きるため、対象を絞ることで走行中相手に対する既存の
   安全マージンには一切手を入れない)

```python
if _dlat_ttc_veto and _plan_ok and _plan_side != 0:
    _real_trend_veto = self._dlat_closing_trend(
        opp_sit.fwd_dlat, lat_dec.dlat_v_ema, lat_dec.dlat_shrink_run,
        footprint_risk=False)
    _opponent_stopped = (opp_sit.fwd_vopp is not None
                          and opp_sit.fwd_vopp < self._opp_obstacle_speed)
    if not _real_trend_veto and _opponent_stopped:
        _dlat_ttc_veto = False
        # [DLAT-TTC-VETO-SELFLOCK-RELEASE]ログを1回だけ出力
```

**変更していないもの**:
- `_dlat_closing_trend()`関数自体(共有元)は無変更
- G2-RELEASE・force_include_vidの2消費先は無変更(このメソッド内
  ローカル計算のみ)
- `engage_cooldown`(giveup直後の固定/実測解除タイマー)は無変更
  (この修正が効くのはcooldown解除**後**の段階のみ)
- `_plan_pass()`自体(側選択・物理的空き幅判定)は無変更

## このプロジェクトの既往症(慎重に扱うべき理由)

このプロジェクトの安全判定ロジック(`switchback`/`cleared`/`giveup`周り)
は過去に重大リグレッションを起こした実績がある: 「switchback判定へ
`not cleared`という一見合理的なガードを追加」→ 実測で衝突4.3倍・
完走ラップ半減という重大な悪化を招き、revert済み(2026-07-15前後)。
以降、この種の判定ロジック(`cleared`/`switchback`/`giveup`周辺)への
変更は非常に慎重に扱う運用にしている(社内CLAUDE.md §1.3に明記)。

## レビューしてほしい観点

1. **この修正が同型の「良かれと思ったガードが裏目に出る」パターンに
   該当しないか**。特に、走行中の相手(`fwd_vopp >= opp_obstacle_speed`)
   を除外した条件設計で本当に十分か。
2. `_plan_pass()`が「物理的に妥当」と判定する基準(`along_min_width`、
   カート幅ベースの物理下限)が、footprint_riskが想定する接触リスクの
   基準(`along_min_width`/`along_min_length`)と同じ土俵の値かどうか
   ——ここが乖離していると「plan_okだが実際には危険」という穴になりうる。
3. `_real_trend_veto`の再計算(`footprint_risk=False`での再評価)が、
   実際に「相手が停止しているのに横方向で急接近している」ような
   別の危険ケースを見逃さないか。
4. 実地検証(dev3・予選)で確認すべき最優先の観測項目は何か
   (例: `[DLAT-TTC-VETO-SELFLOCK-RELEASE]`ログの発火頻度・その後の
   footprint_risk再発の有無等)。

## 差分ファイル

`aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/multi_purpose_mpc_ros/mpc_controller.py`
の`_evaluate_engage_readiness`メソッド内(`git diff`で確認可能、
まだコミットしていない)。単体回帰(`pytest test/`、3236件)はPASS済み、
dev3・予選での実地検証はこれから。
