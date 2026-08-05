---
title: opp_lat_pred根本修正+並走ガード+離脱意味論 設計レビュー依頼プロンプト
date: 2026-08-06
status: draft
---

# 依頼

これまでの相談(13Hzマッチング仮説の確定、対策候補②④の合意)を踏まえ、
実装前の詳細設計を完成させました。**実装前にこの設計自体のレビュー**を
お願いします。設計書全文は
`design_docs/opp_lat_pred_overlap_guard_design_20260806.md`です(以下に
要旨を再掲)。

## 設計の要旨

### Fix A(根本原因の修正): 対象車位置の自前差分をやめ、既存の
V2Xトラッカー速度推定を再利用する

```python
def _estimate_opp_lateral_velocity(self, vid, wp):
    tracker = getattr(self, "_v2x_tracker", None)
    if tracker is None:
        return None
    vx, vy = tracker.velocity(vid)  # 既存の窓端点差分(13Hzスパイク耐性)
    return -math.sin(wp.psi) * vx + math.cos(wp.psi) * vy
```

`_scan_traffic()`の`v_long`計算(既存、mpc_controller.py:3416-3417)と
同一の`tracker.velocity()`・同一の回転行列を再利用し、自前の位置差分
(旧実装、13Hz階段状データを40Hz固定dtで割っていた誤り)を廃止する。
副産物として、EMA関連の状態変数4個(`_ot_opp_lat_prev`等)と、それに
伴う4箇所の重複リセットコードが不要になり削除できる。

### Fix B(並走中オフセット床): 縦オーバーラップ判定を共有ヘルパー化し、
2箇所(OVERTAKING分岐・STOPPING/proactive-bias分岐)から呼ぶ

```python
def _update_overlap_state(self, opp_ds_now):
    """ヒステリシス付き並走判定。along_min_length(footprint_risk判定と
    共通の既存定数)を流用、新規距離定数は導入しない。"""
    if opp_ds_now is None:
        return self._ot_overlapping  # 鮮度切れ=継続中とみなす(保守側)
    enter_thr = self._along_min_length + self._ot_overlap_margin_m
    exit_thr = self._along_min_length + self._ot_overlap_margin_m * 2.0
    d = abs(opp_ds_now)
    self._ot_overlapping = (d < exit_thr) if self._ot_overlapping else (d < enter_thr)
    return self._ot_overlapping

def _apply_overlap_floor(self, target_mag, opp_ds_now):
    """並走中は既存の168節フリーズ値(self._ot_last_valid_target_mag、
    corr_bound込みで既に安全な値)を床として使う(新規床専用変数を
    増やさない)。コリドー実測は常に優先(このフリーズ値自体が
    corr_bound込みのため、壁より広い床は原理的に発生しない)。"""
    if self._update_overlap_state(opp_ds_now) and self._ot_last_valid_target_mag is not None:
        target_mag = max(target_mag, self._ot_last_valid_target_mag)
    return target_mag
```

### Fix C(並走中の離脱保留): giveup判定成立時、緊急系(footprint_risk)
以外は並走解消まで有限時間保留する

```python
_giveup_now = (self._ot_giveup_count >= self._ot_giveup_cycles
               or _locked == 0 or _side_blocked)
if _giveup_now and not _lat_dec.footprint_risk_triggered:
    if self._update_overlap_state(_opp_sit.fwd_ds):
        self._ot_pending_disengage_count += 1
        if self._ot_pending_disengage_count < self._ot_pending_disengage_max_cycles:
            _giveup_now = False  # 保留、OVERTAKING継続
    else:
        self._ot_pending_disengage_count = 0
if _giveup_now:
    self._ot_pending_disengage_count = 0
    # ↓ 既存のgiveup処理(state=STOPPING/side=0/_reset_ot_offset_state()等)、無変更
    ...
```

**必須のフェイルセーフ**: `_ot_pending_disengage_max_cycles`
(既定`_ot_giveup_cycles*2`目安)に達したら、並走が解消していなくても
強制的に通常giveupへ合流する(無期限保留の禁止、82/83節の教訓を反映)。

### リセット統合

既存の4箇所(側反転・rescue反転・新規エンゲージ・NORMAL復帰)に重複実装
されていた6行のリセットブロックを、Fix A適用で2行(`target_mag`関連の
み)に縮小した上で、Fix B/Cの新規状態を含めた共有ヘルパー
`_reset_ot_episode_tracking_state()`へ統合する。

## このプロジェクトの制約(再掲・重要)

- switchback/rescue判定・`cleared`判定そのものには一切触れない設計に
  なっている(82/83節: `cleared`判定周りへの安易なガード追加が衝突4.3倍・
  完走ラップ半減という重大リグレッションを招いた前例があるため)。
- footprint_risk・LAT-TTC強制giveupは緊急系として全て現行の即時挙動を
  維持し、Fix B/Cの対象外としている。
- `_reset_ot_offset_state()`(`lateral_target=0.0`即時ゼロ化)は
  230節続報で「stale offsetがinfeasibilityカスケードを招く」問題への
  対処として導入された経緯があり、単純な呼び出し削除・遅延はできない
  ため、Fix Cは「離脱そのものを有限時間保留する」形にした(ゼロ化自体は
  無変更のまま)。

## レビューしていただきたい観点

1. **状態遷移の一貫性**: `_ot_state`(NORMAL/STOPPING/OVERTAKING)・
   `_ot_side`/`_ot_side_locked`・`_ot_pending_disengage_count`の組み合わせで、
   矛盾する状態(例: pending中に側だけ変わる等)が生じないか。特に
   switchback(側反転、Fix Cのgiveup判定より前で評価される既存分岐)と
   Fix Cの保留が同一周期で競合するケースはないか。
2. **Fix A(tracker.velocity()再利用)の妥当性**: `tracker.velocity()`は
   `v2x_vehicle_tracker.py`の`clamp_hold_enabled`(既定false、d2固有の
   V2X異常対策、8月3日実装)とも関わる値です。この既存機構との相互作用に
   問題はないか。
3. **Fix B(オフセット床)の副作用**: 既存の`_ot_last_valid_target_mag`
   (168節、コリドー崩壊時の凍結)を並走ガードの床として二重利用する
   設計にしていますが、この2つの用途(コリドー崩壊対策/並走ガード)を
   同一変数に持たせることで生じうる想定外の相互作用はないか。
4. **Fix C(保留)のフェイルセーフ設計**: `_ot_pending_disengage_max_cycles`
   の妥当な既定値、および保留中にegoが実際にSTUCK(v≈0)へ陥った場合の
   相互作用(設計書ではSTUCK検知は独立と想定していますが、確認をお願い
   したいです)。
5. **低侵襲性**: Fix Cの実装方式(新規カウンタ`_ot_pending_disengage_count`
   の追加)より、既存の`_ot_giveup_count`を転用する等、より状態変数を
   増やさない実装方法はありますか。
6. **見落としている箇所**: `_target_mag`計算・giveup判定は
   OVERTAKING分岐以外にも(STOPPING/proactive-bias分岐、STUCK-PUSH等)
   類似ロジックがある可能性があります。Fix B/Cを適用すべき他の箇所を
   見落としていないか確認をお願いします。

以上、率直な意見をお願いします。この設計で問題なければ実装へ進みます。
