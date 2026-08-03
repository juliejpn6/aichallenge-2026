# 遅延較正プロトコル整備(2026-08-03、Phase C-0-2)

## 目的

予選環境の実効遅延(指示→アクチュエータ動作)は変動し得るため、予選走行の都度実測し、
ローカル試験の`debug_extra_actuator_delay_s`投入値をそれに合わせる常設手順を整備する。

## 実装: `analyze_actuator_delay.py`にyawrateモードを追加

予選rosbagには実測操舵角(`/vehicle/status/steering_status`)が含まれない制約(前回報告済み)
があるため、`/localization/kinematic_state`(nav_msgs/msg/Odometry、標準型)の
`twist.twist.angular.z`(ヨーレート)を実測応答として使う**yawrateモード**を追加した。
これは操舵角そのものの遅れ(tau)とは異なる、**操舵指令→ヨーレート応答という車両
ダイナミクス込みの「ループ全体の実効遅延」(L_eff)**であることに注意。

## 較正結果(0803-03/04 予選 vs dev3 ローカル)

| 環境 | FOPDT L[ms] | FOPDT tau_eff[ms] | edge中央値[ms] |
|---|---|---|---|
| 予選0803-03 | 240 | 50 | 245 |
| 予選0803-04 | 240 | 50 | 149 |
| dev3_1本目(delay=0) | 180 | 50 | 48 |
| dev3_2本目(delay=0) | 180 | 50 | 78 |

**FOPDT Lベースの較正**: delta = L_eff_予選(240ms) - L_eff_ローカル(180ms) = **60ms**

## 結論: 既存の較正値(0.055)は妥当

**delta=60msは、既存の`debug_extra_actuator_delay_s=0.055`(55ms)とほぼ一致**しており、
196節で言及されていた過去の知見(「予選環境の実効遅延がローカルより+50-60ms大きい」)とも
整合する。今回、新しい測定方法(ヨーレートベース、予選rosbagでも直接測定可能)で
独立に再検証しても、既存の較正値が妥当であることが裏付けられた。

**今回のPart C実験では、既存の`debug_extra_actuator_delay_s=0.055`をそのまま使用する**
(較正値の変更は不要と判断)。

## 正直な限界

- FOPDT tau_effが両ログとも50ms(グリッドサーチ候補の下限)に張り付いている。実際の
  tau_effが50ms未満である可能性があり、候補範囲を広げた再検証の余地がある(今回は
  delta算出には影響しないため見送り)。
- edge法の較正値(予選197ms平均 vs ローカル63ms平均、delta≈134ms)はFOPDT基準の
  deltaと大きく異なる。edge法は有効サンプル数が少なく(2-6件)ノイズに敏感なため、
  FOPDTグリッドサーチの方を採用する。
- ヨーレートベースのL_effは「ループ全体の実効遅延」であり、AXIS06のtau(操舵角自体の
  一次遅れ、190ms)とは異なる量。この較正結果は`debug_extra_actuator_delay_s`(196節の
  パイプライン遅延差の注入用パラメータ)の妥当性確認であり、tau=190ms自体の再検証には
  直接使えない(tauの再検証は`analyze_actuator_delay.py`のsteeringモード、ローカル実験
  でのみ可能)。

## 運用ルール(今後の標準手順)

1. 予選走行のたびに、回収rosbagへ`analyze_actuator_delay.py --mode yawrate`を適用し
   L_eff_予選を記録する(design_docsへの追記として時系列蓄積、変動監視も兼ねる)。
2. ローカルで同条件(delay=0)のL_eff_ローカルを測り、delta = L_eff_予選 - L_eff_ローカル
   を算出する。
3. delta が現行の`debug_extra_actuator_delay_s`(0.055)から大きく乖離した場合のみ、
   較正値の更新を検討する。
