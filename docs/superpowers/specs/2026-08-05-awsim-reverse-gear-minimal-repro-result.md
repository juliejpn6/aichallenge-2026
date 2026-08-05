# AWSIM自律モードREVERSEギア不受理問題 — 最小再現ノードでの決定的結果

前回のフォローアップ相談(`2026-08-05-awsim-reverse-gear-consultation-followup.md`)で
提案いただいた「最小再現ノード」を実施し、決定的な結果が得られたので報告する。

## 実施した試験

自作の`mpc_controller`(本線コントローラ)を**完全に停止**した状態で、それとは
無関係な最小限の一時的なrclpyノードを新規作成し、以下を10秒間・約20Hzで
継続的に実行した。

```python
# 毎周期(約20Hz)実行
gc = GearCommand()
gc.stamp = now.to_msg()          # publishごとに更新
gc.command = GearCommand.REVERSE  # =20
gear_pub.publish(gc)

cmd = AckermannControlCommand()
cmd.stamp = now.to_msg()
cmd.lateral.steering_tire_angle = 0.0
cmd.longitudinal.speed = -1.0        # 後退方向の目標速度
cmd.longitudinal.acceleration = -1.0  # 後退方向の目標加速度
cmd_pub.publish(cmd)
```

同時に`/vehicle/status/gear_status`と`/localization/kinematic_state`
(実速度)を購読し、1秒ごとに値を記録した。

## 環境の前提条件(前回相談での推奨事項を全て満たした)

- 車両1台のみ(solo dev環境、`make dev`)
- 本線コントローラ(`mpc_controller`)は`kill -TERM`で完全停止済み
  (プロセスリストで消えていることを確認)、他に競合するpublisherは存在しない
- `/control/command/actuation_cmd`は購読者はいるが発行者0(未使用の経路と確認済み)、
  `/control/command/control_cmd`(AckermannControlCommand)が実際にAWSIMが
  subscribeしている経路であることを`ros2 topic info -v`で確認済み
- `GearCommand.stamp`は毎周期(publish直前に)更新している
- 直前に、同じ車両で「手動(Manual)でRへ入れた状態のままAutonomousへ復帰」する
  差分試験も実施し、**mpc_controller停止時はRレンジが維持される**ことを確認済み
  (=Autonomousモードへの切替自体はギアをリセットしない)

## 結果(タイムスタンプ付き実測ログ)

```
t=0.0s gear_report=20 v=None       # REVERSE、速度未受信
t=0.1s gear_report=20 v=None       # REVERSE維持
t=1.0s gear_report=22 v=0.0002     # PARKへ変化、速度はほぼゼロ(ノイズレベル)
t=2.0s gear_report=22 v=0.00005
t=3.0s gear_report=22 v=0.0002
t=4.0s gear_report=22 v=0.0000
...(以降t=9.1sまで一貫してgear_report=22、速度は全て±0.001未満のノイズレベル)
```

## 結論

1. **REVERSE要求は一瞬(0.1秒未満)は受理されている**(t=0.0〜0.1sでgear_report=20)。
2. **しかし1秒以内に自動的にPARK(22)へ強制的に戻され、以降10秒間ずっと戻らない**
   (継続的にREVERSE要求+後退速度/加速度指令を送り続けているにもかかわらず)。
3. **速度は試験全体を通じて実質ゼロ**(ノイズレベル)であり、車両は一切後退していない。
4. 本線コントローラを完全に排除し、Manual/Autonomous切替もこの結果に影響しない
   ことを既に確認済みのため、**この現象は自作コード・状態機械の設計とは無関係で、
   AWSIM自体の自律モード向けROSギア入力経路の内部実装(または既知の仕様、
   あるいはバグ)に起因すると、これまでで最も確度高く特定できた**。

これにより、前回相談時点の優先順位付き仮説リストのうち、以下が確定的に判断できる
ようになった:

- 「運動指令が伴わないためPARKへ戻る」仮説 → **否定**(継続的に運動指令を
  送り続けても結果は同じだった)。
- 「Autonomous初期化・mode watchdogがRを強制解除している」仮説 → **否定**
  (Manual時にRへ入れた状態でAutonomousへ復帰する差分試験で、mpc_controller
  停止時はRが維持されることを確認済み)。
- 「本線コントローラ内の競合・上書き」仮説 → **否定**(本線コントローラを
  完全停止した状態での再現のため)。
- 「AWSIM自律モード用ROSギア入力経路でR遷移そのものが拒否されている」仮説
  → **最有力かつほぼ確定**。

## 相談したいこと

1. この結果を踏まえ、これ以上ROS側の送信パターン(タイミング・待ち時間・
   運動指令の有無・組み合わせ等)を変えても解決しない可能性が高いと考えている。
   この判断は妥当か。
2. 妥当な場合、次の現実的な選択肢はどちらが良いか:
   - (a) 大会運営・AWSIM提供元へ、この最小再現結果(本ドキュメント)を添えて
     問い合わせる
   - (b) この制約を前提として受け入れ、自動化されたREVERSE復帰には頼らず、
     既存の安全網(shuffle_hard_limit、経路非依存でSTUCK復帰を断念しNORMALへ
     委譲する仕組み)で被害を限定する運用へ切り替える
3. もし(a)を選ぶ場合、運営への問い合わせ文面として、本ドキュメントの
   「結果」セクションをそのまま使ってよいか、他に追加すべき情報はあるか
   (AWSIM.zipのバージョン情報等、まだ添付できていないものがある)。

## 参考: 直前の差分試験(Manual→Autonomous、Rレンジ維持を確認)

同一のsolo dev環境で、`mpc_controller`停止後に以下を実施し、Rレンジが
Autonomousへの切替後も維持されることを確認した(本ドキュメントの試験の
前提となる重要な補助的事実):

1. Autonomous状態で通常走行中、AWSIM画面のA→Mボタンで手動へ切替
2. 手動操作で停止・R レンジへ変更(GearReport: DRIVE→REVERSEを確認)
3. アクセルを踏まずにM→Aボタンで自律モードへ復帰
4. **GearReportはREVERSEのまま維持された**(ユーザー実測確認、
   「維持できました!」)
