# AWSIM自律モードREVERSEギア不受理問題 — 最小再現ノードでの結果(訂正版)

前回のフォローアップ相談(`2026-08-05-awsim-reverse-gear-consultation-followup.md`)で
提案いただいた「最小再現ノード」を実施した。**2026-08-05追記(第3回相談での
指摘を受けた訂正)**: 当初「決定的」と記載したが、以下2点の解釈上の誤りが
指摘されたため訂正する。(1)本試験は直前の「Manual→Autonomous R維持」差分試験の
直後に同一車両で実施したため、t=0時点で既にManual操作由来のREVERSE状態が
残っていた可能性があり、「ROS要求が受理された」ことの証拠にはならない。
(2)AWSIM公式仕様では`longitudinal.speed`は未使用で、実際に効いていたのは
`longitudinal.acceleration=-1.0`のみであり、これが「後退駆動」と「ブレーキ」の
どちらとして解釈されたかは本試験だけでは分離できていない。以下の記述はこれらの
限界を踏まえて読むこと。

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

## 結論(訂正版、確実に言えることのみ)

1. Manual操作で設定し、Autonomous移行後もmpc_controller停止中は維持されていた
   REVERSE状態に対し、最小ノードから`GearCommand.REVERSE`と
   `AckermannControlCommand`(acceleration=-1.0)の送信を開始した後、**1秒以内に
   GearReportがPARK(22)へ変化した**。
2. その後、REVERSEを約20Hzで10秒間継続送信しても、**REVERSEへ復帰しなかった**。
3. **速度は試験全体を通じて実質ゼロ**(ノイズレベル)であり、車両は一切後退していない
   (ただし`longitudinal.speed`はAWSIM未使用のため、この試験の`acceleration=-1.0`が
   実際に「後退駆動」として解釈されていた保証はない)。
4. 本線コントローラの状態機械やpublisher競合は、この現象の再現に必要ない。
5. 少なくとも「REVERSE=20を単純に繰り返し送ればよい」というインターフェースには
   なっていない。

**訂正**: 「REVERSE要求がROS経由で一瞬受理された」「AWSIM側の入力経路でR遷移
そのものが拒否されている」と断定するのは時期尚早である。t=0の状態がManual操作の
残存かROS要求の受理結果かを分離できていないこと、および`GearCommand.REVERSE`
単体・ゼロ制御単体・負加速度単体をそれぞれ分離した試験がまだ無いことが理由。
これらを分離する追加試験(4種類)を第3回相談で提案されており、次回実施する。

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
