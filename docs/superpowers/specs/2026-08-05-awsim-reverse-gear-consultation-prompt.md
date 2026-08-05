# AWSIM自律モードでのREVERSEギアシフト不受理問題 — 外部相談用プロンプト

## 背景

ROS2 Humble + Autoware ベースの自律走行レーシングカート(AWSIM シミュレータ)で、
STUCK(スタック)復帰時に後退(REVERSE)ギアへシフトする自動化処理が、
自律(Autonomous)モードでは一貫して失敗する。手動(Manual、AWSIM画面のA/Mトグル
ボタン経由でキーボード/ジョイスティック操作)では即座に成功する。

## 環境

- ROS2 Humble、`autoware_auto_vehicle_msgs`(GearCommand/GearReport)
- AWSIM(Unity製、クローズドソースバイナリ、`.x86_64`実行形式で配布)
- 車両制御ノードは自作Python(rclpy)、STUCK復帰の状態機械を実装
- Docker Compose環境(自律走行コンテナ + AWSIMシミュレータコンテナ)
- 公式仕様: `/control/command/gear_cmd`のcommandフィールドは
  `1=NEUTRAL、2=DRIVE、20=REVERSE`の3値のみが仕様に明記されている(PARKは非対応)

## これまでに確認した事実

1. **DDS配線は正しい**: `ros2 topic info -v`で確認済み。
   - `/control/command/gear_cmd`(型`autoware_auto_vehicle_msgs/msg/GearCommand`、
     QoS: RELIABLE/KEEP_LAST(1)/VOLATILE)は自作ノード(`mpc_controller`)が
     publish、AWSIM側ノード(`awsim_d1`)が正しくsubscribeしている(GID一致確認済み)。
   - `/vehicle/status/gear_status`(型`GearReport`、同QoS)はAWSIM側(`awsim_d1`)が
     publish、自作ノードが正しくsubscribeしている。
2. **enum数値は正しい**: コンテナ内で実測確認。
   `GearCommand.NEUTRAL=1`、`GearCommand.DRIVE=2`、`GearCommand.REVERSE=20`、
   `GearCommand.PARK=22`(ただし仕様外)。`GearReport`側も同一数値体系。
3. **中間ギアの誤りを発見・修正済み**: 「D→Rへ直接シフト不可、中間ギアを経由する
   必要がある」という既知知見(実車/AWSIM共通仕様と思われる)に対し、当初は
   一般的な自動車知識から中間ギアに`PARK`(22、仕様外の値)を選んでいた。
   これを仕様通りの`NEUTRAL`(1)へ修正した結果、**NEUTRAL確認成功率は
   0%近く→100%(dev3実測24/24)まで劇的に改善**した。
4. **しかしREVERSE自体は依然として一貫して失敗する**(dev3実測、NEUTRAL修正後も
   confirmed 0件)。GearCommand.REVERSE(20)を送り続けても、GearReportは
   `PARK`(22)を返し続ける(NEUTRALやREVERSEには遷移しない)。
5. **待ち時間の長さは原因ではない**: 確認タイムアウトを段階的に
   0.5秒→1.25秒→5.0秒(200周期@40Hz)まで拡大したが、5秒待っても改善しなかった
   (正確に5.00秒後にタイムアウトすることをログで確認済み、待機ロジック自体は
   正しく機能している)。
6. **毎周期の連続再パブリッシュが原因という仮説も棄却**: 元々は40Hzで
   GearCommandを継続的に再送していた(状態が変わらない限り同一メッセージを
   送り続ける設計)。これがAWSIM側のシフト処理と干渉している可能性を疑い、
   エッジトリガー化(状態遷移時のみ即座に送信、以後は0.2秒間隔のハートビート
   再送のみ)へ変更したが、これも改善に寄与しなかった。
7. **AWSIM実車体側は後退可能なことを確認済み**: AWSIM画面のA(自律)/M(手動)
   トグルボタンをクリックして手動モードへ切り替え、キーボード/ジョイスティックで
   直接ギア操作すると、**一発でRレンジへの切り替えが成功する**。これはAWSIMの
   車両物理モデル自体がREVERSE動作をサポートしていることの直接証拠である。
8. **稀な自律モードでの成功例が1件観測された**(dev3、複数回のログ調査で
   confirmed=2件を記録した唯一のログファイル): 同一STUCKエピソード内で
   WAIT_PARK→WAIT_REVERSE→BACKUP失敗を8回繰り返した末、8回目の
   WAIT_PARK突入時点で「前回(7回目)の遅延応答がちょうど間に合っていた」ことで
   偶然confirmedになった。つまり自律モードでのREVERSE成功は完全に不可能ではなく、
   極めて低確率で(かつ原因不明のタイミングで)発生することがある。

## 未確認・未検証の項目

- AWSIM.zip自体は2026-07-26版から更新していない(TIER4提供のSharePoint共有
  リンクからの再ダウンロードは、非対話環境での自動化に懸念がありまだ試していない)。
- `/awsim/control_mode_request_topic`(`std_msgs/Bool`型、A/Mトグルボタンが
  publishする)をコードから能動的に発火させる実験は、現在値を保持しない
  (echoで数秒待っても何も受信できない)一発トリガー的トピックであることが
  判明したため、意図せず自律モードを解除したまま戻せなくなるリスクを懸念し
  未実施。
- 実測した「STUCK detected」時点の実速度は0.00〜0.09m/s程度(閾値0.3m/s未満で
  検知)。REVERSE要求時点でこれが十分にゼロに近いかは確認しているが、AWSIM側が
  内部的にどの程度の精度・継続時間で「静止」を要求しているかは不明。
- 実車の自動変速機のような「ブレーキペダル信号」「パーキングブレーキ」に相当する
  追加のインターロック信号がAWSIM側に存在し、自律モード(ROS経由)では
  それが送られていない可能性は未検証。

## 相談したいこと

1. AWSIM(Unity製クローズドソース、`autoware_auto_vehicle_msgs`のGearCommand/
   GearReportインターフェースを持つ)の自律モードにおけるギアシフト(特に
   REVERSE)処理の実装について、既知の制約・ドキュメント・GitHub Issue等の
   心当たりはあるか。
2. 上記の事実(6項目)から推測できる、REVERSEシフト固有の未知の成立条件は
   何が考えられるか(例: より厳密な静止判定、別トピック/サービス経由の追加信号、
   特定のcontrol_mode状態、等)。
3. `/awsim/control_mode_request_topic`を安全に(自律状態を失わずに)実験する
   方法はあるか。
4. AWSIM.zipの再取得(TIER4提供のSharePoint共有リンク)を非対話環境で
   安全に自動化する現実的な方法はあるか。
5. その他、この種の「手動では成功するが自動化パイプラインでは一貫して失敗する」
   パターンに対する一般的なデバッグ手法で、まだ試していないものはあるか。

## 参考: 関連コード(抜粋)

- STUCK復帰の状態機械: `_handle_stuck_recovery()`
  (`multi_purpose_mpc_ros/mpc_controller.py`)
- エッジトリガー化ヘルパー: `_publish_gear_cmd_throttled()`(同ファイル)
- 設定値: `config/config.yaml`の`stuck_recovery.gear_settle_cycles`(200)、
  `stuck_recovery.gear_cmd_heartbeat_cycles`(8)
