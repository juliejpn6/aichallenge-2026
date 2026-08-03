# 初日測定キット: ステアリング・ステップ応答テスト(2026-08-03)

## 背景

大会運営から2026-08-03、AWSIMアップデート告知(今週金曜日適用予定)があり、その中に
「steerのrate limitが実機に合わせて0.8→0.6に変更される」という項目が含まれていた。
Geminiのレビューでも「MPCの内部モデル(現状`steer_rate_max=1.1 raw`)とシミュレータの
実態が乖離すれば、MPCが『曲がれる』と予測して出した指令がAWSIM側で頭打ちになり、
予測モデルが崩壊する(ワインドアップ、蛇行の直接悪化)」との指摘があった。

アップデート適用後、直ちに`steer_rate_max`を実測較正する必要があるが、既存の
`analyze_actuator_delay.py`は走行中の自然な指令変化を受動的に観測する手法であり、
意図的なステップ応答テストの手段が無かった(Geminiからの質問「具体的な計測手順は
確立されているか」に対し、正直に「未確立」と回答し、本キットを準備した)。

## 実装: `scripts/step_response_test.py`

`/control/command/control_cmd`(AckermannControlCommand)へ、既知の振幅・タイミングの
ステップ列を直接publishする独立ノード。制御ロジック(mpc_controller.py等)には一切
変更を加えない。

**ステップパターン**: `0° -> +step -> 0 -> -step -> 0` を1サイクルとし、指定回数
繰り返す(既定: 振幅15°・各フェーズ3秒・5サイクル=正負各5回分のエッジサンプル)。

## 実施手順

1. `make dev`(または`dev3`等)でAWSIM+bag-recorderを起動する
2. **mpc_controllerノードを停止する**(本ツールと二重にpublishすると競合するため)。
   具体的な停止コマンドは環境依存だが、`docker exec <container> bash -c "pkill -f mpc_controller"`
   等が候補(**次回実走行時に確定させる、正直な未検証事項**)
3. コンテナ内で`python3 step_response_test.py --speed 2.0 --step-deg 15.0`を実行
4. 完了後(自動終了する)、bag-recorderを停止してrosbagを回収
5. `analyze_actuator_delay.py --mode steering`(ローカルの実測操舵角あり)または
   `--mode yawrate`(予選と同じ量での検証)で解析する

## 用途

- **金曜日のAWSIMアップデート後**: 直ちに本キットで`steer_rate_max`の実測較正を行う
  (Phase1の再同定タスクの一部)
- **平時**: シミュレータの挙動が変わったか疑わしい場合の健全性チェックとしても再利用できる

## 正直な限界(次回実走行で検証すべき事項)

- mpc_controllerノードの停止方法は理論上の候補のみで、実地検証していない。
  docker-compose構成上、autowareサービス全体を止めずにmpc_controllerプロセスだけを
  安全に止められるか(他の依存ノードへの影響有無)は次回確認が必要。
- ステップ振幅(15°)・周期(3秒)は暫定値。実際のAWSIM応答を見て、飽和領域まで
  振幅を広げる追加テスト(例: 30°、45°)が必要になる可能性がある。
- 本ツール自体はまだ実機(AWSIM)で一度も実行しておらず、publish自体が意図通り
  動くかの動作確認は次回実走行時が初回になる。
