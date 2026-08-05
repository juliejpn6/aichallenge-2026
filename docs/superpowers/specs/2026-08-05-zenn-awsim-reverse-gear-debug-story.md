---
title: "「バックギアが1000回に1回しか成功しない」謎を追いかけた話 ― AWSIM × ROS2 デバッグ推理譚"
emoji: "🕵️"
type: "tech"
topics: ["ros2", "autoware", "awsim", "debug", "自動運転"]
published: false
---

## はじめに

こんにちは。自律走行のレーシングカートを開発している者です。

今回は、自動運転シミュレータ「AWSIM」上で**バックギアがどうしても入らない**という、地味だけどかなり厄介なバグを追いかけた記録です。

「手動でバックできるプログラムを書いたのに、なぜかシミュレータ上でだけ全然動かない」――こういう、いわゆる「環境依存っぽいのに原因が全く見えない」系のバグ、経験ある方も多いんじゃないでしょうか。同じように「動くはずなのに動かない」で頭を抱えている方に、少しでも「こういう考え方もあるんだな」というヒントになれば嬉しいです。

結論から言うと、最終的にたどり着いた真犯人は、**私たち自身が「良かれと思って」入れた対処そのものでした**。ミステリー小説だと「一番怪しくない人が犯人」というパターンがありますが、まさにそれに近い展開でした。順を追ってお話しします。

## 事件の発端: 動かないバックギア

自律走行では、壁際で立ち往生してしまった時に「一度バックして態勢を立て直す」という復帰動作が必要になります。この復帰処理として、以下のようなシンプルな手順を実装していました。

1. `GearCommand.REVERSE` (バックギアへの切り替え要求) をパブリッシュする
2. `GearReport` (実際の車両側のギア状態) が `REVERSE` になったのを確認する
3. 確認できたら、後退の速度指令を送る

書いた本人としては「まあ普通に動くだろう」というくらいの気持ちで実装したのですが、いざローカル環境で動かしてみると――**ほぼ100%失敗する**のです。何度再試行しても`GearReport`はずっと`PARK`(駐車)のまま。後退なんて夢のまた夢、という状態でした。

一方で不思議なことがありました。AWSIMの画面にある「Autonomous(自律) / Manual(手動)」の切り替えボタンを押して手動操作に切り替え、キーボードでギアをRに入れてみると――**一発で成功する**んです。

シミュレータ自体はバックできる。なのに、プログラムから自動でやろうとすると、なぜか全く受け付けてくれない。この「手動なら一瞬、自動なら絶望的」というギャップこそが、今回の事件の一番の手がかりであり、同時に一番の罠でもありました。

## 手がかり1: そもそも「存在しない値」を送っていた

実は「D(前進)からR(後退)へは直接シフトできず、必ず何か中間のギアを経由する必要がある」ということは、以前の調査で既にわかっていました。そこで復帰処理は、

D → (中間ギア) → R

という手順にしていたのですが、この中間ギアに、なんとなく自動車の一般常識に従って**PARK(駐車)**を選んでいました。「Dから直接Rに入らないなら、一旦Pを経由するのが自然だろう」というくらいの、ごく自然な発想です。

ところが後になって、Autowareの公式インターフェース仕様をあらためて確認したところ、`GearCommand`の`command`フィールドとして明記されていたのは、

```
1  = NEUTRAL (ニュートラル)
2  = DRIVE   (前進)
20 = REVERSE (後退)
```

の3つだけでした。**PARK(22)は、そもそも仕様に載っていない値だった**のです。

これは完全に見落としでした。中間ギアをNEUTRALに変更したところ、NEUTRALへの遷移確認は、それまでのほぼ0%から**100%**まで劇的に改善しました。

「解決した!」と思いきや――主役であるREVERSEへの遷移は、相変わらず一度も成功しませんでした。半分だけ前進、という、なんとも歯がゆい状況です。

## 手がかり2: 「動き続けなければ拒否される」という、もっともらしい仮説

行き詰まったので、ここで複数のAIに相談してみました。すると、次のような分析が返ってきました。

> シフトが受理された後も後退の速度指令が続いていないと、AWSIM側が「止まったまま何も操作していない」と判断して、自動的にPARKへ戻してしまっているのではないでしょうか。手動操作は「Rに入れてすぐアクセルを踏む」という一連の動作なので、確認待ちの"間"がないこととも辻褄が合います。

なるほど、確かに筋が通っています。実際の実装を見返すと、REVERSEの確認を待っている間は、速度指令をずっとゼロに保持していました。「確認を待たずに、REVERSE要求と同時に後退の速度指令も送り続ける」方式に変更してみることにしました。

……結果は、**変わりませんでした**。それどころか、5秒間ずっと後退の指令を送り続けても、実際の車速は完全にゼロ(センサーのノイズレベル)のまま、微動だにしませんでした。

もっともらしく聞こえる仮説が、実測データにあっさり裏切られる。デバッグあるあるですが、ここで「じゃあもっと長く待てば?」「送信頻度を上げれば?」と場当たり的なパラメータ調整に走らなかったのが、後々効いてきます。

## 転機: 「容疑者」を一人ずつ独房に入れる

手詰まりになったところで、別のAIからハッとする指摘をもらいました。

> 直前の観測で「t=0の時点でもうREVERSEが受理されているように見える」とのことですが、それってもしかして、その前に手動操作でRに入れていた状態が、たまたま残っていただけではないですか? ROSからの要求が本当に受理された証拠にはならないと思います。

言われてみればその通りでした。自分たちのメインの制御プログラムを都合よく信じすぎていたのです。

そこで、思い切って**メインの制御プログラムを完全に停止**し、それとは無関係などまで最小限のスクリプトだけを使って、初期状態をきちんと確認したうえで、もう一度試してみることにしました。犯人候補を一人ずつ独房に入れて、他の要因が混ざらない状態で尋問する、というイメージです。

### 実際に使った再現用スクリプト

そのまま動かせるように、必要な部分を全部書いておきます。手元の環境(ROS2 Humble + Autoware系メッセージ)で試したい方は、以下をコピーして`reverse_repro.py`のような名前で保存してください。

```python
#!/usr/bin/env python3
"""AWSIM上でのギアシフト挙動を確認するための最小再現スクリプト。
メインの制御ノードを止めた状態で実行することで、他の要因が
混ざらない「クリーンな」条件でギアシフトの挙動だけを観察する。
"""
import time
import sys

import rclpy
from rclpy.node import Node
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import GearCommand, GearReport
from nav_msgs.msg import Odometry

GEAR_LABELS = {
    0: "NONE", 1: "NEUTRAL", 2: "DRIVE", 20: "REVERSE",
    21: "REVERSE_2", 22: "PARK", 23: "LOW", 24: "LOW_2",
}


def make_zero_cmd(node, now):
    """速度・加速度とも完全にゼロの制御コマンドを作る(運動指令なし)。"""
    cmd = AckermannControlCommand()
    cmd.stamp = now.to_msg()
    cmd.lateral.stamp = now.to_msg()
    cmd.lateral.steering_tire_angle = 0.0
    cmd.lateral.steering_tire_rotation_rate = 0.0
    cmd.longitudinal.stamp = now.to_msg()
    cmd.longitudinal.speed = 0.0
    cmd.longitudinal.acceleration = 0.0
    return cmd


def main():
    rclpy.init()
    node = rclpy.create_node("reverse_repro")

    gear_pub = node.create_publisher(GearCommand, "/control/command/gear_cmd", 1)
    cmd_pub = node.create_publisher(AckermannControlCommand, "/control/command/control_cmd", 1)

    state = {"gear": None, "v": None}

    def on_gear(msg):
        state["gear"] = msg.report

    def on_odom(msg):
        state["v"] = msg.twist.twist.linear.x

    node.create_subscription(GearReport, "/vehicle/status/gear_status", on_gear, 10)
    node.create_subscription(Odometry, "/localization/kinematic_state", on_odom, 10)

    def spin_a_bit():
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.02)

    def gear_label():
        return GEAR_LABELS.get(state["gear"], f"raw={state['gear']}")

    # --- 0. 現在のギア状態を確認しておく(重要: 前の状態が残っていないか確認) ---
    spin_a_bit()
    spin_a_bit()
    print(f"[INIT] 現在のgear_report = {state['gear']} ({gear_label()})", flush=True)

    # --- 1. NEUTRALへ、加速度は完全ゼロを維持したまま要求する ---
    t0 = time.time()
    while time.time() - t0 < 3.0:
        now = node.get_clock().now()
        gc = GearCommand()
        gc.stamp = now.to_msg()
        gc.command = GearCommand.NEUTRAL
        gear_pub.publish(gc)
        cmd_pub.publish(make_zero_cmd(node, now))
        spin_a_bit()
        if state["gear"] == GearReport.NEUTRAL:
            break
    print(f"[STEP1] NEUTRAL要求 -> {gear_label()} (経過{time.time()-t0:.2f}s)", flush=True)

    # --- 2. NEUTRALのまま1秒キープ(ここでも加速度はゼロのまま) ---
    t0 = time.time()
    while time.time() - t0 < 1.0:
        now = node.get_clock().now()
        gc = GearCommand()
        gc.stamp = now.to_msg()
        gc.command = GearCommand.NEUTRAL
        gear_pub.publish(gc)
        cmd_pub.publish(make_zero_cmd(node, now))
        spin_a_bit()
    print(f"[STEP2] 1秒キープ後 -> {gear_label()}", flush=True)

    # --- 3. 本命のREVERSEへ。ここでもまだ加速度はゼロのまま、確認できるまで待つ ---
    t0 = time.time()
    next_log = 0.0
    while time.time() - t0 < 5.0:
        now = node.get_clock().now()
        gc = GearCommand()
        gc.stamp = now.to_msg()
        gc.command = GearCommand.REVERSE
        gear_pub.publish(gc)
        cmd_pub.publish(make_zero_cmd(node, now))
        spin_a_bit()
        elapsed = time.time() - t0
        if elapsed >= next_log:
            print(f"[STEP3] t={elapsed:.1f}s gear={gear_label()} v={state['v']}", flush=True)
            next_log += 0.5

    print("[DONE] 試験終了", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

実行する時は、Autoware系のメッセージ型が使えるROS2環境(コンテナの中など)で、以下のように動かします。ROS_DOMAIN_IDはご自身の環境に合わせて読み替えてください。

```bash
# ROSの環境をsourceしてから実行する例
source /opt/ros/humble/setup.bash
source /path/to/your/workspace/install/setup.bash
export ROS_DOMAIN_ID=1   # 実際に使っているドメインIDに合わせる

python3 reverse_repro.py
```

このスクリプトを、**メインの制御ノードを止めた状態**(`ros2 node list`で自分の制御ノードのプロセスを見つけて`kill`するか、そもそも起動しないでおく)で走らせるのがポイントです。他の要因を排除して、純粋にAWSIM側の挙動だけを見るためです。

### 出てきた結果

```
[INIT]  現在のgear_report = 2 (DRIVE)
[STEP1] NEUTRAL要求 -> NEUTRAL (経過0.08s)
[STEP2] 1秒キープ後 -> NEUTRAL
[STEP3] t=0.0s gear=NEUTRAL v=None
[STEP3] t=0.5s gear=REVERSE v=7e-06   ← 確認できた!
[STEP3] t=1.0s gear=REVERSE v=2e-05
[STEP3] t=1.5s gear=REVERSE v=1e-05
[STEP3] t=2.0s gear=REVERSE v=2e-06
   ... (以下t=5.0sまでずっとREVERSEのまま安定)
```

**確認できました。しかも5秒間、一度もPARKに戻りませんでした。**

念のため、この後REVERSEが確定した状態から、負の加速度(後退方向の駆動指令)を送ってみても、ギアはREVERSEのまま維持されました。つまり――

> 「動き続けなければ拒否される」という、あれだけもっともらしく聞こえた仮説は、**完全に逆だった**

ということが、ここでようやくハッキリしたのです。

## 真犯人: 実車の「シフトインターロック」に近い挙動

2つの試験結果を並べてみると、違いはたった一点でした。

|                | ギア遷移の要求 | 運動指令 | 結果 |
|----------------|----------------|----------|------|
| ❌ 失敗したパターン | REVERSE要求と**同時に** | 非ゼロの値を送信 | 1秒以内に強制的にPARKへ |
| ✅ 成功したパターン | REVERSEを単独で要求 | 確認できるまで**完全にゼロ** | 安定して維持される |

つまりAWSIMは、**「ギア遷移がまだ確定していないうちに、速度や加速度の指令が入り込んでいる」ことを、暗黙の拒否条件として扱っていた**ようなのです。実車のオートマ車でよくある「ブレーキペダルを踏んでいないとシフトレバーが動かせない」というインターロックに、かなり近い挙動だと思います。

これに気づいた瞬間、それまでのバラバラだった手がかりが、一本の線でつながりました。

- ローカル環境で一度だけ偶然成功した記録がありました。よく見ると、何度もリトライを繰り返した末、たまたま前のサイクルの遅延応答が「運動指令が入っていないタイミング」にちょうど間に合っただけだったのです。
- 手動操作で一発成功していたのも納得です。シフト操作とアクセル操作はUI上まったく別の入力であり、シフトを入れる瞬間には(たとえ直後にアクセルを踏むとしても)運動指令が競合していなかったからでした。

## 直した内容

スタック復帰の状態機械を、次のようにシンプルに直しました。

- REVERSEを要求してから`GearReport`で確認できるまでは、**速度・加速度とも完全にゼロ**を送り続ける
- 確認が取れて、はじめて後退のための本当の速度・加速度指令を送り始める

コードの変更量としては数行程度なのですが、そこにたどり着くまでに何度も「それっぽいけど的外れな仮説」を試すことになりました。

## 直った結果

3台での自己対戦環境で確かめたところ、修正前は確認成功率がほぼ0%だったのが、修正後は**19回中19回、全部成功**しました。実際に車速がだんだん後退方向に伸びていく様子(-0.07 → -0.29 → -0.51 → -0.74 m/sと、ちゃんと加速していく様子)もログでしっかり確認できました。

同じように「動くはずなのに動かない」で悩んでいる方がいたら、少しでも励みになれば嬉しいです。

## 今回の教訓

最後に、今回の一件から自分なりに持ち帰った教訓を4つ書いておきます。

1. **「もっともらしい仮説」ほど、実測で裏を取ってから採用する。** 「動き続けるべきだ」という仮説は理屈としては筋が通っていましたが、実際には正反対でした。仮説をいきなり本番コードに組み込む前に、影響範囲の小さい場所で単独検証してみる価値は大きいです。
2. **観測結果の「初期条件」を、まず疑ってみる。** 直前の手動操作の残り香を、自分の変更の効果だと勘違いしかけました。時系列の前後関係を丁寧に確認しないと、原因と結果を取り違えてしまいます。
3. **本線のコードを思い切って削ぎ落とした最小再現は、遠回りに見えて実は一番の近道になることがある。** 複雑な処理の中でパラメータをあれこれいじり続けるより、無関係な要因を物理的に取り除いてしまったほうが、結果的に早く核心にたどり着けました。
4. **公式の仕様書は、まず最初に読む。** 中間ギアにPARKを使っていたバグは、仕様書のたった3行を読んでいれば一瞬で気づけたはずでした。「たぶんこうだろう」という思い込みで書いた箇所は、あとから仕様と突き合わせてみる価値があります。

シミュレータでも実機でも、「手動では動くのに、自動化すると動かない」という現象に出会ったときは、両者の入力の性質の違い――特に**タイミング**と**排他性**――を疑ってみると、突破口が見つかるかもしれません。

最後まで読んでいただき、ありがとうございました。同じ沼にハマっている方の助けに少しでもなれば幸いです。
