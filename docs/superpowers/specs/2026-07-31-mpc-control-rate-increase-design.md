# MPC制御周期引き上げ(40Hz→72Hz)設計書

- 日付: 2026-07-31
- 対象パッケージ: `aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros`
- 関連: design_docs/stage15_perf_20260707.html(時系列の全履歴)、課題棚卸しダッシュボード

## 1. 目的・背景

車速を今後36km/h程度まで引き上げていく計画に伴い、40Hzの制御周期では「1周期あたりに車両が進む距離」が速度に比例して増大し、対戦車状況の再評価・ENGAGE/STOPPING/switchback等の判断頻度が相対的に粗くなる。これは主に**反応速度**の課題であり、蛇行(AXIS06、限界サイクル振動)への効果は期待していない——蛇行の根本原因はアクチュエータの物理遅延(tau=190ms)とQ[e_y]チューニングの相互作用であり、40Hzは既にこの190ms系を制御するのに十分な速さ(約7.6周期/tau)であるため、サンプリング周期を上げても物理遅延そのものは変わらない。

### 目標値の算出根拠

現在の運用速度(v_max=20km/h)における「1周期あたりの走行距離」を、目標速度36km/hでも維持する:

```
新周期 = 40Hz × (36km/h ÷ 20km/h) = 72Hz
```

## 2. スコープ

### 対象
- `control_rate`をconfig.yamlで72.0Hzへ変更可能にする(値自体の変更は本設計の実装完了後、実測確認を経て行う)
- `control_rate`に依存する全ての周期数ベース閾値・EMA/ローパスゲインを、周期を変えても実時間の意味を保つよう自動換算する仕組みを導入する
- 診断ログ(`[PERF]`)のハードコードされた25ms閾値を修正する

### 非対象(明示的に対象外)
- AXIS06(蛇行/限界サイクル)の根本対処——別スレッドの課題として継続
- MPCホライズン自体の変更——`core/MPC.py`のホライズンはwaypoint間隔(空間ベース、`delta_s = wp_next - wp`)で離散化されており、`control_rate`とは独立している。本設計では変更しない
- ローカリゼーション(EKF)側のプロセスノイズ等——別サブシステム、本設計のスコープ外

## 3. 現状分析: control_rate依存箇所の分類

`config.yaml`・`mpc_controller.py`・`lateral_ttc_monitor.py`を網羅的に調査し、4カテゴリに分類した。

### カテゴリA: 周期数(整数カウント)閾値 — 線形換算が必要

`新値 = round(旧値 × control_rate / 40.0)`

| パラメータ | 現在値(40Hz基準) | 実時間 | 定義箇所 |
|---|---|---|---|
| `hold_cycles` | 60 | 1.5s | config.yaml:427(STUCK検知) |
| `gear_settle_cycles` | 20 | 0.5s | config.yaml:432 |
| `stall_hold_cycles` | 400 | 10.0s | config.yaml:453 |
| `infeas_thr` | 300 | 7.5s | config.yaml:428 |
| `ghost_block_hold_cycles` | 40 | 1.0s | config.yaml:525 |
| `giveup_cycles` | 40 | 1.0s | config.yaml:548 |
| `engage_debounce` | 8 | 0.2s | config.yaml:549(6箇所以上で共有) |
| `engage_cooldown` | 160 | 4.0s | config.yaml:567 |
| `def_enter_cycles` | 5 | 0.12s | config.yaml:589 |
| `def_exit_cycles` | 15 | 0.38s | config.yaml:590 |
| `unlock_inf_cycles` | 80 | 2.0s | config.yaml:593 |
| `unlock_hold_cycles` | 60 | 1.5s | config.yaml:594 |
| `collision_cum_window_cycles` | 5 | 0.125s | config.yaml:521 |
| `min_trend_cycles` | 3 | 0.075s | config.yaml:397(LAT-TTC) |
| `infeasible_latch` | 40 | ≈1.0s | config.yaml:653 |
| `course_in_count` | 5 | ヒステリシス | config.yaml:693 |
| `osqp_shadow_cycles` | 50 | 起動時のみ | config.yaml:329 |
| `_pf_report_every`(コード内リテラル) | 400 | 10s | mpc_controller.py:4186 |

**例外**: `shuffle_max_cycles`(config.yaml:497)は試行回数のカウントであり、周期時間ではないため換算対象外。

### カテゴリB: 減衰係数型(EMA/ローパスゲイン) — 指数換算が必要

`新値 = 1 - (1 - 旧値) ** (40.0 / control_rate)`(単純な線形換算では時定数がずれる)

| パラメータ | 現在値 | 定義箇所 |
|---|---|---|
| `ema_alpha` | 0.05 | config.yaml:564 |
| `r_delta_swing_ema_beta` | 0.15 | config.yaml:105 |
| `beta`(LAT-TTC v_inst平滑化) | 0.15 | config.yaml:388付近 |
| `accel_low_pass_gain` | 0.35 | config.yaml:129 |
| `steer_low_pass_gain` | 0.35 | config.yaml:138(蛇行対策で慎重にチューニング済み) |

### カテゴリC: ハードコードされた実時間定数 — 直接修正が必要

- `mpc_controller.py:4226` `if work > 0.025:` — 40Hzの周期(25ms)を直書き。`1.0 / control_rate`を参照するよう修正する。

### カテゴリD: 既に実時間ベースで安全(変更不要、確認のみ)

- `backup_timeout_s`/`push_timeout_s`(252節で実時間判定と確認済み)
- `ttc_danger_s`/`ttc_critical_s`
- `dt = (now - self._last_t).nanoseconds / 1e9`(実測経過時間)
- `self._loop % int(self._mpc_cfg.control_rate)`系の間引きロジック(既に動的参照)
- MPCホライズン本体(空間ベース)
- `_lag_step = 0.025`(mpc_controller.py:2425、V2X予測の内部積分刻み。control_rateと無関係な独立パラメータ)
- **操舵スルーレートリミッタ**(`core/MPC.py:648`、`max_delta_change = self.max_steering_rate * self.model.Ts`)——Geminiレビュー(Phase 0調査項目1)で懸念された「周期あたり最大変化量」だが、`self.model.Ts`は`mpc_controller.py:525`で`Ts = 1.0 / self._cfg.mpc.control_rate`と既にcontrol_rateから動的に計算されている。したがって`max_delta_change × control_rate = max_steering_rate`(不変)が常に成立し、**1秒あたりの実効レート上限はcontrol_rateを変えても自動的に一定に保たれる**。新規のカテゴリE(逆比換算)は不要と判断した
- `collision_cum_window_cycles`が対象とする累積量(`_collision_v_window`等、mpc_controller.py:4451)は生の実速度値(m/s)であり、「周期ごとの増分」ではない。この量自体はレート非依存の物理量であり、窓長(周期数)をカテゴリAとして線形換算するだけで正しく実時間の意味を保てる(追加対応は不要)
- OSQPソルバー設定(`core/MPC.py:402,462`の`.setup()`呼び出し)は`max_iter`/`time_limit`を明示的に指定しておらずライブラリ既定値に依存している。72Hzの13.9ms予算と直接コンフリクトする設定は見つからなかったが、逆に言うと「予算内で強制打ち切る」安全網はソルバー側に元々存在しない(72Hz化固有の新規リスクではなく、既存のリスクがそのまま残る)

### 副次的な影響(正・負の両方、実測で確認が必要)

- **正の可能性**: 72Hz化で1周期あたりの状態変化(位置・姿勢・障害物配置)が相対的に小さくなるため、OSQPのウォームスタート初期値の精度が上がり、反復回数・solve時間が短縮される可能性がある(未検証、実測で確認)
- **負の可能性**: `/control/command/control_cmd`等の発行頻度が1.8倍になることで、下流(ログ収集・可視化・他ノードのSubscriber)側のQoS/Queueサイズが不足し、メッセージ溢れ・可視化の取りこぼしが起きる可能性がある(本パッケージ内のsubscriptionは全てdepth=1のKEEP_LASTで確認済みだが、下流の他ノード側は未調査)
- 入力トピック(EKF自己位置・V2X対戦車認識)の実際の更新レートは本リポジトリのソースからは確認できなかった(GNSS生値は既存ドキュメントで20Hzと判明しているが、EKF出力・V2X配信レートはAutoware本体側の設定に依存すると見られる)。72Hzで再計算しても、これらの入力がより低頻度でしか更新されない場合、一部の周期は同一のセンサ値を再利用するだけになり、期待するほどの反応速度向上が得られない可能性がある。実測で確認する

## 4. 採用する実装方針: 明示的ヘルパー + デバッグログ

config.yamlの数値・コメントは一切変更しない(「40Hz基準で書かれた値」という意味を保つ)。読み込み側で以下2つのヘルパーを新設し、カテゴリA・Bの各読み込み箇所で明示的にラップする。

```python
_RATE_SCALE_REFERENCE_HZ = 40.0  # config.yamlの数値は全てこの基準周期で書かれている

def _rate_scaled_cycles(self, name: str, cycles_at_ref: int) -> int:
    """周期数閾値をcontrol_rateに合わせて線形換算する(カテゴリA用)。
    config.yamlの値は_RATE_SCALE_REFERENCE_HZ(40Hz)基準で書かれているため、
    実際のcontrol_rateとの比で換算し直す。起動時に換算前後の値をログする。
    max(1, ...)は、将来control_rateを大きく下げた場合に閾値が0(=即発火)へ
    潰れる事故を防ぐガード(Geminiレビュー指摘、2026-07-31反映)。"""
    scaled = max(1, round(
        cycles_at_ref * self._mpc_cfg.control_rate / self._RATE_SCALE_REFERENCE_HZ))
    self.get_logger().info(
        f"[RATE-SCALE] {name}: {cycles_at_ref}周期@{self._RATE_SCALE_REFERENCE_HZ:.0f}Hz "
        f"({cycles_at_ref / self._RATE_SCALE_REFERENCE_HZ:.3f}s) "
        f"-> {scaled}周期@{self._mpc_cfg.control_rate:.0f}Hz")
    return scaled

def _rate_scaled_gain(self, name: str, gain_at_ref: float) -> float:
    """EMA/ローパスゲインをcontrol_rateに合わせて指数換算する(カテゴリB用)。
    離散一次遅れの時定数を保つ式(1-(1-gain)^(ref_rate/rate))を使う
    (単純な線形換算(gain×ref_rate/rate)では時定数がずれるため不可)。"""
    scaled = 1.0 - (1.0 - gain_at_ref) ** (
        self._RATE_SCALE_REFERENCE_HZ / self._mpc_cfg.control_rate)
    self.get_logger().info(
        f"[RATE-SCALE] {name}: gain={gain_at_ref:.4f}@{self._RATE_SCALE_REFERENCE_HZ:.0f}Hz "
        f"-> {scaled:.4f}@{self._mpc_cfg.control_rate:.0f}Hz")
    return scaled
```

呼び出し例:

```python
self._stuck_hold_cycles = self._rate_scaled_cycles("hold_cycles", int(_stkget("hold_cycles", 60)))
self._steer_low_pass_gain = self._rate_scaled_gain("steer_low_pass_gain", float(_mpcget("steer_low_pass_gain", 0.35)))
```

起動時に約23行の`[RATE-SCALE]`ログが出力され、換算前後の値を目視確認できる。`shuffle_max_cycles`等の換算不要な値はヘルパーを呼ばないことで自然に除外される。

### この方針を選んだ理由

- 既存コードの慣用パターン(`_room_to_wall`・`_vid_changed_reset`等、命名規則による自動判定ではなく呼び出し箇所ごとに明示的にヘルパーを呼ぶ流儀)に整合する
- config.yamlのスキーマ変更(キー名の秒単位への一括改名)を避けられ、既存のdesign_docs記述・コメントとの整合を壊さない
- `control_rate`を変更するだけで全ての依存値が自動追従するため、将来の周期変更(例: 72→100Hzへの再挑戦、デバッグ用に40Hzへ戻す)が構造的に安全になる

## 5. テスト方針

- 新規: `_rate_scaled_cycles`/`_rate_scaled_gain`の単体テスト(数式の正しさ、丸め挙動、境界値、ログ出力の存在、`_RATE_SCALE_REFERENCE_HZ`定数の再利用確認)
- 既存テストのうち、カテゴリA・Bの「読み込み後の実値」(`self._stuck_hold_cycles`等)をリテラルで検証しているものは、72Hz環境下での期待値(例: 60→108)に更新する(弱化ではなく追随)
- 回帰スイート全体を実行し、PASSを確認する

### 丸め誤差チェックリスト(Geminiレビュー指摘、2026-07-31追加)

カテゴリAの線形換算(`round(旧値 × 新rate / 旧rate)`)は、値が小さいほど丸め誤差による実時間の相対的なズレが大きくなる。特にデバウンス・ヒステリシス系の小周期値は、安全側に倒れているか(=誤検知を増やす方向のズレでないか)を目視確認する:

| パラメータ | 40Hz時 | 72Hz換算値(丸め後) | 実時間換算(真値) | 相対誤差(真値比較) |
|---|---|---|---|---|
| `min_trend_cycles` | 3 | 5(=3×1.8=5.4→5) | 0.0694s(真値0.075s比 -7.4%) | 短くなる方向(誤発火防止が僅かに緩む) |
| `def_enter_cycles` | 5 | 9(=9.0、丸め無し) | 0.125s(真値0.125s比 誤差なし) | 誤差なし |
| `collision_cum_window_cycles` | 5 | 9(=9.0、丸め無し) | 0.125s(真値と一致) | 誤差なし |

**2026-07-31再訂正(Geminiレビュー指摘)**: `def_enter_cycles`の当初の「+4.2%」という記載は誤りだった。40Hz時の真値(5/40=0.125s)をconfig.yamlのコメント(「≈0.12s」という丸め表示)と比較してしまっていたのが原因で、真値同士(0.125s vs 0.125s)を比較すれば誤差はゼロである(`def_enter_cycles`は5×1.8=9.0と割り切れるため、この項目には本来丸め誤差は一切発生しない)。実際に丸め誤差が生じるのは`min_trend_cycles`(3×1.8=5.4→5)のように、40Hz時の値と1.8の積が整数にならない場合のみである。

`min_trend_cycles`のように丸めが切り捨て方向(短くなる)に働く場合、意図した誤発火防止時間より僅かに早くTTC評価が始まる。この程度の誤差(-7.4%)は許容範囲と判断するが、実装後に`[RATE-SCALE]`ログで全対象の丸め方向を確認し、安全側に倒れていないものがあれば個別に検討する。

## 6. ロールアウト・検証手順

1. 本設計の実装(ヘルパー新設+23箇所のラップ+`[PERF]`ハードコード修正)を40Hzのまま(`control_rate`は変更せず)完了させ、`[RATE-SCALE]`ログが全て「換算前後で同じ値」(40Hz→40Hzなので変化なし)になることを確認する(①非矛盾性: 既存挙動が一切変わらないことの確認)
2. `control_rate`を72.0へ変更し、`[RATE-SCALE]`ログで想定通りの換算値(表の「72Hz換算後」列と一致)になっていることを確認する
3. 実運用環境(予選環境)で処理落ち率を実測する。現状40Hzでの実測平均処理時間は10.2ms(25ms予算の約40%)。72Hzでは1周期の予算が13.9msとなり、平均処理時間がほぼ変わらないと仮定すると使用率は約72%——スパイク(過去に平常の3〜5倍を記録した事例あり)への耐性は不明であり、この実測が本設計の成否を分ける
4. **判定基準(Geminiレビュー指摘、2026-07-31追加)**: 10分間以上の連続走行ログで、1周期の処理時間が13.9msを超えた回数の割合が**0.1%以下**であればPass、それを超えればFailとする。**さらに「最大連続超過数 ≦ 3周期」を併記する**(2026-07-31再追加、Geminiレビュー指摘)——散発的な単発超過と、3周期以上連続する超過は安全上の意味が異なる(後者は実際の反応遅延として体感されうる)ため、超過率が0.1%以下でも連続超過が続く場合は追加調査の対象とする
5. **段階的フォールバック**: 4の判定がFailの場合、まず`control_rate`を**60Hz(予算16.66ms、想定使用率約61%)**へ下げて同じ基準で再測定する。60Hzでも基準を満たさない場合は、コリドー計算(現状最重量、平均4.01ms)の最適化を別課題として着手した上で72Hz(または60Hz)への再挑戦を検討する。`control_rate`を変更するだけで全ての依存値([RATE-SCALE]ログ経由)が自動追従するため、この段階的な後退・再測定は追加の実装変更なしに行える
6. **実測開始前の前提条件確認(2026-07-31追加、Geminiレビュー指摘)**: 手順3の実測を始める前に、`ros2 topic hz /localization/kinematic_state`・`ros2 topic hz /v2x/vehicle_positions`でEKF自己位置・V2X認識トピックの実際の更新レートを確認する(3章「副次的な影響」で触れた通り、これらが72Hzより低頻度の場合、期待する反応速度向上が得られない可能性があるため)。また、rosbag記録系が1.8倍のメッセージ量(`/control/command/control_cmd`等)を取りこぼさずに記録できることも確認する
7. **タイマーのオーバーラン挙動に関する脚注(2026-07-31追加、Phase 0調査結果)**: 制御ループは`create_timer`のコールバック方式ではなく、`run()`内の`while rclpy.ok(): self._control()`ループ+`rclpy.Rate.sleep()`で駆動されている。したがって、上記4の「周期超過」は**コールバックのスキップや欠落を意味しない**——1周期の処理が予算を超えても、その周期の実行自体は必ず最後まで完了し、次の周期がその分だけ後ろ倒しで開始されるだけである(処理落ち=「制御が飛ぶ」ではなく「制御が遅延する」)。判定基準の数値はこの前提の下で解釈すること

8. **dt分布ベースの判定基準を併記する(2026-07-31追加、クローズ作業Phase 3)**: 手順7の脚注の通り、「周期超過」は処理時間(work)だけでなく`rclpy.Rate.sleep()`自体のジッタも含めた「実際に何秒おきに`_control()`が呼ばれたか」(dt、`_control()`呼び出し開始時刻の連続差分)で評価する必要がある。そのための専用計装`[PERF-DT]`(`_dtperf_record`、既存の`_pf_report_every`窓・`self._pf_over_budget_s`をそのまま再利用、新規パラメータなし)を実装した。上記4の処理時間ベース基準に加え、以下をAND条件として満たすことをPassの条件とする:
   - **実効平均レート ≧ 70Hz**(`[PERF-DT]`の`eff_rate`、目標72Hzの約97%。集計窓の周期数÷経過実時間)
   - **dtのp99 ≦ 予算×1.5**(13.9ms×1.5=20.85ms。処理時間p99ではなく実際の呼び出し間隔p99であることに注意)
   - **dt>予算の連続回数 ≦ 3周期**(4の連続超過基準と同じ閾値を、処理時間ではなくdt定義で読み替えたもの。`[PERF-DT]`の`max_consec_over`)
   処理時間ベースの基準(4)とdt分布ベースの基準(本項目)のどちらか一方でも不合格ならFailとする(両方が独立に重要な情報を持つため、いずれかで代替しない)
9. **40Hzベースラインの取得(2026-07-31追加、クローズ作業Phase 3)**: 手順1(40Hz恒等確認)を実施する際、`[PERF-DT]`のログも合わせて1回分採取し、`rclpy.Rate.sleep()`自体が40Hzで実際にどの程度のジッタを持つか(dtのp50/p95/p99/max、実効平均レート)をベースラインとして記録しておくこと。72Hz時の評価(手順8)は、この40Hzベースラインとの相対的な悪化度合いも合わせて確認する(40Hz自体が既に想定より大きいジッタを持っていた場合、72Hz時の悪化を過大評価しないため)
10. **rclpy.Rate精度の確認(2026-07-31追加、クローズ作業Phase 3)**: 13.9ms(72Hz)という短い周期での`rclpy.Rate.sleep()`のジッタは、40Hz(25ms周期)の実績からは外挿できない(OSスケジューラの分解能・スリープ精度の非線形性による)。したがって、72Hz切替直後の**最初の確認項目**として、`[PERF-DT]`のdt分布(特にp99・max)を確認し、処理時間(work)がほぼゼロに近い早い段階のログでも顕著なジッタが見られる場合は、負荷とは独立した`rclpy.Rate`自体の精度限界の可能性を疑うこと

## 7. 未解決のリスク・正直な限界

- 計算負荷(処理落ち率)は実測するまで確定しない。72Hzが実運用環境で成立するかどうかは、本設計の実装だけでは保証できない
- `steer_low_pass_gain`は蛇行対策として過去に慎重に手動チューニングされた値であり、指数換算式で機械的に算出した値(0.35→**0.213**、検算式: `1-(1-0.35)^(40/72)=0.2128`。旧版の本書に「0.231」との誤記があったが2026-07-31のGeminiレビューで指摘され訂正した)が同等に機能するかは実地での再検証が必要
- 本設計は「反応速度」の向上のみを目的としており、蛇行(AXIS06)への効果は見込んでいない。これを混同して評価しないこと
- **Phase 4(40Hz恒等性の実機ログ確認)は本セッションのサンドボックス環境では実施できなかった**: このリポジトリのROS2ワークスペース(`install/`)のシンボリックリンクが別環境(コンテナ)の絶対パスを指しており、`get_package_share_directory`がパッケージを解決できない(未ビルド相当)。数式の恒等性(40Hz時に入力=出力)は単体テストで証明済み、また回帰スイート全体(既存1213件、新規20件、計1233件)がPASSし挙動に変化がないことも確認済みだが、実際にノードを起動して`[RATE-SCALE]`ログの実出力を目視確認する作業は、正しくビルドされた開発/実運用環境で改めて実施する必要がある(ロールアウト手順1に相当)

### 実装時に判明した追加事項(design書レビュー時点では未確定だった詳細)

- **`osqp_shadow_cycles`のみ、ヘルパーが`self._mpc_cfg.control_rate`ではなく`self._cfg.mpc.control_rate`を参照する**: `osqp_shadow_cycles`は`create_mpc()`内、すなわち`self._mpc_cfg`へ代入される**前**に読まれるため、当初の設計通り`self._mpc_cfg.control_rate`を参照すると初期化順序でAttributeErrorになることが実装時に判明した。両ヘルパー(`_rate_scaled_cycles`/`_rate_scaled_gain`)は、`__init__`時点から常に利用可能な生の設定値`self._cfg.mpc.control_rate`(`self._mpc_cfg.control_rate`と同一値)を参照するよう実装した。この変更は全23箇所に一貫して適用されており、挙動への影響はない
- **`space_ema_alpha`(LateralTTCMonitorのコンストラクタ引数)はヘルパーでラップしない**: config.yamlに明示的な`space_ema_alpha`キーは存在せず、コード上のデフォルト値は`self._ot_ema_alpha`(既にヘルパーで換算済み)への参照になっている。ここをさらにヘルパーでラップすると、通常運用時(config.yamlに明示設定がない場合)に二重換算になってしまうため、意図的にラップ対象から除外した
- **`accel_low_pass_gain`/`steer_low_pass_gain`のROS2動的パラメータコールバック(`ros2 param set`によるライブ変更)はヘルパーでラップしない**: これらは起動後に運用者が値を直接変更できる別経路であり、変更時点で運用者が意図する値は「現在アクティブなcontrol_rateでの値」であって「40Hz基準の値」ではないと考えられるため、config.yaml読み込み時のみを対象とする本設計のスコープ外とした
- **動的パラメータ経路の意味論(2026-07-31追加、クローズ作業Phase 1)**: 上記除外により、`accel_low_pass_gain`/`steer_low_pass_gain`には「config.yaml経由(40Hz基準・`control_rate`に応じて自動換算)」と「`ros2 param set`経由(現在の`control_rate`における生値、換算なし)」という**意味の異なる2つの設定経路**が存在することになる。`control_rate`が40Hzのままの間はこの差が顕在化しないが、72Hzへ切替後に運用者がこの違いに気づかず`ros2 param set`で値を調整すると、意図と異なる実効時定数になりうる。この混同を防ぐため、両パラメータの動的コールバック内に`control_rate != 40Hz`の場合のみ発火する1回警告(`_warn_if_dynamic_gain_param_unscaled`)を追加した(換算はしない、生値適用のまま、ログ追加のみで挙動は変えない)
