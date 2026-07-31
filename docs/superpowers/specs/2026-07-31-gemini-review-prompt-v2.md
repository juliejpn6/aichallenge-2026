あなたは自動運転レーシングカート(ROS2 Humble / Autoware、spatial MPCで制御)のソフトウェアアーキテクトです。以下は、あなたが以前レビューした設計書の**改訂版**です。前回指摘した内容がどう反映されたかを確認し、再レビューしてください。この設計は、まだ実装前の「設計段階」のドキュメントです。実装コードは書かず、設計の妥当性のみを評価してください。

# 前回レビューで指摘された内容と、それへの対応(サマリ)

1. **`steer_low_pass_gain`の換算値誤記**(0.35→0.231)——検算の結果、正しくは**0.213**(`1-(1-0.35)^(40/72)=0.2128`)と判明し、修正した
2. **処理落ちリスクへの段階的フォールバック(60Hz)と定量判定基準の欠如**——「10分間走行で13.9ms超過が0.1%以下ならPass」という基準と、Fail時に60Hzへ後退する段階的フォールバック手順を追加した
3. **小整数値の丸め誤差への注意不足**——`min_trend_cycles`等の小周期値について、丸め方向(切り捨てか切り上げか)と実時間への影響を明記したチェックリストを追加した
4. **二次的依存経路の見落とし懸念**——以下のPhase 0調査を実施し、結果を「カテゴリD」「副次的な影響」として反映した:
   - **スルーレートリミッタの調査**: `core/MPC.py`に`max_delta_change = self.max_steering_rate * self.model.Ts`という「周期あたり最大操舵変化量」の制約が実在した。しかし`self.model.Ts`は`Ts = 1.0 / control_rate`と既にcontrol_rateから動的に計算されており、`max_delta_change × control_rate = max_steering_rate`(不変)が常に成立するため、**この制約は既にcontrol_rate非依存に設計されている**と判断し、新規のカテゴリ(逆比換算)は追加しなかった
   - `collision_cum_window_cycles`の累積対象が生の実速度値(m/s、レート非依存の物理量)であることを確認し、窓長のカテゴリA換算のみで十分と判断した
   - OSQPソルバーの`max_iter`/`time_limit`が明示設定されておらずライブラリ既定値であることを確認した(72Hz予算との直接的なコンフリクトはないが、予算内強制打ち切りの安全網も元々ない)
   - 入力トピック(EKF・V2X)の実際の更新レートは本リポジトリのソースから確認できず、「未確認」として正直に記録した(GNSS生値のみ既存資料で20Hzと判明)
   - 副次的な影響として、OSQPウォームスタート改善の可能性(正)と、発行頻度上昇に伴うROS2 Subscriber側のQoS/Queue溢れリスク(負)を明記した

# 改訂後の設計書全文

---

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

| パラメータ | 40Hz時 | 72Hz換算値(丸め後) | 実時間換算 | 相対誤差 |
|---|---|---|---|---|
| `min_trend_cycles` | 3 | 5(=3×1.8=5.4→5) | 0.0694s(意図0.075s比 -7.4%) | 短くなる方向(誤発火防止が僅かに緩む) |
| `def_enter_cycles` | 5 | 9(=9.0) | 0.125s(意図0.12s比 +4.2%) | 誤差小 |
| `collision_cum_window_cycles` | 5 | 9(=9.0) | 0.125s(意図と一致) | 誤差なし |

`min_trend_cycles`のように丸めが切り捨て方向(短くなる)に働く場合、意図した誤発火防止時間より僅かに早くTTC評価が始まる。この程度の誤差(-7.4%)は許容範囲と判断するが、実装後に`[RATE-SCALE]`ログで全対象の丸め方向を確認し、安全側に倒れていないものがあれば個別に検討する。

## 6. ロールアウト・検証手順

1. 本設計の実装(ヘルパー新設+23箇所のラップ+`[PERF]`ハードコード修正)を40Hzのまま(`control_rate`は変更せず)完了させ、`[RATE-SCALE]`ログが全て「換算前後で同じ値」(40Hz→40Hzなので変化なし)になることを確認する(①非矛盾性: 既存挙動が一切変わらないことの確認)
2. `control_rate`を72.0へ変更し、`[RATE-SCALE]`ログで想定通りの換算値(表の「72Hz換算後」列と一致)になっていることを確認する
3. 実運用環境(予選環境)で処理落ち率を実測する。現状40Hzでの実測平均処理時間は10.2ms(25ms予算の約40%)。72Hzでは1周期の予算が13.9msとなり、平均処理時間がほぼ変わらないと仮定すると使用率は約72%——スパイク(過去に平常の3〜5倍を記録した事例あり)への耐性は不明であり、この実測が本設計の成否を分ける
4. **判定基準(Geminiレビュー指摘、2026-07-31追加)**: 10分間以上の連続走行ログで、1周期の処理時間が13.9msを超えた回数の割合が**0.1%以下**であればPass、それを超えればFailとする
5. **段階的フォールバック**: 4の判定がFailの場合、まず`control_rate`を**60Hz(予算16.66ms、想定使用率約61%)**へ下げて同じ基準で再測定する。60Hzでも基準を満たさない場合は、コリドー計算(現状最重量、平均4.01ms)の最適化を別課題として着手した上で72Hz(または60Hz)への再挑戦を検討する。`control_rate`を変更するだけで全ての依存値([RATE-SCALE]ログ経由)が自動追従するため、この段階的な後退・再測定は追加の実装変更なしに行える

## 7. 未解決のリスク・正直な限界

- 計算負荷(処理落ち率)は実測するまで確定しない。72Hzが実運用環境で成立するかどうかは、本設計の実装だけでは保証できない
- `steer_low_pass_gain`は蛇行対策として過去に慎重に手動チューニングされた値であり、指数換算式で機械的に算出した値(0.35→**0.213**、検算式: `1-(1-0.35)^(40/72)=0.2128`。旧版の本書に「0.231」との誤記があったが2026-07-31のGeminiレビューで指摘され訂正した)が同等に機能するかは実地での再検証が必要
- 本設計は「反応速度」の向上のみを目的としており、蛇行(AXIS06)への効果は見込んでいない。これを混同して評価しないこと

---

# 再レビューでお願いしたいこと

1. 前回指摘した4点(誤記・フォールバック基準・丸め誤差チェックリスト・二次的依存経路)が、それぞれ適切に反映されているか確認してください。特に「スルーレートリミッタは既にTs=1/control_rateで動的計算されているため対応不要」という判断が本当に正しいか、数式を検算した上で確認してほしいです。
2. 新たに追加した「副次的な影響」セクション(OSQPウォームスタート改善の可能性、QoS/Queue溢れリスク、入力トピックレート未確認の件)について、実装前にさらに調査すべき点があれば指摘してください。
3. この設計書は、あなたの前回レビューを経て**このまま実装フェーズへ進めるレベルに達しているか**、それとも追加の修正が必要か、率直な総合判断をお願いします。

# 出力形式

- 前回指摘した4点それぞれについて「反映済み・妥当」「反映されているが不十分」「未反映」のいずれかを明示してください。
- 新規指摘があれば、具体的な理由とともに列挙してください。
- 最後に、実装フェーズへ進むことへのGo/No-Go判断を一言でまとめてください。
