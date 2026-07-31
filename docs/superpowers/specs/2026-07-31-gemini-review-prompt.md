あなたは自動運転レーシングカート(ROS2 Humble / Autoware、spatial MPCで制御)のソフトウェアアーキテクトです。以下の設計書をレビューしてください。この設計は、まだ実装前の「設計段階」のドキュメントです。実装コードは書かず、設計の妥当性のみを評価してください。

# 前提知識(このプロジェクト固有の文脈)

- 制御アルゴリズムは「空間バイシクルモデル」ベースのMPC(OSQPソルバー)。予測ホライズンは弧長(waypoint間隔、resolution[m/wp])で離散化されており、時間(dt=1/control_rate)では離散化されていない。この点は設計書内でも独立したものとして扱われている。
- `control_rate`(現行40Hz)は、MPCの内部モデルそのものではなく、①制御ループの再計算頻度(対戦車判断・状態遷移の再評価頻度)、②アクチュエータへの指令送信間隔、③多数の「周期数」ベースの閾値(例: STUCK検知までの継続周期数)の実時間換算基準、の3つに効いている。
- 車速を将来36km/h程度まで引き上げる計画があり、その一環として「1周期あたりに車両が進む距離を一定に保つ」という考え方で、現在の運用速度(v_max=20km/h)を基準に40Hz×(36/20)=72Hzという目標値を算出した。
- 蛇行(限界サイクル振動、周期0.6〜0.7Hz)は既に別途調査済みで、根本原因はアクチュエータの物理遅延(一次遅れ、tau=190ms)とコスト関数の横位置重み(Q[e_y])の相互作用と特定されている。本設計(制御周期の引き上げ)はこの蛇行問題の解決を目的としておらず、狙いは「対戦車への反応速度」の向上のみである。

# レビューしてほしい設計書

以下、設計書の全文です(Markdown形式)。

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

## 4. 採用する実装方針: 明示的ヘルパー + デバッグログ

config.yamlの数値・コメントは一切変更しない(「40Hz基準で書かれた値」という意味を保つ)。読み込み側で以下2つのヘルパーを新設し、カテゴリA・Bの各読み込み箇所で明示的にラップする。

```python
_RATE_SCALE_REFERENCE_HZ = 40.0  # config.yamlの数値は全てこの基準周期で書かれている

def _rate_scaled_cycles(self, name: str, cycles_at_ref: int) -> int:
    """周期数閾値をcontrol_rateに合わせて線形換算する(カテゴリA用)。
    config.yamlの値は_RATE_SCALE_REFERENCE_HZ(40Hz)基準で書かれているため、
    実際のcontrol_rateとの比で換算し直す。起動時に換算前後の値をログする。"""
    scaled = round(cycles_at_ref * self._mpc_cfg.control_rate / self._RATE_SCALE_REFERENCE_HZ)
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

## 6. ロールアウト・検証手順

1. 本設計の実装(ヘルパー新設+23箇所のラップ+`[PERF]`ハードコード修正)を40Hzのまま(`control_rate`は変更せず)完了させ、`[RATE-SCALE]`ログが全て「換算前後で同じ値」(40Hz→40Hzなので変化なし)になることを確認する(①非矛盾性: 既存挙動が一切変わらないことの確認)
2. `control_rate`を72.0へ変更し、`[RATE-SCALE]`ログで想定通りの換算値(表の「72Hz換算後」列と一致)になっていることを確認する
3. 実運用環境(予選環境)で処理落ち率を実測する。現状40Hzでの実測平均処理時間は10.2ms(25ms予算の約40%)。72Hzでは1周期の予算が13.9msとなり、平均処理時間がほぼ変わらないと仮定すると使用率は約72%——スパイク(過去に平常の3〜5倍を記録した事例あり)への耐性は不明であり、この実測が本設計の成否を分ける
4. 処理落ち率が許容範囲を超える場合は、`control_rate`を60Hz等へ下げて再測定するか、コリドー計算(現状最重量、平均4.01ms)の最適化を別課題として着手する

## 7. 未解決のリスク・正直な限界

- 計算負荷(処理落ち率)は実測するまで確定しない。72Hzが実運用環境で成立するかどうかは、本設計の実装だけでは保証できない
- `steer_low_pass_gain`は蛇行対策として過去に慎重に手動チューニングされた値であり、指数換算式で機械的に算出した値(0.35→0.231)が同等に機能するかは実地での再検証が必要
- 本設計は「反応速度」の向上のみを目的としており、蛇行(AXIS06)への効果は見込んでいない。これを混同して評価しないこと

---

# レビュー観点(特に厳しく見てほしい点)

1. **物理・制御理論の妥当性**: 「40Hzは190ms(≈5.3Hzの一次遅れ)のアクチュエータを制御するのに既に十分速く、サンプリング周期を上げても蛇行(限界サイクル)は改善しない」という主張は正しいか。反証となる制御理論上の観点(例: サンプリング遅延自体がむだ時間として効く、ゼロ次ホールドの影響、離散化誤差等)が見落とされていないか。

2. **EMA/ローパスゲインの換算式の正しさ**: `新gain = 1 - (1 - 旧gain) ** (旧rate / 新rate)` という式は、離散一次遅れフィルタの時定数を周期非依存に保つための正しい式か。丸め誤差や、gainが大きい(0.35等)場合の近似の妥当性についてもコメントしてほしい。

3. **周期数閾値の線形換算の妥当性**: `新値 = round(旧値 × 新rate / 旧rate)` という単純な比例配分で、STUCK検知やヒステリシス系のタイミング閾値を換算することに問題はないか(特に整数への丸めによる、小さい値(例: 3周期)での相対誤差の大きさ)。

4. **見落とされている可能性のある依存箇所**: 設計書の「カテゴリA/B/C/D」分類は網羅的か。spatial MPCのホライズン自体は空間ベースで周期非依存という主張は本当に正しいか(設計書はコード調査に基づきそう主張しているが、二次的な依存経路——例えばOSQPのウォームスタート挙動や、ソルバーのタイムアウト設定等——が見落とされていないか)。

5. **計算負荷(処理落ち率)のリスク評価**: 実運用環境での実測平均処理時間(10.2ms@40Hz、25ms予算の40%)から、72Hzでの予算(13.9ms)に対する使用率(約72%)という見積もりの妥当性。この見積もり自体に見落としがないか(例: 周期を上げることで処理内容自体が変化する可能性——OSQPのwarm-start効果、キャッシュ効果等)。

6. **設計方針(明示的ヘルパー vs 他の方式)の妥当性**: config.yamlのスキーマを変えず、読み込み時に明示的なヘルパー関数でラップするという方針は、保守性・安全性の観点で妥当か。より良い代替案があれば提示してほしい。

7. **ロールアウト手順の十分性**: 6章の検証手順(40Hzのままヘルパー導入→ログ確認→72Hzへ変更→処理落ち率実測)は、リスクを段階的に検出する順序として十分か。抜けている安全策があれば指摘してほしい。

# 出力形式

- 各レビュー観点について、「問題なし」「軽微な懸念」「重大な懸念」のいずれかを明示した上で、具体的な理由と(あれば)改善提案を述べてください。
- 設計書全体としてこのまま実装フェーズへ進めてよいか、それとも設計を修正してから進めるべきかの総合判断を末尾にまとめてください。
