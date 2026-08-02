#!/usr/bin/env python3
"""横方向TTC(TTClat)ベースの3分岐リアクションロジック(2026-07-11設計、ROS非依存コア)。

0711-02ログ3件の実測分析で確認した真因(「側選択は正しいが、ロック後2〜4秒の再評価
空白の間に相手車の幅寄せで空きが閾値到達前に潰れる」)に対処する。

現時点(2026-07-11)ではmpc_controller.py側からシャドウモード(判定結果をログするのみ
で実際の側/速度/状態には反映しない)で呼び出している。実際の挙動への統合はユーザー
承認後に別途行う。
"""
from __future__ import annotations
import dataclasses
from typing import Optional


@dataclasses.dataclass
class TTCDecision:
    side_override: Optional[int] = None      # 分岐A発火時のみ +1/-1
    v_safe_cap: Optional[float] = None        # 分岐C-stage1発火時のみ [m/s]
    v_safe_cap_label: str = ""                # v_safe_cand用のラベル(既存の命名規約に合わせる)
    force_giveup: bool = False                # 分岐C-stage2発火時のみ True
    ttc_lat: Optional[float] = None           # 診断ログ用
    branch: str = "none"                      # 診断ログ用: none/warmup/stable/A/A_dlat/A_lookahead/A_rescue/A_rescue_relaxed/B/B_cleared/C1/C1_deferred/C2/C2_cleared/FOOTPRINT_RISK
    # 2026-07-26追加(186節続報): branch="A_dlat"はfwd_dlat(相手との実測横間隔)
    # 起点の早期switchbackトリガー(壁コリドーが安定でも発火しうる)。可否条件・
    # 発火時の状態リセットはbranch="A"と同一。詳細はupdate()内のコメント参照。
    # 2026-07-11追加: branch=Aが発火しない原因切り分け用の内部状態スナップショット。
    is_side_by_side: bool = False
    has_switched: bool = False
    # 2026-07-15追加: 断念(giveup)直前の「最終救済」反転(下記has_rescued参照)を
    # 消費したかどうかの診断用スナップショット。
    has_rescued: bool = False
    v_corridor_ema: float = 0.0
    shrink_run: int = 0
    # 2026-07-13追加: 呼び出し元のclearedをそのままエコーする(ログ単体でB_cleared/
    # C2_clearedが選ばれた理由を追えるようにするため)。
    cleared: bool = False
    # 2026-07-14追加: margin不足(opp_space<space)により分岐Aへの遷移が抑制された
    # ことを示す。過去ログ検証で「現在側より狭い側への反転は21件中21件(100%)が
    # 直後giveupに終わる」ことが確認できたための対処(下記switchback_space_m参照)。
    switchback_suppressed: bool = False
    # 2026-07-14追加: v_instが物理妥当性クランプ(v_inst_max)に引っかかったかどうか。
    # 呼び出し元でのエッジトリガーログ用(このcycleでクランプが実際に発動したかを
    # 単体で追跡できるようにする)。
    v_inst_clamped: bool = False
    # 2026-07-16追加(79節: 0715-07/08実測で発覚したトークン浪費バグの根治):
    # 反転先が直近のカーブで閉じるためswitchback_suppressedとなった場合に、
    # その理由がmargin不足(opp_space<space)ではなくcurvature(呼び出し元の
    # new_side_blocked=True)であったことを示す。呼び出し元のログで
    # switchback_suppressedのreasonを"margin"/"k_corner"に区別するために使う。
    switchback_curvature_blocked: bool = False
    # 2026-07-22追加(157節、0722-03実測でk_cornerが実時間的に有利な反転を過剰に
    # 抑制していたことを確認): 静的曲率が懸念を示しても、反転先の実測opp_spaceが
    # switchback_space_mを満たしていた(=overrideが成立しうる状況だった)ことを
    # 示す。switchback_curvature_blocked(生の曲率判定、無変更)とは独立して常に
    # 記録し、「曲率は懸念ありだったが実測で通した」ケースを次回ログから直接
    # 追跡できるようにする。
    switchback_curvature_overridden: bool = False
    # 2026-07-16追加(84節、ユーザー承認済み設計): cleared中の通常switchbackに、
    # margin(opp_space-space)が既存のswitchback_space_m-giveup_space_m(0.5m)
    # 以上であることを追加要求した結果、marginが薄いことのみを理由に抑制された
    # ことを示す(81→82節で試みた「cleared中は一律禁止」案とは異なり、marginが
    # 薄い場合のみに限定した緩やかな制限。82節案は0716-03実測で重大な回帰を
    # 招いたため83節でrevertし、本フィールドに置き換えた)。
    switchback_cleared_margin_blocked: bool = False
    # 2026-07-19追加(103/106/107節で発見・部分対処された非対称性の解消、120節続報):
    # switchback_curvature_blocked(静的トラック曲率のみ)とは別に、反転先について
    # 対象車両IDごとの学習済み走行ライン(OpponentSpeedMap.lat_mean)ベースの
    # room先読みが物理下限を下回ると予測されたためswitchback_suppressedとなった
    # ことを示す。103/107節でnew_side_room_blockedは既にA_rescue_relaxedの
    # 適格判定には統合済みだったが、通常のswitchback(branch=A/A_lookahead)には
    # 意図的に未適用のままだった(CPU予算配慮の段階的アプローチ)。0719-03実測
    # (lap2 wp189-193、速い対戦車との間合いがfwd_dlat=0.017mまで潰れる直前に
    # branch=Aで側反転しttc=0.0の緊急giveup→壁挟み込み)で、この未対処範囲が
    # 実害に直結することを確認したため、branch=A/A_lookaheadにも適用する。
    switchback_room_blocked: bool = False
    # 2026-07-20追加(125節、A-1): switchback_curvature_blocked(静的トラックkappaのみ)
    # とは別に、反転先を含む先読み区間で、MPC自身が毎周期解いている動的コリドー
    # (壁+占有格子込み、wall_slow・124節と同一データソース)の幅がalong_min_width
    # (カート幅未満の物理下限)を下回る区間があったためswitchback_suppressedと
    # なったことを示す。new_side_room_blocked(対象車両IDごとの学習済み走行ライン
    # ベース、A_rescue_relaxedのみ適用)とは異なり、switchback_curvature_blockedと
    # 同じスコープ(通常のswitchback/A_lookahead、および厳密なA_rescueの両方)に
    # 適用する。
    switchback_wall_blocked: bool = False
    # 2026-07-22追加(159節): new_side_wall_blocked(コリドー全体幅)・space/opp_space
    # (壁基準の空き)のいずれも「反転先方向に実際にオフセット目標が到達できるか」を
    # 見ておらず、判定層と実行層(_corr_bound_ahead)の指標が食い違っていた
    # (0722-03実測、A_rescue成立直後にオフセット目標が実質ゼロへクランプされる
    # 内部矛盾を確認)。判定層に実行層と同一の関数・同一閾値(along_min_width)を
    # 追加し両層を一致させる。switchback_wall_blockedと同じスコープ(通常の
    # switchback/A_lookahead、および厳密なA_rescueの両方)に適用する。
    switchback_offset_blocked: bool = False
    # 2026-07-26追加(191節、AXIS03: switchback/A_rescueの縦方向盲点対処):
    # switchback_curvature_blocked/wall_blocked/room_blocked/offset_blockedは
    # いずれも「反転先(-side)の横方向の空き」だけを見ており、相手との縦距離
    # (fwd_ds)を一切見ていなかった。予選ログ(wp185-198)実測で、A_rescueが
    # opp_space=3.22・space=2.62(横空間の条件は満たす)を根拠に反転したが、
    # その0.7秒後にfwd_ds=0.99mでfootprint_risk giveup、さらに0.24秒後に
    # COLLISION-SUSPECTEDが発生した事例を確認。相手が既にalong_min_length
    # (footprint_risk本体と同一の物理下限)未満まで接近している状態では、
    # オフセットをゼロから作り直す反転を完了する前に接触域へ入ってしまうため、
    # 側変更ではなく減速(既存footprint_risk/wall_slow)に委ねるべきと判断した。
    switchback_ds_blocked: bool = False
    # 2026-07-16追加(84節): 呼び出し元のlookahead_favor_switch(前方カーブにより
    # 現在側が閉じ、反対側は閉じないと判明している状態)をそのままエコーする。
    # 診断ログで「この周期、先回り切り替えの条件自体は成立していたか」を
    # branch(A_lookahead発火有無)と切り離して追跡できるようにするため。
    lookahead_favor_switch: bool = False
    # 2026-07-17追加(92節①、カーブ起因のTTC猶予): C2判定時、現在側の収縮が
    # カーブ起因と分かっている間、何周期連続で猶予中かを示す(診断用)。
    critical_curvature_run: int = 0
    # 2026-07-20追加(127節続報、0720-01予選ログwp173の異常接近分析): 呼び出し元の
    # footprint_risk(fwd_dlat<along_min_widthかつfwd_ds<along_min_length、実際に
    # 車体が重なるリスクがある状態)がTrueだったため、トレンド判定を待たず即座に
    # 強制giveupしたことを示す(診断用)。
    footprint_risk_triggered: bool = False
    # 2026-07-20追加(132節、Gap①Phase0、診断専用・実挙動へは未配線): fwd_dlat
    # (自車〜対象車の実測横間隔)の縮小速度[m/s、負値=縮小中]と、その連続縮小
    # 周期数。既存のspace(壁〜対象車)ベースのv_corridor_ema/shrink_runとは別に、
    # side==0(未エンゲージ)の間も_ot_side/reset_episode()の影響を受けず追跡し
    # 続ける(下記_update_dlat_trend参照)。ENGAGE可否判定にはまだ使わない。
    dlat_v_ema: float = 0.0
    dlat_shrink_run: int = 0
    # 2026-07-21追加(149節続報、③診断): dlat_v_ema/dlat_shrink_runが0のまま
    # footprint_risk発火に至るケースが実測(0720-05/07/08/0721-01の64件全件)で
    # 確認されたため、_update_dlat_trend()内のどのリセット経路が今回発火したか
    # (または正常にトレンドを蓄積できたか)を記録する診断専用フィールド。
    # "none"=正常にトレンド計算済み、"dlat_none"=対象車ロスト、
    # "vid_changed"=対象車切替、"warmup"=切替/ロスト直後で蓄積周期数不足。
    dlat_trend_reset_reason: str = "none"


class LateralTTCMonitor:
    """横方向空間の縮小トレンドを監視し、TTClatに基づく3分岐(スイッチバック/
    並走時は速度介入なし/フォールバック2段)を判定する。1オーバーテイクepisode単位で
    reset_episode()を呼び、update()を毎周期呼ぶ想定。

    設計方針(2026-07-11、0711-02ログ3件の実測分析に基づく、ユーザー承認済み):
    - 側選択(_plan_pass)自体は正しく機能しているため変更しない。
    - 「側ロック後、2〜4秒の再評価空白」という確認済みの真因にのみ対処する。
    - 分岐A(スイッチバック): 未並走・未反転・反対側が十分広ければ即座に側反転(1エンゲージ1回のみ)。
    - 分岐B(並走中): 側は完全ロックし、速度には一切介入しない(アクセル全開を妨げない)。
      並走状態を素早く抜け切ることで、危険な横並びの継続時間そのものを短くする狙い。
    - 分岐C-stage1(未並走・スイッチバック不可・危険): 相手の縦速度基準で一段引き、
      縦間合いを確保する(vopp - caution_speed_margin)。
    - 分岐C-stage2(さらに悪化): 実測の空き幅がgiveup_space_mを割っていなくても、
      TTClatがcritical閾値を下回った時点で強制的にオーバーテイクを断念する。
    - 既存の_ot_side_block_ema(閾値到達で追従へ離脱)を将来的に"置き換える"形で統合する
      前提。並存させると二重判定になるため、実挙動へ統合する際はそちらを置換すること。

    2026-07-13追加(cleared緩和、ユーザー承認済み設計): 旧_ot_side_block_ema方式は
    「真横到達(_ot_cleared)後は閾値を物理下限(along_min_width)へ緩和する」という
    二段階閾値を持っていたが、本モジュール(LAT-TTC)にはこれが無かった。
    is_side_by_side(dlat<=1.6m)は「まだ相手に寄せている最中」の狭い窓であるのに対し、
    _ot_cleared(dlat>=1.8m以上)は「既に十分離れた」広い窓であり、両者は逆方向の条件
    である。そのため真横到達(cleared)した瞬間にis_side_by_sideから外れ、C1/C2の
    「コリドー縮小トレンド」判定に投げ出される、という逆転現象が起きていた。C1/C2は
    相手との実距離を見ないため、コーナー形状によるコリドー自然収束を「相手に迫られて
    いる」と誤認し、fwd_dlatが3m超でもgiveupが発火する事例を実測(0713-04)。
    対処: cleared=Trueの間はC1(縦速度キャップ)を課さず、C2の残存判定(最終防波堤)
    のみ、閾値をgiveup_space_mからcleared_space_m(物理下限、既定=along_min_width)へ
    緩和して残す。旧方式の「無効化はしない、本当に閉じれば必ず離脱する」思想をそのまま
    踏襲する。
    """

    def __init__(self,
                 beta: float = 0.15,                  # 微分値(v_inst)のEMA平滑化係数
                 space_ema_alpha: float = 0.05,        # 2026-07-12追加: 生spaceの事前平滑化係数。
                 # 既定値は既存_ot_side_block_ema(mpc_controller.py)のema_alphaと同一値を踏襲。
                 # 呼び出し元でself._ot_ema_alphaをそのまま渡すこと(新規チューニング値を
                 # 増やさず、実績あるパラメータを再利用する)。
                 ttc_danger_s: float = 2.0,            # 危険判定の閾値
                 ttc_critical_s: float = 0.8,          # 強制giveupの閾値(C-stage2)
                 giveup_space_m: float = 1.85,         # 既存 alongside_lane_need と同一値
                 switchback_space_m: float = 2.35,     # giveup_space_m + 反転マージン0.5m
                 side_by_side_dlat_m: float = 1.6,     # 既存 _clear_lat_reacquire と同一値
                 side_by_side_dlat_release_m: float = 2.1,  # 2026-07-14追加: 既存clear_lat_releaseと同一値
                 # (is_side_by_sideの離脱側ヒステリシス。単一閾値だと境界付近のノイズで
                 #  is_side_by_sideが周期ごとに反転し、branch B(速度介入なし)⇔C1/C2(速度
                 #  キャップ/強制giveup)がチャーンする恐れがあった。既にmpc_controller.py側の
                 #  _ot_cleared(clear_lat_reacquire=1.6で再取得/clear_lat_release=2.1で解放)
                 #  が全く同じ二段階閾値を持っているため、その値をそのまま再利用する。
                 side_by_side_ds_m: float = 1.0,       # 既存 _clear_ds_beside と同一値(真横判定の縦方向条件)
                 caution_speed_margin_kmh: float = 2.0,
                 min_trend_cycles: int = 3,            # 誤発火防止: 連続N周期縮小継続で初めてTTC評価
                 cleared_space_m: float = 1.45,        # 2026-07-13追加: 既存along_min_widthと同一値。
                 # cleared=True時の緩和後閾値(カート幅未満の物理下限)。
                 # 既存策(_vid_changed検知・生spaceの事前平滑化、2026-07-12)は対象車切替に
                 # 起因する外れ値(実測-32.6m/s)には対処済みだが、対象車切替が無い周期でも
                 # v_inst=-22〜-27m/s級の値が発生し、fwd_dlat=3m超でもgiveupする事例を2件
                 # 実測(0713-05 wp16-21、0713-06 wp243-246、いずれもfwd_vid不変)。これらの
                 # 既存策では捕捉できない外れ値であるため、原因(dt異常かspace算出の瞬間的
                 # 外れ値か)を特定せずとも一律に無害化できる、v_inst自体への物理的妥当性
                 # クランプを追加する。既定5.0m/sは実測の正当な危険域(-1〜-5m/s程度)を
                 # 覆いつつ、上記の異常値(-22〜-27m/s)を確実に弾ける水準。
                 v_inst_max: float = 5.0,
                 # 2026-07-20追加(144節続報): C1_deferred(カーブ起因のTTC猶予)中に
                 # 逃げ道(switchback/rescue)が壁/相手位置先読みでブロックされていると
                 # 分かっている場合、通常のvoppベースの緩いキャップの代わりに、既存の
                 # wall_slow_speed(壁際減速・footprint_risk等、他の緊急層と共通の保守
                 # 速度定数)を使う。詳細はupdate()内のC1_deferred分岐を参照。
                 wall_slow_speed: float = 2.0):
        self.beta = beta
        self.space_ema_alpha = space_ema_alpha
        self.ttc_danger_s = ttc_danger_s
        self.ttc_critical_s = ttc_critical_s
        self.giveup_space_m = giveup_space_m
        self.switchback_space_m = switchback_space_m
        self.side_by_side_dlat_m = side_by_side_dlat_m
        self.side_by_side_dlat_release_m = side_by_side_dlat_release_m
        self.side_by_side_ds_m = side_by_side_ds_m
        self.caution_speed_margin_kmh = caution_speed_margin_kmh
        self.min_trend_cycles = min_trend_cycles
        self.cleared_space_m = cleared_space_m
        self.v_inst_max = v_inst_max
        self.wall_slow_speed = wall_slow_speed
        # 2026-07-20追加(132節、Gap①Phase0): fwd_dlat縮小トレンドの追跡状態。
        # reset_episode()の対象に含めない(下記reset_episode()のdocstring参照) —
        # side==0の間も対象車(fwd_vid)が変わらない限りトレンドを保持し続けるための、
        # 意図的な設計上の分離。__init__でのみ初期化し、_update_dlat_trend()が
        # fwd_vid変化またはfwd_dlat=Noneの周期にのみリセットする。
        self._dlat_ema: Optional[float] = None
        self._prev_dlat_ema: Optional[float] = None
        self._v_dlat_ema: float = 0.0
        self._dlat_shrink_run: int = 0
        self._dlat_prev_vid: Optional[str] = None
        # 2026-07-21追加(149節続報、③診断): _update_dlat_trend()が今回どの経路を
        # 通ったかの記録(TTCDecision.dlat_trend_reset_reasonへそのまま渡す)。
        self._dlat_trend_reset_reason: str = "none"
        self.reset_episode()

    def reset_episode(self) -> None:
        """新規エンゲージ(_can_engage成立)のたびに呼ぶ。has_switchedもここでリセット
        する=「1エンゲージにつき1回」の反転制限。呼び出し元は side==0 の間も毎周期
        本メソッドを呼んでいる(=エンゲージ前の「対象車と接触せずにいる区間」全体で
        トレンドを持たない設計)。

        2026-07-20追加(132節、Gap①Phase0): 上記の「side==0の間毎周期呼ばれる」
        性質により、_prev_space/_space_ema等(壁ベースのspaceトレンド)は元々
        エンゲージ前は意味を持たない(「現在側」自体が存在しないため)値として
        妥当だが、副作用としてfwd_dlat(自車〜対象車の実測間隔)の縮小トレンドも
        エンゲージのたびに失われていた。実測(0720-02予選ログwp284)で、
        giveup直後に同一対象車(vid=d3)を4.2秒後に再エンゲージした際、その間の
        間合いの推移を一切知らないまま判定していたことを確認した。本メソッドは
        意図的に_dlat_ema等(下記_update_dlat_trend参照)をリセットしない
        (fwd_vid変化時のみ_update_dlat_trend側でリセットする、独立したライフ
        サイクル)。"""
        self._prev_space: Optional[float] = None
        self._space_ema: Optional[float] = None  # 2026-07-12追加: 平滑化済みspace
        self._v_corridor_ema: float = 0.0
        self._shrink_run: int = 0
        self.has_switched: bool = False
        # 2026-07-15追加: 「断念直前の最終救済」反転を1エピソードにつき1回だけ許可する
        # ラッチ。has_switched(通常のswitchback、危険域に入った時点で反対側が明確に
        # 広ければ即反転)とは独立にカウントする。0715-03実測(t=16.90に通常switchback
        # 発火→ has_switched=True消費済み → t=21.5〜23.3に選択側が縮み反対側が拡大する
        # 明確な逆転トレンドが生じたが、has_switched消費済みのため2回目の反転ができず、
        # t=23.42にgiveupしていた)で確認した抜け穴への対処。「もう失うものがない」
        # giveup直前の局面に限定するため、通常のmid-pass反転回数制限(過去61件の
        # 実測検証で確立済み)を緩めることにはならない。
        self.has_rescued: bool = False
        self.is_side_by_side: bool = False
        self._prev_vid: Optional[str] = None  # 2026-07-12追加: 対象車切替検知用
        # 2026-07-17追加(92節①): カーブ起因のTTC猶予の連続周期数。
        self._critical_curvature_run: int = 0

    def force_rescue_switch(self) -> None:
        """247節(2026-07-30)追加: OVERTAKING継続中にroom_exhausted(既存側の
        先読みroomが尽きた)がgiveupへ合流する直前、mpc_controller.py側が
        反対側の安全性(is_side_by_side/switchback_space_m/new_side_*_blocked/
        fwd_ds_overlap_risk、通常のswitchbackと全く同一の可否条件)を確認した
        上で、最終手段として側を切り替えると決めた時に呼ぶ。

        中身はbranch=A/A_dlat成立時の状態リセット(586-598行目付近)と完全に
        同一(①非矛盾性: 経路が違うだけで「側が変わった」という事実に対する
        後始末は1種類のみ)。has_switchedもここで消費するため、通常の
        switchback同様「1エンゲージにつき1回」の制限にそのまま従う
        (③ハンチング防止: 新しいラッチを増やさず既存のhas_switchedを共有する
        ことで、この経路が発火してもその後の通常switchbackやこの経路自身の
        再発火が同一エンゲージ内では起きない)。"""
        self.has_switched = True
        self._prev_space = None
        self._space_ema = None
        self._v_corridor_ema = 0.0
        self._shrink_run = 0
        self._critical_curvature_run = 0
        self._dlat_ema = None
        self._prev_dlat_ema = None
        self._v_dlat_ema = 0.0
        self._dlat_shrink_run = 0

    def _update_dlat_trend(self, fwd_dlat: Optional[float], fwd_vid: Optional[str],
                            dt: float) -> None:
        """2026-07-20追加(132節、Gap①Phase0、診断専用): fwd_dlatの縮小トレンドを
        side/reset_episode()と無関係に常時追跡する。update()の先頭から毎周期
        (side==0を含む)呼ぶ。space系トレンド(_prev_space/_space_ema/beta/
        space_ema_alpha/v_inst_max)と全く同じ平滑化→微分→クランプの式をそのまま
        再利用し、新規の数式・新規パラメータは追加しない。リセットするのは
        fwd_dlat=None(対象車ロスト)またはfwd_vid変化(対象車切替、既存の
        _vid_changed処理と同じ考え方)の周期のみで、side==0(未エンゲージ)や
        reset_episode()では消去しない。
        """
        if fwd_dlat is None:
            self._dlat_ema = None
            self._prev_dlat_ema = None
            self._v_dlat_ema = 0.0
            self._dlat_shrink_run = 0
            self._dlat_prev_vid = fwd_vid
            self._dlat_trend_reset_reason = "dlat_none"
            return
        _dlat_vid_changed = (self._dlat_prev_vid is not None and fwd_vid is not None
                              and fwd_vid != self._dlat_prev_vid)
        self._dlat_prev_vid = fwd_vid
        if _dlat_vid_changed:
            self._dlat_ema = fwd_dlat
            self._prev_dlat_ema = None
            self._v_dlat_ema = 0.0
            self._dlat_shrink_run = 0
            self._dlat_trend_reset_reason = "vid_changed"
            return
        if self._dlat_ema is None:
            self._dlat_ema = fwd_dlat
        else:
            self._dlat_ema += self.space_ema_alpha * (fwd_dlat - self._dlat_ema)
        if self._prev_dlat_ema is None:
            self._prev_dlat_ema = self._dlat_ema
            self._dlat_trend_reset_reason = "warmup"
            return
        _d = self._dlat_ema - self._prev_dlat_ema
        self._prev_dlat_ema = self._dlat_ema
        _v_inst_raw = _d / max(dt, 1e-3)
        _v_inst = max(-self.v_inst_max, min(self.v_inst_max, _v_inst_raw))
        self._v_dlat_ema += self.beta * (_v_inst - self._v_dlat_ema)
        if self._v_dlat_ema < 0.0:
            self._dlat_shrink_run += 1
        else:
            self._dlat_shrink_run = 0
        self._dlat_trend_reset_reason = "none"

    def _decision(self, **kwargs) -> TTCDecision:
        """2026-07-20追加(132節、Gap①Phase0): 全てのTTCDecision生成箇所を経由させる
        薄いラッパー。dlat_v_ema/dlat_shrink_run(診断専用フィールド)を、既存の
        11箇所超あるreturn文それぞれへ個別に追記するのではなく、ここ1箇所へ
        集約して常に付与する(②非冗長性)。"""
        kwargs.setdefault("dlat_v_ema", self._v_dlat_ema)
        kwargs.setdefault("dlat_shrink_run", self._dlat_shrink_run)
        kwargs.setdefault("dlat_trend_reset_reason", self._dlat_trend_reset_reason)
        return TTCDecision(**kwargs)

    def update(self, side: int, space: Optional[float], opp_space: Optional[float],
               fwd_dlat: Optional[float], fwd_ds: Optional[float],
               vopp: Optional[float], dt: float,
               fwd_vid: Optional[str] = None, cleared: bool = False,
               new_side_blocked: bool = False,
               new_side_curvature_override: bool = False,
               lookahead_favor_switch: bool = False,
               current_side_closing_ahead: bool = False,
               new_side_room_blocked: bool = False,
               new_side_wall_blocked: bool = False,
               new_side_offset_blocked: bool = False,
               footprint_risk: bool = False,
               fwd_ds_overlap_risk: bool = False) -> TTCDecision:
        """1周期分の更新。space=現在サイドの空き幅、opp_space=反対サイドの空き幅、
        fwd_dlat=対象車との実横間隔、fwd_ds=対象車との実縦間隔(いずれもis_side_by_side判定用)、
        vopp=対象車の縦速度[m/s]、fwd_vid=対象車ID(対象車切替検知用、2026-07-12追加)、
        cleared=呼び出し元の_ot_cleared(真横到達ラッチ、2026-07-13追加)、
        new_side_blocked=反転先(-side)が直近のカーブで閉じるため反転させるべきでないか
        どうか(2026-07-16追加、79節)。呼び出し元がmpc_controller.pyの
        _switchback_curvature_veto(-side)をupdate()呼び出し前に計算して渡す。
        new_side_curvature_override=new_side_blockedが静的曲率のみに基づく懸念で
        あっても、反転先の実測opp_spaceが既にswitchback_space_m(通常のswitchback
        自体が要求する既存の実測ベース閾値)を満たしている場合にTrue(2026-07-22
        追加、157節、0722-03実測でk_cornerが実時間的に有利な反転を過剰に抑制して
        いたことを確認)。new_side_blocked自体は診断用(switchback_curvature_blocked)
        の意味を保つため無変更のまま渡し、可否判定(545/589行目)側でのみ
        `not new_side_blocked or new_side_curvature_override`として考慮する。
        lookahead_favor_switch=現在側(side)が直近のカーブで閉じ、かつ反対側は
        閉じないと判明しているかどうか(2026-07-16追加、84節)。呼び出し元が
        _switchback_curvature_veto(side)(現在側)と_switchback_curvature_veto(-side)
        (=new_side_blocked)の両方を計算し、「現在側が閉じる かつ 反対側は閉じない」
        の場合にTrueを渡す。Trueの場合、通常のmargin/cleared_margin判定を待たずに
        早めの反転(branch=A_lookahead)を許可する。
        current_side_closing_ahead=現在側(side)が直近のカーブで閉じるかどうか
        (2026-07-17追加、92節①、lookahead_favor_switchの算出に使う値と同一の
        _switchback_curvature_veto(side)をそのまま再利用、新規スキャン処理0個)。
        Trueの場合、C2(強制giveup)判定時に「相手に迫られている」のではなく
        「コーナー形状で一時的に狭まっている」可能性が高いと判断し、
        min_trend_cyclesの2倍まで強制giveupを猶予する(branch=C1_deferred)。
        カーブを抜ければv_corridor_emaが自然に回復し、猶予中に危機を脱する想定。
        new_side_room_blocked=反転先(-side)について、対象車両IDごとの学習済み
        走行ライン(OpponentSpeedMap.lat_mean)ベースの先読みroomが物理下限を
        下回ると予測されるかどうか(2026-07-18追加、107節案C、103節Phase 1)。
        new_side_blocked(静的トラック曲率のみ)とは独立した、相手車位置認識
        ベースの判定。呼び出し元がmpc_controller.pyの_opponent_room_ahead()を
        update()呼び出し前に計算して渡す。_rescue_relaxed_eligible(最終救済の
        適格緩和)のみに影響し、通常のswitchback/A_lookahead/A_rescue(厳格閾値)
        には影響しない。データ未学習時はFalse(素通し、fail-open)。
        new_side_offset_blocked=反転先(-side)方向への実測先読み最小値
        (_corr_bound_ahead、既存関数、オフセット目標のクランプに実際に使われている
        のと同一の指標・同一配列)がalong_min_width未満かどうか(2026-07-22追加、
        159節)。new_side_wall_blocked(コリドー全体幅)・space/opp_space(壁基準の
        空き)はいずれも「反転先方向に実際に到達可能か」を見ておらず、0722-03実測で
        A_rescueが成立し側が反転したにも関わらずオフセット目標が実質ゼロへ
        クランプされる(=判定層と実行層の指標不一致)ケースを確認した。判定層に
        実行層と同一の指標を追加することで両層を一致させる。通常のswitchback
        (branch=A/A_lookahead)・A_rescue(厳格閾値)の両方に適用する
        (new_side_wall_blockedと同じスコープ)。
        footprint_risk=fwd_dlat<along_min_width かつ fwd_ds<along_min_length(呼び出し元
        mpc_controller.pyが算出、2026-07-20追加、127節続報)。本メソッドが使うspace/
        opp_space(_scan_traffic内のlf/rf、壁〜相手の隙間の広さ)は自車の現在位置を
        式に含まないため、fwd_dlatが物理下限を割っていてもspaceが「安全」に見える
        矛盾が実測(0720-01予選ログwp173)で確認された。footprint_risk=Trueの場合、
        トレンド判定(shrink_run/v_corridor_ema)を待たず即座に強制giveupする
        (branch=FOOTPRINT_RISK)。
        fwd_ds_overlap_risk(2026-07-26追加、191節、AXIS03対処): 呼び出し元が
        abs(fwd_ds) < along_min_length(footprint_risk本体と同一の物理下限)から
        算出。footprint_riskとは異なりfwd_dlatは問わない(反転の可否そのものを
        塞ぎたいので、まだ横に十分離れている=fwd_dlatが大きい状況でも、縦距離が
        近ければ塞ぐ)。A/A_lookahead/A_dlat/A_rescueいずれの反転経路にも適用する
        (new_side_wall_blockedと同じスコープ)。Trueの間は他の条件を満たしても
        反転せず、既存のfootprint_risk/wall_slowによる減速へ委ねる。
        2026-07-18追加(100節、Tier1裁定の外出し): 旧fwd_is_obstacle_classパラメータ
        (92節続報)は削除した。障害物クラス(vopp<opp_obstacle_speed)の間はF3-TAPER
        (icc_f3)へ委譲しC1のv_safe_capを使わない、という判定(旧branch=
        C1_obstacle_yield)は、C1自体の値(vopp基準キャップ)を変えるものではなく
        「その値を使うかどうか」という呼び出し元の裁定だったため、mpc_controller.py
        側(Tier1)へ移設した。本メソッドは障害物クラスに関わらず常にC1のv_capを計算
        して返し、コリドー物理量(縮小トレンド)のみに専念する本来の役割へ戻す。

        2026-07-16追加の設計意図(79節、0715-07/08実測で発覚したトークン浪費バグの根治):
        旧実装(77節)はmpc_controller.py側でupdate()の戻り値(side_override)を受け取った
        後に別途curvature vetoを行い、反転の実行のみを止めていた。しかしhas_switched/
        has_rescuedは本メソッド内で既にTrueへ更新済みのため、veto発生時にこの
        エピソードの反転トークンが両方とも浪費され、直後に本当の危機的TTCが来ても
        選択肢が無く緊急giveupに至っていた(0715-08実測: wp61switchback成功→
        wp73 A_rescue veto[トークン消費]→wp75 ttc=0.09秒で緊急giveup、1.3秒後)。
        本修正はnew_side_blockedを既存の_switchback_eligible/_rescue_eligible
        判定式にAND条件として追加することで、curvatureでブロックされる反転は
        そもそも「不成立」として扱い、has_switched/has_rescuedを消費させない。
        2026-07-11修正: is_side_by_sideはdlatのみでなくds(縦間隔)も要求するよう修正した。
        旧実装はdlatのみで判定しており、「まだ相手の後方にいて真横に並んでいない
        (ds大)のに、たまたまレースライン基準の横位置が近い(dlat小)だけでis_side_by_side=Trueと
        誤判定され、分岐A(スイッチバック)の権利を失っていた」可能性がある(0711-03/04で
        branch=Aが1度も発火しなかった件の有力仮説)。既存の_ot_cleared判定(2597-2609行目
        付近)が「真横到達」をdlat AND ds<=clear_ds_besideで定義しているのに合わせた。

        2026-07-12修正(0712-01の実挙動ログで発覚した誤スイッチバック・誤giveupへの対処):
        _scan_traffic の対象車選択(mpc_controller.py)にはヒステリシスが無く、fwd_vidが
        周期ごとに別の車へ切り替わり得る。旧_ot_side_block_ema(alpha=0.05、既存の側検出
        処理)は生値を直接強く平滑化していたためこの飛びを吸収できていたが、本モジュールは
        「生値を先に微分してからEMA」という順序だったため、1周期の外れ値がv_corridor_emaへ
        直接混入していた(実測: -32.6m/sという物理的にあり得ない値、および広い側から狭い側へ
        逆向きにスイッチバックする誤動作を確認)。対処として(a)対象車IDが変わった周期は
        トレンドを汚さず静かに再スタートする、(b)既存と同じ考え方で生spaceを先に平滑化
        してから微分する、の2点を追加した。"""
        # 2026-07-20追加(132節、Gap①Phase0): side/reset_episode()の影響を受けない
        # fwd_dlat縮小トレンドを、下記のside==0早期returnより前(=side==0の周期も
        # 含め毎回)更新する。診断専用(dlat_v_ema/dlat_shrink_run)、ENGAGE可否判定
        # にはまだ使わない。
        self._update_dlat_trend(fwd_dlat, fwd_vid, dt)
        if side == 0 or space is None:
            self._prev_space = None
            self._space_ema = None
            self._shrink_run = 0
            self._critical_curvature_run = 0
            self._prev_vid = fwd_vid
            return self._decision(branch="none")

        # 2026-07-20追加(127節続報): footprint_riskは「壁〜相手の隙間」ではなく
        # 「自車〜相手の実測距離」ベースの、既存space/opp_spaceより優先度が高い
        # 緊急信号。トレンド(shrink_run)の蓄積を待たず最優先で強制giveupする。
        # side_override(反転)は返さない(反転自体が相手へ幅寄せする動きになり
        # うるため、125節のswitchback_wall_veto等と同じ「不明な場合は反転させない」
        # 保守的な設計と一貫させる)。
        if footprint_risk:
            self._prev_space = None
            self._space_ema = None
            self._shrink_run = 0
            self._critical_curvature_run = 0
            self._prev_vid = fwd_vid
            return self._decision(force_giveup=True, ttc_lat=0.0, branch="FOOTPRINT_RISK",
                                is_side_by_side=self.is_side_by_side,
                                has_switched=self.has_switched,
                                footprint_risk_triggered=True)

        if fwd_dlat is not None:
            _ds_ok = fwd_ds is not None and fwd_ds <= self.side_by_side_ds_m
            # 2026-07-14追加: dlatにヒステリシスを設ける(水平展開: 事象C対策で発見)。
            # 単一閾値(side_by_side_dlat_m)だと境界付近の測位ノイズでis_side_by_sideが
            # 周期ごとに反転し、branch B(速度介入なし)⇔C1/C2(速度キャップ/強制giveup)が
            # チャーンしうる。既にis_side_by_side=Trueの間はside_by_side_dlat_release_m
            # (より大きい、遠ざかる方向)を上回るまで維持する。逆方向(dlatが縮み危険が
            # 増す方向)には遅延を入れない(離脱側=安全な方向にのみヒステリシスをかける)。
            # 2026-07-14再修正(0714-04実測、副次事象): 上記dlatのヒステリシスに対し、
            # fwd_ds(_ds_ok)は「既にTrueの間」も含めて毎周期チェックされたままだったため、
            # 真横到達(cl相当)直後にfwd_dsがclear_ds_beside(1.0m)境界を1周期だけ跨ぐと
            # is_side_by_sideが即座にFalseへ反転し、蓄積済みのshrink_runに基づくC2判定が
            # 露出して危険水準ではないspace(実測2.56m)で強制giveupが発火していた。
            # ds条件はエントリー(誤って「まだ縦に離れているのに横位置だけ近い」を弾く役割、
            # 2026-07-11修正)にのみ必要であり、一度真に横並びに達した後まで毎周期要求する
            # 理由はない。dlatのヒステリシスと同じ設計(離脱判定はdlatのみに一元化)に揃える。
            if self.is_side_by_side:
                self.is_side_by_side = fwd_dlat < self.side_by_side_dlat_release_m
            else:
                self.is_side_by_side = (fwd_dlat <= self.side_by_side_dlat_m) and _ds_ok

        _vid_changed = (self._prev_vid is not None and fwd_vid is not None
                         and fwd_vid != self._prev_vid)
        self._prev_vid = fwd_vid
        if _vid_changed:
            # 対象車が切り替わった周期は「別の車の空き幅」が飛び込むため、トレンドを
            # 汚さずに新しい基準値へ静かに再スタートする。
            self._prev_space = None
            self._space_ema = None
            self._shrink_run = 0
            self._critical_curvature_run = 0
            return self._decision(branch="warmup", is_side_by_side=self.is_side_by_side,
                                has_switched=self.has_switched)

        # 生spaceを先に平滑化(既存_ot_side_block_emaと同じ考え方)してから微分する。
        if self._space_ema is None:
            self._space_ema = space
        else:
            self._space_ema += self.space_ema_alpha * (space - self._space_ema)

        if self._prev_space is None:
            self._prev_space = self._space_ema
            return self._decision(branch="warmup", is_side_by_side=self.is_side_by_side,
                                has_switched=self.has_switched)
        d_space = self._space_ema - self._prev_space
        self._prev_space = self._space_ema
        v_inst_raw = d_space / max(dt, 1e-3)
        # 2026-07-14追加: 物理的に妥当な範囲へクランプする(v_inst_max参照、コンストラクタ
        # docstring参照)。dt異常等による瞬間的な外れ値がv_corridor_emaへ混入するのを防ぐ。
        v_inst = max(-self.v_inst_max, min(self.v_inst_max, v_inst_raw))
        _v_inst_clamped = abs(v_inst_raw) > self.v_inst_max
        self._v_corridor_ema += self.beta * (v_inst - self._v_corridor_ema)

        _diag = dict(is_side_by_side=self.is_side_by_side, has_switched=self.has_switched,
                     has_rescued=self.has_rescued,
                     v_corridor_ema=self._v_corridor_ema, shrink_run=self._shrink_run,
                     cleared=cleared, v_inst_clamped=_v_inst_clamped,
                     lookahead_favor_switch=lookahead_favor_switch,
                     # 2026-07-22追加(157節): new_side_blocked(生の静的曲率判定、
                     # switchback_curvature_blockedの意味を保つため無変更)とは別に、
                     # 「今回overrideが実際に成立しうる状況だったか」を常に記録する。
                     switchback_curvature_overridden=(new_side_blocked
                                                       and new_side_curvature_override))

        # 2026-07-26追加(186節続報、クロスライン対策、ユーザー承認済み設計):
        # 第3コーナー追突(0726-01実測、wp161 ENGAGE→wp170 footprint_risk giveup)の
        # 分析で、壁コリドー(_v_corridor_ema)は終始「安定」(直後のif文でbranch=
        # "stable"へ抜ける)だったにも関わらず、fwd_dlat(自車〜相手の実測横間隔)
        # だけが11周期連続で縮小し続けていたことを確認した。壁ベースの判定は
        # 「壁〜相手の隙間」(space/opp_space)のみを見ており、相手が斜行等で
        # 自車へ直接幅寄せしてくる動きを検知できない構造的な盲点だった。
        # 下のif文(壁ベース)より前に評価し、壁が「安定」と言っていても
        # このfwd_dlatベースの独立トリガーだけで発火できるようにする。
        #
        # 対処方針は既存のswitchback判定式(_switchback_eligible/veto群)を一切
        # 変更せず、そのTTCゲートだけをfwd_dlat起点で追加で用意する形にした
        # (①上位〜下位レイヤの一貫性: 反転可否条件・has_switchedラッチ・
        # 発火時の状態リセットは branch=A と完全に同一。②非冗長性: 新規の
        # 安全弁・新規チューニング値は追加しない。ttc_danger_s(既存branch=Aの
        # 危険判定閾値)とcleared_space_m(=along_min_width、footprint_riskが
        # 実際に強制giveupへ切り替える閾値と同一値を予測的に転用)・
        # min_trend_cycles(壁ベースと共用)の3つの既存定数のみを使う。
        # ③ハンチング: has_switchedを共有するため、このトリガーが発火しても
        # 通常branch=Aの反転トークンを追加消費するだけで、独自のラッチは
        # 持たない=1エンゲージ1回のみに自動的に制限される。④単純さ: 「反転
        # トリガーが壁基準・相手基準の2種類に増えたが、どちらも同じ実行経路
        # (side_override・同一veto群)へ収束する」という一段のみの追加に留めた)。
        #
        # 0726-01実測による閾値の裏付け: footprint_risk発火時点(wp170)で
        # dlat_shrink_run=11。min_trend_cycles=3で本トリガーを評価すると、
        # 当時のdlat_v_ema推移(-0.365〜-1.06m/s)から概算して発火の約1.2秒前
        # (11周期中の3周期目付近)には既に評価可能だった計算になる(周期毎の
        # 詳細ログが残っていないため概算。イベントトリガー式ログの範囲内で
        # 妥当性を確認済み)。
        if (fwd_dlat is not None and self._v_dlat_ema < 0.0
                and self._dlat_shrink_run >= self.min_trend_cycles):
            _dlat_residual = fwd_dlat - self.cleared_space_m
            ttc_dlat = (0.0 if _dlat_residual <= 0.0
                        else _dlat_residual / abs(self._v_dlat_ema))
            if ttc_dlat <= self.ttc_danger_s:
                _dlat_switchback_eligible = (not self.is_side_by_side and not self.has_switched
                                              and opp_space is not None
                                              and opp_space >= self.switchback_space_m)
                if _dlat_switchback_eligible:
                    # branch=Aの_margin/_cleared_margin_okと完全に同一の式
                    # (549-559行目参照)。トリガー源(壁/dlat)が違うだけで、
                    # 可否判定そのものは1種類しか存在しないことを保つ。
                    _dlat_margin = opp_space - space
                    _dlat_cleared_margin_required = self.switchback_space_m - self.giveup_space_m
                    _dlat_cleared_margin_ok = (not cleared) or (_dlat_margin >= _dlat_cleared_margin_required)
                    _dlat_reactive_ok = (_dlat_margin >= 0.0) and _dlat_cleared_margin_ok
                    if ((not new_side_blocked or new_side_curvature_override)
                            and not new_side_wall_blocked and not new_side_room_blocked
                            and not new_side_offset_blocked
                            and not fwd_ds_overlap_risk
                            and _dlat_margin >= 0.0
                            and (lookahead_favor_switch or _dlat_reactive_ok)):
                        self.has_switched = True
                        self._prev_space = None
                        self._space_ema = None
                        self._v_corridor_ema = 0.0
                        self._shrink_run = 0
                        self._critical_curvature_run = 0
                        # 反転先で基準が変わるため、自身のdlatトレンドもここで
                        # リセットする(古い縮小方向を残すと反転直後に同じ
                        # トレンドで誤って再評価されるおそれがあるため、
                        # branch=A発火時の壁トレンドリセットと同じ考え方)。
                        self._dlat_ema = None
                        self._prev_dlat_ema = None
                        self._v_dlat_ema = 0.0
                        self._dlat_shrink_run = 0
                        return self._decision(side_override=(-side), ttc_lat=ttc_dlat,
                                           branch="A_dlat", **_diag)
                    _diag["switchback_suppressed"] = True
                    if new_side_blocked:
                        _diag["switchback_curvature_blocked"] = True
                    elif new_side_wall_blocked:
                        _diag["switchback_wall_blocked"] = True
                    elif new_side_room_blocked:
                        _diag["switchback_room_blocked"] = True
                    elif new_side_offset_blocked:
                        _diag["switchback_offset_blocked"] = True
                    elif fwd_ds_overlap_risk:
                        _diag["switchback_ds_blocked"] = True
                    elif cleared and _dlat_margin >= 0.0 and not _dlat_cleared_margin_ok:
                        _diag["switchback_cleared_margin_blocked"] = True

        if self._v_corridor_ema >= 0.0:
            self._shrink_run = 0
            self._critical_curvature_run = 0
            return self._decision(branch="stable", **_diag)
        self._shrink_run += 1
        _diag["shrink_run"] = self._shrink_run
        if self._shrink_run < self.min_trend_cycles:
            return self._decision(branch="warmup", **_diag)

        # 2026-07-13追加: cleared(真横到達)後は閾値をcleared_space_m(物理下限)へ緩和する
        # (旧_ot_side_block_emaの二段階閾値と同一の考え方)。コーナー形状によるコリドー
        # 自然収束と、実際に相手に幅寄せされている状況を区別する。
        _threshold = self.cleared_space_m if cleared else self.giveup_space_m
        residual = space - _threshold
        ttc_lat = 0.0 if residual <= 0.0 else residual / abs(self._v_corridor_ema)
        if ttc_lat > self.ttc_danger_s:
            return self._decision(ttc_lat=ttc_lat, branch="stable", **_diag)

        # 2026-07-16追加(81節)→82節で実装→0716-03実測で回帰確認、83節でrevert:
        # 81節時点の仮説は「clearedはmpc_controller.py側の_ot_cleared(dlat>=1.8〜2.1で
        # 成立)、is_side_by_sideは本クラス内部の判定(dlat<=1.6かつds<=1.0で成立)で
        # 判定条件が異なるため、横には十分離れたがまだ縦に追いついていない局面
        # (cleared=True かつ is_side_by_side=False)でswitchbackが確保済みの横間隔を
        # 無駄に破棄する(ep5、0716-02 wp298-328)」というもので、82節でclearedを
        # _switchback_eligibleのガードに追加した。
        # しかし0716-03実測で、clearedは実際には長時間(数秒〜十数秒)成立したままの
        # 局面が大半であり、その間もコリドーは変動し続けるにも関わらず通常switchback
        # (branch A)が一律に封じられた結果、反対側が明確に(時に2倍以上)広くても
        # 安全に反転できず、コリドー限界(offset=±3.00)に長時間張り付いたまま
        # COLLISION-SUSPECTEDが多発した(Lap1 wp217-240・wp196-205、Lap2 wp202-232で
        # 計5件、いずれも「switchback_suppressed reason=cleared」直後に発生)。
        # 82節の前提(cleared中の反転は得るものがない)は実測で反証されたため、
        # clearedによるガードを削除しrevertする(83節、ユーザー承認済み)。
        _switchback_eligible = (not self.is_side_by_side and not self.has_switched
                                 and opp_space is not None
                                 and opp_space >= self.switchback_space_m)
        if _switchback_eligible:
            # 2026-07-14追加: 過去ログ10本・switchback発火61件の定量検証で、
            # opp_space<space(反対側の方が実は現在側より狭い)への反転は21件中21件
            # (100%)が直後(15秒以内)のgiveupに終わり、一度も有効だった例が無いことを
            # 確認した。絶対閾値(switchback_space_m)だけでは「現在側がまだ十分広いのに
            # 縮小トレンドだけで反転してしまう」ケースを弾けないため、現在側より
            # 反対側が広い場合のみ反転を許可する(新規パラメータ不要、既存space/
            # opp_spaceの比較のみ)。過去の成功例5件は全てmargin>=0だったため、この
            # ガードで犠牲になる成功例は無い。
            _margin = opp_space - space
            # 2026-07-16追加(84節①、margin 0.5m案): 82節の「cleared中は一律禁止」
            # (83節でrevert)に代わる、より狭いガード。cleared中(既に十分な横間隔を
            # 確保済み)は、marginが既存のswitchback_space_m-giveup_space_m(0.5m、
            # 新規パラメータ0個)以上であることを追加要求する。0716-02実測ep5
            # (margin=0.05)のような「ほぼ得るものが無い反転」だけを狙い撃ちで抑制し、
            # 0716-03実測(wp202-204、margin=1.6超)のような「反対側が明確に広い」
            # 場面は従来通り反転できる。
            _cleared_margin_required = self.switchback_space_m - self.giveup_space_m
            _cleared_margin_ok = (not cleared) or (_margin >= _cleared_margin_required)
            _reactive_ok = (_margin >= 0.0) and _cleared_margin_ok
            # 2026-07-16追加(84節②、カーブ先回り切り替え): 現在側が前方カーブで
            # 閉じ、反対側は閉じないと分かっている場合(呼び出し元が
            # _switchback_curvature_veto()を現在側/反対側の両方へ適用して算出、
            # 新規スキャン処理0個)、通常のmargin/cleared_margin判定を待たず早めに
            # 反転する。直線走行中はlookahead_favor_switch自体がFalseのままのため、
            # 無駄な反転は増えない。
            # 2026-07-16追加(79節、継続): curvatureでブロックされる反転は「不成立」
            #   として扱い、has_switchedを消費しない(下記new_side_blocked参照)。
            # 2026-07-18追加(107節、案A'): 84節①が過去ログ61件の定量検証で確立した
            #   「opp_space<space(反対側の方が狭い)への反転は100%が15秒以内に
            #   giveupに終わる」というmargin>=0ガードを、lookahead_favor_switch
            #   (84節②)のOR分岐が完全にバイパスしていた(0718-05実測T=701.15、
            #   opp_space=2.07<space=2.19でmargin=-0.12なのにA_lookaheadが発火し、
            #   直後にwall_slow・COLLISION-SUSPECTEDが発生)。lookahead_favor_switch
            #   は引き続き_cleared_margin_ok(clearedの間だけ要求される追加0.5m
            #   バッファ)のみをバイパスできるが、margin>=0自体は常時必須とする。
            # 2026-07-19追加(103/106/107節が確立した非対称性の解消、120節続報):
            #   new_side_blocked(静的トラック曲率のみ)と同じ考え方・同じ位置で、
            #   new_side_room_blocked(反転先の対象車両ID込みroom先読み、103/107節で
            #   既にA_rescue_relaxedへは統合済みの既存入力をそのまま再利用)も
            #   通常のswitchback(branch=A/A_lookahead)の成立条件へ追加する。
            #   新規パラメータ0個(引数は既にupdate()シグネチャに存在、呼び出し元
            #   mpc_controller.pyも既に毎周期計算・伝搬済み)。
            if ((not new_side_blocked or new_side_curvature_override)
                    and not new_side_wall_blocked and not new_side_room_blocked
                    and not new_side_offset_blocked and not fwd_ds_overlap_risk
                    and _margin >= 0.0 and (lookahead_favor_switch or _reactive_ok)):
                self.has_switched = True
                self._prev_space = None
                self._space_ema = None
                self._v_corridor_ema = 0.0
                self._shrink_run = 0
                self._critical_curvature_run = 0
                _branch = "A" if _reactive_ok else "A_lookahead"
                return self._decision(side_override=(-side), ttc_lat=ttc_lat, branch=_branch, **_diag)
            _diag["switchback_suppressed"] = True
            if new_side_blocked:
                _diag["switchback_curvature_blocked"] = True
            elif new_side_wall_blocked:
                _diag["switchback_wall_blocked"] = True
            elif new_side_room_blocked:
                _diag["switchback_room_blocked"] = True
            elif new_side_offset_blocked:
                _diag["switchback_offset_blocked"] = True
            elif fwd_ds_overlap_risk:
                _diag["switchback_ds_blocked"] = True
            elif cleared and _margin >= 0.0 and not _cleared_margin_ok:
                _diag["switchback_cleared_margin_blocked"] = True

        if self.is_side_by_side:
            return self._decision(ttc_lat=ttc_lat, branch="B", **_diag)

        if ttc_lat <= self.ttc_critical_s:
            # 2026-07-15追加(ユーザー指摘の深掘りで確認、0715-03実測t=16.90〜23.42):
            #   通常のswitchback(上のブロック)は1エピソード1回のみだが、既にそれを
            #   消費した後でも、断念(giveup)に至る過程で反対側が明確に広がっている
            #   ケースが実測で確認された(t=21.5: space=3.14/opp_space=1.86 → t=22.9:
            #   space=2.54/opp_space=3.14、完全に逆転)。この時点でopp_space>=space
            #   かつopp_space>=switchback_space_mという通常のswitchbackと全く同じ
            #   条件を満たしていたにも関わらず、has_switched消費済みのため反転できず、
            #   断念する以外の選択肢が無かった。
            #   「もはや断念する寸前」という最終局面に限り、既存のswitchback判定式を
            #   そのまま再利用して最後にもう一度だけ側変更を試す(has_rescuedで
            #   1エピソード1回のみに制限。通常のmid-pass反転回数制限を緩和するもの
            #   ではなく、断念と表裏一体のこの1点のみの例外)。
            # 2026-07-16追加(79節): new_side_blockedをAND条件に追加し、curvatureで
            #   ブロックされる救済反転はhas_rescuedを消費しない「不成立」として扱う
            #   (0715-08実測: wp73のA_rescueがcurvature vetoで実行されなかったにも
            #   関わらずhas_rescuedが消費され、1.47秒後のwp75で本当に危機的な
            #   giveup(ttc=0.09秒)の際に再挑戦できなかった)。
            _rescue_eligible = (not self.has_rescued
                                 and opp_space is not None
                                 and opp_space >= self.switchback_space_m
                                 and opp_space >= space
                                 and (not new_side_blocked or new_side_curvature_override)
                                 and not new_side_wall_blocked
                                 and not new_side_offset_blocked
                                 and not fwd_ds_overlap_risk)
            if _rescue_eligible:
                self.has_rescued = True
                self._prev_space = None
                self._space_ema = None
                self._v_corridor_ema = 0.0
                self._shrink_run = 0
                self._critical_curvature_run = 0
                return self._decision(side_override=(-side), ttc_lat=ttc_lat,
                                   branch="A_rescue", **_diag)
            if new_side_blocked:
                _diag["switchback_curvature_blocked"] = True
            elif new_side_wall_blocked:
                _diag["switchback_wall_blocked"] = True
            elif new_side_offset_blocked:
                _diag["switchback_offset_blocked"] = True
            elif fwd_ds_overlap_risk:
                _diag["switchback_ds_blocked"] = True

            # 2026-07-17追加(92節①、カーブ起因のTTC猶予、ユーザー承認済み設計):
            #   current_side_closing_ahead(現在側がこの先カーブで閉じると判明)が
            #   Trueの間は、「相手に迫られている」のではなく「コーナー形状で
            #   一時的に狭まっている」可能性が高い。コーナーを抜ければ
            #   v_corridor_emaは自然に回復し上のstable分岐へ抜けるため、
            #   min_trend_cyclesの2倍(新規パラメータ0個、既存定数の再利用)まで
            #   強制giveupを猶予し、その間はC1相当の縦速度キャップのみ課す
            #   (完全ノーガードにはしない)。この分岐へ来る時点で既に
            #   is_side_by_side=False(上のブロックで確定済み)であることが保証される。
            if current_side_closing_ahead:
                self._critical_curvature_run += 1
            else:
                self._critical_curvature_run = 0
            _diag["critical_curvature_run"] = self._critical_curvature_run
            if (current_side_closing_ahead
                    and self._critical_curvature_run < self.min_trend_cycles * 2):
                v_cap = None
                if vopp is not None:
                    v_cap = max(0.0, vopp - self.caution_speed_margin_kmh / 3.6)
                # 2026-07-20追加(144節続報、0720-05実測wp320-326の衝突分析):
                #   猶予中に逃げ道(switchback/rescue)が壁(new_side_wall_blocked)
                #   または相手位置先読み(new_side_room_blocked)で既にブロックされて
                #   いると分かっている場合、「カーブを抜ければ回復する」という92節①の
                #   前提(猶予中に危機を脱する想定)が成立しない可能性が高い。この場合、
                #   通常のvoppベースの緩いキャップ(相手と同程度の速度を許容)ではなく、
                #   壁際減速・footprint_risk等、他の緊急層と共通のwall_slow_speed
                #   (より保守的な既存定数)をmin()で重ねる。猶予の長さ・switchback/
                #   rescue自体の成立条件は一切変更しない(82/83節の教訓: 逃げ道の
                #   成立条件を広く制限すると重大な回帰を招いた実測がある)。
                if new_side_wall_blocked or new_side_room_blocked:
                    v_cap = (self.wall_slow_speed if v_cap is None
                              else min(v_cap, self.wall_slow_speed))
                return self._decision(
                    v_safe_cap=v_cap,
                    v_safe_cap_label=("lat_ttc(C1相当: カーブ起因のTTC猶予中、逃げ道封鎖でキャップ強化)"
                                        if (new_side_wall_blocked or new_side_room_blocked)
                                        else "lat_ttc(C1相当: カーブ起因のTTC猶予中)"),
                    ttc_lat=ttc_lat, branch="C1_deferred", **_diag)

            # 2026-07-17追加(92節②、最終救済の適格緩和、ユーザー承認済み設計):
            #   上の猶予が対象外(カーブ起因ではない=相手に実際に迫られている)、
            #   または猶予を使い切っても解消しなかった場合、断念する以外の
            #   選択肢が無い最終局面に限り、反対側の必要幅を物理下限
            #   (cleared_space_m)まで緩和して再挑戦する。has_rescuedは通常の
            #   A_rescueと共有し、1エピソード1回のみ(新規トークンは増やさない)。
            #
            #   2026-07-18追記(102節続報、検討結果): 0718-03実測(wp≈333〜335で
            #   2回再現)を受け、current_side_closing_ahead=True時のみ閾値を
            #   switchback_space_mへ引き上げる案(案2)を試みたが、これは上の
            #   _rescue_eligible(常時switchback_space_m基準)と完全に同一の式に
            #   なるため、_rescue_eligibleが不成立だった時点でこちらも必ず不成立
            #   になる到達不能コードだと判明し撤回した(実装・検証はdesign_docs
            #   102節参照)。根本対処は103節でOpponentSpeedMap.lat_mean(対象車両
            #   IDごとの学習済み走行ライン)を使った先読みroom計算として設計中
            #   (Phase 0: mpc_controller.py側の診断ロギングのみ、本メソッドの
            #   判定ロジックは現時点で未変更)。
            _rescue_relaxed_eligible = (not self.has_rescued
                                         and opp_space is not None
                                         and opp_space >= self.cleared_space_m
                                         and opp_space >= space
                                         and not new_side_blocked
                                         and not new_side_wall_blocked
                                         and not new_side_room_blocked
                                         and not fwd_ds_overlap_risk)
            if _rescue_relaxed_eligible:
                self.has_rescued = True
                self._prev_space = None
                self._space_ema = None
                self._v_corridor_ema = 0.0
                self._shrink_run = 0
                self._critical_curvature_run = 0
                return self._decision(side_override=(-side), ttc_lat=ttc_lat,
                                   branch="A_rescue_relaxed", **_diag)

            # cleared中でも最終防波堤として残す(閾値は上でcleared_space_mへ緩和済み)。
            _branch = "C2_cleared" if cleared else "C2"
            return self._decision(force_giveup=True, ttc_lat=ttc_lat, branch=_branch, **_diag)

        if cleared:
            # 2026-07-13追加: 真横到達済みならC1(縦速度キャップ)は課さない。相手サイドへの
            # 接近自体にはペナルティが無く、壁距離は別機構(ICC/G-2/G-3/F3taper)が担当する
            # ため、ここでの速度介入は「コーナー形状によるコリドー収束」への誤反応でしか
            # ない。空きが物理下限に迫った場合のみ上のC2_clearedで対処する。
            return self._decision(ttc_lat=ttc_lat, branch="B_cleared", **_diag)

        # 2026-07-18追加(100節、Tier1裁定の外出し): 旧C1_obstacle_yield分岐
        # (92節続報)はここで削除した。障害物クラス判定(vopp<opp_obstacle_speed)は
        # コリドー物理量と無関係な外部裁定であり、mpc_controller.py側でこの返り値
        # (branch="C1"のv_safe_cap)を候補として使うかどうかを判定する
        # (詳細はdesign_docs 100節参照)。本メソッドはC1のv_capを障害物クラスに
        # 関わらず常に計算して返す。

        v_cap = None
        if vopp is not None:
            v_cap = max(0.0, vopp - self.caution_speed_margin_kmh / 3.6)
        return self._decision(
            v_safe_cap=v_cap,
            v_safe_cap_label="lat_ttc(C-stage1: 縦間合い確保)",
            ttc_lat=ttc_lat, branch="C1", **_diag)
