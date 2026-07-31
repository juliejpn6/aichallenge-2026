"""254節(2026-07-31): 後方車両を「前方追従対象」として扱ってしまうバグの根本修正。

背景: ユーザーから「WP240付近で後続車両に追い付かれた際に停車している。前方も
塞がれておらず普通に走行しなければならない状況だった」との指摘を受け調査した。
実測(0731-03 wp243、v2x生データ+ログ照合): 対戦車d3はwp234-238付近でegoに
追い抜かれ、ds符号が反転(+1.0m→-1.0m、以降ずっと後方2m前後に固定)した。
その後d3自身が停止(vopp: 5.9→2.4→1.5→0.7→0.0)するのに追従する形でicc_stopの
v_safeが0.0に張り付き、egoも完全停止した。この間、前方(ds>0)には誰もいなかった
(v2x実測・コード自身のd_minログ双方で確認)。

根本原因: `_scan_traffic`のcars構築条件(2026-07-19、105/110節)が前方車判定の
下限を`-along_min_length < ds`まで緩和しており、これと2026-07-20(129節続報)の
対象車選択フォールバック(「前方候補が無ければ後方along_min_length以内で代替」)
が組み合わさることで、以下3箇所が後方2m以内の車を「前方の対象車」として扱って
いた:
  ① `_follow_speed_limit`(icc_stop本体): 後方車のds(負値)とvopp(≈0、停止車)を
     G2ブレーキ式`rad = v_fwd² + 2*a_brake*(ds-margin_center)`へ渡すと、
     ds<0でさらに保守的側(radがより負)へ振れ、v_safe=0(完全停止)を返し続ける。
  ② `_n_fwd = len(scan["cars"])`(_ot_state遷移の起点判定): 後方車だけでも
     真になり、`_fwd_clear_count`が毎周期リセットされ続けてSTOPPING→NORMAL
     への復帰が妨げられる。
  ③ NO-VSAFEブリッジ(139/138-5節): `abs(_fwd_ds) < along_min_length`と
     符号を見ずに判定していた。

ユーザー確認: 後方車両を追従する必要は無い。水平展開の結果、`_rear_clearance_m`
(意図的に後方専用)・K-CHECK(`_side_blocked_by_other_car`、既にds>0限定)・
along_lat/being_overtaken(2026-07-26修正済み、ds>=0限定)は影響を受けないことを
確認した。また131-6節のOFFSET-RETURN判定は既に`fwd_ds_now > 0.0`を独立に
明示チェックしており、105/110節が懸念していたcliff問題はこの安全策により
緩和窓が無くても再発しないと判断し、`_scan_traffic`のcars構築をds>=0限定へ
戻した(該当ファイル: test_scan_traffic_ds_cliff.py参照)。

対処(単一の根本修正): `_scan_traffic`のcars構築条件を`0.0 <= ds`へ限定。
これにより①②は自動的に解消し(carsに後方車が入らなくなるため)、③は
abs()を撤去して契約を明示した(test_stopping_no_vsafe_bridge.py参照)。
`_ds_priority`の後方フォールバック分岐は事実上到達不能になったが、対象車
選択の優先度キーという役割・関数自体は不変のため維持し、docstringのみ更新した。

このファイルは①(icc_stop本体)のG2式ミラーによる遡及検証と、_ds_priorityの
契約変更を確認する構造的ソーステキスト検証を行う。②③は各専用テストファイル
(test_scan_traffic_ds_cliff.py / test_stopping_no_vsafe_bridge.py)で検証済み。
"""
import os
import math

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

FWD_A_BRAKE = 1.3
FWD_MARGIN_CENTER = 4.0


def _g2_speed(v_fwd, ds, a_brake=FWD_A_BRAKE, margin_center=FWD_MARGIN_CENTER):
    """mpc_controller.py _g2_speedのミラー実装。"""
    rad = v_fwd * v_fwd + 2.0 * a_brake * (ds - margin_center)
    return math.sqrt(max(0.0, rad))


def _is_forward_candidate(ds, max_consider=25.0):
    """254節後のcars構築条件(0.0<=ds<=max_consider、lat_band省略)のミラー。"""
    return 0.0 <= ds <= max_consider


def test_retroactive_0731_03_rear_car_g2_speed_was_zero_pre_fix():
    """遡及検証: 0731-03実測相当(後方d3、ds≈-1.95、停止直前vopp≈0.0)を
    G2式へ直接代入すると、修正前の設計(この値がbestとして選ばれ得た)では
    v_safe=0.0(完全停止)を返していたことを数式的に確認する
    (バグそのものの再現、対処法の正当性の裏付け)。"""
    v_safe = _g2_speed(v_fwd=0.0, ds=-1.95)
    assert v_safe == 0.0


def test_rear_car_g2_speed_collapses_to_zero_as_stopping_sequence_progresses():
    """遡及検証: 0731-03実測のvopp減衰列(5.9→2.4→1.5→0.7→0.0)を後方車
    (ds=-1.95)として G2式へ通すと、単調非増加でv_safeが低下し、
    vopp<=2.4(実測でv_safe=0.0が確認された水準)以降は完全に0へ張り付くことを
    確認する。ds<0(後方)自体がmargin_centerとの差をより負に振らせ、通常の
    前方追従より低いvoppでも早く0へ落ちる(=後方車の存在だけでegoが不要に
    停止し得る)ことがこの数式的性質から裏付けられる。"""
    v_safes = [_g2_speed(v_fwd=vopp, ds=-1.95) for vopp in (5.9, 2.4, 1.5, 0.7, 0.0)]
    assert v_safes == sorted(v_safes, reverse=True)  # vopp低下と共に単調非増加
    assert v_safes[-1] == 0.0  # vopp=0.0(完全停止)では確実に0
    assert v_safes[1] == 0.0  # vopp=2.4(実測でv_safe=0.0が確認された水準)でも0


def test_fix_excludes_rear_car_from_forward_candidates():
    """対処の核心: 254節後のcars構築条件では、後方2m前後(ds<0)の車は
    そもそも前方候補として登場しない。したがって_follow_speed_limitの
    best選択にすら渡らず、上記のG2式評価自体が発生しない。"""
    for ds in (-1.95, -1.97, -1.01, -0.98, -0.01):
        assert _is_forward_candidate(ds) is False


def test_fix_still_allows_genuine_forward_car_through_g2():
    """非退行の確認: 真に前方(ds>=0)の車は従来通りcandidateとなり、
    G2式による速度上限計算も従来通り機能する(後方車の除外だけを行い、
    前方追従ロジック自体は無変更であることの確認)。"""
    assert _is_forward_candidate(ds=1.0) is True
    v_safe = _g2_speed(v_fwd=0.0, ds=1.0)
    assert v_safe == 0.0  # 前方1mで停止車がいれば従来通り止まる、これは正しい挙動


# ---------------------------------------------------------------------------
# ②④ _ds_priority: 後方フォールバック分岐は到達不能になったが関数自体は維持
# ---------------------------------------------------------------------------

def test_ds_priority_function_unchanged_but_docstring_updated():
    """_ds_priority自体の実装(優先度キー算出式)は変更していないことを確認する
    (対象車選択という役割・関数シグネチャは不変。修正はもっぱら呼び出し元
    (_scan_traffic)がds>=0のみを渡すようになった点)。"""
    idx = _SRC.index("def _ds_priority(ds: float) -> float:")
    idx_end = _SRC.index("def _dlat_closing_trend(")
    snippet = _SRC[idx:idx_end]
    assert "return ds if ds >= 0.0 else (1e9 - ds)" in snippet
    assert "254節" in snippet
    assert "到達不能" in snippet


def test_ds_priority_rear_branch_never_reached_from_scan_traffic():
    """配線確認: _ds_priorityの呼び出し元(_scan_traffic内のbest選択、
    _follow_speed_limit内のbest選択)がいずれもscan["cars"]由来の値のみを
    渡しており、carsが0.0<=ds限定になった今、ds<0の入力で_ds_priorityが
    呼ばれることは無いことを確認する。"""
    idx1 = _SRC.index("if best is None or self._ds_priority(ds) < self._ds_priority(best[0]):")
    idx2 = _SRC.index(
        "if best is None or self._ds_priority(ds) < self._ds_priority(best[0]):",
        idx1 + 1)
    assert idx1 > 0 and idx2 > idx1
    # 直前の"cars"要素はいずれもfor ds, ... in scan["cars"] / out["cars"].append の
    # ループ変数由来であり、その"cars"はいずれも0.0<=ds限定条件を通過済み。
    idx_cars_build = _SRC.index('if 0.0 <= ds <= self._ot_max_consider')
    idx_cars_append = _SRC.index('out["cars"].append((ds, lat, v_long, dlat, vid, wp_i))')
    assert idx_cars_build < idx_cars_append < idx1


# ---------------------------------------------------------------------------
# 水平展開: 意図的にds<0を扱う既存箇所が今回の修正で壊れていないことの確認
# ---------------------------------------------------------------------------

def test_rear_clearance_m_intentionally_rear_only_unaffected():
    """_rear_clearance_m()はV2Xトラッカーを直接走査する独立実装であり、
    scan["cars"]を再利用しないため254節の変更の影響を受けない
    (意図的にds<0=後方専用、BACKUP時の後退距離制限用)。"""
    idx = _SRC.index("def _rear_clearance_m(")
    idx_end = _SRC.index("def _compute_stuck_push_steer(")
    snippet = _SRC[idx:idx_end]
    assert 'scan["cars"]' not in snippet
    assert "if ds >= 0.0 or ds < -self._stuck_rear_scan_max_dist_m:" in snippet


def test_k_check_already_forward_only_unaffected():
    """_side_blocked_by_other_car(K-CHECK)はscan["cars"]を再利用するが、
    既に`0.0 < c_ds`で前方限定していたため、254節でcars自体が0.0<=ds
    限定になったことと矛盾せず(むしろ整合性が上がる)、動作は変わらない。"""
    idx = _SRC.index("def _side_blocked_by_other_car(")
    idx_end = _SRC.index("def _g2_speed(") if _SRC.index(
        "def _g2_speed(") > idx else len(_SRC)
    snippet = _SRC[idx:idx + 2000]
    assert "if not (0.0 < c_ds <= ds_end):" in snippet


def test_along_lat_already_ds_ge_zero_unaffected():
    """along_lat/being_overtaken計算(2026-07-26修正済み)は既に
    0.0<=ds<=def_alongside_distで前方〜真横限定であり、254節の変更と
    重複・矛盾しない。"""
    idx = _SRC.index('if (0.0 <= ds <= self._def_alongside_dist')
    assert idx > 0
