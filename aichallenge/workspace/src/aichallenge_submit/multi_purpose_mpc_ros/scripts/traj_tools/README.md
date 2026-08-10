# traj_tools

25km/hディフェンス・35km/hオフェンスのトラジェクトリCSV生成スクリプト(2026-08-09〜10)。
経緯・失敗履歴の詳細はdesign_docs opp_lat_pred_overlap_guard_design_20260806.md §47系参照。

## 前提

`traj_mincurv.csv`(TUM形式: `s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2`)は、
実行時に**x_m,y_m列のみ**が使われる(`mpc_controller.py`/`path_constraints_provider.py`が
`load_ref_path()`の戻り値のうちpsi/kappaを`_, _`で破棄し、`ReferencePath`クラスが
x,yのみから自前でリサンプリング・psi/kappa再計算・速度プロファイル計算を行うため)。
本ディレクトリのスクリプトはこの前提のもと、x,yの編集に専念している。

## ファイル

- `geometry.py`: TUM CSV read/write + x,y編集後のs_m/psi_rad/kappa_radpm再計算
  (3点外接円による符号付き曲率、TUM規約[psi=0は北基準]に対応)。
- `build_two_trajectories.py`: 初期実装(2026-08-09)。
  - オフェンス版(`traj_offense_35kmh.csv`)生成部分は**現役**(現行ジオメトリを
    そのままコピー、既存traj_mincurv.csvのvx_mpsが既にay_max~12設計だったため)。
  - タイトコーナー拡幅(移動平均+raised-cosine taper)は**3回とも失敗・不採用**
    (taper境界でピーク曲率がむしろ悪化)。関数は残置しているが呼び出していない。
- `build_defense_v2.py`: ディフェンス版生成の初期版(2026-08-10、**現在不使用、
  経緯保存のため残置**)。taperなしの生イン側バイアス+全体への軽いLaplacian平滑化
  で不連続を均す設計だったが、隣接コーナー間の隙間・単一コーナー境界で平滑化だけ
  では吸収しきれない曲率スパイクが8箇所系統的に残ることが事後判明(design_docs
  §51、うち1箇所はwp269-282帯のwp280で最重要ホットスポットに直撃)。
- `build_defense_v3.py`: ディフェンス版(`traj_defense_25kmh.csv`)生成の**現行版**
  (2026-08-10)。v2からの変更点:
  1. 各ゆるいコーナー内部でバイアス振幅をsmoothstep(3t²-2t³)で0→最大→0とテーパー
     (taper幅=コーナー長の1/3・最小4pt、taper余地が確保できないコーナーは
     バイアス自体を見送る)。smoothstepは両端で導関数=0のため、隣接コーナーとの
     隙間が数ptでも不連続を作らない。
  2. (200,210,"L")は原本ピーク曲率0.146(R≈6.8m)と「ゆるい」というよりタイトな
     部類で、内側オフセットカーブの曲率増幅(半径R・オフセットdで新曲率=1/(R-d))
     と相性が悪くtaperでも境界で原本超えの曲率が残ったため、GENTLE_CORNERSから除外。
  3. 全体Laplacian平滑化は「仕上げ」として残すが、実際にはalpha=0/passes=0
     (平滑化なし)が最も良好な結果になった——taper自体が連続性を担保するため。
  4. `kaleidoscope.trajectory_clearance.validate_clearance`によるclearance検証に
     加え、原本ジオメトリに対する新規曲率スパイク検出(隣接点比1.6倍超)も
     合否条件に追加(v2はclearanceのみでスパイクを見逃していた)。
  詳細な発見経緯・検証結果はdesign_docs opp_lat_pred_overlap_guard_design_20260806.md
  §51参照。

## 実行方法

リポジトリルートから:

```bash
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/build_two_trajectories.py
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/straighten_regions.py
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/consolidate_corner.py
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/build_defense_v3.py
```

いずれも`env/final_ver3/traj_mincurv.csv`は変更せず、`traj_offense_35kmh.csv`/
`traj_defense_25kmh.csv`を新規生成する。

## 2026-08-10追記: wp135-166・wp0周辺の追加編集(ユーザー指示)

`straighten_regions.py`(wp135-166直線化)・`consolidate_corner.py`(wp349-18の
3コブをwp4付近を頂点とする1つのコーナーへ統合)に続き、ユーザーの目視フィードバックを
受けて以下の追加編集を行った(いずれも`traj_mincurv_straightened.csv`に対する
インラインスクリプトでの手直し、再現用の独立スクリプトとしては未整理):

- wp163-169・wp345-3のごく軽いLaplacian平滑化(接続部の小さな段差解消)
- **wp26-41の直線化+Hermite補間ブレンド**(wp24-28・wp39-43で位置・接線方向とも
  連続にする、C1連続の正式なブレンド。単純なLaplacian平滑化の反復は収束せず、
  別箇所に問題を移すだけだったため、この手法へ切替。詳細はdesign_docs
  opp_lat_pred_overlap_guard_design_20260806.md §47.9参照)

## 2026-08-10さらに追記: wp170出口境界キンクの修正・オフェンス生成経路の是正

ユーザーが実走行でWP170付近の「震え」を指摘。原本`traj_mincurv.csv`では
なめらかだが、wp135-166直線化(`straighten()`、位置は一致させるが接線方向を
考慮しない単純chord)の出口境界(wp166/167)で接線が不連続になり、その帳尻が
数点先のwp170のスパイク(近傍比1.7倍)として現れていた。`fix_straighten_exit_kink.py`
を新設し、wp164-172をHermite補間ブレンドで置き換えて解消(`build_defense_v3.py`と
同じ技法)。**当初wp158-178の広いウィンドウを試したが、コーナー内側へ膨らみすぎ
FOOTPRINT_COLLISIONを引き起こした**——taper/ブレンドの窓は広ければ良いわけではなく、
壁との実クリアランスをkaleidoscopeで都度検証しながら狭める必要がある教訓。
全区間スキャンで他に同種の欠陥(意図的に編集していない箇所の曲率変化)が
無いことも確認済み。

**もう一つの発見**: `build_two_trajectories.py`の`TRAJ_PATH`が`traj_mincurv.csv`
(未編集の完全な原本)を指したままだった(2026-08-09時点のまま更新漏れ)。
このスクリプトを実行すると当日の直線化・consolidate・Hermiteブレンドが
全て失われる状態だったため、`traj_offense_35kmh.csv`は現在
`traj_mincurv_straightened.csv`からの直接コピーで生成する運用に変更した
(`build_two_trajectories.py`自体は修正せず現状のまま放置、次に使う際は
TRAJ_PATHの更新が必須)。

**重要な教訓**: `build_defense_v2.py`の`GENTLE_CORNERS`・`protected_points()`は、
ベースジオメトリ(`traj_mincurv_straightened.csv`)を編集するたびに、消滅した
コーナー(直線化で吸収された区間)をリストから外し、新しく直線化した区間を保護点に
追加する必要がある。これを怠ると、ディフェンス生成側の生バイアスや全体平滑化が
既に修正済みの区間を再び乱す(実際に`(34,35,"R")`を消し忘れてkappa+0.17級の
スパイクを再発させた)。ベースジオメトリを変更したら、必ず
`traj_offense_35kmh.csv`(ベースのコピー)と`traj_defense_25kmh.csv`の該当区間の
kappaを突き合わせて一致を確認すること。
