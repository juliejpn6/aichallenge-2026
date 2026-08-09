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
- `build_defense_v2.py`: ディフェンス版(`traj_defense_25kmh.csv`)生成の最終版
  (2026-08-10)。`tools/kaleidoscope`(実occupancy_grid_map+実車体寸法での
  clearance検証エンジン)を使い、以下を実装:
  1. ゆるいコーナーのうちタイトコーナー2箇所・S字wp340-40帯に近接しすぎる区間
     (隙間6pt未満)は丸ごとバイアス対象から除外。
  2. 残った区間へtaperなしの生イン側バイアスを適用。
  3. 全体にLaplacian平滑化(kaleidoscopeのsmooth_all_pointsと同一アルゴリズム)。
  4. `kaleidoscope.trajectory_clearance.validate_clearance`で実マップに対し検証、
     `is_safe`になるまで振幅を段階的に下げて2-4を再試行。

## 実行方法

リポジトリルートから:

```bash
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/build_two_trajectories.py
python3 aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/scripts/traj_tools/build_defense_v2.py
```

いずれも`env/final_ver3/traj_mincurv.csv`は変更せず、`traj_offense_35kmh.csv`/
`traj_defense_25kmh.csv`を新規生成する。
