#!/bin/bash
# 25km/hディフェンストラジェクトリ(traj_defense_25kmh.csv)をkaleidoscopeで開く。
# コンテナ内(make autoware-bash後)で実行すること。
set -e
cd "$(dirname "$0")"
python3 -m kaleidoscope \
  --trajectory ../../env/final_ver3/traj_defense_25kmh.csv \
  --osm ../../../aichallenge_submit_launch/map/lanelet2_map.osm \
  --circular
