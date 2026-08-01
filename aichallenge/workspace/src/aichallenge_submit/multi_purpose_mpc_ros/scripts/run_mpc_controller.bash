#!/bin/bash
# shellcheck disable=SC1091
source "$(ros2 pkg prefix multi_purpose_mpc_ros)/.venv/bin/activate"
# 2026-08-01追加(262節続報、判定基準改訂+work_cpu計装Phase 3、BLAS/OMPスレッド監査):
#   numpy importより前にBLASワーカースレッド数を制限する必要があるため、Pythonの
#   config.yaml読み込みを待たずこのシェル起動スクリプトで環境変数として設定する。
#   監査結果(コンテナ内numpy.show_config()): BLASバックエンドはOpenBLAS
#   (ビルド時 MAX_THREADS=2, USE_OPENMP=無効=pthreadsベース, NO_AFFINITY=1)。
#   OMP_NUM_THREADS等は未設定=既定でmin(cpu_count, 2)=2ワーカースレッドを使う。
#   MPCのQP規模(N=20)ではこの並列化がほぼ効果を持たない一方、cpu_affinity
#   (262節続報Phase 4)で専有コアを絞った際にBLASワーカースレッド同士がSMT
#   兄弟スレッドを奪い合う自己競合を招くことがC3実測(work>予算割合が無介入時
#   より悪化)で確認された。既定でスレッド数を1へ制限する。
#   MPC_BLAS_THREAD_LIMIT=0を明示指定すると制限をスキップ(旧来のOpenBLAS既定
#   挙動=ビルド時上限の2スレッドまで)できる(docker-compose.ymlのenvironment
#   経由で上書き可能)。
if [ "${MPC_BLAS_THREAD_LIMIT:-1}" != "0" ]; then
    export OMP_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"
    export OPENBLAS_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"
    export MKL_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"
fi
python3 "$(ros2 pkg prefix multi_purpose_mpc_ros)/lib/multi_purpose_mpc_ros/mpc_controller" "$@"
