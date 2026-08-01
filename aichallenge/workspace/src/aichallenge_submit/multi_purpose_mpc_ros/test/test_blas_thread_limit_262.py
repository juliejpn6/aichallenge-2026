"""262節続報(2026-08-01、判定基準改訂+work_cpu計装Phase 3): BLAS/OMPスレッド監査と制限。

背景: コンテナ内でnumpy.show_config()を実測した結果、BLASバックエンドは
OpenBLAS(ビルド時MAX_THREADS=2、USE_OPENMP=無効=pthreadsベース、NO_AFFINITY=1)
であり、OMP_NUM_THREADS等の環境変数は未設定(既定でmin(cpu_count,2)=2ワーカー
スレッドを使う)ことが判明した。C3実測(cpu_affinity=[2,3]、論理2コアへ固定)で
work>予算割合が無介入時より悪化したことから、狭いコア集合下でBLASワーカー
スレッド同士がSMT兄弟スレッドを奪い合う自己競合を疑い、run_mpc_controller.bash
でnumpy importより前にOMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADSを
既定1へ制限するようにした(MPC_BLAS_THREAD_LIMIT=0で無効化可能)。

40Hz・cpu_affinityなしでの単体効果測定では over_pct 0.000%→0.018%(3/16800
周期)とわずかに悪化したが、誤差範囲内の小さな差であり単独では結論が出ない。
本来の目的(cpu_affinityで狭いコアに固定した際の自己競合緩和)はC4実験
(cpu_affinity+スレッド制限の組み合わせ)で判定する。
"""
import os

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "run_mpc_controller.bash")
with open(_SCRIPT_PATH) as _f:
    _SCRIPT_SRC = _f.read()


def test_blas_thread_limit_set_before_python_invocation():
    """python3呼び出しより前(=numpy importより前)に環境変数がexportされること。"""
    idx_export = _SCRIPT_SRC.index('export OMP_NUM_THREADS')
    idx_python = _SCRIPT_SRC.index('python3 "$(ros2 pkg prefix')
    assert idx_export < idx_python


def test_blas_thread_limit_exports_all_three_backends():
    idx = _SCRIPT_SRC.index('if [ "${MPC_BLAS_THREAD_LIMIT:-1}" != "0" ]; then')
    idx_end = _SCRIPT_SRC.index('fi', idx)
    snippet = _SCRIPT_SRC[idx:idx_end]
    assert 'export OMP_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"' in snippet
    assert 'export OPENBLAS_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"' in snippet
    assert 'export MKL_NUM_THREADS="${MPC_BLAS_THREAD_LIMIT:-1}"' in snippet


def test_blas_thread_limit_defaults_to_1_when_unset():
    """MPC_BLAS_THREAD_LIMIT未設定時のデフォルト値(${VAR:-1}のシェル構文)が
    1であることを確認する(既定ON、この規模の行列では並列が逆効果という判断)。"""
    assert '${MPC_BLAS_THREAD_LIMIT:-1}' in _SCRIPT_SRC


def test_blas_thread_limit_can_be_disabled_via_env_var():
    """MPC_BLAS_THREAD_LIMIT=0を明示指定するとexportブロックがスキップされ、
    OpenBLASの従来既定動作(ビルド時上限2スレッドまで)へ戻せることを確認する。"""
    assert 'if [ "${MPC_BLAS_THREAD_LIMIT:-1}" != "0" ]; then' in _SCRIPT_SRC


def mirror_effective_thread_limit(env_value):
    """シェルの${MPC_BLAS_THREAD_LIMIT:-1}構文のミラー(未設定/空文字なら1)。"""
    return env_value if env_value else "1"


def test_effective_limit_mirror_default():
    assert mirror_effective_thread_limit(None) == "1"
    assert mirror_effective_thread_limit("") == "1"


def test_effective_limit_mirror_explicit_override():
    assert mirror_effective_thread_limit("4") == "4"
