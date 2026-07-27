"""Regression guard for config.yaml vs. mpc_controller.py default drift (85節, 2026-07-16).

Background: 64/72節 changed the CODE-side default for `giveup_space_m` from
along_lane_need(1.85m) to along_min_width(1.45m), to fix an "engage then
immediately giveup between 1.45-1.84m" loop (verified via pytest and design_docs
at the time). However `config/config.yaml`'s `lat_ttc:` section hardcoded
`giveup_space_m: 1.85` explicitly, and mpc_controller.py's `_lget()` helper reads
`lat_ttc_cfg` attributes (i.e. the YAML value) BEFORE falling back to the code
default — so the verified fix silently never took effect in any real run since
64/72節, discovered only in 85節's horizontal-sweep audit (2026-07-16).

This test parses config.yaml directly (not mpc_controller.py, which cannot be
imported standalone due to rclpy/autoware module-scope imports) and asserts the
two related constants stay consistent, so a future edit to one YAML value
without the other silently re-introduces this exact class of bug.
"""
import os

import pytest
import yaml

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml")


def _load_lat_ttc_config():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["lat_ttc"]


def test_giveup_space_m_matches_along_min_width_not_stale_along_lane_need():
    """回帰の核心: giveup_space_mが64/72節の意図通りalong_min_width(1.45m)に
    なっており、85節で発見した古いalong_lane_need(1.85m)のハードコードへ
    戻っていないことを確認する。"""
    lat_ttc = _load_lat_ttc_config()
    assert lat_ttc["giveup_space_m"] == 1.45


def test_switchback_space_m_preserves_0_5m_margin_over_giveup_space_m():
    """回帰: switchback_space_mは常にgiveup_space_m+0.5m(反転マージン)を
    維持する。giveup_space_mだけを更新してswitchback_space_mを更新し忘れると、
    このテストが失敗する(85節で発見した「片方だけ直す」ミスの再発防止)。"""
    lat_ttc = _load_lat_ttc_config()
    assert lat_ttc["switchback_space_m"] == pytest.approx(
        lat_ttc["giveup_space_m"] + 0.5)


def test_cleared_space_m_still_matches_along_min_width_physical_floor():
    """回帰: cleared_space_m(真横到達後の緩和閾値)もalong_min_width(1.45m)
    のままであることを確認する(85節監査ではこちらは問題なしと確認済み)。"""
    lat_ttc = _load_lat_ttc_config()
    assert lat_ttc["cleared_space_m"] == 1.45


def test_lat_ttc_enabled_key_removed_no_longer_read_by_code():
    """回帰(2026-07-17): 旧`enabled`キー(ロールバック用キルスイッチ)は
    mpc_controller.py側の読み取りごと削除したため、config.yamlにも残っていない
    ことを確認する(死んだ設定キーが復活しないようにするための回帰テスト)。"""
    lat_ttc = _load_lat_ttc_config()
    assert "enabled" not in lat_ttc
