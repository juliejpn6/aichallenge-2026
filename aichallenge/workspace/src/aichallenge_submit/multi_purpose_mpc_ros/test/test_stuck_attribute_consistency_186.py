"""Regression test for 186節(2026-07-26): STUCK-BACKUPシャッフル(184節)の実環境初回テストで、
`self._stuck_rear_scan_max_dist_m`(__init__での定義)を`_rear_clearance_m()`内で
`self._stuck_rear_scan_max_dist`(末尾の`_m`が抜けたタイポ)として参照しており、
BACKUP状態に入った瞬間に`AttributeError`でmpc_controllerノードそのものがクラッシュし、
車両が後退ギアに入ったまま二度と動かなくなる(予選走行が2分弱で強制終了)という
重大な実害が発生した。

背景: mpc_controller.pyはrclpy依存のため単体テストで直接importできず、既存の
STUCK/PUSH関連テストはロジックをミラー実装して検証していた。ミラー実装は「意図した
正しい名前」で書かれるため、実ファイル側の属性名タイポはミラーテストでは原理的に
検出できない(897件のテストが全てPASSしていたにも関わらず実環境で即座にクラッシュした)。
この構造的な盲点を埋めるため、実ファイルのソーステキストを静的に走査し、
「self._stuck_*/_r_delta_*という名前で参照されているが、ファイル中のどこにも
代入(self.X = ...)されておらずメソッド定義(def X(...)でもない」属性が
存在しないことを機械的に検証する。
"""
import os
import re

_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "multi_purpose_mpc_ros", "mpc_controller.py")
with open(_SRC_PATH) as _f:
    _SRC = _f.read()

# 意図的に「メソッドのみ・属性として代入されない」名前(誤検知除外リスト)。
# 新しいメソッドを追加した場合はここに追記する(属性代入が無いことを確認した上で)。
_METHOD_ONLY_ALLOWLIST = {
    "_stuck_recovery_complete",
    "_stuck_target_steer",
    "_stuck_update_shuffle_cycle",
}


def _referenced_attrs(prefix):
    return set(re.findall(rf'self\.({re.escape(prefix)}[a-zA-Z0-9_]+)', _SRC))


def _assigned_attrs(prefix):
    return set(re.findall(rf'self\.({re.escape(prefix)}[a-zA-Z0-9_]+)\s*=', _SRC))


def _defined_methods(prefix):
    return set(re.findall(rf'def\s+({re.escape(prefix)}[a-zA-Z0-9_]+)\s*\(', _SRC))


def _undefined_for_prefix(prefix):
    referenced = _referenced_attrs(prefix)
    assigned = _assigned_attrs(prefix)
    methods = _defined_methods(prefix)
    return sorted(referenced - assigned - methods - _METHOD_ONLY_ALLOWLIST)


def test_no_stuck_attribute_typos():
    """184/185/186節で追加した_stuck_*属性に、定義漏れ(=タイポ)がないことを確認する。
    実際に186節で発見された`_stuck_rear_scan_max_dist`(正: `..._m`)級のバグを
    機械的に検出する回帰テスト。"""
    undefined = _undefined_for_prefix("_stuck_")
    assert undefined == [], (
        f"以下の_stuck_*属性は参照されているが定義(代入/メソッド定義)が見つからない: "
        f"{undefined}"
    )


def test_no_r_delta_attribute_typos():
    """177節で追加した_r_delta_*属性についても同様に確認する。"""
    undefined = _undefined_for_prefix("_r_delta_")
    assert undefined == [], (
        f"以下の_r_delta_*属性は参照されているが定義(代入/メソッド定義)が見つからない: "
        f"{undefined}"
    )


def test_rear_scan_max_dist_m_name_matches_definition_exactly():
    """186節で発見・修正した具体的なバグの遡及回帰確認: `_rear_clearance_m()`内の
    参照が、__init__での定義と1文字も違わず一致していること。"""
    assert "self._stuck_rear_scan_max_dist_m" in _SRC
    # 末尾の`_m`が欠けた誤った参照が復活していないことを直接確認する
    assert "self._stuck_rear_scan_max_dist:" not in _SRC
    assert "self._stuck_rear_scan_max_dist " not in _SRC
    assert "self._stuck_rear_scan_max_dist\n" not in _SRC
