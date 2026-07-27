#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相手車の「位置別 速度包絡線」をオンライン学習する(ROS非依存コア)。

思想:相手は決定論的制御器で毎周ほぼ同じ挙動をするため、"意図(加速/減速)"は
  意図 = f(コース位置) として学習できる。リアルタイムの加速度は2階微分でノイズ過大
  (実測 SNR<1)なので測らず、位置別の速度包絡線 v(s) を貯め、その空間微分
  a(s) = v·dv/ds で"予測的・低ノイズ"な加速度を得る。

- 索引 = reference_path の wp_id(= 使用中トラジェクトリを resolution で再サンプルしたもの)。
- 値 = max包絡線(上げるだけ)。相手が追い越し/衝突/干渉で実力を出せない時は必ず遅く
  なるので、max は真の実力(本気の値)だけを拾い、干渉に堅牢に収束する。
- スパイク除去 = 物理天井超えを棄却(トラッカー側でも位置跳び→速度0リセット済み)。
- 40Hz制御ループには一切触れない:更新は V2X レート、参照は O(1) 配列読みのみ。
"""
from typing import Dict, List, Optional
import numpy as np


class OpponentSpeedMap:
    def __init__(self, n_wp: int, s_cum, v_ceiling: float = 12.0,
                 min_count: int = 1, smooth_k: int = 5, a_clip: float = 6.0):
        self.n = int(n_wp)
        s = np.asarray(s_cum, dtype=float) if s_cum is not None else np.arange(self.n) * 0.6
        # 各wpの区間長[m](空間微分・先読みに使用)。末尾は近傍値で補う。
        seg = np.diff(s)
        self.seg = np.concatenate([seg, seg[-1:]]) if len(seg) else np.ones(self.n) * 0.6
        if len(self.seg) < self.n:
            self.seg = np.pad(self.seg, (0, self.n - len(self.seg)), mode="edge")
        self.v_ceiling = float(v_ceiling)     # [m/s] 物理天井(43km/h相当)。これ超はスパイクとして棄却
        self.min_count = int(min_count)
        self.smooth_k = int(smooth_k)
        self.a_clip = float(a_clip)           # [m/s²] a(s)の物理クランプ(疎データ・段差での非物理スパイク抑制)
        self._v: Dict[str, np.ndarray] = {}   # vid -> 速度包絡線[m/s]
        self._c: Dict[str, np.ndarray] = {}   # vid -> 観測回数
        self._cap: Dict[str, float] = {}      # vid -> 全体最高速(スカラ)
        self._lat_sum: Dict[str, np.ndarray] = {}  # vid -> 横位置latの累積(平均=走行ライン学習)
        self._lat_cnt: Dict[str, np.ndarray] = {}  # vid -> latの観測回数
        self._dirty: Dict[str, bool] = {}
        self._cache: Dict[str, tuple] = {}    # vid -> (v_smooth, accel) 遅延キャッシュ

    def _ensure(self, vid: str) -> None:
        if vid not in self._v:
            self._v[vid] = np.zeros(self.n)
            self._c[vid] = np.zeros(self.n, dtype=np.int32)
            self._cap[vid] = 0.0
            self._lat_sum[vid] = np.zeros(self.n)
            self._lat_cnt[vid] = np.zeros(self.n, dtype=np.int32)
            self._dirty[vid] = True

    # ---- 更新(V2Xレート, 40Hzループ外) ----
    def update(self, vid: str, wp_id: int, v_long: float, settled: bool = True,
               lat: Optional[float] = None) -> None:
        """速度=max包絡線(上げるだけ)。lat=走行ライン(平均)。settled=トラッカー窓が満杯。"""
        self._ensure(vid)
        w = int(wp_id) % self.n
        self._c[vid][w] += 1                              # 観測済み(有効性判定用)
        if not settled:
            return                                         # 未成熟は速度・ラインとも学習しない
        # 走行ライン学習: 相手が各wpで実際に走る横位置(平均)。「どちら側が空くか」の実測根拠。
        #   相手がコーナー外に膨らむ挙動もこの平均に自然に入る。|lat|>6mはコース外=射影破綻として棄却。
        if lat is not None and abs(lat) <= 6.0:
            self._lat_sum[vid][w] += float(lat)
            self._lat_cnt[vid][w] += 1
        if not (0.0 <= v_long <= self.v_ceiling):
            return                                         # 速度スパイクは包絡線に入れない
        if v_long > self._v[vid][w]:
            self._v[vid][w] = v_long
            self._dirty[vid] = True
        if v_long > self._cap[vid]:
            self._cap[vid] = v_long

    def _rebuild(self, vid: str) -> None:
        v = self._v[vid]
        obs = self._c[vid] > 0
        vs = v.copy()
        if obs.sum() >= 2:
            idx = np.arange(self.n)
            vs = np.interp(idx, idx[obs], v[obs], period=self.n)   # 未観測は近傍で周回補間
            k = max(1, self.smooth_k)
            ker = np.ones(k) / k
            pad = np.concatenate([vs[-k:], vs, vs[:k]])            # 周回平滑
            vs = np.convolve(pad, ker, mode="same")[k:-k]
        # a(s) = v·dv/ds(中央差分, 周回)
        dv = np.roll(vs, -1) - np.roll(vs, 1)
        ds = np.roll(self.seg, -1) + self.seg
        ds = np.where(ds > 1e-3, ds, 1e-3)
        acc = np.clip(vs * dv / ds, -self.a_clip, self.a_clip)  # 物理範囲にクランプ
        self._cache[vid] = (vs, acc)
        self._dirty[vid] = False

    def _g(self, vid: str):
        if vid not in self._v:
            return None
        if self._dirty.get(vid, True):
            self._rebuild(vid)
        return self._cache[vid]

    # ---- 参照(40Hz, O(1)) ----
    def has_data(self, vid: str, wp_id: int) -> bool:
        return vid in self._c and self._c[vid][int(wp_id) % self.n] >= self.min_count

    def v_pred(self, vid: str, wp_id: int) -> Optional[float]:
        g = self._g(vid)
        return None if g is None else float(g[0][int(wp_id) % self.n])

    def v_pred_ahead(self, vid: str, wp_id: int, n_ahead: int):
        g = self._g(vid)
        if g is None:
            return None
        idx = (int(wp_id) + np.arange(int(n_ahead))) % self.n
        return g[0][idx]

    def accel(self, vid: str, wp_id: int) -> Optional[float]:
        g = self._g(vid)
        return None if g is None else float(g[1][int(wp_id) % self.n])

    def cap(self, vid: str) -> float:
        return float(self._cap.get(vid, 0.0))

    def lat_mean(self, vid: str, wp_id: int) -> Optional[float]:
        """相手の学習済み走行ライン(wp別の平均横位置)。未学習は None。"""
        if vid not in self._lat_cnt:
            return None
        w = int(wp_id) % self.n
        c = int(self._lat_cnt[vid][w])
        return float(self._lat_sum[vid][w] / c) if c > 0 else None

    def v_seg_mean(self, vid: str, wp_id: int, n_ahead: int) -> Optional[float]:
        """先n_ahead区間の平滑速度の平均[m/s]。パス所要時間(closing)の見積りに使う。"""
        g = self._g(vid)
        if g is None:
            return None
        idx = (int(wp_id) + np.arange(max(1, int(n_ahead)))) % self.n
        return float(np.mean(g[0][idx]))

    # ---- 検証出力 ----
    def to_flat(self) -> List[float]:
        """Float32MultiArray 用 flat。レイアウト(2026-07-04 lat追加でv2):
        [n_wp, n_vid, {vid番号, v_env[n], count[n], accel[n], lat_mean[n](未学習=99)} × n_vid]"""
        vids = sorted(self._v.keys())
        out = [float(self.n), float(len(vids))]
        for vid in vids:
            g = self._g(vid)
            digits = "".join(ch for ch in vid if ch.isdigit())
            out.append(float(int(digits) if digits else 0))
            out.extend(self._v[vid].astype(float).tolist())
            out.extend(self._c[vid].astype(float).tolist())
            out.extend((g[1] if g is not None else np.zeros(self.n)).astype(float).tolist())
            cnt = np.maximum(self._lat_cnt[vid], 1)
            lat = self._lat_sum[vid] / cnt
            lat[self._lat_cnt[vid] == 0] = 99.0            # 未学習マーカ
            out.extend(lat.astype(float).tolist())
        return out

    def summary(self) -> str:
        """ログ用1行要約:vidごと cap[km/h]・速度学習wp数・ライン学習wp数。"""
        parts = []
        for vid in sorted(self._v.keys()):
            learned = int((self._c[vid] > 0).sum())
            lat_n = int((self._lat_cnt[vid] > 0).sum())
            parts.append(f"{vid}:cap={self.cap(vid) * 3.6:.1f}km/h wp={learned}/{self.n} lat={lat_n}")
        return " ".join(parts) if parts else "(no opponents yet)"

    def vids(self) -> List[str]:
        return sorted(self._v.keys())
