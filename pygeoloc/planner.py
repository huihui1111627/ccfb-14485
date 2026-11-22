"""planner.py — 探测点选取与测量预算分配。

原则
----
* 探测点不是越多越好: 离目标近的探测点圆小而紧, 远的圆又大又容易受路径异常影响。
* 先粗定位: 用少量"分布均匀"的探测点 (全球几何良好) 得到初始区域;
  再按粗区域中心挑选**附近**的探测点精测 (圆更小, 交集收缩更快)。
* 远探测点仍有价值 (约束方向), 但只保留少量作为"几何支撑", 防止区域退化成一条窄带。
"""
from __future__ import annotations

import numpy as np

from .geo import haversine_km, bearing_deg


class ProbePlanner:
    def __init__(self, probes_latlon, rng=None, coarse_ring=True):
        self.lat = np.asarray(probes_latlon[:, 0], np.float64)
        self.lon = np.asarray(probes_latlon[:, 1], np.float64)
        self.rng = rng or np.random.default_rng(0)
        self.coarse_ring = coarse_ring

    # ------------------------------------------------------------ 粗定位
    def coarse_pids(self, n=14):
        """几何上均匀散布的探测点 (最远点采样)。"""
        idx = [int(self.rng.integers(len(self.lat)))]
        d = haversine_km(self.lat[idx[0]], self.lon[idx[0]], self.lat, self.lon)
        for _ in range(1, n):
            k = int(np.argmax(d))
            if k in idx:
                break
            idx.append(k)
            d = np.minimum(d, haversine_km(self.lat[k], self.lon[k], self.lat, self.lon))
        return idx

    # ------------------------------------------------------------ 精测选点
    def refine_pids(self, center_lat, center_lon, radius_km, exclude=(), n_near=8,
                    n_mid=4, r_max=4000.0):
        """围绕粗定位中心选: 最近的若干 + 中距离各方向若干 (保证角度张开)。"""
        d = haversine_km(center_lat, center_lon, self.lat, self.lon)
        brg = bearing_deg(center_lat, center_lon, self.lat, self.lon)
        ex = set(int(e) for e in exclude)
        out = []
        # 近点: 距离最小, 且做方向去重 (同一个方向只取 2 个, 避免基线共线)
        order = np.argsort(d)
        bins = {}
        for p in order:
            p = int(p)
            if p in ex or d[p] > r_max:
                continue
            b = int(brg[p] // 30) % 12
            if bins.get(b, 0) >= 2:
                continue
            bins[b] = bins.get(b, 0) + 1
            out.append(p)
            if len(out) >= n_near:
                break
        # 中距离: 在近点之外, 补齐方向覆盖
        used_b = set(int(brg[p] // 45) % 8 for p in out)
        for p in order:
            p = int(p)
            if p in ex or p in out or d[p] > r_max:
                continue
            if d[p] < np.median(d[out]) if out else 0:
                continue
            b = int(brg[p] // 45) % 8
            if b in used_b:
                continue
            used_b.add(b)
            out.append(p)
            if len(out) >= n_near + n_mid:
                break
        return out

    # ------------------------------------------------------------ 选最近探测点
    def nearest(self, lat, lon, k=1):
        d = haversine_km(lat, lon, self.lat, self.lon)
        return [int(i) for i in np.argsort(d)[:k]]
