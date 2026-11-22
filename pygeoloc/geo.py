"""geo.py — 球面几何工具 (纯 NumPy 实现)。"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0088          # 地球平均半径 (km)
V_LIGHT_KM_S = 299792.458       # 真空光速 (km/s)
V_FIBER_KM_S = 200000.0         # 光纤中光速约 2/3 c (~200 km/ms)
MS_PER_KM_AT_C = 1000.0 / V_LIGHT_KM_S   # 真空光速下 1 km 单程时延 (ms) ≈ 0.003336
MS_PER_KM_FIBER = 1000.0 / V_FIBER_KM_S  # 光纤中 1 km 单程时延 (ms) ≈ 0.005


def wrap_lon(d):
    """把经度差/经度折叠到 [-180, 180)。"""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _prep(lat1, lon1, lat2, lon2):
    return np.broadcast_arrays(
        np.asarray(lat1, np.float64), np.asarray(lon1, np.float64),
        np.asarray(lat2, np.float64), np.asarray(lon2, np.float64))


def haversine_km(lat1, lon1, lat2, lon2):
    """大圆距离 (km)，支持任意广播。"""
    lat1, lon1, lat2, lon2 = _prep(lat1, lon1, lat2, lon2)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(wrap_lon(lon2 - lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """从点1指向点2的初始方位角 (0=北, 90=东)。"""
    lat1, lon1, lat2, lon2 = _prep(lat1, lon1, lat2, lon2)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(wrap_lon(lon2 - lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination_point(lat, lon, bearing, dist_km):
    """沿方位角走 dist_km 后的经纬度。"""
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    brg = np.radians(np.asarray(bearing, np.float64))
    d = np.asarray(dist_km, np.float64) / R_EARTH_KM
    p1 = np.radians(lat)
    sin_p2 = np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(brg)
    p2 = np.arcsin(np.clip(sin_p2, -1.0, 1.0))
    l2 = np.radians(lon) + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(p1),
                                      np.cos(d) - np.sin(p1) * sin_p2)
    return np.degrees(p2), wrap_lon(np.degrees(l2))


def fast_dist_km(lat0, lon0, lat, lon):
    """以两点平均纬度为基准的等距圆柱近似距离 (km)。

    这是小区域/中距离上常用的近似, 在 <3000km 范围内相对误差 <0.15%,
    比"以一点纬度为准"的写法好一个量级。用于网格筛选;
    最终区域/验收判定仍用 haversine。
    """
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    latm = np.radians(0.5 * (lat + lat0))
    kx = R_EARTH_KM * np.cos(latm) * np.pi / 180.0
    ky = R_EARTH_KM * np.pi / 180.0
    dx = wrap_lon(lon - lon0) * kx
    dy = (lat - lat0) * ky
    return np.hypot(dx, dy)


def circle_polyline(lat, lon, radius_km, n=64):
    """测地圆的多边形近似，返回 (lon[], lat[])。"""
    brg = np.linspace(0.0, 360.0, n + 1)[:-1]
    la, lo = destination_point(lat, lon, brg, radius_km)
    la = np.append(la, la[0]); lo = np.append(lo, lo[0])
    return lo, la


class LocalFrame:
    """以某点为中心的局部平面近似 (km)。用于粗网格/快速距离，最终判定仍用 haversine。"""

    def __init__(self, lat0, lon0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.kx = R_EARTH_KM * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self.ky = R_EARTH_KM * np.pi / 180.0

    def to_xy(self, lat, lon):
        x = wrap_lon(np.asarray(lon) - self.lon0) * self.kx
        y = (np.asarray(lat) - self.lat0) * self.ky
        return x, y

    def to_lonlat(self, x, y):
        lon = wrap_lon(np.asarray(x) / self.kx + self.lon0)
        lat = np.asarray(y) / self.ky + self.lat0
        return lat, lon

    def dist(self, lat, lon, lat2, lon2):
        x1, y1 = self.to_xy(lat, lon)
        x2, y2 = self.to_xy(lat2, lon2)
        return np.hypot(x1 - x2, y1 - y2)


def local_xy(lat0, lon0, lat, lon):
    """便捷函数: 以 (lat0,lon0) 为原点的 km 坐标。"""
    return LocalFrame(lat0, lon0).to_xy(lat, lon)


def lonlat_to_mercator(lon, lat):
    """Web Mercator 归一化坐标 x∈[0,1], y∈[0,1] (y 向下)。供前端地图使用。"""
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    x = (lon + 180.0) / 360.0
    lat_c = np.clip(lat, -85.05112878, 85.05112878)
    y = (1.0 - np.log(np.tan(np.pi / 4 + np.radians(lat_c) / 2)) / np.pi) / 2.0
    return x, y


def mercator_to_lonlat(x, y):
    lon = np.asarray(x, np.float64) * 360.0 - 180.0
    lat = np.degrees(2.0 * np.arctan(np.exp(np.pi * (1.0 - 2.0 * np.asarray(y, np.float64)))) - np.pi / 2)
    return lon, lat


def polygon_area_km2(lon, lat):
    """球面多边形面积 (km²)。经度需已展开为连续序列。"""
    lon = np.radians(np.asarray(lon, np.float64))
    lat = np.radians(np.asarray(lat, np.float64))
    if len(lon) > 1 and lon[0] == lon[-1] and lat[0] == lat[-1]:
        lon = lon[:-1]; lat = lat[:-1]
    if len(lon) < 3:
        return 0.0
    lo2 = np.roll(lon, -1); la2 = np.roll(lat, -1)
    s = np.sum((lo2 - lon) * (2.0 + np.sin(lat) + np.sin(la2)))
    return abs(s) * R_EARTH_KM ** 2 / 2.0


def point_in_ring(lon, lat, ring_lon, ring_lat):
    """单点是否在环内 (偶奇规则)。环经度已展开; 点经度会相对环中位经度折叠。"""
    lon = float(lon); lat = float(lat)
    rl = np.asarray(ring_lon, np.float64)
    ra = np.asarray(ring_lat, np.float64)
    if len(rl) > 1 and rl[0] == rl[-1]:
        rl = rl[:-1]; ra = ra[:-1]
    c0 = float(np.median(rl))
    lon = c0 + float(wrap_lon(lon - c0))
    x1, y1 = rl, ra
    x2, y2 = np.roll(rl, -1), np.roll(ra, -1)
    cond = (y1 > lat) != (y2 > lat)
    dy = np.where(y2 - y1 == 0.0, 1e-300, y2 - y1)
    xint = x1 + (lat - y1) * (x2 - x1) / dy
    return bool(np.count_nonzero(cond & (xint > lon)) % 2)


def points_in_rings(points_lon, points_lat, rings):
    """多点 vs 多个环 (每个环为 (lon[], lat[]) 已是连续经度)。返回 bool 数组 (偶奇并集)。"""
    plon = np.asarray(points_lon, np.float64)
    plat = np.asarray(points_lat, np.float64)
    out = np.zeros(plon.shape, dtype=bool)
    for rl, ra in rings:
        rl = np.asarray(rl, np.float64); ra = np.asarray(ra, np.float64)
        x1, y1 = rl, ra
        x2, y2 = np.roll(rl, -1), np.roll(ra, -1)
        # 每个点单独处理 (环的经度跨度通常 < 360; 点经度折叠到环中位附近)
        c0 = float(np.median(rl))
        lon = c0 + wrap_lon(plon - c0)
        cond = (y1[None, :] > plat[:, None]) != (y2[None, :] > plat[:, None])
        dy = np.where(y2 - y1 == 0.0, 1e-300, y2 - y1)
        xint = x1[None, :] + (plat[:, None] - y1[None, :]) * (x2 - x1)[None, :] / dy[None, :]
        out ^= (np.count_nonzero(cond & (xint > lon[:, None]), axis=1) % 2).astype(bool)
    return out
