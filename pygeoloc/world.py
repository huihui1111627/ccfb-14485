"""world.py — 离线世界数据: 陆地栅格、城市人口、机场(三字码)与网络节点。"""
from __future__ import annotations

import os
import numpy as np

from .geo import haversine_km, R_EARTH_KM

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _load(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            '缺少数据文件 %s。请先运行 tools/build_data.py 生成 (见 README)。' % p)
    return np.load(p, allow_pickle=True)


class LandMask:
    """陆地多边形集合 + 点包含判定 (带经纬度包围盒剔除)。"""

    def __init__(self, land_npz=None):
        d = _load('world_land.npz') if land_npz is None else np.load(land_npz)
        pts = np.asarray(d['pts'], np.float64)
        lens = np.asarray(d['lens'], np.int64)
        off = np.concatenate([[0], np.cumsum(lens)])
        self.rings = []
        bbs = np.empty((len(lens), 4), np.float64)
        for i in range(len(lens)):
            r = pts[off[i]:off[i + 1]]
            self.rings.append((r[:, 0].copy(), r[:, 1].copy()))
            bbs[i] = (r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max())
        self.bbs = bbs

    def candidates(self, lon, lat):
        b = self.bbs
        return np.nonzero((b[:, 0] <= lon) & (b[:, 1] >= lon) &
                          (b[:, 2] <= lat) & (b[:, 3] >= lat))[0]

    def contains(self, lon, lat):
        from .geo import point_in_ring
        for i in self.candidates(lon, lat):
            rl, ra = self.rings[i]
            if point_in_ring(lon, lat, rl, ra):
                return True
        return False

    def contains_many(self, lon, lat, return_frac=False):
        """cKDTree 加速的批量判定: 返回每个点是否在陆地 (经度按 360 模后取最近环)。"""
        from scipy.spatial import cKDTree
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        allx = np.concatenate([r[0] for r in self.rings])
        ally = np.concatenate([r[1] for r in self.rings])
        ring_id = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(self.rings)])
        # 把环点复制到经度 ±360 上, 使得查询点经度折叠后仍能找到邻近环
        tree = cKDTree(np.column_stack([allx, ally]))
        out = np.zeros(len(lon), bool)
        # 先用包围盒粗筛
        b = self.bbs
        cand = ((b[None, :, 0] <= lon[:, None]) & (b[None, :, 1] >= lon[:, None]) &
                (b[None, :, 2] <= lat[:, None]) & (b[None, :, 3] >= lat[:, None]))
        from .geo import point_in_ring
        any_cand = cand.any(axis=1)
        idxs = np.nonzero(any_cand)[0]
        for k in idxs:
            for i in np.nonzero(cand[k])[0]:
                rl, ra = self.rings[i]
                if point_in_ring(lon[k], lat[k], rl, ra):
                    out[k] = True
                    break
        return out


class LandRaster:
    """0.05° 全球陆地栅格 (按位打包)。查询为 O(1), 适合海量点。"""

    def __init__(self):
        d = _load('land_mask.npz')
        self.shape = tuple(int(x) for x in d['shape'])
        self.res = float(np.asarray(d['res']).ravel()[0])
        n = self.shape[0] * self.shape[1]
        self.mask = np.unpackbits(np.asarray(d['bits']))[:n].reshape(self.shape).astype(bool)

    def lookup(self, lon, lat):
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        j = np.clip(((lon + 180.0) / self.res).astype(np.int64), 0, self.shape[1] - 1)
        i = np.clip(((lat + 90.0) / self.res).astype(np.int64), 0, self.shape[0] - 1)
        return self.mask[i, j]

    def land_fraction_km2(self, lon, lat, radius_km, n=9):
        """圆内陆地面积占比 (在圆内采样 n x n 网格)。"""
        from .geo import destination_point
        if radius_km <= 0:
            return float(self.lookup(lon, lat))
        offs = np.linspace(-1, 1, n)
        gx, gy = np.meshgrid(offs, offs)
        sel = gx * gx + gy * gy <= 1.0
        pts_lon = []
        pts_lat = []
        for fx, fy in zip(gx[sel], gy[sel]):
            la, lo = destination_point(lat, lon, np.degrees(np.arctan2(fy, fx)) % 360.0,
                                       np.hypot(fx, fy) * radius_km)
            pts_lon.append(float(lo)); pts_lat.append(float(la))
        return float(self.lookup(np.array(pts_lon), np.array(pts_lat)).mean())


class Cities:
    """城市点: 经纬度/人口/最近机场索引/名称。"""

    def __init__(self):
        d = _load('world_cities.npz')
        self.lat = np.asarray(d['lat'], np.float64)
        self.lon = np.asarray(d['lon'], np.float64)
        self.pop = np.asarray(d['pop'], np.float64)
        self.c2a = np.asarray(d['c2a'], np.int64)
        self.name = [str(x) for x in d['name']]
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            x, y, z = self._xyz()
            self._tree = cKDTree(np.column_stack([x, y, z]))
        return self._tree

    def _xyz(self):
        la, lo = np.radians(self.lat), np.radians(self.lon)
        return np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)

    def near(self, lat, lon, radius_km, max_n=None):
        """半径内城市 (精确 haversine 过滤)。返回索引数组。"""
        x, y, z = self._xyz()
        la, lo = np.radians(lat), np.radians(lon)
        q = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        chord = 2.0 * np.sin(min(radius_km / R_EARTH_KM / 2.0, np.pi / 2))
        idx = self.tree.query_ball_point(q, r=chord * 1.0001)
        idx = np.asarray(sorted(idx), np.int64)
        if len(idx):
            d = haversine_km(lat, lon, self.lat[idx], self.lon[idx])
            idx = idx[d <= radius_km]
        if max_n is not None and len(idx) > max_n:
            idx = idx[np.argsort(-self.pop[idx])[:max_n]]
        return idx


class Airports:
    """机场 (IATA 三字码): 经纬度、大型标志、城市名。"""

    def __init__(self):
        d = _load('world_airports.npz')
        self.lat = np.asarray(d['lat'], np.float64)
        self.lon = np.asarray(d['lon'], np.float64)
        self.big = np.asarray(d['big'], np.float64)
        self.code = [str(x) for x in d['code']]
        self.municipality = [str(x) for x in d['municipality']]
        self.country = [str(x) for x in d['country']]
        self.index = {c: i for i, c in enumerate(self.code)}
        x, y, z = self._xyz()
        from scipy.spatial import cKDTree
        self.tree = cKDTree(np.column_stack([x, y, z]))

    def _xyz(self):
        la, lo = np.radians(self.lat), np.radians(self.lon)
        return np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)

    def nearest(self, lat, lon):
        la, lo = np.radians(lat), np.radians(lon)
        q = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        _, i = self.tree.query(q)
        return int(i)

    def near(self, lat, lon, radius_km):
        la, lo = np.radians(lat), np.radians(lon)
        q = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        chord = 2.0 * np.sin(min(radius_km / R_EARTH_KM / 2.0, np.pi / 2))
        idx = np.asarray(sorted(self.tree.query_ball_point(q, r=chord * 1.0001)), np.int64)
        if len(idx):
            d = haversine_km(lat, lon, self.lat[idx], self.lon[idx])
            idx = idx[d <= radius_km]
        return idx


class World:
    """惰性加载的全局数据容器。"""

    def __init__(self, with_land=True):
        self.airports = Airports()
        self.cities = Cities()
        self._land = None
        self._with_land = with_land

    @property
    def land(self):
        """陆地栅格 (优先) 或多边形集合。"""
        if self._land is None:
            if not self._with_land:
                return None
            try:
                self._land = LandRaster()
            except FileNotFoundError:
                self._land = LandMask()
        return self._land


_WORLD = None


def get_world(with_land=True):
    global _WORLD
    if _WORLD is None:
        _WORLD = World(with_land=with_land)
    return _WORLD
