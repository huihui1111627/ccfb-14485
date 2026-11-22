"""region.py — 时延圆的交集区域、空交集诊断与放宽、人口加权后验。

每个约束: "目标到探测点 i 的距离 <= R_i" —— 球面圆盘 (cap)。
区域: 所有圆盘的交集 ∩ 陆地。
后验: 在交集网格上, 用每个探测点的时延似然 × 人口密度先验, 得到"区域内哪里更可能"。
空交集: 说明至少一个测量不可信 (持续拥塞/路径异常/标定失效)。用候选点投票找出
        与大多数约束矛盾的探测点, 先放宽、再不行就剔除。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geo import (haversine_km, fast_dist_km, destination_point, wrap_lon,
                  circle_polyline, R_EARTH_KM)


@dataclass
class Constraint:
    pid: int
    lat: float
    lon: float
    radius_km: float
    rtt: float = np.nan
    a: float = 0.0
    b: float = 0.0
    logt_x: np.ndarray | None = None     # 标定残差 log(rtt/min-model) 的分布网格
    logt_p: np.ndarray | None = None
    kind: str = 'probe'
    trusted: bool = True
    note: str = ''
    label: str = ''
    grad: float = 1.0       # 保守余量倍数 (1.0 = 标定线原值; >1 表示额外放宽)
    corner_safe: bool = True   # 网格判定是否按单元最远角 (保守); False 按中心 (仅用于点估计)

    def dist_km(self, lat, lon):
        return haversine_km(self.lat, self.lon, lat, lon)

    def distance_bound(self, rtt_ms):
        """与 CalibLine.distance_bound 同一口径的距离上界 (线性区 + 真空光速兜底)。"""
        from .calib import BOUND_MARGIN, SLOPE_HARD_MIN
        rtt_ms = np.asarray(rtt_ms, np.float64)
        lin = np.maximum(rtt_ms - self.a, 0.0) / max(self.b, 1e-12) * BOUND_MARGIN
        phys = np.maximum(rtt_ms, 0.0) / SLOPE_HARD_MIN
        trust = rtt_ms >= self.a + 4.0
        return np.where(trust, lin, phys)

    def soft(self, factor=1.3):
        c = Constraint(**{**self.__dict__})
        c.radius_km = self.radius_km * factor
        c.trusted = True
        return c

    def value_at(self, lat, lon):
        """f_i(p) = c_i - b_i * d_i(p), 其中 c_i = rtt_i - a_i (线性区)。
        目标所有约束共享同一固定开销 E, 因此 E <= min_i f_i(p), 且真值点处取等号。"""
        if self.b <= 0:
            return np.inf
        return self.rtt - self.a - self.b * haversine_km(self.lat, self.lon, lat, lon)


@dataclass
class RegionResult:
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    mask: np.ndarray                     # (ny, nx) bool
    posterior: np.ndarray                # (ny, nx) float, 未归一化 (区域内)
    area_km2: float
    polygon: list = field(default_factory=list)   # 每个环 [(lon,lat), ...]
    confidence: float = 0.0
    best_lat: float = np.nan
    best_lon: float = np.nan
    n_cells: int = 0
    cell_km: float = 0.0
    empty: bool = False
    diagnostics: list = field(default_factory=list)
    peak_lat: float = np.nan
    peak_lon: float = np.nan

    def region_contains_point(self, lat, lon):
        """连续点是否在网格区域内 (含一个单元宽度的边界容差, 与角点保守判定一致)。"""
        if self.empty:
            return False
        ny = len(self.lat_edges) - 1
        nx = len(self.lon_edges) - 1
        i0 = int(np.clip(np.searchsorted(self.lat_edges, lat) - 1, 0, ny - 1))
        j0 = int(np.clip(np.searchsorted(self.lon_edges, lon) - 1, 0, nx - 1))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i = i0 + di; j = j0 + dj
                if 0 <= i < ny and 0 <= j < nx and self.mask[i, j]:
                    # 该单元四角都满足约束? 用区域自带的约束不可见, 退化为单元命中
                    return True
        return False

    def contains(self, lat, lon):
        """真实位置是否落在区域内 (网格单元判定; 用于验收)。"""
        if self.empty:
            return False
        i = np.searchsorted(self.lat_edges, lat) - 1
        j = np.searchsorted(self.lon_edges, lon) - 1
        if i < 0 or j < 0 or i >= len(self.lat_edges) - 1 or j >= len(self.lon_edges) - 1:
            return False
        return bool(self.mask[i, j])


def _land_ok(world, lat, lon):
    if world is None or world.land is None:
        return True
    return bool(world.land.lookup(lon, lat))


def circle_circle_points(lat1, lon1, r1, lat2, lon2, r2):
    """两球面圆 (小圆盘) 的交点。返回 [(lat,lon), ...] (0/1/2 个)。"""
    def v(lat, lon):
        la, lo = np.radians(lat), np.radians(lon)
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    c1, c2 = v(lat1, lon1), v(lat2, lon2)
    d = float(np.clip(np.dot(c1, c2), -1.0, 1.0))
    dth = float(np.arccos(d))
    if not np.isfinite(dth) or dth < 1e-12:
        return []
    t1, t2 = float(r1) / R_EARTH_KM, float(r2) / R_EARTH_KM
    if dth > t1 + t2 or dth < abs(t1 - t2):
        return []
    # 沿 c1->c2 的弧上的"垂足"角距 a; 垂直偏移 h
    a = (dth ** 2 + t1 ** 2 - t2 ** 2) / (2.0 * dth)
    h2 = t1 ** 2 - a ** 2
    h = float(np.sqrt(max(h2, 0.0)))
    e1 = c1
    e2 = c2 - d * c1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-12:
        return []
    e2 = e2 / n2
    u = np.cross(c1, c2)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return []
    u = u / nu
    base = np.cos(a) * e1 + np.sin(a) * e2
    pts = []
    for sgn in (1.0, -1.0):
        p = np.cos(h) * base + sgn * np.sin(h) * u
        p = p / np.linalg.norm(p)
        lat = float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0))))
        lon = float(np.degrees(np.arctan2(p[1], p[0])))
        pts.append((lat, lon))
    if h < 1e-9:
        return pts[:1]
    return pts


def _grid_edges(lo, hi, step):
    n = max(1, int(np.ceil((hi - lo) / step)))
    return lo + np.arange(n + 1) * step


def _cell_centers(lat_edges, lon_edges):
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return latc, lonc


def _constraint_ok(c, latc, lonc):
    """(ny,nx) 网格上是否满足约束 (平面近似 + 1km 余量, 保守)。"""
    return fast_dist_km(c.lat, c.lon, latc[:, None], lonc[None, :]) <= c.radius_km + 1.0


def coverage_sweep(mask, lat_edges, lon_edges, point, prior=None, levels=(0.5, 0.8, 0.95, 0.995)):
    """从后验最高点出发, 按后验质量增量扩区, 给出不同置信水平下的区域大小。

    返回 [{level, area_km2, polygon}]; level 表示该区域至少包含 level 的后验质量。
    """
    ny, nx = mask.shape
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    w = np.where(mask, prior if prior is not None else mask.astype(float), 0.0)
    if w.sum() <= 0:
        w = mask.astype(float)
    flat = w.ravel()
    order = np.argsort(-flat)
    cum = np.cumsum(flat[order])
    tot = float(cum[-1]) if len(cum) else 0.0
    cell_area = (R_EARTH_KM ** 2) * np.radians(abs(lat_edges[1] - lat_edges[0])) ** 2 * \
        np.cos(np.radians(latc))[:, None]
    cell_area = cell_area.ravel()
    out = []
    for lv in levels:
        k = int(np.searchsorted(cum, lv * tot)) + 1 if tot > 0 else 0
        k = max(k, 1)
        sel = np.zeros(flat.shape, bool)
        sel[order[:k]] = True
        m = sel.reshape(ny, nx)
        out.append(dict(level=float(lv), area_km2=float(cell_area[sel].sum()),
                        polygon=mask_to_polygons(m, lat_edges, lon_edges)))
    return out


def _logt_likelihood(c, latc, lonc, E=0.0):
    """时延似然: p(log t), t = (rtt - E) / (a + b*d)。

    t = 1 表示"该单元的时延恰好等于按标定线预测的最小值", 越小越可能是真值
    (因为测量噪声只会把 RTT 抬高); 标定残差分布给出"抬多少算正常"。
    """
    if c.logt_x is None or c.logt_p is None or c.b <= 0:
        return np.ones((len(latc), len(lonc)))
    d = haversine_km(c.lat, c.lon, latc[:, None], lonc[None, :])
    model = c.a + c.b * np.maximum(d, 0.0)
    model = np.maximum(model, 1e-6)
    t = np.maximum((np.maximum(c.rtt - E, 0.0)) / model, 1.0)
    lt = np.log(t)
    w = np.interp(lt, c.logt_x, c.logt_p, left=c.logt_p[0], right=c.logt_p[-1])
    return np.maximum(w, 1e-6)


def _population_prior(world, latc, lonc, base=0.02, mask=None, max_cities=260):
    """人口密度先验: 城市人口核密度 + 陆地基础值。

    只在待评分的网格单元内计算 (mask 之外无所谓), 并限制参与计算的城市数,
    以免在超大区域上做无谓的 O(单元×城市) 运算。
    """
    ny, nx = len(latc), len(lonc)
    prior = np.full((ny, nx), base, np.float64)
    if world is None or not hasattr(world, 'cities'):
        return prior
    cities = world.cities
    if mask is not None and mask.any():
        ii, jj = np.nonzero(mask)
        lat0, lon0 = float(np.mean(latc[ii])), float(np.mean(lonc[jj]))
    else:
        lat0, lon0 = float(latc.mean()), float(lonc.mean())
    span = max(np.ptp(latc), np.ptp(lonc) * np.cos(np.radians(lat0)), 1e-3)
    rad = min(span * 1.2 + 100.0, 20000.0)
    idx = cities.near(lat0, lon0, rad, max_n=None)
    if len(idx) == 0:
        return prior
    if len(idx) > max_cities:
        # 人口大的城市优先, 其余合并进基础密度
        idx = idx[np.argsort(-cities.pop[idx])[:max_cities]]
    clat, clon, cpop = cities.lat[idx], cities.lon[idx], cities.pop[idx]
    # 每个城市对网格的贡献用高斯核 (尺度 ~ 城市影响半径)
    la = np.radians(latc)[:, None, None]
    # 分块计算避免内存爆炸
    step_city = 64
    for s in range(0, len(idx), step_city):
        sl = slice(s, s + step_city)
        d = haversine_km(latc[:, None, None], lonc[None, :, None],
                         clat[None, None, sl], clon[None, None, sl])
        sigma = np.clip(20.0 + 0.25 * np.sqrt(cpop[sl]) * 1.0, 20.0, 220.0)[None, None, :]
        g = np.exp(-0.5 * (d / sigma) ** 2)
        prior += np.sum(g * (cpop[sl][None, None, :] / 1e5) ** 0.6, axis=2)
    return prior


def _minimax_violation(cons, lat, lon):
    v = np.zeros(len(cons))
    for k, c in enumerate(cons):
        v[k] = haversine_km(c.lat, c.lon, lat, lon) - c.radius_km
    return v


def _refine_feasible(cons, lat0, lon0, iters=80):
    """从初值出发最小化 max_i(d_i - R_i) (Nelder-Mead 近似: 逐步坐标下降)。"""
    lat, lon = float(lat0), float(lon0)
    best = float(_minimax_violation(cons, lat, lon).max())
    step = 200.0  # km
    for _ in range(iters):
        improved = False
        for dlat, dlon in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            step_deg = step / 111.0
            nlat = np.clip(lat + dlat * step_deg, -89.5, 89.5)
            cosl = max(np.cos(np.radians(lat)), 0.05)
            nlon = lon + dlon * step_deg / cosl
            if nlon > 180.0: nlon -= 360.0
            if nlon < -180.0: nlon += 360.0
            v = float(_minimax_violation(cons, nlat, nlon).max())
            if v < best - 1e-9:
                lat, lon, best = nlat, nlon, v
                improved = True
        if not improved:
            step *= 0.5
            if step < 0.02:
                break
    return lat, lon, best


def feasible_anchor(cons, tol_km=0.5):
    """找一个可行点 (若存在)。返回 (lat, lon, min_max_violation) 或 None。"""
    cands = []
    for c in cons:
        cands.append((c.lat, c.lon))
    for i in range(len(cons)):
        for j in range(i + 1, len(cons)):
            cands.extend(circle_circle_points(cons[i].lat, cons[i].lon, cons[i].radius_km,
                                              cons[j].lat, cons[j].lon, cons[j].radius_km))
    # 粗略全球采样兜底 (区域很大且没有交点时, 圆心本身就是可行点; 一般够用)
    if len(cands) < 8:
        for la in np.arange(-60.0, 76.0, 5.0):
            for lo in np.arange(-180.0, 180.0, 5.0):
                cands.append((float(la), float(lo)))
    best = None
    for la, lo in cands:
        v = float(_minimax_violation(cons, la, lo).max())
        if best is None or v < best[2]:
            best = (la, lo, v)
        if v <= tol_km:
            return float(la), float(lo), v
    # 局部精修最好的候选
    la, lo, v = _refine_feasible(cons, best[0], best[1])
    if v <= tol_km:
        return float(la), float(lo), v
    return None


def _gauss_prior(latc, lonc, center, sigma_km):
    d = haversine_km(center[0], center[1], latc[:, None], lonc[None, :])
    return np.exp(-0.5 * (d / max(sigma_km, 1e-3)) ** 2)


def compute_region(constraints, world=None, res_coarse=1.0, max_cells=320,
                   land_only=True, soft_factor=1.3, min_cell_deg=0.004,
                   use_posterior=True, bbox_pad=0.3, prior_center=None,
                   posterior_fast=False, max_cells_lon=3.0):
    """计算所有约束的交集区域 + 后验。空集时返回 empty=True (调用者用 relax)。"""
    cons = [c for c in constraints if c.trusted]
    empty = RegionResult(np.array([0.0]), np.array([0.0]), np.zeros((0, 0), bool),
                         np.zeros((0, 0)), 0.0, [], 0.0, empty=True)
    if not cons:
        return empty
    # ---- 粗网格 (1°) 找可行范围
    lat_c = np.arange(-89.0, 89.0 + 1e-9, res_coarse)
    lon_c = np.arange(-179.0, 179.0 + 1e-9, res_coarse)
    ok = np.ones((len(lat_c), len(lon_c)), bool)
    if land_only and world is not None and world.land is not None:
        ok &= world.land.lookup(lon_c[None, :], lat_c[:, None])
    for c in cons:
        ok &= _constraint_ok(c, lat_c, lon_c)
    # 约束交集一定在每个约束圆盘的包围盒内 => 取各约束包围盒的交集, 作为范围硬保证
    bb_lat_lo, bb_lat_hi = -89.5, 89.5
    bb_lon_lo, bb_lon_hi = -180.0, 180.0
    lon_ref = cons[0].lon
    for c in cons:
        t = min(float(c.radius_km), 20000.0) / R_EARTH_KM      # 角半径 (rad)
        lat_rad = np.radians(c.lat)
        la_lo = max(-90.0, c.lat - np.degrees(t))
        la_hi = min(90.0, c.lat + np.degrees(t))
        if la_lo <= -90.0 or la_hi >= 90.0:
            dlon = 180.0
        else:
            st = np.sin(t)
            ct = np.cos(lat_rad)
            dlon = 180.0 if st >= ct else float(np.degrees(np.arcsin(st / ct)))
        rel_lo = float(wrap_lon(c.lon - dlon - lon_ref))
        rel_hi = float(wrap_lon(c.lon + dlon - lon_ref))
        bb_lat_lo = max(bb_lat_lo, la_lo)
        bb_lat_hi = min(bb_lat_hi, la_hi)
        if dlon >= 180.0:
            continue        # 该圆覆盖全部经度, 不给经度上加约束
        bb_lon_lo = max(bb_lon_lo, lon_ref + rel_lo)
        bb_lon_hi = min(bb_lon_hi, lon_ref + rel_hi)
    bb_ok = bb_lat_lo <= bb_lat_hi and bb_lon_lo <= bb_lon_hi
    lat_min = bb_lat_lo if bb_ok else max(min(c.lat - c.radius_km / 111.0 for c in cons), -89.5)
    lat_max = bb_lat_hi if bb_ok else min(max(c.lat + c.radius_km / 111.0 for c in cons), 89.5)
    if ok.any():
        ii, jj = np.nonzero(ok)
        lat_lo, lat_hi = lat_c[ii].min() - res_coarse, lat_c[ii].max() + res_coarse
        lon_lo, lon_hi = lon_c[jj].min() - res_coarse, lon_c[jj].max() + res_coarse
        # 与约束包围盒取并 (保证不漏掉粗网格没落点的窄条)
        if bb_ok:
            lat_lo = min(lat_lo, lat_min); lat_hi = max(lat_hi, lat_max)
            lon_lo = min(lon_lo, bb_lon_lo); lon_hi = max(lon_hi, bb_lon_hi)
    else:
        # 粗网格太粗 (小区域) 或真为空: 用可行锚点定位
        anchor = feasible_anchor(cons)
        if anchor is None:
            return empty
        la, lo, _ = anchor
        lat_lo, lat_hi = min(lat_min, la - 0.15), max(lat_max, la + 0.15)
        lon_lo, lon_hi = lo - 0.15, lo + 0.15
        # 保证覆盖所有与锚点"可能相连"的区域: 用粗网格可行单元再检查一次
        ok2 = np.ones((len(lat_c), len(lon_c)), bool)
        for c in cons:
            ok2 &= _constraint_ok(c, lat_c, lon_c)
        if ok2.any():
            ii, jj = np.nonzero(ok2)
            lat_lo = min(lat_lo, lat_c[ii].min() - res_coarse)
            lat_hi = max(lat_hi, lat_c[ii].max() + res_coarse)
            lon_lo = min(lon_lo, lon_c[jj].min() - res_coarse)
            lon_hi = max(lon_hi, lon_c[jj].max() + res_coarse)
    # 硬保证: 交集必在约束包围盒交集 bb 内。无论走哪条分支, 最终窗口都取并集,
    # 这样绝不会因为粗网格/锚点估计偏差而把真值所在位置截在窗口外。
    if bb_ok:
        lat_lo = min(lat_lo, bb_lat_lo); lat_hi = max(lat_hi, bb_lat_hi)
        lon_lo = min(lon_lo, bb_lon_lo); lon_hi = max(lon_hi, bb_lon_hi)
    # 细网格范围必须向四周多留一格, 否则边界上的可行单元会被网格边缘截掉
    lat_lo = max(lat_lo, -89.9); lat_hi = min(lat_hi, 89.9)
    lon_lo = max(lon_lo, -180.0); lon_hi = min(lon_hi, 180.0)
    pad_lat = max(res_coarse, (lat_hi - lat_lo) * 0.05, 0.05)
    lat_lo -= pad_lat; lat_hi += pad_lat
    lon_pad = pad_lat / max(np.cos(np.radians((lat_lo + lat_hi) / 2)), 0.05)
    lon_lo = max(lon_lo - lon_pad, -180.0); lon_hi = min(lon_hi + lon_pad, 180.0)
    # ---- 细网格 (自适应步长)
    span_lat = lat_hi - lat_lo
    span_lon = lon_hi - lon_lo
    cosmid = max(np.cos(np.radians((lat_lo + lat_hi) / 2)), 0.05)
    step = max(max(span_lat, span_lon * cosmid) / max_cells, min_cell_deg)
    # 再保险: 两个方向格数都不超过 max_cells_lon * max_cells
    step = max(step, span_lat / (2.0 * max_cells), span_lon / (max_cells_lon * max_cells))
    # 角点保守判定会轻微放大区域; 用更细的网格抵消 (角点扩大 ~O(step))
    step *= 0.7

    def _grid_check(lo_lat, hi_lat, lo_lon, hi_lon, st, corner_safe, keep_cells=True):
        # 硬上限: 任何情况下网格单元数不超过 160k (防止病态输入把内存/时间拖爆)
        while ((hi_lat - lo_lat) / st + 1) * ((hi_lon - lo_lon) / st + 1) > 160000:
            st *= 1.4
        if keep_cells:
            # 网格边界对齐到步长的整数倍, 否则 bbox 边缘附近的可行单元会因
            # "最后一格被截短"而让保守判定失效 (真值可能正好落在被截掉的部分)
            le = np.arange(np.floor(lo_lat / st) * st, hi_lat + st, st)
            oe = np.arange(np.floor(lo_lon / st) * st, hi_lon + st, st)
            le = le[le <= hi_lat + st * 0.5]
            oe = oe[oe <= hi_lon + st * 0.5]
        else:
            le = _grid_edges(lo_lat, hi_lat, st)
            oe = _grid_edges(lo_lon, hi_lon, st)
        latc_, lonc_ = _cell_centers(le, oe)
        g = np.ones((len(latc_), len(lonc_)), bool)
        if land_only and world is not None and world.land is not None:
            g &= world.land.lookup(lonc_[None, :], latc_[:, None])
        if corner_safe:
            clat = np.stack([le[:-1], le[:-1], le[1:], le[1:]])
            clon = np.stack([oe[:-1], oe[1:], oe[:-1], oe[1:]])
            for c in cons:
                # 以探测点为中心的等距圆柱近似 (误差 <0.4%, 且对判定留了余量)
                okc = np.ones(g.shape, bool)
                if not getattr(c, 'corner_safe', True):
                    okc &= fast_dist_km(c.lat, c.lon, latc_[:, None], lonc_[None, :]) \
                        <= c.radius_km + 1.0
                else:
                    lim = c.radius_km + 1.0
                    for k in range(4):
                        okc &= fast_dist_km(c.lat, c.lon, clat[k][:, None],
                                            clon[k][None, :]) <= lim
                g &= okc
        else:
            for c in cons:
                g &= _constraint_ok(c, latc_, lonc_)
        return le, oe, latc_, lonc_, g

    lat_edges, lon_edges, latc, lonc, geom = _grid_check(
        lat_lo, lat_hi, lon_lo, lon_hi, step, corner_safe=True)
    if not geom.any():
        # 网格穿过小区域而未能落点: 围绕可行锚点加密重试
        anchor = feasible_anchor(cons)
        if anchor is None:
            return empty
        la, lo, _ = anchor
        span = max(step * 4.0, 0.05)
        step2 = max(min_cell_deg, 2.0 * span / max_cells)
        lat_edges, lon_edges, latc, lonc, geom = _grid_check(
            la - span, la + span, lo - span, lo + span, step2, corner_safe=False)
        if not geom.any():
            return empty
    mask = geom.copy()
    if land_only and world is not None and world.land is not None:
        mask &= world.land.lookup(lonc[None, :], latc[:, None])
    post = np.zeros(mask.shape, np.float64)
    if mask.any():
        if use_posterior and not (posterior_fast is None or posterior_fast):
            pass
        if use_posterior:
            if prior_center is not None:
                prior = _gauss_prior(latc, lonc, prior_center[:2], prior_center[2])
            elif posterior_fast:
                prior = _population_prior(world, latc, lonc, mask=mask, max_cities=48)
            else:
                prior = _population_prior(world, latc, lonc, mask=mask)
            lk = np.ones(mask.shape)
            Eoff = float(getattr(prior_center, 'offset', 0.0)) if prior_center is not None \
                and hasattr(prior_center, 'offset') else 0.0
            for c in cons:
                lk *= _logt_likelihood(c, latc, lonc, E=Eoff)
            post = np.where(mask, prior * lk, 0.0)
        else:
            post = mask.astype(float)
    cell_area = (R_EARTH_KM ** 2) * np.radians(step) ** 2 * \
        np.cos(np.radians(latc))[:, None]
    area = float(np.sum(cell_area * mask))
    res = RegionResult(lat_edges, lon_edges, mask, post, area, [],
                       0.0, np.nan, np.nan, int(mask.sum()), float(step) * 111.0,
                       empty=not mask.any())
    if mask.any():
        w = post
        if w.sum() <= 0:
            w = mask.astype(float)
        # 区域内后验峰值; 为减轻网格效应, 在峰值附近取后验加权质心作为点估计
        k = np.unravel_index(np.argmax(w), w.shape)
        res.peak_lat = float(latc[k[0]]); res.peak_lon = float(lonc[k[1]])
        res.best_lat = float(latc[k[0]]); res.best_lon = float(lonc[k[1]])
        # 点估计 = 后验最大单元的中心 (区域内人口×时延最相容处)。
        # 不用质心: 大区域 (大洲尺度) 的质心会落在无人区, 而人口先验的峰在真实城市附近。
        pk = np.unravel_index(np.argmax(w), w.shape)
        res.best_lat = float(latc[pk[0]])
        res.best_lon = float(lonc[pk[1]])
        total = float(w.sum())
        # 置信度: 区域内后验质量 / 放宽 soft_factor 后区域内扩展区间的后验质量
        wide = np.ones(mask.shape, bool)
        if land_only and world is not None and world.land is not None:
            wide &= world.land.lookup(lonc[None, :], latc[:, None])
        for c in cons:
            wide &= _constraint_ok(c.soft(soft_factor), latc, lonc)
        if use_posterior:
            ppost = _population_prior(world, latc, lonc) if prior_center is None else \
                _gauss_prior(latc, lonc, prior_center[:2], prior_center[2])
            lk = np.ones(mask.shape)
            for c in cons:
                lk *= _logt_likelihood(c, latc, lonc)
            wpost = np.where(wide, ppost * lk, 0.0)
        else:
            wpost = wide.astype(float)
        pwide = float(wpost.sum())
        res.confidence = float(min(1.0, total / pwide)) if pwide > 0 else 0.5
        res.polygon = mask_to_polygons(mask, lat_edges, lon_edges, world=world)
    return res


# ------------------------------------------------------------------ 空集诊断
def _candidate_points(cons, world, extra_grid=2.0):
    pts = [(c.lat, c.lon) for c in cons]
    # 两两圆心连线上的点也加入候选 (交集常常是"窄带", 交点反而落在带外)
    for i in range(len(cons)):
        for j in range(i + 1, len(cons)):
            a, b = cons[i], cons[j]
            d = haversine_km(a.lat, a.lon, b.lat, b.lon) / R_EARTH_KM
            if d < 1e-9:
                continue
            n = int(np.clip(d * R_EARTH_KM / 100.0, 2, 40))
            for t in np.linspace(0.0, 1.0, n):
                la = a.lat + (b.lat - a.lat) * t
                lo = a.lon + float(wrap_lon(b.lon - a.lon)) * t
                pts.append((float(la), float(lo)))
    if len(cons) <= 14:
        for i in range(len(cons)):
            for j in range(i + 1, len(cons)):
                pts.extend(circle_circle_points(cons[i].lat, cons[i].lon, cons[i].radius_km,
                                                cons[j].lat, cons[j].lon, cons[j].radius_km))
    if len(pts) < 50:
        # 全球粗网格兜底 (一次性栅格查询, 不做逐点 Python 循环)
        las = np.arange(-60.0, 76.0, extra_grid * 4)
        los = np.arange(-180.0, 180.0, extra_grid * 4)
        LA, LO = np.meshgrid(las, los, indexing='ij')
        if world is not None and world.land is not None and hasattr(world.land, 'lookup'):
            keep = world.land.lookup(LO.ravel(), LA.ravel())
        else:
            keep = np.ones(LA.size, bool)
        pts.extend(zip(LA.ravel()[keep].tolist(), LO.ravel()[keep].tolist()))
    if not pts:
        return np.zeros((0, 2), np.float64)
    return np.asarray(pts, np.float64)


def _best_candidate(cons, pts):
    n = len(cons)
    viol = np.empty((len(pts), n))
    for k, c in enumerate(cons):
        viol[:, k] = np.maximum(0.0, haversine_km(c.lat, c.lon, pts[:, 0], pts[:, 1])
                                - c.radius_km)
    cnt = (viol > 1.0).sum(axis=1)
    score = cnt * 1e5 + np.minimum(viol.sum(axis=1), 9e4)
    b = int(np.argmin(score))
    return b, viol[b], int(cnt[b])


def diagnose_and_relax(constraints, world=None, max_relax=5, soft=1.10,
                       min_relax_km=15.0, verbose=False, max_constraints=18,
                       max_candidates=600):
    """找出与多数约束矛盾的探测点并放宽。

    返回 (新约束列表, 诊断记录)。每个约束可能被: 放宽 (expand) 或剔除 (drop)。
    原则: 每次只放宽"让交集重新非空所需的量"再多留 10%, 不做无根据的大放水。
    """
    cons = [Constraint(**{**c.__dict__}) for c in constraints if c.trusted]
    if len(cons) > max_constraints:
        # 约束很多时, 只保留"最紧的"若干条参与诊断 (圆越小越可能有错),
        # 其余约束只要不与之矛盾即可; 这样把 O(n^2) 的候选点生成控制住
        cons.sort(key=lambda c: c.radius_km)
        keep = cons[:max_constraints]
        cons = keep
    diags = []
    for it in range(max_relax):
        res = compute_region(cons, world=world, land_only=False, max_cells=70,
                             use_posterior=False)
        if not res.empty:
            return cons, diags
        pts = _candidate_points(cons, world)
        if len(pts) == 0:
            break
        if len(pts) > max_candidates:
            # 均匀抽稀 (候选点只需覆盖几何, 不需要全量)
            sel = np.linspace(0, len(pts) - 1, max_candidates).astype(int)
            pts = pts[sel]
        b, v, nbad = _best_candidate(cons, pts)
        if nbad == 0:
            for c in cons:
                c.radius_km *= 1.25
            diags.append(dict(action='expand_all', factor=1.25, iteration=it))
            continue
        bad = np.nonzero(v > 1.0)[0]
        if len(bad) > len(cons) // 2 + 1:
            for c in cons:
                c.radius_km *= 1.6
            diags.append(dict(action='expand_all', factor=1.6, iteration=it))
            continue
        worst = int(bad[np.argmax(v[bad])])
        c = cons[worst]
        old = c.radius_km
        c.radius_km = float(old + v[worst] * (1.0 + max(soft, 0.5)) + min_relax_km)
        c.note = '放宽: 与其余约束矛盾, 半径 %.0f→%.0f km' % (old, c.radius_km)
        diags.append(dict(action='expand', pid=c.pid, label=c.label,
                          excess_km=float(v[worst]), old_radius_km=float(old),
                          new_radius_km=float(c.radius_km), iteration=it))
        if verbose:
            print('  relax iter', it, 'expand', c.label, 'by %.0f km' % v[worst])
    # 仍为空: 逐个剔除"最不可信"的约束 (超出量最大者)
    guard = 0
    while len(cons) > 2 and guard < len(cons) + 4:
        guard += 1
        pts = _candidate_points(cons, world)
        if len(pts) == 0:
            break
        if len(pts) > max_candidates:
            sel = np.linspace(0, len(pts) - 1, max_candidates).astype(int)
            pts = pts[sel]
        b, v, nbad = _best_candidate(cons, pts)
        if nbad == 0:
            for c in cons:
                c.radius_km *= 1.25
            diags.append(dict(action='expand_all', factor=1.25))
            continue
        worst = int(np.argmax(v))
        c = cons.pop(worst)
        c.trusted = False
        diags.append(dict(action='drop', pid=c.pid, label=c.label,
                          excess_km=float(v[worst])))
        if verbose:
            print('  drop', c.label)
        res = compute_region(cons, world=world, land_only=False, max_cells=70,
                             use_posterior=False)
        if not res.empty:
            break
    return cons, diags


# ------------------------------------------------------------------ 轮廓提取
def mask_to_polygons(mask, lat_edges, lon_edges, world=None, simplify_km=2.0):
    """把布尔掩码转成边界多边形 (块状轮廓 + 简化)。返回 [[(lon,lat),...], ...]。"""
    ny, nx = mask.shape
    if not mask.any():
        return []
    from collections import defaultdict
    segs = defaultdict(list)
    mask = _fill_tiny_holes(mask)
    ii, jj = np.nonzero(mask)
    inside = set(zip(ii.tolist(), jj.tolist()))
    lat_e, lon_e = lat_edges, lon_edges
    for i, j in inside:
        x0, x1 = float(lon_e[j]), float(lon_e[j + 1])
        y0, y1 = float(lat_e[i]), float(lat_e[i + 1])
        if (i - 1, j) not in inside: segs[(x0, y0)].append((x1, y0))
        if (i + 1, j) not in inside: segs[(x1, y1)].append((x0, y1))
        if (i, j - 1) not in inside: segs[(x0, y1)].append((x0, y0))
        if (i, j + 1) not in inside: segs[(x1, y0)].append((x1, y1))
    used = set()
    polys = []
    for start in list(segs.keys()):
        if start in used:
            continue
        # 只从"没有入边"或任意点出发, 沿唯一未用出边前进
        ring = [start]
        cur = start
        prev = None
        for _ in range(len(segs) * 3 + 20):
            outs = [q for q in segs.get(cur, []) if (cur, q) not in used]
            if not outs:
                break
            q = outs[0]
            used.add((cur, q))
            used.add((q, cur))
            ring.append(q)
            prev, cur = cur, q
            if cur == start:
                break
        if len(ring) >= 4:
            polys.append(_simplify_closed(ring, tol_deg=max(0.004, simplify_km / 111.0)))
    polys.sort(key=lambda r: -abs(_ring_area(r)))
    return polys[:12]


def _fill_tiny_holes(mask, max_hole=12):
    """填掉掩码内部的小孔 (网格锯齿造成), 只处理不接触边界的连通空洞。"""
    ny, nx = mask.shape
    inv = ~mask
    if not inv.any():
        return mask
    from scipy.ndimage import label as _label
    lab, n = _label(inv)
    if n == 0:
        return mask
    edge_ids = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) | \
        set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    sizes = np.bincount(lab.ravel())
    out = mask.copy()
    for k in range(1, n + 1):
        if k in edge_ids:
            continue
        if sizes[k] <= max_hole:
            out[lab == k] = True
    return out


def _ring_area(ring):
    x = np.array([p[0] for p in ring]); y = np.array([p[1] for p in ring])
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _simplify_closed(ring, tol_deg=0.01, max_pts=500):
    """闭合折线简化: 先按步长抽稀, 再 Douglas-Peucker。"""
    pts = np.asarray(ring, np.float64)
    if len(pts) <= 4:
        return ring
    # 抽稀: 最多 ~2000 点
    if len(pts) > 2000:
        step = int(np.ceil(len(pts) / 2000))
        pts = pts[::step]
    a = _dp(pts, tol_deg)
    if len(a) > max_pts:
        step = int(np.ceil(len(a) / max_pts))
        a = a[::step]
    return a


def _dp(pts, tol):
    """迭代版 Douglas-Peucker (避免深度递归)。"""
    n = len(pts)
    if n < 3:
        return pts.tolist()
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = pts[i0], pts[i1]
        ab = b - a
        L = np.hypot(ab[0], ab[1])
        seg = pts[i0 + 1:i1]
        if L < 1e-12:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (a[1] - seg[:, 1]) - (a[0] - seg[:, 0]) * ab[1]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            kk = i0 + 1 + k
            keep[kk] = True
            stack.append((i0, kk))
            stack.append((kk, i1))
    return pts[keep].tolist()
