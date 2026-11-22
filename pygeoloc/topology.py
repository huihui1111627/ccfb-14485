"""topology.py — 路由器定位 (主机名城市线索 + 递归测量) 与"目标在末跳附近"的拓扑先验。

两个独立的线索:
  (1) 主机名: 形如 `shl-cr2.isp-core.example.net` / `cr3.hyderabad.example.net` /
      `ae7.pvg1.example.net`。解析出 IATA 三字码或城市名, 定位到城市 (metro 级, 数十 km)。
      主机名可能过时/张冠李戴, 因此只作为弱先验, 与测量冲突时以测量为准。
  (2) 递归测量: 用与定位目标完全相同的"标定圆交集"方法去定位路由器本身
      (路由器会回 ICMP, 多测几次取最小)。若交集小, 精度可达几十 km。

目标与末跳路由器之间的接入距离上界:
       RTT_target - RTT_lastHop
  扣掉 ICMP 回复处理时延后, 剩下的时间只可能花在"最后一公里"上, 再经标定线换算成
  距离上界。目标一定落在以路由器位置为圆心、这个上界为半径的圆内 —— 这是极强的约束。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

from .geo import haversine_km, MS_PER_KM_AT_C, R_EARTH_KM

IATA_RE = re.compile(r'(?:^|[.\-_])([a-z]{3})(\d)?(?:[.\-_]|$)')
TOKEN_RE = re.compile(r'[a-z0-9]+')


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.lower()


@dataclass
class GeoHint:
    lat: float
    lon: float
    unc_km: float
    source: str          # 'iata' | 'city'
    token: str


@dataclass
class RouterEstimate:
    node: int
    lat: float
    lon: float
    unc_km: float           # 1-sigma 等效半径
    method: str             # 'hostname+delay' | 'delay' | 'hostname'
    hint: GeoHint | None = None
    delay_area_km2: float = 0.0
    pings: int = 0
    n_probes: int = 0
    note: str = ''


class NameResolver:
    """把路由器主机名解析成城市地理线索。"""

    def __init__(self, world):
        self.world = world
        self.airports = world.airports
        self.cities = world.cities
        self.city_index = {}
        for i, nm in enumerate(self.cities.name):
            k = norm_name(nm).replace(' ', '-')
            j = self.city_index.get(k)
            if j is None or self.cities.pop[i] > self.cities.pop[j]:
                self.city_index[k] = i
        # 机场城市名 (OurAirports 的 municipality) 也建索引
        self.airport_city = {}
        for i, m in enumerate(self.airports.municipality):
            k = norm_name(m).replace(' ', '-')
            if k:
                self.airport_city.setdefault(k, []).append(i)

    def hint_from_name(self, name):
        if not name:
            return None
        n = norm_name(name)
        toks = TOKEN_RE.findall(n)
        # (a) IATA 三字码 (须是全小写 token 且全部字符为字母)
        for t in toks:
            if len(t) == 3 and self.airports.index.get(t.upper()) is not None and \
                    len(t) == 3 and re.fullmatch(r'[a-z]{3}', t):
                ai = self.airports.index[t.upper()]
                # 机场 5km 内取城市中心 (更稳)
                j = self.cities.near(self.airports.lat[ai], self.airports.lon[ai], 60.0, max_n=1)
                if len(j):
                    return GeoHint(float(self.cities.lat[j[0]]), float(self.cities.lon[j[0]]),
                                   18.0, 'iata', t)
                return GeoHint(float(self.airports.lat[ai]), float(self.airports.lon[ai]),
                               12.0, 'iata', t)
        # (b) 城市名 (可能带机场编号后缀, 如 pvg1 / hyderabad)
        joined = '-'.join(toks)
        for k in sorted(self.city_index, key=len, reverse=True):
            if len(k) < 4:
                continue
            if k in joined:
                i = self.city_index[k]
                return GeoHint(float(self.cities.lat[i]), float(self.cities.lon[i]),
                               25.0, 'city', k)
        for k in sorted(self.airport_city, key=len, reverse=True):
            if len(k) < 4:
                continue
            if k in joined:
                ais = self.airport_city[k]
                best = max(ais, key=lambda a: self.airports.big[a])
                return GeoHint(float(self.airports.lat[best]), float(self.airports.lon[best]),
                               20.0, 'city', k)
        return None


class RouterLocator:
    """递归定位路由器, 带缓存。"""

    def __init__(self, meas, calib, probes_latlon, world, max_probes=9,
                 n_packets=6, max_radius_km=3500.0):
        self.meas = meas
        self.calib = calib
        self.probes_lat = probes_latlon[:, 0]
        self.probes_lon = probes_latlon[:, 1]
        self.world = world
        self.names = NameResolver(world)
        self.max_probes = max_probes
        self.n_packets = n_packets
        self.max_radius_km = max_radius_km
        self.cache = {}

    # ---------------------------------------------------------------- 内部
    def _pick_probes(self, lat, lon, exclude=()):
        d = haversine_km(lat, lon, self.probes_lat, self.probes_lon)
        order = np.argsort(d)
        # 选不同方向的近探测点: 先找 8 个较近的, 再按方向粗分箱各取一个, 保证几何张开
        near = [p for p in order if p not in exclude][:24]
        picked = []
        used_bins = set()
        for p in near:
            if d[p] > self.max_radius_km:
                break
            brg = self._bearing(lat, lon, self.probes_lat[p], self.probes_lon[p])
            b = int(brg // 45) % 8
            if b in used_bins:
                continue
            used_bins.add(b)
            picked.append(p)
            if len(picked) >= self.max_probes:
                break
        if len(picked) < 4:
            picked = [p for p in near if d[p] <= self.max_radius_km][:self.max_probes]
        return picked, d

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        from .geo import bearing_deg
        return float(bearing_deg(lat1, lon1, lat2, lon2))

    def locate(self, node, name=None, target_hint=None):
        """定位路由器 node。target_hint=(lat,lon) 用于挑选附近的探测点。"""
        key = int(node)
        if key in self.cache:
            return self.cache[key]
        hint = self.names.hint_from_name(name)
        if target_hint is not None:
            center = target_hint
        elif hint is not None:
            center = (hint.lat, hint.lon)
        else:
            center = (float(np.median(self.probes_lat)), float(np.median(self.probes_lon)))
        pings0 = self.meas.total_packets()
        pids, dist = self._pick_probes(center[0], center[1])
        cons = []
        for pid in pids:
            rtt = self.meas.node_rtt(int(pid), key, n=self.n_packets, samples=2)
            if not np.isfinite(rtt):
                continue
            line = self.calib[int(pid)]
            R = float(line.distance_bound(rtt))
            if R <= 0:
                continue
            cons.append(_mk_constraint(int(pid), self.probes_lat[pid], self.probes_lon[pid],
                                       R, rtt, line, label='router-%d' % key))
        from .region import compute_region
        est = None
        if len(cons) >= 3:
            prior = (hint.lat, hint.lon, hint.unc_km) if hint is not None else None
            res = compute_region(cons, world=None, land_only=False, max_cells=140,
                                 use_posterior=True, min_cell_deg=0.01, prior_center=prior)
            if not res.empty:
                lat, lon = res.best_lat, res.best_lon
                region_unc = _region_unc_km(res)
                method, note, unc = 'delay', '', region_unc
                if hint is not None:
                    d = haversine_km(hint.lat, hint.lon, lat, lon)
                    if d <= max(3.0 * hint.unc_km, 150.0):
                        method = 'hostname+delay'
                        unc = max(region_unc, hint.unc_km * 0.6)
                    else:
                        note = 'hostname 与测量矛盾, 已忽略主机名'
                est = RouterEstimate(key, float(lat), float(lon), float(unc), method, hint,
                                     float(res.area_km2), 0, len(cons), note)
        if est is None and hint is not None:
            est = RouterEstimate(key, hint.lat, hint.lon, hint.unc_km, 'hostname', hint,
                                 0.0, 0, len(cons), '时延圆无法三角定位')
        if est is None:
            est = RouterEstimate(key, center[0], center[1], 3000.0, 'none', hint, 0.0, 0,
                                 len(cons), '无线索')
        est.pings = int(self.meas.total_packets() - pings0)
        self.cache[key] = est
        return est


def _mk_constraint(pid, lat, lon, R, rtt, line, label=''):
    from .region import Constraint
    return Constraint(pid=pid, lat=float(lat), lon=float(lon), radius_km=float(R),
                      rtt=float(rtt), a=line.a, b=line.b, label=label)


def _region_unc_km(res):
    """区域等效半径: sqrt(area/pi), 但不小于网格尺度。"""
    return float(max(np.sqrt(max(res.area_km2, 0.0) / np.pi), res.cell_km))


def estimate_router_region(meas, calib, probes_latlon, world, node, name=None,
                           n_packets=6, max_probes=7):
    """一次性接口 (无缓存)。"""
    loc = RouterLocator(meas, calib, probes_latlon, world, max_probes=max_probes,
                        n_packets=n_packets)
    return loc.locate(node, name=name)


def last_hop_anchor(meas, calib, probes_latlon, world, pid, target, n_packets=6,
                    router_cache=None):
    """对 (探测点, 目标) 做 traceroute, 定位末跳路由器, 返回接入距离上界和锚点。"""
    hops, target_rtt = meas.route(pid, target)
    if not hops or target_rtt is None or not np.isfinite(target_rtt):
        return None
    last = hops[-1]
    loc = router_cache
    if loc is None:
        loc = RouterLocator(meas, calib, probes_latlon, world, n_packets=n_packets)
    # 兜底: 用测量自身的几何来收窄路由器位置 (不能用目标真值 —— 那是作弊)。
    # 用 traceroute 里**前一跳**的标定圆交集给出一个大致的搜索中心。
    hint_center = None
    if len(hops) >= 2:
        prev = hops[-2]
        try:
            prev_est = loc.locate(prev.node, name=prev.name)
            hint_center = (prev_est.lat, prev_est.lon)
        except Exception:
            hint_center = None
    est = loc.locate(last.node, name=last.name, target_hint=hint_center)
    # 接入距离上界: 目标 RTT 减去末跳路由器 RTT (均为多次测量的最小值)
    if np.isfinite(last.rtt_ms):
        extra = max(0.0, float(target_rtt - last.rtt_ms))
    else:
        extra = np.inf
    # 接入段距离的**物理**上界: 多出来的 RTT 只可能花在往返接入链路上,
    # 传播速度不可能超过真空光速 => d_access <= extra/2 * c
    if np.isfinite(extra):
        access_km = float(extra) * 0.5 / MS_PER_KM_AT_C
        access_km = min(max(access_km, 15.0), 2.0e4)
    else:
        access_km = 2.0e4
    return dict(est=est, access_km=access_km, router_rtt=float(last.rtt_ms),
                target_rtt=float(target_rtt), hops=hops, locator=loc)
