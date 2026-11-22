"""sim.py — 仿真互联网 (真实时延现象: 绕路/拥塞/不对称/接入时延/丢包)。

拓扑
----
节点 = 全球机场 (IATA)。边 = k 近邻短链 + 长途中继 + 最小生成树保连通。
每条边有独立的绕路系数 (光纤不沿直线走) 和方向性权重 (去/回程可能走不同路径,
模拟 hot-potato 路由与 ISP 策略不对称)。

时延组成
--------
  RTT = 去程(传播 + 每跳处理 + 排队) + 回程(同上) + 目标接入时延 + 主机处理
  · 排队: 每次测量随机, 重尾; 多次测量取最小值可滤掉
  · 拥塞: 少数目标处于持续拥塞 (最小值也滤不掉) → 圆被放大 (保守)
  · 不对称: 去/回程边权独立, 接入上下行不对称 (ADSL/卫星)
  · 接入: FTTH/DSL/移动/卫星, 最后一段时延往往远大于骨干
  · 丢包: 少量测量直接失败

本模块只提供"网络真实"和测量接口; 定位算法不知道任何真值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree, connected_components
from scipy.spatial import cKDTree

from .geo import haversine_km, R_EARTH_KM
from .world import get_world

V_FIBER_KM_S = 200000.0                 # 光纤中光速 ~200 km/ms
MS_PER_KM = 1000.0 / V_FIBER_KM_S       # 0.005 ms/km 单程

ACCESS_KINDS = ('fiber', 'dsl', 'mobile', 'satellite', 'idc')
ACCESS_P = np.array([0.62, 0.10, 0.16, 0.02, 0.10])


@dataclass
class SimConfig:
    seed: int = 7
    n_probes: int = 260
    knn: int = 14
    max_regional_km: float = 4800.0
    hub_max_km: float = 13000.0
    n_hubs: int = 90
    loss_p: float = 0.003
    # 裸光纤速度 (供 naive 基线使用)
    fiber_ms_per_km: float = MS_PER_KM


@dataclass
class Target:
    tid: int
    lat: float
    lon: float
    kind: str
    node: int
    prop_f: float          # 接入段去程传播 (ms)
    prop_r: float          # 接入段回程传播 (ms)
    base_f: float          # 接入技术固定时延 (ms, 去程)
    base_r: float
    host_ms: float         # 目标主机协议栈处理 (ms, 单程)
    persist: float         # 持续拥塞 (ms, 只影响目标)
    upstream_persist: float  # 共享段持续拥塞 (ms, 目标与末跳路由器都受影响)
    bloat_p: float         # 每次测量的接入排队概率
    _base: dict = field(default_factory=dict, repr=False)


@dataclass
class Probe:
    pid: int
    node: int
    lat: float
    lon: float


@dataclass
class Hop:
    node: int
    name: str
    rtt_ms: float
    loss: bool


class Stats:
    def __init__(self):
        self.packets = 0
        self.pings = 0
        self.traceroutes = 0

    def reset(self):
        self.packets = self.pings = self.traceroutes = 0

    def snapshot(self):
        return dict(packets=self.packets, pings=self.pings,
                    traceroutes=self.traceroutes)


class Simulation:
    def __init__(self, world=None, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.world = world or get_world()
        self.rng = np.random.default_rng(self.cfg.seed)
        ap = self.world.airports
        self.n_nodes = len(ap.lat)
        self.node_lat = np.asarray(ap.lat, np.float64)
        self.node_lon = np.asarray(ap.lon, np.float64)
        self.node_big = np.asarray(ap.big, np.float64)
        self.node_code = list(ap.code)
        self.node_city = [m if m else '' for m in ap.municipality]
        self.node_country = list(ap.country)
        self._fill_city_names()
        self._build_topology()
        self._select_probes()
        self._build_router_meta()
        self.stats = Stats()

    # ------------------------------------------------------------------ 拓扑
    def _fill_city_names(self):
        c = self.world.cities
        for i in range(self.n_nodes):
            if self.node_city[i]:
                continue
            j = c.near(self.node_lat[i], self.node_lon[i], 120.0, max_n=1)
            self.node_city[i] = c.name[j[0]] if len(j) else ''

    def _xyz(self, lat, lon):
        la, lo = np.radians(lat), np.radians(lon)
        return np.column_stack([np.cos(la) * np.cos(lo),
                                np.cos(la) * np.sin(lo), np.sin(la)])

    def _build_topology(self):
        rng = self.rng
        n = self.n_nodes
        xyz = self._xyz(self.node_lat, self.node_lon)
        tree = cKDTree(xyz)
        k = min(self.cfg.knn + 1, n)
        _, nbr = tree.query(xyz, k=k)
        pairs = {}
        for i in range(n):
            for j in nbr[i, 1:]:
                j = int(j)
                if j == i:
                    continue
                d = float(haversine_km(self.node_lat[i], self.node_lon[i],
                                       self.node_lat[j], self.node_lon[j]))
                if d <= self.cfg.max_regional_km:
                    a, b = (i, j) if i < j else (j, i)
                    pairs[(a, b)] = min(d, pairs.get((a, b), np.inf))
        # 长途中继: 大城市之间互联 (模拟国际骨干/海缆)
        big = np.nonzero(self.node_big > 0)[0]
        if len(big) > self.cfg.n_hubs:
            city = self.world.cities
            score = np.zeros(len(big))
            for t, i in enumerate(big):
                j = city.near(self.node_lat[i], self.node_lon[i], 30.0, max_n=1)
                score[t] = city.pop[j[0]] if len(j) else 0.0
            big = big[np.argsort(-score)[:self.cfg.n_hubs]]
        kk2 = min(6, len(big))
        _, nb2 = cKDTree(xyz[big]).query(xyz[big], k=kk2)
        for a in range(len(big)):
            for b in nb2[a, 1:]:
                i, j = int(big[a]), int(big[int(b)])
                d = float(haversine_km(self.node_lat[i], self.node_lon[i],
                                       self.node_lat[j], self.node_lon[j]))
                if d <= self.cfg.hub_max_km:
                    a2, b2 = (i, j) if i < j else (j, i)
                    pairs[(a2, b2)] = min(d, pairs.get((a2, b2), np.inf))
        # 孤立节点 (偏远小机场) 连到最近邻居, 保证每个节点都有接入
        deg = np.zeros(n, np.int64)
        for (a, b) in pairs:
            deg[a] += 1; deg[b] += 1
        for i in np.nonzero(deg == 0)[0]:
            dd, jj = tree.query(xyz[i], k=2)
            j = int(jj[1])
            a, b = (int(i), j) if i < j else (j, int(i))
            d = float(haversine_km(self.node_lat[a], self.node_lon[a],
                                   self.node_lat[b], self.node_lon[b]))
            pairs[(a, b)] = d
        keys = np.array(list(pairs.keys()), np.int64)
        dkm = np.array([pairs[(int(a), int(b))] for a, b in keys], np.float64)
        # 连通性: 用最小生成树 (按球面弦长) 补齐跨洋长链
        def _comp_count(kk):
            ix = np.concatenate([kk[:, 0], kk[:, 1]])
            jx = np.concatenate([kk[:, 1], kk[:, 0]])
            return connected_components(
                csr_matrix((np.ones(len(ix)), (ix, jx)), shape=(n, n)), directed=False)

        ncomp, lab = _comp_count(keys)
        if ncomp > 1:
            with np.errstate(all='ignore'):
                chord = np.clip(1.0 - xyz @ xyz.T, 1e-6, 2.0)
            T = minimum_spanning_tree(csr_matrix(chord)).tocoo()
            have = set((int(a), int(b)) for a, b in keys)
            add = []
            for a, b in zip(T.row, T.col):
                a, b = int(a), int(b)
                if a == b:
                    continue
                a2, b2 = (a, b) if a < b else (b, a)
                if (a2, b2) not in have:
                    add.append((a2, b2))
            if add:
                k2 = np.array(add, np.int64)
                d2 = haversine_km(self.node_lat[k2[:, 0]], self.node_lon[k2[:, 0]],
                                  self.node_lat[k2[:, 1]], self.node_lon[k2[:, 1]])
                keys = np.vstack([keys, k2]); dkm = np.concatenate([dkm, d2])
            ncomp, lab = _comp_count(keys)
            if ncomp != 1:
                raise RuntimeError('仿真拓扑不连通: %d' % ncomp)
        self.node_xyz = xyz
        # 去重 (MST 边可能已存在但方向相反)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys, dkm = keys[order], dkm[order]
        keep = np.ones(len(keys), bool)
        keep[1:] = (keys[1:, 0] != keys[:-1, 0]) | (keys[1:, 1] != keys[:-1, 1])
        keys, dkm = keys[keep], dkm[keep]
        m = len(dkm)
        # 绕路: 光纤不沿大圆走 (城市/地形/海缆登陆点), 少数链路明显绕远
        detour = np.clip(rng.lognormal(0.06, 0.12, m), 1.02, 2.2)
        weird = rng.random(m) < 0.02
        detour[weird] = rng.uniform(1.8, 4.0, weird.sum())
        # 每跳存储转发/交换时延 (RTT 中除传播外的主要成分)。
        # 异质但同量级: 骨干轻载 ~0.05, 普通 0.08~0.2, 少数 0.3~0.5。
        # (这是"处理时延可加、异质"的现象; 若把它设得过大, 标定线的截距/斜率比会失衡)
        proc = rng.choice([0.04, 0.10, 0.20, 0.38],
                          size=m, p=[0.35, 0.40, 0.19, 0.06]) * rng.uniform(0.75, 1.35, m)
        base = dkm * detour * MS_PER_KM + proc
        asym = np.exp(rng.normal(0.0, 0.09, (m, 2)))
        hot = rng.random(m) < 0.06
        asym[hot, 1] *= rng.uniform(1.5, 3.5, hot.sum())
        wf = base * asym[:, 0]
        wr = base * asym[:, 1]
        self.edge_i = keys[:, 0].astype(np.int32)
        self.edge_j = keys[:, 1].astype(np.int32)
        self.edge_km = dkm
        self.edge_detour = detour
        ncomp, _ = connected_components(
            csr_matrix((np.ones(2 * m), (np.concatenate([self.edge_i, self.edge_j]),
                                         np.concatenate([self.edge_j, self.edge_i]))),
                       shape=(n, n)), directed=False)
        if ncomp != 1:
            raise RuntimeError('仿真拓扑不连通: %d 个连通分量' % ncomp)
        rows = np.concatenate([self.edge_i, self.edge_j])
        cols = np.concatenate([self.edge_j, self.edge_i])
        self.Wf = csr_matrix((np.concatenate([wf, wr]), (rows, cols)), shape=(n, n))
        self.WrT = csr_matrix((np.concatenate([wr, wf]), (cols, rows)), shape=(n, n))
        self._tree_cache = {}

    def _select_probes(self):
        """探测点选址: 最大最小距离 (farthest-point) + 小随机扰动, 兼顾覆盖与随机性。"""
        rng = self.rng
        cand = np.nonzero(self.node_big > 0)[0]
        if len(cand) < self.cfg.n_probes:
            cand = np.arange(self.n_nodes)
        xyz = self.node_xyz[cand]
        chosen = [int(rng.integers(len(cand)))]
        dist = np.linalg.norm(xyz - xyz[chosen[0]], axis=1)
        for _ in range(1, self.cfg.n_probes):
            score = dist * (0.85 + 0.3 * rng.random(len(dist)))
            k = int(np.argmax(score))
            chosen.append(k)
            dist = np.minimum(dist, np.linalg.norm(xyz - xyz[k], axis=1))
        nodes = cand[np.array(chosen)]
        self.probes = [Probe(pid=i, node=int(nd),
                             lat=float(self.node_lat[nd]), lon=float(self.node_lon[nd]))
                       for i, nd in enumerate(nodes)]
        self.probe_lat = np.array([p.lat for p in self.probes])
        self.probe_lon = np.array([p.lon for p in self.probes])
        self.probe_node = np.array([p.node for p in self.probes], np.int64)

    def _build_router_meta(self):
        """路由器主机名与真实位置 (主机名里的城市码只有 metro 级精度)。"""
        rng = np.random.default_rng(self.cfg.seed + 991)
        n = self.n_nodes
        self.router_name = [None] * n
        self.router_lat = self.node_lat.copy()
        self.router_lon = self.node_lon.copy()
        self.router_jitter_km = np.zeros(n)
        self.router_name_wrong = np.zeros(n, bool)
        city = self.world.cities
        for i in range(n):
            # 真实路由器位置 = 机场/城市附近 5~40km 的机房
            j = city.near(self.node_lat[i], self.node_lon[i], 60.0, max_n=1)
            if len(j):
                tlat, tlon = city.lat[j[0]], city.lon[j[0]]
            else:
                tlat, tlon = self.node_lat[i], self.node_lon[i]
            d = float(haversine_km(self.node_lat[i], self.node_lon[i], tlat, tlon))
            t = rng.uniform(0.15, 0.85) if d > 1 else 1.0
            self.router_lat[i] = tlat * t + self.node_lat[i] * (1 - t)
            self.router_lon[i] = tlon * t + self.node_lon[i] * (1 - t)
            self.router_jitter_km[i] = float(haversine_km(
                self.node_lat[i], self.node_lon[i],
                self.router_lat[i], self.router_lon[i]))
            u = rng.random()
            code = self.node_code[i].lower()
            city_name = self.node_city[i]
            # 少数路由器的反解主机名指向别处 (ISP 命名习惯/搬迁), 用于检验"主机名不可全信"
            if rng.random() < 0.08:
                other = int(rng.integers(n))
                city_name = self.node_city[other] or city_name
                code = self.node_code[other].lower()
                self.router_name_wrong[i] = True
            slug = _slug(city_name)
            if u < 0.58:
                self.router_name[i] = '%s-cr%d.isp-core.example.net' % (code, rng.integers(1, 9))
            elif u < 0.75 and slug:
                self.router_name[i] = 'cr%d.%s.example.net' % (rng.integers(1, 9), slug)
            elif u < 0.90:
                self.router_name[i] = 'ae%d.%s1.example.net' % (rng.integers(1, 9), code)
            else:
                self.router_name[i] = 'ae%d.core%d.example.net' % (
                    rng.integers(1, 9), rng.integers(1, 30))
        self.router_proc = rng.uniform(0.25, 1.8, n)   # ICMP 应答处理时延

    # ------------------------------------------------------------- 最短路树
    def _trees(self, pid):
        t = self._tree_cache.get(pid)
        if t is None:
            p = self.probes[pid].node
            df, pf = dijkstra(self.Wf, directed=True, indices=p, return_predecessors=True)
            dr, pr = dijkstra(self.WrT, directed=True, indices=p, return_predecessors=True)
            t = (np.asarray(df), np.asarray(pf), np.asarray(dr), np.asarray(pr))
            self._tree_cache[pid] = t
        return t

    def probe_path(self, pid, node):
        """探测点 pid 到 node 的去程节点序列。"""
        _, pf, _, _ = self._trees(pid)
        path = []
        cur = int(node)
        guard = 0
        while cur >= 0 and cur != self.probes[pid].node and guard < self.n_nodes:
            path.append(cur)
            cur = int(pf[cur])
            guard += 1
        path.append(int(self.probes[pid].node))
        return path[::-1]

    # ------------------------------------------------------------- 噪声
    def _transient_noise(self, n, rtt_base):
        rng = self.rng
        hop_scale = 1.0 + 0.25 * rtt_base
        z = rng.gamma(1.4, 0.12, n) * hop_scale
        z += (rng.random(n) < 0.035) * rng.exponential(5.0, n)
        z += (rng.random(n) < 0.003) * rng.exponential(22.0, n)
        return z

    # ------------------------------------------------------------- 目标
    def sample_targets(self, n, rng=None, city_frac=0.88):
        rng = rng or np.random.default_rng(self.cfg.seed + 12345)
        city = self.world.cities
        land = self.world.land
        out = []
        tries = 0
        while len(out) < n and tries < n * 40:
            tries += 1
            if rng.random() < city_frac:
                w = city.pop ** 0.75
                w = w / w.sum()
                i = int(rng.choice(len(city.pop), p=w))
                lat, lon = float(city.lat[i]), float(city.lon[i])
                # 城市 jitter 约 1~15km
                lat += rng.normal(0, 0.07)
                lon += rng.normal(0, 0.07) / max(np.cos(np.radians(lat)), 0.2)
            else:
                i = int(rng.integers(self.n_nodes))
                lat = float(self.node_lat[i]) + rng.normal(0, 0.02)
                lon = float(self.node_lon[i]) + rng.normal(0, 0.02)
            if not bool(land.lookup(lon, lat)):
                continue
            out.append(self._make_target(len(out), lat, lon, rng, idc=(rng.random() < 0.12)))
        return out

    def _make_target(self, tid, lat, lon, rng, idc=False):
        # 接入 PoP: 最近机场; 少量目标接入点离真实位置较远 (区域汇聚/托管)
        d = haversine_km(lat, lon, self.node_lat, self.node_lon)
        order = np.argsort(d)
        node = int(order[0])
        if rng.random() < 0.06:
            far = order[(d[order] > 80) & (d[order] < 900)]
            if len(far):
                node = int(far[int(rng.integers(len(far)))])
        if idc:
            kind = 'idc'
        else:
            kind = str(rng.choice(ACCESS_KINDS, p=ACCESS_P))
        dn = float(haversine_km(lat, lon, self.node_lat[node], self.node_lon[node]))
        wind_f = rng.uniform(1.05, 1.45)
        wind_r = rng.uniform(1.05, 1.45)
        prop_f = dn * wind_f * MS_PER_KM
        prop_r = dn * wind_r * MS_PER_KM
        if kind == 'fiber':
            base_f, base_r = rng.uniform(0.25, 1.1), rng.uniform(0.25, 1.1)
        elif kind == 'dsl':
            base_f, base_r = rng.uniform(0.8, 3.0), rng.uniform(3.0, 12.0)
        elif kind == 'mobile':
            base_f, base_r = rng.uniform(4.0, 18.0), rng.uniform(4.0, 18.0)
        elif kind == 'satellite':
            base_f, base_r = rng.uniform(240.0, 340.0), rng.uniform(240.0, 340.0)
        else:  # idc
            base_f, base_r = rng.uniform(0.15, 0.7), rng.uniform(0.15, 0.7)
        host = rng.uniform(0.25, 2.2)
        persist = rng.uniform(8.0, 80.0) if rng.random() < 0.07 else 0.0
        up_persist = rng.uniform(3.0, 15.0) if rng.random() < 0.04 else 0.0
        bloat_p = 0.12 if kind in ('fiber', 'dsl', 'mobile') else 0.02
        return Target(tid, float(lat), float(lon), kind, node,
                      prop_f, prop_r, base_f, base_r, host, persist, up_persist, bloat_p)

    def _target_base(self, pid, t: Target):
        key = pid
        v = t._base.get(key)
        if v is None:
            df, _, dr, _ = self._trees(pid)
            v = (float(df[t.node] + dr[t.node]), float(self.router_proc[t.node]))
            t._base[key] = v
        return v

    # ------------------------------------------------------------- 测量
    def ping(self, pid, target: Target, n=1):
        """返回 n 次 RTT 采样 (ms); inf 表示丢包。"""
        self.stats.pings += 1
        self.stats.packets += n
        base, _ = self._target_base(pid, target)
        base = base + target.prop_f + target.prop_r + target.base_f + target.base_r \
            + 2.0 * target.host_ms + target.persist + target.upstream_persist
        rng = self.rng
        vals = base + self._transient_noise(n, base)
        bloat = rng.random(n) < target.bloat_p
        vals = vals + bloat * rng.uniform(20.0, 200.0, n)
        loss = rng.random(n) < self.cfg.loss_p
        vals = np.where(loss, np.inf, vals)
        return vals

    def ping_node(self, pid, node, n=1):
        """探测点直接 ping 某台路由器 (ICMP 应答)。"""
        self.stats.pings += 1
        self.stats.packets += n
        df, _, dr, _ = self._trees(pid)
        # 共享段拥塞: 用 node 自身的确定性伪随机量, 使同一 epoch 内稳定
        base = float(df[node] + dr[node] + self.router_proc[node])
        rng = self.rng
        vals = base + self._transient_noise(n, base)
        loss = rng.random(n) < self.cfg.loss_p
        return np.where(loss, np.inf, vals)

    def ping_probe(self, pid, qid, n=1):
        """探测点之间的 RTT (用于标定)。"""
        self.stats.pings += 1
        self.stats.packets += n
        df, _, dr, _ = self._trees(pid)
        q = self.probes[qid].node
        base = float(df[q] + dr[q]) + 0.3
        rng = self.rng
        vals = base + self._transient_noise(n, base)
        loss = rng.random(n) < self.cfg.loss_p
        return np.where(loss, np.inf, vals)

    def traceroute(self, pid, target: Target, tries=3):
        """返回 (hops, target_rtt_min)。hops 含末跳路由器与目标。"""
        self.stats.traceroutes += 1
        self.stats.packets += tries * (len(self.probe_path(pid, target.node)) + 2)
        path = self.probe_path(pid, target.node)
        hops = []
        for nd in path[-6:]:          # 只报告靠近目标的几跳
            vals = self.ping_node(pid, nd, tries)
            m = np.min(vals[np.isfinite(vals)]) if np.any(np.isfinite(vals)) else np.inf
            hops.append(Hop(node=int(nd), name=self.router_name[int(nd)],
                            rtt_ms=float(m), loss=not np.isfinite(m)))
        vals = self.ping(pid, target, tries)
        tm = np.min(vals[np.isfinite(vals)]) if np.any(np.isfinite(vals)) else np.inf
        return hops, float(tm)

    # --------------------------------------------------------------- 真值
    def true_router_pos(self, node):
        return float(self.router_lat[node]), float(self.router_lon[node])


def _slug(name):
    s = ''.join(ch.lower() if (ch.isalnum() and ord(ch) < 128) else '-' for ch in name)
    s = '-'.join(p for p in s.split('-') if p)
    return s if len(s) >= 4 else ''
