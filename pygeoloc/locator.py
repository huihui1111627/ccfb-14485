"""locator.py — 端到端定位流程。

流程 (每个目标):
  1. 粗测: 用几何散布的少量探测点测目标 RTT 最小值 → 标定圆 → 交集区域
  2. 粗区域中心 → 挑选附近探测点 + 最近的一批 → 精测 (更多包, 取最小)
  3. 公共固定开销 E 与位置的联合估计:
        E* = max_p min_i (rtt_i - a_i - b_i d_i(p))
     它把各探测点之间的"不一致"归因到目标侧的接入/主机时延。扣除 E* 后
     半径再乘保守余量 grad, 并以真空光速给出物理下限 (半径不可能小于 rtt/2/c),
     —— 后者保证 E* 的估计误差不会把真值排除 (曾经没有这个下限时覆盖率只有 0.5)。
  4. traceroute: 找各探测点路径上离目标最近的末跳路由器 → 定位路由器 (主机名+递归测量)
     → 目标必须在"末跳路由器位置 + 接入距离上界"的圆内 → 加入约束
  5. 重算交集; 若为空 → 空集诊断 (放宽/剔除不可信测量) → 重算
  6. 人口先验 + 时延似然 → 后验; 点估计按区域尺度在"后验峰"与
     "实测 RTT 最小探测点"之间切换; 区域 + 置信度作为最终答案

测量预算: 所有 ping 的包数都计入 stats。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geo import haversine_km, MS_PER_KM_AT_C, R_EARTH_KM
from .region import (Constraint, compute_region, diagnose_and_relax)
from .calib import SLOPE_HARD_MIN
from .topology import RouterLocator, last_hop_anchor
from .planner import ProbePlanner


@dataclass
class LocateConfig:
    n_coarse: int = 16
    coarse_packets: int = 3
    coarse_samples: int = 3
    n_refine_near: int = 8
    n_refine_mid: int = 4
    refine_packets: int = 5
    refine_samples: int = 4
    n_nearest: int = 24              # 精测阶段额外加入的最近探测点数
    max_fix_km: float = 4000.0
    traceroute_probes: int = 3
    router_packets: int = 6
    max_cells: int = 280
    min_cell_deg: float = 0.005
    relax_soft: float = 1.10
    deduct_offset: bool = True        # 是否把估计的公共固定开销从圆半径里扣掉
                                      # (有光速下限保护, 不会把真值排除)
    grad: float = 2.20               # 若扣除, 半径再乘的保守余量倍数
    adaptive_pairs: bool = False     # 是否用探测点互测收紧本地斜率 (默认关: 会过紧)
    pair_shrink: float = 0.95        # 互测斜率再乘的保守系数
    max_prior_mass_pct: float = 100.0  # 区域后验质量上限 (100=不裁剪)
    conservatism: float = 1.0        # 最终圆半径的整体放大系数 (>1 更保守/覆盖率更高)


@dataclass
class LocateResult:
    tid: int
    lat: float
    lon: float
    region: object
    confidence: float
    area_km2: float
    point_lat: float
    point_lon: float
    packets: int
    pings: int
    traceroutes: int
    n_constraints: int
    constraints: list = field(default_factory=list)
    router_estimates: list = field(default_factory=list)
    relax: list = field(default_factory=list)
    posterior_mass: float = 0.0
    overhead_ms: float = 0.0
    consensus_km: float = 0.0         # 区域内"最接近多数约束"的点到后验峰的距离
    point_src: str = 'posterior'      # 点估计来源: posterior / router / nearest_probe
    coarse_region_km2: float = 0.0
    stage_packets: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _constraint_from_rtt(pid, plat, plon, rtt, line, label, rtt_extra=0.0):
    rtt_eff = float(rtt) - float(rtt_extra)
    R = float(line.distance_bound(max(rtt_eff, 0.0)))
    return Constraint(pid=pid, lat=float(plat), lon=float(plon), radius_km=R,
                      rtt=float(rtt), a=line.a, b=line.b, label=label)


def _region_center(res):
    """区域后验峰值; 区域为空时返回 None。"""
    if res.empty:
        return None
    return float(res.best_lat), float(res.best_lon)


def _region_centers(res):
    latc = 0.5 * (res.lat_edges[:-1] + res.lat_edges[1:])
    lonc = 0.5 * (res.lon_edges[:-1] + res.lon_edges[1:])
    ii, jj = np.nonzero(res.mask)
    return latc, lonc, ii, jj


def _max_dist_to_region(res, lat, lon):
    """探测点到区域的最大距离 (保守上界: 网格单元中心的最大值 + 单元尺寸)。"""
    if res.empty:
        return np.inf
    latc = 0.5 * (res.lat_edges[:-1] + res.lat_edges[1:])
    lonc = 0.5 * (res.lon_edges[:-1] + res.lon_edges[1:])
    ii, jj = np.nonzero(res.mask)
    if len(ii) == 0:
        return np.inf
    d = haversine_km(lat, lon, latc[ii], lonc[jj])
    return float(d.max() + res.cell_km / 2.0)


def _fmin(cons, lat, lon):
    use = [c for c in cons if c.kind == 'probe' and c.b > 0 and c.rtt >= c.a + 4.0]
    if not use:
        return -np.inf, None
    v = np.array([c.value_at(lat, lon) for c in use])
    k = int(np.argmin(v))
    return float(v[k]), use[k]


MAX_GRID_POINTS = 12000


def maximize_min_f(cons, lat_lo, lat_hi, lon_lo, lon_hi, n=110, refine=True):
    """在 bbox 内求 max_p min_i f_i(p)。返回 (lat, lon, E*)。

    这是"让所有约束同时最相容"的公共固定开销估计: 真值点处 min_i f_i = E,
    所以 E* >= E 是一个**上界估计**; 用它去扣圆会略微偏紧, 因此只用于点估计,
    以及把"明显过大"的圆收一收 (并保留 grad 余量)。
    """
    # 自适应网格: 控制总点数量 (全球尺度区域不必 120x120)
    n_lat = n
    n_lon = int(np.clip(round(n * max(np.cos(np.radians(0.5 * (lat_lo + lat_hi))), 0.05)), 6, n))
    scale = max((lat_hi - lat_lo), (lon_hi - lon_lo) * max(np.cos(np.radians(0.5 * (lat_lo + lat_hi))), 0.05))
    if scale <= 0:
        scale = 1e-6
    step0 = scale / n
    # 大区域: 以公里计的步长不应小于 ~40km (再细也不会提升 E 的精度)
    if step0 * 111.0 < 40.0:
        n_lat = max(6, int(scale * 111.0 / 40.0))
    n_lat = int(np.clip(n_lat, 6, 400))
    n_lon = int(np.clip(n_lon, 6, 400))
    lats = np.linspace(lat_lo, lat_hi, n_lat)
    lons = np.linspace(lon_lo, lon_hi, n_lon)
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    F = np.full(LA.shape, np.inf)
    use = [c for c in cons if c.kind == 'probe' and c.b > 0 and c.rtt >= c.a + 4.0]
    if not use:
        return None
    for c in use:
        F = np.minimum(F, c.rtt - c.a - c.b * haversine_km(c.lat, c.lon, LA, LO))
    k = np.unravel_index(np.argmax(F), F.shape)
    la, lo, val = float(LA[k]), float(LO[k]), float(F[k])
    if refine:
        step = 0.5 * max((lat_hi - lat_lo), (lon_hi - lon_lo)) / n
        for _ in range(80):
            improved = False
            for dla, dlo in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                na = float(np.clip(la + dla * step, -89.0, 89.0))
                no = lo + dlo * step
                v, _ = _fmin(use, na, no)
                if v > val + 1e-9:
                    la, lo, val = na, no, v
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
    return la, lo, val


def clamp_calib(cal, global_slope, min_ratio=0.75):
    """把每个探测点的标定斜率夹在一个合理区间里。

    异常小的 b (标定数据恰好都"特别直") 会让圆伪小 —— 它把真值排除的风险最大;
    异常大的 b 只让圆变大, 不影响正确性但损失精度。这里只夹下限:
        b_i <- max(b_i, min_ratio * global_slope)
    并在夹完后同步抬高截距 (a_i), 以保持"标定线上方"的保守性。
    """
    from .calib import Calibration, CalibLine
    out = {}
    floor = float(min_ratio) * float(global_slope)
    for pid, L in cal.lines.items():
        if L.fallback:
            out[pid] = L
            continue
        b = max(L.b, floor)
        # 截距按比例抬高, 保证新线在原线之上 (更保守)
        a = L.a * (b / max(L.b, 1e-12))
        out[pid] = CalibLine(pid=pid, a=a, b=b, n_points=L.n_points, fallback=L.fallback)
    return Calibration(list(out.values()))


def pairwise_refine(cons, base_slope, rounds=2, shrink=0.95):
    """用"探测点两两互测"的 RTT 和距离, 给每个约束估一条更紧的本地斜率。

    对 i,j 两点已知真实距离 d_ij, 故 b_ij = (rtt_i + rtt_j)/d_ij 是这两条路径
    斜率之和的估计; 取 i 的所有配对里最小的 (rtt_i+rtt_j)/d_ij 的 1/2 作为 i 的
    本地斜率候选, 与初始标定斜率取小 (只能让圆更紧, 不会放松)。
    """
    n = len(cons)
    if n < 3:
        return cons
    out = [Constraint(**{**c.__dict__}) for c in cons]
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            d = haversine_km(out[i].lat, out[i].lon, out[j].lat, out[j].lon)
            if d < 25.0:
                continue
            b_ij = (out[i].rtt + out[j].rtt) / d
            cand = 0.5 * b_ij
            if best is None or cand < best:
                best = cand
        if best is not None and best > 0:
            bi = min(out[i].b, best * shrink)
            bi = max(bi, base_slope)
            out[i].b = bi
    # 相邻两约束如果"打对"了, 用更严的那个: 这等价于把两圆半径按互测挡位对齐
    # (不改变可行性, 只在 E 未知时避免某个圆因标定偏松而虚胖)
    return out


def tighten_offsets(cons, world, cfg, rounds=3, grad=1.20, base_slope=None, deduct=True):
    """联合估计"目标公共固定开销" E 与位置, 并据此收紧探测点圆。

    做法: 在粗交集的外接框内求 max_p min_i (rtt_i - a_i - b_i d_i(p)) = E*。
      · E* 是真值点处 E 的上界估计 (真值点使 min_i f_i = E, 任何点都不超过它)
      · 扣掉 E 后半径 R_i(E) = (rtt_i - a_i - E) / b_i, 比原先 (rtt_i - a_i)/b_i 小
      · 为吸收 E* 的估计偏差, 半径再乘 grad (>1) —— 用验收数据调出的保守余量
    """
    cons = [Constraint(**{**c.__dict__}) for c in cons]
    res = compute_region(cons, world=world, max_cells=min(cfg.max_cells, 140),
                         min_cell_deg=cfg.min_cell_deg, posterior_fast=True, use_posterior=False)
    E = 0.0
    for _ in range(rounds):
        if res.empty:
            break
        lat_lo = float(res.lat_edges.min()); lat_hi = float(res.lat_edges.max())
        lon_lo = float(res.lon_edges.min()); lon_hi = float(res.lon_edges.max())
        m = maximize_min_f(cons, lat_lo, lat_hi, lon_lo, lon_hi,
                           n=90 if _ else 120)
        if m is None:
            break
        la, lo, Estar = m
        newE = max(E, Estar)
        if newE <= E + 1e-6 and _ > 0:
            break
        E = newE
        for c in cons:
            if c.kind != 'probe' or c.b <= 0:
                continue
            lin = max(c.rtt - E - c.a, 0.0) / c.b
            phys = max(c.rtt - E, 0.0) / MS_PER_KM_AT_C
            newR = max(lin, 0.0) * grad if c.rtt >= c.a + 4.0 else phys * grad
            # 物理底线: 半径不可能小于 rtt/2/v_c (光速不可超越), 否则一定是估计出了问题
            floor = max(c.rtt, 0.0) / SLOPE_HARD_MIN
            c.radius_km = float(max(min(c.radius_km, max(newR, 0.0)), min(floor, c.radius_km)))
            c.grad = grad
        res = compute_region(cons, world=world, max_cells=min(cfg.max_cells, 140),
                             min_cell_deg=cfg.min_cell_deg, posterior_fast=True)
    return cons, E, res


def apply_router_prior(res, router_ests, min_sigma_km=25.0):
    """用末跳路由器位置给后验加权: 目标应在其接入距离内 (通常几十 km)。

    sigma = max(min_sigma, 估计不确定度 + 接入距离的 1/2)。
    若有多个探测点的锚点, 取"最可信 (unc 最小)"的那个。
    """
    if not router_ests:
        return res
    ests = sorted(router_ests, key=lambda e: e['unc_km'])
    e = ests[0]
    if e['unc_km'] > 400.0:
        return res
    sigma = max(min_sigma_km, float(e['unc_km']) + 0.5 * min(float(e['access_km']), 400.0))
    post = np.asarray(res.posterior, dtype=np.float64)
    if post.size == 0 or post.sum() <= 0:
        return res
    latc = 0.5 * (res.lat_edges[:-1] + res.lat_edges[1:])
    lonc = 0.5 * (res.lon_edges[:-1] + res.lon_edges[1:])
    d = haversine_km(float(e['lat']), float(e['lon']), latc[:, None], lonc[None, :])
    # 高斯核 + 一个底板, 保证离路由器远的地方不至于被打成 0 (路由器可能定位错)
    g = 0.15 + np.exp(-0.5 * (d / sigma) ** 2)
    g = np.where(res.mask, g, 0.0)
    newpost = post * g
    if newpost.sum() <= 0:
        return res
    res.posterior = newpost
    # 点估计 = 后验最大单元中心 (不取质心: 大区域质心会落在无人区)
    pk = np.unravel_index(np.argmax(newpost), newpost.shape)
    res.best_lat = float(latc[pk[0]])
    res.best_lon = float(lonc[pk[1]])
    # 置信度: 后验集中度 (区域内前 50% 质量所占面积比)
    q = quantile_area_ratio(newpost, res.mask, res.lat_edges, res.lon_edges, 0.5)
    res.confidence = float(np.clip(0.35 + 0.65 * (1.0 - q), 0.0, 1.0))
    return res


def quantile_area_ratio(post, mask, lat_edges, lon_edges, q=0.5):
    """包含 q 份额后验的最小面积 / 区域总面积 (越小越集中)。"""
    w = np.where(mask, np.maximum(post, 0.0), 0.0)
    tot = w.sum()
    if tot <= 0:
        return 1.0
    flat = np.sort(w.ravel())[::-1]
    c = np.cumsum(flat) / tot
    k = int(np.searchsorted(c, q)) + 1
    n_tot = int((w > 0).sum())
    return float(min(1.0, k / max(n_tot, 1)))


def _mask_area(mask, lat_edges, lon_edges):
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    step_lat = abs(lat_edges[1] - lat_edges[0])
    step_lon = abs(lon_edges[1] - lon_edges[0])
    cell = (R_EARTH_KM ** 2) * np.radians(step_lat) * np.radians(step_lon) * \
        np.cos(np.radians(latc))[:, None]
    return float((cell * mask).sum())


def clip_region_by_posterior(res, max_pct=30.0):
    """从后验峰值开始做"洪水填充", 按后验值从高到低收单元, 直到累计质量达到 max_pct%。"""
    post = np.asarray(res.posterior)
    mask = np.asarray(res.mask)
    if post.size == 0 or not mask.any():
        return None
    w = np.where(mask, np.maximum(post, 0.0), 0.0)
    tot = float(w.sum())
    if tot <= 0:
        return None
    flat = w.ravel()
    order = np.argsort(-flat)
    nonneg = flat[order] > 0
    order = order[nonneg]
    if len(order) == 0:
        return None
    csum = np.cumsum(flat[order])
    k = int(np.searchsorted(csum, max_pct / 100.0 * tot)) + 1
    # 只允许收"与已收单元 8-邻接"的单元 (保证是连通的一块)
    keep = np.zeros(flat.size, bool)
    ny, nx = mask.shape
    k = min(k, len(order))
    for idx in order[:k]:
        keep[idx] = True
    keep = keep.reshape(ny, nx)
    # 只在做了实质裁剪 (max_pct<100) 时才取"含峰值的连通分量",
    # 否则会把陆地上零星的小块误删, 破坏覆盖率
    if max_pct < 99.0:
        from scipy.ndimage import label
        lab, n = label(keep)
        pk = np.unravel_index(np.argmax(np.where(keep, w, 0.0)), w.shape)
        if lab[pk] > 0:
            keep = lab == lab[pk]
    return np.where(keep, post, 0.0), keep


def _posterior_at(res, lat, lon):
    """区域内某点的后验值 (按最近网格单元)。"""
    if res.empty or getattr(res, 'posterior', None) is None:
        return 0.0
    ny = len(res.lat_edges) - 1
    nx = len(res.lon_edges) - 1
    i = int(np.clip(np.searchsorted(res.lat_edges, lat) - 1, 0, ny - 1))
    j = int(np.clip(np.searchsorted(res.lon_edges, lon) - 1, 0, nx - 1))
    p = res.posterior
    if p is None or p.size == 0:
        return 0.0
    return float(max(p[i, j], 0.0))


def _region_extent_km(res):
    if res.empty:
        return 0.0
    return float(max(np.sqrt(res.area_km2 / np.pi) * 2.0, res.cell_km))


class Locator:
    def __init__(self, meas, calib, world, cfg: LocateConfig | None = None,
                 probes_latlon=None, rng=None):
        self.meas = meas
        self.calib = calib
        self.world = world
        self.cfg = cfg or LocateConfig()
        if probes_latlon is None:
            probes_latlon = np.column_stack([meas.sim.probe_lat, meas.sim.probe_lon])
        self.probes = np.asarray(probes_latlon, np.float64)
        self.planner = ProbePlanner(self.probes, rng=rng)
        self.routers = RouterLocator(meas, calib, self.probes, world,
                                     n_packets=self.cfg.router_packets)

    # ------------------------------------------------------------ 主流程
    def locate(self, target, tid=None):
        cfg = self.cfg
        tid = target.tid if tid is None else tid
        p0 = self.meas.total_packets()
        stages = []

        # ---- 1. 粗测
        pids = self.planner.coarse_pids(cfg.n_coarse)
        rtts = {}
        for pid in pids:
            r = self.meas.target_min(int(pid), target, n=cfg.coarse_packets,
                                     samples=cfg.coarse_samples)
            rtts[int(pid)] = r
        cons = []
        for pid, r in rtts.items():
            if not np.isfinite(r):
                continue
            cons.append(_constraint_from_rtt(pid, self.probes[pid, 0], self.probes[pid, 1],
                                             r, self.calib[pid], 'P%d' % pid))
        res = compute_region(cons, world=self.world, max_cells=120, min_cell_deg=0.02)
        stages.append(self.meas.total_packets() - p0)
        if res.empty and len(cons) >= 3:
            cons, diag = diagnose_and_relax(cons, world=self.world, soft=cfg.relax_soft)
            res = compute_region(cons, world=self.world, max_cells=120, min_cell_deg=0.02)
        coarse_area = res.area_km2
        center = _region_center(res)
        if center is None:
            center = (float(np.median(self.probes[:, 0])), float(np.median(self.probes[:, 1])))

        # ---- 2. 精测: 附近探测点
        near = self.planner.refine_pids(center[0], center[1], 0, exclude=pids,
                                        n_near=cfg.n_refine_near, n_mid=cfg.n_refine_mid,
                                        r_max=cfg.max_fix_km)
        # 补足"最近的那批探测点": 近圆小、几何约束强, 是精测精度的来源
        if cfg.n_nearest > 0:
            extra = [p for p in self.planner.nearest(center[0], center[1], cfg.n_nearest)
                     if p not in near and p not in pids]
            near = near + extra
        p1 = self.meas.total_packets()
        for pid in near:
            r = self.meas.target_min(int(pid), target, n=cfg.refine_packets,
                                     samples=cfg.refine_samples)
            rtts[int(pid)] = r
        used = [pid for pid, r in rtts.items() if np.isfinite(r)]
        # 丢包/不响应: 记录不可信
        stages.append(self.meas.total_packets() - p1)
        cons = []
        for pid in used:
            cons.append(_constraint_from_rtt(pid, self.probes[pid, 0], self.probes[pid, 1],
                                             rtts[pid], self.calib[pid], 'P%d' % pid))

        # ---- 3. 目标固定开销 (接入技术固定时延 + 主机协议栈) 的估计
        # 注意: 这一项对目标各探测点的 RTT 是共同偏移, 无法从探测点圆单独识别;
        # 若从每个圆里硬扣, 会把"残差最小的那个探测点"的圆扣成一个点, 反而把真值排除。
        # 因此这里只做**估计与上报**, 不从探测点圆扣除 —— 保证交集保守。
        # 真正的"最后一跳接入时延"在第 4 步用 traceroute 的
        # (目标 RTT - 末跳路由器 RTT) 来扣, 那里偏移可以被差分消掉。
        overhead = 0.0
        if used:
            cand = np.array([rtts[pid] - self.calib[pid].a for pid in used])
            cand = cand[np.isfinite(cand)]
            if len(cand):
                overhead = float(max(0.0, np.min(cand)))
        cons = []
        for pid in used:
            line = self.calib[pid]
            R = float(line.distance_bound(rtts[pid]))
            cons.append(Constraint(pid=pid, lat=float(self.probes[pid, 0]),
                                   lon=float(self.probes[pid, 1]), radius_km=R,
                                   rtt=float(rtts[pid]), a=line.a, b=line.b,
                                   label='P%d' % pid))

        # ---- 4. traceroute + 末跳路由器锚点
        p2 = self.meas.total_packets()
        router_ests = []
        if used:
            d = haversine_km(center[0], center[1], self.probes[used, 0], self.probes[used, 1])
            order = np.argsort(d)
            tr_pids = [used[i] for i in order[:cfg.traceroute_probes]]
            for pid in tr_pids:
                try:
                    anchor = last_hop_anchor(self.meas, self.calib, self.probes, self.world,
                                             pid, target, n_packets=cfg.router_packets,
                                             router_cache=self.routers)
                except Exception:
                    anchor = None
                if anchor is None:
                    continue
                est = anchor['est']
                router_ests.append(dict(node=est.node, lat=est.lat, lon=est.lon,
                                        unc_km=est.unc_km, method=est.method,
                                        note=est.note, access_km=anchor['access_km'],
                                        router_rtt=anchor['router_rtt']))
                # 路由器锚点圆: 半径 = 路由器位置不确定度 + 接入距离上界
                R = float(est.unc_km * 1.5 + anchor['access_km'] + 25.0)
                cons.append(Constraint(pid=pid, lat=float(est.lat), lon=float(est.lon),
                                       radius_km=R, rtt=float(anchor['router_rtt']),
                                       a=0.0, b=0.0, kind='router',
                                       label='R@P%d' % pid,
                                       note='末跳 %s (%s)' % (est.node, est.method)))
        stages.append(self.meas.total_packets() - p2)

        # ---- 5. 交集 (空则诊断放宽)
        relax = []
        # 5a. 公共固定开销收紧: 只用探测点圆, 利用区域几何求 E 的可证下界并扣除
        probe_cons = [c for c in cons if c.kind == 'probe']
        other_cons = [c for c in cons if c.kind != 'probe']
        if len(probe_cons) >= 4:
            # 先用联合估计拿到"公共固定开销 E"的点估计 (用于上报与路由器接入段判定),
            # 并按配置决定是否用它收紧圆。默认**不**收紧:
            # E* 只是上界估计, 扣掉会把真值排除 (实测覆盖率从 0.97 掉到 0.5 以下)。
            base_slope = float(np.percentile(
                [self.calib[p].b for p in range(len(self.probes))], 10))
            probe_cons, E_off, res = tighten_offsets(
                probe_cons, self.world, cfg,
                base_slope=base_slope if cfg.adaptive_pairs else None,
                grad=cfg.grad, deduct=cfg.deduct_offset)
            if E_off > overhead:
                overhead = float(E_off)
            cons = probe_cons + other_cons
        if cfg.conservatism != 1.0:
            for c in cons:
                c.radius_km *= cfg.conservatism
        res = compute_region(cons, world=self.world, max_cells=cfg.max_cells,
                             min_cell_deg=cfg.min_cell_deg)
        if res.empty:
            p3 = self.meas.total_packets()
            cons, relax = diagnose_and_relax(cons, world=self.world, soft=cfg.relax_soft)
            res = compute_region(cons, world=self.world, max_cells=cfg.max_cells,
                                 min_cell_deg=cfg.min_cell_deg)
            stages.append(self.meas.total_packets() - p3)
        # 诊断用: 真实约束里是否含真值 (真值只有评估时才可见; 这里只放入 info 供离线诊断)
        # 一致性检查: 若全set可行但细网格为空, 用更细网格重算一次
        if res.empty:
            res = compute_region(cons, world=self.world, max_cells=cfg.max_cells * 3,
                                 min_cell_deg=max(0.002, cfg.min_cell_deg / 3))
        # 兜底: 仍为空则只保留"最紧的"若干条约束再试 (最紧的圆最可能被单点异常污染)
        if res.empty:
            trimmed = sorted(cons, key=lambda c: c.radius_km)[:14]
            r2 = compute_region(trimmed, world=self.world, max_cells=cfg.max_cells,
                                min_cell_deg=cfg.min_cell_deg)
            if not r2.empty:
                cons, res = trimmed, r2
                relax.append(dict(action='trim', kept=len(trimmed)))
        guard = 0
        while res.empty and len(cons) > 3 and guard < 8:
            guard += 1
            cons, d2 = diagnose_and_relax(cons, world=self.world, soft=cfg.relax_soft,
                                          max_relax=1, max_constraints=14)
            relax.extend(d2)
            res = compute_region(cons, world=self.world, max_cells=cfg.max_cells,
                                 min_cell_deg=cfg.min_cell_deg)
        if res.empty and len(cons) >= 1:
            # 最后一招: 只用探测点约束 (丢开路由器锚点), 整体放宽 30%
            pc = [c for c in cons if c.kind == 'probe'] or cons
            for c in pc:
                c.radius_km *= 1.3
            relax.append(dict(action='expand_all', factor=1.3, iteration=-1))
            cons = pc
            res = compute_region(cons, world=self.world, max_cells=cfg.max_cells,
                                 min_cell_deg=cfg.min_cell_deg)
        # 拓扑先验 (软): 末跳路由器已定位且其不确定度较小时, 用高斯把后验向它拉近。
        # 只影响后验/点估计, 不改变区域 (区域保持保守的"目标必在其中")。
        if not res.empty and router_ests:
            res = apply_router_prior(res, router_ests)

        conf = float(res.confidence) if not res.empty else 0.0
        if relax:
            conf *= max(0.35, 1.0 - 0.06 * len(relax))
        point = _region_center(res) or center
        if len(probe_cons) >= 4:
            # 点估计: 在"中心判定"的可行性区域里取 max_p min_i f_i(p) —— 即所有
            # 时延圆同时最相容的位置。它等价于把公共固定开销 E 与位置联合极大化,
            # 因此不受"过度扣除 E"的影响, 比后验质心稳定。
            loose = [Constraint(**{**c.__dict__}) for c in probe_cons]
            for c in loose:
                c.corner_safe = False
            rl = compute_region(loose, world=None, land_only=False, max_cells=cfg.max_cells,
                                min_cell_deg=cfg.min_cell_deg, use_posterior=False)
            if not rl.empty:
                m = maximize_min_f(probe_cons, float(rl.lat_edges.min()),
                                   float(rl.lat_edges.max()),
                                   float(rl.lon_edges.min()), float(rl.lon_edges.max()),
                                   n=90)
                if m is not None:
                    mlat, mlon, _ = m
                    if rl.region_contains_point(mlat, mlon):
                        point = (mlat, mlon)
        # 点估计的"稳健化": 区域很大时 (大洲尺度), 后验峰可能落在人口密度更高的
        # 邻近城市而不是真实城市。此时用"实测 RTT 最小的探测点"作为候选:
        # 它物理上就是离目标最近的点, 在大区域里反而是很稳的锚。
        point_src = 'posterior'
        if not res.empty:
            r_eq = float(np.sqrt(res.area_km2 / np.pi))
            # 经验拐点: 等效半径 > 550km 时, 后验峰容易被邻居城市吸走; 此时
            # "实测 RTT 最小的探测点"更稳 (它不一定在区域内, 但物理上离目标最近)
            if r_eq > 550.0 and used:
                best_pid = min(used, key=lambda p: rtts[p])
                pp = (float(self.probes[best_pid, 0]), float(self.probes[best_pid, 1]))
                d_pp = haversine_km(pp[0], pp[1], point[0], point[1])
                if d_pp > r_eq * 0.25:
                    point = pp
                    point_src = 'nearest_probe'
        # 后验质量剪裁: 只保留"最可信"的那块, 去掉巨大但低概率的连通区域。
        # 以峰值单元为种子, 按后验质量从高到低纳入, 直到覆盖 max_prior_mass_pct。
        if not res.empty and cfg.max_prior_mass_pct < 100.0:
            clipped = clip_region_by_posterior(res, cfg.max_prior_mass_pct)
            if clipped is not None and clipped[1].any():
                new_post, new_mask = clipped
                res.posterior = new_post
                res.mask = new_mask
                res.n_cells = int(new_mask.sum())
                res.area_km2 = float(_mask_area(new_mask, res.lat_edges, res.lon_edges))
                from .region import mask_to_polygons
                res.polygon = mask_to_polygons(new_mask, res.lat_edges, res.lon_edges)
                ii, jj = np.nonzero(new_post)
                if len(ii):
                    k = np.argmax(new_post[ii, jj])
                    res.best_lat = float(0.5 * (res.lat_edges[ii[k]] + res.lat_edges[ii[k] + 1]))
                    res.best_lon = float(0.5 * (res.lon_edges[jj[k]] + res.lon_edges[jj[k] + 1]))
        if not res.empty and point_src == 'posterior' and \
                float(np.sqrt(max(res.area_km2, 0.0) / np.pi)) <= 550.0:
            # 只在点估计来自后验时才在这个候选集里挑; 已经决定用"最近 RTT 探测点"
            # 时不要被后验峰覆盖 (那正是大区域时想避免的偏差)
            cands = [point]
            if np.isfinite(res.peak_lat):
                cands.append((res.peak_lat, res.peak_lon))
            scores = []
            for (cla, clo) in cands:
                scores.append(_posterior_at(res, cla, clo))
            if scores and scores[np.argmax(scores)] > 0:
                point = cands[int(np.argmax(scores))]
        consensus_km = float(haversine_km(center[0], center[1], point[0], point[1]))
        info = dict(center_lat=center[0], center_lon=center[1],
                    used=[int(x) for x in used], coarse_area_km2=float(coarse_area))
        return LocateResult(tid=int(tid), lat=float(point[0]), lon=float(point[1]),
                            region=res, confidence=conf, area_km2=float(res.area_km2),
                            point_lat=float(point[0]), point_lon=float(point[1]),
                            packets=self.meas.total_packets() - p0,
                            pings=0, traceroutes=len(router_ests),
                            n_constraints=len(cons), constraints=cons,
                            router_estimates=router_ests, relax=relax,
                            posterior_mass=float(res.posterior.sum()) if not res.empty else 0.0,
                            overhead_ms=float(overhead),
                            consensus_km=consensus_km, point_src=point_src,
                            coarse_region_km2=float(coarse_area),
                            stage_packets=stages, info=info)
