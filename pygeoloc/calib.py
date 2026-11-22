"""calib.py — 每个探测点的时延-距离下包络标定线。

原理
----
物理上, 单程传播时延 d 满足  d >= dist / v_fiber  (真空光速是不可超越的上界,
光纤中更慢)。但只用真空光速得到的圆半径 R = rtt/2 * v_c 太大。
每个探测点可以和它"看得到"的其他探测点互测 RTT, 同时这些探测点的位置已知,
于是得到一批 (距离, RTT) 数据点。真实目标满足:
    rtt >= 2*dist/v_eff(探测点)   (v_eff <= v_fiber, 由路径绕行/设备决定)
我们把一条直线 y = a + b x 压在所有这些数据点的**下方** (最小二乘 + 强制向下平移),
于是由 rtt 反解的距离 R = (rtt - a)/b 是目标真实距离的**上界** —— 圆仍然是保守的,
但比真空光速圆小得多 (b 通常比 1/v_c 大 2 倍以上)。

实现细节
--------
* 只用足够远的点 (>= MIN_KM), 近处点的 RTT 被接入/处理时延主导, 斜率无信息。
* 斜率取"稳健下斜率"与""物理下限"之间: b >= 1/v_c (任何时延都不可能对应更快的速度)。
* 截距 a >= 0; 下包络平移量 = max over points of (y_i - (a + b x_i)) 的负数, 保证
  a + b x <= y 对每个标定点成立, 并留 1% 余量防止测量噪声把真值排除。
* 若标定点不足, 退化为纯物理下界 a=0, b=1/v_c。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geo import MS_PER_KM_AT_C


@dataclass
class CalibLine:
    pid: int
    a: float            # ms  (截距, >=0)
    b: float            # ms/km (斜率, 含绕路/设备)
    n_points: int
    fallback: bool = False
    logt_x: np.ndarray | None = None    # 标定残差 log t 的分位网格
    logt_p: np.ndarray | None = None    # 对应的 (相对) 密度

    @property
    def max_km(self):
        return np.inf

    def distance_bound(self, rtt_ms):
        """由 RTT 给出的距离上界 (km)。rtt 已扣除目标侧固定开销。

        若 rtt 低于标定线截距 (目标接得比标定点更"近"), 说明固定开销已被扣得差不多,
        此时退回真空光速给出的物理上界 (很松但一定成立), 而不是 0。
        """
        rtt_ms = np.asarray(rtt_ms, np.float64)
        lin = np.maximum(rtt_ms - self.a, 0.0) / self.b * BOUND_MARGIN
        phys = np.maximum(rtt_ms, 0.0) / SLOPE_HARD_MIN
        # 只有 RTT 明显高于标定线截距时, 线性外推才可信; 否则退回真空光速物理上界
        trust = rtt_ms >= self.a + 4.0
        return np.where(trust, lin, phys)

    def rtt_of(self, km):
        return self.a + self.b * np.asarray(km, np.float64)


SLOPE_HARD_MIN = MS_PER_KM_AT_C      # 1/真空光速: 不可能有更快的"时延-距离"关系
MIN_KM = 250.0                      # 标定用点的最小距离
RTT_OVERHEAD_ABSORB_KM = 110.0       # 标定线额外保守量 (换算成距离, 被截距吸收)
BOUND_MARGIN = 1.04                 # 最终距离上界再乘的余量 (统计意义上的保险)
SLOPE_SAFETY = 0.92                 # 斜率再乘的安全系数 (<1 => 圆更大)
CONSERVATIVE_PCT = 15.0             # 斜率取标定斜率的 (100-p) 分位数: 更小的斜率=更大的圆


def residual_logt(dist, rtt, a, b, nq=41):
    """标定残差 t = rtt / (a + b*d) 的对数分布, 返回 (logt_x, logt_p) 分位网格。

    用途: 目标时延似然。对目标而言, 若已知它扣掉的固定开销, 则
    t = (rtt - E) / (a + b*d) 在真值处接近 1; 这个分布给出"接近 1 到什么程度算正常"。
    """
    dist = np.asarray(dist, np.float64)
    rtt = np.asarray(rtt, np.float64)
    ok = np.isfinite(dist) & np.isfinite(rtt) & (dist > 1.0)
    if ok.sum() < 4:
        return None, None
    model = np.maximum(a + b * dist[ok], 1e-6)
    t = np.maximum(rtt[ok] / model, 1.0)
    lt = np.log(t)
    xs = np.linspace(float(lt.min()), float(lt.max()), nq)
    # 核密度 (gaussian), 归一化为相对密度
    bw = max(0.05, 0.5 * np.std(lt))
    dens = np.exp(-0.5 * ((xs[:, None] - lt[None, :]) / bw) ** 2).sum(axis=1)
    dens = dens / max(dens.max(), 1e-12)
    return xs, dens


def fit_lower_envelope(dist, rtt, margin=0.03, min_dist=MIN_KM):
    """拟合 y=a+b*x 使 a+b*x <= y 对所有点成立, 且尽量贴近下方。"""
    dist = np.asarray(dist, np.float64)
    rtt = np.asarray(rtt, np.float64)
    ok = np.isfinite(dist) & np.isfinite(rtt) & (dist > 0)
    dist, rtt = dist[ok], rtt[ok]
    n = len(dist)
    if n < 3 or np.ptp(dist) < 40:
        return 0.0, SLOPE_HARD_MIN, n, True
    # 只用较远点估斜率 (近点被接入/处理时延主导); 全部点用于压包络
    far = dist >= min_dist
    if far.sum() < 3:
        far = np.ones(len(dist), bool)
    dist_s, rtt_s = dist[far], rtt[far]
    # 分箱取最小值, 压制重尾噪声
    nb = int(np.clip(len(dist_s) // 3, 4, 24))
    edges = np.quantile(dist_s, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-6
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist_s >= lo) & (dist_s < hi)
        if m.sum() >= 1:
            bx.append(np.median(dist_s[m]))
            by.append(np.min(rtt_s[m]))
    bx = np.asarray(bx); by = np.asarray(by)
    # 稳健斜率: 用下 20% 分位的点做最小二乘 (斜率尽量小 => 圆尽量大, 保守)
    k = max(2, int(np.ceil(0.12 * len(bx))))
    # 用 (斜率候选, 截距) 组中"综合线最低"的若干点: by - s0*bx 最小
    s0 = max(np.median(by / bx), SLOPE_HARD_MIN)
    order = np.argsort(by - s0 * bx)[:k]
    # 过原点的稳健斜率 (几何上更稳, 避免远点截距为负时的耦合)
    b0 = float(np.median(by[order] / bx[order]))
    A = np.column_stack([np.ones(len(order)), bx[order]])
    # 在 b 固定为 b0 时最小二乘求 a
    a0 = float(np.median(by[order] - b0 * bx[order]))
    coef = np.array([a0, b0])
    a, b = float(coef[0]), float(coef[1])
    b = max(b, SLOPE_HARD_MIN * 1.02)          # 物理下限
    # 下包络平移: 保证所有原始数据点都在线上方
    viol = np.max(dist * b - rtt)              # >0 表示有点在线下方
    a = max(0.0, -viol) * (1.0 + margin)
    # 用分箱点再检查一遍
    viol2 = np.max(bx * b - by)
    a = max(a, -viol2 * (1.0 + margin), 0.0)
    return a, b, n, False


class Calibration:
    """全体探测点的标定线集合。"""

    def __init__(self, lines):
        self.lines = {l.pid: l for l in lines}

    def __getitem__(self, pid):
        return self.lines[pid]

    def as_dict(self):
        out = {}
        for pid, l in self.lines.items():
            d = dict(a=l.a, b=l.b, n=l.n_points, fallback=bool(l.fallback))
            if l.logt_x is not None:
                d['logt_x'] = np.asarray(l.logt_x).tolist()
                d['logt_p'] = np.asarray(l.logt_p).tolist()
            out[pid] = d
        return out

    @classmethod
    def from_dict(cls, d):
        out = []
        for k, v in d.items():
            out.append(CalibLine(pid=int(k), a=v['a'], b=v['b'], n_points=v['n'],
                                 fallback=bool(v.get('fallback', False)),
                                 logt_x=np.asarray(v['logt_x']) if 'logt_x' in v else None,
                                 logt_p=np.asarray(v['logt_p']) if 'logt_p' in v else None))
        return cls(out)


def calibrate(meas, probes, max_partners=45, pair_packets=5, min_km=MIN_KM,
              rng=None, local=True, local_min_km=40.0, local_fallback_km=MIN_KM):
    """为每个探测点拟合标定线。只与较近的伙伴互测 (远距离标定更贵也更保守)。

    probes: (n,2) 的 (lat,lon)。返回 Calibration。
    """
    from .geo import haversine_km
    rng = rng or np.random.default_rng(0)
    n = len(probes)
    lat, lon = probes[:, 0], probes[:, 1]
    lines = []
    for i in range(n):
        d = haversine_km(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        dmin = local_min_km if local else min_km
        cand = np.nonzero((d >= dmin) & np.isfinite(d))[0]
        if len(cand) > max_partners:
            # 距离分层采样: 兼顾不同量程
            order = cand[np.argsort(d[cand])]
            pick = np.linspace(0, len(order) - 1, max_partners).astype(int)
            cand = order[np.unique(pick)]
        rtts = []
        dists = []
        for j in cand:
            r = meas.probe_rtt(i, int(j), n=pair_packets)
            if np.isfinite(r):
                rtts.append(r); dists.append(float(d[j]))
        if local:
            la, lb, ln, lfb = fit_lower_envelope(dists, rtts, min_dist=local_min_km)
            if lfb:
                la, lb, ln, lfb = fit_lower_envelope(dists, rtts, min_dist=local_fallback_km)
            a, b, npts, fb = la, lb, ln, lfb
        else:
            a, b, npts, fb = fit_lower_envelope(dists, rtts)
        if len(dists) >= 4:
            # 稳健化: 直接取 (a,b) 线上方最近的那批点重新拟合小斜率
            # —— 保证标定线上方, 同时不让个别"绕得特别少"的链路把斜率压得过小
            rr = np.asarray(rtts, np.float64); dd = np.asarray(dists, np.float64)
            mind = local_min_km if local else min_km
            m = np.isfinite(rr) & (dd > 0) & (dd >= mind)
            if m.sum() < 4:
                m = np.isfinite(rr) & (dd > 0)
            rr, dd = rr[m], dd[m]
            if len(rr) >= 4:
                gap = dd * b - rr                  # >0: 该点在线下方 (需要抬线)
                k = max(2, int(np.ceil(CONSERVATIVE_PCT / 100.0 * len(rr))))
                sel = np.argsort(np.abs(gap))[:k]  # 最贴近线的点
                bx2 = np.median(dd[sel])
                if bx2 > 0:
                    b = min(b, float(np.median(rr[sel]) / bx2))
        b = min(b, 0.25) * SLOPE_SAFETY
        # 保守兜底: 把"每跳处理+路径绕行"折算成一个不随距离增长的截距 (km)
        # 目标自身的接入/主机时延由 locator 估计后扣除; 未扣除的部分在这里吸收
        a = a + RTT_OVERHEAD_ABSORB_KM * b
        lx, lp = residual_logt(np.asarray(dists, np.float64),
                               np.asarray(rtts, np.float64), a, b)
        lines.append(CalibLine(pid=i, a=a, b=b, n_points=npts, fallback=fb,
                               logt_x=lx, logt_p=lp))
    return Calibration(lines)
