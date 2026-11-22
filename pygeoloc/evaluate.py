"""evaluate.py — 批量验收: 对几千个位置已知的仿真目标定位并统计指标。

指标
----
* 误差距离分布: 点估计与真值的 haversine 距离 (p50/p75/p90/p95/最大, 以及不同阈值命中率)
* 覆盖率: 真实位置落在给出区域内的比例
* 区域面积: 中位数/几何平均/分位数
* 测量开销: 每目标包数、traceroute 次数
* 基线对比: ①"最近探测点当答案" ②"只用最近探测点圆交集" ③"真空光速圆"
* 置信度校准: 按置信度分箱的实际命中率

长跑 (几千个目标) 用 tools/run_benchmark.py 并行执行; 这里提供单进程可复现的入口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import numpy as np

from .geo import haversine_km, MS_PER_KM_AT_C, MS_PER_KM_FIBER
from .region import Constraint, compute_region
from .planner import ProbePlanner


@dataclass
class TargetRecord:
    tid: int
    kind: str
    true_lat: float
    true_lon: float
    est_lat: float
    est_lon: float
    err_km: float
    covered: bool
    area_km2: float
    confidence: float
    packets: int
    traceroutes: int
    n_constraints: int
    relax: int
    regime: str = ''
    base_errs: dict = field(default_factory=dict)
    base_cover: dict = field(default_factory=dict)
    base_areas: dict = field(default_factory=dict)


def _quantiles(x, qs=(5, 25, 50, 75, 90, 95, 99)):
    x = np.asarray([v for v in x if np.isfinite(v)], np.float64)
    if len(x) == 0:
        return {q: float('nan') for q in qs}
    return {q: float(np.percentile(x, q)) for q in qs}


def summarize(records):
    err = np.array([r.err_km for r in records], np.float64)
    area = np.array([r.area_km2 for r in records], np.float64)
    cov = np.array([r.covered for r in records], np.float64)
    pk = np.array([r.packets for r in records], np.float64)
    tr = np.array([r.traceroutes for r in records], np.float64)
    conf = np.array([r.confidence for r in records], np.float64)
    q = _quantiles(err)
    aq = _quantiles(area)
    out = {
        'n': len(records),
        'err_km': dict(q=q, mean=float(np.mean(err)),
                       within_10km=float(np.mean(err <= 10)),
                       within_25km=float(np.mean(err <= 25)),
                       within_50km=float(np.mean(err <= 50)),
                       within_100km=float(np.mean(err <= 100)),
                       within_250km=float(np.mean(err <= 250)),
                       within_500km=float(np.mean(err <= 500))),
        'coverage': float(np.mean(cov)),
        'area_km2': dict(q=aq, geo_mean=float(np.exp(np.mean(np.log(np.maximum(area, 1.0)))))),
        'packets_per_target': dict(median=float(np.median(pk)), mean=float(np.mean(pk)),
                                   p90=float(np.percentile(pk, 90)),
                                   total=float(np.sum(pk))),
        'traceroutes_per_target': dict(median=float(np.median(tr)), mean=float(np.mean(tr))),
        'confidence': dict(median=float(np.median(conf)), mean=float(np.mean(conf))),
    }
    # 置信度校准
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    cal = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 3:
            cal.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                            hit=float(np.mean(cov[m])), mean_conf=float(np.mean(conf[m]))))
    out['calibration'] = cal
    # 基线
    base = {}
    for name in ('nearest_probe', 'nearest_circles', 'fiber_speed'):
        be = [r.base_errs.get(name, np.nan) for r in records]
        bc = [r.base_cover.get(name, np.nan) for r in records]
        ba = [r.base_areas.get(name, np.nan) for r in records]
        be = np.array(be, np.float64); bc = np.array(bc, np.float64); ba = np.array(ba, np.float64)
        ok = np.isfinite(be)
        if ok.sum() == 0:
            continue
        cov_v = bc[np.isfinite(bc)]
        base[name] = dict(err_km=dict(q=_quantiles(be[ok])),
                          coverage=float(cov_v.mean()) if len(cov_v) else float('nan'),
                          area_km2=dict(q=_quantiles(ba)))
    out['baselines'] = base
    return out


def baseline_constraints(kind, pids, rtts, calib, probes, E=0.0):
    cons = []
    for pid in pids:
        r = rtts.get(pid, np.inf)
        if not np.isfinite(r):
            continue
        lat, lon = float(probes[pid, 0]), float(probes[pid, 1])
        if kind == 'fiber_speed':
            # 最朴素的物理上界: 真空光速, 且不扣除任何固定开销
            R = float(r) * 0.5 / MS_PER_KM_AT_C
            a = b = 0.0
        elif kind == 'nearest_circles':
            line = calib[pid]
            R = float(line.distance_bound(r - E))
            a, b = line.a, line.b
        else:
            raise ValueError(kind)
        cons.append(Constraint(pid, lat, lon, R, float(r), a, b, label='B%d' % pid))
    return cons


def run_targets(sim, meas, calib, locator, targets, world, cfg=None,
                baseline_coarse=14, baseline_K=(2, 4, 8), progress=None):
    """对一批目标跑完整定位 + 基线。返回 TargetRecord 列表。

    基线选点**不看真值**: 用与主方法粗测阶段相同的"几何散布"探测点集合,
    再按实测 RTT 大小决定"最近探测点"。基线与主方法共享同一批测量结果,
    这样比较的是"从同样的测量里能榨出多少信息", 而不是测量预算。
    """
    from .locator import LocateConfig
    from .region import Constraint, compute_region
    cfg = cfg or LocateConfig()
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    recs = []
    for k, t in enumerate(targets):
        res = locator.locate(t)
        rec = TargetRecord(
            tid=t.tid, kind=t.kind, true_lat=t.lat, true_lon=t.lon,
            est_lat=res.point_lat, est_lon=res.point_lon,
            err_km=float(haversine_km(res.point_lat, res.point_lon, t.lat, t.lon)),
            covered=bool(res.region.contains(t.lat, t.lon)),
            area_km2=float(res.area_km2), confidence=float(res.confidence),
            packets=int(res.packets), traceroutes=int(res.traceroutes),
            n_constraints=int(res.n_constraints), relax=len(res.relax))
        # ---- 基线: 用**主方法实际测过的那批探测点**, 只换聚合方式。
        # 这样比较的是"同样的测量, 不同算法能榨出多少", 而不是测量预算。
        rtts = {}
        for c in res.constraints:
            if c.kind == 'probe' and c.label.startswith('P'):
                try:
                    rtts[int(c.label[1:])] = float(c.rtt)
                except ValueError:
                    pass
        valid = sorted([p for p, r in rtts.items() if np.isfinite(r)], key=lambda p: rtts[p])
        if valid:
            best = valid[0]      # 实测 RTT 最小 = 物理上最近的探测点
            rec.base_errs['nearest_probe'] = float(
                haversine_km(probes[best, 0], probes[best, 1], t.lat, t.lon))
            rec.base_cover['nearest_probe'] = np.nan
            rec.base_areas['nearest_probe'] = np.nan
        for K in baseline_K:
            pids_k = valid[:K]
            if len(pids_k) < 2:
                continue
            for kind in ('nearest_circles', 'fiber_speed'):
                cons = baseline_constraints(kind, pids_k, rtts, calib, probes)
                if len(cons) < 2:
                    continue
                r = compute_region(cons, world=world, max_cells=120, min_cell_deg=0.01)
                name = '%s_K%d' % (kind, K)
                rec.base_errs[name] = float(haversine_km(r.best_lat, r.best_lon, t.lat, t.lon)) \
                    if not r.empty else float('nan')
                rec.base_cover[name] = bool(r.contains(t.lat, t.lon)) if not r.empty else False
                rec.base_areas[name] = float(r.area_km2) if not r.empty else float('nan')
        for name in list(rec.base_errs.keys()):
            if name.endswith('_K2'):
                short = name[:-3]
                rec.base_errs[short] = rec.base_errs[name]
                rec.base_cover[short] = rec.base_cover.get(name, np.nan)
                rec.base_areas[short] = rec.base_areas.get(name, np.nan)
        recs.append(rec)
        if progress is not None:
            progress(k + 1, len(targets), rec)
    return recs


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def records_to_json(records):
    out = []
    for r in records:
        out.append(dict(tid=r.tid, kind=r.kind,
                        true=[r.true_lat, r.true_lon],
                        est=[r.est_lat, r.est_lon],
                        err_km=r.err_km, covered=r.covered,
                        area_km2=r.area_km2, confidence=r.confidence,
                        packets=r.packets, traceroutes=r.traceroutes,
                        relax=r.relax, base_errs=r.base_errs,
                        base_cover=r.base_cover, base_areas=r.base_areas))
    return out
