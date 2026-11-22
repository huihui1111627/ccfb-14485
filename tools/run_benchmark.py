#!/usr/bin/env python3
"""run_benchmark.py — 批量验收 (可并行, 结果可复现)。

用法:
  python tools/run_benchmark.py --n 3000 --workers 8 --seed 7 --out out/bench_seed7.json

说明:
  * 仿真拓扑用固定 seed 构建一次; 每个 worker 用同一个网络、不同的目标/探测点状态
    (测量噪声用 worker 各自的随机流), 保证结果可复现。
  * 输出 JSON: 汇总指标 + 每个目标的明细 (真值/估计/区域面积/覆盖率/包数/基线)。
  * 同时打印人类可读的汇总表。
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.locator import Locator, LocateConfig
from pygeoloc.evaluate import run_targets, summarize, save_json, records_to_json


def worker(args):
    seed, n_targets, cfg_dict, net_seed, calib_path, tseed = args
    cfg = SimConfig(seed=net_seed)
    sim = Simulation(cfg=cfg)
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes)
    lcfg = LocateConfig(**cfg_dict)
    loc = Locator(meas, cal, sim.world, cfg=lcfg)
    rng = np.random.default_rng(tseed)
    tg = sim.sample_targets(n_targets, rng=rng)
    import sys as _sys
    def _prog(done, total, rec):
        if done % 25 == 0 or done == total:
            print('[w%d] %d/%d' % (tseed % 100, done, total), file=_sys.stderr, flush=True)
    recs = run_targets(sim, meas, cal, loc, tg, sim.world, cfg=lcfg, progress=_prog)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='out/bench.json')
    ap.add_argument('--max-cells', type=int, default=280)
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--no-router', action='store_true')
    ap.add_argument('--conservatism', type=float, default=None)
    a = ap.parse_args()

    cfg = LocateConfig(max_cells=a.max_cells)
    if a.no_refine:
        cfg.n_refine_near = 0; cfg.n_refine_mid = 0
    if a.no_router:
        cfg.traceroute_probes = 0
    if a.conservatism is not None:
        cfg.conservatism = a.conservatism
    cfg_dict = cfg.__dict__.copy()

    t0 = time.time()
    if a.workers <= 1:
        recs = worker((a.seed, a.n, cfg_dict, a.seed, None, a.seed + 1))
    else:
        import multiprocessing as mp
        # 按目标数切块
        per = int(np.ceil(a.n / a.workers))
        jobs = [(a.seed, min(per, a.n - i * per), cfg_dict, a.seed, None,
                 a.seed + 1000 + i) for i in range(a.workers) if i * per < a.n]
        with mp.get_context('fork').Pool(len(jobs)) as pool:
            chunks = pool.map(worker, jobs)
        recs = [r for ch in chunks for r in ch]
    dt = time.time() - t0
    summ = summarize(recs)
    summ['elapsed_s'] = dt
    summ['config'] = cfg_dict
    summ['seed'] = a.seed
    save_json(dict(summary=summ, records=records_to_json(recs)), a.out)

    def pr(k, v):
        print('  %-24s %s' % (k, v))
    print('== %d targets, %.1fs (%.2fs/target) ==' % (len(recs), dt, dt / max(len(recs), 1)))
    e = summ['err_km']
    print('误差距离 (km): p50 %.0f  p75 %.0f  p90 %.0f  p95 %.0f' % (
        e['q'][50], e['q'][75], e['q'][90], e['q'][95]))
    print('  命中率: <=10km %.1f%%  <=25km %.1f%%  <=50km %.1f%%  <=100km %.1f%%  <=250km %.1f%%' % (
        100 * e['within_10km'], 100 * e['within_25km'], 100 * e['within_50km'],
        100 * e['within_100km'], 100 * e['within_250km']))
    print('覆盖率 (真值在区域内): %.3f' % summ['coverage'])
    ar = summ['area_km2']
    print('区域面积 km²: p50 %.3g  p75 %.3g  p90 %.3g  几何均值 %.3g' % (
        ar['q'][50], ar['q'][75], ar['q'][90], ar['geo_mean']))
    p = summ['packets_per_target']
    print('测量: 每目标包数 中位 %.0f 均值 %.0f p90 %.0f ; traceroute 中位 %.0f' % (
        p['median'], p['mean'], p['p90'], summ['traceroutes_per_target']['median']))
    print('基线:')
    for name, b in summ['baselines'].items():
        q = b['err_km']['q']
        print('  %-16s err p50 %6.0f p90 %8.0f  cover %s  area p50 %s' % (
            name, q[50], q[90],
            ('%.3f' % b['coverage']) if b['coverage'] == b['coverage'] else 'n/a',
            ('%.3g' % b['area_km2']['q'][50]) if b['area_km2']['q'][50] == b['area_km2']['q'][50] else 'n/a'))
    print('置信度分箱:')
    for c in summ['calibration']:
        print('  conf %.2f-%.2f n=%4d 实际命中 %.3f' % (c['lo'], c['hi'], c['n'], c['hit']))
    print('结果已写入', a.out)


if __name__ == '__main__':
    main()
