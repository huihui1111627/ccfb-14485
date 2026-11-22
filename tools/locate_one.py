#!/usr/bin/env python3
"""locate_one.py — 定位单个目标并打印研判结论 (演示 / 手工核对用)。

用法:
  python tools/locate_one.py --seed 7 --index 3
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.locator import Locator, LocateConfig
from pygeoloc.report import summary_text, cities_in_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--n', type=int, default=40)
    a = ap.parse_args()
    sim = Simulation(cfg=SimConfig(seed=a.seed))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes)
    loc = Locator(meas, cal, sim.world, cfg=LocateConfig())
    tg = sim.sample_targets(a.n)
    t = tg[a.index % len(tg)]
    r = loc.locate(t)
    print(summary_text(r, sim.world, truth=(t.lat, t.lon)))
    print('\n约束明细 (半径/真实距离):')
    for c in sorted(r.constraints, key=lambda c: c.radius_km)[:12]:
        d = float(np.sqrt((c.lat - t.lat) ** 2 + (c.lon - t.lon) ** 2)) if False else None
        print('  %-10s %-6s R=%8.1f km  rtt=%6.2f ms' % (c.label, c.kind, c.radius_km, c.rtt))
    print('\n末跳路由器:')
    for e in r.router_estimates:
        print('  node %-6s %-15s unc=%6.0f km access<=%6.0f km %s' % (
            e['node'], e['method'], e['unc_km'], e['access_km'], e['note']))
    print('\n测量: %d 包, %d 次 traceroute' % (r.packets, r.traceroutes))


if __name__ == '__main__':
    main()
