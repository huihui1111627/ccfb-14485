#!/usr/bin/env python3
"""export_demo.py — 生成地图页面数据 (web/data/demo.json)。

用法:
  python tools/export_demo.py --cases 8 --out web/data/demo.json

每个 case 包含: 视野 (归一化 mercator)、探测点圆、路由器锚点圆、区域多边形、
后验热力 PNG (base64)、点估计、真值。
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.locator import Locator, LocateConfig
from pygeoloc.webexport import build_case, world_layers
from pygeoloc.geo import lonlat_to_mercator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='web/data/demo.json')
    ap.add_argument('--max-circles', type=int, default=40)
    a = ap.parse_args()
    sim = Simulation(cfg=SimConfig(seed=a.seed))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes)
    loc = Locator(meas, cal, sim.world, cfg=LocateConfig())
    cases = []
    for t in sim.sample_targets(a.cases):
        r = loc.locate(t)
        c = build_case(r, target_truth=(t.lat, t.lon), max_circles=a.max_circles)
        cases.append(c)
        print('case %d: %s err_covered=%s area=%.3g' % (
            t.tid, t.kind, r.region.contains(t.lat, t.lon), r.area_km2))
    probe_xy = []
    for p in sim.probes:
        x, y = lonlat_to_mercator(p.lon, p.lat)
        probe_xy.append([float(x), float(y)])
    data = dict(cases=cases, world=world_layers(sim.world), probes=probe_xy,
                meta=dict(n_probes=len(sim.probes), seed=a.seed))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print('written', a.out, os.path.getsize(a.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
