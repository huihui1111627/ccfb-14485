import json
import os
import tempfile
import numpy as np

from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.locator import Locator, LocateConfig
from pygeoloc.webexport import build_case, world_layers, export_case, export_world
from pygeoloc.png import heatmap_rgba, write_rgba


def _ctx():
    sim = Simulation(cfg=SimConfig(n_probes=40))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=16)
    loc = Locator(meas, cal, sim.world, cfg=LocateConfig(n_coarse=10, n_refine_near=4,
                                                         n_refine_mid=2, traceroute_probes=1))
    return sim, loc


def test_case_payload_has_circles_region_and_truth():
    sim, loc = _ctx()
    t = sim.sample_targets(2)[0]
    r = loc.locate(t)
    c = build_case(r, target_truth=(t.lat, t.lon))
    assert c['circles'], '至少应有一个探测点圆'
    for f in c['circles']:
        assert len(f['path']) > 3
        assert 0 <= f['path'][0][0] <= 1
    assert c['truth'] == [t.lat, t.lon]
    assert c['view']['w'] > 0 and c['view']['h'] > 0
    assert c['packets'] > 0


def test_export_writes_valid_json():
    sim, loc = _ctx()
    t = sim.sample_targets(1)[0]
    r = loc.locate(t)
    with tempfile.TemporaryDirectory() as d:
        p = export_case(r, sim.world, (t.lat, t.lon), out_dir=d, name='case')
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        assert data['tid'] == r.tid
        wp = export_world(sim.world, out_dir=d)
        with open(wp, encoding='utf-8') as f:
            w = json.load(f)
        assert len(w['land']) > 100
        assert len(w['cities']) > 100


def test_heatmap_png_is_valid_png():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'h.png')
        post = np.random.default_rng(0).random((30, 40))
        write_rgba(p, heatmap_rgba(post, post > 0.1))
        raw = open(p, 'rb').read()
        assert raw[:8] == b'\x89PNG\r\n\x1a\n'
        assert len(raw) > 100
