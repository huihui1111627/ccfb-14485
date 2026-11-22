import numpy as np
import pytest

from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.topology import NameResolver, RouterLocator, last_hop_anchor
from pygeoloc.geo import haversine_km


@pytest.fixture(scope='module')
def ctx():
    sim = Simulation(cfg=SimConfig(n_probes=60))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=18)
    return sim, meas, probes, cal


def test_name_resolver_iata(ctx):
    sim, *_ = ctx
    r = NameResolver(sim.world)
    h = r.hint_from_name('shl-cr2.isp-core.example.net')
    assert h is not None
    # SHL 是西隆机场 (印度), 城市线索应在机场附近
    assert h.unc_km < 60
    assert h.source in ('iata', 'city')


def test_name_resolver_city_name(ctx):
    sim, *_ = ctx
    r = NameResolver(sim.world)
    h = r.hint_from_name('cr7.hyderabad.example.net')
    assert h is not None
    assert h.lat != 0


def test_name_resolver_unknown(ctx):
    sim, *_ = ctx
    r = NameResolver(sim.world)
    assert r.hint_from_name('ae3.core24.example.net') is None
    assert r.hint_from_name('') is None


def test_router_localization_error(ctx):
    sim, meas, probes, cal = ctx
    loc = RouterLocator(meas, cal, probes, sim.world, n_packets=5)
    errs = []
    for t in sim.sample_targets(8):
        hops, _ = meas.route(0, t)
        if not hops:
            continue
        h = hops[-1]
        est = loc.locate(h.node, name=h.name)
        tl = sim.true_router_pos(h.node)
        errs.append(haversine_km(est.lat, est.lon, tl[0], tl[1]))
    assert np.median(errs) < 300


def test_last_hop_anchor_yields_usable_radius(ctx):
    sim, meas, probes, cal = ctx
    loc = RouterLocator(meas, cal, probes, sim.world, n_packets=5)
    ok = 0
    for t in sim.sample_targets(8):
        a = last_hop_anchor(meas, cal, probes, sim.world, 0, t, router_cache=loc)
        if a is None:
            continue
        e = a['est']
        d = haversine_km(e.lat, e.lon, t.lat, t.lon)
        R = e.unc_km * 1.5 + a['access_km'] + 25.0
        ok += (d <= R)
    assert ok >= 5


def test_router_locator_cache(ctx):
    sim, meas, probes, cal = ctx
    loc = RouterLocator(meas, cal, probes, sim.world, n_packets=4)
    t = sim.sample_targets(1)[0]
    hops, _ = meas.route(0, t)
    a = loc.locate(hops[-1].node, name=hops[-1].name)
    n0 = meas.total_packets()
    b = loc.locate(hops[-1].node, name=hops[-1].name)
    assert meas.total_packets() == n0     # 命中缓存
    assert a.lat == b.lat
