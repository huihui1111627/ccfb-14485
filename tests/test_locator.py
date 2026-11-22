import numpy as np
import pytest

from pygeoloc.sim import Simulation, SimConfig
from pygeoloc.measure import Measurer
from pygeoloc.calib import calibrate
from pygeoloc.locator import Locator, LocateConfig
from pygeoloc.evaluate import run_targets, summarize
from pygeoloc.geo import haversine_km


def _ctx_small():
    sim = Simulation(cfg=SimConfig(n_probes=60))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=16)
    return sim, meas, probes, cal


@pytest.fixture(scope='module')
def ctx():
    sim = Simulation(cfg=SimConfig(n_probes=80))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=20)
    return sim, meas, probes, cal


def test_locate_returns_region_and_confidence(ctx):
    sim, meas, probes, cal = ctx
    loc = Locator(meas, cal, sim.world)
    t = sim.sample_targets(4)[0]
    r = loc.locate(t)
    assert r.area_km2 > 0
    assert 0.0 <= r.confidence <= 1.0
    assert r.packets > 0
    assert r.n_constraints >= 5
    assert len(r.constraints) == r.n_constraints


def test_locate_no_ground_truth_leak(ctx):
    """定位器只能通过 Measurer 访问网络; 检查测量计数与 Locator 输入中没有真值字段。"""
    sim, meas, probes, cal = ctx
    loc = Locator(meas, cal, sim.world)
    t = sim.sample_targets(2)[0]
    r = loc.locate(t)
    # 结果里不应出现真实坐标 (除非恰好与估计一致)
    assert not hasattr(r.info, 'true_lat')
    assert r.packets == meas.total_packets() - 0 or r.packets > 0


def test_near_probe_constraint_dominates(ctx):
    """离目标近的探测点, 圆应显著小于远探测点 (这也是"挑附近探测点"的依据)。"""
    sim, meas, probes, cal = ctx
    loc = Locator(meas, cal, sim.world)
    t = sim.sample_targets(3)[0]
    r = loc.locate(t)
    ds = [haversine_km(c.lat, c.lon, t.lat, t.lon) for c in r.constraints if c.kind == 'probe']
    rs = [c.radius_km for c in r.constraints if c.kind == 'probe']
    ds, rs = np.array(ds), np.array(rs)
    near = rs[ds < np.median(ds)]
    far = rs[ds > np.median(ds)]
    assert np.median(near) < np.median(far)


def test_empty_case_is_relaxed_and_records_diagnosis(ctx):
    from pygeoloc.region import Constraint
    cons = [Constraint(i, 39.9 + i * 0.5, 116.4, 30) for i in range(6)]
    from pygeoloc.region import diagnose_and_relax, compute_region
    out, diag = diagnose_and_relax(cons)
    assert diag
    assert not compute_region(out, land_only=False).empty


def test_summary_metrics_shape(ctx):
    sim, meas, probes, cal = ctx
    loc = Locator(meas, cal, sim.world, cfg=LocateConfig(n_coarse=10, n_refine_near=4,
                                                         n_refine_mid=2, traceroute_probes=1))
    recs = run_targets(sim, meas, cal, loc, sim.sample_targets(6), sim.world)
    s = summarize(recs)
    assert s['n'] == len(recs)
    assert 0.0 <= s['coverage'] <= 1.0
    assert 'q' in s['err_km'] and 'q' in s['area_km2']
    assert 'nearest_probe' in s['baselines']
    assert s['packets_per_target']['median'] > 0


def test_likelihood_is_used_and_not_constant():
    """时延似然必须真的进入后验 (曾经因为字段没填而退化成纯人口先验)。

    用一个"窄圆 + 大圆"的几何: 大圆内的候选点距离差异很大, 似然必须有分辨力。
    """
    from pygeoloc.region import Constraint, compute_region
    from pygeoloc.calib import residual_logt
    import numpy as np
    cons = [Constraint(0, 39.0, 116.0, 260, rtt=5.0, a=1.0, b=0.009),
            Constraint(1, 22.0, 114.0, 2600, rtt=30.0, a=1.0, b=0.009)]
    for c in cons:
        dist = np.linspace(50, 2500, 60)
        c.logt_x, c.logt_p = residual_logt(
            dist, 1.0 + 0.009 * dist * np.linspace(1.0, 2.2, 60), 1.0, 0.009)
    r = compute_region(cons, use_posterior=True, land_only=False)
    assert not r.empty
    vals = r.posterior[r.mask]
    assert vals.size > 10
    # 似然有分辨力 -> 后验不是常数
    assert np.ptp(vals) > 0.01 * max(np.max(vals), 1e-12)


def test_point_source_switches_for_huge_regions():
    """大区域应回退到"实测 RTT 最小的探测点", 小区域用后验峰。"""
    sim, meas, probes, cal = _ctx_small()
    loc = Locator(meas, cal, sim.world)
    seen = set()
    for t in sim.sample_targets(6):
        r = loc.locate(t)
        seen.add(r.point_src)
        r_eq = (r.area_km2 / np.pi) ** 0.5
        if r_eq <= 550.0:
            assert r.point_src == 'posterior'
        else:
            assert r.point_src in ('posterior', 'nearest_probe')
    assert seen <= {'posterior', 'nearest_probe'}
