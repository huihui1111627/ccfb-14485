import numpy as np
import pytest

from pygeoloc.calib import fit_lower_envelope, CalibLine, Calibration, calibrate, \
    SLOPE_HARD_MIN, BOUND_MARGIN
from pygeoloc.geo import MS_PER_KM_AT_C


def test_envelope_stays_below_all_points():
    rng = np.random.default_rng(1)
    dist = rng.uniform(50, 6000, 60)
    rtt = 3 + dist * 0.009 + rng.uniform(0, 3, 60)
    a, b, n, fb = fit_lower_envelope(dist, rtt)
    assert n >= 40
    # 标定线在所有数据点下方 (留 3% 余量, 所以允许极小越界)
    gap = rtt - (a + b * dist)
    assert gap.min() > -0.05 * rtt.max()
    assert b >= SLOPE_HARD_MIN


def test_envelope_slope_not_above_physical_ratio_absurdly():
    dist = np.linspace(100, 5000, 30)
    rtt = 2 + dist * 0.008         # 干净数据: 斜率 0.008 ms/km
    a, b, n, fb = fit_lower_envelope(dist, rtt)
    assert 0.006 < b < 0.03


def test_distance_bound_falls_back_to_light_speed():
    L = CalibLine(pid=0, a=5.0, b=0.009, n_points=40)
    # 低于截距: 走真空光速兜底 (而不是 0)
    assert float(L.distance_bound(1.0)) > 100.0
    # 正常区间: 线性换算
    assert abs(float(L.distance_bound(5.0 + 0.009 * 1000)) - 1000 * BOUND_MARGIN) < 2


def test_calibration_roundtrip_dict():
    cal = Calibration([CalibLine(0, 1.0, 0.008, 30), CalibLine(1, 2.0, 0.009, 25)])
    d = cal.as_dict()
    cal2 = Calibration.from_dict(d)
    assert cal2[0].a == 1.0 and cal2[1].b == 0.009


def test_calibrate_is_conservative_on_synthetic_pairs():
    """用真实仿真: 标定线应压在所有探测点互测数据下方, 并给出更紧的圆。"""
    from pygeoloc.sim import Simulation, SimConfig
    from pygeoloc.measure import Measurer
    from pygeoloc.geo import haversine_km
    sim = Simulation(cfg=SimConfig(n_probes=40))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=25)
    # 互测数据全部在线上方 (允许 5% 噪声余量)
    viol = 0
    tot = 0
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            d = haversine_km(probes[i, 0], probes[i, 1], probes[j, 0], probes[j, 1])
            if d < 200:
                continue
            r = meas.probe_rtt(i, j, n=5)
            if not np.isfinite(r):
                continue
            tot += 1
            if r < cal[i].a + cal[i].b * d * 0.95:
                viol += 1
    assert tot > 100
    assert viol / tot < 0.05


def test_calibrated_circles_smaller_than_light_speed():
    from pygeoloc.sim import Simulation, SimConfig
    from pygeoloc.measure import Measurer
    from pygeoloc.calib import calibrate
    sim = Simulation(cfg=SimConfig(n_probes=40))
    meas = Measurer(sim)
    probes = np.column_stack([sim.probe_lat, sim.probe_lon])
    cal = calibrate(meas, probes, max_partners=20)
    rtt = 60.0
    naive = rtt * 0.5 / MS_PER_KM_AT_C
    for pid in (0, 5, 10):
        assert float(cal[pid].distance_bound(rtt)) < naive
