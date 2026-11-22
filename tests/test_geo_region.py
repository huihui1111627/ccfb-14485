import numpy as np
import pytest

from pygeoloc.geo import (haversine_km, destination_point, polygon_area_km2,
                          point_in_ring, lonlat_to_mercator, mercator_to_lonlat,
                          R_EARTH_KM, MS_PER_KM_AT_C, fast_dist_km)
from pygeoloc.region import (Constraint, compute_region, diagnose_and_relax,
                             circle_circle_points, mask_to_polygons)


def test_haversine_known_distance():
    # 北京-伦敦 约 8140 km
    d = haversine_km(39.9, 116.4, 51.5, -0.13)
    assert 8050 < d < 8250


def test_destination_roundtrip():
    lat, lon = destination_point(10.0, 20.0, 45.0, 100.0)
    assert abs(haversine_km(10.0, 20.0, lat, lon) - 100.0) < 1e-6


def test_mercator_roundtrip():
    lon, lat = mercator_to_lonlat(*lonlat_to_mercator(116.4, 39.9))
    assert abs(lon - 116.4) < 1e-9 and abs(lat - 39.9) < 1e-9


def test_polygon_area_one_degree_cell():
    area = polygon_area_km2([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
    expect = R_EARTH_KM ** 2 * np.radians(1.0) ** 2
    assert abs(area - expect) / expect < 0.01


def test_point_in_ring_simple():
    ring_lon = np.array([0.0, 1.0, 1.0, 0.0])
    ring_lat = np.array([0.0, 0.0, 1.0, 1.0])
    assert point_in_ring(0.5, 0.5, ring_lon, ring_lat)
    assert not point_in_ring(1.5, 0.5, ring_lon, ring_lat)


def test_fast_dist_accuracy():
    for lat0, lon0 in ((0, 0), (45, 100), (-30, -70)):
        lat, lon = destination_point(lat0, lon0, 73.0, 800.0)
        approx = float(fast_dist_km(lat0, lon0, lat, lon))
        exact = float(haversine_km(lat0, lon0, lat, lon))
        assert abs(approx - exact) / exact < 0.01


def test_region_intersection_contains_truth():
    cons = [Constraint(0, 39.9, 116.4, 150),
            Constraint(1, 31.2, 121.5, 1300),
            Constraint(2, 23.1, 113.3, 2100)]
    r = compute_region(cons)
    assert not r.empty
    assert r.contains(39.9, 116.4)
    assert r.contains(39.0, 117.5)      # 北京东南约 130km, 三个圆内
    assert not r.contains(45.0, 126.0)  # 哈尔滨在北边第一个圆外
    assert not r.contains(20.0, 100.0)  # 缅甸: 不在北京圆内
    assert r.area_km2 > 0
    assert r.polygon


def test_region_empty_when_disjoint_and_relax_recovers():
    cons = [Constraint(0, 39.9, 116.4, 5, label='a'),
            Constraint(1, 39.9, 116.6, 5, label='b')]
    r = compute_region(cons, land_only=False)
    assert r.empty
    c2, diag = diagnose_and_relax(cons)
    assert diag
    r2 = compute_region(c2, land_only=False)
    assert not r2.empty


def test_circle_intersection_points_on_both_circles():
    pts = circle_circle_points(0, 0, 500, 0, 5.0, 500)
    assert len(pts) == 2
    for la, lo in pts:
        d1 = haversine_km(0, 0, la, lo)
        d2 = haversine_km(0, 5.0, la, lo)
        assert abs(d1 - 500) < 5 and abs(d2 - 500) < 5
    # 两圆相离: 无交点
    assert circle_circle_points(0, 0, 100, 0, 5.0, 100) == []


def test_region_never_excludes_truth_for_random_containers():
    """几何不变量: 若真值满足所有约束, 网格区域必须包含它 (离散化不得切掉边界)。"""
    rng = np.random.default_rng(3)
    for _ in range(30):
        tlat, tlon = rng.uniform(-50, 60), rng.uniform(-170, 170)
        cons = []
        for k in range(4):
            plat, plon = tlat + rng.normal(0, 3), tlon + rng.normal(0, 3)
            d = haversine_km(tlat, tlon, plat, plon)
            cons.append(Constraint(k, plat, plon, d * rng.uniform(1.0, 1.3)))
        r = compute_region(cons, land_only=False, max_cells=150, min_cell_deg=0.01)
        assert r.contains(tlat, tlon), (tlat, tlon, r.n_cells)
