#!/usr/bin/env python3
"""build_data.py — 把公开源数据裁剪压缩成本项目自带的轻量数据包 (pygeoloc/data/*.npz)

数据源 (均为公开数据, 下载地址见 README):
  ne_50m_land.geojson                 Natural Earth 1:50m 陆地多边形 (海岸线)
  nearest-cities / geonames cities15000  全球人口>=1.5万的城市 (人口/坐标)
  OurAirports airports.csv            机场 (含 IATA 三字码/城市名/规模/是否有定期航班)

用法:
  python tools/build_data.py --land ne50.geojson --airports airports.csv --cities cities15000.txt

输出:
  world_land.npz     陆地多边形环 (float32 经纬度)
  world_cities.npz   城市: 经度/纬度/人口
  world_airports.npz 机场: 经度/纬度/是否大型机场
  world_topology.npz 网络: 机场间邻接距离、城市邻居、城市->最近机场
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'pygeoloc', 'data')
R_EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_land(path):
    g = json.load(open(path))
    rings = []
    for f in g['features']:
        geom = f['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for poly in polys:
            for ring in poly:
                a = np.asarray(ring, dtype=np.float64)
                if len(a) >= 4:
                    # 去掉重复的闭合点, 统一为标准环
                    if np.allclose(a[0], a[-1]):
                        a = a[:-1]
                    # 环内经度展开成连续序列: 跨 180 经线处 ±360, 保证相邻点不跳变
                    a = a.copy()
                    a[:, 0] = np.rad2deg(np.unwrap(np.deg2rad(a[:, 0])))
                    rings.append(a)
    lens = np.array([len(r) for r in rings], dtype=np.int64)
    pts = np.concatenate(rings, axis=0) if rings else np.zeros((0, 2))
    return pts.astype(np.float32), lens


def load_cities(path, min_pop=15000):
    lat, lon, pop, names = [], [], [], []
    for line in open(path, encoding='utf-8'):
        f = line.rstrip('\n').split('\t')
        if len(f) < 15:
            continue
        try:
            p = int(f[14]) if f[14] else 0
        except ValueError:
            p = 0
        if p < min_pop:
            continue
        lat.append(float(f[4])); lon.append(float(f[5])); pop.append(p)
        names.append(f[2])
    return (np.asarray(lat, np.float64), np.asarray(lon, np.float64),
            np.asarray(pop, np.float64), names)


def load_airports(path, min_type=('large_airport', 'medium_airport')):
    rows = list(csv.DictReader(open(path, encoding='utf-8', newline='')))
    lat, lon, big, code, mun, country = [], [], [], [], [], []
    for r in rows:
        if r['type'] not in min_type or not r['iata_code']:
            continue
        if r.get('scheduled_service') != 'yes':
            continue
        lat.append(float(r['latitude_deg'])); lon.append(float(r['longitude_deg']))
        big.append(1.0 if r['type'] == 'large_airport' else 0.0)
        code.append(r['iata_code']); mun.append(r.get('municipality') or '')
        country.append(r.get('iso_country') or '')
    return (np.asarray(lat, np.float64), np.asarray(lon, np.float64),
            np.asarray(big, np.float64), code, mun, country)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--land'); ap.add_argument('--airports'); ap.add_argument('--cities')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.land:
        pts, lens = load_land(a.land)
        np.savez_compressed(os.path.join(OUT, 'world_land.npz'), pts=pts, lens=lens)
        print('land rings=%d pts=%d' % (len(lens), len(pts)))

    air = None
    if a.airports:
        alat, alon, abig, acode, amun, ac = load_airports(a.airports)
        air = (alat, alon, abig)
        np.savez_compressed(os.path.join(OUT, 'world_airports.npz'),
                            lat=alat.astype(np.float32), lon=alon.astype(np.float32),
                            big=abig.astype(np.float32),
                            code=np.array(acode), municipality=np.array(amun),
                            country=np.array(ac))
        print('airports=%d' % len(alat))

    if a.cities:
        clat, clon, cpop, names = load_cities(a.cities)
        if air is not None:
            alat, alon, _ = air
            c2a = np.empty(len(clat), np.int64)
            for i in range(0, len(clat), 512):
                sl = slice(i, i + 512)
                d = haversine_km(clat[sl][:, None], clon[sl][:, None], alat[None, :], alon[None, :])
                c2a[sl] = d.argmin(axis=1)
        else:
            c2a = np.full(len(clat), -1, np.int64)
        np.savez_compressed(os.path.join(OUT, 'world_cities.npz'),
                            lat=clat.astype(np.float32), lon=clon.astype(np.float32),
                            pop=cpop.astype(np.float32), c2a=c2a,
                            name=np.array(names))
        print('cities=%d' % len(clat))


if __name__ == '__main__':
    main()
