#!/usr/bin/env python3
"""build_landmask.py — 由陆地多边形生成 0.05° 全球陆地栅格 (pygeoloc/data/land_mask.npz)。

做法: 把所有环的边 (跨 180° 经线的边拆成两段) 收集起来, 对每一行纬度做扫描线,
按偶奇规则填充。结果按位打包 (7200x3600 bit ≈ 3.2MB, 压缩后 ~120KB)。
用途: 生成仿真目标时剔除海面、区域后验的海陆判定, 比多边形逐点测试快 ~1000 倍。
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'pygeoloc', 'data')
RES = 0.05  # 度


def collect_edges(path):
    g = json.load(open(path))
    X1, Y1, X2, Y2 = [], [], [], []
    for f in g['features']:
        geom = f['geometry']
        polys = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for poly in polys:
            for ring in poly:
                p = np.asarray(ring, np.float64)
                if len(p) < 4:
                    continue
                a, b = p[:-1], p[1:]
                dx = b[:, 0] - a[:, 0]
                for i in np.nonzero(np.abs(dx) <= 180.0)[0]:
                    X1.append(a[i, 0]); Y1.append(a[i, 1]); X2.append(b[i, 0]); Y2.append(b[i, 1])
                for i in np.nonzero(np.abs(dx) > 180.0)[0]:
                    x1, y1 = a[i]; x2, y2 = b[i]
                    if x1 > 0:       # 向西跨 +180
                        t = (180.0 - x1) / dx[i]; ym = y1 + t * (y2 - y1)
                        X1 += [x1, -180.0]; Y1 += [y1, ym]; X2 += [180.0, x2]; Y2 += [ym, y2]
                    else:            # 向东跨 -180
                        t = (-180.0 - x1) / dx[i]; ym = y1 + t * (y2 - y1)
                        X1 += [x1, 180.0]; Y1 += [y1, ym]; X2 += [-180.0, x2]; Y2 += [ym, y2]
    return (np.asarray(X1), np.asarray(Y1), np.asarray(X2), np.asarray(Y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--land', required=True)
    a = ap.parse_args()
    X1, Y1, X2, Y2 = collect_edges(a.land)
    nx, ny = 7200, 3600
    mask = np.zeros((ny, nx), dtype=bool)
    for row in range(ny):
        lat = -90.0 + (row + 0.5) * (180.0 / ny)
        cond = (Y1 > lat) != (Y2 > lat)
        if not cond.any():
            continue
        xs = X1[cond] + (lat - Y1[cond]) * (X2[cond] - X1[cond]) / (Y2[cond] - Y1[cond])
        xs = np.sort(xs)
        if len(xs) % 2:
            xs = xs[:-1]
        lo = np.clip(np.ceil((xs[0::2] + 180.0) / RES).astype(np.int64), 0, nx)
        hi = np.clip(np.ceil((xs[1::2] + 180.0) / RES).astype(np.int64), 0, nx)
        # 用差分数组批量填充 (避免逐区间赋值)
        diff = np.zeros(nx + 1, np.int32)
        np.add.at(diff, lo, 1)
        np.add.at(diff, hi, -1)
        mask[row] = np.cumsum(diff[:-1]) > 0
    np.savez_compressed(os.path.join(OUT, 'land_mask.npz'),
                        bits=np.packbits(mask.reshape(-1)),
                        shape=np.array([ny, nx], np.int64), res=np.array([RES]))
    print('landmask %dx%d land_frac=%.4f' % (ny, nx, mask.mean()))


if __name__ == '__main__':
    main()
