"""webexport.py — 生成地图页面数据。

页面是纯静态 HTML + 内联 JS (无 CDN 依赖), 数据来自本模块导出的 JSON/PNG。
图层:
  · 底图: 打包的陆地轮廓 + 城市点 (人口越大点越大)
  · 每个探测点: 标定圆 (半径由该探测点对目标的 RTT 反解), 颜色按"该圆是否参与交集"区分
  · 最终区域: 后验热力 (PNG 贴图) + 多边形边界
  · 点估计: marker; 真值: 另一 marker (仅验收时显示)
"""
from __future__ import annotations

import base64
import json
import os
import numpy as np

from .geo import lonlat_to_mercator, mercator_to_lonlat, circle_polyline, haversine_km
from .png import write_rgba, heatmap_rgba, resample_to_mercator


def _merc_path(lon, lat):
    x, y = lonlat_to_mercator(lon, lat)
    return [[float(a), float(b)] for a, b in zip(x, y)]


def region_layers(res):
    """区域边界 (归一化 mercator 坐标) + 热力贴图 (base64 PNG)。"""
    out = dict(polygons=[_merc_path(np.array([p[0] for p in ring]),
                                    np.array([p[1] for p in ring])) for ring in res.polygon],
               heatmap=None, heat_extent=None)
    if not res.empty and getattr(res, 'posterior', None) is not None and res.posterior.size:
        try:
            hm = resample_to_mercator(res.posterior, res.lat_edges, res.lon_edges,
                                      out_h=220, out_w=220)
            mask = resample_to_mercator(res.mask.astype(float), res.lat_edges,
                                        res.lon_edges, out_h=220, out_w=220) > 0.5
            rgba = heatmap_rgba(hm, mask)
            rgba = rgba[::-1]  # 图像第 0 行在顶部
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), '_geo_heat.png')
            write_rgba(tmp, rgba)
            with open(tmp, 'rb') as f:
                out['heatmap'] = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
            x0, y0 = lonlat_to_mercator(res.lon_edges[0], res.lat_edges[-1])
            x1, y1 = lonlat_to_mercator(res.lon_edges[-1], res.lat_edges[0])
            out['heat_extent'] = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
        except Exception:
            out['heatmap'] = None
    return out


def circles_geojson(constraints, max_circles=40):
    """每个约束画一个测地圆 (归一化 mercator 折线)。"""
    feats = []
    for c in constraints[:max_circles]:
        if c.kind != 'probe':
            continue
        lon, lat = circle_polyline(c.lat, c.lon, min(c.radius_km, 20000.0), n=72)
        feats.append(dict(kind='circle', label=c.label,
                          center=[float(c.lat), float(c.lon)],
                          radius_km=float(c.radius_km),
                          path=_merc_path(lon, lat)))
    # 路由器锚点圆 (半径=不确定度+接入上界) 用不同颜色
    for c in constraints:
        if c.kind != 'router':
            continue
        lon, lat = circle_polyline(c.lat, c.lon, min(c.radius_km, 20000.0), n=48)
        feats.append(dict(kind='router', label=c.label, center=[float(c.lat), float(c.lon)],
                          radius_km=float(c.radius_km), path=_merc_path(lon, lat)))
    return feats


def build_case(result, target_truth=None, max_circles=40, view_pad=0.25):
    """单个目标的页面数据。

    视野策略: 以区域为准; 只把"圆心离区域不远、且圆半径不至于大到离谱"的圆纳入视野,
    否则几十个大洲级圆会把视野撑到全球, 页面失去可读性 (那些圆仍会画出来, 但被裁掉)。
    """
    res = result.region
    plat, plon = result.point_lat, result.point_lon
    if not res.empty:
        lat0 = float(res.lat_edges.min()); lat1 = float(res.lat_edges.max())
        lon0 = float(res.lon_edges.min()); lon1 = float(res.lon_edges.max())
    else:
        lat0, lat1 = plat - 1.0, plat + 1.0
        lon0, lon1 = plon - 1.0, plon + 1.0
    # 视野 = 区域包围盒 (再留 40% 余量)。圆如果比视野大很多, 自然被裁掉一部分 ——
    # 那是正确的: 大圆对"局部在哪"没有信息量, 撑大视野只会让区域看不清。
    lat0 = min(lat0, plat); lat1 = max(lat1, plat)
    lon0 = min(lon0, plon); lon1 = max(lon1, plon)
    # 至少让"离估计点最近的几个圆"的圆心进入视野, 否则页面上一个圆都看不到
    probes = [c for c in result.constraints if c.kind == 'probe']
    probes.sort(key=lambda c: haversine_km(plat, plon, c.lat, c.lon))
    for c in probes[:6]:
        lat0 = min(lat0, c.lat); lat1 = max(lat1, c.lat)
        lon0 = min(lon0, c.lon); lon1 = max(lon1, c.lon)
    # 防止视野被拉到全球: 纬度跨度最多 55°, 经度最多 110°
    if lat1 - lat0 > 55.0:
        c0 = 0.5 * (lat0 + lat1); lat0, lat1 = c0 - 27.5, c0 + 27.5
    if lon1 - lon0 > 110.0:
        c0 = 0.5 * (lon0 + lon1); lon0, lon1 = c0 - 55.0, c0 + 55.0
    span_lat = max(lat1 - lat0, 0.05)
    span_lon = max(lon1 - lon0, 0.05)
    clat, clon = 0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1)
    half_lat = 0.5 * span_lat * (1 + view_pad)
    half_lon = 0.5 * span_lon * (1 + view_pad)
    lat0 = max(clat - half_lat, -70.0); lat1 = min(clat + half_lat, 80.0)
    lon0 = max(clon - half_lon, -180.0); lon1 = min(clon + half_lon, 180.0)
    x0, y0 = lonlat_to_mercator(lon0, lat1)
    x1, y1 = lonlat_to_mercator(lon1, lat0)
    view = dict(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0))
    x, y = lonlat_to_mercator(plat, plon)
    covered = None
    if target_truth is not None:
        covered = bool(res.contains(float(target_truth[0]), float(target_truth[1])))
    case = dict(
        tid=result.tid,
        view=view,
        point=[float(x), float(y)],
        point_latlon=[float(plat), float(plon)],
        point_src=getattr(result, 'point_src', 'posterior'),
        area_km2=float(result.area_km2),
        confidence=float(result.confidence),
        packets=int(result.packets),
        n_constraints=int(result.n_constraints),
        covered=covered,
        relax=[dict(a=d.get('action'), label=d.get('label'), pid=d.get('pid'),
                    factor=d.get('factor'), excess_km=d.get('excess_km'))
               for d in result.relax],
        overhead_ms=float(result.overhead_ms),
        circles=circles_geojson(result.constraints, max_circles=max_circles),
        routers=[dict(node=e['node'], lat=e['lat'], lon=e['lon'], unc_km=e['unc_km'],
                      method=e['method'], access_km=e['access_km'], note=e.get('note', ''))
                 for e in result.router_estimates],
        truth=[float(target_truth[0]), float(target_truth[1])] if target_truth else None,
    )
    case.update(region_layers(res))
    return case


def world_layers(world, lat_min=-60, lat_max=82, n_cities=900):
    """底图: 陆地轮廓 (归一化 mercator 折线) + 城市点。"""
    land = []
    lm = world.land
    if lm is not None:
        if hasattr(lm, 'rings'):                 # LandMask (多边形)
            for rl, ra in lm.rings:
                land.append(_merc_path(rl, ra))
        elif hasattr(lm, 'mask'):                # LandRaster (栅格) -> 抽取轮廓
            land = _raster_outlines(lm)
    cities = world.cities
    sel = np.argsort(-cities.pop)[:n_cities]
    pts = []
    for i in sel:
        x, y = lonlat_to_mercator(cities.lon[i], cities.lat[i])
        pts.append([float(x), float(y), float(cities.pop[i])])
    return dict(land=land, cities=pts)


def _raster_outlines(raster, max_rings=1400, min_cells=6):
    """从陆地栅格抽取单元级轮廓 (每行连续陆地段, 作为独立矩形环)。

    对前端绘制足够 (栅格分辨率 0.05°, 视觉上与多边形底图差别很小),
    且 JSON 体积可控。较大的连通块会按行段分别成环, 填充效果正确。
    """
    mask = raster.mask
    ny, nx = mask.shape
    res = raster.res
    rings = []
    row_any = mask.any(axis=1)
    rows = np.nonzero(row_any)[0]
    for i in rows:
        row = mask[i]
        if not row.any():
            continue
        d = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        lat0 = -90.0 + i * res
        lat1 = lat0 + res
        for s, e in zip(starts, ends):
            if e - s < min_cells:
                continue
            lon0 = -180.0 + s * res
            lon1 = -180.0 + e * res
            ring_lon = np.array([lon0, lon1, lon1, lon0])
            ring_lat = np.array([lat0, lat0, lat1, lat1])
            rings.append(_merc_path(ring_lon, ring_lat))
    rings.sort(key=lambda r: -(r[1][0] - r[0][0] + r[2][1] - r[1][1]))
    return rings[:max_rings]


def export_case(result, world=None, target_truth=None, out_dir='web/data', name=None):
    os.makedirs(out_dir, exist_ok=True)
    name = name or ('t%d' % result.tid)
    case = build_case(result, target_truth=target_truth)
    p = os.path.join(out_dir, name + '.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(case, f, ensure_ascii=False)
    return p


def export_world(world, out_dir='web/data'):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'world.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(world_layers(world), f)
    return p
