"""report.py — 把定位结果写成"研判人员能直接用"的文字结论。

输出示例:
    目标 #12  (置信度 0.82, 区域 4.3×10⁵ km²)
      最可能位置: 印度 马哈拉施特拉邦 浦那附近 (18.5°N, 73.9°E)
      区域内城市 (按后验从高到低): 浦那、孟买、纳西克…
      结论: 高置信 —— 区域覆盖 1 个城市群, 点估计误差中位数约 40 km
    低置信示例:
      结论: 低置信 —— 区域横跨 3 个国家, 时延约束只能排除远处; 建议补充测量或结合其它线索
"""
from __future__ import annotations

import numpy as np

from .geo import haversine_km


def _hemi(lat, lon):
    return '%.2f°%s, %.2f°%s' % (abs(lat), 'N' if lat >= 0 else 'S',
                                 abs(lon), 'E' if lon >= 0 else 'W')


def nearest_city(world, lat, lon, radius_km=250.0):
    """点估计附近的**最近**城市 (用于"最可能位置"); 250km 内没有城市则返回 None。"""
    idx = world.cities.near(lat, lon, radius_km)
    if len(idx) == 0:
        return None
    d = haversine_km(lat, lon, world.cities.lat[idx], world.cities.lon[idx])
    i = int(idx[int(np.argmin(d))])
    return dict(name=world.cities.name[i], lat=float(world.cities.lat[i]),
                lon=float(world.cities.lon[i]), pop=float(world.cities.pop[i]),
                dist_km=float(haversine_km(lat, lon, world.cities.lat[i], world.cities.lon[i])))


def cities_in_region(world, res, max_n=6):
    """区域内按后验排序的城市 (用于"区域里人口密度高的地方更可能")。"""
    if res.empty:
        return []
    latc = 0.5 * (res.lat_edges[:-1] + res.lat_edges[1:])
    lonc = 0.5 * (res.lon_edges[:-1] + res.lon_edges[1:])
    lat0, lat1 = float(latc[0]), float(latc[-1])
    lon0, lon1 = float(lonc[0]), float(lonc[-1])
    span = max(lat1 - lat0, (lon1 - lon0) * np.cos(np.radians(0.5 * (lat0 + lat1))), 0.05)
    rad = span * 111.0 * 1.2 + 50.0
    idx = world.cities.near(0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1), rad)
    out = []
    post = res.posterior
    for i in idx:
        la = float(world.cities.lat[i]); lo = float(world.cities.lon[i])
        if not (lat0 <= la <= lat1 and lon0 <= lo <= lon1):
            continue
        ii = int(np.clip(np.searchsorted(latc, la), 0, len(latc) - 1))
        jj = int(np.clip(np.searchsorted(lonc, lo), 0, len(lonc) - 1))
        if post is not None and post.size:
            w = float(post[ii, jj])
            if w <= 0 and not res.mask[ii, jj]:
                continue
        else:
            w = 0.0
        out.append(dict(name=world.cities.name[i], lat=la, lon=lo,
                        pop=float(world.cities.pop[i]), weight=w))
    out.sort(key=lambda d: -d['weight'])
    return out[:max_n]


def confidence_word(c):
    if c >= 0.8:
        return '高置信'
    if c >= 0.5:
        return '中等置信'
    return '低置信'


def summary_text(result, world, truth=None):
    lines = []
    nc = nearest_city(world, result.point_lat, result.point_lon)
    lines.append('目标 #%d  (%s, 置信度 %.2f, 区域 %.2g km²)' % (
        result.tid, confidence_word(result.confidence), result.confidence, result.area_km2))
    if nc:
        lines.append('  最可能位置: %s 附近 (%s, 距估计点 %.0f km)' % (
            nc['name'], _hemi(result.point_lat, result.point_lon), nc['dist_km']))
    else:
        lines.append('  最可能位置: %s (250 km 内无已知城市, 可能在海面/偏远地区)' %
                     _hemi(result.point_lat, result.point_lon))
    cities = cities_in_region(world, result.region, max_n=5)
    if cities:
        lines.append('  区域内城市(按后验从高到低): ' + '、'.join(
            '%s' % c['name'] for c in cities))
    elif not result.region.empty:
        lines.append('  区域内无人口>1.5万的城市 (区域可能覆盖海面/荒漠, 或区域过大)')
    if truth is not None:
        err = haversine_km(result.point_lat, result.point_lon, truth[0], truth[1])
        lines.append('  验收: 点估计误差 %.0f km, 真值%s在区域内' % (
            err, '' if result.region.contains(truth[0], truth[1]) else '不'))
    if result.relax:
        lines.append('  注意: %d 条约束被放宽/剔除 (测量异常), 结论应打折' % len(result.relax))
    if result.router_estimates:
        e = min(result.router_estimates, key=lambda x: x['unc_km'])
        how = {'hostname+delay': '主机名+时延三角定位',
               'delay': '时延三角定位, 主机名不可信已忽略',
               'hostname': '仅主机名定位, 时延圆无法三角定位',
               'none': '无线索'}.get(e['method'], e['method'])
        if e['access_km'] > 2000:
            note = '上界很松 (接入/拥塞时延占比大, 这条约束几乎不起作用)'
        else:
            note = '上界较紧, 对排除远处很有效'
        lines.append('  拓扑: 末跳路由器 %s, 不确定度 %.0f km; 目标到末跳的距离上界 %.0f km —— %s'
                     % (how, e['unc_km'], e['access_km'], note))
    return '\n'.join(lines)
