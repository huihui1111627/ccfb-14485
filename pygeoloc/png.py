"""png.py — 极简 PNG 编码器 (仅用 zlib/struct, 无第三方依赖)。

用于把后验热力图导出成前端可直接贴到地图上的位图。
"""
from __future__ import annotations

import struct
import zlib
import numpy as np


def _chunk(tag, data):
    c = tag + data
    return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)


def write_rgba(path, rgba):
    """rgba: (h, w, 4) uint8 数组。右下角为第 h-1 行 (图像坐标自上而下)。"""
    a = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w, _ = a.shape
    raw = b''.join(b'\x00' + a[y].tobytes() for y in range(h))
    data = (b'\x89PNG\r\n\x1a\n'
            + _chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + _chunk(b'IDAT', zlib.compress(raw, 9))
            + _chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(data)


def heatmap_rgba(post, mask=None, cmap='inferno', alpha_max=190):
    """把后验网格转成 RGBA 位图 (低值透明, 高值不透明)。"""
    p = np.array(post, dtype=np.float64, copy=True)
    if mask is not None:
        p = np.where(mask, p, 0.0)
    mx = float(np.nanmax(p)) if np.isfinite(p).any() else 0.0
    if mx <= 0:
        return np.zeros(p.shape + (4,), np.uint8)
    v = np.clip(p / mx, 0.0, 1.0)
    # alpha: 低于 2% 的透明, 再按 sqrt 提亮 (让弱区也能看见)
    a = np.clip((v - 0.02) / 0.98, 0, 1) ** 0.5 * alpha_max
    r = np.clip(255 * (0.05 + 0.95 * v ** 0.7), 0, 255)
    g = np.clip(255 * (0.02 + 1.35 * v - 0.55 * v ** 2), 0, 255) * (v > 0.001)
    b = np.clip(255 * (0.35 * (1 - v) ** 2 + 0.45 * v ** 3), 0, 255)
    img = np.zeros(p.shape + (4,), np.uint8)
    img[..., 0] = r.astype(np.uint8)
    img[..., 1] = g.astype(np.uint8)
    img[..., 2] = b.astype(np.uint8)
    img[..., 3] = a.astype(np.uint8)
    return img


def resample_to_mercator(post, lat_edges, lon_edges, out_h=260, out_w=260):
    """把 (等经纬) 后验网格重采样到以 bbox 为准的 Web Mercator 像素网格。"""
    from .geo import lonlat_to_mercator, mercator_to_lonlat
    ny, nx = post.shape
    latc = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lonc = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ys = np.linspace(0.5 / out_h, 1 - 0.5 / out_h, out_h)     # 归一化 mercator y (向下)
    xs = np.linspace(0.5 / out_w, 1 - 0.5 / out_w, out_w)
    # 先映射回经纬度
    # bbox 的 mercator 范围由经纬度范围决定
    x0, y0 = lonlat_to_mercator(lon_edges[0], lat_edges[-1])
    x1, y1 = lonlat_to_mercator(lon_edges[-1], lat_edges[0])
    lon_q, lat_q = mercator_to_lonlat(x0 + xs[None, :] * (x1 - x0),
                                      y0 + ys[:, None] * (y1 - y0))
    i = np.clip(np.searchsorted(latc, lat_q.ravel()) - 0, 0, ny - 1).reshape(lat_q.shape)
    j = np.clip(np.searchsorted(lonc, lon_q.ravel()) - 0, 0, nx - 1).reshape(lon_q.shape)
    out = post[i, j]
    return out
