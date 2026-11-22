#!/usr/bin/env python3
"""plot_results.py — 从 benchmark JSON 生成验收图表 (纯 SVG, 无第三方绘图库)。

用法:
  python tools/plot_results.py --in out/bench_seed7.json --outdir out/figs

输出:
  err_cdf.svg       误差距离 CDF (主方法 vs 基线)
  area_hist.svg     区域面积分布 (log10)
  coverage_by_kind.svg  按接入类型的分项覆盖率
  calib.svg         置信度校准曲线
"""
import argparse, json, os, math


def _svg(w, h, body, title=''):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d" font-family="-apple-system,PingFang SC,sans-serif">'
            '<rect width="100%%" height="100%%" fill="#0b0f14"/>'
            '<text x="%d" y="22" fill="#d7e0ea" font-size="14">%s</text>%s</svg>'
            % (w, h, w, h, w // 2, title, body))


def _axes(w, h, xlab, ylab, pad=48):
    body = ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#2a3644"/>'
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#2a3644"/>'
            % (pad, h - pad, w - pad, h - pad, pad, h - pad, pad, pad))
    body += ('<text x="%d" y="%d" fill="#7d8b9c" font-size="11" text-anchor="middle">%s</text>'
             % ((w + pad) // 2, pad - 22, xlab))
    body += ('<text x="16" y="%d" fill="#7d8b9c" font-size="11" transform="rotate(-90 16 %d)">%s</text>'
             % ((h - pad) // 2, (h - pad) // 2, ylab))
    return body


COLORS = ['#4ea8ff', '#ffb454', '#4ade80', '#ff5d5d', '#c084fc', '#22d3ee']


def cdf_svg(errs_by_name, w=680, h=400):
    pad = 56
    pts = [(n, sorted(e)) for n, e in errs_by_name.items() if e]
    if not pts:
        return _svg(w, h, '')
    xmax = max(max(e) for _, e in pts)
    xmax = max(xmax, 1.0)
    body = _axes(w, h, '误差距离 (km, 对数刻度)', '累积比例', pad)
    # log x ticks
    for v in (1, 10, 100, 1000, 10000):
        if v > xmax * 1.2:
            continue
        x = pad + (w - 2 * pad) * (math.log10(max(v, 1)) / math.log10(max(xmax, 10)))
        body += ('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#1b2530"/>'
                 '<text x="%.1f" y="%d" fill="#5d6b7c" font-size="10" text-anchor="middle">%d</text>'
                 % (x, pad, x, h - pad, x, h - pad + 14, v))
    for k, (name, e) in enumerate(pts):
        col = COLORS[k % len(COLORS)]
        pts_s = []
        n = len(e)
        for i, v in enumerate(e):
            x = pad + (w - 2 * pad) * (math.log10(max(v, 1)) / math.log10(max(xmax, 10)))
            y = h - pad - (h - 2 * pad) * ((i + 1) / n)
            pts_s.append('%.1f,%.1f' % (x, y))
        body += ('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                 % (col, ' '.join(pts_s)))
        body += ('<text x="%d" y="%d" fill="%s" font-size="11">%s (n=%d)</text>'
                 % (w - pad - 150, 42 + 16 * k, col, name, n))
    return _svg(w, h, body, '误差距离 CDF')


def hist_svg(vals, w=680, h=360, title='', xlabel='log10(面积 km²)'):
    pad = 56
    if not vals:
        return _svg(w, h, '')
    lg = [math.log10(max(v, 1.0)) for v in vals]
    lo, hi = min(lg), max(lg)
    if hi - lo < 1e-6:
        hi = lo + 1
    nb = 30
    bins = [0] * nb
    for v in lg:
        k = min(nb - 1, int((v - lo) / (hi - lo) * nb))
        bins[k] += 1
    mx = max(bins)
    body = _axes(w, h, xlabel, '目标数', pad)
    bw = (w - 2 * pad) / nb
    for i, b in enumerate(bins):
        bh = (h - 2 * pad) * b / mx
        body += ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#4ea8ff" opacity="0.75"/>'
                 % (pad + i * bw, h - pad - bh, max(bw - 1, 1), bh))
    for i in range(0, nb + 1, 5):
        v = lo + (hi - lo) * i / nb
        body += ('<text x="%.1f" y="%d" fill="#5d6b7c" font-size="10" text-anchor="middle">%.1f</text>'
                 % (pad + i * bw, h - pad + 14, v))
    return _svg(w, h, body, title)


def coverage_by_kind(records):
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, []])
    for r in records:
        k = r.get('kind', '?')
        agg[k][0] += 1
        agg[k][1] += 1 if r.get('covered') else 0
        agg[k][2].append(r.get('err_km', 0))
    rows = []
    for k, (n, c, errs) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        errs = sorted(errs)
        rows.append((k, n, c / max(n, 1), errs[len(errs) // 2] if errs else 0))
    return rows


def coverage_svg(rows, w=680, h=320):
    pad = 56
    body = _axes(w, h, '接入类型', '区域覆盖率 (真值在区域内)', pad)
    n = len(rows)
    bw = (w - 2 * pad) / max(n, 1)
    for i, (k, cnt, cov, med) in enumerate(rows):
        bh = (h - 2 * pad) * cov
        x = pad + i * bw + bw * 0.15
        body += ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#4ade80" opacity="0.8"/>'
                 % (x, h - pad - bh, bw * 0.7, bh))
        body += ('<text x="%.1f" y="%d" fill="#9fb0c3" font-size="10" text-anchor="middle">%s</text>'
                 % (x + bw * 0.35, h - pad + 16, k))
        body += ('<text x="%.1f" y="%.1f" fill="#d7e0ea" font-size="10" text-anchor="middle">%.0f%%</text>'
                 % (x + bw * 0.35, h - pad - bh - 6, cov * 100))
    return _svg(w, h, body, '分接入类型的覆盖率')


def calib_svg(cal, w=560, h=320):
    pad = 56
    body = _axes(w, h, '系统给出的置信度', '实际命中率', pad)
    body += ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#2a3644" stroke-dasharray="4 4"/>'
             % (pad, h - pad, w - pad, pad))
    for c in cal:
        x = pad + (w - 2 * pad) * ((c['lo'] + c['hi']) / 2)
        y = h - pad - (h - 2 * pad) * c['hit']
        body += '<circle cx="%.1f" cy="%.1f" r="6" fill="#ffb454"/>' % (x, y)
        body += ('<text x="%.1f" y="%.1f" fill="#9fb0c3" font-size="10" text-anchor="middle">'
                 '%.2f</text>' % (x, y - 12, c['hit']))
    return _svg(w, h, body, '置信度校准 (点=实测, 虚线=理想)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--outdir', default='out/figs')
    a = ap.parse_args()
    d = json.load(open(a.inp))
    recs = d['records']
    summ = d['summary']
    os.makedirs(a.outdir, exist_ok=True)
    series = {'本系统': [r['err_km'] for r in recs if r.get('err_km') is not None]}
    for name in ('nearest_probe', 'nearest_circles', 'fiber_speed'):
        vals = [r['base_errs'][name] for r in recs
                if r.get('base_errs', {}).get(name) is not None]
        label = {'nearest_probe': '最近探测点坐标',
                 'nearest_circles': '最近 2 点标定圆交集',
                 'fiber_speed': '最近 2 点真空光速圆'}[name]
        if vals:
            series[label] = vals
    open(os.path.join(a.outdir, 'err_cdf.svg'), 'w').write(cdf_svg(series))
    open(os.path.join(a.outdir, 'area_hist.svg'), 'w').write(
        hist_svg([r['area_km2'] for r in recs], title='区域面积分布'))
    open(os.path.join(a.outdir, 'coverage_by_kind.svg'), 'w').write(
        coverage_svg(coverage_by_kind(recs)))
    open(os.path.join(a.outdir, 'calib.svg'), 'w').write(
        calib_svg(summ.get('calibration', [])))
    print('wrote 4 figures to', a.outdir)


if __name__ == '__main__':
    main()
