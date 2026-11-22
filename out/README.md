# out/ — 验收产物

由以下命令生成（耗时随机器而定；8 进程约 0.4–0.5 s/目标）：

```bash
python tools/run_benchmark.py --n 3000 --workers 8 --seed 7 --out out/bench_seed7.json
python tools/plot_results.py   --in out/bench_seed7.json --outdir out/figs
```

| 文件 | 内容 |
|---|---|
| `bench_seed7.json` | 3000 目标的完整明细 + 汇总 (真值/估计/区域面积/覆盖率/包数/基线) |
| `bench_v5.json` / `bench_v4.json` | 600 目标的中间验证跑 (调参对照) |
| `bench_cons115.json` | 保守性旋钮 1.15 的对照 (说明默认已处拐点) |
| `figs/err_cdf.svg` | 误差距离 CDF: 本系统 vs 三种基线 |
| `figs/area_hist.svg` | 区域面积分布 |
| `figs/coverage_by_kind.svg` | 分接入类型覆盖率 |
| `figs/calib.svg` | 置信度校准 |

`bench_seed7.json` 是最终验收来源；其余是调参对照。
`data/raw/` 与 `web/data/` 已在 `.gitignore` 中（原始下载数据与页面数据可由脚本重建）。
`out/figs/*.svg` 由 `tools/plot_results.py` 从 `bench_seed7.json` 生成，可直接用浏览器查看。
