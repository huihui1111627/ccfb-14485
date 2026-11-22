"""measure.py — 测量采集: 多次 ping 取最小、traceroute 解析、测量预算记账。

算法侧只通过这个类访问网络; 所有"真值"都不可见。
"""
from __future__ import annotations

import numpy as np

from .sim import Simulation, Target


class Measurer:
    """带缓存的测量接口。同一 (探测点,目标) 的 ping 结果按需增量采样。"""

    def __init__(self, sim: Simulation):
        self.sim = sim
        self._pings = {}       # (pid, tid) -> list[float]  已完成的最小值样本 (每次 n 次 ping 的 min)
        self._tr = {}          # (pid, tid) -> (hops, rtt)
        self._probe_ping = {}  # (pid, qid) -> min 样本列表
        self._node_ping = {}   # (pid, node) -> min 样本列表

    # ---- 目标 ping
    def target_rtt(self, pid, target: Target, n=1, samples=1):
        """返回 samples 次 (每次 n 个包) 的最小 RTT 列表。"""
        key = (pid, target.tid)
        cur = self._pings.setdefault(key, [])
        vals = []
        for _ in range(samples):
            v = self.sim.ping(pid, target, n)
            v = v[np.isfinite(v)]
            cur.append(float(v.min()) if len(v) else np.inf)
        return cur

    def target_min(self, pid, target, n=3, samples=1):
        cur = self.target_rtt(pid, target, n=n, samples=samples)
        f = [v for v in cur if np.isfinite(v)]
        return float(min(f)) if f else np.inf

    # ---- traceroute
    def route(self, pid, target: Target, tries=3):
        key = (pid, target.tid)
        if key not in self._tr:
            self._tr[key] = self.sim.traceroute(pid, target, tries=tries)
        return self._tr[key]

    # ---- 探测点之间
    def probe_rtt(self, pid, qid, n=5):
        key = (pid, qid)
        if key not in self._probe_ping:
            vals = self.sim.ping_probe(pid, qid, n)
            vals = vals[np.isfinite(vals)]
            self._probe_ping[key] = float(vals.min()) if len(vals) else np.inf
        return self._probe_ping[key]

    # ---- 探测点到路由器
    def node_rtt(self, pid, node, n=4, samples=1):
        key = (pid, int(node))
        cur = self._node_ping.setdefault(key, [])
        for _ in range(samples):
            vals = self.sim.ping_node(pid, int(node), n)
            vals = vals[np.isfinite(vals)]
            cur.append(float(vals.min()) if len(vals) else np.inf)
        f = [v for v in cur if np.isfinite(v)]
        return float(min(f)) if f else np.inf

    def stats(self):
        return self.sim.stats.snapshot()

    def total_packets(self):
        return self.sim.stats.packets
