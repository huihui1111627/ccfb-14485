import numpy as np
import pytest

from pygeoloc.sim import Simulation, SimConfig, ACCESS_KINDS
from pygeoloc.geo import haversine_km


@pytest.fixture(scope='module')
def sim():
    return Simulation(cfg=SimConfig(n_probes=40))


def test_topology_connected_and_has_edges(sim):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = sim.n_nodes
    G = csr_matrix((np.ones(2 * len(sim.edge_i)),
                    (np.r_[sim.edge_i, sim.edge_j], np.r_[sim.edge_j, sim.edge_i])),
                   shape=(n, n))
    ncomp, _ = connected_components(G, directed=False)
    assert ncomp == 1
    assert len(sim.edge_km) > n


def test_paths_have_detour_and_asymmetry(sim):
    # 往返时延不对称: 至少 5% 的边对方向权重明显不同
    # 一次传输在正向边上的去程权重 wf 与它在反向图上的"回程"权重不同
    Wf = sim.Wf.tocoo()
    Wr_back = sim.WrT.tocsr()
    ratios = []
    for i, j, w in zip(Wf.row, Wf.col, Wf.data):
        wb = Wr_back[j, i]        # 反向传输 (j->i) 的去程 = i->j 的回程
        if w > 0 and wb > 0:
            ratios.append(float(wb) / float(w))
    ratio = np.asarray(ratios)
    assert np.percentile(ratio, 99) > 1.15
    # 绕行: 路径长度 / 大圆距离 > 1
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse import csr_matrix
    G = csr_matrix((sim.edge_km, (sim.edge_i, sim.edge_j)),
                   shape=(sim.n_nodes, sim.n_nodes)).maximum(
        csr_matrix((sim.edge_km, (sim.edge_j, sim.edge_i)),
                   shape=(sim.n_nodes, sim.n_nodes)))
    dd = dijkstra(G, indices=sim.probes[0].node)
    d = haversine_km(sim.probe_lat[0], sim.probe_lon[0], sim.probe_lat, sim.probe_lon)
    m2 = d > 1000
    assert (dd[sim.probe_node[m2]] / d[m2]).max() > 1.2


def test_min_of_repeats_filters_queueing(sim):
    t = sim.sample_targets(6)[0]
    v = sim.ping(0, t, 400)
    v = v[np.isfinite(v)]
    # 多次测量取最小值能滤掉排队时延: 单包最小值 < 单包中位数
    assert v.min() < np.median(v)
    # 且 200 次里的最小值 明显低于 20 次一批的"批次中位数"
    batch = v[: (len(v) // 20) * 20].reshape(-1, 20)
    assert v.min() < np.median(batch.min(axis=1)) or v.min() < np.median(v) * 0.98
    # 确实存在重尾 (有比最小值大 5ms 以上的样本)
    assert v.max() > v.min() + 5


def test_access_kinds_present(sim):
    kinds = {t.kind for t in sim.sample_targets(120)}
    assert 'fiber' in kinds and 'idc' in kinds


def test_persistent_congestion_exists(sim):
    ts = sim.sample_targets(200)
    bad = [t for t in ts if t.persist > 0]
    assert len(bad) > 0


def test_losses_happen(sim):
    t = sim.sample_targets(5)[0]
    v = sim.ping(0, t, 2000)
    assert np.isinf(v).sum() > 0


def test_targets_on_land(sim):
    for t in sim.sample_targets(60):
        assert sim.world.land.lookup(t.lon, t.lat)
