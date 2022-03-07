from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
import numpy as np
from src.data_structures.hypergraph import HyperEdge, HyperGraph
from src.data_structures.graph import Graph, Node, Edge
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_r_d(hypergraph: HyperGraph) -> Tuple[float, float]:
    """
    Simple function that computes the d parameter and the r parameter.
    d for d-regularity
    r for r-uniformity
    If the hypergraph is not d-regular or r-uniform, assertion error is returned.
    @param hypergraph: 
    @returns r
    @returns d
    """
    d = None
    for hn in hypergraph.hypernodes:
        d_new = len(hypergraph.adj_list[hn.id])
        if d is None:
            d = d_new
        assert d_new == d
    r = None
    for he in hypergraph.hyperedges:
        r_new = len(he.hypernodes)
        if r is None:
            r = r_new
        assert r_new == r
    return r, d


def build_graph(hypergraph: HyperGraph, p: np.array) -> Tuple[Graph, List[Tuple[int, int]]]:
    """
    Build the graph according to Kshiteej's process, namely:
    The vertex set for the graph G_t is the same as the one of the hypergraph.
    Regarding the edges: for every hyper-edge e, collapse it into an edge connecting 
    the two hypernodes u, v\in e s.t. 
    u = argmin_u p(u), u\in e and 
    v = argmax_v p(v), v\in e
    In case of a tie (for both u and v), take one at random with low (or high) probability.
    Since we want our collapsed graph to retain the d-regularity property, 
    we add self-loops to every node until its degree is d.
    Complexity: O(n + m*r)
    @param hypergraph
    @param p: probability vector for every vertex of the hypergraph.
    @returns graph: this graph has the same vertex set as the hypergraph,
        and an edge set made by hyperedges collapsed:
        for every hyper edge he, make an edge that connects v, u s.t. 
        p_max(v) and p_min(u), v, u\in hyperedge he
        according to the probability vector p.
    @returns map_hyperedge_edge: for every hyperedge 
        (in the order they appear in hypergraph.hyperedges), map the corresponding graph edge
        so that the first vertex is the min, and the last vertex is the max 
        (according to probability vector p).
    """
    # Assert that there is one probability entry per hypernode.
    assert len(p) == len(hypergraph.hypernodes)  
    nodes = [Node(hn.id, hn.id) for hn in hypergraph.hypernodes]
    edges = []
    nodes_edge_counter = {}  # Used to know how many self-loops to add.
    # In the position of the hyperedge, there is the corresponding edge
    # Note that since the graph is undirected, there are actualy 2 edges v->u and u->v
    # so that the map will have only one of the two (specifically, min->max).
    map_hyperedge_edge = []  
    for n in nodes:
        nodes_edge_counter[n.id] = 0
    for he in hypergraph.hyperedges:
        v_max = None
        for hn in he.hypernodes:
            if (v_max is None) or (p[v_max.id] < p[hn.id]):
                v_max = hn
        # Scan hypernodes in reversed fashion, so that v_min != v_max
        v_min = None
        for hn in reversed(list(he.hypernodes)):
            if v_min is None or p[v_min.id] > p[hn.id]:
                v_min = hn
        assert v_min.id != v_max.id
        edges.append(Edge(nodes[v_min.id], nodes[v_max.id], 1.0))
        edges.append(Edge(nodes[v_max.id], nodes[v_min.id], 1.0))
        map_hyperedge_edge.append((v_min.id, v_max.id))
        nodes_edge_counter[v_min.id] += 1
        nodes_edge_counter[v_max.id] += 1
        
    # Add self loops.
    for node in nodes:
        diff = np.sum([he.weight for he in hypergraph.adj_list[node.id]]) - nodes_edge_counter[node.id]
        if diff != 0:
            assert diff > 0
            edges.append(Edge(node, node, diff))
    return Graph(nodes, edges), map_hyperedge_edge
    

if __name__ == "__main__":
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    r, d = compute_r_d(hypergraph)

    dt = 1/4  # any number below 1/2

    np.random.seed(2)
    p0 = [np.random.rand() for i in range(len(hypergraph.hypernodes))]
    p0 /= np.sum(p0)  # So that it sums up to 1.

    pt = p0

    # Theorem 3 says convergence happens in:
    # I_t(k) \leq \min(\sqrt{k/d}, \sqrt{(2m - k)/d}) * \exp{-t * \phi^2} + k/2m
    # Hence: we increasingly compute the sweep cut, estimate \phi, and see if the time for convergence makes sense.
    x_s, y_s = [], []

    epochs = 100
    conductances_by_epoch = []
    for t in tqdm(range(epochs)):
        graph_t, map_hyperedge_edge_t = build_graph(hypergraph=hypergraph, p=pt)
        x_t, y_t = graph_t.compute_lovasz_simonovits_curve(pt)
        ls_sweep = hypergraph.compute_lovasz_simonovits_sweep(pt)
        conductances_by_epoch.append(hypergraph.compute_conductance(ls_sweep))
        x_s.append(x_t)
        y_s.append(y_t)
        # Multiply d by two since there are d additional self-loops.
        Mt = ((1 - dt) * (np.eye(len(graph_t.nodes))) + dt / d * graph_t.A)
        p_t_dt = Mt @ pt
        
        pt = p_t_dt
        
    fig, (ax1, ax2) = plt.subplots(2)
    for t in range(int(np.ceil(epochs))):
        ax1.plot(x_s[t], y_s[t])
    
    ax2.plot(range(int(np.ceil(epochs))), conductances_by_epoch)
    plt.show()