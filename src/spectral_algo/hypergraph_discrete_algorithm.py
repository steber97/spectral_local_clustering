import math
import time

from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
import numpy as np
from src.data_structures.hypergraph import HyperEdge, HyperGraph, HyperNode
from src.data_structures.graph import Graph, Node, Edge
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import identity


class HyperGraphLocalClusteringDiscrete:
    def __init__(self, hypergraph: HyperGraph):
        # Turn hypergraph into a standard graph with clique transformation
        self.dt = 1/4
        
    def build_graph(self, hypergraph: HyperGraph, p: np.array) -> Tuple[Graph, List[Tuple[int, int]]]:
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
        nodes_degree_counter = {}  # Used to know how many self-loops to add.
        # In the position of the hyperedge, there is the corresponding edge
        # Note that since the graph is undirected, there are actualy 2 edges v->u and u->v
        # so that the map will have only one of the two (specifically, min->max).
        map_hyperedge_edge = []
        p_weighted_by_deg = p / hypergraph.deg_by_node
        for n in nodes:
            nodes_degree_counter[n.id] = 0
        for he in hypergraph.hyperedges:
            v_max = None
            for hn in he.hypernodes:
                if (v_max is None) or (p_weighted_by_deg[v_max.id] < p_weighted_by_deg[hn.id]):
                    v_max = hn
            # Scan hypernodes in reversed fashion, so that v_min != v_max
            v_min = None
            for hn in reversed(list(he.hypernodes)):
                if v_min is None or p_weighted_by_deg[v_min.id] > p_weighted_by_deg[hn.id]:
                    v_min = hn
            assert v_min != v_max or len(he.hypernodes) == 1
            if v_min != v_max:
                edges.append(Edge(nodes[v_min.id], nodes[v_max.id], he.weight))
                edges.append(Edge(nodes[v_max.id], nodes[v_min.id], he.weight))
                map_hyperedge_edge.append((v_min.id, v_max.id))
                nodes_degree_counter[v_min.id] += he.weight
                nodes_degree_counter[v_max.id] += he.weight
            
        # Add self loops.
        for node in nodes:
            diff = hypergraph.deg_by_node[node.id] - nodes_degree_counter[node.id]
            if diff != 0:
                assert diff > 0
                edges.append(Edge(node, node, diff))
        # print("Time for nodes and edges = {}".format(time.time() - time_start))
        graph = Graph(nodes, edges)
        # print("Time for graph initialization: {}".format(time.time() - time_start))
        return graph, map_hyperedge_edge

    def perform_one_iteration(self, hypergraph: HyperGraph, pt: np.array, mu: float):
        graph_t, map_hyperedge_edge_t = self.build_graph(hypergraph=hypergraph, p=pt)
        ls_sweep = hypergraph.compute_lovasz_simonovits_sweep(pt, mu)
        conductance = hypergraph.compute_conductance(ls_sweep)
        # Evolve pt
        self.M_t = ((1 - self.dt) * (identity(len(graph_t.nodes))) + self.dt * graph_t.getA() * graph_t.getDInv())
        p_t_dt = self.M_t.dot(pt)

        return ls_sweep, conductance, p_t_dt, graph_t
    
    def hypergraph_local_clustering(self, hypergraph: HyperGraph, v: HyperNode, epochs: float, mu: float = 0.1, phi: float = 0.0) -> np.array:
        p_0 = np.zeros(len(hypergraph.hypernodes))
        p_0[v.id] = 1.0
        pt = p_0.copy()
        best_cut = None
        best_conductance = 1.1
        conductances = []
        for epoch in range(int(epochs)):
            cut, conductance, p_t_dt, _ = self.perform_one_iteration(hypergraph, pt, mu)
            conductances.append(conductance)
            if best_cut is None or conductance < best_conductance:
                best_cut = cut
                best_conductance = conductance
            pt = p_t_dt

        conductances_distinc_value_per_iter = []
        last_cond = conductances[0]
        for i, conductance in enumerate(conductances):
            if i == 0:
                conductances_distinc_value_per_iter.append((conductance, 0))
            elif np.abs(last_cond - conductance) > 10e-10:
                conductances_distinc_value_per_iter.append((conductance, i))
                last_cond = conductance

        return best_cut


if __name__ == "__main__":
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    
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

    hglckp = HyperGraphLocalClusteringKshiteejProcess()
    d, r = hglckp.compute_r_d(hypergraph=hypergraph)
    for t in tqdm(range(epochs)):
        graph_t, map_hyperedge_edge_t = hglckp.build_graph(hypergraph=hypergraph, p=pt)
        x_t, y_t = graph_t.compute_lovasz_simonovits_curve(pt)
        ls_sweep = hypergraph.compute_lovasz_simonovits_sweep(pt)
        conductances_by_epoch.append(hypergraph.compute_conductance(ls_sweep))
        x_s.append(x_t)
        y_s.append(y_t)
        # Multiply d by two since there are d additional self-loops.
        Mt = ((1 - hglckp.dt) * (np.eye(len(graph_t.nodes))) + hglckp.dt / d * graph_t.A)
        p_t_dt = Mt @ pt
        
        pt = p_t_dt
        
    fig, (ax1, ax2) = plt.subplots(2)
    for t in range(int(np.ceil(epochs))):
        ax1.plot(x_s[t], y_s[t])
    
    ax2.plot(range(int(np.ceil(epochs))), conductances_by_epoch)
    plt.show()