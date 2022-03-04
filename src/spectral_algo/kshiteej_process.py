from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
import numpy as np
from src.data_structures.hypergraph import HyperEdge, HyperGraph
from src.data_structures.graph import Graph, Node, Edge
from typing import Tuple, List
from tqdm import tqdm


def compute_r_d(hypergraph: HyperGraph) -> Tuple[float, float]:
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
    @returns map_hyperedge_edge: for every hyperedge 
        (in the order they appear in hypergraph.hyperedges), map the corresponding graph edge
        so that the first vertex is the min, and the last vertex is the max.
    """
    nodes = [Node(hn.id, hn.id) for hn in hypergraph.hypernodes]
    edges = []
    nodes_edge_counter = {}
    # In the position of the hyperedge, there is the corresponding edge
    # Note that since the graph is undirected, there are actualy 2 edges v->u and u->v
    # so that the map will have only one of the two.
    map_hyperedge_edge = []  
    for n in nodes:
        nodes_edge_counter[n.id] = 0
    for he in hypergraph.hyperedges:
        v_max = None
        for hn in he.hypernodes:
            if (v_max is None) or (p[v_max.id] < p[hn.id]):
                v_max = hn
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
        diff += d  # Add d self loops, to be sure that we have enough to remove! 
        if diff != 0:
            assert diff > 0
            edges.append(Edge(node, node, diff))
    return Graph(nodes, edges), map_hyperedge_edge


def evolve_graph(graph_t: Graph, 
                 old_graph: Graph, 
                 bipartition: np.array, 
                 hypergraph: HyperGraph, 
                 n: Node,
                 p_t_dt: np.array,
                 p_t: np.array):
    """
    Add one edge from n, so that the new edge is v_min_t->v_min_t_dt or v_max_t->v_max_t_dt
    """
    pass
    

if __name__ == "__main__":
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    r, d = compute_r_d(hypergraph)

    dt = 1/4  # any number below 1/2

    np.random.seed(2)
    p0 = [np.random.rand() for i in range(len(hypergraph.hypernodes))]
    p0 /= np.sum(p0)  # So that it sums up to 1.

    pt = p0

    for t in tqdm(range(100)):
    
        # print(pt)
        # print(graph_t.nodes)
        # print(graph_t.edges_list)
        # print(pt)
        graph_t, map_hyperedge_edge_t = build_graph(hypergraph=hypergraph, p=pt)
        # print("Graph_t")
        # print(graph_t)
        # Multiply d by two since there are d additional self-loops.
        Mt = ((1 - dt) * (np.eye(len(graph_t.nodes))) + dt / (d*2) * graph_t.A)
        p_t_dt = Mt @ pt
        # print("Transition probability matrix")
        # print(Mt)
        # print(p_t_dt)
        graph_t_dt, map_hyperedge_edge_t_dt = build_graph(hypergraph, p_t_dt)
        # print("graph t dt")
        # print(graph_t_dt)
        
        # Here d is a constant, but whatever just for completeness and later 
        # generalization to standard hypergraphs.
        nodes_sorted = [(p_t_dt[i], graph_t.nodes[i]) for i in range(len(graph_t.nodes))]
        nodes_sorted = sorted(nodes_sorted, key=lambda x: x[0], reverse=True)
        nodes_edge_counter = {}
        for hn in hypergraph.hypernodes:
            nodes_edge_counter[hn.id] = 0
        edges_t_tilde = []  
        nodes_t_tilde = [Node(i, i) for i in range(len(hypergraph.hypernodes))]
        for i, he in enumerate(hypergraph.hyperedges):
            for j in range(1, len(pt) - 1):
                Sj = [n[1] for n in nodes_sorted][: j]
                bipartition = [False for i in range(len(graph_t.nodes))]
                for n in Sj:
                    bipartition[n.id] = True
            
                if bipartition[map_hyperedge_edge_t[i][0]] == bipartition[map_hyperedge_edge_t[i][1]] and \
                    bipartition[map_hyperedge_edge_t_dt[i][0]] != bipartition[map_hyperedge_edge_t_dt[i][1]]:
                    # The old graph has a cutting edge less!
                    # map_hyperedge has always (min, max)
                    
                    if bipartition[map_hyperedge_edge_t[i][0]] != bipartition[map_hyperedge_edge_t_dt[i][0]]:
                        # add v_min_t -> v_min_t_dt
                        v_min_t = map_hyperedge_edge_t[i][0]
                        v_min_t_dt = map_hyperedge_edge_t_dt[i][0]
                        edges_t_tilde.append(
                                Edge(nodes_t_tilde[v_min_t], nodes_t_tilde[v_min_t_dt], 1.0))
                        edges_t_tilde.append(
                                Edge(nodes_t_tilde[v_min_t_dt], nodes_t_tilde[v_min_t], 1.0))
                        nodes_edge_counter[v_min_t] += 1
                        nodes_edge_counter[v_min_t_dt] += 1
                    else:
                        assert bipartition[map_hyperedge_edge_t[i][1]] != bipartition[map_hyperedge_edge_t_dt[i][1]]
                        # Add v_max_t -> v_max_t_dt
                        v_max_t = map_hyperedge_edge_t[i][1]
                        v_max_t_dt = map_hyperedge_edge_t_dt[i][1]
                        edges_t_tilde.append(
                                Edge(nodes_t_tilde[v_max_t], nodes_t_tilde[v_max_t_dt], 1.0))
                        edges_t_tilde.append(
                                Edge(nodes_t_tilde[v_max_t_dt], nodes_t_tilde[v_max_t], 1.0))
                        nodes_edge_counter[v_max_t] += 1
                        nodes_edge_counter[v_max_t_dt] += 1
                    break
            # then add the edge in Et (this is always done!)
            v_min_t = map_hyperedge_edge_t[i][0]
            v_max_t = map_hyperedge_edge_t[i][1]
            nodes_edge_counter[v_max_t] += 1
            nodes_edge_counter[v_min_t] += 1
            edges_t_tilde.append(
                    Edge(nodes_t_tilde[v_min_t], nodes_t_tilde[v_max_t], 1.0))
            edges_t_tilde.append(
                    Edge(nodes_t_tilde[v_max_t], nodes_t_tilde[v_min_t], 1.0))
        # Add self loops.
        for node in hypergraph.hypernodes:
            diff = np.sum([he.weight for he in hypergraph.adj_list[node.id]]) - nodes_edge_counter[node.id]
            diff += d  # Add d self loops, just to be sure it is positive! 
            if abs(diff) > 0.0001:
                assert diff > 0
                edges_t_tilde.append(
                        Edge(nodes_t_tilde[node.id], nodes_t_tilde[node.id], diff))
        
        graph_t_tilde = Graph(nodes_t_tilde, edges_t_tilde)
        
        # Evolve pt on the new graph tilde. Multiply d by 2
        Mt_tilde = ((1 - dt) * (np.eye(len(graph_t_tilde.nodes))) + dt / (d*2) * graph_t_tilde.A)
        pt_dt_tilde = Mt_tilde @ pt
        
        # print(pt_dt_tilde)
        # print("graph tilde")
        # print(graph_t_tilde)
        # print("Mt_tilde")
        # print(Mt_tilde)
        pt = pt_dt_tilde
    print(pt)