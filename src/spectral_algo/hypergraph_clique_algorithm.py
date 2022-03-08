from src.data_structures.hypergraph import HyperGraph, HyperNode, HyperEdge
from src.data_structures.graph import Graph, Node, Edge
from src.spectral_algo.page_rank import pagerank
import numpy as np


def hypergraph_local_clustering_by_clique(hypergraph: HyperGraph, v: HyperNode, mu: float = 0.1):
    """
    """
    # Turn hypergraph into a standard graph with clique transformation
    nodes = []
    edges = []
    for hn in sorted(hypergraph.hypernodes, key=lambda x: x.id):
        nodes.append(Node(hn.id, hn.id))
    for he in hypergraph.hyperedges:
        for hn in he.hypernodes:
            for hn2 in he.hypernodes:
                if hn.id != hn2.id:
                    edges.append(Edge(start=nodes[hn.id], end=nodes[hn2.id], weight=he.weight))
    graph = Graph(nodes, edges)
    # print(graph.M)
    best_cut = None
    best_conductance = None
    for alpha in range(5, 100, 5):
        # Center the probability in the starting vertex.
        p_0 = np.zeros(len(nodes))
        p_0[v.id] = 1.0
        
        ppr = pagerank(graph.M, p_0, delta=1e-8, alpha=alpha/100)
        # print(ppr)
        # Take sweep on the ppr, as long as the cut is not larger than mu * vol(H)
        cut = hypergraph.compute_lovasz_simonovits_sweep(ppr, mu=mu)
        conductance = hypergraph.compute_conductance(cut)
        if best_conductance is None or conductance < best_conductance:
            best_conductance = conductance
            best_cut = cut
    
    return best_cut


