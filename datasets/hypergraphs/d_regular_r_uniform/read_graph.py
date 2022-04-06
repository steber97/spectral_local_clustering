
from src.data_structures.hypergraph import HyperEdge, HyperGraph, HyperNode


def read_graph(file: str ='datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt') -> HyperGraph:
    with open(file, 'r') as f:
        hyperedges = []
        hypernodes = []
        for i, l in enumerate(f):
            if i != 0:
                hyperedge = [int(x)-1 for x in l.split()]
                weight = 1
                for hn in hyperedge:
                    for n in range(len(hypernodes), hn+1):
                        hypernodes.append(HyperNode(n))

                hyperedge = [int(h) for h in hyperedge]
                hyperedges.append(HyperEdge([hypernodes[j] for j in hyperedge], weight, _id=i-1))
    return HyperGraph(hypernodes=hypernodes, hyperedges=hyperedges)