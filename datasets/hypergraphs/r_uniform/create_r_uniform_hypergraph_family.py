import argparse
import numpy as np

from src.data_structures.hypergraph import HyperGraph, HyperEdge, HyperNode

args_main = argparse.ArgumentParser(
    "Create a family of r-uniform hypergraphs with equal conductance, but different r")

args_main.add_argument("--conductance", type=float, help='conductance of the family')
args_main.add_argument("--r", nargs='+', help='list of r values')
args_main.add_argument('--n', type=int, help='number of nodes')

args = args_main.parse_args()

"""
The idea for creating a hypergraph with desired conductance is to
split the n vertices in 2 equal groups A, B, add random hyperedges 
of size r to both bipartitions, 
and then add in between A, B the correct amount of
hyperedges so that the cut A,B has the desired conductance.
The ideal thing would be that the volume is the same for both graphs.
"""

if __name__ == "__main__":
    n = args.n
    r_list = [int(x) for x in args.r]
    conductance = args.conductance

    # Create the two bipartitions
    A = [HyperNode(i) for i in range(0, int(n/2))]
    B = [HyperNode(i) for i in range(int(n/2), n)]


    for r in r_list: 
        volume = 10000

        # First we add all edges crossing, until we have added a fraction of the volume which is
        #  (1-phi)* volume (we hope that the weight will be distributed approximately 1/2 on both sides)
        hyperedges = []
        hypernodes = [HyperNode(i) for i in range(n)]
        edge_counter = 0
        for i in range(int(conductance * volume / 2)):
            done = False
            while not done:
                hyperedge = np.random.permutation(A+B)[:r]
                ids = np.array([int(h.id) for h in hyperedge])
                if ids[ids < int(n/2)].sum() > 0 and ids[ids >= int(n/2)].sum() > 0:
                    hyperedges.append(HyperEdge(hyperedge, 1.0, edge_counter))
                    edge_counter += 1
                    done = True
        hypergraph1 = HyperGraph(hyperedges=hyperedges, hypernodes=hypernodes)
        print(np.sum(hypergraph1.deg_by_node), 
              np.sum(hypergraph1.deg_by_node[:int(0.5*n)]), 
              np.sum(hypergraph1.deg_by_node[int(0.5*n):]))
        volume_so_far = np.sum(hypergraph1.deg_by_node)
        for i in range(int((volume-volume_so_far) /(r * 2))):
            # Add hyperedges inside the two bipartitions
            hyperedges.append(HyperEdge(np.random.permutation(A)[:r], 1.0, edge_counter))
            hyperedges.append(HyperEdge(np.random.permutation(B)[:r], 1.0, edge_counter + 1))
            edge_counter += 2
        hypergraph2 = HyperGraph(hyperedges=hyperedges, hypernodes=A+B)
        print(np.sum(hypergraph2.deg_by_node),
              np.sum(hypergraph2.deg_by_node[:int(0.5*n)]), 
              np.sum(hypergraph2.deg_by_node[int(0.5*n):]))
        print(hypergraph2.compute_conductance([i < n/2 for i in range(n)]))



