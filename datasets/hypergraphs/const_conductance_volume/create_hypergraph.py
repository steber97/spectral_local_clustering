import argparse
import numpy as np
import os

from src.data_structures.hypergraph import HyperGraph, HyperEdge, HyperNode

args_main = argparse.ArgumentParser(
    "Create an hypergraph with given volume and conductance")

args_main.add_argument("--conductance", type=float, help='conductance of the family')
args_main.add_argument('--n', type=int, help='number of nodes')
args_main.add_argument('--volume', type=int, help='total volume of the hypergraph')
args_main.add_argument("--out_folder", type=str, help='Output folder for the hypergraph.'
                                                  'The name of the files will be hypergraph_vol_VOL_n_N_cond_COND.txt')

args = args_main.parse_args()


"""
The idea for creating a hypergraph with desired conductance is to
split the n vertices in 2 equal groups A, B, add random hyperedges
to both bipartitions, 
and then add in between A, B the correct amount of
hyperedges so that the cut A,B has the desired conductance.
"""


if __name__ == "__main__":
    n = args.n
    conductance = args.conductance
    volume = args.volume

    # Create the two bipartitions
    A = [HyperNode(i) for i in range(0, int(n/2))]
    B = [HyperNode(i) for i in range(int(n/2), n)]

    # First we add all edges crossing, until we have added a fraction of the volume which is
    #  (1-phi)* volume (we hope that the weight will be distributed approximately 1/2 on both sides)
    hyperedges = []
    hypernodes = [HyperNode(i) for i in range(n)]
    edge_counter = 0
    for i in range(int(conductance * volume / 2)):
        done = False
        while not done:
            # The maximum size of the hyperedge is 1/conductance.
            hyperedge = np.random.permutation(A+B)[:np.random.randint(2, int(1/conductance)+1)]
            ids = np.array([int(h.id) for h in hyperedge])
            if ids[ids < int(n/2)].sum() > 0 and ids[ids >= int(n/2)].sum() > 0:
                hyperedges.append(HyperEdge(hyperedge, 1.0, edge_counter))
                edge_counter += 1
                done = True
    hypergraph1 = HyperGraph(hyperedges=hyperedges, hypernodes=hypernodes)
    volume_so_far = hypergraph1.deg_by_node.sum()
    print("Volume so far: {}, edges crossing: {}".format(volume_so_far, len(hyperedges)))
    
    # Add edges to the two bipartitions
    for bip in [A, B]:
        vol_bip = hypergraph1.deg_by_node[[x.id for x in bip]].sum()
        vol_added = 0
        while vol_bip + vol_added < volume / 2.0:
            # Add hyperedges inside the two bipartitions
            r = np.random.randint(2, int(1/conductance))
            hyperedges.append(HyperEdge(np.random.permutation(bip)[:r], 1.0, edge_counter))
            edge_counter += 1
            vol_added += r
    
    
    hypergraph2 = HyperGraph(hyperedges=hyperedges, hypernodes=A+B)
    print("Total volume: {}, Volume A: {}, Volume B: {}".format(
        np.sum(hypergraph2.deg_by_node),
        np.sum(hypergraph2.deg_by_node[:int(0.5*n)]), 
        np.sum(hypergraph2.deg_by_node[int(0.5*n):])))
    my_conductance = hypergraph2.compute_conductance([(i in set([x.id for x in A])) for i in range(n)])
    print("Conductance: {}".format(my_conductance))
    
    # Take some random cuts and check that indeed the conductance for them is high!
    for i in range(1, 100):
        cut1, cut2, cut3 = np.zeros(n), np.zeros(n), np.zeros(n)
        while not((cut1.max() > 0.5 and cut2.max() > 0.5 and cut3.max() > 0.5) and (
            cut1.min() < 0.5 and cut2.min() < 0.5 and cut3.min() < 0.5)):
            cut1 = np.array([False if j > n/2 else np.random.rand() > i/(100) for j in range(n)])
            cut2 = np.array([False if j < n/2 else np.random.rand() > i/(100) for j in range(n)])
            cut3 = np.array([np.random.rand() > i/(100) for j in range(n)])

        assert hypergraph2.compute_conductance(cut1) > my_conductance - 0.001
        assert hypergraph2.compute_conductance(cut2) > my_conductance - 0.001
        assert hypergraph2.compute_conductance(cut3) > my_conductance - 0.001

    # Assert that the graph is connected.
    assert np.all([len(hypergraph2.adj_list[x]) > 0 for x in hypergraph2.adj_list])
    
    filename = "{out_folder}/hypergraph_vol_{vol}_n_{n}_cond_{conductance}.txt".format(
        out_folder=args.out_folder,
        vol=str(volume),
        n=str(n),
        conductance=str(conductance).replace(".", "_"))  
    os.makedirs(args.out_folder, exist_ok=True)
    
    with open(filename, "w") as f:
        for he in hypergraph2.hyperedges:
            row = ""
            assert len(he.hypernodes) > 1
            for hn in he.hypernodes:
                hn: HyperNode
                row += "{} ".format(hn.id)
            row += "1.0" # The weight of the hyperedge.
            print(row, file=f)
    print("Output written to file {}".format(filename))



