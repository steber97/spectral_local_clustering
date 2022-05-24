from src.data_structures.hypergraph import HyperGraph, HyperNode, HyperEdge
import argparse
import numpy as np
import pandas as pd

argparse = argparse.ArgumentParser("Random d-regular r-uniform Hypergraph generator")
argparse.add_argument("--n", help="Number of nodes in the hypergraph", type=int)
argparse.add_argument("--d", help="Degree of every node", type=int)
argparse.add_argument('--r', help='Number of hypernodes per hyperedge', type=int)
argparse.add_argument("--output_file", help="file where the hypergraph is saved", type=str)

args = argparse.parse_args()


if __name__ == "__main__":
    n = args.n
    d = args.d
    r = args.r

    # Assert that n, d, r are compatible.
    assert (n * d) / r == (n * d) // r

    hypernodes = [HyperNode(i) for i in range(n)]
    print("Edges: {}".format((n * d) // r))
    edges_set = [set() for i in range((n * d) // r)]
    
    possibilities = [i for i in range(len(edges_set))]
    for i in range(d):
        for hn in hypernodes:
            permut_pos = np.random.permutation(possibilities)
            put = False
            for j in permut_pos:
                if hn.id not in edges_set[j]:
                    if len(edges_set[j]) < r - 1:
                        edges_set[j].add(hn.id)
                        put = True
                        break 
                    else:
                        assert len(edges_set[j]) == r - 1
                        edges_set[j].add(hn.id)
                        good = True
                        for k in range(len(edges_set)):
                            if k != j:
                                if edges_set[k].intersection(edges_set[j]) == r:
                                    good = False
                        if good:
                            put = True
                            possibilities = [p for p in possibilities if p != j]
                            break
                        else:
                            edges_set[j].remove(hn.id)
    
    hyperedges = [HyperEdge([hypernodes[i] for i in edges_set[j]], weight=1, _id=j) for j in range(len(edges_set))]
    assert np.sum([len(he.hypernodes) == r for he in hyperedges]) == len(hyperedges)
    for he in hyperedges:
        conflicts = set()
        for hn in he.hypernodes:
            assert hn.id not in conflicts
            conflicts.add(hn.id)
    hypergraph = HyperGraph(hypernodes, hyperedges)

    with open(args.output_file, 'w') as f:
        f.write("{} {}\n".format(len(hypernodes), len(hyperedges)))
        for i, edge in enumerate(edges_set):
            row = ""
            for l, hn in enumerate(list(edge)):
                row += str(int(hn)) + " "
            f.write(row + "\n")
        
    print("Number of ccs: {}".format(len(hypergraph.get_CCs())))
    
    
