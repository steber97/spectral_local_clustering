import argparse
import time

import numpy as np
from matplotlib import pyplot as plt

from src.data_structures.hypergraph import HyperGraph
from src.spectral_algo.hypergraph_discrete_algorithm import HyperGraphLocalClusteringDiscrete

input_dataset_map = {
    "graphprod": "Hypergraph_clustering_based_on_PageRank/instance/graphprod_LCC.txt",
    "netscience": "Hypergraph_clustering_based_on_PageRank/instance/netscience_LCC.txt",
    "arxiv": "Hypergraph_clustering_based_on_PageRank/instance/opsahl-collaboration_LCC.txt"
}

if __name__ == "__main__":
    args = argparse.ArgumentParser("Perform local graph clustering algorithm")
    args.add_argument("--dataset", type=str, choices=[
        'graphprod', 'netscience', 'arxiv', 'dblp_kdd', 'dbpedia_writer',
        "n_400_d_10_r_8"
    ])
    args = args.parse_args()
    hypergraph = HyperGraph.read_hypergraph(input_dataset_map[args.dataset])

    solver = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
    mu = 1/10

    p0 = np.zeros(len(hypergraph.hypernodes))
    p0[5] = 1.0
    pt = p0

    conductances = []

    start = time.time()
    delta = 1e-8
    while True:
        cut, conductance, pt_1 = solver.perform_one_iteration(hypergraph=hypergraph, pt=pt, mu=mu)

        conductances.append(conductance)
        pt = pt_1

        p_d = pt / hypergraph.deg_by_node
        diff = np.abs(np.min(p_d) - np.max(p_d))
        if diff < delta:
            break

    print("Total time: {}".format(time.time() - start))
    constant_pt = hypergraph.compute_lovasz_simonovits_sweep(
        np.array([hypergraph.deg_by_node[i] for i in range(len(hypergraph.hypernodes))]),
        mu)
    print(hypergraph.compute_conductance(constant_pt))
    plt.plot(range(len(conductances)), conductances)
    plt.title(args.dataset)
    plt.xlabel("iteration")
    plt.ylabel("conductance")
    plt.savefig("Hypergraph_clustering_based_on_PageRank/output/conductance_by_iteration_{}_delta_{}.png".format(
        args.dataset, delta), dpi=300)
    plt.show()
