from ast import arg
from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from src.spectral_algo.hypergraph_clique_algorithm import HyperGraphLocalClusteringByClique
from src.spectral_algo.input_loader import input_loader_hypergraph
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset", type=str, choices=[
    'network_theory', 'opsahl_collaboration'
])
args = args.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    hypergraph = input_loader_hypergraph(args.dataset)

    print("Input taken in {}s".format(time.time() - start_time))
    mu = 0.1
    conductances = []
    local_clustering_clique = HyperGraphLocalClusteringByClique(hypergraph=hypergraph)
    for i in tqdm(range(50)):
        # Take a random vertex.
        v = hypergraph.hypernodes[np.random.randint(0, len(hypergraph.hypernodes))]
        cut = local_clustering_clique.hypergraph_local_clustering_by_clique(hypergraph, v, mu)
        conductance = hypergraph.compute_conductance(cut)
        conductances.append(conductance)
    
    plt.plot(range(len(conductances)), sorted(conductances))
    plt.show()
