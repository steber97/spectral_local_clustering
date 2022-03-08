from ast import arg
from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from src.spectral_algo.hypergraph_clique_algorithm import HyperGraphLocalClusteringByClique
from src.spectral_algo.hypergraph_star_algorithm import HyperGraphLocalClusteringByStar
from src.spectral_algo.input_loader import input_loader_hypergraph
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset", type=str, choices=[
    'network_theory', 'opsahl_collaboration', 'dblp_kdd'
])
args = args.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    hypergraph = input_loader_hypergraph(args.dataset)

    print("Input taken in {}s".format(time.time() - start_time))
    mu = 0.1
    conductances_clique, conductances_star = [], []
    local_clustering_clique = HyperGraphLocalClusteringByClique(hypergraph=hypergraph)
    local_clustering_star = HyperGraphLocalClusteringByStar(hypergraph=hypergraph)
    for i in tqdm(range(50)):
        # Take a random vertex.
        v = hypergraph.hypernodes[np.random.randint(0, len(hypergraph.hypernodes))]
        cut_clique = local_clustering_clique.hypergraph_local_clustering(hypergraph, v, mu)
        cut_star = local_clustering_star.hypergraph_local_clustering(hypergraph, v, mu)
        conductance_clique = hypergraph.compute_conductance(cut_clique)
        conductance_star = hypergraph.compute_conductance(cut_star)
        conductances_clique.append(conductance_clique)
        conductances_star.append(conductance_star)
    
    plt.plot(range(len(conductances_clique)), sorted(conductances_clique))
    plt.plot(range(len(conductances_star)), sorted(conductances_star))
    plt.show()
