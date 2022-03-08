from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from src.spectral_algo.hypergraph_clique_algorithm import hypergraph_local_clustering_by_clique
from src.spectral_algo.input_loader import input_loader_hypergraph
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


if __name__ == "__main__":
    start_time = time.time()
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    # hypergraph = input_loader_hypergraph("network_theory")
    hypergraph = input_loader_hypergraph("opsahl_collaboration")
    print("Input taken in {}s".format(time.time() - start_time))
    mu = 0.1
    conductances = []
    for i in tqdm(range(50)):
        # Take a random vertex.
        v = hypergraph.hypernodes[np.random.randint(0, len(hypergraph.hypernodes))]
        cut = hypergraph_local_clustering_by_clique(hypergraph, v, mu)
        conductance = hypergraph.compute_conductance(cut)
        conductances.append(conductance)
    print(conductances)
    plt.plot(range(len(conductances)), sorted(conductances))
    plt.show()
