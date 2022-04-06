from ast import arg
from multiprocessing import Pool
from src.data_structures.hypergraph import HyperEdge, HyperGraph, HyperNode
from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from src.plots.plotter_conductances import plot_results
from src.spectral_algo.hypergraph_discrete_algorithm import HyperGraphLocalClusteringDiscrete
from src.spectral_algo.hypergraph_clique_algorithm import HyperGraphLocalClusteringByClique
from src.spectral_algo.hypergraph_random_algo import HyperGraphLocalClusteringRandom
from src.spectral_algo.hypergraph_star_algorithm import HyperGraphLocalClusteringByStar
from src.spectral_algo.input_loader import input_loader_hypergraph
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

from src.spectral_algo.result import Result

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset", type=str, choices=[
    'graphprod', 'netscience', 'arxiv', 'dblp_kdd', 'dbpedia_writer',
    "n_400_d_10_r_8"
])
args = args.parse_args()


def local_clustering_multithread(v: HyperNode, solver, hypergraph: HyperGraph, mu: float):
    cut = solver.hypergraph_local_clustering(hypergraph, v, mu)
    conductance = hypergraph.compute_conductance(cut)
    return conductance 


input_dataset_map = {
    "graphprod": "Hypergraph_clustering_based_on_PageRank/instance/graphprod_LCC.txt",
    "netscience": "Hypergraph_clustering_based_on_PageRank/instance/netscience_LCC.txt",
    "arxiv": "Hypergraph_clustering_based_on_PageRank/instance/opsahl-collaboration_LCC.txt",
    "n_400_d_10_r_8": "datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt"
}


if __name__ == "__main__":
    start_time = time.time()
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    hypergraph = input_loader_hypergraph(args.dataset)

    print("Input taken in {}s".format(time.time() - start_time))
    mu = 0.1
    solvers = [HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph),
               # HyperGraphLocalClusteringByStar(hypergraph=hypergraph),
               # HyperGraphLocalClusteringRandom(hypergraph=hypergraph)
              ]
    labels = ['discrete',
              # 'star',
              # 'random'
             ]
    conductances_per_solver = {}
    running_time_per_label = {}
    for label in labels:
        conductances_per_solver[label] = []
        running_time_per_label[label] = []

    # Restore these!
    alphas = np.array([0.05, 0.1, 0.2, 0.5])
    epochs = 1.0 / alphas * 2.0
    params_list = [epochs, alphas, epochs]
    repetitions = 10
    results = {}
    starting_vertices = np.random.permutation(range(len(hypergraph.hypernodes)))
    for i, algo in tqdm(enumerate(solvers)):
        result = []
        for param in params_list[i]:
            res = Result()
            res.param = param
            for rep in range(repetitions):
                v = hypergraph.hypernodes[starting_vertices[rep]]
                res.startVertices.append(v.id)
                start = time.time()
                cut = algo.hypergraph_local_clustering(hypergraph, v, param, mu)
                end = time.time()
                conductance = hypergraph.compute_conductance(cut)
                res.conductance.append(conductance)
                res.time.append(end - start)
            result.append(res)
            for rep in range(repetitions):
                v = hypergraph.hypernodes[starting_vertices[rep]]
                start = time.time()
                cut = algo.hypergraph_local_clustering(hypergraph,
                                                       v,
                                                       np.log(len(hypergraph.hyperedges)) / res.conductance[rep]**2, # Epochs
                                                       mu,
                                                       res.conductance[rep])
                end = time.time()
                conductance = hypergraph.compute_conductance(cut)
            result.append(res)
        results[labels[i]] = result

    # Check that the rule I_t(k) \leq min(sqrt(k/d), sqrt((m - k)/d)) e^(-phi^2 t) + k/2m
    plot_results(results, args.dataset)
