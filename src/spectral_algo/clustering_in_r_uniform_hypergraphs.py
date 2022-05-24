from src.data_structures.hypergraph import HyperEdge, HyperGraph, HyperNode
from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from src.plots.plotter_conductances import plot_results
from src.spectral_algo.hypergraph_discrete_algorithm import HyperGraphLocalClusteringDiscrete
from src.spectral_algo.hypergraph_clique_algorithm import HyperGraphLocalClusteringByClique
from src.spectral_algo.hypergraph_random_algo import HyperGraphLocalClusteringRandom
from src.spectral_algo.hypergraph_star_algorithm import HyperGraphLocalClusteringByStar
from src.spectral_algo.input_loader import input_loader_hypergraph

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import os
from ast import arg
from multiprocessing import Pool

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset_folder", type=str, choices=[
    "hypergraph_conductance_0_01_vol_10000_n_100"
])
args = args.parse_args()

input_dataset_map = {
    "hypergraph_conductance_0_01_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_10000_n_100"
}

REPETITIONS = 10
MU = 0.5  # take cuts as large as 1/2 the volume (no local clustering)

if __name__=="__main__":
    iterations = {}
    np.random.seed(42)
    print(os.listdir(input_dataset_map[args.dataset_folder]))
    for file in os.listdir(input_dataset_map[args.dataset_folder]):
        iterations[file] = []
        if ".txt" in file:
            hypergraph = input_loader_hypergraph("{}/{}".format(input_dataset_map[args.dataset_folder], file))
            stationary_distribution = np.array([hypergraph.deg_by_node[n.id] / np.sum(hypergraph.deg_by_node) for n in hypergraph.hypernodes])
            starting_vertices = np.random.permutation(range(len(hypergraph.hypernodes)))
            n = len(hypergraph.hypernodes)
            for rep in range(REPETITIONS):
                algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
                # Starting vector centered in one vertex.
                p_t = np.zeros(n)
                p_t[starting_vertices[rep]] = 1.0
                p_t_dt = np.zeros(n)
                
                iteration = 0
                conductances = []
                best_cut = None
                best_conductance = 1.0
                # Repeat until we have converged
                while True:
                    cut, conductance, p_t_dt = algo.perform_one_iteration(hypergraph, p_t, MU)
                    conductances.append(conductance)
                    if best_cut is None or conductance < best_conductance:
                        best_cut = cut
                        best_conductance = conductance
                    iteration += 1
                    if np.max(np.abs(stationary_distribution - p_t_dt)) < 1 / n:
                        break
                    p_t = p_t_dt
                    assert np.abs(np.sum(p_t) - 1.0) < 0.00001
                iterations[file].append(iteration)
            
    for file in iterations:
        iterations[file] = np.array(iterations[file])
        print("{}: {} +- {}".format(file, iterations[file].mean(), iterations[file].std()))

