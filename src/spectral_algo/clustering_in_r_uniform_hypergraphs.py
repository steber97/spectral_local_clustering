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
import unicodedata

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset_folder", type=str, choices=[
    "hypergraph_conductance_0_01_vol_10000_n_100", "hypergraph_conductance_0_1_vol_10000_n_100",
    "hypergraph_conductance_0_01_vol_1000_n_100", "hypergraph_conductance_0_05_vol_10000_n_100"
])
args = args.parse_args()

input_dataset_map = {
    "hypergraph_conductance_0_01_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_10000_n_100",
    "hypergraph_conductance_0_1_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_1_vol_10000_n_100",
    "hypergraph_conductance_0_01_vol_1000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_1000_n_100",
    "hypergraph_conductance_0_05_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_05_vol_10000_n_100"
}

REPETITIONS = 1
MU = 0.5  # take cuts as large as 1/2 the volume (no local clustering)

def get_r(file: str) -> int:
    return int(file.split("r_")[1].split("_")[0])


if __name__=="__main__":
    iterations = {}
    np.random.seed(42)
    deltas_by_file = {}
    for file in os.listdir(input_dataset_map[args.dataset_folder]):
        if ".txt" in file:
            iterations[file] = []
            print("Processing file {}".format(file))
            hypergraph = input_loader_hypergraph("{}/{}".format(input_dataset_map[args.dataset_folder], file))
            stationary_distribution = np.array([hypergraph.deg_by_node[n.id] / np.sum(hypergraph.deg_by_node) for n in hypergraph.hypernodes])
            starting_vertices = np.random.permutation(range(len(hypergraph.hypernodes)))
            n = len(hypergraph.hypernodes)
            best_conductances = []
            
            for rep in tqdm(range(REPETITIONS)):
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
                max_delta = []
                while True:
                    cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
                    conductances.append(conductance)
                    if best_cut is None or conductance < best_conductance:
                        best_cut = cut
                        best_conductance = conductance
                    iteration += 1
                    delta = np.max(np.abs(stationary_distribution - p_t_dt))
                    max_delta.append(delta)
                    if delta < (1 / (n**(2))):
                        break
                    p_t = p_t_dt
                    assert np.abs(np.sum(p_t) - 1.0) < 0.00001
                iterations[file].append(iteration)
                best_conductances.append(best_conductance)
                deltas_by_file[file] = max_delta
    for file in sorted(deltas_by_file.keys(), key=lambda x: get_r(x)):
        plt.plot([x for x in range(len(deltas_by_file[file]))], 
                    (deltas_by_file[file]), 
                    label="r={}".format(get_r(file)))
    plt.title("Mixing time in r-uniform hypergraphs")
    plt.ylabel("l-infinity norm (p_t - {})".format(unicodedata.lookup("GREEK SMALL LETTER PI")))
    plt.yscale("log")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig("{}/mixing_r_uniform_hypergraph.png".format(input_dataset_map[args.dataset_folder]), dpi=300)
