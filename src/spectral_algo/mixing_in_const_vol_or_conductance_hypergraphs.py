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

args = argparse.ArgumentParser("Perform discrete algo, and checks that mixing happens within correct time.")
args.add_argument("--dataset", type=str, choices=[
    "cond_0_05", "vol_10000"
])
args = args.parse_args()

input_dataset_map = {
    "cond_0_05": "datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05",
    "vol_10000": "datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000"
}

REPETITIONS = 5
MU = 0.5  # take cuts as large as 1/2 the volume (no local clustering)

def inv_sq(x: float):
    return 1/np.array(x)**2

def fit_function(fun, x: np.array, y: np.array) -> float:
    valmin = np.min(y / fun(x))
    valmax = np.max(y / fun(x))
    bestval = valmin
    i = valmin
    while i < valmax:
        if np.sum(np.power(np.abs(y - fun(x) * i), 2)) < np.sum(np.power(np.abs(y - fun(x) * bestval), 2)):
            bestval = i
        i += (valmax - valmin) / 100
    return bestval

if __name__=="__main__":
    iterations = {}
    np.random.seed(42)
    for file in os.listdir(input_dataset_map[args.dataset]):
        if ".txt" in file:
            iterations[file] = []
            hypergraph = input_loader_hypergraph("{}/{}".format(input_dataset_map[args.dataset], file))
            stationary_distribution = np.array([hypergraph.deg_by_node[n.id] / np.sum(hypergraph.deg_by_node) for n in hypergraph.hypernodes])
            starting_vertices = np.random.permutation(range(len(hypergraph.hypernodes)))
            n = len(hypergraph.hypernodes)
            print("Volume: {}, Conductance: {}".format(
                np.sum(hypergraph.deg_by_node),
                hypergraph.compute_conductance([(True if x < len(hypergraph.hypernodes)//2 else False) for x in range(len(hypergraph.hypernodes))])
            ))
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
                    cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
                    conductances.append(conductance)
                    if best_cut is None or conductance < best_conductance:
                        best_cut = cut
                        best_conductance = conductance
                    iteration += 1
                    if np.max(np.abs(stationary_distribution - p_t_dt)) < 1 / n**2:
                        break
                    p_t = p_t_dt
                    assert np.abs(np.sum(p_t) - 1.0) < 0.00001
                iterations[file].append(iteration)
            iterations[file] = np.array(iterations[file])
    x = []
    y = []
    error = []
    data = []
    if "cond" in args.dataset:
        for file in iterations:
            vol = int(file.split("vol_")[1].split("_")[0])
            conductance = float(file.split("cond_")[1].split(".")[0].replace("_", "."))
            iter = iterations[file].mean()
            std = iterations[file].std()
            data.append((vol, iter, std))
            xlabel = "volume"
            title = "Constant conductance, varying volume"
            img_name = "const_cond.png"
    elif "vol" in args.dataset:
        for file in iterations:
            vol = int(file.split("vol_")[1].split("_")[0])
            conductance = float(file.split("cond_")[1].split(".")[0].replace("_", "."))
            iter = iterations[file].mean()
            std = iterations[file].std()
            data.append((conductance, iter, std))
            xlabel = "conductance"
            title = "Constant volume, varying conductance"
            img_name = "const_vol.png"
    
    data = sorted(data, key=lambda x: x[0])
    x = [x[0] for x in data]
    y = [x[1] for x in data]
    error = [x[2] for x in data]
    plt.errorbar(x, y, yerr=error, fmt='-o' )
    plt.xlabel(xlabel)
    plt.ylabel("iterations")
    plt.title(title)
    # Plot the fit.
    # x_fit = np.array([x[0] for x in data])
    # if "cond" in args.dataset:
    #     print("Hyperparam: {}".format(fit_function(np.log, x, y)))
    #     y_fit = np.log(x_fit) * fit_function(np.log, x, y)
    # else:
    #     print("Hyperparameter: {}".format(fit_function(inv_sq, x, y)))
    #     y_fit = inv_sq(x) * fit_function(inv_sq, x, y)
    # plt.plot(x_fit, y_fit, label="fit")
    # plt.show()
    plt.savefig("{}/{}".format(input_dataset_map[args.dataset], img_name))
