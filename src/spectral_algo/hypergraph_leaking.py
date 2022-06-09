from tracemalloc import start
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
import argparse
import os

args = argparse.ArgumentParser("Perform local graph clustering algorithm")
args.add_argument("--dataset", type=str, choices=[
    "hypergraph_conductance_0_01_vol_10000_n_100", "hypergraph_conductance_0_1_vol_10000_n_100",
    "hypergraph_conductance_0_01_vol_1000_n_100", "cond_0_05", "vol_10000"
])
args = args.parse_args()

input_dataset_map = {
    "hypergraph_conductance_0_01_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_10000_n_100",
    "hypergraph_conductance_0_1_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_1_vol_10000_n_100",
    "hypergraph_conductance_0_01_vol_1000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_1000_n_100",
    "cond_0_05": "datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05",
    "vol_10000": "datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000"
}

MU = 0.5

# It is assumed that the best bipartition is A,B with vertices (0,...,n/2-1), (n/2, ..., n)
if __name__ == "__main__":
    for file in os.listdir(input_dataset_map[args.dataset]):
        if ".txt" in file:
            print("Processing file {}".format(file))
            hypergraph = input_loader_hypergraph("{}/{}".format(input_dataset_map[args.dataset], file))
            stationary_distribution = np.array([hypergraph.deg_by_node[n.id] / np.sum(hypergraph.deg_by_node) for n in hypergraph.hypernodes])
            n = len(hypergraph.hypernodes)
            starting_vertices = [x for x in range(n//2)]
            # First we are going to check that the general Escaping Mass result is indeed correct:
            algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
            # Starting vector psi_S
            p_t = np.zeros(n)
            p_t[starting_vertices] = np.array(hypergraph.deg_by_node[starting_vertices]) / np.sum(hypergraph.deg_by_node[starting_vertices])
            p_0 = p_t.copy()
            p_t_dt = np.zeros(n)
            # Repeat until we have converged:
            iteration = 0
            conductance_star = float(file.split("cond_")[1].split(".")[0].replace("_", "."))
            conductance_true = hypergraph.compute_conductance([x <= n//2 for x in range(n)])
            print("Theoretical conductance: {}, conductance in practice: {}".format(conductance_star, conductance_true))
            cumulative_M = np.eye(n)
            cumulative_M_DS = np.eye(n)
            D_S = np.diag(np.array([x < n//2 for x in range(n)]))
            while True:
                cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
                iteration += 1
                if (iteration * conductance_star) >= 1:
                    break
                cumulative_M_DS_dt = (D_S @ algo.M_t) @ cumulative_M_DS
                cumulative_M_dt = algo.M_t @ cumulative_M
                p_t = p_t_dt
                assert np.abs(np.sum(p_t) - 1.0) < 0.00001
                # assert leaking is correct
                assert np.sum(p_t_dt[:n//2]) >= 1 - (iteration * conductance_star) / 2
                assert np.ones(n).T @ cumulative_M_DS_dt @ p_0 >= 1 - (iteration * conductance_star) / 2
                prob_in_S_at_t = np.sum(p_t_dt[:n//2])
                prob_always_in_S = np.ones(n).T @ cumulative_M @ p_0
                assert np.sum(p_t_dt[:n//2]) >= np.ones(n).T @ cumulative_M_DS_dt @ p_0 - 1e-6
                val1 = np.ones(n).T @ cumulative_M_DS @ p_0
                val2 = np.ones(n).T @ cumulative_M_DS_dt @ p_0
                assert val1 - val2 <= algo.dt * conductance_star + 1e-05

                cumulative_M = cumulative_M_dt
                cumulative_M_DS = cumulative_M_DS_dt
            
            # Now check that the volume of the starting vertices for which it is true that 
            # the probability escaping is small 
            # >= vol(S) / 2
            vol_vertices_ok = 0
            for v in tqdm(starting_vertices):
                algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
                # Starting vector psi_S
                p_t = np.zeros(n)
                p_t[v] = 1.0
                p_0 = p_t.copy()
                p_t_dt = np.zeros(n)
                cumulative_M_t_DS = np.eye(n)
                cumulative_M_t = np.eye(n)
                algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
                vertex_is_ok = True
                iteration = 0
                while True:
                    cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
                    iteration += 1
                    if (iteration * conductance_star) >= 1/10:
                        break
                    cumulative_M_t_dt_DS = (D_S @ algo.M_t) @ cumulative_M_t_DS
                    cumulative_M_t_dt = algo.M_t @ cumulative_M_t
                    if np.array([x >= n//2 for x in range(n)]).T @ cumulative_M_t_dt @ p_0 > \
                        (iteration * conductance_true) + 1e-6:
                        vertex_is_ok = False
                    # Assert that Proposition 6.3 (Proposition 2.2 in Spielman) 
                    assert np.max(graph_t.getDInv() @ p_t) >= np.max(graph_t.getDInv() @ algo.M_t @ p_t) - 1e-6
                    # Assert that Proposition 6.4 (Proposition 2.4 in Spielman)
                    assert p_t.T @ algo.M_t @ p_t >= p_t.T @ (D_S @ algo.M_t) @ p_t - 1e-6
                    assert p_t.T @ cumulative_M_t_dt @ p_t >= p_t.T @ cumulative_M_t_dt_DS @ p_t - 1e-6
                    # Assert Lemma 2.7 in Spielman
                    assert (np.array([x >= n//2 for x in range(n)]).T @ (cumulative_M_dt) @ p_0) <= \
                        (1 - np.ones(n).T @ cumulative_M_DS_dt @ p_0)
                    cumulative_M_t_DS = cumulative_M_t_dt_DS
                    cumulative_M_t = cumulative_M_t_dt
                if vertex_is_ok:
                    vol_vertices_ok += hypergraph.deg_by_node[v]
            print("Vol S^g: {}".format(vol_vertices_ok))
            print("Vol S: {}".format(np.sum([hypergraph.deg_by_node[starting_vertices]])))
            assert vol_vertices_ok >= 0.5 * np.sum([hypergraph.deg_by_node[starting_vertices]])