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
    "hypergraph_conductance_0_01_vol_1000_n_100", "hypergraph_conductance_0_05_vol_10000_n_100",
     "cond_0_05", "vol_10000"
])
args = args.parse_args()

input_dataset_map = {
    "hypergraph_conductance_0_01_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_10000_n_100",
    "hypergraph_conductance_0_1_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_1_vol_10000_n_100",
    "hypergraph_conductance_0_01_vol_1000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_01_vol_1000_n_100",
    "hypergraph_conductance_0_05_vol_10000_n_100": "datasets/hypergraphs/r_uniform/hypergraph_conductance_0_05_vol_10000_n_100", 
    "cond_0_05": "datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05",
    "vol_10000": "datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000"
}

MU = 0.5

def check_rule_for_psi_S(hypergraph, n, starting_vertices, conductance_true):
    # First we are going to check that the general Escaping Mass result is indeed correct:
    algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
    # Starting vector psi_S
    p_t = np.zeros(n)
    p_t[starting_vertices] = np.array(hypergraph.deg_by_node[starting_vertices]) / np.sum(hypergraph.deg_by_node[starting_vertices])
    p_0 = p_t.copy()
    p_t_dt = np.zeros(n)
    # Repeat until we have converged:
    iteration = 0
    cumulative_M = np.eye(n)
    cumulative_M_DS = np.eye(n)
    D_S = np.diag(np.array([x < n//2 for x in range(n)]))
    while True:
        cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
        iteration += 1  # TODO: restore dt * 2
        if (iteration * conductance_true) >= 1:
            break
        cumulative_M_DS_dt = (D_S @ algo.M_t) @ cumulative_M_DS
        cumulative_M_dt = algo.M_t @ cumulative_M
        p_t = p_t_dt
        # Assert that cumulative_M_dt @ p_0 is exactly the same as p_t_dt
        assert np.max(np.abs(cumulative_M_dt @ p_0 - p_t_dt)) < 1e-6
        # Assert that (D_S @ M_t) ^ t @ p_0 < p_t_dt in every coordinate
        assert np.min(p_t_dt - (cumulative_M_DS_dt @ p_0)) >= -1e-6
        # Assert that p_t sums to 1.
        assert np.abs(np.sum(p_t) - 1.0) < 1e-6
        # assert leaking is correct
        assert np.sum(p_t_dt[:n//2]) >= 1 - (iteration * conductance_true) / 2
        # Lemma 2.7 Spielman
        assert np.ones(n).T @ cumulative_M_DS_dt @ p_0 >= 1 - (iteration * conductance_true) / 2
        
        val1 = np.ones(n).T @ cumulative_M_DS @ p_0
        val2 = np.ones(n).T @ cumulative_M_DS_dt @ p_0
        assert val1 - val2 <= algo.dt * conductance_true + 1e-05
        cumulative_M = cumulative_M_dt
        cumulative_M_DS = cumulative_M_DS_dt

# It is assumed that the best bipartition is A,B with vertices (0,...,n/2-1), (n/2, ..., n)
if __name__ == "__main__":
    iterations = 30
    for file in os.listdir(input_dataset_map[args.dataset]):
        if ".txt" in file:
            print("Processing file {}".format(file))
            hypergraph = input_loader_hypergraph("{}/{}".format(input_dataset_map[args.dataset], file))
            
            pi = np.array([hypergraph.deg_by_node[n.id] / np.sum(hypergraph.deg_by_node) for n in hypergraph.hypernodes])
            n = len(hypergraph.hypernodes)
            starting_vertices = [x for x in range(n//2)]
            psi_S = np.zeros(n)
            psi_S[starting_vertices] = np.array(hypergraph.deg_by_node[starting_vertices]) / np.sum(hypergraph.deg_by_node[starting_vertices])
    
            D_S = np.diag(np.array([x < n//2 for x in range(n)]))
            
            conductance_star = float(file.split("cond_")[1].split(".")[0].replace("_", "."))
            conductance_true = hypergraph.compute_conductance([x <= n//2 for x in range(n)])
            print("Theoretical conductance: {}, conductance in practice: {}".format(conductance_star, conductance_true))
            
            # Check that the rule is true when the starting distribution is psi_S (general Escape Mass Lemma)
            check_rule_for_psi_S(hypergraph, n, starting_vertices, conductance_true)
            
            # Now check that the volume of the starting vertices for which it is true that 
            # the probability escaping is small 
            # >= vol(S) / 2
            vol_vertices_ok = 0
            S_g = set()
            S_prime = set()
            probability_leaking_per_t = [[0 for v in starting_vertices] for i in range(iterations)]
            for v in tqdm(starting_vertices):
                algo = HyperGraphLocalClusteringDiscrete(hypergraph=hypergraph)
                r = np.max([len(x.hypernodes) for x in hypergraph.hyperedges])
                # Starting vector psi_S
                p_t = np.zeros(n)
                p_t[v] = 1.0
                p_0 = p_t.copy()
                p_t_dt = np.zeros(n)
                cumulative_M_t_DS = np.eye(n)
                cumulative_M_t = np.eye(n)
                v_in_S_g = True
                t = 0
                chi_S = np.array([x < n//2 for x in range(n)])
                chi_S_bar = np.ones(n) - chi_S
                ls_curve = []
                while True:
                    cut, conductance, p_t_dt, graph_t = algo.perform_one_iteration(hypergraph, p_t, MU)
                    t += 1  # TODO: Restore dt * 2
                    if t == iterations:
                        break
                    cumulative_M_t_dt_DS = (D_S @ algo.M_t) @ cumulative_M_t_DS
                    cumulative_M_t_dt = algo.M_t @ cumulative_M_t
                    assert np.max(np.abs(p_t_dt - cumulative_M_t_dt @ p_0)) < 1e-6
                    # Assert that probability going out at any step is smaller than 
                    # 1 - prob that we always stay in S
                    assert chi_S_bar.T @ cumulative_M_t_dt @ p_0 <= (1 - np.ones(n).T @ cumulative_M_t_dt_DS @ p_0) + 1e-6
                    # Assert that Proposition 6.3 (Proposition 2.2 in Spielman) 
                    assert np.max(graph_t.getDInv() @ p_t) >= np.max(graph_t.getDInv() @ algo.M_t @ p_t) - 1e-6
                    # Assert that Proposition 6.4 (Proposition 2.4 in Spielman)
                    assert p_t.T @ algo.M_t @ p_t >= p_t.T @ (D_S @ algo.M_t) @ p_t - 1e-6
                    assert np.ones(n).T @ cumulative_M_t_dt @ p_0 >= \
                        np.ones(n).T @ cumulative_M_t_dt_DS @ p_0 - 1e-6
                    # Assert Lemma 2.7 in Spielman
                    assert (chi_S_bar.T @ (cumulative_M_t_dt) @ p_0) <= \
                        (1 - np.ones(n).T @ cumulative_M_t_dt_DS @ p_0) + 1e-6
                    
                    # Assert inequalities in proof of lemma on size of S^g
                    assert 0.5 * t * conductance_true >= 1 - (np.ones(n).T @ cumulative_M_t_dt_DS @ psi_S) - 1e-6
                    tot_sum = 0
                    val_eq = 1 - (np.ones(n).T @ cumulative_M_t_dt_DS @ psi_S)
                    for v_2 in starting_vertices:
                        chi_v_2 = np.zeros(n)
                        chi_v_2[v_2] = 1.0
                        tot_sum += hypergraph.deg_by_node[v_2] / np.sum(hypergraph.deg_by_node[starting_vertices]) * \
                            (1 - np.ones(n).T @ cumulative_M_t_dt_DS @ chi_v_2)
                    assert np.abs(val_eq - tot_sum) < 1e-6
                    probability_leaking = chi_S_bar.T @ cumulative_M_t_dt @ p_0
                    probability_leaking_per_t[t][v] = probability_leaking
                    # if probability_leaking >= \
                    #     (1 - np.sum([hypergraph.deg_by_node[x] for x in starting_vertices])) / hypergraph.get_volume() :
                    #     # (t * conductance_true * r) + 1e-6:
                    #     # Vertex does not belong to S^g
                    #     v_in_S_g = False
                    #     break
                    cumulative_M_t_DS = cumulative_M_t_dt_DS
                    cumulative_M_t = cumulative_M_t_dt
                    x_t, y_t, _ = graph_t.compute_lovasz_simonovits_curve(p_t)
                    ls_curve.append((x_t, y_t))
                    p_t = p_t_dt
                
                for (x, y) in ls_curve:
                    plt.plot(x, y)
                plt.title("LS-curve convergence")
                plt.xlabel("Volume (k)")
                plt.ylabel("I_t(k)")
                plt.show()
                
                if v_in_S_g:
                    vol_vertices_ok += hypergraph.deg_by_node[v]
                    S_g.add(v)

            print("Vol S^g: {}".format(vol_vertices_ok))
            print("Vol S: {}".format(np.sum([hypergraph.deg_by_node[starting_vertices]])))
            assert vol_vertices_ok >= 0.5 * np.sum([hypergraph.deg_by_node[starting_vertices]])
            