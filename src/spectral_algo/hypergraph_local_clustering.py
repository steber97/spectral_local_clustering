from ast import arg
from multiprocessing import Pool
from data_structures.hypergraph import HyperEdge, HyperGraph, HyperNode
from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph
from spectral_algo.kshiteej_process import HyperGraphLocalClusteringKshiteejProcess
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
    'network_theory', 'opsahl_collaboration', 'dblp_kdd', 'dbpedia_writer',
    "n_400_d_10_r_8"
])
args = args.parse_args()


def local_clustering_multithread(v: HyperNode, solver, hypergraph: HyperGraph, mu: float):
    cut = solver.hypergraph_local_clustering(hypergraph, v, mu)
    conductance = hypergraph.compute_conductance(cut)
    return conductance 


if __name__ == "__main__":
    start_time = time.time()
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_6_d_2_r_4.txt")
    # hypergraph = read_graph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")
    hypergraph = input_loader_hypergraph(args.dataset)

    print("Input taken in {}s".format(time.time() - start_time))
    mu = 0.1
    solvers = [HyperGraphLocalClusteringByClique(hypergraph=hypergraph), 
               HyperGraphLocalClusteringByStar(hypergraph=hypergraph),
               HyperGraphLocalClusteringKshiteejProcess(hypergraph=hypergraph)]
    labels = ['clique', 'star', 'kshiteej']
    conductances_per_solver = {}
    running_time_per_label = {}
    for label in labels:
        conductances_per_solver[label] = []
        running_time_per_label[label] = []
    
    for i in tqdm(range(50)):
        # Take a random vertex.
        v = hypergraph.hypernodes[np.random.randint(0, len(hypergraph.hypernodes))]
        conductances = []
        for label, solver in zip(labels, solvers):
            start_time = time.time()
            conductances.append(local_clustering_multithread(v, solver, hypergraph, mu))
            running_time_per_label[label].append(time.time() - start_time)
        
        for j, c in enumerate(conductances):
            conductances_per_solver[labels[j]].append(c)
    
    fig, axes = plt.subplots(1, 2)
    for label in labels:
        x_s = []
        y_s = []
        conductances_sorted = sorted(conductances_per_solver[label])
        x_s.append(0)
        y_s.append(conductances_sorted[0])
        for i in range(1, len(conductances_sorted)):
            x_s.append(i)
            y_s.append(conductances_sorted[i-1])
            x_s.append(i)
            y_s.append(conductances_sorted[i])
            
        axes[0].plot(x_s, y_s, label=label)
    
    axes[1].boxplot([running_time_per_label[label] for label in labels], labels=labels)

    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].legend()

    plt.show()
