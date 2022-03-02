import argparse
import numpy as np
from src.spectral_algo.input_loader import input_loader_graph

parser = argparse.ArgumentParser(description="Run Lovasz-Simonovits Random Walk process on graph.")
parser.add_argument("--dataset", 
                    help="dataset name", 
                    choices=["email_eu", "five_nodes_graph_1", "lastfm_asia"],
                    type=str)
args = parser.parse_args()

if __name__ == "__main__":
    read_graph = input_loader_graph(args.dataset)
    graph = read_graph()
    print("Number of vertices: {}, edges: {}".format(len(graph.nodes), len(graph.edges_list)))
    largest_cc_graph = graph.get_largest_cc()
    print("Largest CC nodes: {}, edges: {}".format(len(largest_cc_graph.nodes), len(largest_cc_graph.edges_list)))
    
    # p_0 concentrated in a single node at the beginning.
    p_0 = np.zeros(len(largest_cc_graph.nodes))
    starting_vertex = np.random.randint(len(largest_cc_graph.nodes))
    p_0[starting_vertex] = 1.0
    print(p_0)
    
    # Simulate the random walk process.
    eigvals, eigvect = np.linalg.eig(largest_cc_graph.M)
    mu = sorted(eigvals, reverse=True)[1]  # Take second largest eigenvalue
    print("eigenvalues: {}".format(eigvals))
    # print("eigenvectors: {}".format(eigvect))
    print("second largest eigenvalue: {}".format(mu))
    p_t = p_0
    pi_v = np.array(
        [largest_cc_graph.D[i, i] / np.sum(np.diag(largest_cc_graph.D)) for i in range(
            len(largest_cc_graph.nodes))])
    print("static probability vector: {}".format(pi_v))
    for t in range(1000):
        p_t_1 = largest_cc_graph.M @ p_t
        print(p_t_1)
        print(np.abs(p_t_1 - pi_v) <= np.sqrt(
                np.diag(largest_cc_graph.D) / largest_cc_graph.D[starting_vertex, starting_vertex]
            ) * (mu ** t) + 0.00001)
        p_t = p_t_1

