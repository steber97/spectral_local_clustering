from src.data_structures.graph import Graph, Node, Edge
import argparse
import numpy as np
from src.spectral_algo.input_loader import input_loader_graph

parser = argparse.ArgumentParser(description="Run Cheeger's Sweep algorithm on a graph.")
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
    
    min_conductance_cut = largest_cc_graph.cheeger_sweep()

    