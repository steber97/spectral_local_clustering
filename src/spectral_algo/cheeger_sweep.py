from src.data_structures.graph import Graph, Node, Edge
# from datasets.graphs.lastfm_asia.read_graph import read_graph
from datasets.graphs.email_eu.read_graph import read_graph
# from datasets.graphs.five_nodes_graph_1.read_graph import read_graph
import numpy as np

if __name__ == "__main__":
    graph = read_graph()
    print("Number of vertices: {}, edges: {}".format(len(graph.nodes), len(graph.edges_list)))
    largest_cc_graph = graph.get_largest_cc()
    print("Largest CC nodes: {}, edges: {}".format(len(largest_cc_graph.nodes), len(largest_cc_graph.edges_list)))
    
    min_conductance_cut = largest_cc_graph.cheeger_sweep()
    
    