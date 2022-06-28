def input_loader_graph(dataset_name: str):
    """
    Given the dataset name, return the function with the name read_graph that loads the graph 
    for the given input.
    dataset_name: choices are
        - email_eu
        - five_nodes_graph_1
        - lastfm_asia
    """
    if dataset_name == "email_eu":
        from datasets.graphs.email_eu.read_graph import read_graph
    elif dataset_name == "five_nodes_graph_1":
        from datasets.graphs.five_nodes_graph_1.read_graph import read_graph
    elif dataset_name == "lastfm_asia":
        from datasets.graphs.lastfm_asia.read_graph import read_graph
    else:
        print("Dataset name not supported!")
        exit(1)
    return read_graph


input_dataset_map = {
    "graphprod": "Hypergraph_clustering_based_on_PageRank/instance/graphprod_LCC.txt",
    "netscience": "Hypergraph_clustering_based_on_PageRank/instance/netscience_LCC.txt",
    "arxiv": "Hypergraph_clustering_based_on_PageRank/instance/opsahl-collaboration_LCC.txt",

}


def input_loader_hypergraph(dataset_name: str):
    if dataset_name == "network_theory":
        from datasets.hypergraphs.hypergraph_pagerank_paper.read_hypergraph import read_hypergraph
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/network_theory/download.tsv.dimacs10-netscience/dimacs10-netscience/out.dimacs10-netscience",
                               skiprows=1)
    if dataset_name == "opsahl_collaboration":
        from datasets.hypergraphs.hypergraph_pagerank_paper.read_hypergraph import read_hypergraph
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/opsahl_collaboration/out.opsahl-collaboration",
                               skiprows=2)
    if dataset_name == "dblp_kdd":
        from datasets.hypergraphs.hypergraph_pagerank_paper.read_hypergraph import read_hypergraph
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/dblp_kdd/out.dblp-kdd",
                               skiprows=2)
    if dataset_name == "dbpedia_writer":
        from datasets.hypergraphs.hypergraph_pagerank_paper.read_hypergraph import read_hypergraph
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/dbpedia_writer/download.tsv.dbpedia-writer/out.dbpedia-writer",
                               skiprows=2)
    if dataset_name == "n_400_d_10_r_8":
        from datasets.hypergraphs.d_regular_r_uniform.read_graph import read_graph as read_uniform_regular_hypergraph
        return read_uniform_regular_hypergraph("datasets/hypergraphs/d_regular_r_uniform/n_400_d_10_r_8.txt")

    if dataset_name == "graphprod" or dataset_name == "netscience" or dataset_name == "arxiv":
        from src.data_structures.hypergraph import HyperGraph
        return HyperGraph.read_hypergraph(input_dataset_map[dataset_name])

    if "hypergraph_conductance_0_01_vol_10000_n_100" or "hypergraph_conductance_0_1_vol_10000_n_100" or "hypergraph_conductance_0_1_vol_1000_n_100" in dataset_name:
        return HyperGraph.read_hypergraph(dataset_name)
    # Input file not found.
    raise FileNotFoundError