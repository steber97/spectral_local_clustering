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


def input_loader_hypergraph(dataset_name: str):
    from datasets.hypergraphs.hypergraph_pagerank_paper.read_hypergraph import read_hypergraph
    if dataset_name == "network_theory":
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/network_theory/download.tsv.dimacs10-netscience/dimacs10-netscience/out.dimacs10-netscience",
                               skiprows=1)
    if dataset_name == "opsahl_collaboration":
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/opsahl_collaboration/out.opsahl-collaboration",
                               skiprows=2)
    if dataset_name == "dblp_kdd":
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/dblp_kdd/out.dblp-kdd",
                               skiprows=2)
    if dataset_name == "dbpedia_writer":
        return read_hypergraph("datasets/hypergraphs/hypergraph_pagerank_paper/dbpedia_writer/download.tsv.dbpedia-writer/out.dbpedia-writer",
                               skiprows=2)