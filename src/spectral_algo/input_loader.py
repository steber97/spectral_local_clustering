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
