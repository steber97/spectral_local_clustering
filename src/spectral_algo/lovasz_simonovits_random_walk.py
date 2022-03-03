import argparse
import numpy as np
from src.spectral_algo.input_loader import input_loader_graph
from src.data_structures.graph import Edge, Graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description="Run Lovasz-Simonovits Random Walk process on graph.")
parser.add_argument("--dataset", 
                    help="dataset name", 
                    choices=["email_eu", "five_nodes_graph_1", "lastfm_asia"],
                    type=str)
args = parser.parse_args()


def compute_lovasz_simonovits_curve(graph: Graph, probabilities: np.array):
    """
    - \sigma := \sum_{v\in V} d(v)
    - p := vector probabilities
    - A := adjacency matrix
    - d := diagonal of D matrix (simply the vertex degree)


    Curve I(x): [0, \sigma] -> [0, 1] = \sum_{e \in E} p(e) = \sum_{(u, v) \ in E} p[u] * A[u][v] / d[u]
    after having sorted the edges by \rho_{u,v} = p[u] / d[u] = 

    Since the Lovasz-Simonovits curve is pointwise linear, it is enough to list the x,y coordinates of 
    the non-derivabile points. It is assured that the first point is (0, 0) and the last is (sigma, 1)
    """
    x = []
    y = []
    edge_rho = []
    for node in graph.nodes:
        probability_edge = 0.0
        for edge in graph.adj_list[node.id]:
            probability_edge += edge.weight
            edge_rho.append((edge, probabilities[node.id] / graph.D[node.id, node.id]))
        probability_edge /= graph.D[node.id, node.id]
        assert np.abs(probability_edge - 1.0) < 0.00001
    edges_sorted = sorted(edge_rho, key=lambda x: (x[1], x[0].start.id), reverse=True)
    x_iter = 0
    y_iter = 0
    assert np.abs(np.sum(graph.A) - np.sum(np.diag(graph.D))) < 0.0001
    for edge, rho in edges_sorted:
        edge: Edge
        x_iter += edge.weight
        y_iter += rho * edge.weight
        x.append(x_iter)
        y.append(y_iter)
    
    return x, y



def compute_cut_lovasz_simonovits_sweep(graph: Graph, probabilities: np.array) -> np.array:
    """
    Given a vertex probability vector and a graph, compute a bipartition (sweep)
    using as vertex sorting quantity the value \rho(u) = p(u)/d(u)
    If everything goes well, we should be able to find a good conductance cut.
    """
    vector_rho = [(n, probabilities[n.id]/graph.D[n.id, n.id]) for n in graph.nodes]
    rho_sorted = sorted(vector_rho, key=lambda x: (x[1], -x[0].id), reverse=True)
    bipartition = np.array([False for i in range(len(graph.nodes))])
    best_bipartition = None
    best_conductance = 1.1  # Max value, will always be updated.
    volume_S = 0.0
    volume_S_compl = np.sum(np.diag(graph.D))
    edges_crossing = 0
    print("Perform Lovasz Simonovits sweep")
    # Compute the conductance taking into account that, at every iteration,
    # We only need to edit the in/out-coming edges from node. 
    for i in tqdm(range(0, len(graph.nodes)-1)):
        # Add node i to the bipartition
        node = rho_sorted[i][0]
        bipartition[node.id] = True
        volume_S += graph.D[node.id, node.id]
        volume_S_compl -= graph.D[node.id, node.id]
        # Exactly the same outcome as
        # conductance = self.compute_conductance(bipartition)
        # But computed faster.
        for i in range(len(graph.nodes)):
            if i != node.id:
                if bipartition[i] != bipartition[node.id]:
                    # Add outcoming edges from node.
                    edges_crossing += graph.A[node.id, i]
                else:
                    # Remove outcoming edges from node.
                    edges_crossing -= graph.A[node.id, i]
            
        conductance_online = edges_crossing / min(volume_S, volume_S_compl)
        assert conductance_online <= 1
        if conductance_online < best_conductance:
            # Found a better cut, update the best bipartition
            best_bipartition = bipartition.copy()
            best_conductance = conductance_online
    print("Best conductance: {}".format(best_conductance))
    # print("Best bipartition: {}".format(best_bipartition))
    # print(rho_sorted)
    return best_bipartition



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
    # print(p_0)
    
    # Simulate the random walk process.
    start_time = time.time()
    eigvals, eigvect = np.linalg.eig(largest_cc_graph.M)
    print("computed eigenvalue in {} seconds".format(time.time() - start_time))
    mu = sorted(eigvals, reverse=True)[1]  # Take second largest eigenvalue
    # print("eigenvalues: {}".format(eigvals))
    # print("eigenvectors: {}".format(eigvect))
    print("second largest eigenvalue: {}".format(mu))
    p_t = p_0
    pi_v = np.array(
        [largest_cc_graph.D[i, i] / np.sum(np.diag(largest_cc_graph.D)) for i in range(
            len(largest_cc_graph.nodes))])
    # print("static probability vector: {}".format(pi_v))

    # In order to get an estimate of how many iterations we need in order to converge, we can use the 
    # theorem 5 of lecture 5 TTCS CS455 2019, which states that 
    # I^t(x) <= min(\sqrt(x), \sqrt(\sigma - x)) * (1 - \phi^2/2)^t + (x / \sigma)
    # Notice that x / \sigma is simply the straight line. So if we want the error to be small, we need t 
    # large enough so that (1 - \phi^2 / 2)^t << \sqrt(x) or \sqrt(\sigma - x)
    # Of course we cannot estimate the correct \phi, so that we can at most use the cheeger-sweep approximation.

    cheeger_sweep_bipartition = largest_cc_graph.cheeger_sweep()
    # Estimation of phi, let's call it phi hat
    phi_hat = largest_cc_graph.compute_conductance(bipartition=cheeger_sweep_bipartition)
    # We know that phi_hat >= phi => (1 - phi^2/2)^t >= (1 - phi_hat^2/2)
    # So that if I set t large enough to converge for 1 - phi_hat^2/2, it will hold also for the optimal phi.
    # I will set t such that min(\sqrt(x), \sqrt(\sigma - x) * (1 - \phi_hat ^ 2 / 2)^t = 1/10
    # t = log_{1 - phi_hat^2 / 2}(1/(10 * min(\sqrt(x), \sqrt(sigma - x)))) using t = \sigma/2 because it is the largest value
    t_last = np.log(1 / (10 * np.sqrt(np.sum(np.diag(largest_cc_graph.D)) / 2))) / np.log(1 - ((phi_hat**2) / 2))
    print("t last: {}".format(t_last))
    ls_curve_per_t = []
    epochs = t_last
    for t in tqdm(range(int(np.ceil(epochs)))):
        # Evolve the probability vector p_{t+1} = M @ p_t
        p_t_1 = largest_cc_graph.M @ p_t
        # Check that for all nodes the theorem 1 notes 4 lecture 455 2019 EPFL
        assert np.sum(np.abs(p_t_1 - pi_v) <= np.sqrt(
                np.diag(largest_cc_graph.D) / largest_cc_graph.D[starting_vertex, starting_vertex]
            ) * (mu ** t) + 0.00001) == len(largest_cc_graph.nodes)
        p_t = p_t_1
        assert np.abs(np.sum(p_t_1) - 1.0) < 0.0001
        partition = compute_cut_lovasz_simonovits_sweep(largest_cc_graph, p_t)

        x, y = compute_lovasz_simonovits_curve(largest_cc_graph, p_t)
        ls_curve_per_t.append((x, y))
    
    for t in range(int(np.ceil(epochs))):
        plt.plot(ls_curve_per_t[t][0], ls_curve_per_t[t][1])
    plt.show()