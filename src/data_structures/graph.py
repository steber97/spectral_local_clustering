from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from data_structures.merge_find_set import MergeFindSet
from tqdm import tqdm
import time


class Node:
    def __init__(self, id, value) -> None:
        """
        The id must be an integer, so that it can be easily hashed into a dictionary.
        """
        self.id = id
        self.value = value
    
    def __hash__(self) -> int:
        return int(self.id)
    
    def __eq__(self, o: object) -> bool:
        return int(self.id)

    def __repr__(self):
        return "Node {}".format(self.id)


class Edge:
    def __init__(self, start: Node, end: Node, weight: float) -> None:
        self.start = start
        self.end = end
        self.weight = weight
    
    def __repr__(self):
        # return "{} -> {} with weight {}".format(self.start.id, self.end.id, self.weight)
        return "{} -> {}".format(self.start, self.end)


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        """
        Nodes ids must be incremental ids.
        If the graph is unweighted, leave all edge weights to 1 (float value in the tuple).
        """
        self.nodes = nodes
        # Assert ids are unique for the nodes. They must be in range [0, len(nodes)-1]
        assert len(np.unique([n.id for n in nodes])) == len(nodes) 
        assert np.max([n.id for n in nodes]) == len(nodes)-1 
        assert np.min([n.id for n in nodes]) == 0
        self.edges_list = edges
        self.adj_list = {}
        for node in nodes:
            self.adj_list[node.id] = []
        for edge in edges:
            self.adj_list[edge.start.id].append(edge)
        
        self.A = np.zeros((len(self.nodes), len(self.nodes)))
        for edge in edges:
            self.A[edge.start.id, edge.end.id] += edge.weight
        # Initialize D
        self.D = np.diag(np.sum(self.A, axis=1))
        # Initialize D^{-0.5}
        self.D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(self.A, axis=1)))
        # Initialize D^{-1}
        self.D_inv = np.diag(1.0 / np.sum(self.A, axis=1))
        # Laplacian L = I - D_inv @ A
        self.L = np.eye(len(self.nodes)) - self.D_inv @ self.A
        self.M = 1/2 * (np.eye(len(self.nodes)) + self.D_inv @ self.A)
        # Assert that every row sums up to 1.0
        assert np.sum([np.abs(np.sum(self.M[i, :]) - 1.0) < 0.0001 for i in range(len(self.nodes))]) == len(self.nodes)

    def get_largest_cc(self) -> Graph:
        """
        Return the largest CC as a new graph, with new increasing IDs.
        """
        nodes_mfs = {}
        for n in self.nodes:
            nodes_mfs[n.id] = MergeFindSet(n.id)
        
        for node in self.nodes:
            for edge in self.adj_list[node.id]:
                nodes_mfs[edge.start.id].merge(nodes_mfs[edge.end.id])

        ccs = [nodes_mfs[n.id].get_root() for n in self.nodes]
        largest_cc_root = pd.value_counts(ccs).index[0]
        map_node_old_to_new = {}
        new_id = 0
        new_nodes = []
        for n in self.nodes:
            if nodes_mfs[n.id].get_root() == largest_cc_root:
                map_node_old_to_new[n.id] = new_id
                new_nodes.append(Node(new_id, new_id))
                new_id += 1

        new_edges = []
        for n in self.nodes:
            if nodes_mfs[n.id].get_root() == largest_cc_root:
                for edge in self.adj_list[n.id]:
                    edge: Edge
                    assert edge.start.id in map_node_old_to_new
                    assert edge.end.id in map_node_old_to_new
                    new_edges.append(Edge(start=new_nodes[map_node_old_to_new[edge.start.id]],
                                          end=new_nodes[map_node_old_to_new[edge.end.id]],
                                          weight=edge.weight))
        return Graph(nodes=new_nodes, edges=new_edges)
                
    def compute_conductance(self, bipartition: List[bool]) -> float:
        """
        Given a graph, and a cut encoded as a boolean vector (indexed by the node id),
        return the conductance of the cut as:

        phi(bipartition) = # edges crossing the cut
                        ________________________
                            min(vol(S), vol(V - S)) 
        """
        # number of edges crossing the cut
        assert np.sum(bipartition) > 0 and np.sum(bipartition) != len(bipartition)
        crossing_edges = 0.0
        for edge in self.edges_list:
            if bipartition[edge.start.id] != bipartition[edge.end.id]:
                crossing_edges += edge.weight
        vol_S = 0.0
        vol_S_compl = 0.0
        for node in self.nodes:
            if bipartition[node.id]:
                vol_S += self.D[node.id, node.id]
            else:
                vol_S_compl += self.D[node.id, node.id]
        assert vol_S_compl > 0 and vol_S > 0
        return crossing_edges / min(vol_S, vol_S_compl)

    def cheeger_sweep(self) -> List[bool]:
        """
        Compute the cheeger sweep: 
        1) get the second eigenvector
        2) sort the nodes of the graph by their respective eigenvector value
        3) Perform the sweep: take the cut [v_0, ..., v_k] s.t. the conductance is lowest.
        """
        # Now that we have the Laplacian, we can perform the lambda 
        start_time = time.time()
        eigval, eigvect = np.linalg.eig(self.L)
        print("Compute eigevals in {} seconds".format(time.time() - start_time))
        second_eigenvect = eigvect[:, 1]
        # Make a list of (node, eigenvector_entry)
        nodes_eigvect_val = [(n, v) for (n, v) in zip(self.nodes, second_eigenvect)]
        # Sort by the eigenvector value
        nodes_eigvect_sorted = sorted(nodes_eigvect_val, key=lambda x: x[1])  
        # Perform the sweep cut.
        bipartition = np.array([False for i in range(len(self.nodes))])
        best_bipartition = None
        best_conductance = 1.1  # Max value, will always be updated.
        volume_S = 0.0
        volume_S_compl = np.sum(np.diag(self.D))
        edges_crossing = 0
        print("Perform cheeger sweep")
        # Compute the conductance taking into account that, at every iteration,
        # We only need to edit the in/out-coming edges from node. 
        for i in tqdm(range(0, len(self.nodes)-1)):
            # Add node i to the bipartition
            node = nodes_eigvect_sorted[i][0]
            bipartition[node.id] = True
            volume_S += self.D[node.id, node.id]
            volume_S_compl -= self.D[node.id, node.id]
            # Exactly the same outcome as
            # conductance = self.compute_conductance(bipartition)
            # But computed faster.
            for i in range(len(self.nodes)):
                if i != node.id:
                    if bipartition[i] != bipartition[node.id]:
                        # Add incoming and outcoming edges from node.
                        edges_crossing += self.A[node.id, i]
                    else:
                        # Remove incoming and outcoming edges from node.
                        edges_crossing -= self.A[node.id, i]
                
            conductance_online = edges_crossing / min(volume_S, volume_S_compl)
            assert conductance_online <= 1
            if conductance_online < best_conductance:
                # Found a better cut, update the best bipartition
                best_bipartition = bipartition.copy()
                best_conductance = conductance_online
        print("Best conductance: {}".format(best_conductance))
        # Assert that cheeger inequality is correct
        assert best_conductance <= np.sqrt(eigval[1] * 2.0)
        assert eigval[1]/2.0 <= best_conductance
        return best_bipartition