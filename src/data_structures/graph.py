from __future__ import annotations
from re import A

from typing import List, Tuple
import numpy as np
import pandas as pd
from src.data_structures.merge_find_set import MergeFindSet
from tqdm import tqdm
import time
from scipy.sparse import csr_matrix, identity, dia_matrix


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
        return "{} -> {}, weight {}".format(self.start, self.end, self.weight)


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        """
        Nodes ids must be incremental ids.
        If the graph is unweighted, leave all edge weights to 1 (float value in the tuple).
        """
        self.nodes = nodes
        self.edges = sorted(edges, key=lambda x: (x.start.id, x.end.id))
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
        
        self._A = None  # See method getA()
        self._D = None  # See method getD()
        self._degree_list = None  # See method getDensityList()
        self._D_inv_sqrt = None  # See method getDInvSqrt()
        self._D_inv = None  # See method getDInv()
        self._L = None  # See method getL()
        self._M = None

        # Assert that every column sums up to 1.0
        # assert np.sum([np.abs(np.sum(self.getM()[:, i]) - 1.0) < 0.0001 for i in range(len(self.nodes))]) == len(self.nodes)

    def __repr__(self) -> str:
        res = ""
        for e in self.edges_list:
            res += "{} -> {}, weight {}\n".format(e.start, e.end, e.weight)
        return res

    def getTotVol(self):
        return np.sum(self.getDegreeList())

    def getA(self):
        if self._A is None:
            self._A = csr_matrix((
                    [edge.weight for edge in self.edges], 
                    ([edge.start.id for edge in self.edges], [edge.end.id for edge in self.edges])), 
                (len(self.nodes), len(self.nodes)))
        return self._A
        
    def getDegreeList(self):
        if self._degree_list is None:
            # Sum by row
            self._degree_list = np.array(self.getA().sum(axis=1)).flatten()
        return self._degree_list
    
    def getD(self):
        if self._D is None:
            # Offset is simply zero.
            self._D = dia_matrix((self.getDegreeList(), [0]), (len(self.nodes), len(self.nodes)))
        return self._D

    def getDInvSqrt(self):
        if self._D_inv_sqrt is None: 
            # Initialize D^{-0.5}
            self._D_inv_sqrt = dia_matrix((1 / np.sqrt(self.getDegreeList()), [0]), (len(self.nodes), len(self.nodes)))
        return self._D_inv_sqrt

    def getDInv(self):
        if self._D_inv is None: 
            # Initialize D^{-1}
            self._D_inv = dia_matrix((1 / self.getDegreeList(), [0]), (len(self.nodes), len(self.nodes)))
        return self._D_inv

    def getL(self):
        if self._L is None:
            # Laplacian L = I - D_inv @ A
            self._L = identity(len(self.nodes)) - self.getDInv().dot(self.getA())
        return self._L
        
    def getM(self):
        if self._M is None:
            # Transition probability matrix
            A = self.getA()
            DInv = self.getDInv()
            self._M = 1/2 * (identity(len(self.nodes)) + (A.dot(DInv))) 
        return self._M

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
                vol_S += self.getDegreeList()[node.id]
            else:
                vol_S_compl += self.getDegreeList()[node.id]
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
                        # Add outcoming edges from node.
                        edges_crossing += self.A[node.id, i]
                    else:
                        # Remove outcoming edges from node.
                        edges_crossing -= self.A[node.id, i]
                
            conductance_online = edges_crossing / min(volume_S, volume_S_compl)
            assert conductance_online <= 1
            if conductance_online < best_conductance:
                # Found a better cut, update the best bipartition
                best_bipartition = bipartition.copy()
                best_conductance = conductance_online
        print("Best conductance: {}".format(best_conductance))
        # Assert that cheeger inequality is correct
        print("Second eigenvalue: {}".format(eigval[1]))
        assert best_conductance <= np.sqrt(eigval[1] * 2.0)
        assert eigval[1]/2.0 <= best_conductance
        return best_bipartition
    
    def compute_lovasz_simonovits_curve(self, probabilities: np.array):
        """
        - \sigma := \sum_{v\in V} d(v)
        - p := vector probabilities
        - A := adjacency matrix
        - d := diagonal of D matrix (simply the vertex degree)


        Curve I(x): [0, \sigma] -> [0, 1] = \sum_{e \in E} p(e) = \sum_{(u, v) \ in E} p[u] * A[u][v] / d[u]
        after having sorted the edges by \rho_{u,v} = p[u] / d[u] = 

        Since the Lovasz-Simonovits curve is pointwise linear, it is enough to list the x,y coordinates of 
        the non-derivabile points. It is assured that the first point is (0, 0) and the last is (sigma, 1)

        Returns:
            - x
            - y
            - vertices sorted by prob/deg
        """
        x = []
        y = []
        starting_edges_sorted_by_rho = []
        edge_rho = []
        for node in self.nodes:
            for edge in self.adj_list[node.id]:
                edge_rho.append((edge, probabilities[node.id] / self.getDegreeList()[node.id]))
        edges_sorted = sorted(edge_rho, key=lambda x: (x[1], x[0].start.id), reverse=True)
        x_iter = 0
        y_iter = 0
        assert np.abs(np.sum(self.getA()) - np.sum(self.getDegreeList())) < 0.0001
        for edge, rho in edges_sorted:
            edge: Edge
            x_iter += edge.weight
            y_iter += rho * edge.weight
            x.append(x_iter)
            y.append(y_iter)
            starting_edges_sorted_by_rho.append(edge)
        
        return x, y, starting_edges_sorted_by_rho

    def I_t(self, x: np.array, y: np.array, k: float):
        """
        Given the list of x's and y's of the Lovasz-Simonovits curve (computed with compute_lovasz_simonovits_curve)
        return the value of I_t for the given value k.

        Parameters
        ----------
        x
        y
        k

        Returns
        -------

        """
        # Do a linear search for the moment:
        # Then it can be improved with a binary search.
        assert x[-1] >= k >= 0
        x_ = [0] + x
        y_ = [0] + y
        for i in range(len(x_)-1):
            if k >= x_[i] and k <= x_[i+1]:
                return y_[i] + ((k - x_[i]) / (x_[i+1] - x_[i])) * (y_[i+1] - y_[i])
        assert False  # This should never happen really.

    def compute_j_star(self, k):
        """
        Return the number of vertices such that their volume is >= k, after having sorted them by degree.
        Parameters
        ----------
        k

        Returns
        -------

        """
        nodes_sorted_by_degree = sorted(self.nodes, key=lambda x: self.getDegreeList()[x.id], reverse=False)
        tot_volume = 0
        for i in range(len(nodes_sorted_by_degree)):
            if tot_volume <= k and tot_volume + self.getDegreeList()[nodes_sorted_by_degree[i].id] >= k:
                return i + (k - tot_volume) / (self.getDegreeList()[nodes_sorted_by_degree[i].id])
                # return i + 1
            tot_volume += self.getDegreeList()[nodes_sorted_by_degree[i].id]
