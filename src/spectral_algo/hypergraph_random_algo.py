import numpy as np

from src.data_structures.hypergraph import HyperGraph, HyperNode


class HyperGraphLocalClusteringRandom:
    def __init__(self, hypergraph: HyperGraph):
        pass

    def hypergraph_local_clustering(self, hypergraph: HyperGraph, v: HyperNode, epochs: float,
                                    mu: float = 0.1) -> np.array:
        """
        Instead of taking a sweep over the probability vector, take a sweep over any random permutation where the first
        element is v
        Parameters
        ----------
        hypergraph
        v
        epochs
        mu

        Returns
        -------

        """
        best_cond = 1.1
        best_cut = None
        for epoch in range(int(epochs)):
            permut = np.random.permutation(range(len(hypergraph.hypernodes)))
            for i in range(len(permut)):
                if permut[i] == v.id:
                    # Swap first element with v
                    permut[0], permut[i] = permut[i], permut[0]

            # now ensure that we can take a random sweep using this order!
            prob_vector = np.zeros(len(permut))
            for i in range(len(permut)):
                prob_vector[permut[i]] = (len(permut) - i) * hypergraph.deg_by_node[permut[i]]

            prob_vector /= prob_vector.sum()

            cut = hypergraph.compute_lovasz_simonovits_sweep(prob_vector, mu)
            cond = hypergraph.compute_conductance(cut)
            if cond < best_cond:
                best_cut = cut
                best_cond = cond

        return best_cut