"""PageRank algorithm with explicit number of iterations.
Taken from Wikipedia https://en.wikipedia.org/wiki/PageRank, and partially adapted to our needs
In particular, we changed the stopping criterion so that the distance from the page ranks
of different epochs is small.

Returns
-------
ranking of nodes (pages) in the adjacency matrix

"""

import numpy as np
from scipy.sparse import csr_matrix


def pagerank(M, p_0: np.array, delta: float = 1e-8, alpha: float = 0.85):
    """PageRank: The trillion dollar algorithm.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
        This is actually the transition probability matrix rather than the adjacency matrix.
    p_0 : initial probability vector
    delta : stopping criterion, stop when two consecutive page rank vectors do not differ 
            by more than delta in every dimension
    alpha : float, optional
            damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    assert np.abs(np.sum(p_0) - 1.0) < 0.0001
    assert alpha < 1.0 and alpha > 0
    p_t = p_0.copy()
    while True:
        p_t_1 = alpha * M.dot(p_t) + (1 - alpha) * p_0
        # Perform truncation.
        p_t_1 = np.where(p_t_1 > 1e-5, p_t_1, 0.0)
        if np.max(np.abs(p_t_1 - p_t)) < delta:
            break
        p_t = p_t_1
    return p_t