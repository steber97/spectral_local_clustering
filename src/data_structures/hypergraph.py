import time

import numpy as np
import pandas as pd
from typing import List
from src.data_structures.merge_find_set import MergeFindSet


class HyperNode:
    def __init__(self, id):
        self.id = id
    
    def __eq__(self, __o: object) -> bool:
        return self.id == __o.id

    def __neq__(self, __o: object) -> bool:
        return self.id != __o.id
    
    def __hash__(self) -> int:
        return self.id
    
    def __repr__(self) -> str:
        return str(self.id)


class HyperEdge:
    def __init__(self, hypernodes: List[HyperNode], weight: float, _id: int) -> None:
        self.hypernodes = set()
        for hn in hypernodes:
            self.hypernodes.add(hn)
        self.weight = weight
        self.id = _id

    def __eq__(self, __o: object) -> bool:
        if len(self.hypernodes) != len(__o.hypernodes):
            return False
        return len(self.hypernodes.intersection(__o.hypernodes)) == len(self.hypernodes)

    def __neq__(self, __o: object) -> bool:
        if len(self.hypernodes) != len(__o.hypernodes):
            return True
        return len(self.hypernodes.intersection(__o.hypernodes)) != len(self.hypernodes)
    
    def __hash__(self) -> int:
        return hash(e for e in sorted(list(self.hypernodes), key=lambda x: x.id))


class HyperGraph:
    def __init__(self, hypernodes: List[HyperNode], hyperedges: List[HyperEdge]) -> None:
        """
        Params
        @hypernodes: ids must be incremental ids from 0 to len(hypernodes) - 1
        """
        assert len(np.unique([hn.id for hn in hypernodes])) == len(hypernodes) 
        assert np.min([hn.id for hn in hypernodes]) == 0
        assert np.max([hn.id for hn in hypernodes]) == len(hypernodes) - 1
        self.hypernodes = hypernodes
        # Assert that all hyperedges make sense.
        for he in hyperedges:
            for hn in he.hypernodes:
                assert hn.id >= 0 and hn.id < len(hypernodes)
        self.hyperedges = hyperedges

        self.adj_list = {}
        for hn in self.hypernodes:
            self.adj_list[hn.id] = []
        for he in self.hyperedges:
            for hn in he.hypernodes:
                self.adj_list[hn.id].append(he)

        self.deg_by_node = np.array([0 for i in range(len(self.hypernodes))])
        for hn in self.hypernodes:
            self.deg_by_node[hn.id] = np.sum([he.weight for he in self.adj_list[hn.id]])

    def get_volume(self):
        return np.sum(self.deg_by_node)

    def get_CCs(self) -> List[List[HyperNode]]:
        hn_mfs = {}
        for hn in self.hypernodes:
            hn_mfs[hn.id] = MergeFindSet(hn.id)

        # Merge into one cc all hypernodes in the same edge. 
        for he in self.hyperedges:
            hypernodes_list = list(he.hypernodes)
            for i in range(1, len(hypernodes_list)):
                hn_mfs[hypernodes_list[i].id].merge(hn_mfs[hypernodes_list[i-1].id])
                
            
        ccs = [hn_mfs[hn.id].get_root() for hn in self.hypernodes]
        cc_map = {}
        for hn in self.hypernodes:
            root = ccs[hn.id]
            if root not in cc_map:
                cc_map[root] = []
            cc_map[root].append(hn)

        return sorted(cc_map.values(), key=lambda x: len(x), reverse=True)        
    
    def compute_conductance(self, bipartition: np.array) -> float:
        """
        The conductance of the hypergraph is computed as follows:
        \phi(S) = vol(he \in H s.t. u\in S, v\notin S, u,v\in he) / min(vol(S), vol(V\setminus S))
        """
        assert np.sum(bipartition) > 0 and np.sum(bipartition) < len(bipartition)
        hyperedges_crossing = 0
        for hyperedge in self.hyperedges:
            bipartitions_in_cut = np.array([bipartition[hn.id] for hn in hyperedge.hypernodes])
            # Check that two vertices of the hyperedge lie on different sides of the bipartition.
            hyperedges_crossing += 0 if np.sum(bipartitions_in_cut) == 0 or np.sum(bipartitions_in_cut) == len(bipartitions_in_cut) else hyperedge.weight
        volume_1 = 0.0
        volume_2 = 0.0
        for hn in self.hypernodes:
            if bipartition[hn.id]:
                for he in self.adj_list[hn.id]:
                    volume_1 += he.weight
            else:
                for he in self.adj_list[hn.id]:
                    volume_2 += he.weight
        min_volume = min(volume_1, volume_2)
        assert min_volume > 0
        conductance = hyperedges_crossing / min_volume
        return conductance

    def compute_lovasz_simonovits_sweep(self, p, mu=0.5):
        """
        Return a bipartition, wrt the probability vector:
        sort vertices by decreasing probability, and take the best-conductance sweep cut S_j,
        @param p: probability vector
        @param mu: max fraction of the volume taken by the sweep S_j. Usually used when computing local sweep cuts.
        """
        hypernodes_sorted_by_probability = zip(self.hypernodes, p / self.deg_by_node)
        hypernodes_sorted_by_probability = sorted(hypernodes_sorted_by_probability, 
                                                  key=lambda x: (x[1], -x[0].id), 
                                                  reverse=True)

        S_j = []
        bipartition = np.array([False for i in range(len(self.hypernodes))])
        best_cut = None
        best_conductance = None
        hyperedges_crossing = 0.0
        volume_1 = 0.0
        volume_2 = np.sum([np.sum([he.weight for he in self.adj_list[hn.id]]) for hn in self.hypernodes])
        stop_volume = mu * volume_2  # When volume_1 gets to stop_volume, we need to stop.
        best_index = None
        edge_counter_per_bipartition = np.zeros(len(self.hyperedges))
        for i in range(len(hypernodes_sorted_by_probability) - 1):
            hn = hypernodes_sorted_by_probability[i][0]
            hn: HyperNode
            bipartition[hn.id] = True
            # The edge needs to be added or removed from the crossing hyperedges, if:
            for he in self.adj_list[hn.id]:
                he: HyperEdge
                bipartitions_in_hyperedge = edge_counter_per_bipartition[he.id]
                # Notice that bipartition[hn.id] is always true!!
                if bipartitions_in_hyperedge == 0:
                        hyperedges_crossing += he.weight
                if bipartitions_in_hyperedge == len(he.hypernodes):
                        hyperedges_crossing -= he.weight
                edge_counter_per_bipartition[he.id] += 1
            volume_1 += self.deg_by_node[hn.id]
            volume_2 -= self.deg_by_node[hn.id]
            conductance = hyperedges_crossing / min(volume_1, volume_2)
            if best_conductance is None or best_conductance > conductance:
                best_index = i
                best_conductance = conductance
            if volume_1 >= stop_volume:
                break  # Stop early.
        best_cut = np.array([False for i in range(len(self.hypernodes))])
        for i in range(best_index + 1):
            best_cut[hypernodes_sorted_by_probability[i][0].id] = True
        return best_cut
    
    @staticmethod
    def read_hypergraph(file: str):
        hypernodes = []
        hyperedges = []
        with open(file) as f:
            for i, l in enumerate(f):
                hyperedge = [int(hn) for hn in l.split()[:-1]]
                weight = float(l.split()[-1])
                for hn in hyperedge:
                    for j in range(len(hypernodes), hn + 1):
                        hypernodes.append(HyperNode(j))
                hyperedges.append(HyperEdge([hypernodes[j] for j in hyperedge], weight, _id=i))

        return HyperGraph(hypernodes, hyperedges)