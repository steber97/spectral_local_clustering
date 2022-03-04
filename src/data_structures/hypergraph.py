import numpy as np
import pandas as pd
from typing import List
from src.data_structures.merge_find_set import MergeFindSet


class HyperNode:
    def __init__(self, id):
        self.id = id


class HyperEdge:
    def __init__(self, hypernodes: List[HyperNode], weight: float) -> None:
        self.hypernodes = set()
        for hn in hypernodes:
            self.hypernodes.add(hn)
        self.weight = weight


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
