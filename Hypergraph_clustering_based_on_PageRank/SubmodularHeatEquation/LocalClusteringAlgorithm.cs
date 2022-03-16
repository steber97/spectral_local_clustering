using System;
using System.Collections.Generic;

namespace SubmodularHeatEquation
{
    public interface LocalClusteringAlgorithm
    {
        bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double param);
    }
}