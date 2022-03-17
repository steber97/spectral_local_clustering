using System;
using System.Collections.Generic;
using System.Security.AccessControl;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SubmodularHeatEquation
{
    public class LocalClusteringStar : LocalClusteringAlgorithm
    {
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double alpha)
        {
            // Create start graph
            var time = new System.Diagnostics.Stopwatch();
            time.Start();
            Graph starGraph = createStarGraph(hypergraph);
            
            double min_conductance = double.MaxValue;
            bool[] best_cut = new bool[hypergraph.n];
            
            Vector<double> p0 = DenseVector.Create(starGraph.n, 0.0);
            p0[startingVertex] = 1.0;
            Vector<double> ppr = PageRank.ComputePageRank(starGraph.M, p0, alpha, 1e-8);
            Vector<double> ppr_hypergraph = ppr.SubVector(0, hypergraph.n);
            
            bool[] cut = hypergraph.ComputeBestSweepCut(ppr_hypergraph);
            
            return cut;
        }

        public Graph createStarGraph(Hypergraph hypergraph)
        {
            // special nodes for hyperedges have indices n + i, for i in range(0, m).
            List<List<int>> edges = new List<List<int>>();
            List<double> weights = new List<double>();
            for (int i = 0; i < hypergraph.m; i++)
            {
                foreach (int node in hypergraph.edges[i])
                {   
                    edges.Add(new List<int>(){node, i + hypergraph.n});
                    weights.Add(hypergraph.weights[i]);
                }
            }
            Graph graph = new Graph(edges, weights);
            
            
            return graph;
        }
    }
}