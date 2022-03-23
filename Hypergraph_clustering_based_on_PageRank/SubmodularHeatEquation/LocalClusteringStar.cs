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
            
            // Initialize the probability vector centered in startingVertex (local clustering).
            Vector<double> p0 = DenseVector.Create(starGraph.n, 0.0);
            p0[startingVertex] = 1.0;
            // Whatch out: ppr has n + m entries (because the star graph has one special vertex for every hyperedge).
            Vector<double> ppr = PageRank.ComputePageRank(starGraph.M, p0, alpha, 1e-8);
            // Take only the first n entries (which are related to the nodes of the original hypergraph).
            Vector<double> ppr_hypergraph = ppr.SubVector(0, hypergraph.n);
            ppr_hypergraph /= ppr_hypergraph.Sum();  // Normalize so that the vector sums up to 1.
            
            bool[] cut = hypergraph.ComputeBestSweepCut(ppr_hypergraph);
            
            return cut;
        }

        /**
         * Create a graph of n + m vertices, so that:
         * for every edge e in H, add one special vertex v_e (they go from n to n+m-1).
         * For every edge e in H, for every node v in e, add an edge v -> n+e of weight w[e]/|e|
         */
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
                    // Weight of the edge is w(e) / |e| according to paper https://arxiv.org/pdf/2006.08302.pdf
                    weights.Add(hypergraph.weights[i] / hypergraph.edges[i].Count);
                }
            }
            Graph graph = new Graph(edges, weights);
            
            return graph;
        }
    }
}