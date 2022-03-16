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
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double param)
        {
            // Create start graph
            var time = new System.Diagnostics.Stopwatch();
            time.Start();
            Graph starGraph = createStarGraph(hypergraph);
            var A_cand = new List<double>();
            const double eps = 0.9;
            for (int i = 0; i <= Math.Log(hypergraph.n * hypergraph.m) / Math.Log(1 + eps); i++)
            {
                A_cand.Add(Math.Pow(1 + eps, i) / (hypergraph.n * hypergraph.m));
            }
            
            double min_conductance = double.MaxValue;
            bool[] best_cut = new bool[hypergraph.n];
            foreach (double alpha in A_cand)
            {
                Vector<double> p0 = DenseVector.Create(starGraph.n, 0.0);
                p0[startingVertex] = 1.0;
                Vector<double> ppr = PageRank.ComputePageRank(starGraph.M, p0, alpha, 1e-8);
                ppr = ppr.SubVector(0, hypergraph.n);
                
                bool[] cut = hypergraph.ComputeBestSweepCut(ppr);
                double conductance = hypergraph.conductance(cut);
                if (min_conductance > conductance)
                {
                    min_conductance = conductance;
                    best_cut = cut;
                }
            }
            Console.WriteLine("time(s): " + time.ElapsedMilliseconds/1000.0);
            return best_cut;
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