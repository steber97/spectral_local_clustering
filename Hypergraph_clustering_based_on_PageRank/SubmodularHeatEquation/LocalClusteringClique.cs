using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SubmodularHeatEquation
{
    public class LocalClusteringClique : LocalClusteringAlgorithm
    {
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double param)
        {
            Graph cliqueGraph = CreateCliqueGraph(hypergraph);
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
                Vector<double> p0 = DenseVector.Create(cliqueGraph.n, 0.0);
                p0[startingVertex] = 1.0;
                Vector<double> ppr = PageRank.ComputePageRank(cliqueGraph.M, p0, alpha, 1e-8);
                
                bool[] cut = hypergraph.ComputeBestSweepCut(ppr);
                double conductance = hypergraph.conductance(cut);
                if (min_conductance > conductance)
                {
                    min_conductance = conductance;
                    best_cut = cut;
                }
            }
            return best_cut;
        }

        private Graph CreateCliqueGraph(Hypergraph hypergraph)
        {
            List<List<int>> edges_graph = new List<List<int>>();
            List<double> weights = new List<double>();
            for (int i = 0; i < hypergraph.edges.Count; i++)
            {
                for (int j = 0; j < hypergraph.edges[i].Count; j++)
                {
                    for (int k = j + 1; k < hypergraph.edges[i].Count; k++)
                    {
                        edges_graph.Add(new List<int>(){hypergraph.edges[i][j], hypergraph.edges[i][k]});
                        weights.Add(hypergraph.weights[i]);
                    }
                }
            }

            return new Graph(edges_graph, weights);
        }
    }
}