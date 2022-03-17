using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Channels;
using System.Web;
using MathNet.Numerics.Integration;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;

namespace SubmodularHeatEquation
{
    public class LocalClusteringDiscreteGraphIteration : LocalClusteringAlgorithm
    {
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double param)
        {
            double dt = 0.25;  // constant
            int epochs = (int) param;
            Vector<double> p0 = CreateVector.Dense(hypergraph.n, 0.0);
            p0[startingVertex] = 1.0;

            Vector<double> pt = CreateVector.Dense(hypergraph.n, 0.0);
            p0.CopyTo(pt);
            double min_conductance = Double.MaxValue;
            bool[] best_cut = new bool[hypergraph.n];

            List<double> max_diff = new List<double>();
            for (int i = 0; i < 1000; i++)
            {
                Graph graph = BuildGraph(hypergraph, pt);
                SparseMatrix Mt = ((1 - dt) * SparseMatrix.CreateDiagonal(hypergraph.n, hypergraph.n, 1.0)) + (dt * graph.A * graph.D_Inv);
                
                Vector<double> pt_1 = Mt * pt;
                bool[] cut = hypergraph.ComputeBestSweepCut(pt_1);
                double conductance = hypergraph.conductance(cut);
                if (min_conductance > conductance)
                {
                    min_conductance = conductance;
                    best_cut = cut;
                }

                pt_1.CopyTo(pt);
            }

            Vector ICurve = new DenseVector(hypergraph.n);
            pt.CopyTo(ICurve);
            double total_probability = pt.Sum();
            // Lovasz-Simonovits curve, useful to check that it actually converges when the number of iterations is high
            for (int i = 0; i < hypergraph.n; i++)
                ICurve[i] /= hypergraph.w_Degree(i);
            return best_cut;
        }
        
        
        private Graph BuildGraph(Hypergraph hypergraph, Vector<double> pt)
        {
            Vector<double> pWeightedByDegree = DenseVector.Create(hypergraph.n, 0.0);
            for (int i = 0; i < hypergraph.n; i++)
                pWeightedByDegree[i] = pt[i] / hypergraph.w_Degree(i);
            List<List<int>> edges = new List<List<int>>();
            List<double> weights = new List<double>();
            Dictionary<int, double> edges_counter_per_node = new Dictionary<int, double>();
            for (int i = 0; i < hypergraph.n; i++)
                edges_counter_per_node.Add(i, 0.0);

            for (int i = 0; i < hypergraph.edges.Count; i++)
            {
                // For every edge, take the simple edge max->min
                int min_prob_node = hypergraph.edges[i][0];
                int max_prob_node = hypergraph.edges[i].Last();
                bool all_equal = true;
                for (int j = 0; j < hypergraph.edges[i].Count; j++)
                {
                    if (Math.Abs(pWeightedByDegree[hypergraph.edges[i][j]] -
                                 pWeightedByDegree[hypergraph.edges[i][0]]) > 1e-8)
                    {
                        all_equal = false;
                        break;
                    }
                }

                if (!all_equal)
                {
                    
                    for (int j = 0; j < hypergraph.edges[i].Count; j++)
                    {
                        if (pWeightedByDegree[hypergraph.edges[i][j]] < pWeightedByDegree[min_prob_node])
                        {
                            min_prob_node = hypergraph.edges[i][j];
                        }

                        if (pWeightedByDegree[hypergraph.edges[i][j]] > pWeightedByDegree[max_prob_node])
                        {
                            max_prob_node = hypergraph.edges[i][j];
                        }
                    }
                    edges.Add(new List<int>(){min_prob_node, max_prob_node});
                    weights.Add(hypergraph.weights[i]);
                    edges_counter_per_node[min_prob_node] += hypergraph.weights[i];
                    edges_counter_per_node[max_prob_node] += hypergraph.weights[i];
                }
                else
                {
                    if (hypergraph.edges[i].Count > 1)
                    {
                        min_prob_node = hypergraph.edges[i][0];
                        max_prob_node = hypergraph.edges[i][1];
                        edges.Add(new List<int>(){min_prob_node, max_prob_node});
                        weights.Add(hypergraph.weights[i]);
                        edges_counter_per_node[min_prob_node] += hypergraph.weights[i];
                        edges_counter_per_node[max_prob_node] += hypergraph.weights[i];
                    }
                    else
                    {
                        // It is the same vertex, do nothing and wait to add a self loop at the end.
                    }
                }
            }
            
            // Add self loops.
            for (int i = 0; i < hypergraph.n; i++)
            {
                if (edges_counter_per_node[i] < hypergraph.w_Degree(i))
                {
                    // half the weight, since it is split in two symmetrical self loops.
                    edges.Add(new List<int>(){i, i});
                    weights.Add((hypergraph.w_Degree(i) - edges_counter_per_node[i]) / 2);
                }
            }
            
            Graph graph = new Graph(edges, weights);
            
            return graph;
        }
    }
}