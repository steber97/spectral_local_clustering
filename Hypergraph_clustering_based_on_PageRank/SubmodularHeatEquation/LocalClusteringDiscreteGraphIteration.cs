using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Channels;
using System.Security.Policy;
using System.Web;
using MathNet.Numerics.Integration;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;

namespace SubmodularHeatEquation
{
    public class LocalClusteringDiscreteGraphIteration : LocalClusteringAlgorithm
    {
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double t)
        {
            double dt = 0.25;  // constant
            int epochs = (int) t;
            Vector<double> p0 = CreateVector.Dense(hypergraph.n, 0.0);
            p0[startingVertex] = 1.0;

            Vector<double> pt = CreateVector.Dense(hypergraph.n, 0.0);
            p0.CopyTo(pt);
            double min_conductance = Double.MaxValue;
            bool[] best_cut = new bool[hypergraph.n];

            for (int i = 0; i < epochs; i++)
            {
                Graph graph = BuildGraph(hypergraph, pt);
                Vector<double> pt_1 = new DenseVector(hypergraph.n);
                for (int j = 0; j < hypergraph.n; j++)
                {
                    // Perform the update: pt_1 = ((1 - dt) * I + dt * A * D^-1) * pt
                    pt_1[j] = (1 - dt) * pt[j];
                    foreach (var v in graph.adj_list[j])
                    {
                        pt_1[j] += dt * pt[v.Key] * v.Value / graph.w_Degree(v.Key);
                    }
                }
                
                bool[] cut = hypergraph.ComputeBestSweepCut(pt_1);
                
                double conductance = hypergraph.conductance(cut);
                if (min_conductance > conductance)
                {
                    min_conductance = conductance;
                    best_cut = cut;
                }
                pt_1.CopyTo(pt);
            }

            return best_cut;
        }
        
        
        /**
         * Create the graph using the following strategy:
         * for every hyperedge e
         *      create an edge e' connecting the v_max and v_min
         *      v_max and v_min \in e, s.t.
         *      v_max is max_{u\in e} p(u)/d(u)
         *      v_min is min_{u\in e} p(u)/d(u)
         * Then, for every node, add self-loops so that original degree d(u) is preserved.
         */
        private Graph BuildGraph(Hypergraph hypergraph, Vector<double> pt)
        {
            Vector<double> pWeightedByDegree = DenseVector.Create(hypergraph.n, 0.0);
            for (int i = 0; i < hypergraph.n; i++)
                pWeightedByDegree[i] = pt[i] / hypergraph.w_Degree(i);
            List<List<int>> edges = new List<List<int>>();
            List<double> weights = new List<double>();
            // counts the weight of the edges added to every node. Eventually, it must be the same as the
            // weight in the original hypergraph.
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
                    // Look for the min and the max, since they are certainly distinct.
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
                    // When they have all equal probability/degree value.
                    if (hypergraph.edges[i].Count > 1)
                    {
                        // If there are more than 1 nodes in the hyperedge e, simply take the edge e[0]->e[1]
                        min_prob_node = hypergraph.edges[i][0];
                        max_prob_node = hypergraph.edges[i][1];
                        edges.Add(new List<int>(){min_prob_node, max_prob_node});
                        weights.Add(hypergraph.weights[i]);
                        edges_counter_per_node[min_prob_node] += hypergraph.weights[i];
                        edges_counter_per_node[max_prob_node] += hypergraph.weights[i];
                    }
                    else
                    {
                        // The hyperedge has a unique node, don't do anything since self loops are added at the end.
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