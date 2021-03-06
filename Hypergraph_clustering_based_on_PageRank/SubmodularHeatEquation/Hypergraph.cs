using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SubmodularHeatEquation
{
    public class Hypergraph
    {
        public List<List<int>> edges = new List<List<int>>();
        public List<List<int>> incident_edges = new List<List<int>>();
        public List<double> weights = new List<double>();
        public Dictionary<List<int>, int> ID = new Dictionary<List<int>, int>();
        public Dictionary<int, List<int>> ID_rev = new Dictionary<int, List<int>>();
        
        /**
         * Return the number of incident edges (NOT the total weight of the incident edges).
         * In order to obtain the total weight, see w_Degree()
         */
        public int Degree(int v)
        {
            return incident_edges[v].Count;
        }

        /**
         * Return total weight volume of the hypergraph, namely the
         * sum of all vertex weights.
         */
        public double TotalVolume()
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
                res += w_Degree(i);
            return res;
        }

        /**
         * Return the total weight of the incident edges.
         */
        public double w_Degree(int v)
        {
            double sum = 0;
            foreach (int e in incident_edges[v])
            {
                sum += weights[e];
            }
            return sum;
        }

        /**
         * Number of nodes.
         */
        public int n
        {
            get
            {
                return incident_edges.Count;
            }
        }

        /**
         * Number of edges.
         */
        public int m
        {
            get
            {
                return edges.Count;
            }
        }

        /**
         * Read a hypergraph from file.
         * Nodes are from 0 to n-1.
         * Every line of the file represents a hyperedge as a list of nodes,
         * the last number being the weight of the hyperedge.
         */
        public static Hypergraph Open(string fn)
        {
            var fs = new FileStream(fn, FileMode.Open);
            var sr = new StreamReader(fs);

            var edges = new List<List<int>>();
            var weights = new List<double>();
            int vertex_num = 0;
            for (string line; (line = sr.ReadLine()) != null;)
            {
                var words = line.Split();
                var edge = new List<int>();
                int i = 0;
                foreach (var word in words)
                {
                    if (i < words.Length - 1)
                    {
                        int v = int.Parse(word);
                        edge.Add(v);
                        if (v >= vertex_num) vertex_num = v + 1;
                        i++;
                    }
                }
                edges.Add(edge);
                // last number in every line is the edge weight.
                weights.Add(double.Parse(words.Last()));  
            }

            var H = new Hypergraph();
            H.edges = edges;
            H.weights = weights;
            for (int v = 0; v < vertex_num; v++)
            {
                H.incident_edges.Add(new List<int>());
            }
            for (int i = 0; i < edges.Count; i++)
            {
                var edge = edges[i];
                foreach (var v in edge)
                {
                    H.incident_edges[v].Add(i);
                }
                H.ID.Add(edge, i);
                H.ID_rev.Add(i, edge);
            }
            fs.Close();
            return H;
        }
        public static double Sqr(double v) { return v * v; }

        /**
         * Compute the hypergraph conductance as
         * phi(S) = volume edges cut / min(volume(S), volume(V - S))
         * where the edge e is cut if there is at least one node v\in e
         * s.t. v \in S and another u \in e s.t. u \notin S
         */
        public double conductance(bool[] cut)
        {
            double volume_cut = 0.0;
            double volume_partition = 0.0;
            for (int i = 0; i < edges.Count; i++)
            {
                bool side_true = false;
                bool side_false = false;
                foreach (int node in edges[i])
                {
                    if (cut[node])
                        side_true = true;
                    else
                        side_false = true;
                }
                // The edge is cut if there is at least one node in both sides of the cut.
                if (side_false && side_true)
                    volume_cut += weights[i];
            }

            for (int i = 0; i < cut.Length; i++)
            {
                if (cut[i])
                    volume_partition += w_Degree(i);
            }
            
            return volume_cut / Math.Min(volume_partition, TotalVolume() - volume_partition);
        }
        
        /**
         * Given a probability vector, return the best sweep cut according to the
         * quantity probability(u)/w_Degree(u).
         * Best sweep cut is the one with lowest conductance.
         */
        public bool[] ComputeBestSweepCut(Vector<double> p)
        {
            Vector<double> vec = DenseVector.Create(p.Count, 0.0);
            double min_conductance = Double.MaxValue;
            var edge_size = new Dictionary<int, int>();
            bool[] best_cut = new bool[n];
            for (int eid = 0; eid < m; eid++)
            {
                edge_size.Add(eid, ID_rev[eid].Count());
            }
            p.CopyTo(vec);
            // normalize vector wrt node degrees.
            for (int i = 0; i < n; i++)
            {
                vec[i] /= w_Degree(i);
            }
            
            // sort by increasing values of prob(u) / vol(u)
            int[] index = Enumerable.Range(0, n).ToArray<int>();
            Array.Sort<int>(index, (a, b) => vec[a].CompareTo(vec[b]));
            // Reverse the array (so that it is sorted in decreasing order).
            Array.Reverse(index);

            double vol_V = 0;
            for (int i = 0; i < n; i++) 
                vol_V += w_Degree(i);

            var num_contained_nodes = new Dictionary<int, int>();
            for (int eid = 0; eid < m; eid++)
            {
                num_contained_nodes.Add(eid, 0);
            }

            double cut_val = 0;
            double vol_S = 0;
            int best_index = -1;

            foreach (int i in index)
            {
                vol_S += w_Degree(i);
                // Return cuts smaller than 1/10 of the total volume.
                // See paper https://arxiv.org/pdf/2006.08302.pdf variable \mu 
                if (vol_S <= vol_V / 10.0)
                {
                    foreach (var e in incident_edges[i])
                    {
                        if (num_contained_nodes[e] == 0)
                        {
                            cut_val += weights[e];
                        }
                        if (num_contained_nodes[e] == edge_size[e] - 1)
                        {
                            cut_val -= weights[e];
                        }
                        num_contained_nodes[e] += 1;
                    }
                    
                    // Since volume of S is at most 1/10 of Volume of V, it is worthless to take the min.
                    double conductance = cut_val / Math.Min(vol_S, vol_V - vol_S);
                    //Console.WriteLine($"{cut_val}, {vol_S}, {vol_V}, {conductance}");
                    if (conductance < min_conductance)
                    {
                        min_conductance = conductance;
                        best_index = i;
                    }
                }
                else
                {
                    break;
                }
            }
            // Update the best cut, set to true all indices until we reach the best_index.
            foreach (int i in index)
            {
                best_cut[i] = true;
                if (i == best_index)
                    break;
            }
            return best_cut;
        }

        /**
         * Given an hypergraph, return the CCs as a list of int (ids of hypernodes).
         * CCs are returned sorted by the size (largest CC first).
         */
        public List<List<int>> getCC()
        {
            List<MergeFindSet> hn_mfs = new List<MergeFindSet>(new MergeFindSet[n]);
            for (int i = 0; i < n; i++)
            {
                hn_mfs[i] = new MergeFindSet(i);
            }

            foreach (List<int> edge in edges)
            {
                for (int j = 1; j < edge.Count; j++)
                    hn_mfs[edge[j]].merge(hn_mfs[edge[j-1]]);
            }

            Dictionary<int, List<int>> map_cc = new Dictionary<int, List<int>>();
            for (int i = 0; i < hn_mfs.Count; i++)
            {
                int root = hn_mfs[i].getRoot().value;
                if (!map_cc.ContainsKey(root))
                {
                    map_cc[root] = new List<int>();
                }
                map_cc[root].Add(i);  
            }

            List<List<int>> cc_list = new List<List<int>>();
            foreach (var a in map_cc)
            {
                cc_list.Add(a.Value);
            }
            // In order to reverse the sorting, simply negate the length of the list.
            cc_list.Sort((a,b) => -a.Count.CompareTo(b.Count));
            return cc_list;
        }
    }

}
