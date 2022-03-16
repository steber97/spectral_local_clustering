using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
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
        
        public static Hypergraph RandomHypergraph(int n, int m, int k)
        {
            var H = new Hypergraph();

            for (int i = 0; i < m; i++)
            {
                var edge = new List<int>();
                for (int j = 0; j < k; j++)
                {
                    int v = (int)(Util.Xor128() % n);
                    edge.Add(v);
                }
                edge = edge.Distinct().ToList();
                H.AddEdge(edge);
            }
            return H;
        }

        public int Degree(int v)
        {
            return incident_edges[v].Count;
        }

        public double TotalVolume()
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
                res += Degree(i);
            return res;
        }

        public double w_Degree(int v)
        {
            double sum = 0;
            foreach (int e in incident_edges[v])
            {
                sum += weights[e];
            }
            return sum;
        }

        public int n
        {
            get
            {
                return incident_edges.Count;
            }
        }

        public int m
        {
            get
            {
                return edges.Count;
            }
        }

        public void AddEdge(List<int> edge, double w = 1)
        {
            int eid = edges.Count;
            edges.Add(edge);
            foreach (var v in edge)
            {
                while (v >= incident_edges.Count)
                {
                    incident_edges.Add(new List<int>());
                }
                incident_edges[v].Add(eid);
            }
            weights[eid] = w;
            ID[edge] = eid;
        }

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

            int[] index = Enumerable.Range(0, n).ToArray<int>();
            Array.Sort<int>(index, (a, b) => vec[a].CompareTo(vec[b]));

            Array.Reverse(index);

            double vol_V = 0;
            for (int i = 0; i < n; i++) vol_V += w_Degree(i);

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
                    double conductance = cut_val / Math.Min(vol_S, vol_V - vol_S);
                    //Console.WriteLine($"{cut_val}, {vol_S}, {vol_V}, {conductance}");
                    if (conductance < min_conductance)
                    {
                        min_conductance = conductance;
                        best_index = i;
                        for (int j = 0; j < best_cut.Length; j++)
                        {
                            best_cut[j] = false;
                        }

                        foreach (var j in index)
                        {
                            best_cut[j] = true;
                            if (j == i)
                                break;
                        }
                    }
                }
                else
                {
                    break;
                }
            }
            return best_cut;
        }
    }

}
