using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace SubmodularHeatEquation
{
    public class LocalClusteringHeatEquation : LocalClusteringAlgorithm
    {
        public bool[] LocalClustering(Hypergraph hypergraph, int startingVertex, double param)
        {
            return Proposed_local_round(hypergraph, startingVertex);
        }
        
        public bool[] Proposed_local_round(Hypergraph hypergraph, int v_init)
        {

            var time = new System.Diagnostics.Stopwatch();
            time.Start();

            int n = hypergraph.n;
            int m = hypergraph.m;

            const double eps = 0.9;
            
            const double dt = 1.0;
            const double T = 30.0;

            var A_cand = new List<double>();
            for (int i = 0; i <= Math.Log(n * m) / Math.Log(1 + eps); i++)
            {
                A_cand.Add(Math.Pow(1 + eps, i) / (n * m));
            }

            var edge_size = new Dictionary<int, int>();
            for (int eid = 0; eid < hypergraph.m; eid++)
            {
                edge_size.Add(eid, hypergraph.ID_rev[eid].Count());
            }

            double min_conductance = double.MaxValue;
            bool[] best_cut = new bool[hypergraph.n];

            foreach (double alpha in A_cand)
            {

                var vec = CreateVector.Dense<double>(n);

                vec[v_init] = 1.0;

                vec = Hypergraph.Simulate_round(hypergraph, vec, v_init, dt, T, alpha);

                for (int i = 0; i < n; i++)
                {
                    vec[i] /= hypergraph.w_Degree(i);
                }

                int[] index = Enumerable.Range(0, n).ToArray<int>();
                Array.Sort<int>(index, (a, b) => vec[a].CompareTo(vec[b]));

                Array.Reverse(index);

                double vol_V = 0;
                for (int i = 0; i < n; i++) vol_V += hypergraph.w_Degree(i);

                var num_contained_nodes = new Dictionary<int, int>();
                for (int eid = 0; eid < hypergraph.m; eid++)
                {
                    num_contained_nodes.Add(eid, 0);
                }

                double cut_val = 0;
                double vol_S = 0;
                double conductance = double.MaxValue;
                int best_index = -1;

                foreach (int i in index)
                {
                    vol_S += hypergraph.w_Degree(i);
                    if (vol_S <= vol_V / 10.0)
                    {
                        foreach (var e in hypergraph.incident_edges[i])
                        {
                            if (num_contained_nodes[e] == 0)
                            {
                                cut_val += hypergraph.weights[e];
                            }
                            if (num_contained_nodes[e] == edge_size[e] - 1)
                            {
                                cut_val -= hypergraph.weights[e];
                            }
                            num_contained_nodes[e] += 1;
                        }
                        conductance = cut_val / Math.Min(vol_S, vol_V - vol_S);
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
            }
            time.Stop();
            TimeSpan ts = time.Elapsed;

            Console.WriteLine("conductance: " + min_conductance);
            Console.WriteLine("time(s): " + time.ElapsedMilliseconds/1000.0);
            return best_cut;
        }
        
        public Vector<double> Simulate_round(Hypergraph H, Vector<double> vec, int v_init, double dt, double T, double alpha)
        {

            var active_edges = new List<List<int>>();
            var active = new List<int>(new int[H.m]);

            foreach (var e in H.incident_edges[v_init])
            {
                active_edges.Add(H.ID_rev[e]);
                active[e] = 1;
            }

            var cur_time = 0.0;
            while (cur_time < T)
            {
                var next_time = Math.Min(cur_time + dt, T);
                vec = Iterate_round(H, vec, v_init, next_time - cur_time, alpha, active_edges, active);
                cur_time = next_time;
            }
            return vec;
        }
        
        public Vector<double> Iterate_round(Hypergraph H, Vector<double> vec, int v_init, double dt, double alpha, List<List<int>> active_edges, List<int> active)
        {
            var dv = T_round(H, vec, v_init, alpha, active_edges);
            var res = vec;
            res -= dv * dt;

            for (int i = 0; i < H.n; i++)
            {
                if (res[i] < 1e-5)
                {
                    res[i] = 0;
                }
            }
            
            var new_active_edges = new List<int>();

            for (int i = 0; i < H.n; i++)
            {
                if (vec[i] == 0 && res[i] != 0)
                {
                    foreach (var f in H.incident_edges[i])
                    {
                        if (active[f] == 0)
                        {
                            new_active_edges.Add(f);
                            active[f] = 1;
                        }
                    }
                }
            }

            foreach (var e in new_active_edges)
            {
                active_edges.Add(H.ID_rev[e]);
            }

            return res;
        }
        
        public Vector<double> T_round(Hypergraph H, Vector<double> vec, int v_init, double alpha, List<List<int>> active_edges)
        {
            var res = CreateVector.Dense<double>(H.n);

            const double eps = 1e-8;

            foreach (var edge in active_edges)
            {
                var argmaxs = new List<int>();
                var argmins = new List<int>();
                double maxval = double.MinValue, minval = double.MaxValue;
                foreach (var v in edge)
                {
                    var val = vec[v] / H.w_Degree(v);
                    if (val > maxval + eps)
                    {
                        maxval = val;
                        argmaxs.Clear();
                        argmaxs.Add(v);
                    }
                    else if (val > maxval - eps)
                    {
                        argmaxs.Add(v);
                    }

                    if (val < minval - eps)
                    {
                        minval = val;
                        argmins.Clear();
                        argmins.Add(v);
                    }
                    else if (val < minval + eps)
                    {
                        argmins.Add(v);
                    }
                }
                foreach (var v in argmaxs)
                {
                    res[v] += H.weights[H.ID[edge]] * (maxval - minval) / argmaxs.Count;
                }
                foreach (var v in argmins)
                {
                    res[v] -= H.weights[H.ID[edge]] * (maxval - minval) / argmins.Count;
                }
            }

            var res_init = CreateVector.Dense<double>(H.n);
            vec.CopyTo(res_init);
            res_init[v_init] -= 1;

            var mix = (1 - alpha) * res + alpha * res_init;

            return mix;
        }
    }
}