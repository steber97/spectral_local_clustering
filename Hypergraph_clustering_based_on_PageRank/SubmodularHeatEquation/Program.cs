using System;

using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SubmodularHeatEquation
{
    class MainClass
    {
        public static double Sqr(double v) { return v * v; }
        const double dt = 0.01;


        public static Vector<double> Integrate(Func<double, Vector<double>> f, double T)
        {
            var dummy = f(0.0);
            var res = CreateVector.Dense<double>(dummy.Count);

            var t = 0.0;
            while (t < T)
            {
                double nt = Math.Min(T - t, dt);
                var v = f(t);
                res += v * nt;
                t += nt;
            }
            return res;
        }

        public static void Main(string[] args)
        {
            Dictionary<string, string> dataset_to_infile = new Dictionary<string, string>();
            Dictionary<string, string> dataset_to_outfile = new Dictionary<string, string>();
            dataset_to_infile["graphprod"] = "../../instance/graphprod_LCC.txt";
            dataset_to_infile["netscience"] = "../../instance/netscience_LCC.txt";
            dataset_to_infile["arxiv"] = "../../instance/opsahl-collaboration_LCC.txt";
            dataset_to_infile["dblp_kdd"] = "../../instance/dblp_kdd_LCC.txt";
            
            dataset_to_outfile["graphprod"] = "../../../output/output_conductances_graphprod.json";
            dataset_to_outfile["netscience"] = "../../../output/output_conductances_netscience.json";
            dataset_to_outfile["arxiv"] = "../../../output/output_conductances_opsahl-collaboration.json";
            dataset_to_outfile["dblp_kdd"] = "../../../output/output_conductances_dblp_kdd.json";

            string dataset = args[0];


            string filename = dataset_to_infile[dataset];
            string outfile = dataset_to_outfile[dataset];
            
            LocalClusteringHeatEquation lche = new LocalClusteringHeatEquation();
            LocalClusteringStar lcs = new LocalClusteringStar();
            LocalClusteringClique lcc = new LocalClusteringClique();
            LocalClusteringDiscreteGraphIteration lcdgi = new LocalClusteringDiscreteGraphIteration();
            
            Hypergraph hypergraph = Hypergraph.Open(filename);
            int[] startingVertices = new int[hypergraph.n];
            for (int i = 0; i < hypergraph.n; i++)
                startingVertices[i] = i;
            Random random = new Random();
            // Take a random permutation of all vertices. We will take the first 50.
            startingVertices = startingVertices.OrderBy(x => random.Next()).ToArray();

            string[] methods = {"Heat_equation", "Star", "Clique", "Discrete"};
            double[,] conductances = new double[methods.Length, 50];
            double[,] times = new double[methods.Length, 50];
            LocalClusteringAlgorithm[] algos = {lche, lcs, lcc, lcdgi};

            List<List<double>> paramslist = new List<List<double>>()
            {
                new List<double> {0.05, 0.1, 0.2, 0.5},
                new List<double> {0.05, 0.1, 0.2, 0.5},
                new List<double> {0.05, 0.1, 0.2, 0.5},
                new List<double> {20, 10, 5, 2},
            };

            Dictionary<string, List<Result>> results = new Dictionary<string, List<Result>>();
            for (int j = 0; j < algos.Length; j++)
            {
                results[methods[j]] = new List<Result>();
                for (int param = 0; param < paramslist[0].Count; param++)
                {
                    Result res = new Result();
                    res.param = paramslist[j][param];
                    
                    for (int i = 0; i < 50; i++)
                    {
                        // start from a random vertex.
                        int vInit = startingVertices[i];
                        res.startVertices.Add(vInit);
                        var time = new System.Diagnostics.Stopwatch();
                        time.Start();
                        bool[] cut = algos[j].LocalClustering(hypergraph, vInit, paramslist[j][param]);
                        res.conductance.Add(hypergraph.conductance(cut));
                        time.Stop();
                        double ts = time.Elapsed.TotalMilliseconds;
                        res.time.Add(ts);
                    }
                    results[methods[j]].Add(res);
                }
            }
            string json = JsonSerializer.Serialize(results);
            File.WriteAllText(outfile, json);
            
            
        }


    }
}
