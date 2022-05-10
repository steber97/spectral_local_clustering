using System;

using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Diagnostics;
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
        private const int repetitions = 50;


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
            dataset_to_infile["fauci_email_no_cc"] = "../../instance/fauci_email_no_cc_LCC.txt";
            dataset_to_infile["fauci_email_cc"] = "../../instance/fauci_email_cc_LCC.txt";
            
            dataset_to_outfile["graphprod"] = "../../../output/output_conductances_graphprod.json";
            dataset_to_outfile["netscience"] = "../../../output/output_conductances_netscience.json";
            dataset_to_outfile["arxiv"] = "../../../output/output_conductances_opsahl-collaboration.json";
            dataset_to_outfile["dblp_kdd"] = "../../../output/output_conductances_dblp_kdd.json";
            dataset_to_outfile["fauci_email_no_cc"] = "../../../output/output_conductances_fauci_email_no_cc.json";
            dataset_to_outfile["fauci_email_cc"] = "../../../output/output_conductances_fauci_email_cc.json";
            
            string dataset = args[0];


            string infile = dataset_to_infile[dataset];
            string outfile = dataset_to_outfile[dataset];
            
            LocalClusteringHeatEquation lche = new LocalClusteringHeatEquation();
            LocalClusteringStar lcs = new LocalClusteringStar();
            LocalClusteringClique lcc = new LocalClusteringClique();
            LocalClusteringDiscreteGraphIteration lcdgi = new LocalClusteringDiscreteGraphIteration();
            
            Hypergraph hypergraph = Hypergraph.Open(infile);
            Debug.Assert(hypergraph.getCC().Count == 1);
            
            // Take a random permutation of the starting vertices. We are going to use the first this.repetitions (50).
            int[] startingVertices = new int[hypergraph.n];
            for (int i = 0; i < hypergraph.n; i++)
                startingVertices[i] = i;
            Random random = new Random();
            startingVertices = startingVertices.OrderBy(x => random.Next()).ToArray();

            // These two lists must be ordered in the same way.
            string[] methods = {"Heat_equation", "Star", "Clique", "Discrete"};
            LocalClusteringAlgorithm[] algos = {lche, lcs, lcc, lcdgi};

            Vector<double> alphas = CreateVector.Dense<double>(new double[]{0.05, 0.1, 0.2, 0.5});
            List<Vector<double>> paramslist = new List<Vector<double>>()
            {
                alphas,  // alpha
                alphas,  // alpha
                alphas,  // alpha
                (1.0 / alphas) * 2.0,         // 1/alpha
            };

            // map method -> list of results ordered by alpha value
            Dictionary<string, List<Result>> results = new Dictionary<string, List<Result>>();
            for (int j = 0; j < algos.Length; j++)
            {
                results[methods[j]] = new List<Result>();
                for (int param = 0; param < paramslist[0].Count; param++)
                {
                    Result res = new Result();
                    res.param = paramslist[j][param];
                    
                    for (int i = 0; i < repetitions; i++)
                    {
                        // start from a random vertex.
                        int vInit = startingVertices[i];
                        res.startVertices.Add(vInit);
                        var time = new System.Diagnostics.Stopwatch();
                        time.Start();
                        // Compute the cut using any local clustering algorithm.
                        bool[] cut = algos[j].LocalClustering(hypergraph, vInit, paramslist[j][param]);
                        time.Stop();
                        // Compute the conductance for the given cut.
                        res.conductance.Add(hypergraph.conductance(cut));
                        double ts = time.Elapsed.TotalMilliseconds;
                        res.time.Add(ts);
                    }
                    results[methods[j]].Add(res);
                }
            }
            
            // Print the json results file.
            string json = JsonSerializer.Serialize(results);
            File.WriteAllText(outfile, json);
            
        }
        
    }
}
