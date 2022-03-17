using System;

using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json.Converters;

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
            
            dataset_to_outfile["graphprod"] = "../../../output/output_conductances_graphprod.csv";
            dataset_to_outfile["netscience"] = "../../../output/output_conductances_netscience.csv";
            dataset_to_outfile["arxiv"] = "../../../output/output_conductances_opsahl-collaboration.csv";
            dataset_to_outfile["dblp_kdd"] = "../../../output/output_conductances_dblp_kdd.csv";

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
            startingVertices = startingVertices.OrderBy(x => random.Next()).ToArray();

            string[] methods = {"Heat_equation", "Star", "Clique", "Discrete"};
            double[,] conductances = new double[methods.Length, 50];
            double[,] times = new double[methods.Length, 50];
            LocalClusteringAlgorithm[] algos = {lche, lcs, lcc, lcdgi};
            for (int i = 0; i < 50; i++)
            {
                // start from a random vertex.
                int vInit = startingVertices[i];
                for (int j = 0; j < algos.Length; j++)
                {
                    var time = new System.Diagnostics.Stopwatch();
                    time.Start();
                    bool[] cut = algos[j].LocalClustering(hypergraph, vInit, 0.0);
                    conductances[j, i] = hypergraph.conductance(cut);
                    time.Stop();
                    double ts = time.Elapsed.TotalMilliseconds;
                    times[j, i] = ts;
                }
            }

            using (StreamWriter writer = new StreamWriter(outfile))  
            {
                for (int i = 0; i < methods.Length; i++)
                {
                    writer.Write(methods[i] + "_conductance,");
                    writer.Write(methods[i] + "_time");
                    if (i != methods.Length - 1)
                        writer.Write(",");
                }
                writer.Write("\n");  
                for (int i = 0; i < 50; i++)
                {
                    string line = "";
                    for (int j = 0; j < methods.Length; j++)
                    {
                        line += conductances[j, i];
                        line += ",";
                        line += times[j, i];
                        if (j != methods.Length - 1)
                            line += ",";
                    }
                    writer.WriteLine(line);
                }
            }  
            
        }


    }
}
