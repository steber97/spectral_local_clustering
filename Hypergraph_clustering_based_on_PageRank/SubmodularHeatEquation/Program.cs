using System;

using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;

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
            string filename = "../../instance/netscience_LCC.txt";

            LocalClusteringHeatEquation lche = new LocalClusteringHeatEquation();
            LocalClusteringStar lcs = new LocalClusteringStar();

            Hypergraph hypergraph = Hypergraph.Open(filename);
            int[] startingVertices = new int[hypergraph.n];
            for (int i = 0; i < hypergraph.n; i++)
                startingVertices[i] = i;
            Random random = new Random();
            startingVertices = startingVertices.OrderBy(x => random.Next()).ToArray();

            string[] methods = {"Heat_equation", "star"};
            double[,] conductances = new double[2, 50];             
            for (int i = 0; i < 50; i++)
            {
                // start from a random vertex.
                int vInit = startingVertices[i];
                bool[] cut_heat_eq = lche.LocalClustering(hypergraph, vInit, 0.0);
                bool[] cut_star = lcs.LocalClustering(hypergraph, vInit, 0.0);  
                conductances[0, i] = hypergraph.conductance(cut_heat_eq);
                conductances[1, i] = hypergraph.conductance(cut_star);
            }

            using (StreamWriter writer = new StreamWriter("../../../output/output_conductances.csv"))  
            {
                for (int i = 0; i < methods.Length; i++)
                {
                    writer.Write(methods[i]);
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
                        if (j != methods.Length - 1)
                            line += ",";
                    }
                    writer.WriteLine(line);
                }
            }  
            
        }


    }
}
