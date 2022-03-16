using System;

using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;

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
            string filename = "../../instance/dbpedia-writer_LCC.txt";
            int vInit = 0;
            LocalClusteringHeatEquation lche = new LocalClusteringHeatEquation();
            Hypergraph hypergraph = Hypergraph.Open(filename);
            bool[] cut1 = lche.LocalClustering(hypergraph, vInit, 0.0);
            Console.WriteLine(hypergraph.conductance(cut1));
        }


    }
}
