using System;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SubmodularHeatEquation
{
    public class PageRank
    {
        public static Vector<double> ComputePageRank(SparseMatrix M, Vector<double> p0, double alpha, double delta)
        {
            Vector<double> p_t = DenseVector.Create(p0.Count, 0.0);
            p0.CopyTo(p_t);
            while (true)
            {
                Vector<double> p_t_1 = alpha * M * p_t + (1 - alpha) * p0;
                bool stop = true;
                for (int i = 0; i < p_t_1.Count; i++)
                {
                    if (Math.Abs(p_t_1[i] - p_t[i]) > delta)
                    {
                        stop = false;
                        break;
                    }
                }
                if (stop)
                    break;
                p_t = p_t_1;
            }
            return p_t;
        }
    }
}