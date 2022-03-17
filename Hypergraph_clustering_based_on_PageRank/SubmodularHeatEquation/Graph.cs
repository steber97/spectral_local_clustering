using System;
using System.Collections.Generic;
using System.ComponentModel;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SubmodularHeatEquation
{
    
    public class Graph
    {
        public List<Dictionary<int, double>> adj_list = new List<Dictionary<int, double>>();
        public SparseMatrix _M = new SparseMatrix(1, 1);
        public SparseMatrix _A = new SparseMatrix(1, 1);
        public SparseMatrix _D = new SparseMatrix(1, 1);
        public SparseMatrix _D_Inv = new SparseMatrix(1, 1);
        
        public int Degree(int v)
        {
             
            return adj_list[v].Count;
        }

        public double w_Degree(int v)
        {
            double sum = 0;
            foreach (var neighbor_val in adj_list[v].Values)
            {
                sum += neighbor_val;
            }
            return sum;
        }

        public int n
        {
            get
            {
                return adj_list.Count;
            }
        }

        public int m
        {
            get
            {
                int sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += adj_list[i].Keys.Count;
                }
                return sum / 2;
            }
        }

        public SparseMatrix D
        {
            get
            {
                if (_D.NonZerosCount == 0)
                {
                    _D = SparseMatrix.CreateDiagonal(n, n, 1.0) ;
                    for (int i = 0; i < n; i++)
                        _D[i, i] = w_Degree(i);
                }

                return _D;
            }
        }

        public SparseMatrix A
        {
            get
            {
                if (_A.NonZerosCount == 0)
                {
                    _A = new SparseMatrix(n, n);
                    for (int i = 0; i < n; i++)
                    {
                        foreach (var v in adj_list[i])
                        {
                            _A[i, v.Key] += v.Value;
                        }
                    }
                }

                return _A;
            }
        }

        public SparseMatrix D_Inv
        {
            get
            {
                if (_D_Inv.NonZerosCount == 0)
                {
                    _D_Inv = new SparseMatrix(n, n);
                    for (int i = 0; i < n; i++)
                    {
                        _D_Inv[i, i] = (1.0 / w_Degree(i));
                    }
                }

                return _D_Inv;
            }
        }

        public SparseMatrix M
        {
            get
            {
                if (_M.NonZerosCount == 0)
                {
                    _M = 0.5 * (SparseMatrix.CreateDiagonal(n, n, 1.0) + (A * D_Inv));
                }

                return _M;
            }
        }

        public void AddEdge(List<int> edge, double w = 1)
        {
            while (Math.Max(edge[0], edge[1]) >= adj_list.Count)
            {
                adj_list.Add(new Dictionary<int, double>());
            }

            if (!adj_list[edge[0]].ContainsKey(edge[1]))
                adj_list[edge[0]][edge[1]] = 0;
            if (!adj_list[edge[1]].ContainsKey(edge[0]))
                adj_list[edge[1]][edge[0]] = 0;
            adj_list[edge[0]][edge[1]] += w; 
            adj_list[edge[1]][edge[0]] += w;
        }

        public Graph(List<List<int>> edges, List<double> weights)
        {
            for (int i = 0; i < edges.Count; i++)
            {
                AddEdge(edges[i], weights[i]);
            }
        }
    }
}