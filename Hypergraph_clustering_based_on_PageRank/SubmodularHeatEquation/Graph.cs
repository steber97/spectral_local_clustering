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
        public SparseMatrix M = new SparseMatrix(1, 1);
        public SparseMatrix A = new SparseMatrix(1, 1);
        public SparseMatrix D = new SparseMatrix(1, 1);
        public SparseMatrix D_Inv = new SparseMatrix(1, 1);
        
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

        public void AddEdge(List<int> edge, double w = 1)
        {
            while (Math.Max(edge[0], edge[1]) >= adj_list.Count)
            {
                adj_list.Add(new Dictionary<int, double>());
            }
            adj_list[edge[0]][edge[1]] = w; 
            adj_list[edge[1]][edge[0]] = w;
        }

        public Graph(List<List<int>> edges, List<double> weights)
        {
            for (int i = 0; i < edges.Count; i++)
            {
                AddEdge(edges[i], weights[i]);
            }

            A = new SparseMatrix(n, n);
            for (int i = 0; i < n; i++)
            {
                foreach (var v in adj_list[i])
                {
                    A[i, v.Key] += v.Value;
                }
            }

            D = SparseMatrix.CreateDiagonal(n, n, 1.0) ;
            for (int i = 0; i < n; i++)
                D[i, i] = w_Degree(i);

            D_Inv = new SparseMatrix(n, n);

            for (int i = 0; i < n; i++)
            {
                D_Inv[i, i] = (1.0 / w_Degree(i));
            }

            M = 0.5 * (SparseMatrix.CreateDiagonal(n, n, 1.0) + (A * D_Inv));
        }

    }
}