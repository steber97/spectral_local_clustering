using System.Collections.Generic;

namespace SubmodularHeatEquation
{
    class Result
    {
        public double param{ get; set; }
        public List<double> time { get; set; }
        public List<double> conductance { get; set; }
        public List<int> startVertices{ get; set; }

        public Result()
        {
            conductance = new List<double>();
            time = new List<double>();
            startVertices = new List<int>();
        }
    }
}