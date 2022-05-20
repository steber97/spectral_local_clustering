import argparse
import numpy as np

args_main = argparse.ArgumentParser(
    "Create a family of r-uniform hypergraphs with equal conductance, but different r")

args_main.add_argument("--conductance", type=float, help='conductance of the family')
args_main.add_argument("--r", nargs='+', help='list of r values')
args_main.add_argument('--n', type=int, help='number of nodes')

args = args_main.parse_args()

"""
The idea for creating a hypergraph with desired conductance is to
split the n vertices in 2 equal groups A, B, add random hyperedges 
of size r to both bipartitions, 
and then add in between A, B the correct amount of
hyperedges so that the cut A,B has the desired conductance.
The ideal thing would be that the volume is the same for both graphs.
"""

if __name__ == "__main__":
    n = args.n
    r_list = [int(x) for x in args.r]
    conductance = args.conductance

    # Create the two bipartitions
    A = [i for i in range(0, int(n/2))]
    B = [i for i in range(int(n/2), n)]

    

    for r in r_list: 
        volume = 100 * r

        hyperedges = []
        for i in range(int((1-conductance)*volume/(r * 2))):
            hyperedges.append(np.random.permutation(A)[:r])
            hyperedges.append(np.random.permutation(B)[:r])
        for i in range(int(conductance * volume/(r*2))):
            done = False
            while not done:
                hyperedge = np.random.permutation(A+B)[:r]
                if hyperedge[hyperedge in A] != 0 and hyperedge[hyperedge in A] != r:
                    hyperedges.append(hyperedge)
                    done = True



