import pandas as pd
import argparse
import matplotlib.pyplot as plt
import json


args = argparse.ArgumentParser("Plot conductances")
args.add_argument("--json_file", type=str)

args = args.parse_args()


if __name__ == "__main__":
    with open(args.json_file) as f:
        results = json.load(f)
    fig, axes = plt.subplots(4, 2)
    
    methods = sorted([method for method in results])
    
    max_conductance = 0
    for method in methods:
        for i, result in enumerate(results[method]):
            x_s = []
            y_s = []
            conductances_sorted = sorted(result['conductance'])
            if conductances_sorted[-1] > max_conductance:
                max_conductance = conductances_sorted[-1]
            x_s.append(0)
            y_s.append(conductances_sorted[0])
            for j in range(1, len(conductances_sorted)):
                x_s.append(j)
                y_s.append(conductances_sorted[j-1])
                x_s.append(j)
                y_s.append(conductances_sorted[j])
            
            axes[i, 0].plot(x_s, y_s, label=method)
            if method == methods[0]:
                axes[i, 0].set_ylabel("conductance")
                axes[i, 0].set_title("Conductance for alpha={}, t={}".format(
                    result['param'], 2*1/result['param']))
                if i != len(results[method]) - 1:
                    axes[i, 0].xaxis.set_visible(False)
            
    for i, result in enumerate(results[methods[0]]):
        if i == len(result) - 1:
            axes[i, 1].boxplot([results[method][i]['time'] for method in methods], labels=methods)
        else:
            axes[i, 1].boxplot([results[method][i]['time'] for method in methods],
                               labels=["" for j in range(len(results[methods[0]]))])
        axes[i, 1].set_title("Time for alpha={}, t={}".format(result['param'], 1/result['param']))
        axes[i, 1].set_ylabel("ms")
        
    for i, result in enumerate(results["Heat_equation"]):
        axes[i, 0].legend()
        axes[i, 0].set_ylim(0, max_conductance)
        
    plt.suptitle(args.json_file.replace(".json", "").\
        replace("Hypergraph_clustering_based_on_PageRank/output/output_conductances_",""))
    fig.set_size_inches(15, 15)
    plt.subplots_adjust()
    plt.show()
