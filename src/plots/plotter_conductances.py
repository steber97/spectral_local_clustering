import pandas as pd
import argparse
import matplotlib.pyplot as plt

args = argparse.ArgumentParser("Plot conductances")
args.add_argument("--csv_file", type=str)

args = args.parse_args()


if __name__ == "__main__":
    df = pd.read_csv(args.csv_file)
    fig, axes = plt.subplots(1, 2)
    methods = sorted(list(set([label.replace("_conductance", "").replace("_time", "") for label in df.columns])))
    running_time_per_method = {}
    for label in df.columns:
        if "_conductance" in label:
            x_s = []
            y_s = []
            conductances_sorted = sorted(df[label])
            x_s.append(0)
            y_s.append(conductances_sorted[0])
            for i in range(1, len(conductances_sorted)):
                x_s.append(i)
                y_s.append(conductances_sorted[i-1])
                x_s.append(i)
                y_s.append(conductances_sorted[i])
                
            axes[0].plot(x_s, y_s, label=label.replace("_conductance", ""))
        elif "_time" in label:
            method = label.split("_time")[0]
            running_time_per_method[method] = df[label]
    axes[1].boxplot([running_time_per_method[method] for method in methods], 
                    labels=methods)
    plt.xticks(rotation=45)

    axes[0].set_ylim(0)
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(args.csv_file.replace(".csv", "").\
        replace("Hypergraph_clustering_based_on_PageRank/output/output_conductances_",""))
    plt.subplots_adjust()
    plt.show()
