import pandas as pd
import argparse
import matplotlib.pyplot as plt

args = argparse.ArgumentParser("Plot conductances")
args.add_argument("--csv_file", type=str)

args = args.parse_args()


if __name__ == "__main__":
    df = pd.read_csv(args.csv_file)
    fig, axes = plt.subplots(1, 2)
    for label in df.columns:
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
            
        axes[0].plot(x_s, y_s, label=label)
    
    # axes[1].boxplot([running_time_per_label[label] for label in labels], labels=labels)

    axes[0].set_ylim(0)
    axes[0].legend()
    # axes[1].legend()
    plt.show()
