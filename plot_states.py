import csv
import sys

import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, rows


def main():
    path = "state_log.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    header, rows = load_csv(path)
    if not rows:
        raise SystemExit("No data found in CSV.")

    data = list(zip(*rows))
    t = data[0]
    series = data[1:]

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(t, series[i])
        ax.set_title(header[i + 1])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("t (s)")
    fig.suptitle("F-18 State Trajectories (Init_Dramatic)", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
