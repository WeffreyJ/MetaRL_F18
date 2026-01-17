import argparse
import csv
import sys


STATE_NAMES = ["V", "beta", "alpha", "p", "q", "r", "phi", "theta", "psi", "pN", "pE", "h"]


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, rows


def parse_args():
    p = argparse.ArgumentParser(description="Plot F-18 state_log.csv")
    p.add_argument("csv", nargs="?", default="state_log.csv", help="Path to CSV (default: state_log.csv)")
    p.add_argument("--mode", choices=["all", "quick"], default="all", help="Plot mode (default: all)")
    p.add_argument("--save", default=None, help="If set, save figure to this path instead of only showing")
    return p.parse_args()


def require_columns(header, cols):
    missing = [c for c in cols if c not in header]
    if missing:
        raise KeyError(f"Missing columns in CSV header: {missing}\nFound: {header}")


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("python3 -m pip install matplotlib")
        return 1

    header, rows = load_csv(args.csv)
    if not rows:
        print("No data found in CSV.")
        return 1

    require_columns(header, ["t"])
    if args.mode == "quick":
        require_columns(header, ["V", "alpha", "beta", "p", "q", "r"])
    else:
        require_columns(header, STATE_NAMES)

    data = list(zip(*rows))
    t = data[header.index("t")]

    def series(name):
        return data[header.index(name)]

    if args.mode == "quick":
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes = axes.flatten()

        axes[0].plot(t, series("V"))
        axes[0].set_title("V")

        axes[1].plot(t, series("alpha"))
        axes[1].set_title("alpha")

        axes[2].plot(t, series("beta"))
        axes[2].set_title("beta")

        axes[3].plot(t, series("p"), label="p")
        axes[3].plot(t, series("q"), label="q")
        axes[3].plot(t, series("r"), label="r")
        axes[3].set_title("p/q/r")
        axes[3].legend()

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("t (s)")

        fig.tight_layout()

    else:
        fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        for i, name in enumerate(STATE_NAMES):
            ax = axes[i]
            ax.plot(t, series(name))
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        for ax in axes[-3:]:
            ax.set_xlabel("t (s)")

        fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
