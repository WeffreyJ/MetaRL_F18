import argparse


STATE_NAMES = ["V", "beta", "alpha", "p", "q", "r", "phi", "theta", "psi", "pN", "pE", "h"]


def parse_args():
    p = argparse.ArgumentParser(description="Plot all F-18 states from CSV.")
    p.add_argument("csv", help="Path to CSV")
    return p.parse_args()


def load_csv(path):
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, rows


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

    data = list(zip(*rows))
    try:
        t = data[header.index("t")]
    except ValueError:
        print("Missing column: t")
        return 1

    for name in STATE_NAMES:
        if name not in header:
            print(f"Missing column: {name}")
            return 1

    fig, axes = plt.subplots(6, 2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        series = data[header.index(name)]
        ax.plot(t, series)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel("t (s)")

    fig.tight_layout()
    out_path = f"{args.csv.rsplit('.', 1)[0]}_states.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
