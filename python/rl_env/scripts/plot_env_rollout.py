import argparse
import csv
from pathlib import Path

import numpy as np


STATE_NAMES = ["V", "beta", "alpha", "p", "q", "r", "phi", "theta", "psi", "pN", "pE", "h"]
ACTION_NAMES = ["u_ail", "u_rud", "u_elev", "u_T"]


def parse_args():
    p = argparse.ArgumentParser(description="Plot rollout CSV.")
    p.add_argument("csv", help="Path to CSV")
    return p.parse_args()


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("pip3 install matplotlib")
        return 1

    header, rows = load_csv(args.csv)
    if not rows:
        print("No data found in CSV.")
        return 1

    if "t" not in header:
        print("Missing column: t")
        return 1
    t = [float(r[header.index("t")]) for r in rows]

    for name in STATE_NAMES + ACTION_NAMES + ["reward"]:
        if name not in header:
            print(f"Missing column: {name}")
            return 1

    csv_path = Path(args.csv)
    stem = csv_path.stem

    fig, axes = plt.subplots(6, 2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(STATE_NAMES):
        ax = axes[i]
        series = [float(r[header.index(name)]) for r in rows]
        ax.plot(t, series)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(csv_path.with_name(f"{stem}_states.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(ACTION_NAMES):
        ax = axes[i]
        series = [float(r[header.index(name)]) for r in rows]
        ax.plot(t, series)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(csv_path.with_name(f"{stem}_actions.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, [float(r[header.index("reward")]) for r in rows])
    plt.title("reward")
    plt.xlabel("t (s)")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(csv_path.with_name(f"{stem}_reward.png"), dpi=150)
    plt.close(fig)

    if len(t) > 1:
        dt = np.diff(t)
        du = []
        for name in ACTION_NAMES:
            u = np.array([float(r[header.index(name)]) for r in rows], dtype=float)
            du.append(np.diff(u) / dt)
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
        axes = axes.flatten()
        for i, name in enumerate(ACTION_NAMES):
            ax = axes[i]
            ax.plot(t[1:], du[i])
            ax.set_title(f"d{name}/dt")
            ax.grid(True, alpha=0.3)
        for ax in axes:
            ax.set_xlabel("t (s)")
        fig.tight_layout()
        fig.savefig(csv_path.with_name(f"{stem}_rates.png"), dpi=150)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
