import argparse
import csv
import sys

from f18sim import F18Sim

try:
    from regimes import REGIMES, stack_x, stack_u
except Exception:
    REGIMES = {}
    stack_x = None
    stack_u = None


# ==========================
# EDIT THIS BLOCK ONLY
dt = 0.02
sim_time = 100.0
out_csv = "state_log.csv"

# Initial condition (12)
x0 = [
    350.0,
    20.0 * 3.141592653589793 / 180.0,
    40.0 * 3.141592653589793 / 180.0,
    10.0 * 3.141592653589793 / 180.0,
    0.0,
    5.0 * 3.141592653589793 / 180.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    25000.0,
]

# Action order must match wrapper: [ail, rud, elev, T]
u = [0.0, 0.0, -0.022, 5470.5]
# ==========================


def parse_args():
    parser = argparse.ArgumentParser(description="Run F-18 experiment.")
    parser.add_argument("--regime", default=None, help="Regime name in regimes.py")
    parser.add_argument("--out", default=None, help="Override output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    if dt <= 0.0 or sim_time <= 0.0:
        print("dt and sim_time must be positive.", file=sys.stderr)
        return 1

    x_init = x0
    u_init = u
    if args.regime:
        if not REGIMES or args.regime not in REGIMES:
            print(f"Unknown regime: {args.regime}", file=sys.stderr)
            return 1
        x_init = stack_x(REGIMES[args.regime])
        u_init = stack_u(REGIMES[args.regime])

    out_path = args.out if args.out else out_csv
    steps = int(round(sim_time / dt))

    sim = F18Sim()
    sim.set_step_size(dt)
    sim.reset(x_init)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "V", "beta", "alpha", "p", "q", "r", "phi", "theta", "psi", "pN", "pE", "h"])
        w.writerow([sim.get_time()] + sim.get_state())
        for _ in range(steps):
            sim.step(u_init)
            w.writerow([sim.get_time()] + sim.get_state())

    sim.close()
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
