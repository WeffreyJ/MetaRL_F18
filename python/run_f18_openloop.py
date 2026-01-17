import argparse
import csv
import sys

from f18sim import F18Sim

try:
    from regimes import REGIMES, stack_x
except Exception:
    REGIMES = {}
    stack_x = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run F-18 open-loop sim.")
    parser.add_argument("--dt", type=float, default=0.02, help="Step size (s)")
    parser.add_argument("--sim_time", type=float, default=200.0, help="Sim time (s)")
    parser.add_argument("--out", default="state_log.csv", help="CSV output path")
    parser.add_argument("--regime", default=None, help="Regime name in regimes.py")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat runs for determinism check")
    return parser.parse_args()


def build_init_from_vars():
    d2r = 3.141592653589793 / 180.0
    r2d = 1.0 / d2r
    _ = r2d

    # Editable initial conditions (MATLAB-like block)
    V = 350.0
    beta = 20.0 * d2r
    alpha = 40.0 * d2r
    p = 10.0 * d2r
    q = 0.0 * d2r
    r = 5.0 * d2r
    phi = 0.0 * d2r
    theta = 0.0 * d2r
    psi = 0.0 * d2r
    pN = 0.0
    pE = 0.0
    h = 25000.0

    return [V, beta, alpha, p, q, r, phi, theta, psi, pN, pE, h]


def build_action():
    # Control inputs (MATLAB-style)
    Con_ail = 0.0
    Con_rud = 0.0
    Con_elev = -0.022
    Con_T = 5470.5
    # Action order: [ail, rud, elev, T]
    return [Con_ail, Con_rud, Con_elev, Con_T]


def run_once(sim, x_init, u, steps, out_path=None):
    sim.reset(x_init)

    if out_path:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t", "V", "beta", "alpha", "p", "q", "r", "phi", "theta", "psi", "pN", "pE", "h"]
            )
            state = sim.get_state()
            writer.writerow([sim.get_time()] + state)
            for _ in range(steps):
                sim.step(u)
                state = sim.get_state()
                writer.writerow([sim.get_time()] + state)
    else:
        for _ in range(steps):
            sim.step(u)

    return sim.get_time(), sim.get_state()


def nearly_equal(a, b, eps=1e-12):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > eps:
            return False
    return True


def main():
    args = parse_args()

    if args.dt <= 0.0:
        print("dt must be positive.", file=sys.stderr)
        return 1
    if args.sim_time <= 0.0:
        print("sim_time must be positive.", file=sys.stderr)
        return 1

    x_init = build_init_from_vars()
    u = build_action()
    if args.regime:
        if not REGIMES or args.regime not in REGIMES:
            print(f"Unknown regime: {args.regime}", file=sys.stderr)
            return 1
        x_init = stack_x(REGIMES[args.regime])

    sim = F18Sim()
    sim.set_step_size(args.dt)
    steps = int(round(args.sim_time / args.dt))

    t1, x1 = run_once(sim, x_init, u, steps, out_path=args.out)
    deterministic = True

    if args.repeat > 1:
        for _ in range(args.repeat - 1):
            t2, x2 = run_once(sim, x_init, u, steps, out_path=None)
            deterministic = deterministic and nearly_equal(x1, x2)

    sim.close()

    print("Final time:", t1)
    print("Final state[0:3]:", x1[0:3])
    print("CSV:", args.out)
    if args.repeat > 1:
        print("Deterministic:", deterministic)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
