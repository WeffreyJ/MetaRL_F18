import argparse
import csv
import sys

import numpy as np

from f18sim import F18Sim
from controller_lqr import lqr_controller
from linmodel_f18 import x_trim, u_trim


def parse_args():
    p = argparse.ArgumentParser(description="Run F-18 closed-loop LQR sim.")
    p.add_argument("--sim_time", type=float, default=30.0, help="Sim time (s)")
    p.add_argument("--dt", type=float, default=0.02, help="Step size (s)")
    p.add_argument("--out", default="lqr_log.csv", help="CSV output path")
    p.add_argument("--repeat", type=int, default=1, help="Repeat runs for determinism check")
    return p.parse_args()


def _is_finite(values):
    return np.isfinite(values).all()


def run_once(sim, x_init, steps, out_path=None):
    sim.reset(x_init)
    if out_path:
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "t",
                    "V",
                    "beta",
                    "alpha",
                    "p",
                    "q",
                    "r",
                    "phi",
                    "theta",
                    "psi",
                    "pN",
                    "pE",
                    "h",
                    "u_ail",
                    "u_rud",
                    "u_elev",
                    "u_T",
                ]
            )
            u0 = u_trim
            w.writerow([sim.get_time()] + sim.get_state() + list(u0))
            for _ in range(steps):
                x = sim.get_state()
                u = lqr_controller(x)
                sim.step(u)
                state = sim.get_state()
                if not _is_finite(state):
                    print(f"TERMINATED: NaN/Inf at t={sim.get_time()}, state={state}")
                    return sim.get_time(), state
                w.writerow([sim.get_time()] + state + list(u))
    else:
        for _ in range(steps):
            x = sim.get_state()
            u = lqr_controller(x)
            sim.step(u)
            state = sim.get_state()
            if not _is_finite(state):
                print(f"TERMINATED: NaN/Inf at t={sim.get_time()}, state={state}")
                return sim.get_time(), state

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
    if args.dt <= 0.0 or args.sim_time <= 0.0:
        print("dt and sim_time must be positive.", file=sys.stderr)
        return 1

    steps = int(round(args.sim_time / args.dt))
    sim = F18Sim()
    sim.set_step_size(args.dt)
    if sim.get_num_states() != len(x_trim):
        print(
            f"State size mismatch: sim={sim.get_num_states()} x_trim={len(x_trim)}",
            file=sys.stderr,
        )
        sim.close()
        return 1

    t1, x1 = run_once(sim, x_trim, steps, out_path=args.out)
    deterministic = True

    if args.repeat > 1:
        for _ in range(args.repeat - 1):
            t2, x2 = run_once(sim, x_trim, steps, out_path=None)
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
