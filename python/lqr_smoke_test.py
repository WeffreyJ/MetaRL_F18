from f18sim import F18Sim
from controller_lqr import lqr_controller
from linmodel_f18 import x_trim


def nearly_equal(a, b, eps=1e-10):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > eps:
            return False
    return True


def is_finite(values):
    for v in values:
        if v != v:
            return False
        if v == float("inf") or v == float("-inf"):
            return False
    return True


def run(sim, steps):
    sim.reset(x_trim)
    for _ in range(steps):
        x = sim.get_state()
        u = lqr_controller(x)
        sim.step(u)
        if not is_finite(sim.get_state()):
            return sim.get_time(), sim.get_state(), False
    return sim.get_time(), sim.get_state(), True


def main():
    dt = 0.02
    sim_time = 5.0
    steps = int(round(sim_time / dt))

    sim = F18Sim()
    sim.set_step_size(dt)

    t1, x1, ok1 = run(sim, steps)
    t2, x2, ok2 = run(sim, steps)

    deterministic = ok1 and ok2 and nearly_equal(x1, x2)
    print("deterministic:", deterministic)
    if not ok1 or not ok2:
        print("NaN/Inf detected.")

    sim.close()


if __name__ == "__main__":
    main()
