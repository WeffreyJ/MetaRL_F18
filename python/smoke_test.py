from f18sim import F18Sim


def nearly_equal(a, b, eps=1e-12):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > eps:
            return False
    return True


def run_once(sim, x0, steps):
    sim.reset(x0)
    for _ in range(steps):
        sim.step()
    return sim.get_time(), sim.get_state()


def main():
    d2r = 3.141592653589793 / 180.0
    h0 = 25000.0
    x0 = [
        350.0,
        20.0 * d2r,
        40.0 * d2r,
        10.0 * d2r,
        0.0 * d2r,
        5.0 * d2r,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        h0,
    ]

    sim = F18Sim()

    t1, x1 = run_once(sim, x0, 10)
    t2, x2 = run_once(sim, x0, 10)

    print("t1:", t1)
    print("t2:", t2)
    print("x1[0:3]:", x1[0:3])
    print("x2[0:3]:", x2[0:3])
    print("deterministic:", nearly_equal(x1, x2))

    sim.close()


if __name__ == "__main__":
    main()
