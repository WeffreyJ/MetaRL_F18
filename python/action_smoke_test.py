from f18sim import F18Sim


def nearly_equal(a, b, eps=1e-12):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > eps:
            return False
    return True


def run(sim, x0, u, steps):
    sim.reset(x0)
    for _ in range(steps):
        sim.step(u)
    return sim.get_time(), sim.get_state()


def main():
    d2r = 3.141592653589793 / 180.0
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
        25000.0,
    ]

    u1 = [0.0, 0.0, -0.022, 5470.5]
    u2 = [0.0, 0.0, -0.05, 6000.0]

    sim = F18Sim()

    t1, x1 = run(sim, x0, u1, 50)
    t2, x2 = run(sim, x0, u1, 50)
    print("deterministic (u1):", nearly_equal(x1, x2))

    t3, x3 = run(sim, x0, u2, 50)
    different = not nearly_equal(x1, x3, eps=1e-9)
    print("different actions diverge:", different)

    sim.close()


if __name__ == "__main__":
    main()
