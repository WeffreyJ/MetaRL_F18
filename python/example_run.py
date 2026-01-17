from f18sim import F18Sim


def main():
    sim = F18Sim()

    d2r = 3.141592653589793 / 180.0
    h0 = 25000.0
    init_dramatic = [
        350.0,         # V
        20.0 * d2r,    # beta
        40.0 * d2r,    # alpha
        10.0 * d2r,    # p
        0.0 * d2r,     # q
        5.0 * d2r,     # r
        0.0 * d2r,     # phi
        0.0 * d2r,     # theta
        0.0 * d2r,     # psi
        0.0,           # pN
        0.0,           # pE
        h0,            # h
    ]

    sim.reset(init_dramatic)

    steps = 50
    for _ in range(steps):
        sim.step()
        print(f"t={sim.get_time():.2f}, state[0:3]={sim.get_state()[0:3]}")

    sim.close()


if __name__ == "__main__":
    main()
