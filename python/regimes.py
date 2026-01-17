def stack_x(reg):
    return [
        reg["V"],
        reg["beta"],
        reg["alpha"],
        reg["p"],
        reg["q"],
        reg["r"],
        reg["phi"],
        reg["theta"],
        reg["psi"],
        reg["pN"],
        reg["pE"],
        reg["h"],
    ]


def stack_u(reg):
    return [
        reg["u"][0],
        reg["u"][1],
        reg["u"][2],
        reg["u"][3],
    ]


def _d2r():
    return 3.141592653589793 / 180.0


REGIMES = {
    "Init_Dramatic": {
        "V": 350.0,
        "beta": 20.0 * _d2r(),
        "alpha": 40.0 * _d2r(),
        "p": 10.0 * _d2r(),
        "q": 0.0 * _d2r(),
        "r": 5.0 * _d2r(),
        "phi": 0.0 * _d2r(),
        "theta": 0.0 * _d2r(),
        "psi": 0.0 * _d2r(),
        "pN": 0.0,
        "pE": 0.0,
        "h": 25000.0,
        "u": [0.0, 0.0, -0.022, 5470.5],
    },
    "TrimGuess": {
        "V": 500.0,
        "beta": 0.0,
        "alpha": 5.0 * _d2r(),
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "phi": 0.0,
        "theta": 0.0,
        "psi": 0.0,
        "pN": 0.0,
        "pE": 0.0,
        "h": 20000.0,
        "u": [0.0, 0.0, -0.02, 6000.0],
    },
}
