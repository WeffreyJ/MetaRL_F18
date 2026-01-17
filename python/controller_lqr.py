import numpy as np

from linmodel_f18 import x_trim, u_trim
from lqr_design import K


X8_IDX = [0, 1, 2, 3, 4, 5, 6, 7]
X8_TRIM = x_trim[X8_IDX]

print("LQR K shape:", K.shape)
print("dx shape:", X8_TRIM.shape)
print("Action order: [ail, rud, elev, T]")


def _map_x8(x12):
    x12 = np.asarray(x12, dtype=float)
    return x12[X8_IDX]


def lqr_controller(x12):
    x8 = _map_x8(x12)
    dx = x8 - X8_TRIM
    du = -K @ dx
    u = u_trim + du
    return u
