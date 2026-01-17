import numpy as np
from scipy.linalg import solve_continuous_are

from linmodel_f18 import A, B


X8_IDX = [0, 1, 2, 3, 4, 5, 6, 7]
A8 = A[np.ix_(X8_IDX, X8_IDX)]
B8 = B[X8_IDX, :]

Q = np.diag([1, 10, 10, 1, 1, 1, 10, 10])
R = np.diag([100, 100, 100, 0.01])


P = solve_continuous_are(A8, B8, Q, R)
K = np.linalg.solve(R, B8.T @ P)

print("Closed-loop eigenvalues:", np.linalg.eigvals(A8 - B8 @ K))
