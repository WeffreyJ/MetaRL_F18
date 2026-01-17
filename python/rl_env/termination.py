import numpy as np


def check_termination_drive_to_trim(state, success_counter, success_count_needed, bounds, step, max_steps):
    state = np.asarray(state, dtype=float)
    if not np.isfinite(state).all():
        return True, False, False, "nan_inf", success_counter

    v = state[0]
    beta = state[1]
    alpha = state[2]
    p = state[3]
    q = state[4]
    r = state[5]
    phi = state[6]
    theta = state[7]
    h = state[11]

    if v <= bounds["v_min"]:
        return True, False, False, "v_min", success_counter
    if abs(alpha) > bounds["alpha_max"]:
        return True, False, False, "alpha", success_counter
    if abs(beta) > bounds["beta_max"]:
        return True, False, False, "beta", success_counter
    if h < bounds["h_min"] or h > bounds["h_max"]:
        return True, False, False, "h", success_counter

    tol = bounds["tol"]
    if (
        abs(alpha) < tol["alpha"]
        and abs(beta) < tol["beta"]
        and abs(p) < tol["pqr"]
        and abs(q) < tol["pqr"]
        and abs(r) < tol["pqr"]
        and abs(phi) < tol["phi_theta"]
        and abs(theta) < tol["phi_theta"]
    ):
        success_counter += 1
    else:
        success_counter = 0

    if success_counter >= success_count_needed:
        return True, False, True, "success", success_counter

    if step >= max_steps:
        return False, True, False, "max_steps", success_counter

    return False, False, False, "", success_counter


def check_termination_drive_to_trim_v2(
    state,
    x_trim,
    persist,
    success_counter,
    success_count_needed,
    bounds,
    step,
    max_steps,
    persist_needed,
    terminate_on_success=True,
):
    state = np.asarray(state, dtype=float)
    x_trim = np.asarray(x_trim, dtype=float)
    if not np.isfinite(state).all():
        return True, False, False, "nan_inf", success_counter, persist

    v = state[0]
    beta = state[1] - x_trim[1]
    alpha = state[2] - x_trim[2]
    p = state[3] - x_trim[3]
    q = state[4] - x_trim[4]
    r = state[5] - x_trim[5]
    phi = state[6] - x_trim[6]
    theta = state[7] - x_trim[7]
    h = state[11]
    dv = v - x_trim[0]
    dh = h - x_trim[11]

    h_min = bounds.get("h_min", -np.inf)
    h_max = bounds.get("h_max", np.inf)
    if h <= h_min:
        return True, False, False, "ground", success_counter, persist
    if h >= h_max:
        return True, False, False, "h_max", success_counter, persist

    h_band = bounds.get("h_band", np.inf)
    v_band = bounds.get("v_band", np.inf)
    theta_band = bounds.get("theta_band", np.inf)
    if abs(dh) > h_band:
        return True, False, False, "h_band", success_counter, persist
    if abs(dv) > v_band:
        return True, False, False, "v_band", success_counter, persist
    if abs(theta) > theta_band:
        return True, False, False, "theta_band", success_counter, persist

    persist["v_min"] = persist["v_min"] + 1 if v <= bounds["v_min"] else 0
    persist["alpha"] = persist["alpha"] + 1 if abs(alpha) > bounds["alpha_max"] else 0
    persist["beta"] = persist["beta"] + 1 if abs(beta) > bounds["beta_max"] else 0

    if persist["v_min"] >= persist_needed:
        return True, False, False, "v_min_persist", success_counter, persist
    if persist["alpha"] >= persist_needed:
        return True, False, False, "alpha_persist", success_counter, persist
    if persist["beta"] >= persist_needed:
        return True, False, False, "beta_persist", success_counter, persist

    tol = bounds.get("tol", {})
    v_tol = tol.get("V", np.inf)
    h_tol = tol.get("h", np.inf)
    if (
        abs(alpha) < tol["alpha"]
        and abs(beta) < tol["beta"]
        and abs(p) < tol["pqr"]
        and abs(q) < tol["pqr"]
        and abs(r) < tol["pqr"]
        and abs(phi) < tol["phi_theta"]
        and abs(theta) < tol["phi_theta"]
        and abs(dv) < v_tol
        and abs(dh) < h_tol
    ):
        success_counter += 1
    else:
        success_counter = 0

    if success_counter >= success_count_needed:
        if terminate_on_success:
            return True, False, True, "success", success_counter, persist
        return False, False, True, "success", success_counter, persist

    if step >= max_steps:
        return False, True, False, "max_steps", success_counter, persist

    return False, False, False, "", success_counter, persist
