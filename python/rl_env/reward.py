import numpy as np


def reward_drive_to_trim(
    err,
    u,
    u_trim,
    action_scale,
    w,
    lambda_u,
    step_penalty=0.0,
    terminal_bonus=0.0,
    terminal_penalty=0.0,
    terminated=False,
    success=False,
):
    err = np.asarray(err, dtype=float)
    u = np.asarray(u, dtype=float)
    u_trim = np.asarray(u_trim, dtype=float)
    action_scale = np.asarray(action_scale, dtype=float)
    w = np.asarray(w, dtype=float)
    err_cost = np.sum(w * err * err)
    du_norm = (u - u_trim) / action_scale
    u_cost = lambda_u * np.sum(du_norm * du_norm)
    reward = -(err_cost + u_cost) - step_penalty
    if terminated and success:
        reward += terminal_bonus
    elif terminated and not success:
        reward -= terminal_penalty
    return reward


def reward_drive_to_trim_v2(
    err_cost,
    err_cost_prev,
    u,
    u_trim,
    action_scale,
    u_prev,
    dt,
    rate_limits,
    lambda_u,
    k_prog,
    lambda_du,
    k_hold,
    in_tol,
    lambda_trend=0.0,
    trend_clip=None,
    barrier_cost=0.0,
    lambda_barrier=1.0,
    drift_cost=0.0,
    lambda_sat=0.0,
    sat_margin=0.02,
    u_min=None,
    u_max=None,
    theta_err=None,
    V_err=None,
    h_err=None,
    lambda_theta=0.0,
    lambda_V=0.0,
    lambda_h=0.0,
    step_penalty=0.0,
    terminal_bonus=0.0,
    terminal_penalty=0.0,
    terminated=False,
    success=False,
):
    u = np.asarray(u, dtype=float)
    u_trim = np.asarray(u_trim, dtype=float)
    action_scale = np.asarray(action_scale, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)
    rate_limits = np.asarray(rate_limits, dtype=float)

    du_norm = (u - u_trim) / action_scale
    u_cost = lambda_u * np.sum(du_norm * du_norm)

    du_dt = (u - u_prev) / max(dt, 1e-9)
    slew_cost = 0.0
    for i in range(len(du_dt)):
        if np.isfinite(rate_limits[i]) and rate_limits[i] > 0.0:
            slew_cost += (du_dt[i] / rate_limits[i]) ** 2
    slew_cost *= lambda_du

    prog = 0.0
    trend_reward = 0.0
    if err_cost_prev is not None and lambda_trend != 0.0:
        trend = err_cost_prev - err_cost
        if trend_clip is not None:
            trend = float(np.clip(trend, -trend_clip, trend_clip))
        trend_reward = lambda_trend * trend
    hold_bonus = k_hold if in_tol else 0.0

    sat_cost = 0.0
    if lambda_sat > 0.0 and u_min is not None and u_max is not None:
        u_min = np.asarray(u_min, dtype=float)
        u_max = np.asarray(u_max, dtype=float)
        span = np.maximum(u_max - u_min, 1e-9)
        margin = sat_margin * span
        near_min = u <= (u_min + margin)
        near_max = u >= (u_max - margin)
        sat_cost = lambda_sat * float(np.sum(near_min | near_max))

    theta_cost = 0.0
    if theta_err is not None and lambda_theta != 0.0:
        theta_cost = float(lambda_theta * (float(theta_err) ** 2))
    V_cost = 0.0
    if V_err is not None and lambda_V != 0.0:
        V_cost = float(lambda_V * (float(V_err) ** 2))
    h_cost = 0.0
    if h_err is not None and lambda_h != 0.0:
        h_cost = float(lambda_h * (float(h_err) ** 2))

    reward = (
        -(err_cost + u_cost + slew_cost + theta_cost + V_cost + h_cost)
        + prog
        + trend_reward
        + hold_bonus
        - step_penalty
        - sat_cost
        - (lambda_barrier * barrier_cost)
        - drift_cost
    )
    if terminated and success:
        reward += terminal_bonus
    elif terminated and not success:
        reward -= terminal_penalty

    return reward, u_cost, slew_cost, prog, hold_bonus, sat_cost, theta_cost, V_cost, h_cost, trend_reward


def reward_residual(err, u, u_nom, delta_u_max, w, lambda_u, step_penalty=0.0, terminal_bonus=0.0, terminal_penalty=0.0, terminated=False, success=False):
    err = np.asarray(err, dtype=float)
    u = np.asarray(u, dtype=float)
    u_nom = np.asarray(u_nom, dtype=float)
    delta_u_max = np.asarray(delta_u_max, dtype=float)
    w = np.asarray(w, dtype=float)
    err_cost = np.sum(w * err * err)
    du_norm = (u - u_nom) / delta_u_max
    u_cost = lambda_u * np.sum(du_norm * du_norm)
    reward = -(err_cost + u_cost) - step_penalty
    if terminated and success:
        reward += terminal_bonus
    elif terminated and not success:
        reward -= terminal_penalty
    return reward
