import numpy as np
from collections import deque

from .f18_env_base import F18EnvBase
from .reward import reward_drive_to_trim_v2
from .termination import check_termination_drive_to_trim_v2


class F18EnvDriveToTrim(F18EnvBase):
    def __init__(self, dt, task, seed=None, log_dir=None):
        super().__init__(dt, task, seed=seed, log_dir=log_dir)
        # Action scale maps normalized [-1,1] into conservative physical deltas.
        self._action_scale = np.array([0.05, 0.05, 0.05, 2000.0], dtype=float)
        weight_map = {
            0: 2.0,   # V
            1: 3.0,   # beta
            2: 3.0,   # alpha
            3: 1.0,   # p
            4: 1.5,   # q
            5: 1.0,   # r
            6: 1.0,   # phi
            7: 8.0,   # theta
            11: 2.0,  # h
        }
        if self._task.obs_mode == "inner10_vh":
            base_idx = [0, 1, 2, 3, 4, 5, 6, 7]
            base_w = [weight_map.get(i, 1.0) for i in base_idx]
            self._w = np.array(base_w + [2.0, 2.0], dtype=float)
        elif self._task.obs_mode == "inner11_vhE":
            base_idx = [0, 1, 2, 3, 4, 5, 6, 7]
            base_w = [weight_map.get(i, 1.0) for i in base_idx]
            self._w = np.array(base_w + [2.0, 2.0, 1.0], dtype=float)
        else:
            self._w = np.array([weight_map.get(i, 1.0) for i in self._task.obs_indices], dtype=float)
        self._lambda_u = 0.1
        self._step_penalty = 0.0
        self._terminal_bonus = 0.0
        self._terminal_penalty = 0.0
        self._k_prog = 0.5
        self._lambda_trend = 0.25
        self._trend_clip = 50.0
        self._lambda_du = 0.1
        self._k_hold = 0.2
        self._lambda_theta = 5.0
        self._lambda_V = 0.02
        self._lambda_h = 1e-6
        self._lambda_barrier = 1.0
        self._barrier_margin = 0.7
        self._barrier_shape_gain = 3.0
        self._barrier_shape_p = 4
        self._barrier_k = {
            "theta": 10.0,
            "alpha": 1.0,
            "beta": 1.0,
            "V": 0.5,
            "h": 0.25,
        }
        self._k_drift = 0.05
        self._drift_on = True
        self._drift_ramp_s = 10.0
        self._lambda_sat = 0.0
        self._sat_margin = 0.02
        self._success_counter = 0
        hold_s = 5.0 if self._task.max_steps <= 3000 else 10.0
        self._success_count_needed = max(75, min(1500, int(round(hold_s / self._dt))))
        self._persist_needed = 5
        self._bounds = {
            "v_min": 1.0,
            "v_min_frac": 0.60,
            "v_min_abs": 0.0,
            "v_band": 200.0,
            "alpha_max": 0.7,
            "beta_max": 0.7,
            "h_min": 0.0,
            "h_max": 60000.0,
            "h_band": 3000.0,
            "theta_band": 0.35,
            "terminate_on_success": False,
            "tol": {
                "alpha": 0.015,
                "beta": 0.015,
                "pqr": 0.03,
                "phi_theta": 0.03,
                "V": 150.0,
                "h": 800.0,
            },
        }
        self._rate_limits = np.array(
            [1.7453292519943295, 1.064650843716541, 0.69813170079773179, np.inf],
            dtype=float,
        )
        self._u_prev = self._task.u_trim.copy()
        self._err_prev_cost = 0.0
        self._persist = {"v_min": 0, "alpha": 0, "beta": 0}
        self._verbose_reset = False
        self._in_tol_count = 0
        self._consecutive_in_tol = 0
        self._max_consecutive_in_tol = 0
        self._first_entry_step = -1
        self._behavioral_time_frac = 0.5
        self._behavioral_consec_frac = 0.25
        self._final_window_steps = max(1, int(round(10.0 / self._dt)))
        self._in_tol_window = deque(maxlen=self._final_window_steps)
        self._max_abs_h_err = 0.0
        self._sum_abs_h_err = 0.0
        self._u_min = np.array(
            [-0.43633231299858238, -0.52359877559829882, -0.41887902047863906, 0.0],
            dtype=float,
        )
        self._u_max = np.array(
            [0.43633231299858238, 0.52359877559829882, 0.18325957145940461, 20000.0],
            dtype=float,
        )

    def reset(self, seed=None, options=None):
        ret = super().reset(seed=seed, options=options)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info0 = ret
        else:
            obs, info0 = ret, {}
        self._u_prev = self._task.u_trim.copy()
        self._err_prev_cost = float(np.sum(self._w * np.asarray(obs, dtype=float) ** 2))
        self._persist = {"v_min": 0, "alpha": 0, "beta": 0}
        self._max_abs_alpha = 0.0
        self._max_abs_beta = 0.0
        self._min_v = float("inf")
        self._in_tol_count = 0
        self._consecutive_in_tol = 0
        self._max_consecutive_in_tol = 0
        self._first_entry_step = -1
        self._in_tol_window = deque(maxlen=self._final_window_steps)
        self._max_abs_h_err = 0.0
        self._sum_abs_h_err = 0.0
        self._bounds["v_min"] = max(self._bounds["v_min_abs"], self._bounds["v_min_frac"] * self._task.x_trim[0])
        hold_s = 5.0 if self._task.max_steps <= 3000 else 10.0
        self._success_count_needed = max(75, min(1500, int(round(hold_s / self._dt))))
        info0["v_min"] = self._bounds["v_min"]
        info0["success_count_needed"] = self._success_count_needed
        if self._verbose_reset:
            print(
                "drive_to_trim bounds:"
                f" dt={self._dt}"
                f" max_steps={self._task.max_steps}"
                f" Vtrim={self._task.x_trim[0]:.1f}"
                f" htrim={self._task.x_trim[11]:.1f}"
                f" v_min={self._bounds['v_min']:.1f}"
                f" v_band={self._bounds['v_band']}"
                f" h_band={self._bounds['h_band']}"
                f" theta_band={self._bounds['theta_band']}"
                f" tolV={self._bounds['tol']['V']}"
                f" tolH={self._bounds['tol']['h']}"
                f" hold_steps={self._success_count_needed}"
                f" terminate_on_success={self._bounds['terminate_on_success']}"
            )
        return obs, info0

    def step(self, action):
        self._k += 1
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)
        u = self._task.u_trim + action * self._action_scale
        u = self._apply_disturbance(u)
        u[3] = np.clip(u[3], 0.0, 20000.0)

        self._sim.step(u)
        self._t = self._sim.get_time()
        x = self.get_state12()
        obs = self._observe(x)
        err = obs
        err_cost = float(np.sum(self._w * err * err))
        dx = x - self._task.x_trim
        V_err = dx[0]
        theta_err = dx[7]
        h_err = dx[11]
        tol = self._bounds["tol"]
        in_tol = (
            abs(dx[2]) < tol["alpha"]
            and abs(dx[1]) < tol["beta"]
            and abs(dx[3]) < tol["pqr"]
            and abs(dx[4]) < tol["pqr"]
            and abs(dx[5]) < tol["pqr"]
            and abs(dx[6]) < tol["phi_theta"]
            and abs(dx[7]) < tol["phi_theta"]
            and abs(dx[0]) < tol["V"]
            and abs(dx[11]) < tol["h"]
        )
        if in_tol:
            self._in_tol_count += 1
            self._consecutive_in_tol += 1
            if self._first_entry_step < 0:
                self._first_entry_step = self._k
        else:
            self._consecutive_in_tol = 0
        self._max_consecutive_in_tol = max(self._max_consecutive_in_tol, self._consecutive_in_tol)
        self._in_tol_window.append(1 if in_tol else 0)
        steps_run = self._k
        time_in_tol_frac = self._in_tol_count / max(1, steps_run)
        window_frac = (sum(self._in_tol_window) / len(self._in_tol_window)) if self._in_tol_window else 0.0
        final_window_good = len(self._in_tol_window) == self._final_window_steps and window_frac > 0.9
        hold_success = self._max_consecutive_in_tol >= self._success_count_needed
        behavioral_success = (
            hold_success
            or (
                time_in_tol_frac >= self._behavioral_time_frac
                and self._max_consecutive_in_tol >= int(round(self._behavioral_consec_frac * self._success_count_needed))
            )
        )

        terminated, truncated, success, reason, self._success_counter, self._persist = (
            check_termination_drive_to_trim_v2(
                x,
                self._task.x_trim,
                self._persist,
                self._success_counter,
                self._success_count_needed,
                self._bounds,
                self._k,
                self._task.max_steps,
                self._persist_needed,
                terminate_on_success=self._bounds["terminate_on_success"],
            )
        )
        unsafe_reasons = {
            "h_band",
            "v_band",
            "theta_band",
            "ground",
            "h_max",
            "v_min_persist",
            "alpha_persist",
            "beta_persist",
        }
        unsafe_termination = terminated and reason in unsafe_reasons
        true_success = (not unsafe_termination) and (final_window_good or hold_success)
        success_reason = "unsafe" if unsafe_termination else ("hold" if hold_success else ("final_window" if final_window_good else "none"))
        success = true_success
        if success and terminated and reason not in unsafe_reasons:
            reason = "success"
        def _barrier(margin, k):
            if margin <= self._barrier_margin:
                return 0.0
            excess = margin - self._barrier_margin
            return k * excess * excess * (1.0 + self._barrier_shape_gain * (margin ** self._barrier_shape_p))

        theta_margin = abs(dx[7]) / max(self._bounds["theta_band"], 1e-9)
        alpha_margin = abs(dx[2]) / max(self._bounds["alpha_max"], 1e-9)
        beta_margin = abs(dx[1]) / max(self._bounds["beta_max"], 1e-9)
        v_margin = abs(dx[0]) / max(self._bounds["v_band"], 1e-9)
        h_margin = abs(dx[11]) / max(self._bounds["h_band"], 1e-9)
        barrier_cost = (
            _barrier(theta_margin, self._barrier_k["theta"])
            + _barrier(alpha_margin, self._barrier_k["alpha"])
            + _barrier(beta_margin, self._barrier_k["beta"])
            + _barrier(v_margin, self._barrier_k["V"])
            + _barrier(h_margin, self._barrier_k["h"])
        )
        if self._drift_on and self._first_entry_step >= 0:
            steps_since_entry = self._k - self._first_entry_step
            ramp_steps = max(1, int(round(self._drift_ramp_s / self._dt)))
            drift_ramp = min(1.0, max(0.0, steps_since_entry / float(ramp_steps)))
            drift_cost = self._k_drift * drift_ramp * err_cost
        else:
            drift_ramp = 0.0
            drift_cost = 0.0

        reward, u_cost, slew_cost, prog, hold_bonus, sat_cost, theta_cost, V_cost, h_cost, trend_reward = reward_drive_to_trim_v2(
            err_cost,
            self._err_prev_cost,
            u,
            self._task.u_trim,
            self._action_scale,
            self._u_prev,
            self._dt,
            self._rate_limits,
            self._lambda_u,
            self._k_prog,
            self._lambda_du,
            self._k_hold,
            in_tol,
            lambda_trend=self._lambda_trend,
            trend_clip=self._trend_clip,
            barrier_cost=barrier_cost,
            lambda_barrier=self._lambda_barrier,
            drift_cost=drift_cost,
            lambda_sat=self._lambda_sat,
            sat_margin=self._sat_margin,
            u_min=self._u_min,
            u_max=self._u_max,
            theta_err=theta_err,
            V_err=V_err,
            h_err=h_err,
            lambda_theta=self._lambda_theta,
            lambda_V=self._lambda_V,
            lambda_h=self._lambda_h,
            step_penalty=self._step_penalty,
            terminal_bonus=self._terminal_bonus,
            terminal_penalty=self._terminal_penalty,
            terminated=terminated,
            success=success,
        )
        self._max_abs_alpha = max(self._max_abs_alpha, abs(x[2] - self._task.x_trim[2]))
        self._max_abs_beta = max(self._max_abs_beta, abs(x[1] - self._task.x_trim[1]))
        self._min_v = min(self._min_v, x[0])
        self._max_abs_h_err = max(self._max_abs_h_err, abs(h_err))
        self._sum_abs_h_err += abs(h_err)
        mean_abs_h_err = self._sum_abs_h_err / max(1, steps_run)

        info = {
            "t": self._t,
            "success": success,
            "reason": reason,
            "u": u,
            "in_tol": in_tol,
            "err_cost": err_cost,
            "u_cost": u_cost,
            "slew_cost": slew_cost,
            "prog": prog,
            "trend_reward": trend_reward,
            "hold_bonus": hold_bonus,
            "sat_cost": sat_cost,
            "theta_cost": theta_cost,
            "V_cost": V_cost,
            "h_cost": h_cost,
            "barrier_cost": barrier_cost,
            "drift_cost": drift_cost,
            "drift_ramp": drift_ramp,
            "time_in_tol_frac_so_far": time_in_tol_frac,
            "time_in_tol_frac": time_in_tol_frac,
            "max_consecutive_in_tol": self._max_consecutive_in_tol,
            "max_consecutive_in_tol_s": self._max_consecutive_in_tol * self._dt,
            "first_entry_step": self._first_entry_step,
            "behavioral_success": behavioral_success,
            "hold_success": hold_success,
            "final_window_good": final_window_good,
            "final_window_frac": window_frac,
            "true_success": true_success,
            "success_reason": success_reason,
            "hold_success_s_threshold": self._success_count_needed * self._dt,
            "max_abs_alpha": self._max_abs_alpha,
            "max_abs_beta": self._max_abs_beta,
            "min_v": self._min_v,
            "v_min": self._bounds["v_min"],
            "max_abs_h_err": self._max_abs_h_err,
            "mean_abs_h_err": mean_abs_h_err,
        }

        self._u_prev = u
        self._err_prev_cost = err_cost

        return (obs, reward, terminated, truncated, info)
