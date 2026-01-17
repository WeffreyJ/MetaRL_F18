import csv


class CSVLogger:
    def __init__(self, path, include_u_nom=False):
        self._path = path
        self._include_u_nom = include_u_nom
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        header = [
            "t",
            "V",
            "beta",
            "alpha",
            "p",
            "q",
            "r",
            "phi",
            "theta",
            "psi",
            "pN",
            "pE",
            "h",
            "u_ail",
            "u_rud",
            "u_elev",
            "u_T",
            "reward",
            "terminated",
            "truncated",
            "task_id",
            "success",
            "reason",
            "in_tol",
            "err_cost",
            "u_cost",
            "slew_cost",
            "prog",
            "hold_bonus",
            "sat_cost",
            "theta_cost",
            "V_cost",
            "h_cost",
            "max_abs_alpha",
            "max_abs_beta",
            "min_v",
            "v_min",
        ]
        if include_u_nom:
            header += ["u_nom_ail", "u_nom_rud", "u_nom_elev", "u_nom_T"]
        header += [
            "trend_reward",
            "time_in_tol_frac",
            "max_consecutive_in_tol",
            "first_entry_step",
            "barrier_cost",
            "drift_cost",
            "behavioral_success",
            "final_window_good",
            "drift_ramp",
            "hold_success",
            "final_window_frac",
            "max_consecutive_in_tol_s",
            "max_abs_h_err",
            "mean_abs_h_err",
            "true_success",
        ]
        self._writer.writerow(header)

    def log(
        self,
        t,
        state,
        u,
        reward,
        terminated,
        truncated,
        task_id=0,
        success=False,
        reason="",
        in_tol=False,
        err_cost=0.0,
        u_cost=0.0,
        slew_cost=0.0,
        prog=0.0,
        hold_bonus=0.0,
        sat_cost=0.0,
        theta_cost=0.0,
        V_cost=0.0,
        h_cost=0.0,
        max_abs_alpha=0.0,
        max_abs_beta=0.0,
        min_v=0.0,
        v_min=0.0,
        trend_reward=0.0,
        time_in_tol_frac=0.0,
        max_consecutive_in_tol=0,
        first_entry_step=-1,
        barrier_cost=0.0,
        drift_cost=0.0,
        behavioral_success=False,
        final_window_good=False,
        drift_ramp=0.0,
        hold_success=False,
        final_window_frac=0.0,
        max_consecutive_in_tol_s=0.0,
        max_abs_h_err=0.0,
        mean_abs_h_err=0.0,
        true_success=False,
        u_nom=None,
    ):
        row = (
            [t]
            + list(state)
            + list(u)
            + [
                reward,
                int(terminated),
                int(truncated),
                task_id,
                int(success),
                reason,
                int(in_tol),
                err_cost,
                u_cost,
                slew_cost,
                prog,
                hold_bonus,
                sat_cost,
                theta_cost,
                V_cost,
                h_cost,
                max_abs_alpha,
                max_abs_beta,
                min_v,
                v_min,
            ]
        )
        if self._include_u_nom:
            if u_nom is None:
                row += [0.0, 0.0, 0.0, 0.0]
            else:
                row += list(u_nom)
        row += [
            trend_reward,
            time_in_tol_frac,
            max_consecutive_in_tol,
            first_entry_step,
            barrier_cost,
            drift_cost,
            int(behavioral_success),
            int(final_window_good),
            drift_ramp,
            int(hold_success),
            final_window_frac,
            max_consecutive_in_tol_s,
            max_abs_h_err,
            mean_abs_h_err,
            int(true_success),
        ]
        self._writer.writerow(row)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
