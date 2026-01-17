import argparse
import csv
import glob
import os
import json

import numpy as np

from rl_env.f18_env_drive_to_trim import F18EnvDriveToTrim
from rl_env.f18_env_residual_lqr import F18EnvResidualLqr
from rl_env.task import load_default_task, obs_dim_from_task


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SB3 checkpoints on F-18 env.")
    p.add_argument("--log_dir", required=True, help="Training log dir with checkpoints/")
    p.add_argument("--env", choices=["drive_to_trim", "residual_lqr"], default="drive_to_trim")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument(
        "--obs_mode",
        choices=["inner8", "inner10_vh", "inner11_vhE", "full12"],
        default=None,
        help="Observation mode; must match the trained model.",
    )
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--sim_time", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max_n", type=int, default=None, help="Limit number of models evaluated")
    return p.parse_args()


def _resolve_obs_mode(args):
    if args.obs_mode:
        return args.obs_mode
    config_path = os.path.join(args.log_dir, "run_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "obs_mode" in cfg:
                return str(cfg["obs_mode"])
        except Exception:
            pass
    print("WARNING: run_config.json not found or unreadable; defaulting obs_mode=inner8")
    return "inner8"


def _list_models(log_dir, max_n=None):
    paths = []
    for name in ["model.zip", "best_model.zip"]:
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            paths.append(path)

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.zip")))
    paths.extend(ckpts)

    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if max_n is not None and max_n > 0 and len(uniq) > max_n:
        uniq = uniq[-max_n:]

    return uniq


def _make_env(env_name, dt, sim_time, seed, obs_mode):
    max_steps = int(round(sim_time / dt))
    task = load_default_task(obs_mode=obs_mode, dt=dt, max_steps=max_steps)
    if env_name == "drive_to_trim":
        env = F18EnvDriveToTrim(dt=dt, task=task, seed=seed)
    else:
        env = F18EnvResidualLqr(dt=dt, task=task, seed=seed)
    if hasattr(env, "_verbose_reset"):
        env._verbose_reset = False
    return env, task


def _rollout_model(model, env, steps, deterministic, seed=None):
    ret = env.reset(seed=seed)
    obs = ret[0] if isinstance(ret, tuple) else ret
    total_return = 0.0
    in_tol_count = 0
    consecutive_in_tol = 0
    max_consecutive_in_tol = 0
    max_abs_theta_err = 0.0
    max_abs_V_err = 0.0
    max_abs_alpha_err = 0.0
    max_abs_beta_err = 0.0
    max_abs_phi_err = 0.0
    min_h = float("inf")
    min_V = float("inf")
    max_V = -float("inf")
    final_h = float("nan")
    final_V = float("nan")
    terminated = False
    truncated = False
    reason = ""
    final_err_cost = 0.0
    t_end = 0.0
    behavioral_success = False
    final_window_good = False
    hold_success = False
    true_success = False
    final_window_frac = 0.0
    max_abs_h_err = 0.0
    mean_abs_h_err = 0.0

    steps_run = 0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=float).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        steps_run += 1
        if isinstance(info, dict):
            reason = str(info.get("reason", reason))
            in_tol = bool(info.get("in_tol", False))
            in_tol_count += 1 if in_tol else 0
            if in_tol:
                consecutive_in_tol += 1
            else:
                consecutive_in_tol = 0
            max_consecutive_in_tol = max(max_consecutive_in_tol, consecutive_in_tol)
            final_err_cost = float(info.get("err_cost", final_err_cost))
            t_end = float(info.get("t", t_end))
            behavioral_success = bool(info.get("behavioral_success", behavioral_success))
            final_window_good = bool(info.get("final_window_good", final_window_good))
            hold_success = bool(info.get("hold_success", hold_success))
            true_success = bool(info.get("true_success", true_success))
            final_window_frac = float(info.get("final_window_frac", final_window_frac))
            max_abs_h_err = float(info.get("max_abs_h_err", max_abs_h_err))
            mean_abs_h_err = float(info.get("mean_abs_h_err", mean_abs_h_err))
        state = env.get_state12()
        dx = state - env._task.x_trim
        max_abs_theta_err = max(max_abs_theta_err, abs(dx[7]))
        max_abs_V_err = max(max_abs_V_err, abs(dx[0]))
        max_abs_alpha_err = max(max_abs_alpha_err, abs(dx[2]))
        max_abs_beta_err = max(max_abs_beta_err, abs(dx[1]))
        max_abs_phi_err = max(max_abs_phi_err, abs(dx[6]))
        min_h = min(min_h, state[11])
        min_V = min(min_V, state[0])
        max_V = max(max_V, state[0])
        final_h = state[11]
        final_V = state[0]
        if terminated or truncated:
            break

    time_in_tol_frac = in_tol_count / max(1, steps_run)
    max_consecutive_in_tol_s = max_consecutive_in_tol * env._dt
    success = bool(info.get("success", False)) if isinstance(info, dict) else False
    envelope_reasons = {
        "h_band",
        "v_band",
        "theta_band",
        "ground",
        "h_max",
        "v_min_persist",
        "alpha_persist",
        "beta_persist",
    }
    envelope_reason = reason in envelope_reasons
    return {
        "t_end": t_end,
        "steps": steps_run,
        "terminated": terminated,
        "truncated": truncated,
        "reason": reason,
        "success": success,
        "total_return": total_return,
        "time_in_tol_frac": time_in_tol_frac,
        "max_consecutive_in_tol": max_consecutive_in_tol,
        "max_consecutive_in_tol_s": max_consecutive_in_tol_s,
        "final_err_cost": final_err_cost,
        "max_abs_theta_err": max_abs_theta_err,
        "max_abs_V_err": max_abs_V_err,
        "max_abs_alpha_err": max_abs_alpha_err,
        "max_abs_beta_err": max_abs_beta_err,
        "max_abs_phi_err": max_abs_phi_err,
        "min_h": min_h,
        "min_V": min_V,
        "max_V": max_V,
        "final_h": final_h,
        "final_V": final_V,
        "envelope_reason": envelope_reason,
        "behavioral_success": behavioral_success,
        "final_window_good": final_window_good,
        "hold_success": hold_success,
        "true_success": true_success,
        "final_window_frac": final_window_frac,
        "max_abs_h_err": max_abs_h_err,
        "mean_abs_h_err": mean_abs_h_err,
    }


def main():
    args = parse_args()
    try:
        from stable_baselines3 import PPO, SAC
    except Exception:
        print("pip install stable-baselines3")
        return 1

    resolved_obs_mode = _resolve_obs_mode(args)
    model_paths = _list_models(args.log_dir, max_n=args.max_n)
    if not model_paths:
        print("No model checkpoints found.")
        return 1

    steps = int(round(args.sim_time / args.dt))
    results = []
    env, task = _make_env(args.env, args.dt, args.sim_time, args.seed, resolved_obs_mode)
    obs_dim = obs_dim_from_task(task)
    print(f"eval: obs_mode={resolved_obs_mode}, obs_dim={obs_dim}, dt={args.dt}, sim_time={args.sim_time}")
    env.close()

    for path in model_paths:
        if args.algo == "ppo":
            model = PPO.load(path)
        else:
            model = SAC.load(path)
        env, _ = _make_env(args.env, args.dt, args.sim_time, args.seed, resolved_obs_mode)
        if hasattr(model, "observation_space") and hasattr(model.observation_space, "shape"):
            model_obs_dim = model.observation_space.shape[0]
            if model_obs_dim != obs_dim:
                raise SystemExit(
                    f"Obs mismatch: env obs_dim={obs_dim} model expects {model_obs_dim}. "
                    "Use --obs_mode or ensure run_config.json is present."
                )
        metrics = _rollout_model(model, env, steps, args.deterministic, seed=args.seed)
        env.close()
        metrics["model_path"] = path
        results.append(metrics)

    out_path = os.path.join(args.log_dir, "eval_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_path",
                "t_end",
                "steps",
                "terminated",
                "truncated",
                "reason",
                "success",
                "envelope_reason",
                "total_return",
                "time_in_tol_frac",
                "max_consecutive_in_tol",
                "max_consecutive_in_tol_s",
                "final_err_cost",
                "max_abs_theta_err",
                "max_abs_V_err",
                "max_abs_alpha_err",
                "max_abs_beta_err",
                "max_abs_phi_err",
                "min_h",
                "min_V",
                "max_V",
                "final_h",
                "final_V",
                "behavioral_success",
                "final_window_good",
                "hold_success",
                "true_success",
                "final_window_frac",
                "max_abs_h_err",
                "mean_abs_h_err",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["model_path"],
                    r["t_end"],
                    r["steps"],
                    int(r["terminated"]),
                    int(r["truncated"]),
                    r["reason"],
                    int(r["success"]),
                    int(r["envelope_reason"]),
                    r["total_return"],
                    r["time_in_tol_frac"],
                    r["max_consecutive_in_tol"],
                    r["max_consecutive_in_tol_s"],
                    r["final_err_cost"],
                    r["max_abs_theta_err"],
                    r["max_abs_V_err"],
                    r["max_abs_alpha_err"],
                    r["max_abs_beta_err"],
                    r["max_abs_phi_err"],
                    r["min_h"],
                    r["min_V"],
                    r["max_V"],
                    r["final_h"],
                    r["final_V"],
                    int(r["behavioral_success"]),
                    int(r["final_window_good"]),
                    int(r["hold_success"]),
                    int(r.get("true_success", False)),
                    r["final_window_frac"],
                    r["max_abs_h_err"],
                    r["mean_abs_h_err"],
                ]
            )

    ranked = sorted(
        results,
        key=lambda r: (
            int(r.get("true_success", False)),
            int(r["final_window_good"]),
            int(r["hold_success"]),
            r["time_in_tol_frac"],
            r["max_consecutive_in_tol"],
            r["total_return"],
        ),
        reverse=True,
    )
    print("Top 5 models by true_success, final_window_good, hold_success, time_in_tol, max_consec, return:")
    for r in ranked[:5]:
        print(
            f"{r['model_path']} | true={int(r.get('true_success', False))}"
            f"| win={int(r['final_window_good'])}"
            f"| hold={int(r['hold_success'])}"
            f"| in_tol={r['time_in_tol_frac']:.3f}"
            f"| max_consec={r['max_consecutive_in_tol']}"
            f"| return={r['total_return']:.2f} | reason={r['reason']}"
        )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
