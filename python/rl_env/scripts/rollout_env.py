"""
Examples:
PYTHONPATH=python python3 -m rl_env.scripts.env_smoke_test
PYTHONPATH=python python3 -m rl_env.scripts.rollout_env --env residual_lqr --sim_time 30 --dt 0.02 --out rollout.csv --seed 0 --policy zero
PYTHONPATH=python python3 -m rl_env.scripts.plot_env_rollout rollout.csv
"""
import argparse
import json
import os

import numpy as np

from rl_env.f18_env_drive_to_trim import F18EnvDriveToTrim
from rl_env.f18_env_residual_lqr import F18EnvResidualLqr
from rl_env.task import load_default_task, sample_task, obs_dim_from_task
from rl_env.utils_csv_logger import CSVLogger


def parse_args():
    p = argparse.ArgumentParser(description="Rollout F-18 RL env.")
    p.add_argument("--env", choices=["drive_to_trim", "residual_lqr"], default="drive_to_trim")
    p.add_argument("--sim_time", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--out", default="rollout.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["zero", "random", "sb3"], default="zero")
    p.add_argument("--model", default=None, help="Path to SB3 .zip model")
    p.add_argument("--model_path", dest="model", default=None, help="Alias for --model")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--task_mode", choices=["default", "meta"], default="default")
    p.add_argument(
        "--obs_mode",
        choices=["inner8", "inner10_vh", "inner11_vhE", "full12"],
        default=None,
        help="Observation mode; must match trained policy.",
    )
    p.add_argument("--log_dir", default=None, help="Optional run dir containing run_config.json")
    p.add_argument("--verbose_reset", action="store_true")
    return p.parse_args()


def _resolve_obs_mode(args):
    if args.obs_mode:
        return args.obs_mode
    candidates = []
    if args.log_dir:
        candidates.append(args.log_dir)
    if args.model:
        model_dir = os.path.dirname(os.path.abspath(args.model))
        candidates.append(model_dir)
        candidates.append(os.path.dirname(model_dir))
    for base in candidates:
        config_path = os.path.join(base, "run_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict) and "obs_mode" in cfg:
                    return str(cfg["obs_mode"])
            except Exception:
                pass
    print("WARNING: run_config.json not found; defaulting obs_mode=inner8")
    return "inner8"


def main():
    args = parse_args()
    resolved_obs_mode = _resolve_obs_mode(args)
    max_steps = int(round(args.sim_time / args.dt))
    task = load_default_task(obs_mode=resolved_obs_mode, dt=args.dt, max_steps=max_steps)
    if args.task_mode == "meta":
        task = sample_task(np.random.default_rng(args.seed), task, mode="meta")

    if args.env == "drive_to_trim":
        env = F18EnvDriveToTrim(dt=args.dt, task=task, seed=args.seed)
    else:
        env = F18EnvResidualLqr(dt=args.dt, task=task, seed=args.seed)
    if hasattr(env, "_verbose_reset"):
        env._verbose_reset = args.verbose_reset

    ret = env.reset(seed=args.seed)
    obs = ret[0] if isinstance(ret, tuple) else ret

    steps = int(round(args.sim_time / args.dt))
    rng = np.random.default_rng(args.seed)
    logger = CSVLogger(args.out, include_u_nom=(args.env == "residual_lqr"))

    model = None
    if args.log_dir and not args.model:
        for candidate in ["best_model.zip", "model.zip"]:
            path = os.path.join(args.log_dir, candidate)
            if os.path.exists(path):
                args.model = path
                break
        if not args.model:
            raise SystemExit(f"No model.zip found in log_dir: {args.log_dir}")
    if args.policy == "sb3":
        if args.model is None:
            raise SystemExit("Missing --model for sb3 policy.")
        try:
            from stable_baselines3 import PPO, SAC
        except Exception:
            raise SystemExit("Stable-Baselines3 not installed. pip install stable-baselines3")
        if args.algo == "ppo":
            model = PPO.load(args.model)
        else:
            model = SAC.load(args.model)
        print(
            f"SB3 policy: model={args.model}, algo={args.algo}, deterministic={args.deterministic}"
        )
        env_obs_dim = obs_dim_from_task(task)
        model_obs_dim = None
        if hasattr(model, "observation_space") and hasattr(model.observation_space, "shape"):
            model_obs_dim = model.observation_space.shape[0]
        if model_obs_dim is not None and env_obs_dim != model_obs_dim:
            raise SystemExit(
                f"Obs mismatch: env obs_dim={env_obs_dim} model expects {model_obs_dim}. "
                f"Use --obs_mode to match the trained policy."
            )
        print(f"Loaded model: {args.model}")

    env_obs_dim = obs_dim_from_task(task)
    print(
        f"rollout: env={args.env}, algo={args.algo}, dt={args.dt}, sim_time={args.sim_time}, "
        f"obs_mode={resolved_obs_mode}, obs_dim={env_obs_dim}"
    )

    final_info = {}
    terminated = False
    truncated = False
    steps_run = 0
    for _ in range(steps):
        if args.policy == "zero":
            action = np.zeros(4, dtype=float)
        elif args.policy == "random":
            action = rng.uniform(-1.0, 1.0, size=4)
        else:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            action = np.asarray(action, dtype=float).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info if isinstance(info, dict) else {}
        steps_run += 1
        state = env.get_state12()
        u = info.get("u") if isinstance(info, dict) else env._task.u_trim
        u_nom = info.get("u_nom") if isinstance(info, dict) else None
        logger.log(
            info.get("t", 0.0),
            state,
            u,
            reward,
            terminated,
            truncated,
            task_id=env._task.task_id,
            success=info.get("success", False),
            reason=info.get("reason", ""),
            in_tol=info.get("in_tol", False),
            err_cost=info.get("err_cost", 0.0),
            u_cost=info.get("u_cost", 0.0),
            slew_cost=info.get("slew_cost", 0.0),
            prog=info.get("prog", 0.0),
            hold_bonus=info.get("hold_bonus", 0.0),
            sat_cost=info.get("sat_cost", 0.0),
            theta_cost=info.get("theta_cost", 0.0),
            V_cost=info.get("V_cost", 0.0),
            h_cost=info.get("h_cost", 0.0),
            trend_reward=info.get("trend_reward", 0.0),
            time_in_tol_frac=info.get("time_in_tol_frac", 0.0),
            max_consecutive_in_tol=info.get("max_consecutive_in_tol", 0),
            first_entry_step=info.get("first_entry_step", -1),
            barrier_cost=info.get("barrier_cost", 0.0),
            drift_cost=info.get("drift_cost", 0.0),
            behavioral_success=info.get("behavioral_success", False),
            final_window_good=info.get("final_window_good", False),
            drift_ramp=info.get("drift_ramp", 0.0),
            hold_success=info.get("hold_success", False),
            final_window_frac=info.get("final_window_frac", 0.0),
            max_consecutive_in_tol_s=info.get("max_consecutive_in_tol_s", 0.0),
            max_abs_h_err=info.get("max_abs_h_err", 0.0),
            mean_abs_h_err=info.get("mean_abs_h_err", 0.0),
            true_success=info.get("true_success", False),
            max_abs_alpha=info.get("max_abs_alpha", 0.0),
            max_abs_beta=info.get("max_abs_beta", 0.0),
            min_v=info.get("min_v", 0.0),
            v_min=info.get("v_min", 0.0),
            u_nom=u_nom,
        )
        if terminated or truncated:
            break

    logger.close()
    env.close()
    print(
        f"steps={steps_run}, terminated={terminated}, truncated={truncated}, "
        f"reason={final_info.get('reason', '')}, "
        f"true_success={final_info.get('true_success', False)}, "
        f"hold_success={final_info.get('hold_success', False)}, "
        f"final_window_good={final_info.get('final_window_good', False)}, "
        f"time_in_tol_frac={final_info.get('time_in_tol_frac', 0.0):.3f}, "
        f"max_consec_s={final_info.get('max_consecutive_in_tol_s', 0.0):.1f}"
    )
    print(f"output={args.out}")


if __name__ == "__main__":
    main()
