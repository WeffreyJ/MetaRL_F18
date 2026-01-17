"""
Examples:
PYTHONPATH=python python3 -m rl_env.scripts.train_sb3 --env drive_to_trim --algo ppo --task_mode default --obs_mode inner8 --total_steps 5000 --dt 0.02 --seed 0 --log_dir runs/smoke
PYTHONPATH=python python3 -m rl_env.scripts.train_sb3 --env residual_lqr --algo ppo --task_mode meta --obs_mode inner8 --total_steps 200000 --dt 0.02 --seed 0 --log_dir runs/exp1
"""
import argparse
import csv
import json
import os
from collections import Counter, deque

import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    gym = None
    PPO = None
    SAC = None
    DummyVecEnv = None
    VecMonitor = None
    BaseCallback = object

from rl_env.f18_env_drive_to_trim import F18EnvDriveToTrim
from rl_env.f18_env_residual_lqr import F18EnvResidualLqr
from rl_env.task import load_default_task, sample_task, obs_dim_from_task
from rl_env.utils_csv_logger import CSVLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train SB3 on F-18 env.")
    p.add_argument("--env", choices=["drive_to_trim", "residual_lqr"], default="drive_to_trim")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--task_mode", choices=["default", "meta"], default="default")
    p.add_argument("--obs_mode", choices=["inner8", "inner10_vh", "inner11_vhE", "full12"], default="inner8")
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_dir", default="runs/exp1")
    p.add_argument("--horizon_s", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--terminate_on_success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override terminate-on-success behavior.",
    )
    p.add_argument("--checkpoint_freq", type=int, default=50000)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--policy_layers", default="64,64")
    p.add_argument("--eval_rollout", action="store_true")
    p.add_argument("--plot", action="store_true", help="Plot learning curve at end")
    p.add_argument("--long_horizon", type=float, default=None, help="Convenience horizon in seconds")
    return p.parse_args()


def parse_layers(s):
    return [int(v) for v in s.split(",") if v.strip()]


class MetaTaskWrapper(gym.Wrapper):
    def __init__(self, env, base_task, rng):
        super().__init__(env)
        self._base_task = base_task
        self._rng = rng

    def reset(self, seed=None, options=None):
        task = sample_task(self._rng, self._base_task, mode="meta")
        self.env.set_task(task)
        return self.env.reset(seed=seed, options=options)


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_dir, checkpoint_freq, verbose=0):
        super().__init__(verbose)
        self._log_dir = log_dir
        self._checkpoint_freq = checkpoint_freq
        self._ep_idx = 0
        self._ep_returns = []
        self._ep_lengths = []
        self._recent_returns = deque(maxlen=100)
        self._recent_success = deque(maxlen=100)
        self._recent_time_in_tol = deque(maxlen=100)
        self._recent_max_consec = deque(maxlen=100)
        self._recent_behavioral_success = deque(maxlen=100)
        self._recent_final_window_good = deque(maxlen=100)
        self._recent_hold_success = deque(maxlen=100)
        self._recent_true_success = deque(maxlen=100)
        self._reason_counter = Counter()
        self._csv_path = os.path.join(log_dir, "episode_log.csv")
        self._best_mean_return = None
        self._best_metrics = None
        self._best_model_path = os.path.join(log_dir, "best_model")
        os.makedirs(log_dir, exist_ok=True)
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "episode_idx",
                "timesteps",
                "ep_return",
                "ep_len",
                "success",
                "reason",
                "max_abs_alpha",
                "max_abs_beta",
                "min_v",
                "time_in_tol_frac",
                "max_consecutive_in_tol",
                "first_entry_step",
                "behavioral_success",
                "final_window_good",
                "barrier_cost",
                "drift_cost",
                "drift_ramp",
                "hold_success",
                "final_window_frac",
                "max_consecutive_in_tol_s",
                "max_abs_h_err",
                "mean_abs_h_err",
                "true_success",
            ]
        )
        self._ep_returns_vec = None
        self._ep_lengths_vec = None

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self._ep_returns_vec = np.zeros(n_envs, dtype=float)
        self._ep_lengths_vec = np.zeros(n_envs, dtype=int)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        self._ep_returns_vec += rewards
        self._ep_lengths_vec += 1

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            ep_return = float(self._ep_returns_vec[i])
            ep_len = int(self._ep_lengths_vec[i])
            success = bool(info.get("success", False))
            reason = str(info.get("reason", ""))

            max_abs_alpha = float(info.get("max_abs_alpha", 0.0))
            max_abs_beta = float(info.get("max_abs_beta", 0.0))
            min_v = float(info.get("min_v", 0.0))
            time_in_tol_frac = float(info.get("time_in_tol_frac", 0.0))
            max_consecutive_in_tol = int(info.get("max_consecutive_in_tol", 0))
            first_entry_step = int(info.get("first_entry_step", -1))
            behavioral_success = bool(info.get("behavioral_success", False))
            final_window_good = bool(info.get("final_window_good", False))
            barrier_cost = float(info.get("barrier_cost", 0.0))
            drift_cost = float(info.get("drift_cost", 0.0))
            drift_ramp = float(info.get("drift_ramp", 0.0))
            hold_success = bool(info.get("hold_success", False))
            true_success = bool(info.get("true_success", False))
            final_window_frac = float(info.get("final_window_frac", 0.0))
            max_consecutive_in_tol_s = float(info.get("max_consecutive_in_tol_s", 0.0))
            max_abs_h_err = float(info.get("max_abs_h_err", 0.0))
            mean_abs_h_err = float(info.get("mean_abs_h_err", 0.0))
            self._csv_writer.writerow(
                [
                    self._ep_idx,
                    self.num_timesteps,
                    ep_return,
                    ep_len,
                    int(success),
                    reason,
                    max_abs_alpha,
                    max_abs_beta,
                    min_v,
                    time_in_tol_frac,
                    max_consecutive_in_tol,
                    first_entry_step,
                    int(behavioral_success),
                    int(final_window_good),
                    barrier_cost,
                    drift_cost,
                    drift_ramp,
                    int(hold_success),
                    final_window_frac,
                    max_consecutive_in_tol_s,
                    max_abs_h_err,
                    mean_abs_h_err,
                    int(true_success),
                ]
            )
            self._csv_file.flush()

            self._recent_returns.append(ep_return)
            self._recent_success.append(1 if success else 0)
            self._recent_time_in_tol.append(time_in_tol_frac)
            self._recent_max_consec.append(max_consecutive_in_tol)
            self._recent_behavioral_success.append(1 if behavioral_success else 0)
            self._recent_final_window_good.append(1 if final_window_good else 0)
            self._recent_hold_success.append(1 if hold_success else 0)
            self._recent_true_success.append(1 if true_success else 0)
            self._reason_counter[reason] += 1
            self._ep_idx += 1

            self._ep_returns_vec[i] = 0.0
            self._ep_lengths_vec[i] = 0

            if len(self._recent_returns) >= 20:
                mean_return = float(np.mean(self._recent_returns))
                mean_time_in_tol = float(np.mean(self._recent_time_in_tol)) if self._recent_time_in_tol else 0.0
                mean_max_consec = float(np.mean(self._recent_max_consec)) if self._recent_max_consec else 0.0
                mean_behavioral = float(np.mean(self._recent_behavioral_success)) if self._recent_behavioral_success else 0.0
                mean_final_window = float(np.mean(self._recent_final_window_good)) if self._recent_final_window_good else 0.0
                mean_hold = float(np.mean(self._recent_hold_success)) if self._recent_hold_success else 0.0
                mean_true = float(np.mean(self._recent_true_success)) if self._recent_true_success else 0.0
                candidate = (mean_true, mean_final_window, mean_hold, mean_time_in_tol, mean_max_consec, mean_return)
                if self._best_metrics is None or candidate > self._best_metrics:
                    self._best_metrics = candidate
                    self._best_mean_return = mean_return
                    self.model.save(self._best_model_path)
                    with open(os.path.join(self._log_dir, "best_metrics.json"), "w") as f:
                        json.dump(
                            {
                                "mean_behavioral_success": mean_behavioral,
                                "mean_final_window_good": mean_final_window,
                                "mean_hold_success": mean_hold,
                                "mean_true_success": mean_true,
                                "mean_time_in_tol": mean_time_in_tol,
                                "mean_max_consecutive_in_tol": mean_max_consec,
                                "mean_return": mean_return,
                            },
                            f,
                            indent=2,
                        )

        if self.num_timesteps % self._checkpoint_freq == 0:
            ckpt_dir = os.path.join(self._log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir, f"{self.model.__class__.__name__.lower()}_step{self.num_timesteps}.zip"
            )
            self.model.save(ckpt_path)

        if self._ep_idx > 0 and self._ep_idx % 100 == 0:
            mean_return = float(np.mean(self._recent_returns)) if self._recent_returns else 0.0
            success_rate = float(np.mean(self._recent_success)) if self._recent_success else 0.0
            mean_time_in_tol = float(np.mean(self._recent_time_in_tol)) if self._recent_time_in_tol else 0.0
            mean_max_consec = float(np.mean(self._recent_max_consec)) if self._recent_max_consec else 0.0
            mean_behavioral = float(np.mean(self._recent_behavioral_success)) if self._recent_behavioral_success else 0.0
            mean_final_window = float(np.mean(self._recent_final_window_good)) if self._recent_final_window_good else 0.0
            mean_hold = float(np.mean(self._recent_hold_success)) if self._recent_hold_success else 0.0
            mean_true = float(np.mean(self._recent_true_success)) if self._recent_true_success else 0.0
            top_reasons = dict(self._reason_counter.most_common(5))
            summary = {
                "episodes": self._ep_idx,
                "timesteps": self.num_timesteps,
                "mean_return_last100": mean_return,
                "success_rate_last100": success_rate,
                "behavioral_success_last100": mean_behavioral,
                "final_window_good_last100": mean_final_window,
                "hold_success_last100": mean_hold,
                "true_success_last100": mean_true,
                "mean_time_in_tol_last100": mean_time_in_tol,
                "mean_max_consecutive_in_tol_last100": mean_max_consec,
                "top_reasons": top_reasons,
            }
            print(summary)
            with open(os.path.join(self._log_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()

    def best_model_path(self):
        return f"{self._best_model_path}.zip" if self._best_mean_return is not None else ""


def make_env(env_name, task_mode, obs_mode, seed, dt, horizon_s, max_steps, terminate_on_success):
    base_task = load_default_task(obs_mode=obs_mode, dt=dt, horizon_s=horizon_s, max_steps=max_steps)
    rng = np.random.default_rng(seed)

    def _make():
        if env_name == "drive_to_trim":
            env = F18EnvDriveToTrim(dt=dt, task=base_task, seed=seed)
        else:
            env = F18EnvResidualLqr(dt=dt, task=base_task, seed=seed)
        if terminate_on_success is not None:
            env._bounds["terminate_on_success"] = bool(terminate_on_success)
        if task_mode == "meta":
            env = MetaTaskWrapper(env, base_task, rng)
        return env

    return _make


def eval_rollout(env, out_path, steps):
    logger = CSVLogger(out_path, include_u_nom=False)
    obs, info = env.reset()
    for _ in range(steps):
        action = np.zeros(4, dtype=float)
        obs, reward, terminated, truncated, info = env.step(action)
        state = env.get_state12()
        u = info.get("u") if isinstance(info, dict) else env._task.u_trim
        logger.log(info.get("t", 0.0), state, u, reward, terminated, truncated, task_id=env._task.task_id)
        if terminated or truncated:
            break
    logger.close()


def _rolling_mean(values, window):
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        out.append(float(np.mean(window_vals)))
    return out


def plot_learning_curve(csv_path, out_path, window=50):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("pip3 install matplotlib")
        return

    if not os.path.exists(csv_path):
        print(f"Missing episode log: {csv_path}")
        return

    steps = []
    returns = []
    success = []
    behavioral_success = []
    hold_success = []
    true_success = []
    time_in_tol = []
    max_consec = []
    final_window_frac = []
    barrier_cost = []
    drift_cost = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row.get("timesteps", 0)))
            returns.append(float(row.get("ep_return", 0.0)))
            success.append(int(row.get("success", 0)))
            behavioral_success.append(int(row.get("behavioral_success", 0)))
            hold_success.append(int(row.get("hold_success", 0)))
            true_success.append(int(row.get("true_success", 0)))
            time_in_tol.append(float(row.get("time_in_tol_frac", 0.0)))
            max_consec.append(float(row.get("max_consecutive_in_tol", 0.0)))
            final_window_frac.append(float(row.get("final_window_frac", 0.0)))
            barrier_cost.append(float(row.get("barrier_cost", 0.0)))
            drift_cost.append(float(row.get("drift_cost", 0.0)))

    if not steps:
        print("No episodes in log; skipping plot.")
        return

    returns_sm = _rolling_mean(returns, window)
    success_sm = _rolling_mean(success, window)
    behavioral_sm = _rolling_mean(behavioral_success, window)
    hold_sm = _rolling_mean(hold_success, window)
    true_success_sm = _rolling_mean(true_success, window)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, returns, alpha=0.3)
    plt.plot(steps, returns_sm, linewidth=2)
    plt.title("Episode Return")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    success_path = os.path.join(os.path.dirname(out_path), "rolling_terminal_success.png")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, success, alpha=0.3)
    plt.plot(steps, success_sm, linewidth=2)
    plt.title("Terminal Success Rate (rolling)")
    plt.xlabel("Timesteps")
    plt.ylabel("Success")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(success_path, dpi=150)
    plt.close()
    # Backwards-compatible filename.
    legacy_success_path = os.path.join(os.path.dirname(out_path), "learning_curve_success.png")
    if legacy_success_path != success_path:
        try:
            import shutil

            shutil.copyfile(success_path, legacy_success_path)
        except Exception:
            pass

    behavioral_path = os.path.join(os.path.dirname(out_path), "rolling_behavioral_success.png")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, behavioral_success, alpha=0.3)
    plt.plot(steps, behavioral_sm, linewidth=2)
    plt.title("Behavioral Success Rate (rolling)")
    plt.xlabel("Timesteps")
    plt.ylabel("Behavioral Success")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(behavioral_path, dpi=150)
    plt.close()

    hold_path = os.path.join(os.path.dirname(out_path), "rolling_hold_success.png")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, hold_success, alpha=0.3)
    plt.plot(steps, hold_sm, linewidth=2)
    plt.title("Hold Success Rate (rolling)")
    plt.xlabel("Timesteps")
    plt.ylabel("Hold Success")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(hold_path, dpi=150)
    plt.close()

    true_path = os.path.join(os.path.dirname(out_path), "rolling_true_success.png")
    plt.figure(figsize=(8, 4))
    plt.plot(steps, true_success, alpha=0.3)
    plt.plot(steps, true_success_sm, linewidth=2)
    plt.title("True Success Rate (rolling)")
    plt.xlabel("Timesteps")
    plt.ylabel("True Success")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(true_path, dpi=150)
    plt.close()

    if time_in_tol:
        tol_path = os.path.join(os.path.dirname(out_path), "rolling_time_in_tol_frac.png")
        time_in_tol_sm = _rolling_mean(time_in_tol, window)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, time_in_tol, alpha=0.3)
        plt.plot(steps, time_in_tol_sm, linewidth=2)
        plt.title("Time-in-Tolerance Fraction (rolling)")
        plt.xlabel("Timesteps")
        plt.ylabel("time_in_tol_frac")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(tol_path, dpi=150)
        plt.close()

    if max_consec:
        consec_path = os.path.join(os.path.dirname(out_path), "rolling_max_consecutive_in_tol.png")
        max_consec_sm = _rolling_mean(max_consec, window)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, max_consec, alpha=0.3)
        plt.plot(steps, max_consec_sm, linewidth=2)
        plt.title("Max Consecutive In-Tol (rolling)")
        plt.xlabel("Timesteps")
        plt.ylabel("max_consecutive_in_tol")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(consec_path, dpi=150)
        plt.close()

    if final_window_frac:
        window_path = os.path.join(os.path.dirname(out_path), "rolling_final_window_frac.png")
        window_sm = _rolling_mean(final_window_frac, window)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, final_window_frac, alpha=0.3)
        plt.plot(steps, window_sm, linewidth=2)
        plt.title("Final Window Fraction (rolling)")
        plt.xlabel("Timesteps")
        plt.ylabel("final_window_frac")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(window_path, dpi=150)
        plt.close()

    if barrier_cost:
        barrier_path = os.path.join(os.path.dirname(out_path), "rolling_barrier_cost.png")
        barrier_sm = _rolling_mean(barrier_cost, window)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, barrier_cost, alpha=0.3)
        plt.plot(steps, barrier_sm, linewidth=2)
        plt.title("Barrier Cost (rolling)")
        plt.xlabel("Timesteps")
        plt.ylabel("barrier_cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(barrier_path, dpi=150)
        plt.close()

    if drift_cost:
        drift_path = os.path.join(os.path.dirname(out_path), "rolling_drift_cost.png")
        drift_sm = _rolling_mean(drift_cost, window)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, drift_cost, alpha=0.3)
        plt.plot(steps, drift_sm, linewidth=2)
        plt.title("Drift Cost (rolling)")
        plt.xlabel("Timesteps")
        plt.ylabel("drift_cost")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(drift_path, dpi=150)
        plt.close()


def main():
    args = parse_args()
    if gym is None or PPO is None:
        print("Missing deps. Install with: pip install gymnasium stable-baselines3 torch")
        return 1

    if args.long_horizon is not None:
        args.horizon_s = float(args.long_horizon)
        if args.checkpoint_freq == 50000:
            args.checkpoint_freq = 250000

    os.makedirs(args.log_dir, exist_ok=True)
    policy_layers = parse_layers(args.policy_layers)
    policy_kwargs = dict(net_arch=policy_layers)

    base_task_preview = load_default_task(
        obs_mode=args.obs_mode, dt=args.dt, horizon_s=args.horizon_s, max_steps=args.max_steps
    )
    horizon_steps_60 = int(round(60.0 / args.dt))
    if args.terminate_on_success is None:
        terminate_on_success = base_task_preview.max_steps <= horizon_steps_60
    else:
        terminate_on_success = bool(args.terminate_on_success)
    print(f"max_steps={base_task_preview.max_steps}, terminate_on_success={terminate_on_success}")

    if args.env == "drive_to_trim":
        preview_env = F18EnvDriveToTrim(dt=args.dt, task=base_task_preview, seed=args.seed)
    else:
        preview_env = F18EnvResidualLqr(dt=args.dt, task=base_task_preview, seed=args.seed)
    preview_env._bounds["terminate_on_success"] = terminate_on_success
    obs_dim = obs_dim_from_task(base_task_preview)
    run_config = {
        "env": args.env,
        "algo": args.algo,
        "obs_mode": args.obs_mode,
        "task_mode": args.task_mode,
        "dt": args.dt,
        "horizon_s": args.horizon_s,
        "max_steps": base_task_preview.max_steps,
        "terminate_on_success": terminate_on_success,
        "obs_dim": obs_dim,
        "obs_indices": list(base_task_preview.obs_indices),
        "weights": list(getattr(preview_env, "_w", [])),
        "reward_params": {
            "lambda_trend": getattr(preview_env, "_lambda_trend", None),
            "trend_clip": getattr(preview_env, "_trend_clip", None),
            "lambda_theta": getattr(preview_env, "_lambda_theta", None),
            "lambda_u": getattr(preview_env, "_lambda_u", None),
            "lambda_du": getattr(preview_env, "_lambda_du", None),
            "k_hold": getattr(preview_env, "_k_hold", None),
            "lambda_barrier": getattr(preview_env, "_lambda_barrier", None),
            "barrier_margin": getattr(preview_env, "_barrier_margin", None),
            "k_drift": getattr(preview_env, "_k_drift", None),
        },
        "bounds": getattr(preview_env, "_bounds", {}),
    }
    with open(os.path.join(args.log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    with open(os.path.join(args.log_dir, "run_config.txt"), "w") as f:
        for k, v in run_config.items():
            f.write(f"{k}: {v}\n")
    preview_env.close()

    env_fn = make_env(
        args.env,
        args.task_mode,
        args.obs_mode,
        args.seed,
        args.dt,
        args.horizon_s,
        args.max_steps,
        terminate_on_success,
    )
    vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    if args.algo == "ppo":
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs)
    else:
        model = SAC("MlpPolicy", vec_env, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs)

    callback = EpisodeLoggerCallback(args.log_dir, args.checkpoint_freq)
    model.learn(total_timesteps=args.total_steps, callback=callback)
    final_model_path = os.path.join(args.log_dir, "model")
    model.save(final_model_path)

    if args.eval_rollout:
        eval_env = env_fn()
        steps = int(round(args.total_steps * 0.0 + 1000))
        eval_rollout(eval_env, os.path.join(args.log_dir, "eval_rollout.csv"), steps)
        eval_env.close()

    if args.plot:
        plot_path = os.path.join(args.log_dir, "learning_curve.png")
        plot_learning_curve(os.path.join(args.log_dir, "episode_log.csv"), plot_path)

    vec_env.close()
    print(f"final_model={final_model_path}.zip")
    best_path = callback.best_model_path()
    if best_path:
        print(f"best_model={best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
