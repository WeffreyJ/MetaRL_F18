from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

DEFAULT_HORIZON_S = 60.0


@dataclass
class Task:
    name: str
    x_trim: np.ndarray
    u_trim: np.ndarray
    K: Optional[np.ndarray]
    feedback_state_indices: Optional[List[int]]
    obs_indices: List[int]
    obs_scale: np.ndarray
    init_sampler: Callable[[np.random.Generator], np.ndarray]
    obs_mode: str = "inner8"
    max_steps: int = 3000
    param_vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    obs_builder: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    obs_dim: Optional[int] = None
    disturbance: Optional[Callable[[float, np.ndarray, np.ndarray, np.random.Generator], np.ndarray]] = None
    task_id: int = 0


def obs_dim_from_task(task):
    return task.obs_dim if task.obs_dim is not None else len(task.obs_indices)


def _default_init_sampler(x_trim):
    def _sampler(rng):
        noise = rng.normal(0.0, 1e-3, size=x_trim.shape)
        return x_trim + noise

    return _sampler


def _default_obs_config(obs_mode):
    if obs_mode == "inner8":
        obs_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        obs_scale = np.array([100.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        obs_dim = len(obs_indices)
        obs_builder = None
    elif obs_mode == "inner10_vh":
        # inner8 plus explicit dv/dh error features (no duplicate indices).
        base_idx = [0, 1, 2, 3, 4, 5, 6, 7]
        base_scale = np.array([100.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        obs_indices = base_idx
        obs_scale = base_scale
        obs_dim = 10

        def _builder(x, x_trim):
            x = np.asarray(x, dtype=float)
            x_trim = np.asarray(x_trim, dtype=float)
            base = (x[base_idx] - x_trim[base_idx]) / base_scale
            dv = (x[0] - x_trim[0]) / 500.0
            dh = (x[11] - x_trim[11]) / 5000.0
            return np.concatenate([base, [dv, dh]])

        obs_builder = _builder
    elif obs_mode == "inner11_vhE":
        base_idx = [0, 1, 2, 3, 4, 5, 6, 7]
        base_scale = np.array([100.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
        obs_indices = base_idx
        obs_scale = base_scale
        obs_dim = 11
        g = 32.174

        def _builder(x, x_trim):
            x = np.asarray(x, dtype=float)
            x_trim = np.asarray(x_trim, dtype=float)
            base = (x[base_idx] - x_trim[base_idx]) / base_scale
            dv = (x[0] - x_trim[0]) / 500.0
            dh = (x[11] - x_trim[11]) / 5000.0
            e = 0.5 * x[0] ** 2 + g * x[11]
            e_trim = 0.5 * x_trim[0] ** 2 + g * x_trim[11]
            dE = (e - e_trim) / 100000.0
            return np.concatenate([base, [dv, dh, dE]])

        obs_builder = _builder
    elif obs_mode == "full12":
        obs_indices = list(range(12))
        obs_scale = np.array(
            [100.0, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5000.0, 5000.0, 5000.0],
            dtype=float,
        )
        obs_dim = len(obs_indices)
        obs_builder = None
    else:
        raise ValueError(f"Unknown obs_mode: {obs_mode}")
    return obs_indices, obs_scale, obs_dim, obs_builder


def load_default_task(obs_mode="inner8", dt=0.02, horizon_s=None, max_steps=None):
    x_trim = np.zeros(12, dtype=float)
    u_trim = np.array([0.0, 0.0, -0.022, 5470.5], dtype=float)
    K = None
    feedback_state_indices = None

    try:
        from linmodel_f18 import x_trim as _x_trim, u_trim as _u_trim

        x_trim = np.asarray(_x_trim, dtype=float)
        u_trim = np.asarray(_u_trim, dtype=float)
    except Exception:
        pass

    try:
        from controller_lqr import K as _K, X8_IDX as _X8_IDX

        K = np.asarray(_K, dtype=float)
        feedback_state_indices = list(_X8_IDX)
    except Exception:
        K = None
        feedback_state_indices = None

    obs_indices, obs_scale, obs_dim, obs_builder = _default_obs_config(obs_mode)

    if max_steps is None and horizon_s is not None:
        max_steps = int(round(float(horizon_s) / float(dt)))
    if max_steps is None:
        max_steps = int(round(float(DEFAULT_HORIZON_S) / float(dt)))

    return Task(
        name="default",
        x_trim=x_trim,
        u_trim=u_trim,
        K=K,
        feedback_state_indices=feedback_state_indices,
        obs_indices=obs_indices,
        obs_scale=obs_scale,
        obs_builder=obs_builder,
        obs_dim=obs_dim,
        init_sampler=_default_init_sampler(x_trim),
        disturbance=None,
        task_id=0,
        param_vector=np.zeros(0, dtype=float),
        max_steps=max_steps,
        obs_mode=obs_mode,
    )


def sample_task(rng, base_task, mode="meta"):
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    x_trim = np.array(base_task.x_trim, dtype=float)
    u_trim = np.array(base_task.u_trim, dtype=float)

    if mode == "meta":
        offsets = {
            1: rng.uniform(-0.01, 0.01),  # beta
            2: rng.uniform(-0.01, 0.01),  # alpha
            6: rng.uniform(-0.02, 0.02),  # phi
            7: rng.uniform(-0.02, 0.02),  # theta
        }
        for idx, val in offsets.items():
            x_trim[idx] += val

        def disturbance(t, x, u, rng_inner):
            if 2.0 <= t <= 3.0:
                gust = rng_inner.normal(0.0, 0.002, size=4)
                return gust
            return np.zeros(4, dtype=float)

    else:
        disturbance = None

    return Task(
        name=f"{base_task.name}_{mode}",
        x_trim=x_trim,
        u_trim=u_trim,
        K=base_task.K,
        feedback_state_indices=base_task.feedback_state_indices,
        obs_indices=base_task.obs_indices,
        obs_scale=base_task.obs_scale,
        obs_builder=base_task.obs_builder,
        obs_dim=base_task.obs_dim,
        init_sampler=_default_init_sampler(x_trim),
        disturbance=disturbance,
        task_id=base_task.task_id + 1,
        param_vector=np.zeros(0, dtype=float),
        max_steps=base_task.max_steps,
        obs_mode=base_task.obs_mode,
    )
