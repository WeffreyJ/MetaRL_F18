import numpy as np

from f18sim import F18Sim
from .normalization import normalize_obs

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None


class F18EnvBase(gym.Env if gym else object):
    def __init__(self, dt, task, seed=None, log_dir=None):
        self._dt = float(dt)
        self._task = task
        self._rng = np.random.default_rng(seed)
        self._log_dir = log_dir
        self._sim = F18Sim()
        self._sim.set_step_size(self._dt)
        self._t = 0.0
        self._k = 0
        self._last_state = None
        if gym and spaces:
            obs_dim = self._task.obs_dim if self._task.obs_dim is not None else len(self._task.obs_indices)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float64
            )

    def set_task(self, task):
        self._task = task

    def set_step_size(self, dt):
        self._dt = float(dt)
        self._sim.set_step_size(self._dt)

    def get_state12(self):
        return np.asarray(self._sim.get_state(), dtype=float)

    def set_state12(self, x):
        self._sim.set_state(np.asarray(x, dtype=float))

    def _observe(self, x):
        if self._task.obs_builder is not None:
            return np.asarray(self._task.obs_builder(x, self._task.x_trim), dtype=float)
        return normalize_obs(x, self._task.x_trim, self._task.obs_indices, self._task.obs_scale)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        x0 = self._task.init_sampler(self._rng)
        self._sim.reset(x0)
        self._t = 0.0
        self._k = 0
        self._last_state = self.get_state12()
        obs = self._observe(self._last_state)
        info = {"t": self._t}
        return obs, info

    def step(self, action):
        raise NotImplementedError

    def _apply_disturbance(self, u):
        if self._task.disturbance is None:
            return u
        return u + self._task.disturbance(self._t, self.get_state12(), u, self._rng)

    def close(self):
        if self._sim:
            self._sim.close()
            self._sim = None
