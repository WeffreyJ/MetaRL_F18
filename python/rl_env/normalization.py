import numpy as np


def normalize_obs(x, x_trim, obs_indices, obs_scale):
    x = np.asarray(x, dtype=float)
    x_trim = np.asarray(x_trim, dtype=float)
    obs_indices = np.asarray(obs_indices, dtype=int)
    obs_scale = np.asarray(obs_scale, dtype=float)
    obs = x[obs_indices]
    obs_trim = x_trim[obs_indices]
    return (obs - obs_trim) / obs_scale
