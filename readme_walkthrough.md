# F-18 Kernel + RL Environment Walkthrough

This document is a **detailed, end-to-end walkthrough** of the repository at:

`f18_kernel_codegen_grt_rtw/`

It covers:
- what the repo contains
- how the model is wired
- how to run open-loop tests
- how to run LQR closed-loop tests
- how to train/evaluate RL policies
- where logs, rollouts, and plots live
- how to diagnose common issues

It is written so a Python/C++ newcomer can follow it without prior context.

---

## 1) Repository Structure (what lives where)

Top-level (root):
- `f18_kernel_codegen.cpp/.h/...` — **Simulink-generated C++** model code (do not edit; regen will overwrite).
- `f18_wrap.mk` — Makefile to build a Python-callable shared library (`libf18sim.dylib`).
- `custom/` — **your wrapper source lives here** (safe from regen). Typical file: `custom/f18_wrap.cpp`.
- `python/` — all Python scripts and packages.
- `runs/` — RL training outputs and rollouts (created by scripts).
- `readme_walkthrough.md` — this file.

Key folders:
- `python/f18sim/`
  - `wrapper.py` — ctypes wrapper for the C++ shared library.
  - `__init__.py` — exposes `F18Sim` class.
  - `libf18sim.dylib` — compiled shared library (produced by `make -f f18_wrap.mk`).
- `python/rl_env/`
  - `f18_env_base.py` — common environment base class.
  - `f18_env_drive_to_trim.py` — absolute-control RL env (drive-to-trim).
  - `f18_env_residual_lqr.py` — residual RL env (LQR + delta action).
  - `task.py` — trim/task definitions + observation modes.
  - `reward.py` — reward shaping for RL.
  - `termination.py` — safety/termination logic.
  - `utils_csv_logger.py` — rollout logging utility.
  - `scripts/` — training, rollout, evaluation, plotting scripts.

Other helpful scripts in `python/`:
- `run_f18_openloop.py` — open-loop sim runner with CSV output.
- `experiment.py` — “MATLAB-style” editable open-loop script.
- `plot_states.py` — plot open-loop CSV (quick or full grid).
- `run_f18_lqr_closedloop.py` — LQR closed-loop with CSV output.
- `plot_states_all.py` — plot all 12 states from LQR logs.
- `linmodel_f18.py`, `lqr_design.py`, `controller_lqr.py` — linear model + LQR control.

---

## 2) Core Model Interface (state/action, units)

### State ordering (12)
Always in this order everywhere in Python:
```
[V, beta, alpha, p, q, r, phi, theta, psi, pN, pE, h]
```
Units (from MATLAB conventions):
- V: ft/s
- beta, alpha, phi, theta, psi: rad
- p, q, r: rad/s
- pN, pE: ft
- h: ft

### Action ordering (4)
Always in this order everywhere in Python:
```
[ail, rud, elev, T]
```
Units:
- surfaces in radians
- thrust in lbf (model-specific; used as numeric input)

### Important safety note
The **Simulink model already includes actuator saturation and rate limits.**
Python only applies light scaling and additional penalties to guide RL; it does not re-implement the plant saturation logic.

---

## 3) Building the shared library (C++ → Python)

If `python/f18sim/libf18sim.dylib` is missing or out of date:

```bash
cd /Users/jeffreywalker/Downloads/OPENLOOP_MODELS/f18_kernel_codegen_grt_rtw
make -f f18_wrap.mk clean
make -f f18_wrap.mk
```

Notes:
- `f18_wrap.mk` uses MATLAB headers; edit `MATLAB_ROOT` inside the makefile if your MATLAB path differs.
- Wrapper source is in `custom/f18_wrap.cpp`. **Do not edit generated C++ files.**

Sanity check: load the wrapper
```bash
PYTHONPATH=python python3 -c "from f18sim import F18Sim; s=F18Sim(); print(s.get_num_states(), s.get_num_actions()); s.close()"
```
Expected: `12 4`

---

## 4) Open-loop simulation (Python only)

### Option A — quick CLI runner
File: `python/run_f18_openloop.py`

Example:
```bash
PYTHONPATH=python python3 python/run_f18_openloop.py --sim_time 2 --dt 0.02 --out state_log.csv
```
- Writes `state_log.csv` with header:
  `t,V,beta,alpha,p,q,r,phi,theta,psi,pN,pE,h`
- Uses fixed initial conditions in the script.
- Accepts `--regime <name>` to load from `python/regimes.py` if present.
- `--repeat N` runs determinism checks.

### Option B — MATLAB-style editable script
File: `python/experiment.py`

Edit the top block (dt, sim_time, x0, u), then run:
```bash
PYTHONPATH=python python3 python/experiment.py
```
Optional regime override:
```bash
PYTHONPATH=python python3 python/experiment.py --regime Init_Dramatic
```

### Plotting open-loop outputs
File: `python/plot_states.py`

All-states grid (default):
```bash
PYTHONPATH=python python3 python/plot_states.py state_log.csv
```
Quick view:
```bash
PYTHONPATH=python python3 python/plot_states.py state_log.csv --mode quick
```
Save figures:
```bash
PYTHONPATH=python python3 python/plot_states.py state_log.csv --save states.png
```

---

## 5) LQR closed-loop tests

### Linear model data
- `python/linmodel_f18.py` contains A/B matrices and trim (x_trim/u_trim).

### LQR controller
- `python/lqr_design.py` computes K using CARE.
- `python/controller_lqr.py` defines `lqr_controller(x)`.

### Closed-loop run
File: `python/run_f18_lqr_closedloop.py`
```bash
PYTHONPATH=python python3 python/run_f18_lqr_closedloop.py --sim_time 30 --dt 0.02 --out lqr_log.csv
```

### Plot all states (from LQR log)
File: `python/plot_states_all.py`
```bash
PYTHONPATH=python python3 python/plot_states_all.py lqr_log.csv
```

---

## 6) RL Environment (core concepts)

### Environments
- `F18EnvDriveToTrim` (absolute control) — file: `python/rl_env/f18_env_drive_to_trim.py`
- `F18EnvResidualLqr` (residual control) — file: `python/rl_env/f18_env_residual_lqr.py`

### Action space
- Always normalized to `[-1, 1]` in Python.
- DriveToTrim maps to `u = u_trim + action * action_scale`.
- ResidualLqr maps to `u = u_nom + action * delta_u_max`.

### Observation modes
Defined in `python/rl_env/task.py`:
- `inner8`: 8D normalized state errors `[V, beta, alpha, p, q, r, phi, theta]`.
- `inner10_vh`: 10D = inner8 errors + dv + dh features (explicitly built, no duplicate indices).
- `inner11_vhE`: inner10_vh + energy error feature `dE`.
- `full12`: 12D normalized state errors for all state variables.

### Reward shaping (DriveToTrim)
In `python/rl_env/reward.py` + env:
- Base error cost + effort cost + slew cost.
- Trend shaping: `trend_reward = lambda_trend * clip(err_prev - err)`.
- Barrier cost near envelope bounds (steeper near boundary).
- Drift penalty after first entry into tolerance window (ramps over time).

### Termination
In `python/rl_env/termination.py`:
- NaN/Inf immediate termination.
- Envelope termination: `theta_band`, `h_band`, `v_band`, alpha/beta bounds.
- `terminate_on_success` can be False for long-horizon runs.

### Success definitions
- `hold_success`: sustained in‑tolerance for required time (seconds).
- `final_window_good`: last 10s window is mostly in tolerance.
- `true_success`: safe and (hold_success OR final_window_good).

---

## 7) RL Scripts (train / rollout / eval)

### A) Smoke test
```bash
PYTHONPATH=python python3 -m rl_env.scripts.env_smoke_test
```

### B) Rollout a policy (or random)
File: `python/rl_env/scripts/rollout_env.py`

Random rollout:
```bash
PYTHONPATH=python python3 -m rl_env.scripts.rollout_env \
  --env drive_to_trim --dt 0.02 --sim_time 10 --seed 0 \
  --policy random --obs_mode inner10_vh --out tmp_rand.csv
```

SB3 rollout (explicit model):
```bash
PYTHONPATH=python python3 -m rl_env.scripts.rollout_env \
  --env drive_to_trim --dt 0.02 --sim_time 300 --seed 0 \
  --policy sb3 --algo ppo --model_path runs/<RUN>/checkpoints/ppo_step1000000.zip \
  --deterministic --obs_mode inner10_vh --out runs/<RUN>/rollout_step1000000.csv
```

SB3 rollout (auto model from log_dir):
```bash
PYTHONPATH=python python3 -m rl_env.scripts.rollout_env \
  --env drive_to_trim --dt 0.02 --sim_time 300 --seed 0 \
  --policy sb3 --algo ppo --log_dir runs/<RUN> --deterministic \
  --out runs/<RUN>/rollout_best.csv
```

### C) Plot rollout CSV
File: `python/rl_env/scripts/plot_env_rollout.py`
```bash
PYTHONPATH=python python3 -m rl_env.scripts.plot_env_rollout runs/<RUN>/rollout_best.csv
```
Generates:
- `<basename>_states.png`
- `<basename>_actions.png`
- `<basename>_rates.png`
- `<basename>_reward.png`

### D) Training
File: `python/rl_env/scripts/train_sb3.py`

Short sanity run:
```bash
PYTHONPATH=python python3 -m rl_env.scripts.train_sb3 \
  --env drive_to_trim --algo ppo --obs_mode inner10_vh \
  --dt 0.02 --total_steps 50000 --seed 0 \
  --log_dir runs/ppo_inner10vh_sanity --long_horizon 60 --plot
```

Long-horizon run (300s):
```bash
PYTHONPATH=python python3 -m rl_env.scripts.train_sb3 \
  --env drive_to_trim --algo ppo --obs_mode inner10_vh \
  --dt 0.02 --total_steps 5000000 --seed 0 \
  --log_dir runs/ppo_inner10vh_300s --long_horizon 300 --plot
```

Outputs:
- `run_config.json` and `run_config.txt`
- `model.zip`, `best_model.zip`, `checkpoints/*.zip`
- `episode_log.csv`
- plots: rolling success, hold, barrier, drift, etc.

### E) Evaluate checkpoints
File: `python/rl_env/scripts/eval_checkpoints.py`
```bash
PYTHONPATH=python python3 -m rl_env.scripts.eval_checkpoints \
  --log_dir runs/<RUN> --env drive_to_trim --algo ppo \
  --dt 0.02 --sim_time 300 --seed 0 --deterministic
```
Ranks models by: `true_success → final_window_good → hold_success → time_in_tol → max_consec → return`.

---

## 8) Logs and artifacts (where to find things)

Training run folder (e.g. `runs/ppo_inner10vh_300s/`):
- `run_config.json` — used to auto‑resolve obs_mode for eval/rollout.
- `episode_log.csv` — per‑episode metrics (append‑only schema).
- `model.zip` — final model.
- `best_model.zip` — model selected by hold‑quality metrics.
- `checkpoints/` — step‑based snapshots.

Rollout CSVs:
- Any CSV produced by `rollout_env.py` or open‑loop scripts.
- Plot script saves PNGs beside the CSV.

---

## 9) Common errors and fixes

**Error: Obs mismatch (env obs_dim != model obs_dim)**
- Use `--obs_mode` explicitly when rolling out or evaluating, OR
- Ensure `run_config.json` exists in the run folder.

**Error: libf18sim.dylib not found**
- Build with `make -f f18_wrap.mk clean && make -f f18_wrap.mk`.

**Error: matplotlib missing**
- Install: `python3 -m pip install --user matplotlib`

**Error: stable-baselines3 missing**
- Install: `python3 -m pip install --user stable-baselines3 gymnasium torch`

---

## 10) Quick “start here” flow for new users

1) Build the library (only once):
```bash
cd /Users/jeffreywalker/Downloads/OPENLOOP_MODELS/f18_kernel_codegen_grt_rtw
make -f f18_wrap.mk
```

2) Run a small open‑loop test:
```bash
PYTHONPATH=python python3 python/run_f18_openloop.py --sim_time 2 --dt 0.02 --out state_log.csv
PYTHONPATH=python python3 python/plot_states.py state_log.csv
```

3) Run a small RL smoke test:
```bash
PYTHONPATH=python python3 -m rl_env.scripts.env_smoke_test
```

4) Train a small RL run:
```bash
PYTHONPATH=python python3 -m rl_env.scripts.train_sb3 \
  --env drive_to_trim --algo ppo --obs_mode inner10_vh --dt 0.02 \
  --total_steps 50000 --seed 0 --log_dir runs/ppo_inner10vh_sanity --long_horizon 60 --plot
```

5) Evaluate and roll out the best model:
```bash
PYTHONPATH=python python3 -m rl_env.scripts.eval_checkpoints --log_dir runs/ppo_inner10vh_sanity --env drive_to_trim --algo ppo --dt 0.02 --sim_time 60 --seed 0 --deterministic
PYTHONPATH=python python3 -m rl_env.scripts.rollout_env --env drive_to_trim --dt 0.02 --sim_time 60 --seed 0 --policy sb3 --log_dir runs/ppo_inner10vh_sanity --deterministic --out runs/ppo_inner10vh_sanity/rollout_best.csv
PYTHONPATH=python python3 -m rl_env.scripts.plot_env_rollout runs/ppo_inner10vh_sanity/rollout_best.csv
```

---

## 11) Where to tune behavior (advanced)

- `f18_env_drive_to_trim.py`:
  - `self._barrier_k`, `self._barrier_shape_gain`, `self._barrier_shape_p`
  - `self._k_drift`, `self._drift_ramp_s`
  - `self._lambda_theta`, `self._lambda_V`, `self._lambda_h`
  - `self._success_count_needed` (derived from horizon)

- `task.py`:
  - Observation modes and scales.
  - `DEFAULT_HORIZON_S`.

- `termination.py`:
  - Envelope limits and termination reasons.

- `train_sb3.py`:
  - Logging, plotting, and best‑model selection logic.

---

## 12) One‑line checklist before training

- `libf18sim.dylib` exists and loads
- `PYTHONPATH=python` is set on all commands
- obs_mode matches the model you evaluate
- `run_config.json` exists in run folder (auto‑generated by train script)

---

If you want this walkthrough mirrored as a shorter “quickstart” or as a doc in `python/rl_env/`, say the word and I’ll generate it.
