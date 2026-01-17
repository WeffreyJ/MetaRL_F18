from rl_env.f18_env_drive_to_trim import F18EnvDriveToTrim
from rl_env.task import load_default_task


def main():
    task = load_default_task()
    env = F18EnvDriveToTrim(dt=0.02, task=task, seed=0)
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret
    action = [0.0, 0.0, 0.0, 0.0]
    terminated = False
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    env.close()
    print("terminated:", terminated)
    print("t:", info.get("t"))


if __name__ == "__main__":
    main()
