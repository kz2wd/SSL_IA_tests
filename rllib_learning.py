import gym
import gym_example


def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward


env = gym.make("example-v0")


history = []
for _ in range(10000):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))


import os
import shutil
chkpt_root = "tmp/exa"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

import ray
ray.init(ignore_reinit_error=True)

from ray.tune.registry import register_env
from gym_example.envs.example_env import Example_v0
select_env = "example-v0"
register_env(select_env, lambda config: Example_v0())

import ray.rllib.agents.ppo as ppo
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=select_env)


status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 5
for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)
    print(status.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            chkpt_file
            ))
