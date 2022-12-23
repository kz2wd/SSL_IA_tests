import gym

import ssl_simplified

from gym.utils.env_checker import check_env

from ssl_simplified.envs.ssl_data_structures import TeamColor


env = gym.make("ssl_simplified-v0", render_mode="human", delta_time=0.1)


env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        observation, info = env.reset()

env.close()
