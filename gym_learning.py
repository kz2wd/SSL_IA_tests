import gym

import ssl_simplified

from gym.utils.env_checker import check_env

from ssl_simplified.envs.ssl_data_structures import TeamColor
from ssl_simplified.envs.ssl_simplified_env import WaitingDecisionMaker

enemy_ia = WaitingDecisionMaker(TeamColor.YELLOW)
env = gym.make("ssl_simplified-v0", enemy_ia=enemy_ia, render_mode="human")
check_env(env)

exit()

env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
