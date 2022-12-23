
from ray.rllib.algorithms.ppo import ppo
from ray.tune.registry import register_env


from ssl_simplified.envs.ssl_simplified_env import SSL_Environment


def env_creator(env_config):
    return SSL_Environment()  # return an env instance


register_env("ssl_simplified-v0", env_creator)

algo = ppo.PPO(env="ssl_simplified-v0")
