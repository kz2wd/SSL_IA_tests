from gym.envs.registration import register

register(
    id="ssl_simplified-v0",
    entry_point="ssl_simplified.envs:SSL_Environment",
)
