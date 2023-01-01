from ssl_simplified import ssl_simplified_env_v0


def policy(obs, agent):
    return env.action_space(agent).sample()


env = ssl_simplified_env_v0.env()

env.reset()
for agent in env.agent_iter(10):
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent)
    print(action)
    env.step(action)
