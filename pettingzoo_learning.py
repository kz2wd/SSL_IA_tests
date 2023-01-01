from pettingzoo.butterfly import pistonball_v6


def policy(obs, agent):
    return env.action_space(agent).sample()


env = pistonball_v6.env()

env.reset()
for agent in env.agent_iter(10):
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent)
    print(action)
    env.step(action)
