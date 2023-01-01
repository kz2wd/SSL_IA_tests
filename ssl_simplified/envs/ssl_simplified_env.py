import functools
from collections import OrderedDict
from copy import copy
from typing import Optional, Union

import numpy as np
import pygame
from gymnasium import spaces
from gym.core import ActType, RenderFrame

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from ssl_simplified.envs.ssl_data_structures import Terrain, TeamColor, load_divB_configuration


def _robot_action_space():
    return spaces.Dict({
        "robot_speed": spaces.Box(0, 1),
        "robot_orientation": spaces.Box(0, 360),
        "do_robot_kick": spaces.Discrete(2),
        # "do_robot_drible": spaces.Discrete(2) ADD THIS WHEN IMPLEMENTING DRIBLE :D
    })


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.OrderEnforcingWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = SSL_Environment(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class SSL_Environment(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None,
                 delta_time: float = 1, max_steps: int = 100_000):
        self.delta_time = delta_time
        self.terrain: Terrain = load_divB_configuration()

        self.max_steps = max_steps
        self.current_step = 0

        self.possible_agents = [TeamColor.BLUE, TeamColor.YELLOW]

        # self.spec.max_episode_steps = 100  # guessing a 'good' value
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.display_resolution = 0.1
        self.clock = None

    def step(self, actions: ActType):

        for i, (blue_action, yellow_action) in enumerate(zip(actions[TeamColor.BLUE], actions[TeamColor.YELLOW])):
            self.terrain.execute_action(blue_action, TeamColor.BLUE, i, self.delta_time)
            self.terrain.execute_action(yellow_action, TeamColor.YELLOW, i, self.delta_time)

        self.terrain.update_game_state()
        self.current_step += 1
        observations = {color: self._get_obs(color) for color in TeamColor}
        rewards = {color: self._get_reward(color) for color in TeamColor}
        terminations = {color: False for color in TeamColor}
        truncations = {color: self.max_steps >= self.current_step for color in TeamColor}
        if any(truncations):
            self.agents = []
        infos = {color: self._get_info(color) for color in TeamColor}
        return observations, rewards, terminations, truncations, infos

    def _get_reward(self, agent):
        return self._get_info(agent.opponent())["ball_distance_to_enemy_goal"] \
        + self.terrain.scores[agent] * 1000 \
        - self.terrain.scores[agent.opponent()] * 1000

    def render(self) -> Optional[Union[RenderFrame, list[RenderFrame]]]:
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(tuple(self.terrain.size * self.display_resolution))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(tuple(self.terrain.size * self.display_resolution))
        canvas.fill((51, 153, 51))
        # We have our terrain
        for robot in self.terrain.teams[TeamColor.BLUE].robots:
            pygame.draw.circle(canvas, (0, 102, 255), tuple(robot.position * self.display_resolution),
                               robot.size.width * self.display_resolution)
        for robot in self.terrain.teams[TeamColor.YELLOW].robots:
            pygame.draw.circle(canvas, (255, 255, 0), tuple(robot.position * self.display_resolution),
                               robot.size.width * self.display_resolution)
        pygame.draw.circle(canvas, (255, 0, 102), tuple(self.terrain.ball.position * self.display_resolution),
                           self.terrain.ball.size.width * self.display_resolution)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def _get_info(self, agent: TeamColor):
        return {"score": self.terrain.scores[agent],
                "ball_distance_to_enemy_goal":
                    self.terrain.ball.position.distance(self.terrain.goals[agent.opponent()].center)}

    def _get_obs(self, agent: TeamColor):
        return OrderedDict(
            [("Ally_robots_pos", tuple(
                robot.position.to_np_array() for robot in self.terrain.teams[agent].robots)),
             ("Enemy_robots_pos", tuple(
                 robot.position.to_np_array() for robot in self.terrain.teams[agent.opponent()].robots)),
             ("Ball_pos", self.terrain.ball.position.to_np_array())]
        )

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.terrain: Terrain = load_divB_configuration()

        return {color: self._get_obs(color) for color in TeamColor}

    def _robot_observation_space(self):
        return spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                      self.terrain.size.height]),
                          dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: TeamColor):
        return spaces.Dict(
            {
                "Ally_robots_pos": spaces.Tuple(self._robot_observation_space() for _ in range(
                    len(self.terrain.teams[agent].robots))),
                "Enemy_robots_pos": spaces.Tuple(self._robot_observation_space() for _ in range(
                    len(self.terrain.teams[agent.opponent()].robots))),
                "Ball_pos": spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                                   self.terrain.size.height]),
                                       dtype=np.float32)
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Tuple(_robot_action_space() for _ in range(len(self.terrain.teams[agent].robots)))


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(SSL_Environment(), num_cycles=1_000_000)
