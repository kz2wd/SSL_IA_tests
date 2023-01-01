import functools
from collections import OrderedDict
from copy import copy
from typing import Optional, Union

import gym
import numpy as np
import pygame
from gymnasium import spaces
from gym.core import ActType, ObsType, RenderFrame

from pettingzoo.utils.env import ParallelEnv

from ssl_simplified.envs.ssl_data_structures import Terrain, TeamColor, load_divB_configuration

BLUE_PLAYER = "blue"
YELLOW_PLAYER = "yellow"
PLAYER_COLORS = [BLUE_PLAYER, YELLOW_PLAYER]


def get_opponent(player: str):
    if player == BLUE_PLAYER:
        return YELLOW_PLAYER
    return BLUE_PLAYER


def player_to_teamColor(player: str) -> TeamColor:
    if player == BLUE_PLAYER:
        return TeamColor.BLUE
    return TeamColor.YELLOW


def _robot_action_space():
    return spaces.Dict({
        "robot_speed": spaces.Box(0, 1),
        "robot_orientation": spaces.Box(0, 360),
        "do_robot_kick": spaces.Discrete(2),
        # "do_robot_drible": spaces.Discrete(2) ADD THIS WHEN IMPLEMENTING DRIBLE :D
    })


class SSL_Environment(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None,
                 delta_time: float = 1, max_steps: int = 100_000):
        self.delta_time = delta_time
        self.terrain: Terrain = load_divB_configuration()

        self.max_steps = max_steps
        self.current_step = 0

        self.possible_agents = copy(PLAYER_COLORS)  # TROP chiant que les agents doivent être des STR et pas des ANY

        # self.spec.max_episode_steps = 100  # guessing a 'good' value
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.display_resolution = 0.1
        self.clock = None

    def step(self, actions: ActType):

        for i, (blue_action, yellow_action) in enumerate(zip(actions[BLUE_PLAYER], actions[YELLOW_PLAYER])):
            self.terrain.execute_action(blue_action, TeamColor.BLUE, i, self.delta_time)
            self.terrain.execute_action(yellow_action, TeamColor.YELLOW, i, self.delta_time)

        self.terrain.update_game_state()
        self.current_step += 1
        observations = {color: self._get_obs(color) for color in PLAYER_COLORS}
        rewards = {color: self._get_reward(color) for color in PLAYER_COLORS}
        terminations = {color: False for color in PLAYER_COLORS}
        truncations = {color: self.max_steps >= self.current_step for color in PLAYER_COLORS}
        infos = {color: self._get_info(color) for color in PLAYER_COLORS}
        return observations, rewards, terminations, truncations, infos

    def _get_reward(self, agent: str):
        return self._get_info(get_opponent(agent))["ball_distance_to_enemy_goal"] \
        + self.terrain.scores[player_to_teamColor(agent)] * 1000 \
        - self.terrain.scores[player_to_teamColor(get_opponent(agent))] * 1000

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

    def _get_info(self, agent: str):
        return {"score": self.terrain.scores[player_to_teamColor(agent)],
                "ball_distance_to_enemy_goal":
                    self.terrain.ball.position.distance(self.terrain.goals[player_to_teamColor(get_opponent(agent))].center)}

    def _get_obs(self, agent: str):
        return OrderedDict(
            [("Ally_robots_pos", tuple(
                robot.position.to_np_array() for robot in self.terrain.teams[player_to_teamColor(agent)].robots)),
             ("Enemy_robots_pos", tuple(
                 robot.position.to_np_array() for robot in self.terrain.teams[player_to_teamColor(get_opponent(agent))].robots)),
             ("Ball_pos", self.terrain.ball.position.to_np_array())]
        )

    def reset(self, seed=None, return_info=False, options=None):
        self.agents = copy(self.possible_agents)
        self.terrain: Terrain = load_divB_configuration()

        return {color: self._get_obs(color) for color in PLAYER_COLORS}

    def _robot_observation_space(self):
        return spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                      self.terrain.size.height]),
                          dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return spaces.Dict(
            {
                "Ally_robots_pos": spaces.Tuple(self._robot_observation_space() for _ in range(
                    len(self.terrain.teams[player_to_teamColor(agent)].robots))),
                "Enemy_robots_pos": spaces.Tuple(self._robot_observation_space() for _ in range(
                    len(self.terrain.teams[player_to_teamColor(get_opponent(agent))].robots))),
                "Ball_pos": spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                                   self.terrain.size.height]),
                                       dtype=np.float32)
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Tuple(_robot_action_space() for _ in range(len(self.terrain.teams[player_to_teamColor(agent)].robots)))


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(SSL_Environment(), num_cycles=1_000_000)
