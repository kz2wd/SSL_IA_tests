from collections import OrderedDict
from typing import Optional, Union

import gym
import numpy as np
import pygame
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame

from ssl_simplified.envs.ssl_data_structures import Terrain, TeamColor, load_divB_configuration


class SSL_Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None,
                 delta_time: float = 1):
        self.delta_time = delta_time
        self.terrain: Terrain = load_divB_configuration()
        self.blue_player = None  # OUR IA
        self.yellow_player = None  # Enemy

        self.observation_space = spaces.Dict(
            {
                "Ally_robots_pos": spaces.Sequence(spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                                                          self.terrain.size.height]),
                                                              dtype=np.float32)),
                "Enemy_robots_pos": spaces.Sequence(spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                                                           self.terrain.size.height]),
                                                               dtype=np.float32)),
                "Ball_pos": spaces.Box(np.array([0, 0]), np.array([self.terrain.size.width,
                                                                   self.terrain.size.height]),
                                       dtype=np.float32)
            }
        )
        self.action_space = spaces.Sequence(spaces.Dict({
            "robot_speed": spaces.Box(0, 1),
            "robot_orientation": spaces.Box(0, 360),
            "do_robot_kick": spaces.Discrete(2),
            # "do_robot_drible": spaces.Discrete(2) ADD THIS WHEN IMPLEMENTING DRIBLE :D
        }))

        # self.spec.max_episode_steps = 100  # guessing a 'good' value
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.display_resolution = 0.1
        self.clock = None

    def step(self, actions: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        # blue_actions = self.blue_player.get_decision(self.terrain)

        # yellow_actions = self.yellow_player.get_decision(self.terrain)
        # Lets simplify by saying that yellow never play

        for i, action in enumerate(actions):
            if i > 5:
                break
            self.terrain.execute_action(action, TeamColor.BLUE, i, self.delta_time)

        self.terrain.update_game_state()
        info = self._get_info()
        reward = -info["ball_distance_to_yellow_goal"] \
                 + self.terrain.scores[TeamColor.BLUE] * 1000 \
                 - self.terrain.scores[TeamColor.YELLOW] * 1000
        # Observation, reward, terminated?, truncated?, info
        return self._get_obs(), reward, False, False, info

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
            pygame.draw.circle(canvas, (0, 102, 255), tuple(robot.position * self.display_resolution), robot.size.width * self.display_resolution)
        for robot in self.terrain.teams[TeamColor.YELLOW].robots:
            pygame.draw.circle(canvas, (255, 255, 0), tuple(robot.position * self.display_resolution), robot.size.width * self.display_resolution)
        pygame.draw.circle(canvas, (255, 0, 102), tuple(self.terrain.ball.position * self.display_resolution), self.terrain.ball.size.width * self.display_resolution)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pass

    def _get_obs(self):
        return OrderedDict(
            [("Ally_robots_pos", tuple(
                robot.position.to_np_array() for robot in self.terrain.teams[TeamColor.BLUE].robots)),
             ("Enemy_robots_pos", tuple(
                 robot.position.to_np_array() for robot in self.terrain.teams[TeamColor.YELLOW].robots)),
             ("Ball_pos", self.terrain.ball.position.to_np_array())]
        )

    def _get_info(self):
        return {"ia_score": self.terrain.scores[TeamColor.BLUE],
                "opponent_score": self.terrain.scores[TeamColor.YELLOW],
                "ball_distance_to_yellow_goal":
                    self.terrain.ball.position.distance(self.terrain.goals[TeamColor.YELLOW].center)}

    def reset(self):
        super().reset()
        self.terrain: Terrain = load_divB_configuration()

        return self._get_obs(), self._get_info()
