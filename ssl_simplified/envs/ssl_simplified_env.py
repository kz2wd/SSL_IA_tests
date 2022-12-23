from collections import OrderedDict
from typing import Optional, Union

import gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame

from ssl_simplified.envs.ssl_data_structures import Terrain, TeamColor, load_divB_configuration, Action, Position, \
    MoveTo, Wait


class DecisionMaker:
    def __init__(self, team_color: TeamColor):
        self.team_color: TeamColor = team_color

    def get_decision(self, terrain: Terrain) -> list[Action]:
        return [MoveTo(robot, robot.position + Position(10000, 0)) for robot in terrain.teams[self.team_color].robots]


class WaitingDecisionMaker(DecisionMaker):
    def get_decision(self, terrain: Terrain) -> list[Action]:
        return [Wait(robot) for robot in terrain.teams[self.team_color].robots]


class SSL_Environment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, enemy_ia: DecisionMaker = WaitingDecisionMaker(TeamColor.YELLOW), render_mode=None,
                 delta_time: float = 1):
        self.delta_time = delta_time
        self.terrain: Terrain = load_divB_configuration()
        self.blue_player = DecisionMaker(TeamColor.BLUE)
        self.yellow_player = enemy_ia

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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        # blue_actions = self.blue_player.get_decision(self.terrain)

        print(action)
        input()
        blue_actions = []

        yellow_actions = self.yellow_player.get_decision(self.terrain)

        for b, y in zip(blue_actions, yellow_actions):
            b.execute(self.delta_time)
            y.execute(self.delta_time)

        self.terrain.update_game_state()
        info = self._get_info()
        reward = -info["ball_distance_to_yellow_goal"] \
                 + self.terrain.scores[TeamColor.BLUE] * 1000 \
                 - self.terrain.scores[TeamColor.YELLOW] * 1000
        # Observation, reward, terminated?, truncated?, info
        return self._get_obs(), reward, False, False, info

    def render(self) -> Optional[Union[RenderFrame, list[RenderFrame]]]:
        print(self.terrain)

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terrain: Terrain = load_divB_configuration()
        self.blue_player = DecisionMaker(TeamColor.BLUE)
        self.yellow_player = DecisionMaker(TeamColor.YELLOW)

        return self._get_obs(), self._get_info()
