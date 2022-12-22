from typing import Optional, Union, List, Tuple

import gym
from gym.core import RenderFrame, ActType, ObsType
from gym import Space

from SSL_environment import Terrain, TeamColor, Action, MoveTo, Position, load_divB_configuration


class DecisionMaker:
    def __init__(self, team_color: TeamColor):
        self.team_color: TeamColor = team_color

    def get_decision(self, terrain: Terrain) -> list[Action]:
        return [MoveTo(robot, robot.position + Position(10000, 0)) for robot in terrain.teams[self.team_color].robots]


class SSL_Environment(gym.Env):

    def __init__(self, delta_time: float):
        super().__init__()
        self.delta_time = delta_time
        self.terrain: Terrain = None
        self.blue_player: DecisionMaker = None
        self.yellow_player: DecisionMaker = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        blue_actions = self.blue_player.get_decision(self.terrain)
        yellow_actions = self.yellow_player.get_decision(self.terrain)

        for b, y in zip(blue_actions, yellow_actions):
            b.execute(self.delta_time)
            y.execute(self.delta_time)

        self.terrain.update_game_state()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terrain: Terrain = load_divB_configuration()
        self.blue_player = DecisionMaker(TeamColor.BLUE)
        self.yellow_player = DecisionMaker(TeamColor.YELLOW)


if __name__ == "__main__":
    pass

