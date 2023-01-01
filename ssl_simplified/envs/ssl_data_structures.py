from __future__ import annotations
from enum import Enum
from math import sqrt
from typing import OrderedDict, Union

import numpy as np

# WARNING : Robot speed is the distance in mm that a robot can travel during one second
MAX_ROBOT_SPEED = 1000

BALL_HIT_DISTANCE = 50


class TeamColor(Enum):
    YELLOW = 0
    BLUE = 1


class Position:
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y

    def __add__(self, other: Position) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: Position):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: Position) -> Position:
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> Position:
        return Position(self.x * other, self.y * other)

    def __str__(self) -> str:
        return f"({round(self.x, 2)}, {round(self.y, 2)})"

    def __iter__(self) -> iter:
        return iter((self.x, self.y))

    def to_np_array(self) -> np.array:
        return np.array(list(map(float, self)), dtype=np.float32)

    def distance(self, position: Position) -> float:
        return sqrt((position.x - self.x) ** 2 + (position.y - self.y) ** 2)

    def get_normalized(self, factor: float = 1.0):
        norm = sqrt(self.x ** 2 + self.y ** 2)
        return Position(self.x * factor / norm, self.y * factor / norm)


class Size:
    def __init__(self, width: float, height: float):
        self.width: float = width
        self.height: float = height

    def __mul__(self, factor: float):
        return Size(self.width * factor, self.height * factor)

    def __iter__(self) -> iter:
        return iter((self.width, self.height))


    @property
    def center(self) -> Position:
        return Position(self.width / 2, self.height / 2)


class Area:
    def __init__(self, position: Position, size: Size):
        self.position: Position = position
        self.size: Size = size

    @property
    def center(self) -> Position:
        return self.size.center + self.position

    def __contains__(self, position: Position):
        return self.position.x <= position.x <= self.position.x + self.size.width and \
               self.position.y <= position.y <= self.position.y + self.size.height


class Terrain:
    def __init__(self, terrain_size: Size, goal_keeper_area: Size, goal_size: Size, blue_team: Team, yellow_team: Team):
        self.size: Size = terrain_size
        self.area: Area = Area(Position(0, 0), terrain_size)
        self.goal_keeper_areas = {
            TeamColor.BLUE: Area(Position(0, terrain_size.height - goal_keeper_area.height / 2), goal_keeper_area),
            TeamColor.YELLOW: Area(Position(terrain_size.width - goal_keeper_area.width,
                                            terrain_size.height - goal_keeper_area.height / 2), goal_keeper_area)}
        self.goals: dict[TeamColor: Area] = {
            TeamColor.BLUE: Area(Position(-1000, terrain_size.height / 2 - goal_size.height / 2), goal_size),
            TeamColor.YELLOW: Area(Position(terrain_size.width, terrain_size.height / 2 - goal_size.height / 2),
                                   goal_size)}
        self.ball: Ball = Ball(Size(43, 43), terrain_size.center)
        self.teams: dict[TeamColor: Team] = {TeamColor.BLUE: blue_team, TeamColor.YELLOW: yellow_team}
        self.scores: dict[TeamColor: int] = {TeamColor.BLUE: 0, TeamColor.YELLOW: 0}

    def update_game_state(self) -> None:
        if self.ball.position in self.goals[TeamColor.BLUE]:
            self.scores[TeamColor.YELLOW] += 1
        elif self.ball.position in self.goals[TeamColor.YELLOW]:
            self.scores[TeamColor.BLUE] += 1

        if self.ball.position not in self.area:
            self.ball.position = self.area.center

    @staticmethod
    def _print_team_line(y: int, robots: list[Robot], symbol: str) -> str:
        line_str: str = ""
        for x in range(9):
            case_area: Area = Area(Position(x * 1000, y * 1000), Size(1000, 1000))
            robots_amount = len(list(filter(lambda robot: robot.position in case_area, robots)))
            if robots_amount > 0:
                line_str += f"|{robots_amount} : {symbol}|"
            else:
                line_str += "|     |"
        line_str += "\n"
        return line_str

    def __str__(self) -> str:
        terrain_str: str = "-" * (9 * 7) + "\n"
        for y in range(6):
            for x in range(9):
                case_area: Area = Area(Position(x * 1000, y * 1000), Size(1000, 1000))
                if self.ball.position in case_area:
                    terrain_str += f"|Ball |"
                else:
                    terrain_str += "|     |"
            terrain_str += "\n"

            terrain_str += self._print_team_line(y, self.teams[TeamColor.BLUE].robots, "B")
            terrain_str += self._print_team_line(y, self.teams[TeamColor.YELLOW].robots, "Y")
            terrain_str += "-" * (9 * 7) + "\n"
        return terrain_str

    def execute_action(self, action: OrderedDict[str, Union[float, np.array]], player_color: TeamColor, robot_id: int,
                       delta_time: float):
        robot: Robot = self.teams[player_color].robots[robot_id]
        move_entity(robot, action["robot_orientation"][0], action["robot_speed"][0], delta_time)

        if action["do_robot_kick"] and self.ball.position.distance(robot.position) < BALL_HIT_DISTANCE:
            move_entity(self.ball, action["robot_orientation"][0], action["robot_speed"][0], delta_time)


class Team:
    def __init__(self, color: TeamColor, robot_starting_positions: list[Position]):
        self.color: TeamColor = color
        self.robots = [Robot(Size(180, 180), position, color, i) for i, position in enumerate(robot_starting_positions)]


class Entity:
    def __init__(self, size: Size, position: Position):
        self.size: Size = size
        self.position: Position = position


class Ball(Entity):
    pass


class Robot(Entity):
    def __init__(self, size: Size, position: Position, team_color: TeamColor, id: int):
        super().__init__(size, position)
        self.team_color: TeamColor = team_color
        self.id: int = id


def load_divB_configuration() -> Terrain:
    goal_size: Size = Size(1000, 2000)
    blue_team = Team(TeamColor.BLUE, [Position(500, 3000)] + [Position(2225, 500 + 1000 * i) for i in range(5)])
    yellow_team = Team(TeamColor.YELLOW, [Position(8500, 3000)] + [Position(6725, 500 + 1000 * i) for i in range(5)])

    return Terrain(Size(9000, 6000), goal_size, Size(1000, 1000), blue_team, yellow_team)


def move_entity(entity: Entity, orientation_in_degree: float, speed: float, delta_time: float) -> None:
    rad_angle: float = np.deg2rad(orientation_in_degree)
    movement_vector: Position = Position(np.cos(rad_angle), np.sin(rad_angle))  # Get normalised vector
    distance_traveled = MAX_ROBOT_SPEED * speed * delta_time
    entity.position += movement_vector * distance_traveled
