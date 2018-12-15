from collections import defaultdict
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Tuple

import numpy as np


class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass(eq=True, frozen=True)
class State:
    """ Class for keeping track of state within the maze.
    """
    x: int
    y: int


@dataclass(eq=True, frozen=True)
class Move:
    """ Class for keeping track of moving within the maze.
    """
    dx: int
    dy: int


class Maze:
    free_space = 0
    wall_location = 1
    robot_location = 2

    action_space = {
        Action.UP: Move(0, -1),
        Action.DOWN: Move(0, 1),
        Action.LEFT: Move(-1, 0),
        Action.RIGHT: Move(1, 0),
    }

    def __init__(self):
        self.reset()
        self.construct_allowed_states()

    def __str__(self) -> str:
        return str(self.maze)

    def reset(self) -> None:
        self.maze = np.array([
            [2, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ])
        self.robot_position = State(0, 0)
        self.steps = 0

    def is_allowed_move(self, state: State, action: Action) -> bool:
        x, y = state.x, state.y
        x += Maze.action_space[action].dx
        y += Maze.action_space[action].dy
        if y < 0 or x < 0 or y > 5 or x > 5:
            return False
        return True if self.maze[y, x] != Maze.wall_location else False

    def construct_allowed_states(self) -> Dict[State, List[Action]]:
        allowed_states = defaultdict(list)
        for y, row in enumerate(self.maze):
            for x, _ in enumerate(row):
                if self.maze[(y, x)] != Maze.wall_location:
                    allowed_state = State(x, y)
                    for action in Maze.action_space:
                        if self.is_allowed_move(allowed_state, action):
                            allowed_states[allowed_state].append(action)
        self.allowed_states = allowed_states

    def update_maze(self, action: Action) -> None:
        x, y = self.robot_position.x, self.robot_position.y
        self.maze[y, x] = Maze.free_space
        x += Maze.action_space[action].dx
        y += Maze.action_space[action].dy
        self.robot_position = State(x, y)
        self.maze[y, x] = Maze.robot_location
        self.steps += 1

    def is_complete(self) -> bool:
        return self.robot_position == State(5, 5)

    def get_state_and_reward(self) -> Tuple[State, int]:
        reward = self.give_reward()
        return self.robot_position, reward

    def give_reward(self) -> int:
        return 0 if self.robot_position == State(5, 5) else -1
