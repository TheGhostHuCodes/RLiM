import time
from typing import List

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from environment import Maze, State
from agent import Agent


def trial(robot: Agent) -> List[int]:
    maze = Maze()
    move_history = []
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        while not maze.is_complete():
            state, _ = maze.get_state_and_reward()
            action = robot.choose_action(state, maze.allowed_states[state])
            maze.update_maze(action)
            state, reward = maze.get_state_and_reward()
            robot.update_state_history(state, reward)
            if maze.steps > 1000:
                maze.robot_position = State(5, 5)
        robot.learn()
        move_history.append(maze.steps)
        maze.reset()
    return move_history


if __name__ == '__main__':
    allowed_states = Maze().allowed_states

    robot_1 = Agent(allowed_states, alpha=0.1, random_factor=0.25)
    move_history_1 = trial(robot_1)

    robot_2 = Agent(allowed_states, alpha=0.99, random_factor=0.25)
    move_history_2 = trial(robot_2)

    plt.subplot(211)
    plt.semilogy(move_history_1, 'b-')
    plt.xlabel('episode')
    plt.ylabel('steps to solution')
    plt.legend(['alpha=0.1'])
    plt.subplot(212)
    plt.semilogy(move_history_2, 'r-')
    plt.xlabel('episode')
    plt.ylabel('steps to solution')
    plt.legend(['alpha=0.99'])

    plt.show()
