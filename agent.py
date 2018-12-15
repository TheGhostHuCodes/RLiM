from typing import Dict, List

import numpy as np

from environment import Action, Maze, Move, State


class Agent:
    def __init__(self, states, alpha: float = 0.15,
                 random_factor: float = 0.2):
        self.state_history = [(State(0, 0), 0)]
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = Agent.init_reward(states)

    @staticmethod
    def init_reward(states: Dict[State, List[Action]]) -> Dict[State, float]:
        # return {
        #     state: np.random.uniform(low=-1.0, high=-0.1)
        #     for state in states
        # }
        return dict(
            zip(states.keys(),
                np.random.uniform(low=-1.0, high=-0.1, size=len(states))))

    def choose_action(self, state: State,
                      allowed_moves: List[Action]) -> Action:
        max_G = -10e15
        next_move = None
        random_N = np.random.random()
        if random_N < self.random_factor:
            next_move = np.random.choice(allowed_moves)
        else:
            for action in allowed_moves:
                y = Maze.action_space[action].dy
                x = Maze.action_space[action].dx
                new_state = State(state.x + x, state.y + y)
                if self.G[new_state] >= max_G:
                    next_move = action
                    max_G = self.G[new_state]
        return next_move

    def update_state_history(self, state: State, reward: int) -> None:
        self.state_history.append((state, reward))

    def learn(self) -> None:
        target = 0  # We only learn when we beat the maze.
        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []
        self.random_factor -= 10e-5
