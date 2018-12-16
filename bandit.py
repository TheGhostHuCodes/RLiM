from typing import List

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, num_arms: int, true_rewards: List[float],
                 epsilon: float) -> None:
        self.num_arms = num_arms
        self.Q = [0 for _ in range(self.num_arms)]
        self.N = [0 for _ in range(self.num_arms)]
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.last_action = None

    def pull(self) -> float:
        rand = np.random.random()
        if rand <= self.epsilon:
            which_arm = np.random.choice(self.num_arms)
        else:
            a = np.array([approx for approx in self.Q])
            which_arm = np.random.choice(np.where(a == a.max())[0])
        self.last_action = which_arm

        # Return a random reward centered around the true reward for the arm.
        return np.random.randn() + self.true_rewards[which_arm]

    def update_mean(self, sample: float) -> None:
        which_arm = self.last_action
        self.N[which_arm] += 1
        self.Q[which_arm] = self.Q[which_arm] + 1.0 / self.N[which_arm] * (
            sample - self.Q[which_arm])


def simulate(num_arms: int, epsilon: float, num_pulls: int) -> np.ndarray:
    reward_history = np.zeros(num_pulls)
    for j in range(2000):
        rewards = [np.random.randn() for _ in range(num_actions)]
        bandit = Bandit(num_arms, rewards, epsilon)
        if j % 200 == 0:
            print(j)
        for i in range(num_pulls):
            reward = bandit.pull()
            bandit.update_mean(reward)

            reward_history[i] += reward
    return reward_history / 2000


if __name__ == '__main__':
    num_actions = 5
    run1 = simulate(num_actions, epsilon=0.1, num_pulls=1000)
    run2 = simulate(num_actions, epsilon=0.01, num_pulls=1000)
    run3 = simulate(num_actions, epsilon=0.0, num_pulls=1000)

    plt.plot(run1, 'b-', run2, 'r-', run3, 'g-')
    plt.xlabel('pull number')
    plt.ylabel('reward')
    plt.legend(['epsilon-0.1', 'epsilon-0.01', 'epsilon-0, Pure greedy'])
    plt.show()