import numpy as np

N_TASKS = 500
N_EPISODES = 1000


class KArmedBandit:
    def __init__(self, arms=10, seed_value=None):
        if seed_value is not None:
            np.random.seed(seed_value)

        self.means = np.random.normal(size=arms)
        self.actions = np.arange(arms)

    def play(self, action):
        if action not in self.actions:
            raise ValueError

        return np.random.normal(self.means[action])


class SoftMaxPlayer:
    def __init__(self, bandit, temperature, t_factor=1.0001):
        self.bandit = bandit

        self.actions = bandit.actions
        self.softmax = np.ones(self.actions.shape) / self.actions.shape
        self.actions_tried = np.zeros(self.actions.shape, dtype=int)
        self.actions_reward = np.zeros(self.actions.shape)

        self.q_function = np.zeros(self.actions.shape)
        self.temperature = temperature
        self.t_factor = t_factor

        self.total_reward = 0

    def choose_action(self):
        return np.random.choice(self.actions, p=self.softmax)

    def update(self):
        action = self.choose_action()
        reward = self.bandit.play(action)

        self.actions_tried[action] += 1
        self.actions_reward[action] += reward
        self.total_reward += reward
        self.q_function[action] = self.actions_reward[action]
        self.q_function[action] /= self.actions_tried[action]

        softmax = np.exp(self.q_function / self.temperature)
        self.softmax = softmax / softmax.sum()
        self.temperature *= self.t_factor

        return reward


def run(n_tasks, n_episodes, temperature=.2, t_factor=1.00001):
    rewards_sotfmax = []
    optimal_arm_expectation = []

    for task in range(n_tasks):
        bandit = KArmedBandit()
        optimal_arm_expectation.append(max(bandit.means))

        player = SoftMaxPlayer(bandit, temperature, t_factor=t_factor)
        for trial in range(n_episodes):
            player.update()

        rewards_sotfmax.append(player.total_reward)

    mean_softmax = np.mean(rewards_sotfmax)
    mean_optimal = np.mean(optimal_arm_expectation) * N_EPISODES

    print('softmax: ', mean_softmax, '±', np.std(rewards_sotfmax))
    print(
        'optimal: ', mean_optimal, '±',
        np.std(optimal_arm_expectation) * N_EPISODES,
    )

    return mean_softmax / mean_optimal


if __name__ == '__main__':
    run(N_TASKS, N_EPISODES)
