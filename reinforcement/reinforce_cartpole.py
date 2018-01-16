import sys
import time

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

MAX_EPISODES = 1000


class REINFORCEAgent:
    def __init__(
        self,
        state_size,
        action_size,
        discount_factor=0.99,
        learning_rate=0.001,
        hidden1=24,
        hidden2=24,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.hidden1, self.hidden2 = hidden1, hidden2

        self.model = self.build_model()

        self.states, self.actions, self.rewards = [], [], []

    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()

        model.add(
            Dense(
                self.hidden1,
                input_dim=self.state_size,
                activation='relu',
                kernel_initializer='glorot_uniform',
            ),
        )

        model.add(
            Dense(
                self.hidden2,
                activation='relu',
                kernel_initializer='glorot_uniform',
            ),
        )

        model.add(
            Dense(
                self.action_size,
                activation='softmax',
                kernel_initializer='glorot_uniform',
            ),
        )
        model.summary()

        model.compile(
            loss='categorical_crossentropy',  # H(p, q) = A * log(policy(s, a))
            optimizer=Adam(lr=self.learning_rate),
        )

        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        for ix in range(len(rewards) - 1, -1, -1):
            running_add *= self.discount_factor
            running_add += rewards[ix]
            discounted_rewards[ix] = running_add

        return discounted_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []


def show_game(env, agent, repeat_num=5):
    for _ in range(repeat_num):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1, state_size])

            time.sleep(.02)

        env.render()
        time.sleep(1)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)
    scores, episodes = [], []

    for episode in range(MAX_EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if done and score < 499:
                reward = -100

            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                agent.train_model()

                if score < 500:
                    score += 100

                scores.append(score)

                if episode and episode % 10 == 0:
                    print('episode #', episode, '  score:', score)

            if np.mean(scores[-min(10, len(scores)):]) > 490:
                break

    show_game(env, agent)
    env.close()
    sys.exit()
