from __future__ import division
from pathlib import Path

import os
import random
import numpy as np
import tensorflow as tf

TARGET_PATH = os.path.join(os.path.dirname(__file__), 'target.h5')
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), 'weights.h5')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'model.png')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

class QNetwork:
    def __init__(self, state_size=9, discount=1, epsilon=1, epsilon_min=0.0001, epsilon_decay=0.9995):
        self.state_size = state_size
        self.model, self.target = self._create_model()
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.experiences = Memory(32768)
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                          histogram_freq=1000,
                                                          write_graph=True,
                                                          write_images=True)

    def _create_model(self):
        """Returns a new model."""

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])
        target = tf.keras.models.clone_model(model)
        target.load_weights(TARGET_PATH)
        # For the loss function Mean Square Error is used as the problem
        # we are trying to solve is a regression problem
        # rather than a classification problem.
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        model.summary()

        #tf.keras.utils.plot_model(model, IMAGE_PATH, show_shapes=True)

        return model, target

    def act(self, possible_states):
        """Returns the best of multiple states unless it has decided to explore which returns a random one."""

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_states)

        max_rating = None
        best_state = None
        ratings = self._predict_ratings([state for action, state in possible_states])

        for i, (action, state) in enumerate(possible_states):
            rating = ratings[i]
            if not max_rating or rating > max_rating:
                max_rating = rating
                best_state = (action, state)

        return best_state

    def _predict_ratings(self, states):
        """Returns the outputs of the Neural Network for multiple states."""
        input_y = np.array(states)
        predictions = self.model.predict(input_y)
        return [predict[0] for predict in predictions]

    def _target_ratings(self, states):
        input_y = np.array(states)
        predictions = self.target.predict(input_y)
        return [predict[0] for predict in predictions]

    def memorize(self, p, data):
        self.experiences.add(p, data)

    def train(self, env, episodes=1):
        rewards = []
        scores = []
        steps = 0
        for episode in range(episodes):
            obs = env.reset()
            previous_state = env.game.board.get_info([])
            done = False
            total_reward = 0
            while not done:
                action, state = self.act(obs)
                obs, reward, done, info = env.step(action)
                self.memorize(reward, (previous_state, reward, state, done))
                previous_state = state
                steps += 1
                total_reward += reward
                if(total_reward >= 10000):
                    done = True

            rewards.append(total_reward)
            scores.append(env.game.score)

            self.learn()

        return [steps, rewards, scores]

    def load(self):
        """Load the weights."""
        if Path(WEIGHT_PATH).is_file():
            self.model.load_weights(WEIGHT_PATH)

    def save(self):
        """Save the weights."""
        if not os.path.exists(os.path.dirname(WEIGHT_PATH)):
            os.makedirs(os.path.dirname(WEIGHT_PATH))

        self.model.save_weights(WEIGHT_PATH)

    def learn(self, batch_size=512, epochs=1):
        if self.experiences.tree.n_entries < batch_size:
            return

        batch, idxs, is_weight = self.experiences.sample(batch_size)

        train_x = []
        train_y = []
        ratings = self._target_ratings([x[2] for x in list(batch)])

        for i, (previous_state, reward, next_state, done) in enumerate(batch):
            if not done:
                rating = ratings[i]
                q = reward + self.discount * rating
            else:
                q = reward
            train_x.append(previous_state)
            train_y.append(q)

        self.model.fit(np.array(train_x), np.array(train_y), batch_size=len(train_x), verbose=0,
                       epochs=epochs, callbacks=[self.tensorboard], sample_weight=np.array(is_weight))
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.8
    beta = 0.3
    beta_increment_per_sampling = 0.0005

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])