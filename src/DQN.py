import numpy as np
import NN
import copy
import time

IDX_TIME = 0
COPY_TIME = 0
TRAINING_TIME = 0
losses = []

class DQN:
    def __init__(self, num_state_vars, num_actions, epsilon_0 = .9, epsilon_decay=.95, gamma=.9, layers=None, **kwargs):
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_0
        self.num_state_vars = num_state_vars
        self.num_actions = num_actions
        self.gamma = gamma

        self.replay_memory = []

        if layers is None:
            layers = [num_state_vars, 50, num_actions]
        self.q_network = NN.FCNN(layers, **kwargs)
        self.target_network = copy.copy(self.q_network)

    def action(self, state):
        if self.epsilon > np.random.rand():
            return int(np.random.rand() * self.num_actions)
        qs = self.q_network.forward(np.asarray(state))
        return np.argmax(qs)

    def store_in_replay(self, state, action, reward, next_sate, done):
        self.replay_memory.append((state, action, reward, next_sate, done))
        if len(self.replay_memory) > 4096:
            self.replay_memory = self.replay_memory[2048:]

    def replay(self, batch_size=32):
        global IDX_TIME, TRAINING_TIME, COPY_TIME, losses
        if len(self.replay_memory) < batch_size:
            return

        start_time = time.time()
        batch_idxs = np.random.choice(range(len(self.replay_memory)), batch_size)
        IDX_TIME += time.time() - start_time
        for i in batch_idxs:
            s, a, r, n_s, done = self.replay_memory[i]
            if not done:
                r += self.gamma * np.max(self.target_network.forward(n_s))
            y = self.target_network.forward(s)
            y[0,a] = r
            start_time = time.time()
            losses.append(self.q_network.forward(np.asarray(s), y=y)[0])
            TRAINING_TIME += time.time() - start_time
        start_time = time.time()
        self.target_network = copy.copy(self.q_network)
        COPY_TIME += time.time() - start_time
        self.epsilon = max((0.01, self.epsilon*self.epsilon_decay))
