import numpy as np
import NN
import copy


class DQN:
    def __init__(self, state_dim, action_dim, epsilon_0 = .9, epsilon_decay=.95, gamma=.9, layers=None, **kwargs):
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.replay_memory = []

        if layers is None:
            layers = [state_dim, 50, action_dim]
        self.q_network = NN.FCNN(layers, **kwargs)
        self.target_network = copy.copy(self.q_network)

    def action(self, state):
        if self.epsilon > np.random.rand():
            return np.random.rand(self.action_dim)
        return self.q_network.forward(np.asarray(state)).ravel() # only supports one dimensional outputs

    def store_in_replay(self, state, action, reward, next_sate, done):
        self.replay_memory.append((state, action, reward, next_sate, done))
        if len(self.replay_memory) > 4096:
            self.replay_memory = self.replay_memory[2048:]

    def replay(self, batch_size=32):
        if len(self.replay_memory) < batch_size:
            return

        batch_idxs = np.random.choice(range(len(self.replay_memory)), batch_size)
        for i in batch_idxs:
            s, a, r, n_s, done = self.replay_memory[i]
            if not done:
                r += self.gamma * np.max(self.target_network.forward(n_s))
            y = self.target_network.forward(s)
            y[0,a] = r
            self.q_network.forward(np.asarray(s), y=y)
        self.target_network = copy.copy(self.q_network)
        self.epsilon = max((0.01, self.epsilon*self.epsilon_decay))
