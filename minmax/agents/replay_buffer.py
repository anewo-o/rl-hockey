import random
from collections import deque
import numpy as np


class MinimaxReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_a, action_b, reward, next_state, done):
        self.buffer.append((state, action_a, action_b, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions_a, actions_b, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions_a),
            np.array(actions_b),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)