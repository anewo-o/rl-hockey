import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha

        self.pos = 0
        self.size = 0

        # storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # priorities
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            self.states[self.pos] = state
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.next_states[self.pos] = next_state
            self.dones[self.pos] = done

        # assign max priority (new experience = important)
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]

        # compute probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)

        # importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        # gather samples
        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + 1e-6