import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimaxQNetwork(nn.Module):
    def __init__(self, state_dim=128, num_actions=18):
        super().__init__()

        self.num_actions = num_actions

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Q(s, a, b)
        self.fc_out = nn.Linear(256, num_actions * num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q = self.fc_out(x)

        q = q.view(-1, self.num_actions, self.num_actions)

        return q