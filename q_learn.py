"""Modules for implementing q-learning using NNs.

A lot of the below code is taken from
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

"""
import random
from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
GAMMA = 0.999

Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'score'))


class ReplayMemory(object):
    """Class to implement replay memory."""

    def __init__(self, capacity):
        """Constructor."""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Get the length of memory buffer."""
        return len(self.memory)


class QNetwork(nn.Module):
    """Simple feed-forward network to approximate Q-values."""

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """Constructor."""
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        """Forward pass."""
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
