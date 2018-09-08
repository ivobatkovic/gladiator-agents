"""Modules for implementing policy gradient based learning using NNs."""

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """Simple feed-forward network using which we sample actions."""

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """Constructor."""
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        """Forward pass."""
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.softmax(out, dim=1)
