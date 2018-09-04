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


def create_batch_inputs(game, batch):
    """Create batch of inputs to model."""
    batch_size = len(batch.current_state)
    state_batch = torch.zeros(batch_size, 16)
    action_batch = torch.zeros(batch_size, 1)
    reward_batch = torch.zeros(batch_size)
    next_state_batch = torch.zeros(batch_size, 16)

    # create state batch
    for idx, states in enumerate(batch.current_state):
        states_sample = []
        rel_states = game.players[0].calc_rel_states(states)
        for states in rel_states:
            for state_pair in states:
                if state_pair == None:
                    state_inputs = torch.zeros(4)
                else:
                    state_inputs = torch.from_numpy(np.array(state_pair)).float()
                states_sample.append(state_inputs)
        states_sample = torch.cat(states_sample)
        state_batch[idx] = states_sample

    # create action batch
    for idx, actions in enumerate(batch.action):
        # player 1's action
        action_batch[idx] = actions[0]

    # create next state batch
    for idx, states in enumerate(batch.next_state):
        states_sample = []
        rel_states = game.players[0].calc_rel_states(states)
        for states in rel_states:
            for state_pair in states:
                if state_pair == None:
                    state_inputs = torch.zeros(4)
                else:
                    state_inputs = torch.from_numpy(np.array(state_pair)).float()
                states_sample.append(state_inputs)
        states_sample = torch.cat(states_sample)
        next_state_batch[idx] = states_sample

    # create reward batch
    for idx, scores in enumerate(batch.score):
        # player 1's scores
        reward_batch[idx] = scores[0]

    return state_batch, action_batch, next_state_batch, reward_batch


def optimize_model(game, memory, q_net, target_net, optimizer, batch_size):
    """Perform one step of the optimization for q-learning."""
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state_batch, action_batch, next_state_batch, reward_batch = create_batch_inputs(game, batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = q_net(state_batch).gather(1, action_batch.long())

    # Compute V(s_{t+1}) for all next states.
    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # print('Loss: {}'.format(loss.item()))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
