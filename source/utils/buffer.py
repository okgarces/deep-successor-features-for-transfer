# -*- coding: UTF-8 -*-
import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, n_samples=1000000, n_batch=32, device=None):
        """
        Creates a new randomized replay buffer.

        Parameters
        ----------
        n_samples : integer
            the maximum number of samples that can be stored in the buffer
        n_batch : integer
            the batch size
        """
        self.n_samples = n_samples
        self.n_batch = n_batch
        self.device = device

        # When initialize run reset()
        self.reset()

    def reset(self):
        """
        Removes all samples currently stored in the buffer.
        """
        self.buffer = np.empty(self.n_samples, dtype=object)
        self.index = 0
        self.size = 0

    def replay(self):
        """
        Samples a batch of samples from the buffer randomly. If the number of samples
        currently in the buffer is less than the batch size, returns None.

        Returns
        -------
        states : torch.Tensor
            a collection of starting states of shape [n_batch, -1]
        actions : torch.Tensor
            a collection of actions taken in the starting states of shape [n_batch,]
        rewards : torch.Tensor:
            a collection of rewards (for DQN) or features (for SFDQN) obtained of shape [n_batch, -1]
        next_states : torch.Tensor
            a collection of successor states of shape [n_batch, -1]
        gammas : torch.Tensor
            a collection of discount factors to be applied in computing targets for training of shape [n_batch,]
        """
        if self.size < self.n_batch: return None
        indices = np.random.randint(low=0, high=self.size, size=(self.n_batch,))
        states, actions, rewards, phis, next_states, gammas = zip(*self.buffer[indices])
        states = torch.tensor(np.vstack(states)).float().to(self.device)
        actions = torch.tensor(np.vstack(actions)).long().to(self.device)
        rewards = torch.tensor(np.vstack(rewards)).float().to(self.device)
        phis = torch.tensor(np.vstack(phis)).float().to(self.device)
        next_states = torch.tensor(np.vstack(next_states)).float().to(self.device)
        gammas = torch.tensor(np.vstack(gammas)).float().to(self.device)
        return states, actions, rewards, phis, next_states, gammas

    def append(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, phi: np.ndarray, next_state: np.ndarray,
               gamma: np.ndarray) -> None:
        """
        Adds the specified sample to the replay buffer. If the buffer is full, then the earliest added
        sample is removed, and the new sample is added.

        Parameters
        ----------
        state : torch.Tensor
            the encoded state of the task
        action : integer
            the action taken in state
        reward : torch.Tensor
            the reward obtained in the current transition (for DQN) or state features (for SFDQN)
        next_state : torch.Tensor
            the encoded successor state
        gamma : float
            the effective discount factor to be applied in computing targets for training
        """
        # Reward are double. Here we convert to float to store float in Buffer
        self.buffer[self.index] = (state, action, reward, phi, next_state, gamma)
        self.size = min(self.size + 1, self.n_samples)
        self.index = (self.index + 1) % self.n_samples