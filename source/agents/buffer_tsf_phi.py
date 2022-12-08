# -*- coding: UTF-8 -*-
import numpy as np
import torch

from utils.torch import get_torch_device


class ReplayBuffer_TSF_PHI:
    
    def __init__(self, *args, n_samples=1000000, n_batch=32, **kwargs):
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

        self.device = get_torch_device()

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
        states = torch.vstack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.vstack(rewards).to(self.device)
        phis = torch.vstack(phis).to(self.device)
        next_states = torch.vstack(next_states).to(self.device)
        gammas = torch.tensor(gammas).to(self.device)
        return states, actions, rewards, phis, next_states, gammas
    
    def append(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, phi: torch.Tensor, next_state: torch.Tensor, gamma: torch.Tensor) -> None:
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
        self.buffer[self.index] = (state, action, reward.float(), phi, next_state, gamma)
        self.size = min(self.size + 1, self.n_samples)
        self.index = (self.index + 1) % self.n_samples
        
