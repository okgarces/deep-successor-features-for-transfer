# -*- coding: UTF-8 -*-
import gym
from gym.envs.classic_control import CartPoleEnv

import torch
import numpy as np

from tasks.task import Task
from utils.torch import get_torch_device


class Cartpole_PHI(Task):
    
    def __init__(self, task_index, n_features, pole_length=0.5):
        self.task_index = task_index
        self.device = get_torch_device()
        self.n_features = n_features

        self.env = CartPoleEnv()
        self.env.gravity = 9.8
        self.env.masscart = 1.0
        self.env.masspole = 0.1
        self.env.total_mass = self.env.masspole + self.env.masscart
        self.env.length = pole_length if pole_length else 0.5  # actually half the pole's length
        self.env.polemass_length = self.env.masspole * self.env.length

    def clone(self):
        return Cartpole_PHI(self.task_index, self.n_features)
    
    def initialize(self):
        state = torch.tensor(self.env.reset()).to(self.device)
        return state
    
    def action_count(self):
        # Cartpole has two options right or left
        return 2
    
    def transition(self, action: torch.Tensor):
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()

        new_state, reward, done, _ = self.env.step(action)

        new_state = torch.tensor(new_state).to(self.device)
        reward = torch.tensor(reward).to(self.device)

        return new_state, reward, done
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        return torch.tensor(state).reshape((1, -1)).to(self.device) # Flat all [B, D]
    
    def encode_dim(self):
        return self.env.observation_space.shape[0]
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        raise Exception('Phi Version should learn features')
    
    def feature_dim(self):
        return self.n_features
    
    def get_w(self):
        raise Exception('SFDQN should learn weights')

    def action_dim(self):
        # The action dim for Reacher is only a integer number
        return 1