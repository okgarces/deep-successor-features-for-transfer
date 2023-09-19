# -*- coding: UTF-8 -*-
import torch
import numpy as np
import gym
import numpy as np
from gym.envs.mujoco.hopper_v4 import HopperEnv

from tasks.task import Task
from utils.torch import get_torch_device


class Hopper_PHI(Task):
    
    def __init__(self, min_healthy_z, n_features):
        self.env = HopperEnv(healthy_z_range=(min_healthy_z, np.inf))
        
        # make the action lookup from integer to real action
        actions = [-1., 0., 1.]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                for a3 in actions:
                    self.action_dict[len(self.action_dict)] = (a1, a2, a3)

        self.device = get_torch_device()
        self.n_features = n_features
        self.min_healthy_z = min_healthy_z
        
    def clone(self):
        return HopperEnv(healthy_z_range=(self.min_healthy_z, np.inf))
    
    def initialize(self):
        # if self.task_index == 0:
        #    self.env.render('human')
        state = torch.tensor(self.env.reset()).to(self.device)
        return state
    
    def action_count(self):
        return len(self.action_dict)
    
    def transition(self, action: torch.Tensor):
        action_int = int(action)
        real_action = self.action_dict[action_int]
        new_state, reward, done, *_ = self.env.step(real_action)

        new_state = torch.tensor(new_state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        return new_state, reward, done
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state = torch.tensor(state).reshape((1, -1)).to(self.device, dtype=torch.float32)

            norm_state = torch.sigmoid(state)

        return norm_state
    
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
        raise Exception('Get w should learnt')

    def action_dim(self):
        # The action dim for Reacher is only a integer number
        return 1