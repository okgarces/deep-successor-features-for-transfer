# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch
from typing import Tuple

from agents.agent import Agent
from utils.logger import get_logger_level
from utils.torch import get_torch_device
from utils.types import ModelTuple
from utils.torch import update_models_weights


class SFDQN_PHI(Agent):

    def __init__(self, deep_sf, lambda_phi_model, buffer, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
        """
        Creates a new SFDQN agent per the specifications in the original paper.
        
        Parameters
        ----------
        deep_sf : DeepSF
            instance of deep successor feature representation
         buffer : ReplayBuffer
            a replay buffer that implements randomized experience replay
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(SFDQN_PHI, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.buffer = buffer
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.device = get_torch_device()

        # phi learning
        self.lambda_phi_model = lambda_phi_model
        self.phi: Tuple[ModelTuple, ModelTuple]
        self.updates_since_phi_target_updated = []
        
    def get_Q_values(self, s, s_enc):
        q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = self.task_index
        self.c = c
        return q[:, c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # update w
        phi_tuple, *_ = self.phi
        phi_model, *_ = phi_tuple

        #self.sf.update_reward(phi, r, self.task_index)
        
        # remember this experience
        with torch.no_grad():
            input_phi = torch.concat([s_enc.flatten().to(self.device), a.flatten().to(self.device), s1_enc.flatten().to(self.device)]).to(self.device)
            phi = phi_model(input_phi)
            self.buffer.append(s_enc, a, r, phi, s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 10 == 0:
            transitions = self.buffer.replay()
            for index in range(self.n_tasks):
                self.sf.update_successor(transitions, self.phis, index)
        
    def reset(self):
        super(SFDQN_PHI, self).reset()
        self.sf.reset()
        self.buffer.reset()
        self.updates_since_phi_target_updated = []

    def add_training_task(self, task):
        """
        This method does not call parent's method.
        """
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)  
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            # TODO Is this n_features is needed?
            self.n_features = -1
            self.s_enc_dim = task.encode_dim()
            self.action_dim = task.action_dim()
            if self.encoding == 'task':
                self.encoding = task.encode

        self.phis.append(self.init_phi_model())
        self.sf.add_training_task(task, source=None)

    ############# phi Model Learning ####################
    def init_phi_model(self):
        # This is only if there is only one shared phi
        if self.n_tasks > 1:
            return self.phis[0]

        phi_model, phi_loss, phi_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)
        phi_target_model, phi_target_loss, phi_target_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)

        update_models_weights(phi_model, phi_target_model)
        self.updates_since_phi_target_updated.append(0)

        return (phi_model, phi_loss, phi_optim), (phi_target_model, phi_target_loss, phi_target_optim)

    ############## Progress and Stats ###################
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFDQN_PHI, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = torch.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
    ############### Train, Test methods ############
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)
            
        # train each one
        return_data = []
        for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
            self.set_active_training_task(index)
            for t in range(n_samples):
                
                # train
                self.next_sample(viewer, n_view_ev)
                
                # test
                if t % n_test_ev == 0:
                    Rs = []
                    for test_task in test_tasks:
                        R = self.test_agent(test_task)
                        Rs.append(R)
                    print('test performance: {}'.format('\t'.join(map('{:.4f}'.format, Rs))))
                    avg_R = torch.mean(torch.Tensor(Rs).to(self.device))
                    return_data.append(avg_R)
                    self.logger.log_progress(self.get_progress_dict())
                    self.logger.log_average_reward(avg_R, self.total_training_steps)
                    self.logger.log_accumulative_reward(torch.sum(torch.Tensor(return_data).to(self.device)), self.total_training_steps)

                self.total_training_steps += 1
        return return_data
    
    def get_test_action(self, s_enc, w):
        if random.random() <= self.test_epsilon:
            a = torch.tensor(random.randrange(self.n_actions)).to(self.device)
        else:
            q, c = self.sf.GPI_w(s_enc, w)
            q = q[:, c,:]
            a = torch.argmax(q)
        return a
            
    def test_agent(self, task):
        R = 0.0
        # w = task.get_w()
        w = torch.nn.Linear(task.feature_dim(), 1).to(self.device)
        s = task.initialize()
        s_enc = self.encoding(s)
        for _ in range(self.T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            s, s_enc = s1, s1_enc
            R += r
            if done:
                break
        return R
    
    def get_progress_dict(self):
        if self.sf is not None:
            gpi_percent = self.sf.GPI_usage_percent(self.task_index)
            w_error = torch.linalg.norm(self.sf.fit_w[self.task_index].weight - self.sf.true_w[self.task_index])
        else:
            gpi_percent = None
            w_error = None

        return_dict = {
            'task': self.task_index,
            'steps': self.total_training_steps,
            'episodes': self.episode,
            'eps': self.epsilon,
            'ep_reward': self.episode_reward,
            'reward': self.reward,
            'reward_hist': self.reward_hist,
            'cum_reward': self.cum_reward,
            'cum_reward_hist': self.cum_reward_hist,
            'GPI%': gpi_percent,
            'w_err': w_error
            }
        return return_dict
