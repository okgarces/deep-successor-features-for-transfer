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

    def __init__(self, deep_sf, lambda_phi_model, replay_buffer_handle, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
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
        self.replay_buffer_handle = replay_buffer_handle
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.device = get_torch_device()

        # phi learning
        self.lambda_phi_model = lambda_phi_model
        self.phi: Tuple[ModelTuple, ModelTuple]
        self.test_tasks_weights = []
        self.buffers = []

    def set_active_training_task(self, index):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """
        
        # set the task
        self.task_index = index
        self.active_task = self.tasks[index]
        
        # reset task-dependent counters
        self.s = self.s_enc = None
        self.new_episode = True
        self.episode, self.episode_reward = 0, 0.
        self.steps_since_last_episode, self.reward_since_last_episode = 0, 0.
        self.steps, self.reward = 0, 0.
        self.epsilon = self.epsilon_init
        self.episode_reward_hist = []

        # Set buffer to current task buffer
        self.buffer = self.buffers[index]
        
    def get_Q_values(self, s, s_enc):
        q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
        if not self.use_gpi:
            c = self.task_index
        self.c = c
        return q[:, c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # update w
        # phi_tuple, *_ = self.phi
        # phi_model, *_ = phi_tuple

        #self.sf.update_reward(phi, r, self.task_index)
        
        # TODO This changes are to update rewards and phi, psi simultaneously. Different to 
        # Original SFDQN algorithm
        # remember this experience
        #input_phi = torch.concat([s_enc.flatten().to(self.device), a.flatten().to(self.device), s1_enc.flatten().to(self.device)]).to(self.device)

        # Update Reward Mapper
        # phi = phi_model(input_phi)
        # phi = phi.clone().detach().requires_grad_(False)
        # self.sf.update_reward(phi, r, self.task_index)
        # phis should not be in buffer
        self.buffer.append(s_enc, a, r, torch.tensor([]), s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:

            # Apply learning to reward mapper
            # Update successor and phi
            for index in range(self.n_tasks):
                transitions = self.buffers[index].replay()
                # Update successor using transitions per source task and GPI to update successor
                self.sf.update_successor(transitions, self.phi, index, True)

    def reset(self):
        super(SFDQN_PHI, self).reset()
        self.sf.reset()

        # Reset all buffers
        for buffer in self.buffers:
            buffer.reset()

    def add_training_task(self, task):
        """
        This method does not call parent's method.
        """
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)  
        if self.n_tasks == 1:
            # Only one phi approximator

            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            self.s_enc_dim = task.encode_dim()
            self.action_dim = task.action_dim()
            if self.encoding == 'task':
                self.encoding = task.encode

            # After initial values 
            self.phi = self.init_phi_model()
        self.sf.add_training_task(task, source=None)
        self.buffers.append(self.replay_buffer_handle())

    ############# phi Model Learning ####################
    def init_phi_model(self):
        phi_model, phi_loss, phi_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)
        phi_target_model, phi_target_loss, phi_target_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)

        update_models_weights(phi_model, phi_target_model)
        # TODO This variable could be removed
        return (phi_model, phi_loss, phi_optim), (phi_target_model, phi_target_loss, phi_target_optim)

    ############## Progress and Stats ###################
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFDQN_PHI, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = torch.linalg.norm(self.sf.fit_w[self.task_index].weight - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
    ############### Train, Test methods ############
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000, cycles_per_task=1):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)

        for test_task in test_tasks:
            fit_w = torch.Tensor(1, test_task.feature_dim()).uniform_(-0.01, 0.01).to(self.device)
            w_approx = torch.nn.Linear(test_task.feature_dim(), 1, bias=False, device=self.device)
            # w_approx = torch.nn.Linear(test_task.feature_dim(), 1, device=self.device)

            with torch.no_grad():
                w_approx.weight = torch.nn.Parameter(fit_w)

            #fit_w = torch.Tensor(test_task.feature_dim(), 1).uniform_(-0.01, 0.01).to(self.device)
            #w_approx = fit_w

            self.test_tasks_weights.append(w_approx)
            
        # train each one
        return_data = []
        # Cycles per task
        
        for _ in range(cycles_per_task):
            for index, (train_task, viewer) in enumerate(zip(train_tasks, viewers)):
                self.set_active_training_task(index)
                for t in range(n_samples):
                    # train
                    self.next_sample(viewer, n_view_ev)
                    
                    # test
                    if t % n_test_ev == 0:
                        Rs = []
                        for test_index, test_task in enumerate(test_tasks):
                            R = self.test_agent(test_task, test_index)
                            Rs.append(R)
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
            
    def test_agent(self, task, test_index):
        R = 0.0
        w = self.test_tasks_weights[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        for _ in range(self.T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            # loss_t = self.update_test_reward_mapper(w, r, s, a, s1).item()
            loss_t = self.update_test_reward_mapper(w, r, s_enc, a, s1_enc).item()
            accum_loss += loss_t
            # loss_t = self.update_test_reward_mapper_ascent_version(w, r, s, a, s1, test_index)

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T
        self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, accum_loss, test_index, self.T))

        return R

    def update_test_reward_mapper(self, w_approx, r, s_enc, a, s1_enc):

        phi_tuple, *_ = self.phi
        phi_model, *_ = phi_tuple

        input_phi = torch.concat([s_enc.flatten(), a.flatten(), s1_enc.flatten()]).to(self.device)
        phi = phi_model(input_phi).detach()

        optim = torch.optim.SGD(w_approx.parameters(), lr=1e-4, weight_decay=1e-3)
        loss_task = torch.nn.MSELoss()

        r_tensor = torch.tensor(r).float().unsqueeze(0).detach().to(self.device)

        r_fit = w_approx(phi)

        optim.zero_grad()
        loss = loss_task(r_fit, r_tensor)

        # Otherwise gradients will be computed to inf or nan.
        if True or not (torch.isnan(loss) or torch.isinf(loss)):
            if (torch.isnan(loss) or torch.isinf(loss)):
                print(f'loss target task {loss}')
                print(f'task_w weights target {w_approx.weight}')
                print(f'phi model in target {[param.data for param in phi_model.parameters()]}')
                print(f'phis in targte reward mapper {phi}')

            loss.backward()
            optim.step()
        # If inf loss
        return loss
    
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

    def get_target_reward_mapper_error(self, r, loss, task, ts):
        return_dict = {
            'task': task,
            # Total steps and ev_frequency
            'reward': r,
            'steps': ((500 * (self.total_training_steps // 1000)) + ts),
            'w_error': loss
            }
        return return_dict
