# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch

from agents.agent import Agent
from utils.logger import get_logger_level
from utils.torch import get_torch_device


class SFDQN(Agent):

    def __init__(self, deep_sf, buffer_handle, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
        """
        Creates a new SFDQN agent per the specifications in the original paper.
        
        Parameters
        ----------
        deep_sf : DeepSF
            instance of deep successor feature representation
         buffer_handle : () -> ReplayBuffer
            a replay buffer that implements randomized experience replay
        use_gpi : boolean
            whether or not to use transfer learning (defaults to True)
        test_epsilon : float
            the exploration parameter for epsilon greedy used during testing 
            (defaults to 0.03 as in the paper)
        """
        super(SFDQN, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.buffer_handle = buffer_handle
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.device = get_torch_device()
        self.test_tasks_weights = []

        # Sequential Successor Features
        self.buffers = []

    def set_active_training_task(self, index):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """
        
        # Same as Parent Agent
        super(SFDQN, self).set_active_training_task(index)
        
        # Sequential
        self.buffer = self.buffers[index]

    def get_Q_values(self, s, s_enc):
        with torch.no_grad():
            q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
            if not self.use_gpi:
                c = self.task_index
            self.c = c
            return q[:, c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        phi = self.phi(s, a, s1)
        self.buffer.append(s_enc, a, r, phi, s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:
            transitions = self.buffer.replay()
            losses = self.sf.update_successor(transitions, self.task_index, self.use_gpi)
        
            if isinstance(losses, tuple):
                total_loss, psi_loss, phi_loss = losses
                self.logger.log_losses(total_loss.item(), psi_loss.item(), phi_loss.item(), [1], self.total_training_steps)

        # Print weights for Reward Mapper
        if self.total_training_steps % 1000 == 0:
            task_w = self.sf.fit_w[self.task_index]
            print(f'Current task {self.task_index} Reward Mapper {task_w.weight}')

    def reset(self):
        super(SFDQN, self).reset()
        self.sf.reset()

        for buffer in self.buffers:
            buffer.reset()

    def add_training_task(self, task):
        super(SFDQN, self).add_training_task(task)
        self.sf.add_training_task(task, source=None)

        # Append Buffer
        self.buffers.append(self.buffer_handle())

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
    
    def get_progress_strings(self):
        sample_str, reward_str = super(SFDQN, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = torch.linalg.norm(self.sf.fit_w[self.task_index] - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
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
        with torch.no_grad():
            if random.random() <= self.test_epsilon:
                a = torch.tensor(random.randrange(self.n_actions)).to(self.device)
            else:
                psi = self.sf.get_successors(s_enc)
                q = w(psi)[:,:,:,0]
                # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                c = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)

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
            loss_t = self.update_test_reward_mapper(w, task, r, s_enc, a, s1_enc).item()
            accum_loss += loss_t

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T
        self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, accum_loss, test_index, self.T))

        return R

    def update_test_reward_mapper(self, w_approx, task, r, s, a, s1):
        # Return Loss
        phi = task.features(s,a,s1)

        # Learning rate alpha (Weights)
        optim = torch.optim.Adam(w_approx.parameters(), lr=1e-2)
        loss_task = torch.nn.MSELoss()

        with torch.no_grad():
            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

        optim.zero_grad()

        r_fit = w_approx(phi)
        loss = loss_task(r_fit, r_tensor)
        loss.backward()
        
        optim.step()
        return loss

    def get_target_reward_mapper_error(self, r, loss, task_index, ts):
        return_dict = {
            'task': task_index,
            # Total steps and ev_frequency
            'reward': r,
            'steps': ((500 * (self.total_training_steps // 1000)) + ts),
            'w_error': loss,
            #'w_error': torch.linalg.norm(self.test_tasks_weights[task_index] - task.get_w())
            }
        return return_dict