# -*- coding: UTF-8 -*-
import random
import numpy as np
import torch

from agents.agent import Agent
from utils.logger import get_logger_level
from utils.torch import get_torch_device
from utils.torch import update_models_weights


class TSFDQN(Agent):

    def __init__(self, deep_sf, buffer_handle, *args, use_gpi=True, test_epsilon=0.03, **kwargs):
        """
        Creates a new TSFDQN agent per the specifications in the original paper.
        
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
        super(TSFDQN, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.buffer_handle = buffer_handle
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.device = get_torch_device()
        self.test_tasks_weights = []

        # Sequential Successor Features
        self.buffers = []

        # Transformed Successor Features
        self.omegas = []
        self.g_functions = []
        self.h_function = None

    def set_active_training_task(self, index):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """
        
        # Same as Parent Agent
        super(TSFDQN, self).set_active_training_task(index)
        
        # Sequential
        self.buffer = self.buffers[index]

        # Transformed Successor Feature
        self.active_g_function = self.g_functions[index]
        
    def get_Q_values(self, s, s_enc):
        with torch.no_grad():
            q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
            if not self.use_gpi:
                c = self.task_index
            self.c = c
            return q[:, c,:]

    def _init_g_function(self, states_dim, features_dim):
        # g : |S| -> |d|, d features dimension
        g_function = torch.nn.Linear(states_dim, features_dim, bias=True, device=self.device)
        return g_function

    def _init_h_function(self, features_dim):
        # h : |d| -> |d|, d features dimension
        # This affine transformation
        h_function = torch.nn.Linear(features_dim, features_dim, bias=True, device=self.device)
        return h_function

    def _init_omega(self, features_dim):
        omega = torch.Tensor(features_dim).uniform_(0,1).to(self.device).requires_grad_(True)
        return omega
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        phi = self.phi(s, a, s1)
        self.buffer.append(s_enc, a, r, phi, s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:
            transitions = self.buffer.replay()
            losses = self.update_successor(transitions, self.task_index, self.use_gpi)
        
            if isinstance(losses, tuple):
                total_loss, psi_loss, phi_loss = losses
                self.logger.log_losses(total_loss.item(), psi_loss.item(), phi_loss.item(), [1], self.total_training_steps)

        # Print weights for Reward Mapper
        if self.total_training_steps % 1000 == 0:
            task_w = self.sf.fit_w[self.task_index]
            print(f'Current task {self.task_index} Reward Mapper {task_w.weight}')
            print(f'Current omegas {self.task_index} OMEGAS Weights {self.omegas}')

    def update_successor(self, transitions, policy_index, use_gpi=True):
        if transitions is None:
            return

        if self.h_function is None:
            raise Exception('Affine Function (h) is not initialized')

        g_function = self.g_functions[policy_index]

        states, actions, rs, phis, next_states, gammas = transitions
        n_batch = len(gammas)
        indices = torch.arange(n_batch)
        gammas = gammas.reshape((-1, 1))
        task_w = self.sf.fit_w[policy_index]
         
        # next actions come from current Successor Feature
        if use_gpi:
            q1, _ = self.sf.GPI(next_states, policy_index)
            next_actions = torch.argmax(torch.max(q1, axis=1).values, axis=-1)
        else:
            sf = self.sf.get_successor(next_states, policy_index)
            q1 = task_w(sf)
            # Do not forget: argmax according to actions, and squeeze in axis according to get n_batch
            next_actions = torch.squeeze(torch.argmax(q1, axis=1), axis=1)

        # compute the targets and TD errors
        psi_tuple, target_psi_tuple = self.sf.psi[policy_index]
        psi_model, psi_loss, optim = psi_tuple
        target_psi_model, *_ = target_psi_tuple

        current_psi = psi_model(states)

        transformed_state = g_function(states)
        transformed_next_state = g_function(next_states)
        affine_transformed_states = self.h_function(transformed_state) + self.h_function(transformed_next_state)

        transformed_phis = affine_transformed_states * phis

        with torch.no_grad():
            targets = transformed_phis + gammas * target_psi_model(next_states)[indices, next_actions,:]
        
        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets
        # psi_model.train_on_batch(states, current_psi)

        optim.zero_grad()

        r_fit = task_w(transformed_phis)

        l1 = psi_loss(current_psi, merge_current_target_psi)
        l2 = psi_loss(r_fit, rs)

        loss = l1 + l2
        loss.backward()

        # log gradients this is only a way to track gradients from time to time
        if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev - 1:
            print(f'########### BEGIN #################')
            print(f'Policy Index {policy_index}')
            print(f' Update STEP # {self.sf.updates_since_target_updated[policy_index]}')
            for params in psi_model.parameters():
                print(f'Gradients of Psi {params.grad}')

            for params in g_function.parameters():
                print(f'Gradients of G function {params.grad}')

            for params in self.h_function.parameters():
                print(f'Gradients of H function {params.grad}')

            for params in task_w.parameters():
                print(f'Gradients of W {params.grad}')

            for params in target_psi_model.parameters():
                print(f'Gradients of Psi Target {params.grad}')
            print(f'########### END #################')

        optim.step()

        # Finish train the SF network
        # update the target SF network
        self.sf.updates_since_target_updated[policy_index] += 1
        if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev:
            update_models_weights(psi_model, target_psi_model)
            self.sf.updates_since_target_updated[policy_index] = 0

        return loss, l1, l2 

    def reset(self):
        super(TSFDQN, self).reset()
        self.sf.reset()

        for buffer in self.buffers:
            buffer.reset()

    def add_training_task(self, task):
        super(TSFDQN, self).add_training_task(task)
        # Sequential Learning Append Buffer
        self.buffers.append(self.buffer_handle())

        # Transformed Successor Feature
        # Encode Dim encapsulates the state encoding dimension
        g_function = self._init_g_function(task.encode_dim(), task.feature_dim())
        self.g_functions.append(g_function)

        if self.h_function is None:
            self.h_function = self._init_h_function(task.feature_dim())

        self.omegas.append(self._init_omega(task.feature_dim()))
        # SF model will keep the model optimizer
        self.sf.add_training_task(task, None, g_function, self.h_function)

    def get_progress_dict(self):
        if self.sf is not None:
            gpi_percent = self.sf.GPI_usage_percent(self.task_index)
            w_error = torch.linalg.norm(self.sf.fit_w[self.task_index].weight.T - self.sf.true_w[self.task_index])
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
        sample_str, reward_str = super(TSFDQN, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = torch.linalg.norm(self.sf.fit_w[self.task_index].weight.T - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000, cycles_per_task=1):
        if viewers is None: 
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()
        for train_task in train_tasks:
            self.add_training_task(train_task)

        # train each one
        # Regularize sum w_i = 1
        # Unsqueeze to have [n_tasks, n_actions, n_features]
        # Initialize Omegas
        self.omegas = torch.vstack(self.omegas).unsqueeze(1)
        with torch.no_grad():
            self.omegas = (self.omegas / torch.sum(self.omegas, axis=0, keepdim=True)).nan_to_num(0)
        self.omegas = torch.tensor(self.omegas).requires_grad_(True)

        # Initialize Target Reward Mappers and optimizer
        for test_task in test_tasks:
            fit_w = torch.Tensor(1, test_task.feature_dim()).uniform_(-0.01, 0.01).to(self.device)
            w_approx = torch.nn.Linear(test_task.feature_dim(), 1, bias=False, device=self.device)
            # w_approx = torch.nn.Linear(test_task.feature_dim(), 1, device=self.device)

            with torch.no_grad():
                w_approx.weight = torch.nn.Parameter(fit_w)

            #fit_w = torch.Tensor(test_task.feature_dim(), 1).uniform_(-0.01, 0.01).to(self.device)
            #w_approx = fit_w
            # Learning rate alpha (Weights)
            parameters = [
                {'params': w_approx.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2},
                {'params': self.omegas,  'lr': 1e-3, 'weight_decay': 1e-2},
            ]

            optim = torch.optim.Adam(parameters)
            self.test_tasks_weights.append((w_approx, optim))

        return_data = []

        print(f'Self Omegas after requires_grad {self.omegas}')

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
                successor_features = self.sf.get_successors(s_enc)
                tsf = torch.sum(successor_features * self.omegas, axis=1)

                q = w(tsf)
                # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                # Target TSF only Use Q-Learning
                a = torch.argmax(q)
            return a
            
    def test_agent(self, task, test_index):
        R = 0.0
        w, optim = self.test_tasks_weights[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        for _ in range(self.T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            loss_t = self.update_test_reward_mapper(w, optim, task, r, s_enc, a, s1_enc).item()
            accum_loss += loss_t

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T
        self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, accum_loss, test_index, self.T))

        return R

    def update_test_reward_mapper(self, w_approx, optim, task, r, s, a, s1):

        if self.h_function is None:
            raise Exception('Affine Function (h) is not initialized')

        # Return Loss
        phi = task.features(s,a,s1)

        # Transformed States
        t_states = []
        t_next_states = []

        with torch.no_grad():
            for g in self.g_functions:
                state = g(s)
                next_state = g(s1)
                t_states.append(state)
                t_next_states.append(next_state)

            # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
            t_states = torch.vstack(t_states).unsqueeze(1)
            t_next_states = torch.vstack(t_next_states).unsqueeze(1)

        weighted_states = torch.sum(t_states * self.omegas, axis=0)
        weighted_next_states = torch.sum(t_next_states * self.omegas, axis=0)

        with torch.no_grad():
            affine_states = self.h_function(weighted_states) + self.h_function(weighted_next_states)
            transformed_phi = phi * affine_states.squeeze(0)

            successor_features = self.sf.get_successors(s)
            next_successor_features = self.sf.get_next_successors(s)

            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

        tsf = torch.sum(successor_features * self.omegas, axis=1)
        next_tsf = transformed_phi + self.gamma * torch.sum(next_successor_features * self.omegas, axis=1)

        loss_task = torch.nn.MSELoss()

        r_fit = w_approx(transformed_phi)

        l1 = loss_task(tsf, next_tsf)
        l2 = loss_task(r_fit, r_tensor)

        loss = l1 + l2

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Sum_i omega_i = 1
        with torch.no_grad():
            self.omegas.clamp_(0, 1) 
            weight_sum_1 = (self.omegas / torch.sum(self.omegas, axis=0, keepdim=True)).nan_to_num(0)
            self.omegas.copy_(weight_sum_1) 

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
