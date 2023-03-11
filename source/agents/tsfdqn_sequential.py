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
        self.hyperparameters = kwargs.get('hyperparameters', {})

        self.logger = get_logger_level()
        self.device = get_torch_device()
        self.test_tasks_weights = []

        # Sequential Successor Features
        self.buffers = []

        # Transformed Successor Features
        self.omegas_per_source_task = []
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

    def _init_g_function(self, states_dim, output_dim):
        # g : |S| -> |d|, d features dimension
        g_function = torch.nn.Linear(states_dim, output_dim, bias=True, device=self.device)
        #with torch.no_grad():
        #    # 0.001 and 0.01 got from analysis between the max values of g_function [-220, 200] and phi prefixed is [-1.5, 1]
        #    weights = torch.eye(features_dim, states_dim).to(self.device).requires_grad_(False)
        #    bias = torch.Tensor(1, features_dim).uniform_(0.01,0.1).to(self.device).requires_grad_(False)
        #    g_function.weight = torch.nn.Parameter(weights, requires_grad=False)
        #    g_function.bias = torch.nn.Parameter(bias, requires_grad=False)
        return g_function

    def _init_h_function(self, input_dim, features_dim):
        # TODO Remove weights clamp and initial weights
        # h : |d| -> |d|, d features dimension
        # This affine transformation
        h_function = torch.nn.Linear(input_dim, features_dim, bias=True, device=self.device)
        #with torch.no_grad():
        #    # 0.001 and 0.01 got from analysis between the max values of g_function [-220, 200] and phi prefixed is [-1.5, 1]
        #    weights = torch.Tensor(features_dim, features_dim).uniform_(0.001,0.01).to(self.device).requires_grad_(False)
        #    bias = torch.Tensor(1, features_dim).uniform_(0.001,0.01).to(self.device).requires_grad_(False)
        #    h_function.weight = torch.nn.Parameter(weights, requires_grad=True)
        #    h_function.bias = torch.nn.Parameter(bias, requires_grad=True)

        return h_function

    def _init_omega(self, num_source_tasks):
        omega = torch.Tensor(1, num_source_tasks, 1, 1).uniform_(0,1).to(self.device).requires_grad_(True)
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
                
                self.logger.log_losses(total_loss.item(), psi_loss.item(), phi_loss.item(), [self.hyperparameters['beta_loss_coefficient']], self.total_training_steps)

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
        with torch.no_grad():
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
            next_psis = target_psi_model(next_states)[indices, next_actions,:]

        targets = transformed_phis + gammas * next_psis

        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets
        # psi_model.train_on_batch(states, current_psi)

        optim.zero_grad()

        r_fit = task_w(transformed_phis)
        beta_loss_coefficient = torch.tensor(self.hyperparameters['beta_loss_coefficient'])

        l1 = psi_loss(current_psi, merge_current_target_psi)
        l2 = psi_loss(r_fit, rs)
        
        loss = l1 + (beta_loss_coefficient * l2)
        loss.backward()

        # log gradients this is only a way to track gradients from time to time
        if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev - 1:
            print(f'########### BEGIN #################')
            print(f'Affine transformed states {torch.norm(affine_transformed_states, dim=1)} task {policy_index}')
            print(f'g functions values states {torch.norm(transformed_state, dim=1)} task {policy_index}')
            print(f'g functions values next states {torch.norm(transformed_next_state, dim=1)} task {policy_index}')
            print(f'phis values {torch.norm(phis, dim=1)} task {policy_index}')
            print(f'Policy Index {policy_index}')
            print(f' Update STEP # {self.sf.updates_since_target_updated[policy_index]}')

            accum_grads = 0
            accum_weights = 0
            for params in psi_model.parameters():
                accum_grads += torch.norm(params.grad)
                accum_weights += torch.norm(params.data)
            print(f'Gradients of Psi {accum_grads}')
            print(f'Psi weights {accum_weights}')

            accum_grads = 0
            accum_weights = 0
            for params in g_function.parameters():
                accum_grads += torch.norm(params.grad)
                accum_weights += torch.norm(params.data)
            print(f'Gradients of G function {accum_grads}')
            print(f'G function weights {accum_weights}')

            accum_grads = 0
            accum_weights = 0
            for params in self.h_function.parameters():
                accum_grads += torch.norm(params.grad)
                accum_weights += torch.norm(params.data)
            print(f'Gradients of H function {accum_grads}')
            print(f'H function weights {accum_weights}')

            accum_grads = 0
            accum_weights = 0
            for params in task_w.parameters():
                accum_grads += torch.norm(params.grad)
                accum_weights += torch.norm(params.data)
                print(f'W weights {params.data}')
            print(f'Gradients of W {accum_grads}')
            print(f'W weights {accum_weights}')

            accum_grads = 0
            accum_weights = 0
            for params in target_psi_model.parameters():
                if params.grad is not None:
                    accum_grads += torch.norm(params.grad)
                accum_weights += torch.norm(params.data)
            print(f'Gradients of Psi Target {accum_grads}')
            print(f'Weights of Psi Target {accum_weights}')
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
        g_h_function_dims = self.hyperparameters.get('g_h_function_dims')

        g_function = self._init_g_function(task.encode_dim(), g_h_function_dims)
        self.g_functions.append(g_function)

        if self.h_function is None:
            self.h_function = self._init_h_function(g_h_function_dims, task.feature_dim())

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
        # Unsqueeze to have [n_batch, n_tasks, n_actions, n_features]
        # Initialize Omegas
        omegas_temp = self._init_omega(len(train_tasks))
        with torch.no_grad():
            omegas_temp = (omegas_temp / torch.sum(omegas_temp, axis=1, keepdim=True))
        omegas_temp = torch.tensor(omegas_temp).requires_grad_(True)

        # Initialize Target Reward Mappers and optimizer
        for test_task in test_tasks:
            omegas_target_task = omegas_temp.clone().detach().requires_grad_(True)

            fit_w = torch.Tensor(1, test_task.feature_dim()).uniform_(-0.01, 0.01).to(self.device)
            w_approx = torch.nn.Linear(test_task.feature_dim(), 1, bias=False, device=self.device)
            # w_approx = torch.nn.Linear(test_task.feature_dim(), 1, device=self.device)

            with torch.no_grad():
                w_approx.weight = torch.nn.Parameter(fit_w)

            #fit_w = torch.Tensor(test_task.feature_dim(), 1).uniform_(-0.01, 0.01).to(self.device)
            #w_approx = fit_w
            # Learning rate alpha (Weights)
            parameters = [
                {'params': w_approx.parameters(),
                 'lr': self.hyperparameters['learning_rate_w'],
                 'weight_decay': self.hyperparameters['weight_decay_w']},
                {'params': omegas_target_task,
                 'lr': self.hyperparameters['learning_rate_omega'],
                 'weight_decay': self.hyperparameters['weight_decay_omega']},
            ]
            optim = torch.optim.Adam(parameters)

            no_op_lambda = lambda epoch: 1 ** epoch
            lambda_omegas_lr = lambda epoch: (1 - self.hyperparameters['learning_rate_omega_decay']) ** epoch

            # This scheduler manages the lambda according to parameter group
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, [no_op_lambda, lambda_omegas_lr])
            self.test_tasks_weights.append((w_approx, optim, scheduler))
            self.omegas.append(omegas_target_task)

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
    
    def get_test_action(self, s_enc, w, omegas):
        with torch.no_grad():
            if random.random() <= self.test_epsilon:
                a = torch.tensor(random.randrange(self.n_actions)).to(self.device)
            else:
                normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))
                successor_features = self.sf.get_successors(s_enc)
                tsf = torch.sum(successor_features * normalized_omegas, axis=1)

                q = w(tsf)
                # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                # Target TSF only Use Q-Learning
                a = torch.argmax(q)
            return a
            
    def test_agent(self, task, test_index):
        R = 0.0
        w, optim, scheduler = self.test_tasks_weights[test_index]
        omegas = self.omegas[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        total_phi_loss = 0
        total_psi_loss = 0

        for target_ev_step in range(self.T):
            a = self.get_test_action(s_enc, w, omegas)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)
            a1 = self.get_test_action(s1_enc, w, omegas)

            loss_t, phi_loss, psi_loss = self.update_test_reward_mapper(w, omegas, optim, task, r, s_enc, a, s1_enc, a1)
            accum_loss += loss_t.item()
            total_phi_loss += phi_loss.item()
            total_psi_loss += psi_loss.item()

            # Index 1 is for omegas
            scheduler.step()

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T from in a random way
        if(self.total_training_steps % 5000 == 0):
            beta_loss_coefficient = self.hyperparameters['beta_loss_coefficient']
            self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, accum_loss, total_phi_loss, total_psi_loss, test_index, beta_loss_coefficient, self.T))
            self.logger.log_omegas_learning_rate(optim.param_groups[1]['lr'], test_index, (self.total_training_steps))

        # Fix it seems omegas are being cached
        self.omegas[test_index] = omegas

        return R

    def update_test_reward_mapper(self, w_approx, omegas, optim, task, r, s, a, s1, a1):

        if self.h_function is None:
            raise Exception('Affine Function (h) is not initialized')

        # Return Loss
        phi = task.features(s,a,s1)

        # h function eval
        self.h_function.eval()

        # Transformed States
        t_states = []
        t_next_states = []

        normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))

        with torch.no_grad():
            for g in self.g_functions:
                state = g(s)
                next_state = g(s1)
                t_states.append(state)
                t_next_states.append(next_state)

            # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
            t_states = torch.vstack(t_states).unsqueeze(1)
            t_next_states = torch.vstack(t_next_states).unsqueeze(1)

        weighted_states = torch.sum(t_states * normalized_omegas, axis=1)
        weighted_next_states = torch.sum(t_next_states * normalized_omegas, axis=1)
        affine_states = self.h_function(weighted_states) + self.h_function(weighted_next_states)
        transformed_phi = phi * affine_states.squeeze(0)

        with torch.no_grad():
            successor_features = self.sf.get_successors(s)
            next_successor_features = self.sf.get_next_successors(s1)
            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

        next_tsf = transformed_phi + self.gamma * torch.sum(next_successor_features * normalized_omegas, axis=1)[:, a1 ,:]

        tsf = torch.sum(successor_features * normalized_omegas, axis=1)[:, a ,:]
        loss_task = torch.nn.MSELoss()

        r_fit = w_approx(transformed_phi)

        # Hyperparameters and L1 Regularizations
        beta_loss_coefficient = torch.tensor(self.hyperparameters['beta_loss_coefficient'])
        lasso_coefficient = torch.tensor(self.hyperparameters['omegas_l1_coefficient'])
        # L1 Norm
        lasso_regularization = torch.norm(omegas, 1)

        l1 = loss_task(tsf, next_tsf)
        l2 = loss_task(r_fit, r_tensor)

        loss = l1 + (beta_loss_coefficient * l2) + (lasso_coefficient * lasso_regularization)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Sum_i omega_i = 1
        with torch.no_grad():
            epsilon = 1e-7
            omegas.clamp_(epsilon) 

        if(self.total_training_steps % 1000 == 0 and random.randint(1, 1000) < 10):
            print(f'########### BEGIN TARGET TASKS #################')
            print(f'Target Task Index {task}')
            print(f'Target Task {task} Omegas Gradients {omegas.grad}')
            print(f'Target Task {task} Weights {w_approx.weight}')
            print(f'Target Task {task} Weights Gradients {w_approx.weight.grad}')
            print(f'########### END TARGET TASKS #################')

        # h function train
        self.h_function.train()
        # Loss, phi_loss, psi_loss
        return loss, l2, l1

    def get_target_reward_mapper_error(self, r, loss, phi_loss, psi_loss, task_index, target_loss_coefficient, ts):
        return_dict = {
            'task': task_index,
            # Total steps and ev_frequency
            'reward': r,
            'steps': ((500 * (self.total_training_steps // 1000)) + ts),
            'w_error': loss,
            'psi_loss': psi_loss,
            'phi_loss': phi_loss,
            #'w_error': torch.linalg.norm(self.test_tasks_weights[task_index] - task.get_w())
            'target_loss_coefficient': target_loss_coefficient
            }
        return return_dict
