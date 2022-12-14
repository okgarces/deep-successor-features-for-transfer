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


class TSFDQN_PHI(Agent):

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
        super(TSFDQN_PHI, self).__init__(*args, **kwargs)
        self.sf = deep_sf
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.device = get_torch_device()

        # phi learning
        self.replay_buffer_handle = replay_buffer_handle
        self.lambda_phi_model = lambda_phi_model
        self.phi: Tuple[ModelTuple, ModelTuple]
        self.updates_since_phi_target_updated = []
        self.test_tasks_weights = []
        self.buffers = []
        self.loss_coefficients = []
        self.target_loss_coefficients = []

        # TSF
        self.g_functions = []

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

        # Transformed Successor Features 
        self.g_function = self.g_functions[index]

        # Set buffer to current task buffer
        self.buffer = self.buffers[index]
        self.loss_coefficient = self.loss_coefficients[index]
        
    def get_Q_values(self, s, s_enc):
        with torch.no_grad():
            q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
            if not self.use_gpi:
                c = self.task_index
            self.c = c
            return q[:, c,:]
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        # phis should not be in buffer
        # remember this experience
        self.buffer.append(s_enc, a, r, torch.tensor([]), s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:
            transitions = self.buffer.replay()
            losses = self.update_deep_models(transitions, self.phi, self.task_index, self.loss_coefficient, self.use_gpi)

            if isinstance(losses, tuple):
                total_loss, psi_loss, phi_loss, loss_coefficient = losses
                self.logger.log_losses(total_loss.item(), psi_loss.item(), phi_loss.item(), [loss_coefficient.item()], self.total_training_steps)

    def reset(self):
        super(TSFDQN_PHI, self).reset()
        self.sf.reset()
        self.updates_since_phi_target_updated = []
        self.g_functions = []

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
            self.n_actions = task.action_count()
            # TODO Is this n_features is needed?
            self.n_features = task.feature_dim()
            self.s_enc_dim = task.encode_dim()
            self.action_dim = task.action_dim()
            if self.encoding == 'task':
                self.encoding = task.encode

            self.phi = self.init_phi_model()
            self.h_function = self.init_h_function(self.n_features)

        # Transformed Successor features
        self.omegas = self.init_omegas(self.n_features, self.n_tasks)
        g_function = self.init_g_function(self.n_features, self.s_enc_dim)
        self.g_functions.append(g_function)
        # Buffers and Loss coefficients
        self.buffers.append(self.replay_buffer_handle())
        self.loss_coefficients.append(self.init_loss_coefficients())

        self.sf.add_training_task(task, source=None)

    ############# phi Model Learning ####################
    def init_loss_coefficients(self):
        weights = torch.ones(1, requires_grad=True, device=self.device)
        return weights

    def init_omegas(self, n_features, n_tasks):
        # This is to approaxh weighted norm for linear
        omegas = torch.nn.utils.weight_norm(torch.nn.Linear(n_features * n_tasks, n_features, bias=False, device=self.device))
        return omegas

    def init_h_function(self, n_features):
        g_function = torch.nn.Linear(n_features, n_features, bias=False, device=self.device)
        return g_function

    def init_g_function(self, n_features, s_enc_dim):
        g_function = torch.nn.Linear(s_enc_dim, n_features, bias=False, device=self.device)
        return g_function

    def init_phi_model(self):
        # This is only if there is only one shared phi
        phi_model, phi_loss, phi_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)
        phi_target_model, phi_target_loss, phi_target_optim = self.lambda_phi_model(self.s_enc_dim, self.action_dim, self.n_features)

        update_models_weights(phi_model, phi_target_model)
        # TODO This variable could be removed
        self.updates_since_phi_target_updated.append(0)

        return (phi_model, phi_loss, phi_optim), (phi_target_model, phi_target_loss, phi_target_optim)

    def compute_target_phi(self, phi_input):
        # This function is the function which forward using target trasnformed feature function
        phi_model_tuple, *_ = self.phi
        phi_model, *_ = phi_model_tuple

        return phi_model(phi_input) 

    def update_deep_models(self, transitions, phis_model, policy_index, loss_coefficient, use_gpi):
        torch.autograd.set_detect_anomaly(True)

        if transitions is None:
            return

        # phis aren't going to be calculated instead of use ReplayBuffer
        states, actions, rs, _, next_states, gammas = transitions
        n_batch = len(gammas)
        indices = torch.arange(n_batch)
        gammas = gammas.reshape((-1, 1))
        # Compute phis
        # Phi version only have one shared phi
        phi_model_tuple, _ = phis_model
        phi_model, phi_loss, _ = phi_model_tuple
        # target_phi_model, *_ = target_phi_tuple 
        task_w = self.sf.fit_w[policy_index]

        # Compute phi in source tasks
        input_phi = torch.concat([states.to(self.device), actions.reshape((n_batch,1)), next_states.to(self.device)], axis=1)
        phis = phi_model(input_phi) 
        s_transformed = self.h_function(self.g_function(states))
        s1_transformed = self.h_function(self.g_function(next_states))
        # \phi = phis * (h(g^{-1}(s)) + h(g^{-1}(s1)))
        phis = phis * (s_transformed + s1_transformed)

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
        psi_model, psi_loss, _ = psi_tuple
        target_psi_model, *_ = target_psi_tuple

        current_psi = psi_model(states)
        targets = phis + gammas * target_psi_model(next_states)[indices, next_actions,:]

        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets

        # Only one phi vector with a weight_decay to learn smooth functions
        # psi_optim.zero_grad()
        parameters = [
                {'params': self.h_function.parameters(), 'lr': 1e-3},
                {'params': self.g_function.parameters(), 'lr': 1e-3},
                {'params': psi_model.parameters(), 'lr': 1e-3},
                {'params': phi_model.parameters(), 'lr': 1e-3},
                {'params': task_w.parameters(), 'lr': 1e-3},
                #{'params': self.h_function.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': self.g_function.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': psi_model.parameters(), 'lr': 1e-3 , 'weight_decay': 1e-4 },
                #{'params': phi_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': task_w.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': loss_coefficient, 'lr': 1e-3, 'weight_decay': 1e-3 }
                {'params': loss_coefficient, 'lr': 1e-3},
        ]

        #current_psi_clone = current_psi
        #merge_current_target_psi_clone = merge_current_target_psi
        r_fit = task_w(phis)
        phi_loss_value = phi_loss(r_fit, rs).unsqueeze(0)

        optim = torch.optim.Adam(parameters)
        optim.zero_grad()

        psi_loss_value = psi_loss(current_psi, merge_current_target_psi).unsqueeze(0)
        loss = phi_loss_value + loss_coefficient * psi_loss_value

        # This is only to avoid gradient exploiding or vanishing. While we 
        # find a specific lr and wd
        if (torch.isnan(loss) or torch.isinf(loss)):
            print(f'loss {loss}')
            print(f'phi_loss_value {phi_loss_value}')
            print(f'psi_loss_value {psi_loss_value}')
            print(f'task_w weights {task_w.weight}')
            print(f'phi model {[param.data for param in phi_model.parameters()]}')
            print(f'task_w {task_w(phis)}')
            print(f'phis {phis}')

        loss.backward()

        for param_dict in parameters:
            for params in param_dict.get('params', []):
                params.grad.data.clamp_(-1, 1)

        optim.step()


        # Finish train the SF network
        # update the target SF network
        self.sf.updates_since_target_updated[policy_index] += 1
        if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev:
            update_models_weights(psi_model, target_psi_model)
            # We don't need target phi model
            # update_models_weights(phi_model, target_phi_model)
            self.sf.updates_since_target_updated[policy_index] = 0

        # Return loss for statistics
        return loss, psi_loss_value, phi_loss_value, loss_coefficient

    ############## Progress and Stats ###################

    def get_target_reward_mapper_error(self, r, loss, psi_loss, phi_loss, task, target_loss_coefficient, ts):
        return_dict = {
            'task': task,
            # Total steps and ev_frequency
            'reward': r,
            'steps': ((500 * (self.total_training_steps // 1000)) + ts),
            'w_error': loss,
            'psi_loss': psi_loss,
            'phi_loss': phi_loss,
            'target_loss_coefficient': target_loss_coefficient
            }
        return return_dict
    
    def get_progress_strings(self):
        sample_str, reward_str = super(TSFDQN_PHI, self).get_progress_strings()
        gpi_percent = self.sf.GPI_usage_percent(self.task_index)
        w_error = torch.linalg.norm(self.sf.fit_w[self.task_index].weight - self.sf.true_w[self.task_index])
        gpi_str = 'GPI% \t {:.4f} \t w_err \t {:.4f}'.format(gpi_percent, w_error)
        return sample_str, reward_str, gpi_str
            
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

            self.test_tasks_weights.append(w_approx)
            self.target_loss_coefficients.append(self.init_loss_coefficients())

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
                # Flatten n_tasks

                # Swap axes from [n_batch, n_tasks, n_actions, n_features]
                # to [n_batch, n_actions, n_tasks, n_features]
                psi_flatten = psi.swapaxes(1,2).flatten(start_dim=2)
                target_psi = self.omegas(psi_flatten)

                # Target pso only Use Q-Learning
                q = w(target_psi)
                a = torch.argmax(q)
            return a
            
    def test_agent(self, task, test_index):
        R = 0.0
        w = self.test_tasks_weights[test_index]
        target_loss_coefficient = self.target_loss_coefficients[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        total_loss = 0
        total_phi_loss = 0
        total_psi_loss = 0

        for _ in range(self.T):
            a = self.get_test_action(s_enc, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            # loss_t = self.update_test_reward_mapper(w, r, s, a, s1).item()
            loss, psi_loss, phi_loss = self.update_target_models(w, target_loss_coefficient, r, s_enc, a, s1_enc)
            total_loss += loss
            total_phi_loss += phi_loss
            total_psi_loss += psi_loss
            # loss_t = self.update_test_reward_mapper_ascent_version(w, r, s, a, s1, test_index)

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T
        self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, total_loss, total_psi_loss, total_phi_loss, test_index, target_loss_coefficient.item() ,self.T))

        return R

    def update_target_models(self, w_approx, target_loss_coefficient, r, s_enc, a, s1_enc):
        #with torch.no_grad():
        input_phi = torch.concat([s_enc.flatten().to(self.device), a.flatten().to(self.device), s1_enc.flatten().to(self.device)]).to(self.device)
        phi = self.compute_target_phi(input_phi)
        psi = self.sf.get_successors(s_enc)
        next_psi = self.sf.get_next_successors(s1_enc)

        parameters = [
                {'params': w_approx.parameters()},
                {'params': self.omegas.parameters()},
                #{'params': loss_coefficient.parameters(), 'lr': 1e-2}
                #{'params': psi_model.parameters(), 'lr': 1e-3 , 'weight_decay': 1e-4 },
                #{'params': phi_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': task_w.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': loss_coefficient, 'lr': 1e-3, 'weight_decay': 1e-3 }
                {'params': target_loss_coefficient},
        ]
        #optim = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)
        optim = torch.optim.Adam(parameters, lr=1e-3)
        loss_task = torch.nn.MSELoss()

        with torch.no_grad():
            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

            t_s_values = []
            t_s1_values = []
            for g_function in self.g_functions:
                t_s_values.append(g_function(s_enc))
                t_s1_values.append(g_function(s1_enc))

        s_transformed = self.omegas(torch.concat(t_s_values).flatten())
        s1_transformed = self.omegas(torch.concat(t_s1_values).flatten())

        transformed_phi = phi * (self.h_function(s_transformed) + self.h_function(s1_transformed))

        with torch.no_grad():
        # 1.0 is the gamma
            psi_flatten = psi.swapaxes(1,2).flatten(start_dim=2)
            next_psi_flatten = next_psi.swapaxes(1,2).flatten(start_dim=2)

        transformed_psi = self.omegas(psi_flatten)
        transformed_next_psi = transformed_phi + self.gamma * self.omegas(next_psi_flatten)

        r_fit_transfer = w_approx(transformed_phi)

        optim.zero_grad()
        # TODO psi loss missing

        phi_loss = loss_task(r_fit_transfer, r_tensor)
        psi_loss = loss_task(transformed_psi, transformed_next_psi)

        loss =  phi_loss + (target_loss_coefficient * psi_loss)
        # Otherwise gradients will be computed to inf or nan.
        loss.backward()

        if (torch.isnan(loss) or torch.isinf(loss)):
            print(f'loss {loss}')
            print(f's_transformed weights {s_transformed}')
            print(f's1_transformed weights {s1_transformed}')
            print(f'h_transformed weights {self.h_function.weight}')
            print(f'g_transformed weights {self.g_function.weight}')
            print(f'task_w {w_approx}')

        for param_dict in parameters:
            for params in param_dict.get('params', []):
                params.grad.data.clamp_(-1e10, 1e10)

        optim.step()

        with torch.no_grad():
            max_coefficient = 1e6
            min_coefficient = 1e-2
            target_loss_coefficient.data.clamp_(min_coefficient, max_coefficient)

        # If inf loss
        return loss, psi_loss, phi_loss
