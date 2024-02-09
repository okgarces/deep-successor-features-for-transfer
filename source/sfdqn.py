# -*- coding: UTF-8 -*-
import numpy as np
import torch

from tasks.task import Task
from utils.torch import update_models_weights
from utils.logger import get_logger_level
import random


############################################ Successor Features ###################################################

class DeepSF:
    """
    A successor feature representation implemented using Keras. Accepts a wide variety of neural networks as
    function approximators.
    """
    
    def __init__(self, pytorch_model_handle, use_true_reward=False, target_update_ev=1000, device=None, **kwargs):
        """
        Creates a new deep representation of successor features.
        
        Parameters
        ----------
        pytorch_model_handle: () -> ModelTuple
            a function from an input tensor to a ModelTuple
            the ModelTuple model must have outputs reshaped to [None, n_actions, n_features], where
                None corresponds to the batch dimension
                n_actions is the number of actions of the MDP
                n_features is the number of state features to learn SFs
        target_update_ev : integer 
            how often to update the target network, measured by the number of training calls
        """
        #super(DeepSF, self).__init__(*args, **kwargs)
        self.use_true_reward = use_true_reward
        self.hyperparameters = kwargs.get('hyperparameters', {})
        self.alpha_w = self.hyperparameters.get('learning_rate_w')

        self.pytorch_model_handle = pytorch_model_handle
        self.target_update_ev = target_update_ev

        self.device = device
        
    def reset(self):
        """
        Removes all trained successor feature representations from the current object, all learned rewards,
        and all task information.
        """
        self.n_tasks = 0
        self.psi = []
        self.true_w = []
        self.fit_w = []
        self.gpi_counters = []
        self.updates_since_target_updated = []

    def GPI_usage_percent(self, task_index):
        """
        Counts the number of times that actions were transferred from other tasks.
        
        Parameters
        ----------
        task_index : integer
            the index of the task
        
        Returns
        -------
        float : the (normalized) number of actions that were transferred from other
            tasks in GPi.
        """
        counts = self.gpi_counters[task_index]        
        return 1. - (float(counts[task_index]) / np.sum(counts))


    def GPI(self, state, task_index, update_counters=False):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        task_index : integer
            the index of the task in which the GPI action will be used
        update_counters : boolean
            whether or not to keep track of which policies are active in GPI
        
        Returns
        -------
        torch.Tensor : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        torch.Tensor : the tasks that are active in each state of state_batch in GPi
        """
        q, task = self.GPI_w(state, self.fit_w[task_index])
        if update_counters:
            self.gpi_counters[task_index][task] += 1
        return q, task

    def add_training_task(self, task, source=None):
        """
        Adds a successor feature representation for the specified task.
        
        Parameters
        ----------
        task : Task
            a new MDP environment for which to learn successor features
        source : integer
            if specified and not None, the parameters of the successor features for the task at the source
            index should be copied to the new successor features, as suggested in [1]
        """
        
        # build new reward function
        true_w = task.get_w()

        n_features = task.feature_dim()
        fit_w = torch.Tensor(1, n_features).uniform_(-0.01, 0.01).to(self.device)
        w_approx = torch.nn.Linear(n_features, 1, bias=False, device=self.device)

        with torch.no_grad():
            w_approx.weight = torch.nn.Parameter(fit_w)

        self.true_w.append(true_w)
        self.fit_w.append(w_approx)

        # add successor features to the library
        self.psi.append(self.build_successor(task, source, w_approx))
        self.n_tasks = len(self.psi)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))

    def GPI_w(self, state, w):
        """
        Implements generalized policy improvement according to [1]. 
        
        Parameters
        ----------
        state : object
            a state or collection of states of the MDP
        w : numpy array
            the reward parameters of the task to control
        
        Returns
        -------
        torch.Tensor : the maximum Q-values computed by GPI for selecting actions
        of shape [n_batch, n_tasks, n_actions], where:
            n_batch is the number of states in the state argument
            n_tasks is the number of tasks
            n_actions is the number of actions in the MDP 
        torch.Tensor : the tasks that are active in each state of state_batch in GPi
        """
        psi = self.get_successors(state)
        #q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
        # in phi learning w is a linear approximator
        q = w(psi)[:,:,:,0]
        task = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)
        return q, task
        
    def build_successor(self, task, source=None, w_approx={}):
        
        # input tensor for all networks is shared
        # TODO the environment should send the action_count, feature_dim and inputs?
        if self.n_tasks == 0:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            self.inputs = task.encode_dim()
            
        # build SF network and copy its weights from previous task
        # output shape is assumed to be [n_batch, n_actions, n_features]
        model, loss, _ = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)

        if source is not None and self.n_tasks > 0:
            source_psi_tuple, _ = self.psi[source]
            # model.set_weights(source_psi.get_weights())
            source_psi, *_ = source_psi_tuple
            update_models_weights(source_psi, model)
        
        # append predictions of all SF networks across tasks to allow fast prediction
        #expand_output = Lambda(lambda x: K.expand_dims(x, axis=1))(model.output)
         #if self.n_tasks == 0:
         #   self.all_outputs.append
         #else:
         #   self.all_outputs = concatenate([self.all_outputs, expand_output], axis=1)
        #self.all_output_model = Model(inputs=self.inputs, outputs=self.all_outputs)
        #self.all_output_model.compile('sgd', 'mse')  # dummy compile so Keras doesn't complain
        #

        ## build target model and copy the weights 
        # This is ModelTuple
        target_model, target_loss, _ = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)
        # target_model.set_weights(model.parameters())
        update_models_weights(model, target_model)
        self.updates_since_target_updated.append(0)

        # Set target model to eval
        target_model.eval()

        print(f'Initialize optimizer for SF and W {task}')
        params = [
                {'params': model.parameters(), 'lr': self.hyperparameters['learning_rate_sf'], 'weight_decay': self.hyperparameters['weight_decay_sf']},
                {'params': w_approx.parameters(), 'lr': self.hyperparameters['learning_rate_w'], 'weight_decay': self.hyperparameters['weight_decay_w']},
            ]
        optim = torch.optim.Adam(params)

        return (model, loss, optim), (target_model, None, None)
        
    def get_successor(self, state, policy_index):
        psi_tuple, _ = self.psi[policy_index]
        psi, *_ = psi_tuple
        return psi(state)
    
    def get_successors(self, state):
        #return self.all_output_model.predict_on_batch(state)
        predictions = []
        for policy_index in range(len(self.psi)):
            predictions.append( self.get_successor(state, policy_index))

        return torch.stack(predictions, axis=1).to(self.device)

    def update_successor(self, transitions, policy_index, use_gpi=True, to_propagate=True):
        if transitions is None:
            return

        states, actions, rs, phis, next_states, gammas = transitions
        n_batch = len(gammas)
        indices = torch.arange(n_batch)
        gammas = gammas.reshape((-1, 1))
        task_w = self.fit_w[policy_index]

        current_policy_index_for_gpi = [policy_index]

        # next actions come from current Successor Feature
        if use_gpi:
            q1, current_policy_index_for_gpi= self.GPI(next_states, policy_index)
            next_actions = torch.argmax(torch.max(q1, axis=1).values, axis=-1)

        else:
            sf = self.get_successor(next_states, policy_index)
            q1 = task_w(sf)
            # Do not forget: argmax according to actions, and squeeze in axis according to get n_batch
            next_actions = torch.squeeze(torch.argmax(q1, axis=1), axis=1)

        # compute the targets and TD errors
        psi_tuple, target_psi_tuple = self.psi[policy_index]
        psi_model, psi_loss, optim = psi_tuple
        target_psi_model, *_ = target_psi_tuple

        current_psi = psi_model(states)

        with torch.no_grad():
            targets = phis + gammas * target_psi_model(next_states)[indices, next_actions,:]
        
        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets
        # psi_model.train_on_batch(states, current_psi)

        optim.zero_grad()

        r_fit = task_w(phis)
        l1 = psi_loss(current_psi, merge_current_target_psi)
        l2 = psi_loss(r_fit, rs)

        loss = l1 + l2
        loss.backward()
        
        # log gradients this is only a way to track gradients from time to time
        # if self.updates_since_target_updated[policy_index] >= self.target_update_ev - 1:
        #     print(f'########### BEGIN #################')
        #     print(f'Policy Index {policy_index}')
        #     print(f' Update STEP # {self.updates_since_target_updated[policy_index]}')
        #     for params in psi_model.parameters():
        #         print(f'Gradients of Psi {params.grad}')
        #
        #     for params in task_w.parameters():
        #         print(f'Gradients of W {params.grad}')
        #
        #     for params in target_psi_model.parameters():
        #         print(f'Gradients of Psi Target {params.grad}')
        #     print(f'########### END #################')

        optim.step()

        # Finish train the SF network
        # update the target SF network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            update_models_weights(psi_model, target_psi_model)
            self.updates_since_target_updated[policy_index] = 0

        # It is not easy to read but current_policy_index_for_gpi only changes if only if we are using GPI
        if to_propagate and current_policy_index_for_gpi[-1] != policy_index:
            self.update_successor(transitions, current_policy_index_for_gpi[-1], use_gpi, to_propagate=False)

        return loss, l1, l2 

############################ AGENT #################################
class SFDQN:

    def __init__(self, deep_sf, buffer_handle, gamma, T, encoding, epsilon=0.1, epsilon_decay=1, epsilon_min=0, print_ev=1000, 
                 save_ev=100, use_gpi=True, test_epsilon=0.03, device=None, **kwargs):
        """
        Creates a new abstract reinforcement learning agent.
        
        Parameters
        ----------
        gamma : float
            the discount factor in [0, 1]
        T : integer
            the maximum length of an episode
        encoding : function
            encodes the state of the task instance into a numpy array
        epsilon : float
            the initial exploration parameter for epsilon greedy (defaults to 0.1)
        epsilon_decay : float
            the amount to anneal epsilon in each time step (defaults to 1, no annealing)
        epsilon_min : float
            the minimum allowed value of epsilon (defaults to 0)
        print_ev : integer
            how often to print learning progress
        save_ev : 
            how often to save learning progress to internal memory        
        """
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
        self.gamma = gamma
        self.T = T
        if encoding is None:
            encoding = lambda s: s
        self.encoding = encoding
        self.epsilon_init = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.print_ev = print_ev
        self.save_ev = save_ev
        self.total_training_steps = 0

        self.sf = deep_sf
        self.buffer_handle = buffer_handle
        self.use_gpi = use_gpi
        self.test_epsilon = test_epsilon

        self.logger = get_logger_level()
        self.test_tasks_weights = []
        self.hyperparameters = kwargs.get('hyperparameters', {})

        # Sequential Successor Features
        self.buffers = []
        self.device = device

    def set_active_training_task(self, index):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """
        
        # set the task
        self.task_index = index
        self.active_task = self.tasks[index]
        self.phi = self.phis[index]
        
        # reset task-dependent counters
        self.s = self.s_enc = None
        self.new_episode = True
        self.episode, self.episode_reward = 0, 0.
        self.steps_since_last_episode, self.reward_since_last_episode = 0, 0.
        self.steps, self.reward = 0, 0.
        self.epsilon = self.epsilon_init
        
        # Sequential
        self.buffer = self.buffers[index]

    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        phi = self.phi(s, a, s1)
        self.buffer.append(s_enc, a, r, phi, s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:
            transitions = self.buffer.replay()
            losses = self.sf.update_successor(transitions, self.task_index, self.use_gpi)

            # This is to update SFs with the current transitions
            # for index in range(self.n_tasks):
            #     losses = self.sf.update_successor(transitions, index, self.use_gpi)
            if self.total_training_steps % 1000 == 0:
                if isinstance(losses, tuple):
                    total_loss, psi_loss, phi_loss = losses
                    self.logger.log({f'losses/source_task_{self.task_index}/total_loss': total_loss.item(), 'timesteps': self.total_training_steps})
                    self.logger.log({f'losses/source_task_{self.task_index}/phi_loss': phi_loss.item(), 'timesteps': self.total_training_steps})
                    self.logger.log({f'losses/source_task_{self.task_index}/psi_loss': psi_loss.item(), 'timesteps': self.total_training_steps})

                # Print weights for Reward Mapper
                task_w = self.sf.fit_w[self.task_index]
                self.logger.log({f'train/source_task_{self.task_index}/weights': str(task_w.weight.clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})

    def reset(self):
        """
        Resets the agent, including all value functions and internal memory/history.
        """
        self.tasks = []
        self.phis = []
        
        # reset counter history
        self.sf.reset()

        for buffer in self.buffers:
            buffer.reset()

    def add_training_task(self, task):
        """
        Adds a training task to be trained by the agent.
        """
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)  
        self.phis.append(task.features)               
        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            if self.encoding == 'task':
                self.encoding = task.encode

        # Add new successor feature
        self.sf.add_training_task(task, source=None)
        # Append Buffer
        self.buffers.append(self.buffer_handle())

    def next_sample(self, viewer=None, n_view_ev=None):
        """
        Updates the agent by performing one interaction with the current training environment.
        This function performs all interactions with the environment, data and storage manipulations,
        training the agent, and updating all history.
        
        Parameters
        ----------
        viewer : object
            a viewer that displays the agent's exploration behavior on the task based on its update() method
            (defaults to None)
        n_view_ev : integer
            how often (in training episodes) to invoke the viewer to display agent's learned behavior
            (defaults to None)
        """
        # start a new episode
        if self.new_episode:
            self.s = self.active_task.initialize()
            self.s_enc = self.encoding(self.s)
            self.new_episode = False
            self.episode += 1
            self.steps_since_last_episode = 0
            self.episode_reward = self.reward_since_last_episode
            self.reward_since_last_episode = 0.

            # Log when new episode
            self.logger.log({f'train/source_task_{self.task_index}/episode_reward': self.episode_reward, 'episodes': self.episode})
        
        # compute the Q-values in the current state
        # Epsilon greedy exploration/exploitation
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            a = random.randrange(self.n_actions)
        else:
        
            with torch.no_grad():
                s_enc_torch = torch.tensor(self.s_enc).float().to(self.device)
                q, c = self.sf.GPI(s_enc_torch, self.task_index, update_counters=self.use_gpi)
                if not self.use_gpi:
                    c = self.task_index
                self.c = c
                q = q[:, c,:].flatten()
    
                # Assert number of actions and q values
                assert q.size()[0] == self.n_actions

                a = torch.argmax(q).item()
        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # take action a and observe reward r and next state s'
        s1, r, terminal = self.active_task.transition(a)
        s1_enc = self.encoding(s1)
        if terminal:
            gamma = 0.
            self.new_episode = True
        else:
            gamma = self.gamma
        
        # train the agent
        self.train_agent(self.s, self.s_enc, a, r, s1, s1_enc, gamma)
        
        # update counters
        self.s, self.s_enc = s1, s1_enc
        self.steps += 1
        self.reward += r
        self.steps_since_last_episode += 1
        self.reward_since_last_episode += r
        
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
        
        # viewing
        if viewer is not None and self.episode % n_view_ev == 0:
            viewer.update()
            
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
            # w_approx = torch.nn.Linear(test_task.feature_dim(), 1, device=device)

            with torch.no_grad():
                w_approx.weight = torch.nn.Parameter(fit_w)

            # LR 1e-3 and wd 1e-2
            w_hyperparams = {
                    'params': w_approx.parameters(),
                    'lr': self.hyperparameters['learning_rate_w'],
                    'weight_decay': self.hyperparameters['weight_decay_w']},
            optim = torch.optim.Adam(w_hyperparams)
            self.test_tasks_weights.append((w_approx, optim))
            
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
                            R, accum_loss = self.test_agent(test_task, test_index)
                            Rs.append(R)

                            self.logger.log({f'eval/target_task_{test_index}/total_reward': R, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/total_loss': accum_loss, 'timesteps': self.total_training_steps})

                        avg_R = np.mean(Rs)
                        return_data.append(avg_R)

                        # log average return
                        self.logger.log({f'eval/average_reward': avg_R, 'timesteps': self.total_training_steps})

                        # Every n_test_ev we can log the current progress in source task.
                        self.logger.log({f'train/source_task_{self.task_index}/total_reward': self.reward, 'timesteps': self.total_training_steps})

                    self.total_training_steps += 1
        return return_data

    ############# Target Tasks ################
    
    def get_test_action(self, s_enc: torch.Tensor, w: torch.nn.Module) -> int:
        with torch.no_grad():
            if random.random() <= self.test_epsilon:
                a = random.randrange(self.n_actions)
            else:
                psi = self.sf.get_successors(s_enc)
                q = w(psi)[:,:,:,0]
                # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                c = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)

                q = q[:, c,:]
                a = torch.argmax(q).item()
            return a
            
    def test_agent(self, task, test_index):
        R = 0.0
        w, optim = self.test_tasks_weights[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        for _ in range(self.T):
            s_enc_torch = torch.tensor(s_enc).float().to(self.device)
            a = self.get_test_action(s_enc_torch, w)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            # loss_t = self.update_test_reward_mapper(w, r, s, a, s1).item()
            loss_t = self.update_test_reward_mapper(w, optim, task, r, s_enc, a, s1_enc).item()
            accum_loss += loss_t

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        return R, accum_loss

    def update_test_reward_mapper(self, w_approx: torch.nn.Module, optim: torch.optim.Optimizer, task: Task, r: float, s: np.ndarray, a: int, s1: np.ndarray):
        # Return Loss
        phi = task.features(s,a,s1)
        phi_tensor = torch.tensor(phi).float().to(self.device).detach()
        loss_task = torch.nn.MSELoss()

        with torch.no_grad():
            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

        optim.zero_grad()

        r_fit = w_approx(phi_tensor)
        loss = loss_task(r_fit, r_tensor)
        loss.backward()
        
        optim.step()
        return loss