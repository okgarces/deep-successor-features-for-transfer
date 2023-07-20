import numpy as np
import torch

from utils.torch import get_torch_device, update_models_weights
from utils.logger import get_logger_level
import random

class ReplayBuffer:
    
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
        states = torch.vstack(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.vstack(rewards).to(device)
        phis = torch.vstack(phis).to(device)
        next_states = torch.vstack(next_states).to(device)
        gammas = torch.tensor(gammas).to(device)
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

################################ Features ##################################
class PhiFunction(torch.nn.Module):

    def __init__(self, state_space, action_space, feature_dimension):
        super(PhiFunction, self).__init__()

        self.alpha_phi: int = 1e-3

        layers = [
            torch.nn.Linear(state_space * 2 + action_space, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128 * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(128 * 2, feature_dimension)
        ]

        self._model = torch.nn.Sequential(*layers).to(device)
        self.optimiser = torch.optim.Adam(self._model.parameters(), lr=self.alpha_phi)
        self._model.train() # Train mode

    def forward(self, state, action, next_state):
        if action.ndim == 0:
            action = action.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(1)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(0)

        inputs = torch.cat([state, action, next_state], axis=1)
        return self._model(inputs)

    def set_eval(self):
        self._model.eval()

class DeepTSF:
    """
    A successor feature representation implemented using Keras. Accepts a wide variety of neural networks as
    function approximators.
    """
    
    def __init__(self, pytorch_model_handle, use_true_reward, target_update_ev=1000, **kwargs):
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
        self.use_true_reward = use_true_reward
        self.hyperparameters = kwargs.get('hyperparameters', {})
        self.alpha_w = self.hyperparameters.get('learning_rate_w')

        self.pytorch_model_handle = pytorch_model_handle
        self.target_update_ev = target_update_ev
        
        self.device = get_torch_device()
    
    def reset(self):
        """
        Removes all trained successor feature representations from the current object, all learned rewards,
        and all task information.
        """
        self.n_tasks = 0
        self.psi = []
        self.fit_w = []
        self.gpi_counters = []

        self.updates_since_target_updated = []


    def add_training_task(self, task, source=None, g_function_model={}, h_function_model={}):
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

        n_features = task.feature_dim()
        fit_w = torch.Tensor(1, n_features).uniform_(-0.01, 0.01).to(self.device)
        w_approx = torch.nn.Linear(n_features, 1, bias=False, device=self.device)

        with torch.no_grad():
            w_approx.weight = torch.nn.Parameter(fit_w)

        self.fit_w.append(w_approx)

        # add successor features to the library
        self.psi.append(self.build_successor(task, source, w_approx, g_function_model, h_function_model))
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
    
        
    def build_successor(self, task, source=None, task_w={}, g_function={}, h_function={}):
        
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
        params = [
                {'params': model.parameters(),
                 'lr': self.hyperparameters['learning_rate_sf'],
                 'weight_decay': self.hyperparameters['weight_decay_sf']},
                {'params': task_w.parameters(),
                 'lr': self.hyperparameters['learning_rate_w'],
                 'weight_decay': self.hyperparameters['weight_decay_w']},
                {'params': g_function.parameters(),
                 'lr': self.hyperparameters['learning_rate_g'],
                 'weight_decay': self.hyperparameters['weight_decay_g']},
                {'params': h_function.parameters(),
                 'lr': self.hyperparameters['learning_rate_h'],
                 'weight_decay': self.hyperparameters['weight_decay_h']},
            ]

        optim = torch.optim.Adam(params)
        ## build target model and copy the weights 
        # This is ModelTuple
        target_model, _, _ = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)
        # target_model.set_weights(model.parameters())
        update_models_weights(model, target_model)
        self.updates_since_target_updated.append(0)

        # Set target model to eval
        target_model.eval()
        
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
    
    def get_next_successor(self, state, policy_index):
        _, next_psi_tuple = self.psi[policy_index]
        psi, *_ = next_psi_tuple
        return psi(state)

    def get_next_successors(self, state):
        #return self.all_output_model.predict_on_batch(state)
        predictions = []
        for policy_index in range(len(self.psi)):
            predictions.append(self.get_next_successor(state, policy_index))

        return torch.stack(predictions, axis=1).to(self.device)

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

################################## Agent ####################################################
class TSFDQN:

    def __init__(self, deep_sf, buffer_handle, gamma, T, encoding, epsilon=0.1, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=100, use_gpi=True, test_epsilon=0.03, **kwargs):
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
        self.hyperparameters = kwargs.get('hyperparameters', {})

        self.logger = get_logger_level()
        self.device = get_torch_device()
        self.test_tasks_weights = []

        # Sequential Successor Features
        self.buffers = []
        self.learnt_phi = None # This is learnt in pre-train stage

        # Transformed Successor Features
        self.omegas_per_source_task = []
        self.omegas = []
        self.g_functions = []
        self.h_function = None

    # ===========================================================================
    # TASK MANAGEMENT
    # ===========================================================================
    def reset(self):
        """
        Resets the agent, including all value functions and internal memory/history.
        """
        self.tasks = []
        self.phis = []
        
        # reset counter history
        self.cum_reward = 0.
        self.reward_hist = []
        self.cum_reward_hist = []

        # Successor feature reset
        self.sf.reset()

        # Buffer reset
        for buffer in self.buffers:
            buffer.reset()
    
    # ===========================================================================
    # TRAINING
    # ===========================================================================
    def _epsilon_greedy(self, q):
        q = q.flatten()
        assert q.size()[0] == self.n_actions
        
        # sample from a Bernoulli distribution with parameter epsilon
        if random.random() <= self.epsilon:
            a = torch.tensor(random.randrange(self.n_actions)).to(device)
        else:
            a = torch.argmax(q)
        
        # decrease the exploration gradually
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return a
    
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
            if self.episode > 1:
                self.episode_reward_hist.append(self.episode_reward)

            # Log performance in source tasks
            self.logger.log_source_performance(self.task_index, self.episode_reward, self.total_training_steps)

        # compute the Q-values in the current state
        q = self.get_Q_values(self.s, self.s_enc)
        
        # choose an action using the epsilon-greedy policy
        a = self._epsilon_greedy(q)
        
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
        self.cum_reward += r
        
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
            
        if self.steps % self.save_ev == 0:
            self.reward_hist.append(self.reward)
            self.cum_reward_hist.append(self.cum_reward)
        
        # viewing
        if viewer is not None and self.episode % n_view_ev == 0:
            viewer.update()

    def set_active_training_task(self, index):
        """
        Sets the task at the requested index as the current task the agent will train on.
        The index is based on the order in which the training task was added to the agent.
        """
        
        # set the task
        self.task_index = index
        self.active_task = self.tasks[index]

        if self.learnt_phi is not None:
            self.phi = self.learnt_phi
        else:
            self.phi = self.phis[index]
        
        # reset task-dependent counters
        self.s = self.s_enc = None
        self.new_episode = True
        self.episode, self.episode_reward = 0, 0.
        self.steps_since_last_episode, self.reward_since_last_episode = 0, 0.
        self.steps, self.reward = 0, 0.
        self.epsilon = self.epsilon_init
        self.episode_reward_hist = []
        
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
        with torch.no_grad():
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

        if print_debug:
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
        if print_debug:
            if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev - 1:
                print(f'########### BEGIN #################')
                print(f'Affine transformed states {torch.norm(affine_transformed_states, dim=0)} task {policy_index}')
                print(f'g functions values states {torch.norm(transformed_state, dim=0)} task {policy_index}')
                print(f'g functions values next states {torch.norm(transformed_next_state, dim=0)} task {policy_index}')
                print(f'phis values {torch.norm(phis, dim=0)} task {policy_index}')
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
        else:
            gpi_percent = None

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
            'w_err': 0
        }
        return return_dict

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

        num_episodes = 10

        for i in range(num_episodes):
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

        R /= num_episodes

        # Log accum loss for T from in a random way
        beta_loss_coefficient = self.hyperparameters['beta_loss_coefficient']
        self.logger.log_target_error_progress(self.get_target_reward_mapper_error(R, accum_loss, total_phi_loss, total_psi_loss, test_index, beta_loss_coefficient))
        self.logger.log_omegas_learning_rate(optim.param_groups[1]['lr'], test_index, (self.total_training_steps))

        # Fix it seems omegas are being cached
        self.omegas[test_index] = omegas

        return R

    def update_test_reward_mapper(self, w_approx, omegas, optim, task, r, s, a, s1, a1):

        if self.h_function is None:
            raise Exception('Affine Function (h) is not initialized')

        # Return Loss
        with torch.no_grad():
            phi = task.features(s,a,s1) if self.learnt_phi is None else self.phi(s,a,s1)

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

        if print_debug:
            if(self.total_training_steps % 1000 == 0 and random.randint(1, 1000) < 10):
                print(f'########### BEGIN TARGET TASKS #################')
                print(f'Target Task Index {task}')
                print(f'Target Task {task} Omegas Gradients {omegas.grad}')
                print(f'Target Task {task} Weights {w_approx.weight}')
                print(f'Target Task {task} Weights Gradients {w_approx.weight.grad}')
                print(f'Target Task {task} affine states {affine_states.squeeze(0)}')
                print(f'Target Task {task} phi values {phi}')
                print(f'Target Task {task} transformed state {weighted_states}')

                print(f'########### END TARGET TASKS #################')

        # h function train
        self.h_function.train()
        # Loss, phi_loss, psi_loss
        return loss, l2, l1

    def get_target_reward_mapper_error(self, r, loss, phi_loss, psi_loss, task_index, target_loss_coefficient):
        return_dict = {
            'task': task_index,
            # Total steps and ev_frequency
            'reward': r,
            'steps': (self.T * (1 + self.total_training_steps // 1_000)),
            'w_error': loss,
            'psi_loss': psi_loss,
            'phi_loss': phi_loss,
            #'w_error': torch.linalg.norm(self.test_tasks_weights[task_index] - task.get_w())
            'target_loss_coefficient': target_loss_coefficient
            }
        return return_dict

    def pre_train(self, train_tasks, n_samples_pre_train, n_cycles=5):
        # The result of this pre_train is having the first n_samples to fit reconstruct the feature function
        # At the end the idea is to override a phi function: self.learnt_phi
        first_task = train_tasks[0]
        buffer = ReplayBuffer()
        n_actions = first_task.action_count()

        # Here I put all the model
        phi_learn_model = PhiFunction(first_task.encode_dim(), first_task.action_dim(), first_task.feature_dim())

        losses = []

        fit_ws = []
        fit_ws_optim = []

        # Initialize weights
        # Having a random policy using
        for task in train_tasks:
            fit_w = torch.nn.Linear(task.feature_dim(), 1, bias=False, device=device)
            with torch.no_grad():
                fit_w.weight = torch.nn.Parameter(torch.Tensor(1, task.feature_dim()).uniform_(-0.01, 0.01).to(device))

            fit_w_optim = torch.optim.Adam(fit_w.parameters(), lr=1e-3)

            fit_ws.append(fit_w)
            fit_ws_optim.append(fit_w_optim)

        for cycle in range(n_cycles):
            for task_id, task in enumerate(train_tasks):
                # Having a random policy using
                fit_w = fit_ws[task_id]
                fit_w_optim = fit_ws_optim[task_id]

                s_enc = task.encode(task.initialize())

                for sample in range(n_samples_pre_train):
                    a = random.randrange(n_actions)
                    s1, r, terminal = task.transition(a)
                    s1_enc = task.encode(s1)

                    # state, action, reward.float(), phi, next_state, gamma
                    buffer.append(s_enc, torch.tensor(a, device=device),
                                  torch.tensor(r, device=device, dtype=torch.float32), torch.tensor([0]), s1_enc,
                                  torch.tensor([0]))

                    # moving to next_state
                    s_enc = s1_enc

                    if terminal:
                        s_enc = task.encode(task.initialize())

                    # update phi model using mini batch
                    replay = buffer.replay()

                    if replay is not None:
                        state_batch, action_batch, reward_batch, _, next_state_batch, _ = replay

                        phis = phi_learn_model(state_batch, action_batch, next_state_batch)  # [B, feature_dim]
                        linear_combination = fit_w(phis)

                        fit_w_optim.zero_grad()
                        phi_learn_model.optimiser.zero_grad()

                        loss = torch.nn.functional.mse_loss(reward_batch, linear_combination)
                        loss.backward()
                        phi_learn_model.optimiser.step()
                        fit_w_optim.step()

                        losses.append(loss.item())

        buffer.reset()
        self.learnt_phi = phi_learn_model
        self.learnt_phi.set_eval()

        return losses


################################################################################################################
################################################################################################################

# -*- coding: UTF-8 -*-
from tasks.reacher_phi import Reacher_PHI
from tasks.cartpole_phi import Cartpole_PHI
from utils.config import parse_config_file
from utils.torch import set_torch_device, get_activation
from utils.logger import set_logger_level
from utils.types import ModelTuple

import torch
from collections import OrderedDict

import pandas as pd

# read parameters from config file
# config_params = parse_config_file('reacher.cfg')
config_params = parse_config_file('cartpole_phi.cfg')

gen_params = config_params['GENERAL']
n_samples = gen_params['n_samples']
use_gpu = gen_params.get('use_gpu', False)  # Default GPU False
gpu_device_index = gen_params.get('gpu_device_index', 0)  # Default GPU False
use_logger = gen_params.get('use_logger', False)  # Default GPU False
n_cycles_per_task = gen_params.get('cycles_per_task', 1)  # Default GPU False

task_params = config_params['TASK']
goals = task_params['train_targets']
test_goals = task_params['test_targets']
all_goals = goals + test_goals

agent_params = config_params['AGENT']
sfdqn_params = config_params['SFDQN']

phi_learning_params = config_params['PHI']
n_features = phi_learning_params['n_features']

# Config GPU for Torch and logger
device = set_torch_device(use_gpu=use_gpu, gpu_device_index=gpu_device_index)
logger = set_logger_level(use_logger=use_logger)

print_debug = False


# tasks
def generate_tasks(include_target, experiment='reacher'):
    if experiment == 'reacher':
        train_tasks = [Reacher_PHI(all_goals, i, n_features, include_target) for i in range(len(goals))]
        test_tasks = [Reacher_PHI(all_goals, i + len(goals), n_features, include_target) for i in
                      range(len(test_goals))]
    elif experiment == 'cartpole':
        train_tasks = [Cartpole_PHI(i, n_features, goal) for i, goal in enumerate(goals)]
        test_tasks = [Cartpole_PHI(i, n_features, goal) for i, goal in enumerate(test_goals)]

    return train_tasks, test_tasks


def sf_model_lambda(num_inputs: int, output_dim: int, reshape_dim: tuple, reshape_axis: int = 1) -> ModelTuple:
    model_params = sfdqn_params['model_params']

    layers = OrderedDict()
    number_layers = len(model_params['n_neurons'])

    # Layers settings
    first_layer_neurons = model_params['n_neurons'][0]
    last_layer_neurons = model_params['n_neurons'][-1]

    # Input Layer
    input_layer = torch.nn.Linear(num_inputs, first_layer_neurons)
    layers['layer_input'] = input_layer

    # Hidden Layer
    for index, n_neurons, activation in zip(range(number_layers), model_params['n_neurons'],
                                            model_params['activations']):
        linear = torch.nn.Linear(n_neurons, n_neurons)
        activation_function = get_activation(activation)()
        layers[f'layer_{index}'] = linear
        layers[f'activation_layer_{index}'] = activation_function

    # Output Layers
    output_layer = torch.nn.Linear(last_layer_neurons, output_dim)
    unflatten_layer = torch.nn.Unflatten(reshape_axis, reshape_dim)
    layers['layer_output'] = output_layer
    layers['layer_unflatten'] = unflatten_layer

    model = torch.nn.Sequential(layers).to(device)

    loss = torch.nn.MSELoss().to(device)

    return model, loss, None


def replay_buffer_handle():
    return ReplayBuffer(sfdqn_params['buffer_params'])


def main_train(environment):
    train_tasks, test_tasks = generate_tasks(False, environment)

    # build SFDQN
    print('building TSFDQN Sequential')
    print(f'PyTorch Seed {torch.seed()}')
    deep_sf = DeepTSF(pytorch_model_handle=sf_model_lambda, **sfdqn_params)
    sfdqn = TSFDQN(deep_sf=deep_sf, buffer_handle=replay_buffer_handle,
                   **sfdqn_params, **agent_params)

    # Pre train
    print('pre training SFDQN sequential')
    losses = sfdqn.pre_train(train_tasks, 5_000)  # 5_000 steps on each task. as pre train

    pd.DataFrame(losses).plot()
    # # train SFDQN
    print('training TSFDQN Sequential')
    sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'],
                cycles_per_task=n_cycles_per_task)
    print('End Training TSFDQN Sequential')

if __name__ == '__main__':
    main_train('cartpole')