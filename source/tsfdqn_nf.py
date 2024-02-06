import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.torch import update_models_weights
from utils.logger import get_logger_level
import random


class ReplayBuffer:
    
    def __init__(self, *args, n_samples=1000000, n_batch=32, device=None, **kwargs):
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
        self.device = device

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
        states = torch.vstack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.vstack(rewards).to(self.device)
        phis = torch.vstack(phis).to(self.device)
        next_states = torch.vstack(next_states).to(self.device)
        gammas = torch.tensor(gammas).to(self.device)
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
        



################################ Features ##################################3
class DeepTSF:
    """
    A successor feature representation implemented using Keras. Accepts a wide variety of neural networks as
    function approximators.
    """
    
    def __init__(self, pytorch_model_handle, use_true_reward, target_update_ev=1000, device=None, **kwargs):
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
        true_w = task.get_w()

        n_features = task.feature_dim()
        fit_w = torch.Tensor(1, n_features).uniform_(-0.01, 0.01).to(self.device)
        w_approx = torch.nn.Linear(n_features, 1, bias=False, device=self.device)

        with torch.no_grad():
            w_approx.weight = torch.nn.Parameter(fit_w)

        self.true_w.append(true_w)
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


################################## NF ######################################################
# Implementation adapted from github
class Planar(nn.Module):

    """Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)
    ```
        f(z) = z + u * h(w * z + b)
    ```
    """

    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """Constructor of the planar flow

        Args:
          shape: shape of the latent variable z
          h: nonlinear function h of the planar flow (see definition of f above)
          u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2.0 / np.prod(shape))
        lim_u = np.sqrt(2)

        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("Nonlinearity is not implemented.")

    def forward(self, z, return_log_det=False):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim())),
                        keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)  # constraint w.T * u > -1
        if self.act == "tanh":
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + u * self.h(lin)

        if return_log_det:
            log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.reshape(-1))))
            return z_, log_det
        return  z_

    def inverse(self, z, return_log_det=False):
        if self.act != "leaky_relu":
            raise NotImplementedError("This flow has no algebraic inverse.")
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        a = (lin < 0) * (
            self.h.negative_slope - 1.0
        ) + 1.0  # absorb leakyReLU slope into u
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)
        dims = [-1] + (u.dim() - 1) * [1]
        u = a.reshape(*dims) * u
        inner_ = torch.sum(self.w * u, list(range(1, self.w.dim())))
        z_ = z - u * (lin / (1 + inner_)).reshape(*dims)

        if return_log_det:
            log_det = -torch.log(torch.abs(1 + inner_))
            return z_, log_det

        return z_

    @classmethod
    def build_planar_flow(cls, input_dim, output_dim, n_affine_flows):
        raise Exception('This is being modified. It is not implemented yet.')
        # TODO to implement later
        flows = [cls(input_dim).to(device) for _ in range(n_affine_flows)]
        # Last input 
        last_layer = torch.nn.Linear(input_dim, output_dim, bias=True, device=device)
        flows.append(last_layer)

        return torch.nn.Sequential(*flows).to(device)

################################## Agent ####################################################
class TSFDQN:

    def __init__(self, deep_sf, buffer_handle, gamma, T, encoding, epsilon=0.1, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=100, use_gpi=True, test_epsilon=0.03, device=None, **kwargs):
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
        self.device = device
        self.test_tasks_weights = []

        # Sequential Successor Features
        self.buffers = []

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
            a = torch.tensor(random.randrange(self.n_actions)).to(self.device)
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

            # Log when new episode
            self.logger.log({f'train/source_task{self.task_index}/episode_reward': self.episode_reward, 'episodes': self.episode})

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
        self.reward_since_last_episode += r.detach().cpu().numpy()
        self.cum_reward += r
        
        if self.steps_since_last_episode >= self.T:
            self.new_episode = True
        
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

        # Transformed Successor Feature
        self.active_g_function = self.g_functions[index]
        
    def get_Q_values(self, s, s_enc):
        with torch.no_grad():
            q, c = self.sf.GPI(s_enc, self.task_index, update_counters=self.use_gpi)
            if not self.use_gpi:
                c = self.task_index
            self.c = c
            return q[:, c,:]

    def _init_g_function(self, states_dim, output_dim, n_coupling_layers):
        # Number of planarflows = 10
        g_function = Planar(states_dim).to(self.device)
        return g_function

    def _init_h_function(self, input_dim, features_dim):
        # TODO Remove weights clamp and initial weights
        # This affine transformation
        h_function = torch.nn.Linear(input_dim, features_dim, bias=True, device=self.device)
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

            # Print weights for Reward Mapper
            if self.total_training_steps % 1000 == 0:
                if isinstance(losses, tuple):
                    total_loss, psi_loss, phi_loss = losses
                    self.logger.log({f'losses/source_task_{self.task_index}/total_loss': total_loss.clone().detach().cpu().numpy(), 'timesteps': self.total_training_steps})
                    self.logger.log({f'losses/source_task_{self.task_index}/phi_loss': phi_loss.clone().detach().cpu().numpy(), 'timesteps': self.total_training_steps})
                    self.logger.log({f'losses/source_task_{self.task_index}/psi_loss': psi_loss.clone().detach().cpu().numpy(), 'timesteps': self.total_training_steps})

                task_w = self.sf.fit_w[self.task_index]
                self.logger.log({f'train/source_task_{self.task_index}/weights': str(task_w.weight.clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})

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

        optim.zero_grad()

        r_fit = task_w(transformed_phis)
        beta_loss_coefficient = torch.tensor(self.hyperparameters['beta_loss_coefficient'])

        l1 = psi_loss(current_psi, merge_current_target_psi)
        l2 = psi_loss(r_fit, rs)
        
        loss = l1 + (beta_loss_coefficient * l2)
        loss.backward()

        # log gradients this is only a way to track gradients from time to time
        # if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev - 1:
        #     print(f'########### BEGIN #################')
        #     print(f'Affine transformed states {torch.norm(affine_transformed_states, dim=0)} task {policy_index}')
        #     print(f'g functions values states {torch.norm(transformed_state, dim=0)} task {policy_index}')
        #     print(f'g functions values next states {torch.norm(transformed_next_state, dim=0)} task {policy_index}')
        #     print(f'phis values {torch.norm(phis, dim=0)} task {policy_index}')
        #     print(f'Policy Index {policy_index}')
        #     print(f' Update STEP # {self.sf.updates_since_target_updated[policy_index]}')
        #
        #     accum_grads = 0
        #     accum_weights = 0
        #     for params in psi_model.parameters():
        #         accum_grads += torch.norm(params.grad)
        #         accum_weights += torch.norm(params.data)
        #     print(f'Gradients of Psi {accum_grads}')
        #     print(f'Psi weights {accum_weights}')
        #
        #     accum_grads = 0
        #     accum_weights = 0
        #     for params in g_function.parameters():
        #         accum_grads += torch.norm(params.grad)
        #         accum_weights += torch.norm(params.data)
        #     print(f'Gradients of G function {accum_grads}')
        #     print(f'G function weights {accum_weights}')
        #
        #     accum_grads = 0
        #     accum_weights = 0
        #     for params in self.h_function.parameters():
        #         accum_grads += torch.norm(params.grad)
        #         accum_weights += torch.norm(params.data)
        #     print(f'Gradients of H function {accum_grads}')
        #     print(f'H function weights {accum_weights}')
        #
        #     accum_grads = 0
        #     accum_weights = 0
        #     for params in task_w.parameters():
        #         accum_grads += torch.norm(params.grad)
        #         accum_weights += torch.norm(params.data)
        #         print(f'W weights {params.data}')
        #     print(f'Gradients of W {accum_grads}')
        #     print(f'W weights {accum_weights}')
        #
        #     accum_grads = 0
        #     accum_weights = 0
        #     for params in target_psi_model.parameters():
        #         if params.grad is not None:
        #             accum_grads += torch.norm(params.grad)
        #         accum_weights += torch.norm(params.data)
        #     print(f'Gradients of Psi Target {accum_grads}')
        #     print(f'Weights of Psi Target {accum_weights}')
        #     print(f'########### END #################')

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
        n_coupling_layers = self.hyperparameters.get('n_coupling_layers', 1)

        g_function = self._init_g_function(task.encode_dim(), g_h_function_dims, n_coupling_layers)
        self.g_functions.append(g_function)

        if self.h_function is None:
            self.h_function = self._init_h_function(g_h_function_dims, task.feature_dim())

        # SF model will keep the model optimizer
        self.sf.add_training_task(task, None, g_function, self.h_function)

    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000, cycles_per_task=1, learn_omegas=True, use_gpi_eval=False):
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
                            R, accum_loss, total_phi_loss, total_psi_loss = self.test_agent(test_task, test_index, learn_omegas=learn_omegas, use_gpi_eval=use_gpi_eval)
                            Rs.append(R.detach().cpu().numpy())

                            self.logger.log({f'eval/target_task_{test_index}/omegas': str(self.omegas[test_index].clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                            self.logger.log({f'eval/target_task_{test_index}/total_reward': R.detach().cpu().numpy().item(), 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/total_loss': accum_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/phi_loss': total_phi_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/psi_loss': total_psi_loss, 'timesteps': self.total_training_steps})

                        avg_R = np.mean(Rs)
                        return_data.append(avg_R)

                        # log average return
                        self.logger.log({f'eval/average_reward': avg_R, 'timesteps': self.total_training_steps})


                        # Every n_test_ev we can log the current progress in source task.
                        self.logger.log({f'train/source_task_{self.task_index}/total_reward': self.reward.clone().detach().cpu().numpy().item(), 'timesteps': self.total_training_steps})

                    self.total_training_steps += 1
        return return_data
    
    def get_test_action(self, s_enc, w, omegas, use_gpi_eval=False, learn_omegas=True):
        with torch.no_grad():
            if random.random() <= self.test_epsilon:
                a = torch.tensor(random.randrange(self.n_actions)).to(self.device)
            else:
                successor_features = self.sf.get_successors(s_enc)
                if use_gpi_eval:
                    q = w(successor_features)[:, :, :, 0]
                    if learn_omegas:
                        q = omegas[:,:,:,0] * q
                    max_task = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)
                    q = q[:, max_task, :]
                    a = torch.argmax(q)
                else:
                    normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))
                    tsf = torch.sum(successor_features * normalized_omegas, axis=1)
                    q = w(tsf)
                    # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                    # Target TSF only Use Q-Learning
                    a = torch.argmax(q)
            return a
            
    def test_agent(self, task, test_index, use_gpi_eval=True, learn_omegas=True):
        R = 0.0
        w, optim, scheduler = self.test_tasks_weights[test_index]
        omegas = self.omegas[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        total_phi_loss = 0
        total_psi_loss = 0

        for target_ev_step in range(self.T):
            a = self.get_test_action(s_enc, w, omegas, use_gpi_eval)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            if learn_omegas:
                a1 = self.get_test_action(s1_enc, w, omegas, use_gpi_eval)
                loss_t, phi_loss, psi_loss = self.update_test_reward_mapper_omegas(w, omegas, optim, task, test_index, r, s_enc, a, s1_enc, a1)
            else:
                loss_t, phi_loss, psi_loss = self.update_test_reward_mapper(w, optim, task, r, s_enc, a, s1_enc)
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
        #beta_loss_coefficient = self.hyperparameters['beta_loss_coefficient']

        # Fix it seems omegas are being cached
        self.omegas[test_index] = omegas

        return R, accum_loss, total_phi_loss, total_psi_loss

    def update_test_reward_mapper(self, w_approx, optim, task, r, s, a, s1):
        # Return Loss
        phi = task.features(s, a, s1)
        loss_task = torch.nn.MSELoss()

        with torch.no_grad():
            r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device)

        optim.zero_grad()

        r_fit = w_approx(phi)
        loss = loss_task(r_fit, r_tensor)
        loss.backward()

        optim.step()
        return loss, torch.tensor(0), torch.tensor(0)

    def update_test_reward_mapper_omegas(self, w_approx, omegas, optim, task, test_index, r, s, a, s1, a1):

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
            next_target_tsf = torch.sum(next_successor_features * normalized_omegas, axis=1)[:, a1 ,:]

        next_tsf = transformed_phi + self.gamma * next_target_tsf

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

        # Possible we can log the phi values, the transformed states.

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