import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.torch import update_models_weights, get_parameters_norm_mean, project_on_simplex, layer_init
from utils.logger import get_logger_level
import random


################################ Features ##################################3
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

        self._model = torch.nn.Sequential(*layers)
        self.optimiser = torch.optim.Adam(self._model.parameters(), lr=self.alpha_phi)
        self._model.train() # Train mode

    def forward(self, state, action, next_state, has_batch=False):

        if has_batch:
            inputs = torch.cat([state, action.unsqueeze(-1), next_state], dim=-1)
        else:
            inputs = torch.tensor(np.concatenate([state, [action], next_state])).float().to(self.device)

        return self._model(inputs)

    def to(self, device):
        self.device = device
        return super().to(device)

    def set_eval(self):
        self._model.eval()

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
    def build_planar_flow(cls, input_dim, output_dim, include_linear_layer=False, n_affine_flows = 1):
        flows = [cls(input_dim) for _ in range(n_affine_flows)]
        # Last linear layer
        if include_linear_layer:
            last_layer = torch.nn.Linear(input_dim, output_dim, bias=True)
            flows.append(last_layer)

        return torch.nn.Sequential(*flows)


class LinearBatchNorm(nn.Module):
    """
    An (invertible) batch normalization layer.
    This class is mostly inspired from this one:
    https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
    """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, **kwargs):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # log_det = self.log_gamma - 0.5 * torch.log(var + self.eps)

        return y

    def backward(self, x, **kwargs):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_det = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_det.expand_as(x).sum(1)

class MaskedAffineFlow(torch.nn.Module):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    """

    def __init__(self, b, t=None, s=None):
        """Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer("b", self.b_cpu)

        if s is None:
            self.s = torch.zeros_like
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like
        else:
            self.add_module("t", t)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        # log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_  # , log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        # log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_  # , log_det

    @classmethod
    def build_flow(cls, input_dim, output_dim, n_affine_flows = 1):
        n_affine_flows = 1 if n_affine_flows == 0 else n_affine_flows

        assert n_affine_flows >= 2, 'Masked Affine Transformation will keep only part of the variables set. Min 2 layers'

        flows = []

        for n_affine in range(n_affine_flows):
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(input_dim)])
            b = 1 - b if n_affine % 2 == 0 else b # alternate checkboard
            s_init = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2 * input_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * input_dim, output_dim),
            ])
            t_init = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2 * input_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * input_dim, output_dim),
            ])
            s_init.apply(lambda layer: layer_init(layer, method='uniform'))
            t_init.apply(lambda layer: layer_init(layer, method='uniform'))
            nn.init.zeros_(s_init[-1].weight)
            nn.init.zeros_(s_init[-1].bias)
            nn.init.zeros_(t_init[-1].weight)
            nn.init.zeros_(t_init[-1].bias)

            s = torch.nn.Sequential(*s_init)
            t = torch.nn.Sequential(*t_init)
            g_function = cls(b, t, s)
            batch_norm = LinearBatchNorm(input_dim)

            flows.append(g_function)
            flows.append(batch_norm)

        return torch.nn.Sequential(*flows)

################################## Agent ####################################################
class TSFDQN:

    def __init__(self, deep_sf, buffer_handle, gamma, T, encoding, epsilon=0.1, epsilon_decay=1., epsilon_min=0.,
                 print_ev=1000, save_ev=100, use_gpi=True, test_epsilon=0.03, device=None, invertible_flow='planar',
                 learn_transformed_function=False, learn_omegas_source_task=False, omegas_std_mode='average',
                 only_next_states_affine_state=False, **kwargs):
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
        self.invertible_flow = invertible_flow # By default is planar. (MaskedAffineFlow) realnvp. linear.
        self.learnt_phi = None
        self.learn_omegas_source_task = learn_omegas_source_task,
        self.omegas_std_mode = omegas_std_mode
        self.only_next_states_affine_state = only_next_states_affine_state
        self.learn_transformed_function = learn_transformed_function

    # ===========================================================================
    # TASK MANAGEMENT
    # ===========================================================================
    def reset(self):
        """
        Resets the agent, including all value functions and internal memory/history.
        """
        self.tasks = []
        self.phis = []

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
            a = random.randrange(self.n_actions)
        else:
            a = torch.argmax(q).item()
        
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
            self.logger.log({f'train/source_task_{self.task_index}/episode_reward': self.episode_reward, 'episodes': self.episode})

        with torch.no_grad():
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
        self.active_source_task_omegas = self.omegas_per_source_task[index] if self.learn_omegas_source_task else None

    def get_Q_values(self, s, s_enc):
        with torch.no_grad():
            s_enc_torch = torch.tensor(s_enc).float().to(self.device)
            q, c = self.sf.GPI(s_enc_torch, self.task_index, update_counters=self.use_gpi)
            if not self.use_gpi:
                c = self.task_index
            self.c = c
            return q[:, c,:]

    def _init_g_function(self, states_dim, output_dim, n_coupling_layers = 0):

        if self.invertible_flow == 'linear':
            g_function = torch.nn.Linear(states_dim, output_dim, bias=True, device=self.device)
        elif self.invertible_flow == 'realnvp':
            g_function = MaskedAffineFlow.build_flow(states_dim, output_dim, n_affine_flows=n_coupling_layers)
        else:
            # default is planar
            # Number of planarflows = 10
            if n_coupling_layers > 0:
                g_function = Planar.build_planar_flow(states_dim, output_dim, n_affine_flows=n_coupling_layers)
            else:
                g_function = Planar(states_dim)

        return g_function.to(self.device)

    def _init_h_function(self, input_dim, features_dim):
        # TODO Remove weights clamp and initial weights
        # This affine transformation
        h_function = torch.nn.Linear(input_dim, features_dim, bias=True, device=self.device)
        return h_function

    def _init_omega(self, num_source_tasks, method='uniform'):
        if method == 'uniform':
            omega = torch.Tensor(1, num_source_tasks, 1, 1).uniform_(0,1).to(self.device).requires_grad_(True)
        if method == 'constant':
            omega = torch.ones((1, num_source_tasks, 1, 1)).float().to(self.device).requires_grad_(True)
        return omega
    
    def train_agent(self, s, s_enc, a, r, s1, s1_enc, gamma):
        
        # remember this experience
        # This phi should be the known/shared given feature function.
        phi = self.phi(s, a, s1)

        if isinstance(phi, torch.Tensor):
            phi = phi.detach().numpy()

        self.buffer.append(s_enc, a, r, phi, s1_enc, gamma)
        
        # update SFs
        if self.total_training_steps % 1 == 0:
            transitions = self.buffer.replay()
            self.update_successor(transitions, self.task_index, self.use_gpi)

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
        current_qs = task_w(current_psi)  # TODO to use Q in TD.

        transformed_state = g_function(states)
        transformed_next_state = g_function(next_states)
        # affine_transformed_states = self.h_function(transformed_state) + self.h_function(transformed_next_state)
        # affine_transformed_states = self.h_function(transformed_state + transformed_next_state)

        if self.only_next_states_affine_state:
            affine_transformed_states = self.h_function(transformed_next_state)
        else:
            affine_transformed_states = self.h_function(transformed_state + transformed_next_state)

        if self.learn_transformed_function:
            transformed_phis = affine_transformed_states * self.transformed_phi_function(states, actions, states, has_batch=True)
        else:
            transformed_phis = affine_transformed_states * phis

        with torch.no_grad():
            next_psis = target_psi_model(next_states)[indices, next_actions,:]
            next_qs = task_w(next_psis) # TODO to use Q in TD.

        targets = transformed_phis + gammas * next_psis
        targets_q = rs + gammas * next_qs # TODO to use Q in TD.

        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets

        merge_current_target_q = current_qs.clone() # TODO to use Q in TD.
        merge_current_target_q[indices, actions, :] = targets_q

        optim.zero_grad()

        # TODO change the feature function to fit w.
        r_fit = task_w(phis)
        # r_fit_transformed = task_w(transformed_phis)

        psi_loss_coefficient = torch.tensor(self.hyperparameters['source_psi_fit_loss_coefficient'])
        r_fit_loss_coefficient = torch.tensor(self.hyperparameters['source_r_fit_loss_coefficient'])

        l1 = psi_loss(current_psi, merge_current_target_psi)
        l2 = psi_loss(r_fit, rs)
        l3 = psi_loss(current_qs, merge_current_target_q)
        # l4 = psi_loss(r_fit_transformed, rs)

        loss = (psi_loss_coefficient * l1) + (r_fit_loss_coefficient * l2) + (psi_loss_coefficient * l3) # + (r_fit_loss_coefficient * l4)

        if self.learn_transformed_function:
            l5 = psi_loss(phis, transformed_phis)
            loss += psi_loss_coefficient * l5

        loss.backward()

        # log gradients this is only a way to track gradients from time to time
        if self.total_training_steps % 1_000 == 0:
            self.logger.log({f'metrics/source_task_{policy_index}/affine_t_states': str(torch.norm(affine_transformed_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/affine_t_states_mean': str(torch.mean(affine_transformed_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_t_states': str(torch.norm(transformed_state, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_t_next_states': str(torch.norm(transformed_next_state, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_t_states_mean': str(torch.mean(transformed_state, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_t_next_states_mean': str(torch.mean(transformed_next_state, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/phis_values': str(torch.norm(phis, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/phis_values_mean': str(torch.mean(phis, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})

            # Log gradients
            accum_grads, accum_weights, params_mean = get_parameters_norm_mean(psi_model)
            self.logger.log({f'metrics/source_task_{policy_index}/psi_model_gradients_norm':  accum_grads, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/psi_model_weights_norm': accum_weights, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/psi_model_weights_mean': params_mean, 'timesteps': self.total_training_steps})

            accum_grads, accum_weights, params_mean = get_parameters_norm_mean(g_function)
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_gradients_norm': accum_grads, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_weights_norm':  accum_weights, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/g_function_weights_mean': params_mean, 'timesteps': self.total_training_steps})

            accum_grads, accum_weights, params_mean = get_parameters_norm_mean(self.h_function)
            self.logger.log({f'metrics/source_task_{policy_index}/h_function_gradients_norm': accum_grads, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/h_function_weights_norm':  accum_weights, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/h_function_weights_mean': params_mean, 'timesteps': self.total_training_steps})

            accum_grads, accum_weights, params_mean = get_parameters_norm_mean(task_w)
            self.logger.log({f'metrics/source_task_{policy_index}/weights_gradients_norm':  accum_grads, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/weights_norm':  accum_weights, 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/source_task_{policy_index}/weights_mean': params_mean, 'timesteps': self.total_training_steps})

            self.logger.log({f'losses/source_task_{policy_index}/total_loss': loss.item(),
                             'timesteps': self.total_training_steps})
            self.logger.log({f'losses/source_task_{policy_index}/phi_loss': l2.item(),
                             'timesteps': self.total_training_steps})
            self.logger.log({f'losses/source_task_{policy_index}/psi_loss': l1.item(),
                             'timesteps': self.total_training_steps})
            self.logger.log({f'losses/source_task_{policy_index}/q_loss': l3.item(),
                             'timesteps': self.total_training_steps})

            if self.learn_transformed_function:
                self.logger.log({f'losses/source_task_{policy_index}/transformed_phi_loss': l5.item(),
                                 'timesteps': self.total_training_steps})
                accum_grads, accum_weights, params_mean = get_parameters_norm_mean(self.transformed_phi_function)
                self.logger.log({f'metrics/source_task_{policy_index}/transformed_phi_gradients_norm': accum_grads,
                                 'timesteps': self.total_training_steps})
                self.logger.log({f'metrics/source_task_{policy_index}/transformed_phi_norm': accum_weights,
                                 'timesteps': self.total_training_steps})
                self.logger.log({f'metrics/source_task_{policy_index}/transformed_phi_mean': params_mean,
                                 'timesteps': self.total_training_steps})

            task_w = self.sf.fit_w[policy_index]
            self.logger.log({f'train/source_task_{policy_index}/weights': str(task_w.weight.clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})

        optim.step()

        # Finish train the SF network
        # update the target SF network
        self.sf.updates_since_target_updated[policy_index] += 1
        if self.sf.updates_since_target_updated[policy_index] >= self.sf.target_update_ev:
            update_models_weights(psi_model, target_psi_model)
            self.sf.updates_since_target_updated[policy_index] = 0

    def add_training_task(self, task):
        """
        Adds a training task to be trained by the agent.
        """
        self.tasks.append(task)   
        self.n_tasks = len(self.tasks)

        if self.learnt_phi is not None:
            self.phis.append(self.learnt_phi)
        else:
            self.phis.append(task.features)

        if self.n_tasks == 1:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            if self.encoding == 'task':
                self.encoding = task.encode

            if self.learn_transformed_function and self.learnt_phi is None:
                # This transformed phi function is the feature function we will learn.
                self.transformed_phi_function = PhiFunction(task.encode_dim(), 1, task.feature_dim()).to(self.device)
    

        # Sequential Learning Append Buffer
        self.buffers.append(self.buffer_handle())

        # Transformed Successor Feature
        # Encode Dim encapsulates the state encoding dimension
        g_h_function_dims = self.hyperparameters.get('g_h_function_dims')
        n_coupling_layers = self.hyperparameters.get('n_coupling_layers', 0)

        g_function = self._init_g_function(task.encode_dim(), g_h_function_dims, n_coupling_layers)
        self.g_functions.append(g_function)

        if self.h_function is None:
            self.h_function = self._init_h_function(g_h_function_dims, task.feature_dim())

        # SF model will keep the model optimizer
        self.sf.add_training_task(task, None, g_function, self.h_function)

    def train(self, train_tasks, n_samples, viewers=None, n_view_ev=None, test_tasks=[], n_test_ev=1000, cycles_per_task=1, learn_omegas=True, use_gpi_eval_mode='vanilla', omegas_init_method='uniform', learn_weights=True):

        if viewers is None:
            viewers = [None] * len(train_tasks)
            
        # add tasks
        self.reset()

        # train each one
        # Regularize sum w_i = 1
        # Unsqueeze to have [n_batch, n_tasks, n_actions, n_features]
        # Initialize Omegas
        omegas_temp = self._init_omega(len(train_tasks), method=omegas_init_method)
        with torch.no_grad():
            omegas_temp = (omegas_temp / torch.sum(omegas_temp, axis=1, keepdim=True))
        omegas_temp = omegas_temp.requires_grad_(True)

        for train_task in train_tasks:
            self.add_training_task(train_task)
            if self.learn_omegas_source_task:
                omegas_source_task = omegas_temp.clone().detach().requires_grad_(True)
                self.omegas_per_source_task.append(omegas_source_task)

        # Initialize Target Reward Mappers and optimizer
        for test_task in test_tasks:
            omegas_target_task = omegas_temp.clone().detach().requires_grad_(True)

            fit_w = torch.Tensor(1, test_task.feature_dim()).uniform_(-0.01, 0.01).to(self.device)

            if not learn_weights: # to not learn fit_w
                fit_w = torch.tensor(test_task.get_w()).float().to(self.device).reshape(1, test_task.feature_dim()) # for real w.

            w_approx = torch.nn.Linear(test_task.feature_dim(), 1, bias=False, device=self.device)
            # w_approx = torch.nn.Linear(test_task.feature_dim(), 1, device=self.device)

            with torch.no_grad():
                w_approx.weight = torch.nn.Parameter(fit_w)

            #fit_w = torch.Tensor(test_task.feature_dim(), 1).uniform_(-0.01, 0.01).to(self.device)
            #w_approx = fit_w
            # Learning rate alpha (Weights)
            parameters = [
                {'params': w_approx.parameters(),
                 'lr': self.hyperparameters['learning_rate_w_target_task'] if learn_weights else 0.0,
                 'weight_decay': self.hyperparameters['weight_decay_w']},
                {'params': omegas_target_task,
                 'lr': self.hyperparameters['learning_rate_omega'] if learn_omegas else 0.0,
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
                            R, accum_loss, total_psi_loss, total_phi_loss, total_q_value_loss, total_transformed_phi_loss = self.test_agent(test_task, test_index, learn_omegas=learn_omegas, use_gpi_eval_mode=use_gpi_eval_mode)
                            Rs.append(R)

                            self.logger.log({f'eval/target_task_{test_index}/omegas': str(self.omegas[test_index].clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                            self.logger.log({f'eval/target_task_{test_index}/total_reward': R, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/total_loss': accum_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/phi_loss': total_phi_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/psi_loss': total_psi_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/q_value_loss': total_q_value_loss, 'timesteps': self.total_training_steps})
                            self.logger.log({f'losses/target_task_{test_index}/total_transformed_phi_loss': total_transformed_phi_loss, 'timesteps': self.total_training_steps})

                        avg_R = np.mean(Rs)
                        return_data.append(avg_R)

                        # log average return
                        self.logger.log({f'eval/average_reward': avg_R, 'timesteps': self.total_training_steps})


                        # Every n_test_ev we can log the current progress in source task.
                        self.logger.log({f'train/source_task_{self.task_index}/total_reward': self.reward, 'timesteps': self.total_training_steps})

                    self.total_training_steps += 1
        return return_data
    
    def get_test_action(self, s_enc: torch.Tensor, w, omegas, use_gpi_eval_mode='vanilla', learn_omegas=True, test_index=None):
        """
        use_gpi_eval_mode ['vanilla', 'naive', 'argmax_convex', 'affine_similarity',]
        """
        with torch.no_grad():
            if random.random() <= self.test_epsilon:
                a = random.randrange(self.n_actions)
            else:
                if use_gpi_eval_mode != 'argmax_convex':
                    successor_features = self.sf.get_successors(s_enc)

                if use_gpi_eval_mode=='naive':
                    q = w(successor_features)[:, :, :, 0]
                    if learn_omegas:
                        q = omegas[:,:,:,0] * q
                    max_task = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)
                    q = q[:, max_task, :]
                    a = torch.argmax(q)
                elif use_gpi_eval_mode=='argmax_convex':
                    normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))
                    t_states = []
                    for g in self.g_functions:
                        state = g(s_enc)
                        t_states.append(state)
                    # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
                    t_states = torch.vstack(t_states).unsqueeze(1)
                    t_states = torch.sum(t_states * normalized_omegas, axis=1).squeeze(1)
                    t_states = self.h_function(t_states)
                    successor_features = self.sf.get_successors(t_states)

                    q = w(successor_features)[:, :, :, 0]

                    max_task = torch.squeeze(torch.argmax(torch.max(q, axis=2).values, axis=1))  # shape (n_batch,)
                    q = q[:, max_task, :]
                    a = torch.argmax(q)
                elif use_gpi_eval_mode=='affine_similarity':
                    t_states = []
                    for g in self.g_functions:
                        state = g(s_enc)
                        t_states.append(state)
                    # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
                    t_states = torch.vstack(t_states).unsqueeze(1)

                    if learn_omegas:
                        t_states = t_states * omegas.squeeze(0)

                    affine_states = self.h_function(t_states)  # n_tasks, n_actions, n_features

                    ones_torch = torch.ones(affine_states.shape).to(self.device)
                    norm_affine = torch.norm(ones_torch - affine_states, dim=-1)
                    similar_affine_task = torch.argmin(norm_affine, dim=0)
                    q = w(successor_features)[:, similar_affine_task, :, :]
                    a = torch.argmax(q).item()
                elif use_gpi_eval_mode=='affine_similarity_minmax':
                    t_states = []
                    for g in self.g_functions:
                        state = g(s_enc)
                        t_states.append(state)
                    # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
                    t_states = torch.vstack(t_states).unsqueeze(1)

                    if learn_omegas:
                        t_states = t_states * omegas.squeeze(0)

                    affine_states = self.h_function(t_states)  # n_tasks, n_actions, n_features

                    ones_torch = torch.ones(affine_states.shape).to(self.device)
                    norm_affine = torch.norm(ones_torch - affine_states, dim=-1)
                    q = w(successor_features)
                    similar_affine_tasks_order_index = torch.sort(norm_affine, dim=0)[1].reshape(-1)
                    max_q_values_index = torch.sort(torch.max(q, dim=2).values, dim=1, descending=True)[1].reshape(-1)

                    min_max_task = np.zeros(similar_affine_tasks_order_index.shape)
                    # This min_max algorithm. Sorts the similar tasks by affine value: [2, 3, 1, 0]
                    # Max_q values are the max values [3,0,1,2] are sorted by value.
                    # The algorithm try to match the minimum of the affine values with the max values.
                    for min_affine_idx in range(similar_affine_tasks_order_index.shape[0]):
                        max_q_values_idx = torch.where(max_q_values_index == similar_affine_tasks_order_index[min_affine_idx])[0].item()
                        min_max_task[min_affine_idx] = (min_affine_idx + 1) * (max_q_values_idx + 1)

                    min_max_task = similar_affine_tasks_order_index[np.argmin(min_max_task)]
                    q = q[:, min_max_task, :, :]
                    a = torch.argmax(q).item()

                    if self.total_training_steps % 100_000 == 0: # In one million only 500 Test steps * target tasks
                        self.logger.log({f'eval/target_task_{test_index}/similar_affine_order': str(similar_affine_tasks_order_index.detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                        self.logger.log({f'eval/target_task_{test_index}/max_q_values_index': str(max_q_values_index.detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                        self.logger.log({f'eval/target_task_{test_index}/min_max_selected_task': str(min_max_task.detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                elif use_gpi_eval_mode == 'affine_similarity_minmax_expectation':
                    # It is the same as minmax_affine.
                    normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))

                    t_states = []
                    for g in self.g_functions:
                        state = g(s_enc)
                        t_states.append(state)
                    # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
                    t_states = torch.vstack(t_states).unsqueeze(1)
                    t_states = t_states * normalized_omegas

                    affine_states = self.h_function(t_states)  # n_tasks, n_actions, n_features
                    ones_torch = torch.ones(affine_states.shape).to(self.device)

                    expected_transformation = affine_states
                    next_successor_features = successor_features * (ones_torch - expected_transformation.abs())
                    q_values = w(next_successor_features)
                    max_task = torch.squeeze(torch.argmax(torch.max(q_values, dim=2).values, dim=1))
                    q_values = q_values[:, max_task, :, :]
                    a = torch.argmax(q_values).item()

                    if self.total_training_steps % 100_000 == 0:  # In one million only 500 Test steps * target tasks
                        self.logger.log({f'eval/target_task_{test_index}/minmax_expected_task': max_task.item(), 'timesteps': self.total_training_steps})
                else:
                    # Vanilla

                    if self.omegas_std_mode == 'average':
                        normalized_omegas = (omegas / torch.sum(omegas, axis=1, keepdim=True))
                        tsf = torch.sum(successor_features * normalized_omegas, axis=1)
                    if self.omegas_std_mode in ['project_simplex', 'no_constraint']:
                        tsf = torch.sum(successor_features * omegas, axis=1)

                    q = w(tsf)
                    # q = (psi @ w)[:,:,:, 0]  # shape (n_batch, n_tasks, n_actions)
                    # Target TSF only Use Q-Learning
                    a = torch.argmax(q).item()
            return a
            
    def test_agent(self, task, test_index, use_gpi_eval_mode='vanilla', learn_omegas=True):
        R = 0.0
        w, optim, scheduler = self.test_tasks_weights[test_index]
        omegas = self.omegas[test_index]
        s = task.initialize()
        s_enc = self.encoding(s)

        accum_loss = 0
        total_phi_loss = 0
        total_psi_loss = 0
        total_q_value_loss = 0
        total_transformed_phi_loss = 0

        for target_ev_step in range(self.T):
            s_enc_torch = torch.tensor(s_enc).float().to(self.device).detach()
            a = self.get_test_action(s_enc_torch, w, omegas, use_gpi_eval_mode=use_gpi_eval_mode, learn_omegas=learn_omegas, test_index=test_index)
            s1, r, done = task.transition(a)
            s1_enc = self.encoding(s1)

            s1_enc_torch = torch.tensor(s1_enc).float().to(self.device).detach()
            a1 = self.get_test_action(s1_enc_torch, w, omegas, use_gpi_eval_mode=use_gpi_eval_mode, learn_omegas=learn_omegas, test_index=test_index)
            loss_t, psi_loss, phi_loss, q_value_loss, transformed_phi_loss = self.update_test_reward_mapper_omegas(w, omegas, optim, task, test_index, r, s_enc, a, s1_enc, a1, done, eval_step=target_ev_step, scheduler=scheduler)

            accum_loss += loss_t.item()
            total_phi_loss += phi_loss.item()
            total_psi_loss += psi_loss.item()
            total_q_value_loss += q_value_loss.item()
            total_transformed_phi_loss += transformed_phi_loss.item()

            # Update states
            s, s_enc = s1, s1_enc
            R += r

            if done:
                break

        # Log accum loss for T from in a random way
        #beta_loss_coefficient = self.hyperparameters['beta_loss_coefficient']

        # Fix it seems omegas are being cached
        # self.omegas[test_index] = omegas

        return R, accum_loss, total_psi_loss, total_phi_loss, total_q_value_loss, total_transformed_phi_loss

    def update_test_reward_mapper_omegas(self, w_approx, omegas, optim, task, test_index, r, s, a, s1, a1, done, eval_step=0, scheduler=None):
        # GPI modes
        # GPI naive: only argmax_{a} max_{\pi}
        # GPI affine_similarity: argmax_{a} min_{||1-h||} This h could be h(\omega_{i} g^{-i}) or simply h(g^{-i})
        # Vanilla TSF: \sum_{i} \omega_{i}

        if self.h_function is None:
            raise Exception('Affine Function (h) is not initialized')

        # Return Loss
        if self.learnt_phi is not None:
            phi_tensor = self.phi(s.reshape(-1),a,s1.reshape(-1)) # this must be a share
        else:
            phi = task.features(s,a,s1)
            phi_tensor = torch.tensor(phi).float().to(self.device).detach()

            if self.learn_transformed_function:
                transformed_phi_tensor = self.transformed_phi_function(s.reshape(-1),a,s1.reshape(-1))

        s_torch = torch.tensor(s).float().to(self.device).detach()
        s1_torch = torch.tensor(s1).float().to(self.device).detach()

        # h function eval
        self.h_function.eval()
        # g function eval
        [g.eval() for g in self.g_functions]

        # Transformed States
        t_states = []
        t_next_states = []

        with torch.no_grad():
            for g in self.g_functions:
                state = g(s_torch)
                next_state = g(s1_torch)
                t_states.append(state)
                t_next_states.append(next_state)

            # Unsqueeze to be the same shape as omegas [n_batch, n_tasks, n_actions, n_features]
            t_states = torch.vstack(t_states).unsqueeze(1).unsqueeze(0).detach()
            t_next_states = torch.vstack(t_next_states).unsqueeze(1).unsqueeze(0).detach()

        # Code to learn omegas
        weighted_states = torch.sum(t_states * omegas, axis=1)
        weighted_next_states = torch.sum(t_next_states * omegas, axis=1)

        # affine_states = self.h_function(weighted_states) + self.h_function(weighted_next_states)

        # Remember h(\beta g(x) + \alpha  g(y)) = \beta h(g(x)) + \alpha h(g(y)) if \beta + \alpha = 1.
        # Otherwise is the definition of linearity.
        if self.only_next_states_affine_state:
            affine_states = self.h_function(weighted_next_states)
        else:
            affine_states = self.h_function(weighted_states + weighted_next_states)

        ####################### Process latent probability transformation

        if self.learn_transformed_function:
            transformed_phi = transformed_phi_tensor * affine_states.squeeze(0)
        else:
            transformed_phi = phi_tensor * affine_states.squeeze(0)

        successor_features = self.sf.get_successors(s_torch).detach() # n_batch, n_tasks, n_actions, n_features
        next_successor_features = self.sf.get_next_successors(s1_torch).detach()
        r_tensor = torch.tensor(r).float().unsqueeze(0).to(self.device).detach()

        # TODO Remove this. This is to double check that omegas are not being learnt as that suppose to do.
        # next_target_tsf = torch.sum(next_successor_features * omegas, axis=1)[:, a1, :]
        with torch.no_grad():
            next_target_tsf = torch.sum(next_successor_features * omegas, axis=1) # TODO Update the entire q table.
            next_q_value = r_tensor + (1 - float(done)) * self.gamma * w_approx(next_target_tsf).reshape(-1)

        # TODO Remove this. Weights are not being learnt properly.
        # TODO change the feature function to fit w.
        # r_fit_transformed = w_approx(transformed_phi).reshape(-1)
        r_fit = w_approx(phi_tensor).reshape(-1)

        with torch.no_grad():
            next_tsf = transformed_phi + (1 - float(done)) * self.gamma * next_target_tsf
        # tsf = torch.sum(successor_features * omegas, axis=1)[:, a ,:]
        tsf = torch.sum(successor_features * omegas, axis=1) # TODO Update the entire q table
        q_value = w_approx(tsf).reshape(-1)

        loss_task = lambda input, target: torch.sum((input - target) ** 2).mean()

        # Hyperparameters and L1 Regularizations

        psi_loss_coefficient = torch.tensor(self.hyperparameters['target_psi_fit_loss_coefficient'])
        r_loss_coefficient = torch.tensor(self.hyperparameters['target_r_fit_loss_coefficient'])
        q_value_loss_coefficient = torch.tensor(self.hyperparameters['target_q_value_loss_coefficient'])
        lasso_coefficient = torch.tensor(self.hyperparameters['omegas_l1_coefficient'])
        ridge_coefficient = torch.tensor(self.hyperparameters['omegas_l2_coefficient'])
        maxent_coefficient = torch.tensor(self.hyperparameters['omegas_maxent_coefficient'])
        # L1 and L2 Norm
        lasso_regularization = torch.norm(omegas, 1) if lasso_coefficient > 0.0 else torch.tensor(0)
        ridge_regularization = torch.norm(omegas, 2) if ridge_coefficient > 0.0 else torch.tensor(0)
        entropy_regularization = (omegas * torch.log(omegas)).sum() if maxent_coefficient > 0.0 else torch.tensor(0)

        l1 = loss_task(tsf, next_tsf)
        l2 = loss_task(r_fit, r_tensor)
        l3 = loss_task(q_value, next_q_value)
        # l4 = loss_task(r_fit_transformed, r_tensor)
        l5 = torch.tensor(0.0)

        loss = (
                (psi_loss_coefficient * l1)
                + (r_loss_coefficient * l2)
                + (q_value_loss_coefficient * l3)
                # + (r_loss_coefficient * l4)
                # + (r_loss_coefficient * l5)
                + (lasso_coefficient * lasso_regularization)
                + (ridge_coefficient * ridge_regularization)
                + (maxent_coefficient * entropy_regularization))

        if self.learn_transformed_function:
            l5 = loss_task(transformed_phi, phi_tensor)
            loss += l5

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        # optim.zero_grad() # this can be used to make sure no other part we have computed gradients.

        # Sum_i omega_i = 1
        with torch.no_grad():
            epsilon = 1e-7

            if self.omegas_std_mode == 'average':
                omegas.clamp_(epsilon)
                omegas.data = (omegas / torch.sum(omegas, axis=1, keepdim=True)).data
            if self.omegas_std_mode == 'project_simplex':
                omegas.data = project_on_simplex(omegas, epsilon=epsilon, device=self.device).data

        # TODO We can add a log flag to avoid this after debugging.
        if eval_step == 0 or eval_step == self.T - 1: # First and Last eval step
            # Possible we can log the phi values, the transformed states.
            self.logger.log({f'metrics/target_task_{test_index}/affine_t_states': str(torch.norm(affine_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/affine_t_states_mean': str(torch.mean(affine_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/g_function_t_states': str(torch.norm(weighted_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/g_function_t_next_states': str(torch.norm(weighted_next_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/g_function_t_states_mean': str(torch.mean(weighted_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/g_function_t_next_states_mean': str(torch.mean(weighted_next_states, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/phis_values': str(torch.norm(transformed_phi, dim=0).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/phis_values_mean': torch.mean(transformed_phi).item(), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/omegas_lr': scheduler.get_lr()[1], 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/omegas_mean': torch.mean(omegas).item(), 'timesteps': self.total_training_steps})
            self.logger.log({f'metrics/target_task_{test_index}/weights': str(w_approx.weight.data.clone().reshape(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})

            for g_state_index in range(t_states.shape[0]):
                self.logger.log({f'metrics/target_task_{test_index}/g_function_source_task_{g_state_index}_states': str(t_states[g_state_index].view(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})
                self.logger.log({f'metrics/target_task_{test_index}/g_function_source_task_{g_state_index}_next_states': str(t_next_states[g_state_index].view(-1).detach().cpu().numpy()), 'timesteps': self.total_training_steps})


        # h function train
        self.h_function.train()
        # g function eval
        [g.train() for g in self.g_functions]
        # Loss, phi_loss, psi_loss
        return loss, l1, l2, l3, l5 # TODO Remember to restore l3

    def pre_train(self, train_tasks, n_samples_pre_train, n_cycles=5, feature_dim=None, lr=1e-3):
        from utils.buffer import ReplayBuffer

        if feature_dim is None:
            raise Exception('Feature dim should be sent.')

        # The result of this pre_train is having the first n_samples to fit reconstruct the feature function
        # At the end the idea is to override a phi function: self.learnt_phi
        first_task = train_tasks[0]
        buffer = ReplayBuffer()
        n_actions = first_task.action_count()

        # action dim tbd and feature dim.
        # Here I put all the model
        phi_learn_model = PhiFunction(first_task.encode_dim(), 1, first_task.feature_dim()).to(self.device)

        losses = []

        fit_ws = []
        fit_ws_optim = []

        # Initialize weights
        # Having a random policy using
        for task in train_tasks:
            fit_w = torch.nn.Linear(task.feature_dim(), 1, bias=False, device=self.device)
            with torch.no_grad():
                fit_w.weight = torch.nn.Parameter(torch.Tensor(1, task.feature_dim()).uniform_(-0.01, 0.01).to(self.device))

            fit_w_optim = torch.optim.Adam(fit_w.parameters(), lr=lr)

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
                    buffer.append(s_enc, np.array([a]), r, np.array([0]), s1_enc, np.array([0]))

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

                        if sample % 100 == 0:
                            self.logger.log({f'pretrain/source_task_{task_id}/loss': loss.item(), 'episodes': (cycle * n_samples_pre_train) + sample})

                        losses.append(loss.item())

        buffer.reset()
        self.learnt_phi = phi_learn_model
        self.learnt_phi.set_eval()

        return losses