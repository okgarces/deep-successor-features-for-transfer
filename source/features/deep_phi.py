# -*- coding: UTF-8 -*-
from features.successor import SF
from utils.torch import get_torch_device, update_models_weights

import torch
import numpy as np


torch.autograd.set_detect_anomaly(True)

class DeepSF_PHI(SF):
    """
    A successor feature representation implemented using Keras. Accepts a wide variety of neural networks as
    function approximators.
    """
    
    def __init__(self, pytorch_model_handle, *args, target_update_ev=1000, **kwargs):
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
        super(DeepSF_PHI, self).__init__(*args, **kwargs)
        self.pytorch_model_handle = pytorch_model_handle
        self.target_update_ev = target_update_ev
        
        self.device = get_torch_device()
    
    def reset(self):
        SF.reset(self)
        self.updates_since_target_updated = []
        
    def build_successor(self, task, source=None):
        
        # input tensor for all networks is shared
        # TODO the environment should send the action_count, feature_dim and inputs?
        if self.n_tasks == 0:
            self.n_actions = task.action_count()
            self.n_features = task.feature_dim()
            self.inputs = task.encode_dim()
            
        model, loss, optim = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)

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
        target_model, target_loss, target_optim = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)
        # target_model.set_weights(model.parameters())
        update_models_weights(model, target_model)
        self.updates_since_target_updated.append(0)
        
        return (model, loss, optim), (target_model, target_loss, target_optim)
        
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
    
    def update_reward(self, phi: torch.Tensor, r: torch.Tensor, task_index: int, exact=False) -> None:
        """
        Updates the reward parameters for the given task based on the observed reward sample
        from the environment. 
        
        Parameters
        ----------
        phi : torch.Tensor 
            the state features
        r : torch.TensorFloat | torch.Tensor
            the observed reward from the MDP
        task_index : integer
            the index of the task from which this reward was sampled
        exact : boolean
            if True, validates the true reward from the environment and the linear representation
        """
        
        # update reward using linear regression
        w = self.fit_w[task_index]
        loss = torch.nn.MSELoss()
        optim = torch.optim.SGD(w.parameters(), lr=0.005, weight_decay=0.01)
        r_tensor = torch.tensor(r).float().unsqueeze(0).detach().requires_grad_(False).to(self.device)

        optim.zero_grad()
        l1 = loss(w(phi), r_tensor)
        
        # Otherwise gradients will be computed to inf or nan.
        if True or not (torch.isnan(l1) or torch.isinf(l1)):
            if (torch.isnan(l1) or torch.isinf(l1)):
                print(f'loss target task {loss}')
                print(f'task_w weights target {w.weight}')
                print(f'phis in targte reward mapper {phi}')
            l1.backward()

            # Clamp weights between -1 and 1
            #for params in w.parameters():
            #    params.grad.data.clamp_(-1, 1)

            optim.step()

    def update_successor(self, transitions, phis_model, policy_index, loss_coefficient, use_gpi):

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
        task_w = self.fit_w[policy_index]

        # Concat axis = 1 to concat per each batch
        input_phi = torch.concat([states.to(self.device), actions.reshape((n_batch, 1)).to(self.device), next_states.to(self.device)], axis=1).detach()
        phis = phi_model(input_phi)

        # next actions come from current Successor Feature
        if use_gpi:
            q1, _ = self.GPI(next_states, policy_index)
            next_actions = torch.argmax(torch.max(q1, axis=1).values, axis=-1)
        else:
            sf = self.get_successor(next_states, policy_index)
            q1 = task_w(sf)
            # Do not forget: argmax according to actions, and squeeze in axis according to get n_batch
            next_actions = torch.squeeze(torch.argmax(q1, axis=1), axis=1)

        #q1, c = self.GPI(next_states, policy_index)
        #print(f'Q1 here {q1.shape}')
        #print(f'C here {c}')
        #next_actions = torch.argmax(torch.max(q1, axis=1).values, axis=-1)
        
        # compute the targets and TD errors
        psi_tuple, target_psi_tuple = self.psi[policy_index]
        psi_model, psi_loss, _ = psi_tuple
        target_psi_model, *_ = target_psi_tuple

        current_psi = psi_model(states)
        # We don't need target_psi gradients
        targets = phis + gammas * target_psi_model(next_states)[indices, next_actions,:].detach()

        # for param in task_w.parameters():
        #     param.requires_grad = False
        
        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets

        #current_psi_clone = current_psi
        #merge_current_target_psi_clone = merge_current_target_psi
        r_fit = task_w(phis)

        # TODO How many times does phi vector should be updated?
        # Only one phi vector with a weight_decay to learn smooth functions
        # psi_optim.zero_grad()
        # TODO This change is to optimize w with only one Loss. Different to the original.
        # The original first fit the w and then use GPI. This updates simultaneously
        # params = [
        #         {'params': phi_model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-4},
        #         {'params': psi_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
        #         {'params': task_w.parameters(), 'lr': 5e-3, 'weight_decay': 1e-3}
        # ]
        params = [
                #{'params': psi_model.parameters(), 'lr': 1e-3 },
                #{'params': phi_model.parameters(), 'lr': 1e-3},
                #{'params': task_w.parameters(), 'lr': 1e-3},
                #{'params': loss_coefficient.parameters(), 'lr': 1e-2}
                {'params': psi_model.parameters(), 'lr': 1e-3 , 'weight_decay': 1e-4 },
                {'params': phi_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                {'params': task_w.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4 },
                #{'params': loss_coefficient, 'lr': 1e-3, 'weight_decay': 1e-3 }
                {'params': loss_coefficient, 'lr': 1e-3 },
        ]

        phi_loss_value = phi_loss(r_fit, rs).unsqueeze(0)
        optim = torch.optim.Adam(params)
        optim.zero_grad()

        psi_loss_value = psi_loss(current_psi, merge_current_target_psi).unsqueeze(0)

        #input_loss = torch.concat([phi_loss_value, psi_loss_value]).flatten()
        #loss = loss_coefficient(input_loss)
        loss = (phi_loss_value) + (loss_coefficient * psi_loss_value)
        # loss = phi_loss_value + (loss_coefficient(psi_loss_value))

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

        loss.backward(retain_graph=True)

        # Clamp weights between
        for param_dict in params:
            for params in param_dict.get('params', []):
                params.grad.data.clamp_(-1e10, 1e10)
        
        optim.step()

        with torch.no_grad():
            loss_coefficient.data.clamp_(1e-2, 1e3)

        # Finish train the SF network
        # update the target SF network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            update_models_weights(psi_model, target_psi_model)
            # We don't need target phi model
            # update_models_weights(phi_model, target_phi_model)
            self.updates_since_target_updated[policy_index] = 0

        return loss, psi_loss_value, phi_loss_value, loss_coefficient

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
        
        # add successor features to the library
        self.psi.append(self.build_successor(task, source))
        self.n_tasks = len(self.psi)

        true_w = task.get_w()
        # build new reward function
        # TODO Remove task feature_dim should be called from the agent or config
        n_features = task.feature_dim()
        fit_w = torch.nn.Linear(n_features, 1, bias=False).to(self.device)
        
        self.fit_w.append(fit_w)
        self.true_w.append(true_w)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
