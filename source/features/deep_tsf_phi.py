# -*- coding: UTF-8 -*-
from features.successor import SF
from utils.torch import get_torch_device, update_models_weights

import torch
import numpy as np


class DeepSF_TSF_PHI(SF):
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
        super(DeepSF_TSF_PHI, self).__init__(*args, **kwargs)
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

    def update_reward(self, phi: torch.Tensor, r: torch.Tensor, task_index: int, exact=False) -> None:
        raise Exception("This function shouldn't be used")
    
    def update_successor(self, transitions, phis_models, policy_index):
        raise Exception("This function shouldn't be used")

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
        fit_w = torch.nn.Linear(n_features, 1).to(self.device)
        
        self.fit_w.append(fit_w)
        self.true_w.append(true_w)
        
        # add statistics
        for i in range(len(self.gpi_counters)):
            self.gpi_counters[i] = np.append(self.gpi_counters[i], 0)
        self.gpi_counters.append(np.zeros((self.n_tasks,), dtype=int))
