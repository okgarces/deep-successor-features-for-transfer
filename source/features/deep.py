# -*- coding: UTF-8 -*-
from features.successor import SF
from utils.torch import get_torch_device

import torch


class DeepSF(SF):
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
        super(DeepSF, self).__init__(*args, **kwargs)
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
            
        # build SF network and copy its weights from previous task
        # output shape is assumed to be [n_batch, n_actions, n_features]
        model, loss, optim = self.pytorch_model_handle(self.inputs, self.n_actions * self.n_features, (self.n_actions, self.n_features), 1)

        if source is not None and self.n_tasks > 0:
            source_psi_tuple, _ = self.psi[source]
            # model.set_weights(source_psi.get_weights())
            source_psi, *_ = source_psi_tuple
            self._update_models_weights(source_psi, model)
        
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
        self._update_models_weights(model, target_model)
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
    
    def update_successor(self, transitions, policy_index):
        if transitions is None:
            return
        states, actions, phis, next_states, gammas = transitions
        n_batch = len(gammas)
        indices = torch.arange(n_batch)
        gammas = gammas.reshape((-1, 1))
         
        # next actions come from GPI
        q1, _ = self.GPI(next_states, policy_index)
        next_actions = torch.argmax(torch.max(q1, axis=1).values, axis=-1)
        
        # compute the targets and TD errors
        psi_tuple, target_psi_tuple = self.psi[policy_index]
        psi_model, psi_loss, psi_optim = psi_tuple
        target_psi_model, *_ = target_psi_tuple

        current_psi = psi_model(states)
        targets = phis + gammas * target_psi_model(next_states)[indices, next_actions,:]
        
        # train the SF network
        merge_current_target_psi = current_psi.clone()
        merge_current_target_psi[indices, actions,:] = targets
        # psi_model.train_on_batch(states, current_psi)

        psi_optim.zero_grad()
        loss = psi_loss(current_psi, merge_current_target_psi)
        loss.backward()
        psi_optim.step()

        # Finish train the SF network
        
        # update the target SF network
        self.updates_since_target_updated[policy_index] += 1
        if self.updates_since_target_updated[policy_index] >= self.target_update_ev:
            self._update_models_weights(psi_model, target_psi_model)
            self.updates_since_target_updated[policy_index] = 0

    def _update_models_weights(self, model, target_model):
        for target_param, model_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(model_param.data)
