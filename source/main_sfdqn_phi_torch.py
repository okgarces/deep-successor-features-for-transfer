# -*- coding: UTF-8 -*-  
import matplotlib.pyplot as plt

from agents.sfdqn_phi import SFDQN_PHI
from features.deep_phi import DeepSF_PHI
from tasks.reacher_phi import Reacher_PHI
from agents.buffer_phi import ReplayBuffer_PHI
from utils.config import parse_config_file
from utils.torch import set_torch_device, get_activation
from utils.logger import set_logger_level
from utils.types import ModelTuple

from collections import OrderedDict
import torch
import random
import numpy as np

# read parameters from config file
config_params = parse_config_file('reacher_phi.cfg')

gen_params = config_params['GENERAL']
n_samples = gen_params['n_samples']
use_gpu= gen_params.get('use_gpu', False) # Default GPU False
use_logger= gen_params.get('use_logger', False) # Default GPU False
n_cycles_per_task = gen_params.get('cycles_per_task', 1) # Default GPU False

task_params = config_params['TASK']
goals = task_params['train_targets']
test_goals = task_params['test_targets']
all_goals = goals + test_goals
    
agent_params = config_params['AGENT']
sfdqn_params = config_params['SFDQN']

phi_params = config_params['PHI']

# Config GPU for Torch and logger
device = set_torch_device(use_gpu=use_gpu)
logger = set_logger_level(use_logger=use_logger)

# manual seed
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# tasks
def generate_tasks(include_target):
    feature_dim = phi_params['n_features']
    train_tasks = [Reacher_PHI(all_goals, i, feature_dim, include_target) for i in range(len(goals))]
    test_tasks = [Reacher_PHI(all_goals, i + len(goals), feature_dim, include_target) for i in range(len(test_goals))]
    return train_tasks, test_tasks


def phi_model_lambda(s_enc_dim, action_dim, feature_dim) -> ModelTuple:
    model_params = phi_params['model_params']
    learning_rate = model_params['learning_rate']

    feature_dim = phi_params['n_features']

    model = torch.nn.Sequential(
        torch.nn.Linear(np.sum([s_enc_dim, action_dim, s_enc_dim]), 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, feature_dim)
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.MSELoss().to(device)

    return model, loss, optim


def sf_model_lambda(num_inputs: int, output_dim: int, reshape_dim: tuple, reshape_axis: int = 1) -> ModelTuple:
    n_features = len(all_goals)
    model_params = sfdqn_params['model_params']

    layers = OrderedDict() 
    number_layers = len(model_params['n_neurons'])

    # Layers settings
    first_layer_neurons = model_params['n_neurons'][0]
    last_layer_neurons = model_params['n_neurons'][-1]
    learning_rate = model_params['learning_rate']

    # Input Layer
    input_layer = torch.nn.Linear(num_inputs, first_layer_neurons)
    layers['layer_input'] = input_layer
    
    # Hidden Layer
    for index, n_neurons, activation in zip(range(number_layers), model_params['n_neurons'], model_params['activations']):
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

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.MSELoss().to(device)

    return model, loss, optim

def train():
    train_tasks, test_tasks = generate_tasks(False)
    # build SFDQN    
    print('building SFDQN with phi Learning')
    deep_sf = DeepSF_PHI(pytorch_model_handle=sf_model_lambda, **sfdqn_params)
    # sf_model_lambda could be another kind of lambda
    sfdqn = SFDQN_PHI(deep_sf=deep_sf, lambda_phi_model=phi_model_lambda, buffer=ReplayBuffer_PHI(sfdqn_params['buffer_params']),
                  **sfdqn_params, **agent_params)

    # train SFDQN
    print('training SFDQN')
    train_tasks, test_tasks = generate_tasks(False)
    # sfdqn_perf = sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'])
    sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'], cycles_per_task=n_cycles_per_task)

train()
