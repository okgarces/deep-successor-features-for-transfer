# -*- coding: UTF-8 -*- 
from utils.config import parse_config_file
from utils.torch import set_torch_device, get_activation
from utils.logger import set_logger_level
from utils.types import ModelTuple

config_params = parse_config_file('reacher.cfg')

gen_params = config_params['GENERAL']
n_samples = gen_params['n_samples']
use_gpu= gen_params.get('use_gpu', False) # Default GPU False
gpu_device_index = gen_params.get('gpu_device_index', 0) # Default GPU False
use_logger= gen_params.get('use_logger', False) # Default GPU False
n_cycles_per_task = gen_params.get('cycles_per_task', 1) # Default GPU False

task_params = config_params['TASK']
goals = task_params['train_targets']
test_goals = task_params['test_targets']
all_goals = goals + test_goals
    
agent_params = config_params['AGENT']
sfdqn_params = config_params['SFDQN']

# Config GPU for Torch and logger
device = set_torch_device(use_gpu=use_gpu, gpu_device_index=gpu_device_index)
logger = set_logger_level(use_logger=use_logger)

from tsfdqn_nf import TSFDQN, ReplayBuffer, DeepTSF
from tasks.reacher import Reacher

import torch
from collections import OrderedDict

# read parameters from config file

# tasks
def generate_tasks(include_target):
    train_tasks = [Reacher(all_goals, i, include_target) for i in range(len(goals))]
    test_tasks = [Reacher(all_goals, i + len(goals), include_target) for i in range(len(test_goals))]
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

    loss = torch.nn.MSELoss().to(device)

    return model, loss, None

def replay_buffer_handle():
    return ReplayBuffer(sfdqn_params['buffer_params'])

def train():
    train_tasks, test_tasks = generate_tasks(False)
    # build SFDQN    
    print('building TSFDQN With NF Sequential')
    print(f'PyTorch Seed {torch.seed()}')
    deep_sf = DeepTSF(pytorch_model_handle=sf_model_lambda, **sfdqn_params)
    sfdqn = TSFDQN(deep_sf=deep_sf, buffer_handle=replay_buffer_handle,
                  **sfdqn_params, **agent_params)

    # train SFDQN
    print('training TSFDQN With NF Sequential')
    train_tasks, test_tasks = generate_tasks(False)
    sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'], cycles_per_task=n_cycles_per_task)
    print('End Training TSFDQN with NF Sequential')

train()
