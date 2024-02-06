# -*- coding: UTF-8 -*-
import argparse


from sfdqn import SFDQN, ReplayBuffer, DeepSF
from tasks.reacher import Reacher
from utils.config import parse_config_file
from utils.torch import set_torch_device, get_activation, set_random_seed
from utils.logger import set_logger_level
from utils.types import ModelTuple

import torch
from collections import OrderedDict

# read parameters from config file
config_params = parse_config_file('reacher.cfg')

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

# Config GPU for Torch and logger
device = set_torch_device(use_gpu=use_gpu)
logger = None

# tasks
def generate_tasks(include_target):
    train_tasks = [Reacher(all_goals, i, include_target, device=device) for i in range(len(goals))]
    test_tasks = [Reacher(all_goals, i + len(goals), include_target, device=device) for i in range(len(test_goals))]
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
    return ReplayBuffer(**sfdqn_params['buffer_params'], device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSFDQN Reacher environment.')
    parser.add_argument('-seed', type=int, default=26)
    parser.add_argument('-experiment_name', type=str, default=None)

    args = parser.parse_args()

    # Seed and set logger
    set_random_seed(args.seed)
    logger = set_logger_level(use_logger=use_logger, prefix=args.experiment_name)

    train_tasks, test_tasks = generate_tasks(False)
    # build SFDQN    
    print('building SFDQN Sequential')
    deep_sf = DeepSF(pytorch_model_handle=sf_model_lambda, device=device, **sfdqn_params)
    sfdqn = SFDQN(deep_sf=deep_sf, buffer_handle=replay_buffer_handle, device=device,
                  **sfdqn_params, **agent_params)

    # train SFDQN
    print('training SFDQN Sequential')
    train_tasks, test_tasks = generate_tasks(False)
    sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'], cycles_per_task=n_cycles_per_task)
    print('End Training SFDQN Sequential')

