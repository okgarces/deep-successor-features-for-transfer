# -*- coding: UTF-8 -*-  
import matplotlib.pyplot as plt

from agents.sfdqn import SFDQN
from agents.buffer import ReplayBuffer
from features.deep import DeepSF
from tasks.reacher import Reacher
from utils.config import parse_config_file
from utils.torch import set_torch_device, get_activation
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
logger = set_logger_level(use_logger=use_logger)

# tasks
def generate_tasks(include_target):
    train_tasks = [Reacher(all_goals, i, include_target) for i in range(len(goals))]
    test_tasks = [Reacher(all_goals, i + len(goals), include_target) for i in range(len(test_goals))]
    return train_tasks, test_tasks


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
    print('building SFDQN')
    deep_sf = DeepSF(pytorch_model_handle=sf_model_lambda, **sfdqn_params)
    sfdqn = SFDQN(deep_sf=deep_sf, buffer=ReplayBuffer(sfdqn_params['buffer_params']),
                  **sfdqn_params, **agent_params)

    # train SFDQN
    print('training SFDQN')
    train_tasks, test_tasks = generate_tasks(False)
    # sfdqn_perf = sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'])
    sfdqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'], cycles_per_task=n_cycles_per_task)

    # build DQN
    #print('building DQN')
    #dqn = DQN(model_lambda=dqn_model_lambda, buffer=ReplayBuffer(dqn_params['buffer_params']),
    #          **dqn_params, **agent_params)
    
    # training DQN
    #print('training DQN')
    #train_tasks, test_tasks = generate_tasks(True)
    #dqn_perf = dqn.train(train_tasks, n_samples, test_tasks=test_tasks, n_test_ev=agent_params['n_test_ev'])

    # smooth data    
    #def smooth(y, box_pts):
    #    return np.convolve(y, np.ones(box_pts) / box_pts, mode='same')

    #sfdqn_perf = smooth(sfdqn_perf, 10)[:-5]
    #dqn_perf = smooth(dqn_perf, 10)[:-5]
    #x = np.linspace(0, 4, sfdqn_perf.size)
    
    # reporting progress
    #ticksize = 14
    #textsize = 18
    #plt.rc('font', size=textsize)  # controls default text sizes
    #plt.rc('axes', titlesize=textsize)  # fontsize of the axes title
    #plt.rc('axes', labelsize=textsize)  # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=ticksize)  # fontsize of the tick labels
    #plt.rc('ytick', labelsize=ticksize)  # fontsize of the tick labels
    #plt.rc('legend', fontsize=ticksize)  # legend fontsize

    #plt.figure(figsize=(8, 6))
    #ax = plt.gca()
    #ax.plot(x, sfdqn_perf, label='SFDQN')
    #ax.plot(x, dqn_perf, label='DQN')
    #plt.xlabel('training task index')
    #plt.ylabel('averaged test episode reward')
    #plt.title('Testing Reward Averaged over all Test Tasks')
    #plt.tight_layout()
    #plt.legend(frameon=False)
    #plt.savefig('figures/sfdqn_return.png')


train()
