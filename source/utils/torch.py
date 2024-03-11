import torch
import numpy as np
import random
import os

device = None 

def set_torch_device(use_gpu = False, gpu_device_index = 0):
    global device
    if device is None:
        cuda_available = use_gpu and torch.cuda.is_available()
        device = torch.device(f'cuda:{gpu_device_index}' if cuda_available else 'cpu')
        print(f"Using {device}")
    return device

def get_torch_device():
    return device

activations = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
} 

def get_activation(name):
    activation = activations.get(name)
    if activation is None:
        raise Exception('Activation name not supported')
    
    return activation

def update_models_weights(model: torch.nn.Module, target_model: torch.nn.Module):
    for target_param, model_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(model_param.data)

def get_parameters_norm_mean(model: torch.nn.Module):
    with torch.no_grad():
        # Log gradients
        accum_grads = 0
        accum_weights = 0
        mean = 0
        for params in model.parameters():
            accum_grads += torch.norm(params.grad)
            accum_weights += torch.norm(params.data)
            mean += params.data.mean().item()

    return torch.norm(accum_grads).item(), torch.norm(accum_weights).item(), mean

def set_random_seed(seed = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def project_on_simplex(v, z=1, epsilon=1e-7, device='cpu'):
    """ Adapted from https://pythonot.github.io/_modules/ot/utils.html#proj_simplex"""

    initial_shape = v.shape
    v = v.flatten()
    n = v.shape[0]

    # sort u in ascending order
    u = torch.sort(v, axis=0).values

    # take the descending order
    u = torch.flip(u, dims=(0,))
    cssv = torch.cumsum(u, axis=0) - z
    ind = torch.arange(n).to(device) + 1
    cond = u - cssv / ind > 0
    rho = torch.sum(cond, 0)

    theta = cssv[rho - 1] / rho

    w = torch.maximum(v - theta, torch.ones(v.shape).to(device) * epsilon)

    return w.view(initial_shape)
