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


def set_random_seed(seed = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')
