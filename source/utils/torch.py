import torch

device = None 

def set_torch_device(use_gpu = False):
    global device
    if device is None:
        cuda_available = use_gpu and torch.cuda.is_available()
        device = torch.device('cuda' if cuda_available else 'cpu')
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
