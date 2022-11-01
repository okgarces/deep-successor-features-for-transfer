from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.nn import Module

ModelTuple = tuple[Module, Loss, Optimizer]
