from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.nn import Module
from typing import Tuple

ModelTuple = Tuple[Module, Loss, Optimizer|None]
