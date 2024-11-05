import torch
import torch.nn as nn
from ..ml import NeuralNetworkModel
from ..core import SimulationParameters

class Reaction(nn.Module):
    def __init__(self, nn_model=None, use_nn=False):
        super(Reaction, self).__init__()
        self.nn_model = nn_model
        self.use_nn = use_nn

    def forward(self, x):
        if self.use_nn and self.nn_model:
            return self.nn_model(x)
        # Implement default Reaction forward pass
        pass