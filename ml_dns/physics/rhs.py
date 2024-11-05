import torch
import torch.nn as nn
from ..ml import NeuralNetworkModel
from ..core import BaseSimulationState

class RHS(nn.Module):
    def __init__(self, 
                 advection=None, 
                 reaction=None, 
                 diffusion=None, 
                 force=None):
        super(RHS, self).__init__()
        self.advection = advection
        self.reaction = reaction
        self.diffusion = diffusion
        self.force = force

    def forward(self, state: BaseSimulationState):
        result = torch.zeros_like(state.get_solution())
        if self.advection:
            result += self.advection(state)
        if self.reaction:
            result += self.reaction(state)
        if self.diffusion:
            result += self.diffusion(state)
        if self.force:
            result += self.force(state)
        return result

    def toggle_use_nn(self, component):
        if component == 'advection' and self.advection:
            self.advection.set_use_nn(not self.advection.get_use_nn())
        elif component == 'reaction' and self.reaction:
            self.reaction.set_use_nn(not self.reaction.get_use_nn())
        elif component == 'diffusion' and self.diffusion:
            self.diffusion.set_use_nn(not self.diffusion.get_use_nn())
        elif component == 'force' and self.force:
            self.force.set_use_nn(not self.force.get_use_nn())
        else:
            raise ValueError(f"Invalid component '{component}' or component not in use")

    def get_use_nn_status(self):
        status = {}
        if self.advection:
            status['advection'] = self.advection.get_use_nn()
        if self.reaction:
            status['reaction'] = self.reaction.get_use_nn()
        if self.diffusion:
            status['diffusion'] = self.diffusion.get_use_nn()
        if self.force:
            status['force'] = self.force.get_use_nn()
        return status
        

    
