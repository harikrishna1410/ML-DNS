import torch
import torch.nn as nn
from neural_network_model import NeuralNetworkModel

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

    def forward(self, x):
        result = torch.zeros_like(x)
        if self.advection:
            result += self.advection(x)
        if self.reaction:
            result += self.reaction(x)
        if self.diffusion:
            result += self.diffusion(x)
        if self.force:
            result += self.force(x)
        return result

    def integrate(self, x, p):
        """
        Integrate the RHS components forward in time.
        
        Args:
        x (torch.Tensor): The current state of the system
        p (torch.Tensor): The current pressure
        
        Returns:
        torch.Tensor: The updated state after integration
        """
        if self.advection:
            x = self.advection.integrate(x)
        
        if self.force:
            x = self.force.integrate(x, p)
        
        # Ignoring diffusion and reaction for now
        
        return x

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

    def update_forward_method(self):
        method = self.options.get('method', 'incompressible')
        if method == 'compressible':
            self.forward = self.forward_compressible
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'incompressible' or 'compressible'.")
        

    
