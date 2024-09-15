from derivatives import Derivatives
from params import SimulationParameters
import torch
import torch.nn as nn
from neural_network_model import NeuralNetworkModel
from integrate import Integrator

class Force(nn.Module):
    def __init__(self,
                 params: SimulationParameters,
                 derivatives: Derivatives,
                 integrator: Integrator,
                 nn_model: NeuralNetworkModel = None, 
                 use_nn: bool = False,
                 use_buoyancy: bool = False):
        super(Force, self).__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.nn_model = nn_model
        self.use_nn = use_nn
        self.use_buoyancy = use_buoyancy
        self.integrator = integrator
        if(self.use_buoyancy):
            raise ValueError("Can't use buoyancy")
        
        if self.derivatives is None:
            raise ValueError("A 'derivatives' object must be provided.")

    def set_use_nn(self, value: bool):
        self.use_nn = value

    def get_use_nn(self) -> bool:
        return self.use_nn
    
    def forward(self, x, p):
        if self.use_nn and self.nn_model:
            return self.nn_model(x)
        
        # Unpack variables from x
        rho = x[0]
        rho_u = x[1:self.sim_params.ndim+1]
        rho_E = x[self.sim_params.ndim+1]
        rho_Ys = x[self.sim_params.ndim+2:]

        # Initialize the result list
        result = torch.zeros_like(x)
        # Add pressure gradient term
        p_gradients = self.derivatives.gradient(p)
        for i in range(self.sim_params.ndim):
            result[i+1] -= p_gradients[i]  # Negative pressure gradient force on momentum
        ##
        p_u = torch.stack([p*rho_u[i] / rho for i in range(self.sim_params.ndim)],dim=0)
        result[self.sim_params.ndim+1] -= self.derivatives.divergence(p_u)

        return result

    def integrate(self, x, p):
        """
        Integrate the force term forward in time.
        
        Args:
        x (torch.Tensor): The current state of the system
        p (torch.Tensor): The current pressure
        
        Returns:
        torch.Tensor: The updated state after applying the force
        """
        
        rhs = self.forward(x, p)
        
        return self.integrator.integrate(x, rhs)
