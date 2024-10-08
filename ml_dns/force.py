from .derivatives import Derivatives
from .params import SimulationParameters
import torch
import torch.nn as nn
from .neural_network_model import NeuralNetworkModel
from .integrate import Integrator
from .data import SimulationState

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
    
    def forward(self, state: SimulationState,result = None):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        
        # Initialize the result tensor
        # if result is None:
        result = torch.zeros_like(state.soln)
        
        # Add pressure gradient term
        p_gradients = self.derivatives.gradient(state.P)
        for i in range(self.sim_params.ndim):
            result[i+1] -= p_gradients[i]  # Negative pressure gradient force on momentum
        
        # Compute p_u
        p_u = torch.stack([state.P * state.rho_u[i] / state.soln[0] for i in range(self.sim_params.ndim)], dim=0)
        result[self.sim_params.ndim+1] -= self.derivatives.divergence(p_u)

        return result

    def integrate(self, state: SimulationState):
        """
        Integrate the force term forward in time.
        
        Args:
        state (SimulationState): The current state of the system
        
        Returns:
        SimulationState: The updated state after applying the force
        """
        return self.integrator.integrate(state, self.forward)
