from .derivatives import Derivatives
from .params import SimulationParameters
import torch
import torch.nn as nn
from .neural_network_model import NeuralNetworkModel
from .integrate import Integrator
from .data import SimulationState

class Advection(nn.Module):
    def __init__(self,
                 params: SimulationParameters,
                 derivatives: Derivatives, 
                 integrator: Integrator,
                 nn_model: NeuralNetworkModel = None, 
                 use_nn: bool = False, 
                 method: str = 'compressible'):
        super(Advection, self).__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.nn_model = nn_model
        self.use_nn = use_nn
        self.integrator = integrator
        
        if self.derivatives is None:
            raise ValueError("A 'derivatives' object must be provided.")
        
        # Set the forward method based on the method parameter
        if method == 'compressible':
            self.forward = self.forward_compressible
        elif method == 'incompressible':
            self.forward = self.forward_incompressible
        else:
            raise ValueError(f"Invalid method '{method}'. Choose 'compressible' or 'incompressible'.")

    def get_use_nn(self) -> bool:
        """
        Get the current state of the use_nn flag.

        Returns:
            bool: The current state of use_nn
        """
        return self.use_nn

    def set_use_nn(self, value: bool) -> None:
        """
        Set the state of the use_nn flag.

        Args:
            value (bool): The new state for use_nn
        """
        self.use_nn = value

    def forward_incompressible(self, state: SimulationState):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        # Implement incompressible advection scheme
        pass

    def forward_compressible(self, state: SimulationState):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        
        # Access variables directly from SimulationState
        rho = state.rho
        u = state.u
        
        # Prepare tensor for gradient calculation
        flux = [torch.stack([rho*u[i] for i in range(self.sim_params.ndim)],dim=0).unsqueeze(1)]
        
        for i in range(self.sim_params.ndim):
            flux.extend([torch.stack([state.soln[j+1] * u[i] for j in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        flux.extend([torch.stack([state.rho_E * u[i] for i in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        for i in range(self.sim_params.nvars - self.sim_params.ndim - 2):  # For each species
            flux.extend([torch.stack([state.rho_Ys[i] * u[j] for j in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        
        flux = torch.cat(flux,dim=1)
        # Calculate all gradients together
        result = self.derivatives.divergence(flux)

        return result
    
    def integrate(self, state: SimulationState):
        """
        Integrate the advection term forward in time.
        
        Args:
        state (SimulationState): The current state of the system
        
        Returns:
        SimulationState: The updated state after advection
        """
        
        return self.integrator.integrate(state, self.forward)
