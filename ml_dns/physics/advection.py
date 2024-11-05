import torch
import torch.nn as nn
from ..ml import NeuralNetworkModel
from ..numerics import Derivatives
from ..core import BaseSimulationState, SimulationParameters, CompressibleFlowState

class Advection(nn.Module):
    def __init__(self,
                 params: SimulationParameters,
                 derivatives: Derivatives, 
                 nn_model: NeuralNetworkModel = None, 
                 use_nn: bool = False):
        super(Advection, self).__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.nn_model = nn_model
        self.use_nn = use_nn
        
        if self.derivatives is None:
            raise ValueError("A 'derivatives' object must be provided.")
        
        # Set the forward method based on the method parameter
        if self.sim_params.fluidtype == 'compressible_newtonian':
            self.forward = self.forward_compressible
        else:
            raise ValueError(f"Unsupported fluid type {self.sim_params.fluidtype}")

    def get_use_nn(self) -> bool:
        return self.use_nn

    def set_use_nn(self, value: bool) -> None:
        self.use_nn = value

    def forward_compressible(self, state: CompressibleFlowState):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        
        # Access variables directly from SimulationState
        rho = state.get_primitive_var("rho")
        u = state.get_primitive_var("u")
        rho_u = state.get_solution_var("rho_u")
        rho_E = state.get_solution_var("rho_E")
        rho_Y = state.get_solution_var("rho_Y")
        
        # Prepare tensor for gradient calculation
        flux = [torch.stack([rho*u[i] for i in range(self.sim_params.ndim)],dim=0).unsqueeze(1)]
        
        for i in range(self.sim_params.ndim):
            flux.extend([torch.stack([rho_u[i] * u[j] for j in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        flux.extend([torch.stack([rho_E * u[i] for i in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        for i in range(self.sim_params.num_species):  # For each species
            flux.extend([torch.stack([rho_Y[i] * u[j] for j in range(self.sim_params.ndim)],dim=0).unsqueeze(1)])
        
        flux = torch.cat(flux,dim=1)

        # Calculate all gradients together and return the negative
        result = -self.derivatives.divergence(flux)

        return result
