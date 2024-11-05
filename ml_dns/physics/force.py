import torch
import torch.nn as nn
from ..numerics import Derivatives
from ..core import SimulationParameters, BaseSimulationState, CompressibleFlowState
from ..ml import NeuralNetworkModel


class Force(nn.Module):
    def __init__(self,
                 params: SimulationParameters,
                 derivatives: Derivatives,
                 nn_model: NeuralNetworkModel = None, 
                 use_nn: bool = False,
                 use_buoyancy: bool = False):
        super(Force, self).__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.nn_model = nn_model
        self.use_nn = use_nn
        self.use_buoyancy = use_buoyancy
        if(self.use_buoyancy):
            raise ValueError("Can't use buoyancy")

    def set_use_nn(self, value: bool):
        self.use_nn = value

    def get_use_nn(self) -> bool:
        return self.use_nn
    
    def forward(self, state: BaseSimulationState):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)

        u = state.get_primitive_var("u")
        P = state.get_primitive_var("P")
        
        result = torch.zeros_like(state.get_solution())
        
        # Add pressure gradient term
        p_gradients = self.derivatives.gradient(P)
        
        for i in range(self.sim_params.ndim):
            result[i+1] -= p_gradients[i]  # Negative pressure gradient force on momentum
        
        result[self.sim_params.ndim+1] -= self.derivatives.divergence(P*u)



        return result
