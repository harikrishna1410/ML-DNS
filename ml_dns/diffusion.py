import torch
import torch.nn as nn
from .data import SimulationState
from .derivatives import Derivatives
from .params import SimulationParameters
from .properties import FluidProperties
from .integrate import Integrator

class Diffusion(nn.Module):
    def __init__(self, 
                params: SimulationParameters, 
                derivatives: Derivatives, 
                integrator: Integrator,
                fluid_props: FluidProperties, 
                nn_model=None, 
                use_nn=False):
        super().__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.integrator = integrator
        self.fluid_props = fluid_props
        self.nn_model = nn_model
        self.use_nn = use_nn

    def forward(self, state: SimulationState, result = None):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        
        # if result is None:
        result = torch.zeros_like(state.soln)
        
        # Viscous stress tensor contribution
        tau = self.compute_stress_tensor(state)
        heat_flux = self.compute_heat_flux(state)
        for i in range(self.sim_params.ndim):
            result[i+1] += self.derivatives.divergence(tau[i])
            result[self.sim_params.ndim+1] += self.derivatives.divergence(tau[i] * state.u[i])
        
        # Energy equation contribution from heat flux
        result[self.sim_params.ndim+1] -= self.derivatives.divergence(heat_flux)
        
        # Species diffusion contribution
        if self.sim_params.num_species > 0:
            raise ValueError("Diffusion:nspecies > 0")
            # species_fluxes = self.compute_species_diffusion_flux(state)
            # for i in range(self.sim_params.num_species):
            #     result[self.sim_params.ndim+2+i] = self.derivatives.divergence(species_fluxes[i])
        
        return result

    def compute_stress_tensor(self, state: SimulationState):
        # Compute viscosity
        mu = self.fluid_props.calculate_viscosity(state.T * self.sim_params.T_ref)
        
        # Compute strain rate tensor
        gradu = torch.stack(self.derivatives.gradient(state.u),dim=0) ##first dim is the gradient direction
        divu = gradu.diagonal(offset=0,dim1=0,dim2=1).sum(-1).unsqueeze(0).unsqueeze(0)\
                *torch.eye(self.sim_params.ndim).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gradu += gradu.clone().transpose(0,1)
        gradu /=2
        # Compute stress tensor
        tau = 2 * mu * gradu - (2/3) * mu * divu
        
        return tau
    ##
    def compute_heat_flux(self, state: SimulationState):
        # Compute heat flux
        k = self.fluid_props.calculate_thermal_conductivity(state.T * self.sim_params.T_ref)
        heat_flux = -k * torch.stack(self.derivatives.gradient(state.T),dim=0)
        return heat_flux

    def compute_strain_rate_tensor(self, state: SimulationState):
        # Compute strain rate tensor

        return S

    def integrate(self, state: SimulationState):
        return self.integrator.integrate(state, self.forward)

    
    # def compute_species_diffusion_flux(self, state: SimulationState):
    #     # Compute species diffusion coefficients
    #     D = self.calculate_species_diffusion_coefficients(state)
        
    #     # Compute species gradients
    #     Y_gradients = [self.derivatives.gradient(state.Y[i]) for i in range(self.sim_params.num_species)]
        
    #     # Compute species diffusion fluxes
    #     fluxes = [-state.rho * D[i] * Y_gradients[i] for i in range(self.sim_params.num_species)]
        
    #     return fluxes