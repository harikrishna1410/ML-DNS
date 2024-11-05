import torch
import torch.nn as nn
from ..core import BaseSimulationState, SimulationParameters, FluidProperties
from ..core import CompressibleFlowState
from ..numerics import Derivatives

class Diffusion(nn.Module):
    def __init__(self, 
                params: SimulationParameters, 
                derivatives: Derivatives, 
                fluid_props: FluidProperties,
                nn_model=None, 
                use_nn=False):
        super().__init__()
        self.sim_params = params
        self.derivatives = derivatives
        self.fluid_props = fluid_props
        self.nn_model = nn_model
        self.use_nn = use_nn

        if self.sim_params.fluidtype == "compressible_newtonian":
            self.forward = self.forward_compressible
        else:
            raise ValueError("Unsupported fluid type")

    def forward_compressible(self, state: CompressibleFlowState, result = None):
        if self.use_nn and self.nn_model:
            return self.nn_model(state)
        
        # Viscous stress tensor contribution
        u = state.get_primitive_var("u")
        tau = self.compute_stress_tensor(state)
        heat_flux = self.compute_heat_flux(state)

        flux = [torch.zeros_like(tau[:,0:1,:,:,:])]
        flux.extend([tau[:,i:i+1,:,:,:] for i in range(self.sim_params.ndim)])
        flux.append(tau[:,0:1,:,:,:]*u[0:1,:,:,:].unsqueeze(0))
        for i in range(1,self.sim_params.ndim):
            flux[-1] += (tau[:,i:i+1,:,:,:] * u[i:i+1,:,:,:].unsqueeze(0))
        flux[-1] -= heat_flux.unsqueeze(1)
        
        # Species diffusion contribution
        if self.sim_params.num_species > 0:
            raise ValueError("Diffusion:nspecies > 0")
            species_fluxes = self.compute_species_diffusion_flux(state)
            for i in range(self.sim_params.num_species):
                flux.append(species_fluxes[:,i:i+1,:,:,:])
        flux = torch.cat(flux,dim=1)
        result = self.derivatives.divergence(flux)
        return result

    def compute_stress_tensor(self, state: CompressibleFlowState):
        # Compute viscosity
        T = state.get_primitive_var("T") * self.sim_params.T_ref
        u = state.get_primitive_var("u")
        mu = self.fluid_props.calculate_viscosity(T)
        
        # Compute strain rate tensor
        gradu = torch.stack(self.derivatives.gradient(u),dim=0) ##first dim is the gradient direction
        divu = gradu.diagonal(offset=0,dim1=0,dim2=1).sum(-1).unsqueeze(0).unsqueeze(0)\
                *torch.eye(self.sim_params.ndim).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gradu += gradu.clone().transpose(0,1)
        gradu /=2
        # Compute stress tensor
        tau = 2 * mu * gradu - (2/3) * mu * divu
        
        return tau
    ##
    def compute_heat_flux(self, state: CompressibleFlowState):
        # Compute heat flux
        T = state.get_primitive_var("T")
        k = self.fluid_props.calculate_thermal_conductivity(T * self.sim_params.T_ref)
        heat_flux = -k * torch.stack(self.derivatives.gradient(T),dim=0)
        return heat_flux

    def compute_species_diffusion_flux(self, state: CompressibleFlowState):
        T = state.get_primitive_var("T") * self.sim_params.T_ref
        # Compute species diffusion coefficients
        D = self.fluid_props.calculate_species_diffusivity(T)
        rho = self.get_primitive_var("rho")
        Y = state.get_primitive_var("Y")
        # Compute species gradients
        Y_gradients = torch.stack(self.derivatives.gradient(Y),dim=0)
        
        # Compute species diffusion fluxes
        fluxes = -rho * D * Y_gradients #(1,1,...)*(1,nspec,...)*(3,nspec,...)
        
        return fluxes