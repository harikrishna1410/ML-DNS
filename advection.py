from derivatives import Derivatives
from params import SimulationParameters
import torch
import torch.nn as nn
from neural_network_model import NeuralNetworkModel
from integrate import Integrator

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

    def forward_incompressible(self, x):
        if self.use_nn and self.nn_model:
            return self.nn_model(x)
        # Implement incompressible advection scheme
        pass

    def forward_compressible(self, x):
        if self.use_nn and self.nn_model:
            return self.nn_model(x)
        # Unpack variables from x
        rho = x[0]
        rho_u = x[1:self.sim_params.ndim+1]
        rho_E = x[self.sim_params.ndim+1]
        rho_Ys = x[self.sim_params.ndim+2:]  # Species mass fractions

        # Calculate velocities
        u = [rho_u[i] / rho for i in range(self.sim_params.ndim)]

        # Prepare tensor for gradient calculation
        grad_vars = [rho]
        for i in range(self.sim_params.ndim):
            grad_vars.extend([rho_u[j] * u[i] for j in range(self.sim_params.ndim)])
        grad_vars.extend([rho_E * u[i] for i in range(self.sim_params.ndim)])
        for rho_Y in rho_Ys:
            grad_vars.extend([rho_Y * u[i] for i in range(self.sim_params.ndim)])
        
        grad_vars = torch.stack(grad_vars)

        # Calculate all gradients together
        gradients = self.derivatives.gradient(grad_vars)

        # Initialize the result list
        result = []

        # Continuity equation
        result.append(-torch.sum(torch.stack([gradients[i][0] for i in range(self.sim_params.ndim)])))

        # Momentum equations
        for i in range(self.sim_params.ndim):
            result.append(-torch.sum(torch.stack([gradients[j][i+1] for j in range(self.sim_params.ndim)])))

        # Energy equation
        result.append(-torch.sum(torch.stack([gradients[i][self.sim_params.ndim+1] for i in range(self.sim_params.ndim)])))

        # Species conservation equations
        num_species = len(rho_Ys)
        for i in range(num_species):
            result.append(-torch.sum(torch.stack([
                gradients[j][self.sim_params.ndim+2+i] 
                for j in range(self.sim_params.ndim)
            ])))

        # Stack the results into a tensor
        result = torch.stack(result)

        return result
    
    def integrate(self, x):
        """
        Integrate the advection term forward in time.
        
        Args:
        x (torch.Tensor): The current state of the system
        
        Returns:
        torch.Tensor: The updated state after advection
        """
        
        rhs = self.forward(x)
        
        return self.integrator.integrate(x, rhs)