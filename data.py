import torch
from params import SimulationParameters
from properties import FluidProperties

class SimulationState:
    def __init__(self, params:SimulationParameters, props:FluidProperties):
        self.nvars = params.nvars
        self.sim_params = params
        self.Fluid_props = props
        # Create tensors for primitives and solution
        #rho,u,v,w,T,Ys,....,E (total energy),p (pressure)
        self.primitives = torch.zeros((self.nvars+2,) + tuple(params.nl))
        ##soln: rho,rho*u,rho*v,rho*w,rho*E,rho*Ys
        self.soln = torch.zeros((self.nvars,) + tuple(params.nl))
        self.time = 0.0  

    # Initialize properties as variables
    @property
    def rho(self):
        return self.soln[0]

    @rho.setter
    def rho(self, value):
        self.soln[0] = value

    @property
    def rho_u(self):
        return self.soln[1:self.sim_params.ndim+1]

    @rho_u.setter
    def rho_u(self, value):
        self.soln[1:self.sim_params.ndim+1] = value

    @property
    def rho_E(self):
        return self.soln[self.sim_params.ndim+1]

    @rho_E.setter
    def rho_E(self, value):
        self.soln[self.sim_params.ndim+1] = value

    @property
    def rho_Ys(self):
        return self.soln[self.sim_params.ndim+2:]

    @rho_Ys.setter
    def rho_Ys(self, value):
        self.soln[self.sim_params.ndim+2:] = value

    @property
    def u(self):
        return self.primitives[1:self.sim_params.ndim+1]

    @u.setter
    def u(self, value):
        self.primitives[1:self.sim_params.ndim+1] = value

    @property
    def T(self):
        return self.primitives[self.sim_params.ndim+1]

    @T.setter
    def T(self, value):
        self.primitives[self.sim_params.ndim+1] = value

    @property
    def P(self):
        return self.primitives[-1]

    @P.setter
    def P(self, value):
        self.primitives[-1] = value

    @property
    def E(self):
        return self.primitives[-2]
    
    @property
    def Ys(self):
        return self.primitives[self.sim_params.ndim + 2:]

    @Ys.setter
    def Ys(self, value):
        self.primitives[self.sim_params.ndim + 2:] = value
    
    def get_solution(self):
        return self.soln
    
    def compute_primitives_from_soln(self):
        # Compute density (rho)
        self.primitives[0] = self.soln[0]
        
        # Compute velocities (u, v, w)
        for i in range(self.sim_params.ndim):
            self.primitives[i+1] = self.soln[i+1] / self.soln[0]
        
        # Compute mass fractions (Ys)
        for i in range(self.nvars - self.sim_params.ndim - 2):
            self.primitives[self.sim_params.ndim+2+i] = self.soln[self.sim_params.ndim+2+i] / self.soln[0]
        
        self.primitives[-2] = self.soln[self.sim_params.ndim+1]/self.soln[0]
        # Compute temperature (T)
        self.primitives[self.sim_params.ndim+1] = (self.primitives[-2] \
                                            - 0.5 * torch.sum(self.primitives[1:self.sim_params.ndim+1]**2, dim=0))\
                                            * (self.Fluid_props.Cv)
        self.primitives[-1] = self.compute_pressure(self.primitives[0], self.primitives[self.sim_params.ndim+1])

    def compute_soln_from_primitives(self):
        # Compute density (rho)
        self.soln[0] = self.primitives[0]
        
        # Compute momentum (rho * u, rho * v, rho * w)
        for i in range(self.sim_params.ndim):
            self.soln[i+1] = self.primitives[0] * self.primitives[i+1]
        
        # Compute total energy (rho * E)
        kinetic_energy = 0.5 * torch.sum(self.primitives[1:self.sim_params.ndim+1]**2, dim=0)
        internal_energy = self.primitives[self.sim_params.ndim+1] * self.Fluid_props.Cv
        self.soln[self.sim_params.ndim+1] = self.primitives[0] * (kinetic_energy + internal_energy)
        
        # Compute species densities (rho * Ys)
        for i in range(self.nvars - self.sim_params.ndim - 2):
            self.soln[self.sim_params.ndim+2+i] = self.primitives[0] * self.primitives[self.sim_params.ndim+2+i]
        
    def compute_pressure(self, rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure using the ideal gas equation.
        
        Args:
        rho (torch.Tensor): Density in kg/m^3
        T (torch.Tensor): Temperature in K
        
        Returns:
        torch.Tensor: Pressure in Pa
        """
        return rho * self.Fluid_props.R * T

    def compute_density_from_pressure(self, P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute density from pressure using the ideal gas equation.
        
        Args:
        P (torch.Tensor): Pressure in Pa
        T (torch.Tensor): Temperature in K
        
        Returns:
        torch.Tensor: Density in kg/m^3
        """
        return P / (self.Fluid_props.R * T)
    
    

class SimulationData:
    def __init__(self, params : SimulationParameters,state: SimulationState):
        self.state = state
        self.sim_params = params
        ##assuming central difference
        self.halo_depth = params.diff_order // 2
        
        # Initialize halos for all 6 directions (2 per axis)
        self.halos = {}
        directions = ['x', 'y', 'z'][:self.sim_params.ndim]
        for i, direction in enumerate(directions):
            halo_shape = [2, self.state.nvars] + [self.halo_depth if j == i else params.nl[j] for j in range(self.sim_params.ndim)]
            self.halos[direction] = torch.zeros(halo_shape)
            setattr(self, f'halo_{direction}l', self.halos[direction][0])
            setattr(self, f'halo_{direction}r', self.halos[direction][1])
        

    def zero_halos(self):
        """
        Set all halo values to zero.
        """
        for direction in self.halos:
            self.halos[direction].zero_()


