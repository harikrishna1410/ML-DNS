import torch
from params import SimulationParameters
from properties import FluidProperties

class SimulationState:
    def __init__(self, params:SimulationParameters,props:FluidProperties):
        self.nvars = params.nvars
        self.sim_params = params
        self.Fluid_props = props
        # Create tensors for primitives and solution
        #rho,u,v,w,T,Ys,....,E (total energy),p (pressure)
        self.primitives = torch.zeros((self.nvars+2,) + tuple(params.nl))
        ##soln: rho,rho*u,rho*v,rho*w,rho*e,rho*Ys
        self.soln = torch.zeros((self.nvars,) + tuple(params.nl))
        
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

    @property
    def rho_E(self):
        return self.soln[self.sim_params.ndim+1]

    @property
    def rho_Ys(self):
        return self.soln[self.sim_params.ndim+2:]

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
    ##id starts from 0
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
                                            * (self.Fluid_props.gamma - 1) * self.Fluid_props.MW_air \
                                            / self.Fluid_props.R_universal
        self.primitives[-1] = self.compute_pressure(self.primitives[0], self.primitives[self.sim_params.ndim+1])

    def compute_soln_from_primitives(self):
        # Compute density (rho)
        self.soln[0] = self.primitives[0]
        
        # Compute momentum (rho * u, rho * v, rho * w)
        for i in range(self.sim_params.ndim):
            self.soln[i+1] = self.primitives[0] * self.primitives[i+1]
        
        # Compute total energy (rho * E)
        kinetic_energy = 0.5 * torch.sum(self.primitives[1:self.sim_params.ndim+1]**2, dim=0)
        internal_energy = self.primitives[self.sim_params.ndim+1] * self.Fluid_props.R_universal / \
                          (self.Fluid_props.gamma - 1) / self.Fluid_props.MW_air
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
        R_specific = self.Fluid_props.R_universal / self.Fluid_props.MW_air  # Specific gas constant for air
        return rho * R_specific * T

    def compute_density_from_pressure(self, P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute density from pressure using the ideal gas equation.
        
        Args:
        P (torch.Tensor): Pressure in Pa
        T (torch.Tensor): Temperature in K
        
        Returns:
        torch.Tensor: Density in kg/m^3
        """
        R_specific = self.Fluid_props.R_universal / self.Fluid_props.MW_air  # Specific gas constant for air
        return P / (R_specific * T)
    
    

class SimulationData:
    def __init__(self, params : SimulationParameters,state: SimulationState):
        self.state = state
        self.params = params
        self.halo_depth = params.diff_order // 2
        
        # Initialize halos for all 6 directions (2 per axis)
        self.halos = {}
        if params.ndim >= 1:
            self.halos['x'] = torch.zeros((self.state.nvars, self.halo_depth, params.nl[1], params.nl[2]))  # x-direction halos (left and right)
        if params.ndim >= 2:
            self.halos['y'] = torch.zeros((self.state.nvars, params.nl[0], self.halo_depth, params.nl[2]))  # y-direction halos (left and right)
        if params.ndim == 3:
            self.halos['z'] = torch.zeros((self.state.nvars, params.nl[0], params.nl[1], self.halo_depth))   # z-direction halos (left and right)
        
        self.halo_xl = self.halos['x'][0]
        self.halo_xr = self.halos['x'][1]
        if(params.ndim>1):
            self.halo_yl = self.halos['y'][0]
            self.halo_yr = self.halos['y'][1]
        else:
            self.halo_yl = None
            self.halo_yr = None
        
        if params.ndim > 2:
            self.halo_zl = self.halos['z'][0]
            self.halo_zr = self.halos['z'][1]
        else:
            self.halo_zl = None
            self.halo_zr = None

    def zero_halos(self):
        """
        Set all halo values to zero.
        """
        for direction in self.halos:
            self.halos[direction].zero_()


