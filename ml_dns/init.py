import h5py
import torch
from .data import SimulationState
from .params import SimulationParameters
from .grid import Grid
from .properties import FluidProperties

class Initializer:
    def __init__(self, params: SimulationParameters, 
                 sim_state: SimulationState, 
                 grid: Grid, props: FluidProperties):
        self.sim_state = sim_state
        self.params = params
        self.grid = grid
        self.props = props

    def initialize(self):
        if self.params.get('restart'):
            self._init_from_file()
        else:
            case_name = self.params.get('case')
            if case_name == 'pressure_pulse':
                self._init_pressure_pulse()
            elif case_name == 'hotspot':
                if(self.params.ndim == 2):
                    self._init_hotspot()
                else:
                    self._init_hotspot_1d()
            elif("isentropic_vortex" in case_name):
                self._init_isentropic_vortex()
            else:
                raise ValueError(f"Unknown initialization case: {case_name}")
        self.sim_state.compute_soln_from_primitives()
       
    def _init_from_file(self):
        file_path = self.params.get('restart_file')
        if not file_path:
            raise ValueError("restart_file must be provided when restart is True")
        
        with h5py.File(file_path, 'r') as f:
            for var in self.sim_state.variables:
                self.sim_state[var] = torch.tensor(f[var][:])

    def _init_pressure_pulse(self):
        nx, ny, nz = self.params.nl
        x_tup = self.grid.xl()
        
        x_norm_list = []
        for dim, coord in enumerate(["x", "y", "z"][:self.params.ndim]):
            x = x_tup[dim]
            x_norm = (x - self.params.domain_extents[f"{coord}s"]) / \
                     (self.params.domain_extents[f"{coord}e"] - self.params.domain_extents[f"{coord}s"])
            x_norm_list.append(x_norm)

        X, Y, Z = torch.meshgrid(
            x_norm_list[0],
            x_norm_list[1] if self.params.ndim > 1 else torch.tensor([0.0]),
            x_norm_list[2] if self.params.ndim > 2 else torch.tensor([0.0]),
            indexing='ij'
        )
        
        # Create a pressure pulse in the middle of the domain
        center = torch.tensor(self.params.case_params.get('center'))
        r = torch.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        pressure = 1.0 + self.params.case_params.get('amplitude') * torch.exp(-1000*r**2)
        
        
        # Compute density from pressure using the function in SimulationState
        self.sim_state.P = pressure*self.params.case_params.get('P_ambient')*self.params.P_atm/self.params.P_ref
        self.sim_state.T = torch.ones((nx, ny, nz))*self.params.case_params.get('T_ambient')/self.params.T_ref
        self.sim_state.rho = self.sim_state.compute_density_from_pressure(self.sim_state.P, self.sim_state.T)
        
        # Initialize velocity components
        u = torch.zeros((self.params.ndim,) + tuple(self.params.nl))
        self.sim_state.u = u
        if(self.params.num_species > 0):
            self.sim_state.Ys = torch.ones((self.sim_state.sim_params.num_species,) 
                                       + tuple(self.sim_state.sim_params.nl))/self.params.num_species
    
    ##a hotspot with zero pressure gradient
    def _init_hotspot(self):
        if(self.params.ndim > 2):
            raise ValueError("ndim > 2")
        x, y = self.grid.xl()
        x_norm = (x-self.params.domain_extents["xs"])\
            /(self.params.domain_extents["xe"]\
              -self.params.domain_extents["xs"])
        y_norm = (y-self.params.domain_extents["ys"])\
            /(self.params.domain_extents["ye"]\
              -self.params.domain_extents["ys"])
        X, Y = torch.meshgrid(x_norm, y_norm, indexing='ij')
        
        # Create a pressure pulse in the middle of the domain
        center = torch.tensor(self.params.case_params.get('center'))
        r = torch.sqrt((X - center[0])**2 + (Y - center[1])**2).unsqueeze(-1)
        temperature = 1.0 + self.params.case_params.get('amplitude') * torch.exp(-1000*r**2)
        
        # Compute density from pressure using the function in SimulationState
        self.sim_state.P = torch.ones_like(r)*self.params.case_params.get('P_ambient')*self.params.P_atm/self.params.P_ref
        self.sim_state.T = temperature*self.params.case_params.get('T_ambient')/self.params.T_ref
        self.sim_state.rho = self.sim_state.compute_density_from_pressure(self.sim_state.P, self.sim_state.T)
        
        # Initialize velocity components
        u = torch.zeros((self.params.ndim,) + tuple(self.params.nl))
        u[0] = 1.0/self.params.a_ref
        self.sim_state.u = u

    ##a hotspot with zero pressure gradient
    def _init_hotspot_1d(self):
        if(self.params.ndim > 1):
            raise ValueError("ndim > 1")
        x = self.grid.xl()[0]
        x_norm = (x-self.params.domain_extents["xs"])\
            /(self.params.domain_extents["xe"]\
              -self.params.domain_extents["xs"])
        
        # Create a pressure pulse in the middle of the domain
        center = torch.tensor(self.params.case_params.get('center'))
        r = (x_norm - center[0]).unsqueeze(-1).unsqueeze(-1)
        temperature = (1.0 + self.params.case_params.get('amplitude') \
                       * torch.exp(-1000 * r**2.0))
                       #* torch.sin(x_norm*torch.pi*2.0)).unsqueeze(-1).unsqueeze(-1)
        
        # Compute density from pressure using the function in SimulationState
        self.sim_state.P = torch.ones_like(r)*self.params.case_params.get('P_ambient')*self.params.P_atm/self.params.P_ref
        self.sim_state.T = temperature*self.params.case_params.get('T_ambient')/self.params.T_ref
        self.sim_state.rho = self.sim_state.compute_density_from_pressure(self.sim_state.P, self.sim_state.T)
        
        # Initialize velocity components
        u = torch.zeros((self.params.ndim,) + tuple(self.params.nl))
        u[0] = 1.0/self.params.a_ref
        self.sim_state.u = u

    def _init_isentropic_vortex(self):
        if(self.params.ndim != 2):
            raise ValueError("ndim != 2")
        x, y = self.grid.xl()
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X = X.unsqueeze(-1)
        Y = Y.unsqueeze(-1)
    
        # Create an isentropic vortex
        center = torch.tensor([(self.params.domain_extents["xs"] + self.params.domain_extents["xe"])/2.0,
                           (self.params.domain_extents["ys"] + self.params.domain_extents["ye"])/2.0])
        r = torch.sqrt((X - center[0])**2 + (Y - center[1])**2)
        gamma = self.params.gamma
        beta = self.params.case_params.get('beta', 5.0)  # Changed default to typical value
    
        # Calculate perturbations
        f = (1 - r**2) / 2
        du = -beta / (2 * torch.pi) * (Y - center[1]) * torch.exp(f)
        dv = beta / (2 * torch.pi) * (X - center[0]) * torch.exp(f)
        dT = -(gamma - 1) * beta**2 / (8 * gamma * torch.pi**2) * torch.exp(2*f)
    
        # Set background conditions
        u_inf = self.params.case_params.get('u_inf', 1.0)
        v_inf = self.params.case_params.get('v_inf', 1.0)
        T_inf = self.params.case_params.get('T_inf', 1.0)
        p_inf = self.params.case_params.get('p_inf', 1.0)
    
        # Compute final state
        u = u_inf + du
        v = v_inf + dv
        T = T_inf * (1 + dT)
        p = p_inf * (T / T_inf)**(gamma / (gamma - 1))
        rho = p / (self.props.R * T)
    
        self.sim_state.u = torch.stack((u, v), dim=0)
        self.sim_state.T = T
        self.sim_state.rho = rho
        self.sim_state.P = p
        self.sim_state.E = self.props.Cv*T + 0.5*(u**2+v**2)
        