
import h5py
import torch
from data import SimulationState
from params import SimulationParameters
from grid import Grid

class Initializer:
    def __init__(self, params: SimulationParameters, sim_state: SimulationState, grid: Grid):
        self.sim_state = sim_state
        self.params = params
        self.grid = grid

    def initialize(self):
        if self.params.get('restart'):
            self._init_from_file()
        else:
            case_name = self.params.get('case')
            if case_name == 'pressure_pulse':
                self._init_pressure_pulse()
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
        x, y, z = self.grid.xl()
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Create a pressure pulse in the middle of the domain
        center = torch.tensor(self.params.case_params.get('center'))
        r = torch.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
        pressure = 1.0 + self.params.case_params.get('amplitude') * torch.exp(-r**2)
        
        # Compute density from pressure using the function in SimulationState
        self.sim_state.P = pressure*self.params.case_params.get('P_ambient')*self.params.P_atm/self.params.P_ref
        self.sim_state.T = torch.ones((nx, ny, nz))*self.params.case_params.get('T_ambient')/self.params.T_ref
        self.sim_state.rho = self.sim_state.compute_density_from_pressure(self.sim_state.P, self.sim_state.T)
        # Initialize velocity components
        self.sim_state.u = torch.zeros((self.sim_state.sim_params.ndim,) + tuple(self.sim_state.sim_params.nl))
        
        if(self.params.num_species > 0):
            self.sim_state.Ys = torch.ones((self.sim_state.sim_params.num_species,) 
                                       + tuple(self.sim_state.sim_params.nl))/self.params.num_species
        
        
        
