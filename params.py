import json
import numpy as np

class SimulationParameters:
    def __init__(self, json_file, topo, my_idx):
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        self.case = params.get("case", "pressure_pulse")
        self.restart = params.get("restart", False)
        self.restart_file = params.get("restart_file", "restart.h5")
        self.case_params = params.get("case_params", {})

        # number of dimensions
        self.ndim = params.get("ndim", 3)
        # number of grid points
        self.ng = (params.get("nxg", 1), params.get("nyg", 1), params.get("nzg", 1))
        # number of processors
        self.np = (params.get("npx", 1), params.get("npy", 1), params.get("npz", 1))
        # my index
        self.my_idx = my_idx

        # Domain parameters 
        self.domain_extents = params.get("domain_extents", 
                                         {"xs": 0.0, "xe": 1.0, 
                                          "ys": 0.0, "ye": 1.0, 
                                          "zs": 0.0, "ze": 1.0})
        self.grid_stretching_params = params.get("grid_stretching_params", {})
        
        # Periodic boundary conditions
        self.periodic_bc = params.get("periodic_bc", [True, True, True])
        
        # Time stepping parameters
        self.dt = params.get("dt", 0.001)
        self.num_steps = params.get("num_steps", 1000)
        self.cfl = params.get("cfl", 0.5)  # Courant-Friedrichs-Lewy condition

        # Physical parameters
        self.num_species = params.get("num_species", 0)
        if self.num_species > 0:
            raise ValueError("Num species > 0")
        self.use_buoyancy = params.get("use_buoyancy", False)
        self.advection_method = params.get("advection_method", "compressible")

        #fluid properties
        self.gamma = params.get("gamma", 1.4)  # Specific heat ratio
        
        # Numerical method parameters
        self.diff_order = params.get("diff_order", 8)  # Order of difference scheme
        
        # Compute local grid points
        self.nl = tuple([
            self.ng[i] // self.np[i] + 
            (self.ng[i] % self.np[i] if my_idx[i] == self.np[i] - 1 else 0)
            for i in range(3)
        ])
        
        # Solver options
        self.use_nn = params.get("use_nn", False)
        
        # New parameters
        self.nvars = self.ndim + 2 + self.num_species  # Total number of variables

        # Output parameters
        self.output_format = params.get("output_format", "hdf5")
        self.output_frequency = params.get("output_frequency", 100)
        self.output_variables = params.get("output_variables", ["rho", "u", "P", "T"])
        
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

