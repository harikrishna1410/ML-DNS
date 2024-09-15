import json
import numpy as np

class SimulationParameters:
    def __init__(self, json_file, topo, my_idx):
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        # Grid parameters
        self.ng = (params.get("nxg", 1), params.get("nyg", 1), params.get("nzg", 1))
        self.ndim = params.get("ndim", 3)
        self.np = (params.get("npx", 1), params.get("npy", 1), params.get("npz", 1))
        
        # Domain parameters
        self.grid_stretching_params = [
            {'start': params.get('xs', 0), 'end': params.get('xe', 1)},
            {'start': params.get('ys', 0), 'end': params.get('ye', 1)},
            {'start': params.get('zs', 0), 'end': params.get('ze', 1)}
        ]
        
        #periodic bc
        self.periodic_bc = (params.get("periodic_bc", [True, True, True]))
        
        # Time stepping parameters
        self.dt = params.get("dt", 0.001)
        self.num_steps = params.get("num_steps", 1000)
        
        # Physical parameters
        self.num_species = params.get("num_species", 1)
        if(self.num_species>0):
            raise ValueError("Num species > 0")
        self.gamma = params.get("gamma", 1.4)  # Specific heat ratio
        
        # Numerical method parameters
        self.diff_order = params.get("diff_order", 8)  # Order of difference scheme
        
        # Parallel computing parameters
        self.topo = topo
        self.my_idx = my_idx
        
        # Buoyancy parameters
        self.use_buoyancy = params.get("use_buoyancy", False)
        self.rho_ref = params.get("rho_ref", 1.0)  # Reference density
        self.beta = params.get("beta", 0.0)  # Thermal expansion coefficient
        self.T_ref = params.get("T_ref", 300.0)  # Reference temperature
        
        # Compute local grid points
        self.nl = tuple([
            self.ng[i] // self.np[i] + 
            (self.ng[i] % self.np[i] if my_idx[i] == self.np[i] - 1 else 0)
            for i in range(3)
        ])
        
        # Solver options
        self.use_nn = params.get("use_nn", False)
        self.advection_method = params.get("advection_method", "compressible")
        
        
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

