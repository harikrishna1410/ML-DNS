import torch
from params import SimulationParameters

class SimulationData:
    def __init__(self, params):
        self.nvars = params.nvars
        # Create tensors for primitives and solution
        self.primitives = torch.zeros((self.nvars,) + tuple(params.nl))
        self.soln = torch.zeros((self.nvars,) + tuple(params.nl))
        # Calculate halo depth based on the difference order
        halo_depth = params.diff_order // 2
        # Initialize halos for all 6 directions (2 per axis)
        self.halos = {
            'x': torch.zeros((2, self.nvars, params.nl[1], params.nl[2], halo_depth)),  # x-direction halos
            'y': torch.zeros((2, self.nvars, params.nl[0], params.nl[2], halo_depth)),  # y-direction halos
            'z': torch.zeros((2, self.nvars, params.nl[0], params.nl[1], halo_depth))   # z-direction halos
        }
        
        # Initialize properties as variables
        self.rho = self.primitives[0]
        self.u = self.primitives[1]
        self.v = self.primitives[2]
        self.w = self.primitives[3]
        self.T = self.primitives[4]
        
        self.halo_xl = self.halos['x'][0]
        self.halo_xr = self.halos['x'][1]
        self.halo_yl = self.halos['y'][0]
        self.halo_yr = self.halos['y'][1]
        self.halo_zl = self.halos['z'][0]
        self.halo_zr = self.halos['z'][1]

    def Ys(self, id):
        return self.primitives[5 + id]
    
    def get_solution(self):
        return self.soln
    
    def zero_halos(self):
        """
        Set all halo values to zero.
        """
        for direction in self.halos:
            self.halos[direction].zero_()
