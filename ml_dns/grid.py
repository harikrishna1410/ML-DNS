import numpy as np
import torch
from .params import SimulationParameters
class Grid:
    def __init__(self, params:SimulationParameters):
        self.ndim = params.ndim
        self.ng = params.ng
        self.nl = params.nl
        self.sim_params = params
        ##
        self._x, self._dl_dx = self.stretch_coordinates(
            self.ng[0], params.grid_stretching_params
        )  # Stretching in x-direction
        self._x = (self.sim_params.domain_extents['xs'] + \
                    self._x*(self.sim_params.domain_extents['xe'] - \
                             self.sim_params.domain_extents['xs']))
        if(self.ndim > 1):
            self._y, self._dl_dy = self.stretch_coordinates(
                self.ng[1], params.grid_stretching_params
            )  # Stretching in y-direction
            self._y = self.sim_params.domain_extents['ys'] + \
                    self._y*(self.sim_params.domain_extents['ye'] - \
                             self.sim_params.domain_extents['ys'])
        if(self.ndim > 2):
            self._z, self._dl_dz = self.stretch_coordinates(
                self.ng[2], params.grid_stretching_params
            )  # Stretching in z-direction
            self._z = self.sim_params.domain_extents['zs'] + \
                    self._z*(self.sim_params.domain_extents['ze'] - \
                             self.sim_params.domain_extents['zs'])
        ##
        # Non-dimensionalize x, y, z coordinates and their derivatives
        for dim,dir in enumerate(["x","y","z"][:self.ndim]):
            setattr(self, f"_{dir}", getattr(self, f"_{dir}") / self.sim_params.l_ref)
            setattr(self, f"_dl_d{dir}", getattr(self, f"_dl_d{dir}") * self.sim_params.l_ref)
            
        for i, coord in enumerate(['x', 'y', 'z'][:self.ndim]):
            start = params.my_idx[i] * self.nl[i]
            end = None if self.sim_params.my_idx[i] == self.sim_params.np[i] - 1 \
                        else (self.sim_params.my_idx[i] + 1) * self.nl[i]
            setattr(self, f'_{coord}l', getattr(self, f'_{coord}')[start:end])
            setattr(self, f'_dl_d{coord}l', getattr(self, f'_dl_d{coord}')[start:end])

    
    def xl(self, i=None):
        if i is None:
            return tuple([getattr(self, f'_{dir}l') for dir in ['x', 'y', 'z'][:self.ndim]])
        return getattr(self, f'_{["x", "y", "z"][i]}l')

    def dx(self, i=None):
        if i is None:
            return tuple([np.diff(getattr(self, f'_{dir}l')) for dir in ['x', 'y', 'z'][:self.ndim]])
        return np.diff(getattr(self, f'_{["x", "y", "z"][i]}l'))

    def dl_dx(self, i=None):
        if i is None:
            return tuple([getattr(self, f'_dl_d{dir}l') for dir in ['x', 'y', 'z'][:self.ndim]])
        return getattr(self, f'_dl_d{["x", "y", "z"][i]}l')

    def xg(self, i=None):
        if i is None:
            return tuple([getattr(self, f'_{dir}') for dir in ['x', 'y', 'z'][:self.ndim]])
        return getattr(self, f'_{["x", "y", "z"][i]}')

    def dx_g(self, i=None):
        if i is None:
            return tuple([np.diff(getattr(self, f'_{dir}')) for dir in ['x', 'y', 'z'][:self.ndim]])
        return np.diff(getattr(self, f'_{["x", "y", "z"][i]}'))

    def dl_dx_g(self, i=None):
        if i is None:
            return tuple([getattr(self, f'_dl_d{dir}') for dir in ['x', 'y', 'z'][:self.ndim]])
        return getattr(self, f'_dl_d{["x", "y", "z"][i]}')
    ##
    def stretch_coordinates(self, n, stretching_params):
        """
        Generate a uniform grid for now.
        
        Args:
        n (int): Number of grid points
        stretching_params (tuple): Ignored for now, but kept for future implementation
        
        Returns:
        tuple: (coordinates, grid spacing)
        """
        coordinates = torch.linspace(0, 1, n)
        dl_dx = torch.ones(n)  # Uniform grid spacing
        return coordinates, dl_dx
