import numpy as np
class Grid:
    def __init__(self, params,my_pidx):
        self.ndim = params.ndim
        self.ng = params.ng
        self.nl = tuple([
            self.ng[i] // params.np[i] + 
            (self.ng[i] % params.np[i] if my_pidx[i] == params.np[i] - 1 else 0)
            for i in range(3)
        ])  # Compute local grid points
        ##
        self._x, self._dl_dx = self.stretch_coordinates(
            self.ng[0], params.grid_stretching_params[0]
        )  # Stretching in x-direction
        if(self.ndim > 1):
            self._y, self._dl_dy = self.stretch_coordinates(
                self.ng[1], params.grid_stretching_params[1]
            )  # Stretching in y-direction
        if(self.ndim > 2):
            self._z, self._dl_dz = self.stretch_coordinates(
                self.ng[2], params.grid_stretching_params[2]
            )  # Stretching in z-direction
        ##
        for i, coord in enumerate(['x', 'y', 'z'][:self.ndim]):
            start = my_pidx[i] * self.nl[i]
            end = None if my_pidx[i] == params.np[i] - 1 else (my_pidx[i] + 1) * self.nl[i]
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
        coordinates = np.linspace(stretching_params['start'], stretching_params['end'], n)
        dl_dx = np.ones(n)  # Uniform grid spacing
        return coordinates, dl_dx
