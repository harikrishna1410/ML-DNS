import h5py
import os
import numpy as np
import torch
from .params import SimulationParameters
from .data import SimulationData

class IO:
    def __init__(self, params: SimulationParameters, data: SimulationData):
        self.params = params
        self.data = data
        self.state = data.state

    def _get_filename(self):
        return f"{self.params.output_dir}/time_{self.state.time*self.params.time_ref:13.7e}/{self.params.my_rank:d}.h5"

    def write(self):
        filename = self._get_filename()
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('time', data=self.state.time*self.params.time_ref)
            f.create_dataset('dt', data=self.params.dt*self.params.time_ref)
            f.create_dataset('rho', data=self.state.primitives[0].cpu().numpy()*self.params.rho_ref)
            f.create_dataset('u', data=self.state.primitives[1:self.params.ndim+1].cpu().numpy()*self.params.a_ref)
            f.create_dataset('T', data=self.state.primitives[self.params.ndim+1].cpu().numpy()*self.params.T_ref)
            f.create_dataset('E', data=self.state.primitives[-2].cpu().numpy()*self.params.a_ref**2)
            f.create_dataset('P', data=self.state.primitives[-1].cpu().numpy()*self.params.P_ref)
            f.create_dataset('nl', data=self.params.nl)

    def read(self):
        filename = self._get_filename()

        with h5py.File(filename, 'r') as f:
            self.state.time = f['time'][()] / self.params.time_ref
            self.params.dt = f['dt'][()] / self.params.time_ref
            self.state.primitives[0] = torch.from_numpy(f['rho'][:]) / self.params.rho_ref
            self.state.primitives[1:self.params.ndim+1] = torch.from_numpy(f['u'][:]) / self.params.a_ref
            self.state.primitives[self.params.ndim+1] = torch.from_numpy(f['T'][:]) / self.params.T_ref
            self.state.primitives[-2] = torch.from_numpy(f['E'][:]) / self.params.a_ref**2
            self.state.primitives[-1] = torch.from_numpy(f['P'][:]) / self.params.P_ref
            self.params.nl = tuple(f['nl'][()])

        # After reading primitives, compute the solution
        self.state.compute_soln_from_primitives()

    def write_grid(self):
        filename = f"{self.params.output_dir}/grid.h5"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with h5py.File(filename, 'w') as f:
            # Write global grid dimensions
            f.create_dataset('ng', data=self.params.ng)
            f.create_dataset("ndim",self.params.ndim)
            
            # Write domain extents
            f.create_dataset('xs', data=self.params.domain_extents['xs'] * self.params.l_ref)
            f.create_dataset('xe', data=self.params.domain_extents['xe'] * self.params.l_ref)
            f.create_dataset('ys', data=self.params.domain_extents['ys'] * self.params.l_ref)
            f.create_dataset('ye', data=self.params.domain_extents['ye'] * self.params.l_ref)
            f.create_dataset('zs', data=self.params.domain_extents['zs'] * self.params.l_ref)
            f.create_dataset('ze', data=self.params.domain_extents['ze'] * self.params.l_ref)
            
            # Write global grid coordinates
            for i,x in enumerate(["x","y","z"][:self.params.ndim]):
                f.create_dataset(x, data=self.data.grid.xg(i).cpu().numpy() * self.params.l_ref)

        print(f"Global grid written to {filename}")