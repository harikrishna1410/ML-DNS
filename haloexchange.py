import torch
from mpi4py import MPI
from data import SimulationData
from params import SimulationParameters

class HaloExchange:
    def __init__(self, params: SimulationParameters , data: SimulationData, comm: MPI.Comm):
        self.data = data
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.sim_params = params
        self.ndim = params.ndim
        self.np = params.np
        self.my_pidx = tuple(torch.unravel_index(self.rank, params.np))
        self.neighbors = self._get_neighbors()

    def _get_neighbors(self):
        neighbors = {}
        diff = 1
        for dim in range(self.ndim):
            neighbors[f'left_{dim}'] = self.rank - diff
            neighbors[f'right_{dim}'] = self.rank + diff
            if(self.my_pidx[dim]==self.np[dim]-1):
                if(self.sim_params.periodic_bc[dim]):
                    neighbors[f'right_{dim}'] = self.rank - diff*(self.np[dim]-1)
                else:
                    neighbors[f'right_{dim}'] = -1
            if(self.my_pidx[dim]==0):
                if(self.sim_params.periodic_bc[dim]):
                    neighbors[f'left_{dim}'] = self.rank + diff*(self.np[dim]-1)
                else:
                    neighbors[f'left_{dim}'] = -1
            diff = diff*self.np[dim]
        return neighbors

    def Iexchange(self,f):
        if f.dim() == 3:
            nv = 1
        elif f.dim() == 4:
            nv = f.size(0)
        else:
            raise ValueError(f"Unsupported tensor dimension: {f.dim()}. Expected 3 or 4.")
        requests = []
        for dim in range(self.ndim):
            left_neighbor = self.neighbors[f'left_{dim}']
            right_neighbor = self.neighbors[f'right_{dim}']

            # Determine which halos to use based on dimension
            if dim == 0:
                left_halo, right_halo = self.data.halo_xl[:nv], self.data.halo_xr[:nv]
            elif dim == 1:
                left_halo, right_halo = self.data.halo_yl[:nv], self.data.halo_yr[:nv]
            else:  # dim == 2
                left_halo, right_halo = self.data.halo_zl[:nv], self.data.halo_zr[:nv]

            # Send to right, receive from left
            send_right = f.select(dim+1, slice(-left_halo.size(dim+1), None))
            if right_neighbor != -1:
                req = self.comm.Isend(send_right, dest=right_neighbor)
                requests.append(req)
            if left_neighbor != -1:
                req = self.comm.Irecv(left_halo, source=left_neighbor)
                requests.append(req)

            # Send to left, receive from right
            send_left = f.select(dim+1, slice(0, right_halo.size(dim+1)))
            if left_neighbor != -1:
                req = self.comm.Isend(send_left, dest=left_neighbor)
                requests.append(req)
            if right_neighbor != -1:
                req = self.comm.Irecv(right_halo, source=right_neighbor)
                requests.append(req)
        return requests
    
    def wait_dim(self,requests,dim):
        MPI.Request.Waitall(requests[dim*4 : (dim+1)*4])