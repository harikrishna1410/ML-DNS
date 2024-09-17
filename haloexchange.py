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
        self.my_pidx = params.my_idx
        self.halo_depth = self.data.halo_depth
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

    def _Iexchange_dim(self, f: torch.Tensor, dim: int):
        if f.dim() == 3:
            nv = 1
            f = f.unsqueeze(0)
        elif f.dim() == 4:
            nv = f.size(0)
        else:
            raise ValueError(f"Unsupported tensor dimension: {f.dim()}. Expected 3 or 4.")

        left_neighbor = self.neighbors[f'left_{dim}']
        right_neighbor = self.neighbors[f'right_{dim}']

        # Determine which halos to use based on dimension
        if dim == 0:
            left_halo, right_halo = self.data.halo_xl[:nv], self.data.halo_xr[:nv]
            send_right = f[:,-self.halo_depth:,:,:]
            send_left = f[:,:self.halo_depth,:,:]
        elif dim == 1:
            left_halo, right_halo = self.data.halo_yl[:nv], self.data.halo_yr[:nv]
            send_right = f[:,:,-self.halo_depth:,:]
            send_left = f[:,:,:self.halo_depth,:]
        else:  # dim == 2
            left_halo, right_halo = self.data.halo_zl[:nv], self.data.halo_zr[:nv]
            send_right = f[:,:,:,-self.halo_depth:]
            send_left = f[:,:,:,:self.halo_depth]

        requests = []
        # Send to right, receive from left
        if right_neighbor != -1:
            req = self.comm.Isend(send_right, dest=right_neighbor)
            requests.append(req)
        if left_neighbor != -1:
            req = self.comm.Irecv(left_halo, source=left_neighbor)
            requests.append(req)

        # Send to left, receive from right
        if left_neighbor != -1:
            req = self.comm.Isend(send_left, dest=left_neighbor)
            requests.append(req)
        if right_neighbor != -1:
            req = self.comm.Irecv(right_halo, source=right_neighbor)
            requests.append(req)

        return requests

    def Iexchange(self, f: torch.Tensor, dim=None):
        if dim is not None:
            return self._Iexchange_dim(f, dim)
        else:
            all_requests = []
            for d in range(self.ndim):
                all_requests.extend(self._Iexchange_dim(f, d))
            return all_requests
    
    def wait_dim(self,requests,dim):
        MPI.Request.Waitall(requests[dim*4 : (dim+1)*4])