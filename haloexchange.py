import torch
from data import SimulationData
from mpi4py import MPI
from params import SimulationParameters

class HaloExchange:
    def __init__(self, params: SimulationParameters , data: SimulationData, comm):
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
            left_halo_r, right_halo_r = self.data.halo_xlr[:nv], self.data.halo_xrr[:nv]
            left_halo_s, right_halo_s = self.data.halo_xls[:nv], self.data.halo_xrs[:nv]
            right_halo_s[:,:,:,:] = f[:,-self.halo_depth:,:,:]
            left_halo_s[:,:,:,:] = f[:,:self.halo_depth,:,:]
        elif dim == 1:
            left_halo_r, right_halo_r = self.data.halo_ylr[:nv], self.data.halo_yrr[:nv]
            left_halo_s, right_halo_s = self.data.halo_yls[:nv], self.data.halo_yrs[:nv]
            right_halo_s[:,:,:,:] = f[:,:,-self.halo_depth:,:].permute(0,2,1,3)
            left_halo_s[:,:,:,:] = f[:,:,:self.halo_depth,:].permute(0,2,1,3)
        else:  # dim == 2
            left_halo_r, right_halo_r = self.data.halo_zlr[:nv], self.data.halo_zrr[:nv]
            left_halo_s, right_halo_s = self.data.halo_zls[:nv], self.data.halo_zrs[:nv]
            right_halo_s[:,:,:,:] = f[:,:,:,-self.halo_depth:].permute(0,3,1,2)
            left_halo_s[:,:,:,:] = f[:,:,:,:self.halo_depth].permute(0,3,1,2)

        requests = []
        # Send to right, receive from left
        if right_neighbor != -1:
            req = self.comm.Isend(right_halo_s.numpy(), dest=right_neighbor)
            requests.append(req)
        if left_neighbor != -1:
            req = self.comm.Irecv(left_halo_r.numpy(), source=left_neighbor)
            requests.append(req)

        # Send to left, receive from right
        if left_neighbor != -1:
            req = self.comm.Isend(left_halo_s.numpy(), dest=left_neighbor)
            requests.append(req)
        if right_neighbor != -1:
            req = self.comm.Irecv(right_halo_r.numpy(),source=right_neighbor)
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
        MPI.Request.Waitall(requests[dim*4:dim*4+4])
