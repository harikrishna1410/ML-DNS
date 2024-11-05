import torch
from ..core import SimulationData,SimulationParameters
from mpi4py import MPI

class HaloExchange:
    def __init__(self, params: SimulationParameters , comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.sim_params = params
        self.ndim = params.ndim
        self.np = params.np
        self.my_pidx = params.my_idx
        ##assuming central difference
        self.halo_depth = params.diff_order // 2
        # Initialize halos for all 6 directions (2 per axis)
        self.halos = {}
        directions = ['x', 'y', 'z']
        for i, direction in enumerate(directions[:self.sim_params.ndim]):
            halo_shape = [4, self.sim_params.nvars, self.halo_depth] + [params.nl[j] for j in range(len(directions)) if j != i]
            self.halos[direction] = torch.zeros(halo_shape)
            setattr(self, f'halo_{direction}lr', self.halos[direction][0])
            setattr(self, f'halo_{direction}rr', self.halos[direction][1])
            setattr(self, f'halo_{direction}ls', self.halos[direction][2])
            setattr(self, f'halo_{direction}rs', self.halos[direction][3])
        self.neighbors = self._get_neighbors()


    def zero_halos(self):
        for direction in self.halos:
            self.halos[direction].zero_()

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
            left_halo_r, right_halo_r = self.halo_xlr[:nv], self.halo_xrr[:nv]
            left_halo_s, right_halo_s = self.halo_xls[:nv], self.halo_xrs[:nv]
            right_halo_s[:,:,:,:] = f[:,-self.halo_depth:,:,:]
            left_halo_s[:,:,:,:] = f[:,:self.halo_depth,:,:]
        elif dim == 1:
            left_halo_r, right_halo_r = self.halo_ylr[:nv], self.halo_yrr[:nv]
            left_halo_s, right_halo_s = self.halo_yls[:nv], self.halo_yrs[:nv]
            right_halo_s[:,:,:,:] = f[:,:,-self.halo_depth:,:].permute(0,2,1,3)
            left_halo_s[:,:,:,:] = f[:,:,:self.halo_depth,:].permute(0,2,1,3)
        else:  # dim == 2
            left_halo_r, right_halo_r = self.halo_zlr[:nv], self.halo_zrr[:nv]
            left_halo_s, right_halo_s = self.halo_zls[:nv], self.halo_zrs[:nv]
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
