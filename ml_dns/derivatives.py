import torch
import torch.nn.functional as F
from .haloexchange import HaloExchange
from .grid import Grid
from .data import SimulationData

class Derivatives:
    def __init__(self, grid : Grid, halo_exchange: HaloExchange,sim_data : SimulationData):
        self.grid = grid
        self.halo_exchange = halo_exchange
        self.sim_data = sim_data

    def central_difference(self, tensor, axis, left_padding, right_padding, stencil=None):
        """
        Compute central difference along the specified axis for a 3D tensor.
        
        This method uses a central difference stencil to compute gradients.
        By default, it uses an 8th order stencil with coefficients:
        [-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280]
        
        Args:
        tensor (torch.Tensor): Input 3D tensor
        axis (int): Axis along which to compute the gradient (0, 1, or 2)
        left_padding (torch.Tensor): 3D tensor of padding values to be stacked before the input tensor
        right_padding (torch.Tensor): 3D tensor of padding values to be stacked after the input tensor
        stencil (torch.Tensor, optional): Custom stencil coefficients. Defaults to 8th order stencil.
        
        Returns:
        torch.Tensor: Gradient of the input tensor along the specified axis
        """
        if stencil is None:
            stencil = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280])
        
        # Stack left and right padding along the input axis
        ##Note that left and right padding alwlays have depth in first dimension
        if(axis == 1):
            left_padding = left_padding.permute(1,0,2)
            right_padding = right_padding.permute(1,0,2)
        elif(axis == 2):
            left_padding = left_padding.permute(1,2,0)
            right_padding = right_padding.permute(1,2,0)
        padded_tensor = torch.cat([left_padding, 
                                   tensor, 
                                   right_padding], dim=axis)
        
        if axis == 0:
            kernel = stencil.view(-1, 1, 1)
        elif axis == 1:
            kernel = stencil.view(1, -1, 1)
        else:  # axis == 2
            kernel = stencil.view(1, 1, -1)
        ##
        dl_dx_shape = [1,1,1]
        dl_dx_shape[axis] = -1
        return F.conv3d(padded_tensor.unsqueeze(0).unsqueeze(0), 
                        kernel.unsqueeze(0).unsqueeze(0),
                        padding=0).squeeze(0).squeeze(0)*self.grid.dl_dx(axis).view(dl_dx_shape)
    ##
    def central_difference_multi(self, tensor, axis, left_padding, right_padding, stencil=None):
        """
        Compute central difference along the specified axis for multiple 3D tensors.
        
        Args:
        tensor (torch.Tensor): Input 4D tensor of shape (num_variables, *spatial_dims)
        axis (int): Axis along which to compute the gradient (0, 1, or 2 corresponding to x, y, z)
        left_padding (torch.Tensor): 4D tensor of padding values to be stacked before the input tensor
        right_padding (torch.Tensor): 4D tensor of padding values to be stacked after the input tensor
        stencil (torch.Tensor, optional): Custom stencil coefficients. Defaults to 8th order stencil.
        
        Returns:
        torch.Tensor: Gradients of the input tensor along the specified axis
        """
        if stencil is None:
            stencil = torch.tensor([-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280])
        
        # Adjust kernel shape for 4D input
        kernel_shape = [1, 1, 1, 1, 1]
        kernel_shape[axis+2] = -1
        kernel = stencil.view(*kernel_shape).repeat(tensor.size(0),1,1,1,1)
        
        if(axis == 1):
            left_padding = left_padding.permute(0,2,1,3)
            right_padding = right_padding.permute(0,2,1,3)
        elif(axis == 2):
            left_padding = left_padding.permute(0,2,3,1)
            right_padding = right_padding.permute(0,2,3,1)
        # Stack left and right padding along the input axis
        padded_tensor = torch.cat([left_padding, 
                                   tensor, 
                                   right_padding], dim=axis+1)
        
        # Perform convolution
        result = F.conv3d(padded_tensor,
                          kernel,
                          padding=0,
                          groups=tensor.size(0))  # Use groups for parallel computation
        dl_dx_shape = [1,1,1]
        dl_dx_shape[axis] = -1
        return result*self.grid.dl_dx(axis).view(dl_dx_shape)

    def divergence(self, tensor, stencil=None):
        """
        Compute the divergence of a 3D vector field using central_difference.
        
        This method uses the central_difference function to compute the divergence.
        
        Args:
        tensor (torch.Tensor): Input tensor of shape (3, *spatial_dims) or (3, num_variables, *spatial_dims)
        stencil (torch.Tensor, optional): Custom stencil coefficients. Defaults to None.
        
        Returns:
        torch.Tensor: Divergence of the input tensor
        """
        
        ##
        if tensor.dim() == 5:
            nv = tensor.size(1)
            central_diff = self.central_difference_multi
        elif tensor.dim() == 4:
            nv = 1
            central_diff = self.central_difference
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Expected 4 or 5.")
        ##
        ret = []
        for dim in range(self.grid.ndim):
            req=self.halo_exchange.Iexchange(tensor[dim],dim=dim)
            self.halo_exchange.wait_dim(requests=req,dim=0)
            df = central_diff(tensor[dim], 
                              axis=dim,
                              left_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][0][:nv].squeeze(0),
                              right_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][1][:nv].squeeze(0),
                              stencil=stencil)
            ret.append(df)
        self.sim_data.zero_halos()
        return sum(ret)
    
    def gradient(self, tensor, stencil=None,dim=None):
        """
        Compute the gradient of a 3D tensor using central_difference.
        
        This method is a wrapper around central_difference to provide the gradient as a tuple.
        
        Args:
        tensor (torch.Tensor): Input 3D tensor
        left_padding (torch.Tensor): 3D tensor of padding values to be stacked before the input tensor
        right_padding (torch.Tensor): 3D tensor of padding values to be stacked after the input tensor
        stencil (torch.Tensor, optional): Custom stencil coefficients. Defaults to None.
        
        Returns:
        tuple of torch.Tensor: Gradient of the input tensor along all axes
        """
        if tensor.dim() == 4:
            nv = tensor.size(0)
            central_diff = self.central_difference_multi
        elif tensor.dim() == 3:
            nv = 1
            central_diff = self.central_difference
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Expected 3 or 4.")
        req = self.halo_exchange.Iexchange(tensor,dim=dim)
        if(dim):    
            ##req is just a 4 element list
            self.halo_exchange.wait_dim(requests=req,dim=0)
            df = central_diff(tensor, 
                            axis=dim,
                            left_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][0][:nv].squeeze(0),
                            right_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][1][:nv].squeeze(0),
                            stencil=stencil)
            self.sim_data.zero_halos()
            return df
        else:
            ret = []
            for dim in range(self.grid.ndim):
                #Here req is 12 element list
                self.halo_exchange.wait_dim(requests=req,dim=dim)
                df = central_diff(tensor, 
                                axis=dim,
                                left_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][0][:nv].squeeze(0),
                                right_padding=self.sim_data.halos[['x', 'y', 'z'][dim]][1][:nv].squeeze(0),
                                stencil=stencil)
                ret.append(df)
        self.sim_data.zero_halos()
        return tuple(ret)
