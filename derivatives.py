import torch
import torch.nn.functional as F

class Derivatives:
    def __init__(self, grid):
        self.grid = grid

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
        
        stencil = stencil * self.grid.dl_dx(axis)
        
        # Stack left and right padding along the input axis
        padded_tensor = torch.cat([left_padding, tensor, right_padding], dim=axis)
        
        if axis == 0:
            kernel = stencil.view(-1, 1, 1)
        elif axis == 1:
            kernel = stencil.view(1, -1, 1)
        else:  # axis == 2
            kernel = stencil.view(1, 1, -1)
        
        return F.conv3d(padded_tensor.unsqueeze(0).unsqueeze(0), 
                        kernel.unsqueeze(0).unsqueeze(0),
                        padding=0).squeeze(0).squeeze(0)

    def divergence(self, tensor, left_padding, right_padding, stencil=None):
        """
        Compute the divergence of a 3D vector field using central_difference.
        
        This method uses the central_difference function to compute the divergence.
        
        Args:
        tensor (torch.Tensor): Input tensor of shape (3, *spatial_dims)
        left_padding (torch.Tensor): 3D tensor of padding values to be stacked before the input tensor
        right_padding (torch.Tensor): 3D tensor of padding values to be stacked after the input tensor
        stencil (torch.Tensor, optional): Custom stencil coefficients. Defaults to None.
        
        Returns:
        torch.Tensor: Divergence of the input tensor
        """
        # Compute partial derivatives using central_difference
        dx = self.central_difference(tensor, axis=0, left_padding=left_padding, right_padding=right_padding, stencil=stencil)
        dy = self.central_difference(tensor, axis=1, left_padding=left_padding, right_padding=right_padding, stencil=stencil)
        dz = self.central_difference(tensor, axis=2, left_padding=left_padding, right_padding=right_padding, stencil=stencil)

        # Sum the partial derivatives to get the divergence
        return dx + dy + dz
    
    def gradient(self, tensor, left_padding=None, right_padding=None, stencil=None):
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
        return tuple(self.central_difference(tensor, axis=i, left_padding=left_padding, right_padding=right_padding, stencil=stencil) for i in range(3))