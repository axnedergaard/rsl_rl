from torch import Tensor, FloatTensor
import torch

class Geometry():
    """
    A base class representing a geometric structure on a space.
    This class provides a generic interface for computing geometric properties such as distance and interpolation
    and can be extended by specific geometric models.
    """
   
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, x: Tensor, y: Tensor) -> FloatTensor:
        return self.distance_function(x, y)

    @torch.no_grad()
    def broadcast_distance_function(self, x: Tensor, y: Tensor) -> FloatTensor:
      """
      A broadcast version of distance_function.
      Args:
          x (torch.Tensor): First Tensor. (size_x, dim) 
          y (torch.Tensor): Second Tensor. (size_y, dim)
      Returns:
          FloatTensor: Pairwise distances. (size_x, size_y)
      """
      # Note: Can be made more efficient when x is y.
      bx, by = torch.broadcast_tensors(x, y)
      batched_distance_function = torch.vmap(self.distance_function)
      distances = batched_distance_function(bx, by)
      return distances

    def distance_function(self, x: Tensor, y: Tensor) -> FloatTensor:
        """
        Compute the distance between states x and y in the geometric space.
        Args:
            x (torch.Tensor): First Tensor. (dim) 
            y (torch.Tensor): Second Tensor. (dim)
        Returns:
            FloatTensor: (B,) Pairwise distance. (1)
        """
        raise NotImplementedError()

    def interpolate(self, x: Tensor, y: Tensor, alpha: float) -> Tensor:
        """
        Interpolate between states x and y, using a specified weight.
        Args:
            x (torch.Tensor): Point from which to start interpolation.
            y (torch.Tensor): Point towards which x is drifting.
            alpha (float): Interpolation weight of y. Typically in the range [0, 1].
        Returns: torch.Tensor: Interpolated state between x and y.
        """
        raise NotImplementedError()
    
