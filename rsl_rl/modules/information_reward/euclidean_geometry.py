from .geometry import Geometry
from torch import Tensor, FloatTensor
import torch

class EuclideanGeometry(Geometry):

    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    @torch.no_grad()
    def broadcast_distance_function(self, x, y):
      distances = torch.cdist(x, y)
      if distances.dim() == 1:
        return distances.unsqueeze(0)
      else:
        return distances

    @torch.no_grad()
    def distance_function(self, x: Tensor, y: Tensor) -> FloatTensor:
        return torch.norm(x - y, p=2)

    @torch.no_grad()
    def interpolate(self, x: Tensor, y: Tensor, alpha: float) -> Tensor:
        if x.shape != (self.dim,) or y.shape != (self.dim,):
            raise ValueError("Tensors must lie in ambient space")
        return (1 - alpha) * x + alpha * y

    @torch.no_grad()
    def learn(self, states: Tensor = None) -> FloatTensor:
        pass # No learning is required.
