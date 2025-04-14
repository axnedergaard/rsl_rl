import torch
from torch import Tensor

from .information_geometry import InformationGeometry

class AlphaGeometry(InformationGeometry):
    def __init__(self, alpha: float, epsilon: float = 0.0):
      self.alpha = torch.tensor(alpha, requires_grad=True)
      # self.alpha.requires_grad = True
      self.epsilon = torch.tensor(epsilon) # Constant.

    def information_function(self, rec_p_x: Tensor) -> Tensor:
      # The proper form is - f_alpha(u / p(x)) + c(f), where c is a constant 
      # and u is the uniform distribution. We negate (to get entropy) and 
      # ignore c and u, since c vanishes in the derivative and u is absorbed by
      # the learning rate. We similarly ignore the constant term in the
      # alpha != +/ 1 case, since it vanishes in the derivative.
      # The argument rec_p_x represents 1 / p(x). We use this form for 
      # efficient compatibility with the KM and KNN density estimators.
      # We optionally add an epsilon for numerical stability.
      if self.alpha == 1.0:
        return - (rec_p_x + self.epsilon) * torch.log(rec_p_x + self.epsilon)
      elif self.alpha == -1.0: # Shannon.
        return torch.log(rec_p_x + self.epsilon)
      else:
        return (4.0 / (1 - self.alpha ** 2)) * (rec_p_x + self.epsilon) ** ((1.0 + self.alpha) / 2.0)
