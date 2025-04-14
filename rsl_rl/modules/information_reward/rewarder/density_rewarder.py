from rum.rewarder.rewarder import Rewarder
from rum.density import Density 
from torch import Tensor, FloatTensor
import torch

class DensityRewarder(Rewarder):
  def __init__(
      self,
      density: Density,
      form: str, 
      scaling: float = 1.0,
      concurrent: bool = True, # SB3
    ) -> None:
      super(DensityRewarder, self).__init__(scaling, concurrent)
      self.density = density
      if form == 'information':
        self._reward_function = self._reward_information
      elif form == 'entropy':
        self._reward_function = self._reward_entropy
      elif form == 'change_entropy':
        self._reward_function = self._reward_change_entropy
      else:
        raise ValueError("Form must be 'information', 'entropy' or 'change_entropy'.")
      
  def reward_function(self, states: Tensor) -> FloatTensor:
    if not isinstance(states, Tensor) or states.dim() != 2:
      raise ValueError("States must be of shape (num_states, dim_states)")
    rewards = torch.zeros(states.size(0))  # shape: (num_states,)
    if not self.density.ready: # Density estimator may need to learn before computing rewards.
      return rewards
    for i, state in enumerate(states):
        rewards[i] = self._reward_function(state)
    return rewards  # shape: (num_states,)

  def _reward_entropy(self, state: Tensor) -> FloatTensor:
    distances = self.density.simulate_step(state)
    return self.density.compute_entropy(distances)

  def _reward_change_entropy(self, state: Tensor) -> FloatTensor:
    return self._reward_entropy(state) - self.density.entropy()

  def _reward_information(self, state: Tensor) -> FloatTensor:
    return self.density.information(state)

  def learn(self, states: Tensor) -> FloatTensor:
    if not isinstance(states, Tensor) or states.dim() != 2:
        raise ValueError("States must be of shape (num_states, dim_states)")
    self.density.learn(states)
