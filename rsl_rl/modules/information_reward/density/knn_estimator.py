import torch
from torch import Tensor

from rum.density import Density
from rum.geometry import Geometry
from rum.information_geometry import InformationGeometry

# Only entropy-related computations need gradients (for meta learning
# information geometry).

class KNNDensityEstimator(Density):
    def __init__(
      self, 
      k: int, 
      dim: int, 
      buffer_max_size: int, 
      geometry: Geometry,
      information_geometry: InformationGeometry,
      rolling_average: bool = True,
    ):
      self.k = k
      self.dim = dim
      self.buffer_max_size = buffer_max_size
      self.buffer = torch.zeros((buffer_max_size, dim))
      self.buffer_size = 0
      self.rolling_average = rolling_average 
      self.geometry = geometry
      self.information_geometry = information_geometry
      self.ready = False

    @torch.no_grad()
    def learn(self, states: Tensor) -> None:
      # Update the buffer state with incoming states.
      if self.rolling_average:
        self.update_buffer(states, inplace=True)
        if not self.ready and self.buffer_size >= self.k:
          self.ready = True
      else:  # In this case, we reset the buffer every time.
        assert not states.size(0) < self.k, 'Not enough states to not use rolling average.'
        self.buffer = torch.zeros((self.buffer_max_size, self.dim))
        self.buffer_size = 0
        self.update_buffer(states, inplace=True)
        self.ready = True

    @torch.no_grad()
    def simulate_step(self, state: Tensor) -> Tensor:
      # Compute distances between buffer states after adding a state, without 
      # updating the buffer.
      states = torch.unsqueeze(state, 0)
      buffer, buffer_size = self.update_buffer(states, inplace=False)
      return self.compute_distances(buffer[:buffer_size], buffer[:buffer_size]) # shape: (buffer_size, buffer_size)

    @torch.no_grad()
    def pdf(self, x: Tensor) -> float:
      average_distances = self.compute_average_distances_from_point(x)
      return 1.0 / average_distances

    def information(self, x: Tensor) -> float:
      average_distances = self.compute_average_distances_from_point(x)
      return self.information_geometry.information_function(average_distances)

    def entropy(self) -> Tensor:
      distances = self.compute_distances(
        self.buffer[:self.buffer_size], 
        self.buffer[:self.buffer_size]
      )
      return self.compute_entropy(distances)

    def compute_entropy(self, distances: Tensor) -> Tensor:
      average_distances = self.compute_average_distances(distances)
      informations = self.information_geometry.information_function(average_distances)
      return torch.mean(informations)

    @torch.no_grad()
    def compute_average_distances_from_point(self, x: Tensor) -> Tensor:
      distances = self.compute_distances(
        x.unsqueeze(0), 
        self.buffer[:self.buffer_size]
      ).view(-1) # shape: (buffer_size,)
      return self.compute_average_distances(distances)

    @torch.no_grad()
    def compute_average_distances(self, distances: Tensor) -> Tensor:
      dim_to_reduce = 1 if len(distances.shape) > 1 else 0
      return (1.0 / self.k) * torch.sum(distances, dim=dim_to_reduce)

    @torch.no_grad()
    def update_buffer(self, states: Tensor, inplace: bool) -> tuple[Tensor, int]:
      num_new_states = states.size(0)
      if num_new_states > self.buffer_max_size:
        print('Warning: More states than buffer size. Truncating.')
        states = states[:self.buffer_max_size]

      if inplace:
        buffer = self.buffer
      else:
        if self.rolling_average:
          buffer = self.buffer.clone()
        else:
          buffer = torch.zeros((buffer_max_size, dim))

      # Compute the number of slots available in the buffer.
      num_free_slots = self.buffer_max_size - self.buffer_size
      size_overflow = num_new_states - num_free_slots

      # If there is not enough space in the buffer...
      if size_overflow > 0:
        # First add some of states to free slots. 
        buffer[self.buffer_size : self.buffer_size + num_free_slots] = states[:num_free_slots]
        # Then randomly replace slots with the remaining states.
        drop_idx = torch.randperm(self.buffer_size)[:size_overflow]
        buffer[drop_idx] = states[num_free_slots:]
        buffer_size = self.buffer_max_size
        if inplace:
          self.buffer_size = buffer_size
        return buffer, buffer_size
      # If there is enough space in the buffer, just append.
      buffer[self.buffer_size : self.buffer_size + num_new_states] = states
      buffer_size = self.buffer_size + num_new_states
      if inplace:
        self.buffer_size = buffer_size
      return buffer, buffer_size 
    
    @torch.no_grad()
    def compute_distances(self, states: Tensor, buffer: Tensor) -> Tensor:
      distances = self.geometry.broadcast_distance_function(states, buffer)
      k_nearest_distances = torch.topk(distances, self.k, dim=1, largest=False).values
      return k_nearest_distances # shape: (states.size(0), k)
