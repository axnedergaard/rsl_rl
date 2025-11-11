import torch
import rum

class InformationReward:
    def __init__(
        self,
        num_states: int,
        density,
        geom = None,
        scaling: float = 1.0,
        device: str = "cpu",
        min_reward: float = -100.0,
        max_reward: float = 100.0,
        alpha_schedule = None,
        beta_schedule = None,
        # TODO: state_normalization, reward_normalization
    ):
        """Initialize the information reward module."""

        self.num_states = num_states
        self.density = density
        self.scaling = scaling
        self.geom = geom
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.alpha_schedule = alpha_schedule
        self.beta_schedule = beta_schedule
        self.initial_scaling = scaling
        if self.alpha_schedule is not None:
          self.initial_alpha = self.density.information_geometry.alpha
          

    def update_scaling(self, iteration, max_iteration):
        if self.beta_schedule is not None:
            self.scaling = self.beta_schedule(self.initial_scaling, iteration, max_iteration)
        if self.alpha_schedule is not None:
            self.density.information_geometry.alpha = self.alpha_schedule(self.initial_alpha, iteration, max_iteration)


    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.density.information(states)
        torch.clamp(intrinsic_rewards, min=self.min_reward, max=self.max_reward)
        intrinsic_rewards *= self.scaling
        return intrinsic_rewards

    def update(self, states):
        assert False # TODO.
        """Update occupancy density estimate from batch of states."""
        with torch.no_grad():
            self.density.learn(states)
