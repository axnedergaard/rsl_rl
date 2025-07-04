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
        # TODO: state_normalization, reward_normalization, scaling_schedule
    ):
        """Initialize the information reward module."""

        self.num_states = num_states
        self.density = density
        self.scaling = scaling
        self.geom = geom


    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.density.information(states)
        intrinsic_rewards *= self.scaling
        return intrinsic_rewards, states

    def update(self, states):
        assert False # TODO.
        """Update occupancy density estimate from batch of states."""
        with torch.no_grad():
            self.density.learn(states)
