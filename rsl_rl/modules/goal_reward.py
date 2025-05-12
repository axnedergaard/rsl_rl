import torch
import rum
from rum.geometry import EmbeddingGeometry
from rum.rewarder import GoalRewarder

class GoalReward:
    def __init__(
        self,
        geom: EmbeddingGeometry,
        goal_threshold: int,
        goal_update_freq: int,
        scaling: float = 1.0,
        device: str = "cpu",
        # TODO: state_normalization, reward_normalization, scaling_schedule
    ):
        """Initialize the information reward module."""

        self.goal_update_freq = goal_update_freq
        self.geom = geom
        self.num_states = geom.dim
        self.scaling = scaling
        self.rewarder = GoalRewarder(geom, goal_threshold, goal_update_freq, device=device)
        
    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.rewarder.reward_function(states)
        intrinsic_rewards *= self.scaling
        return intrinsic_rewards, states

    def update_goal(self, state):
        self.rewarder.update_goal(state)
