import torch
import rum
from rum.geometry import EmbeddingGeometry
from rum.rewarder import GoalRewarder

class GoalReward:
    def __init__(
        self,
        geometry: EmbeddingGeometry,
        goal_threshold: int,
        goal_update_freq: int,
        weight: float = 1.0,
        device: str = "cpu",
        # TODO: state_normalization, reward_normalization, weight_schedule
    ):
        """Initialize the information reward module."""

        self.goal_update_freq = goal_update_freq
        self.geometry = geometry
        self.num_states = geometry.dim_states
        self.weight = weight
        self.rewarder = GoalRewarder(geometry, goal_threshold)

        self.goal_count = 0
        

    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.rewarder.reward_function(states)
        intrinsic_rewards *= self.weight
        return intrinsic_rewards, states

    def update_goal(self, state):
        self.rewarder.update_goal(state)
        
