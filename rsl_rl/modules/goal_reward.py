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
        optimize_steps: int = 0,
        scaling: float = 1.0,
        beta_schedule = None,
        device: str = "cpu",
        min_reward: float = -100.0,
        max_reward: float = 100.0,
        # TODO: state_normalization, reward_normalization, scaling_schedule
    ):
        """Initialize the information reward module."""

        self.goal_update_freq = goal_update_freq
        self.geom = geom
        self.num_states = geom.dim
        self.initial_scaling = scaling
        self.scaling = scaling
        self.beta_schedule = beta_schedule
        self.rewarder = GoalRewarder(geom, goal_threshold, goal_update_freq, optimize_steps, device=device)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.new_trajectory = 1

    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.rewarder.reward_function(states)
        torch.clamp(intrinsic_rewards, min=self.min_reward, max=self.max_reward)
        intrinsic_rewards *= (1.0 - self.new_trajectory)
        intrinsic_rewards *= self.scaling
        return intrinsic_rewards

    def update_scaling(self, iteration, max_iteration):
        if self.beta_schedule is not None:
            self.scaling = self.beta_schedule(self.initial_scaling, iteration, max_iteration)

    def update_goal(self, state):
        self.rewarder.update_goal(state)
