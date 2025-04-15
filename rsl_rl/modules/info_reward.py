import torch
import information_reward

class InformationReward:
    def __init__(
        self,
        num_states: int,
        density_cfg: dict,
        info_geom_cfg: dict,
        weight: float = 1.0,
        device: str = "cpu",
        # TODO: state_normalization, reward_normalization, weight_schedule
    ):
        """Initialize the information reward module."""

        self.num_states = num_states
        self.weight = weight

        # Initialize information geometry.
        info_geom_cls_name = info_geom_cfg.pop('cls_name')
        info_geom_cls = getattr(
            information_reward, 
            info_geom_cls_name
        )
        info_geom = info_geom_cls(**info_geom_cfg)

        # Initialize occupancy estimator.
        density_cls_name = density_cfg.pop('cls_name')
        density_cls = getattr(
            information_reward, 
            density_cls_name,
        )
        self.density = density_cls(
            **density_cfg,
            dim = num_states, 
            information_geometry = info_geom,
            geometry = information_reward.EuclideanGeometry(num_states),
            device = device,
        )

    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.density.information(states)
        intrinsic_rewards *= self.weight
        return intrinsic_rewards, states

    def update(self, states):
        """Update occupancy density estimate from batch of states."""
        self.density.learn(states)
