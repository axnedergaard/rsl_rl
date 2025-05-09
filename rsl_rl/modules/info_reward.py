import torch
import rum

class InformationReward:
    def __init__(
        self,
        num_states: int,
        density_cfg: dict,
        info_geom_cfg: dict,
        scaling: float = 1.0,
        device: str = "cpu",
        # TODO: state_normalization, reward_normalization, scaling_schedule
    ):
        """Initialize the information reward module."""

        self.num_states = num_states
        self.scaling = scaling

        # Initialize information geometry.
        info_geom_class_name = info_geom_cfg.pop('name')
        info_geom_class = getattr(
            rum.information_geometry, 
            info_geom_class_name
        )
        info_geom = info_geom_class(**info_geom_cfg)

        # Initialize occupancy estimator.
        density_class_name = density_cfg.pop('name')
        density_class = getattr(
            rum.density, 
            density_class_name,
        )
        self.density = density_class(
            **density_cfg,
            dim = num_states, 
            information_geometry = info_geom,
            geometry = rum.geometry.EuclideanGeometry(num_states),
            device = device,
        )

    def get_intrinsic_reward(self, states) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intrinsic reward for batch of states."""
        with torch.no_grad():
            intrinsic_rewards = self.density.information(states)
        intrinsic_rewards *= self.scaling
        return intrinsic_rewards, states

    def update(self, states):
        """Update occupancy density estimate from batch of states."""
        with torch.no_grad():
            self.density.learn(states)
