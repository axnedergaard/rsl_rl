from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlRndCfg
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from dataclasses import MISSING

@configclass
class RslRlInfoRewardCfg:
    info_geom_cfg = {
      'cls_name': 'AlphaGeometry',
      'alpha': 0.0, 
    }
    # k-means.
    # density_cfg = {
    #   'cls_name': 'KMDensityEstimator',
    #   'k': 300,
    #   'learning_rate': 0.05,
    #   'balancing_strength': 0.0001,
    # }
    weight: float = 0.01
    # k nearest neighbors.
    density_cfg = {
     'cls_name': 'KNNDensityEstimator',
     'k': 3,
     'buffer_max_size': 2000,
    }


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

    info_reward_cfg: RslRlInfoRewardCfg | None = None
    """The configuration for the information reward module. Default is None,
    in which case information reward is not used.
    """