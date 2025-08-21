from isaaclab.utils import configclass

@configclass
class InformationGeometryCfg:
  name = 'AlphaGeometry',
  alpha = 0.0,
  offset = 1.0,

@configclass
class DensityCfg:
  name = 'KMDensityEstimator',
  k = 300,
  num_passes = 5,
  homeostasis = False,
  balancing_strength = 0.1, # Not used unless homeostasis = True.

@configclass
class RewarderCfg:
  name='DensityRewarder',
  scaling = 1.0,
  density = DensityCfg(),
  info_geom_cfg = InformationGeometryCfg(),

@configclass
class InfoAlgorithmCfg:
  value_loss_coef = 1.0,
  use_clipped_value_loss = True,
  clip_param = 0.2,
  entropy_coef = 0.01,
  num_learning_epochs = 5,
  num_mini_batches = 4,
  learning_rate = 1.0e-3,
  schedule = "adaptive",
  gamma = 0.99,
  lam = 0.95,
  desired_kl = 0.01,
  max_grad_norm = 1.0,
  rewarder=RewarderCfg(),
