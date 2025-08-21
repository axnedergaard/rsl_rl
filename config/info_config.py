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
