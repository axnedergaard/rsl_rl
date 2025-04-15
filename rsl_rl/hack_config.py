@configclass
class RslRlInfoRewardCfg:
    info_geom_cfg = {
      'cls_name': 'AlphaGeometry',
      'alpha': 0.0, 
    }
    # k-means.
    density_cfg = {
      'cls_name': 'KMDensityEstimator',
      'k': 300,
      'learning_rate': 0.05,
      'balancing_strength': 0.1,
    }
    # k nearest neighbors.
    #density_cfg = {
    #  'cls_name': 'KNNDensityEstimator',
    #  'k': 10,
    #  'buffer_max_size': 1000, 
    #}
