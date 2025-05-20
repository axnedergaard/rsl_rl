import torch
import numpy as np
import wandb
import os

def mae_loss(x, y):
  return np.mean(np.abs(x - y))

def mse_loss(x, y):
  return np.mean(np.square(x - y))

def scale_independent_loss(x, y):
  x = np.array(x, dtype=np.float32)
  y = np.array(y, dtype=np.float32)
  log_diff = np.log(x + 1e-6) - np.log(y + 1e-6)
  squared_log_diff = np.square(log_diff)  # square log diffs to emphasize larger errors
  squared_log_diff_mean = np.mean(squared_log_diff)  # avg squared log diffs for overall error
  log_diff_mean_sq = np.square(np.mean(log_diff))  # square mean log diffs to capture bias
  return squared_log_diff_mean - log_diff_mean_sq  # adjust for bias, emphasizing variance

def intrinsic_reward(rollouts, **kwargs):
  return torch.mean(rollouts['intrinsic_rewards'])

def extrinsic_reward(rollouts, **kwargs):
  return torch.mean(rollouts['extrinsic_rewards'])

def pathological_updates(density, **kwargs):
  return density.n_pathological

def entropy(density, **kwargs):
  return density.entropy().item()

def kmeans_loss(density, manifold, n=1e4, **kwargs):
  samples = torch.Tensor(manifold.sample(int(n)))
  assert samples.dim() == 2, f"Expected 2D tensor, got {samples.dim()}"
  distances, _ = density._find_closest_cluster(samples)
  return density.kmeans_objective(distances).item()

def kmeans_count_variance(density, **kwargs):
  cluster_sizes = density.cluster_sizes
  return torch.var(cluster_sizes).item()

def pdf_loss(manifold, density, n_points=1000, **kwargs):
  samples = manifold.sample(n_points)
  pdf_est = np.zeros(n_points)
  pdf_true = np.zeros(n_points)
  for i, sample in enumerate(samples):
    sample = torch.tensor(sample)
    pdf_true[i] = manifold.pdf(samples)
    pdf_est[i] = density.pdf(sample)
    if isinstance(pdf_est[i], torch.Tensor):
      pdf_est[i] = pdf_est[i].item()
  return scale_independent_loss(pdf_true, pdf_est)

def distance_loss(manifold, geometry, n_points=1000, **kwargs):
  x = torch.tensor(manifold.sample(n_points), dtype=torch.float32)
  y = torch.tensor(manifold.sample(n_points), dtype=torch.float32)
  distances_true = geometry.distance_function(x, y).detach()
  distances_est = manifold.distance_function(x, y)
  return scale_independent_loss(distances_true.numpy(), distances_est.numpy())

def state(rollouts, **kwargs):
  return rollouts['states'] 

def test(success=True, **kwargs):
  if success:
    print('Test succeeded.')
  else:
    print('Test failed (but succeeeded).')

def train_loss(geometry, **kwargs):
  if hasattr(geometry, 'train_loss'):
      loss = geometry.train_loss
      return loss
  return None


def test_loss(geometry, **kwargs):
  if hasattr(geometry, 'compute_test_loss') and callable(geometry.compute_test_loss):
      return geometry.compute_test_loss()
  return None

def control_zero_loss(geometry, **kwargs):
  if hasattr(geometry, 'compute_control_zero_loss') and callable(geometry.compute_control_zero_loss):
      return geometry.compute_control_zero_loss()
  return None

def control_mean_loss(geometry, **kwargs):
  if hasattr(geometry, 'compute_control_mean_loss') and callable(geometry.compute_control_mean_loss):
    return geometry.compute_control_mean_loss()
  return None

def distance_evaluation(geometry, **kwargs):
  if hasattr(geometry, 'distance_comparison_evaluation') and callable(geometry.distance_comparison_evaluation):
    return geometry.distance_comparison_evaluation()
  return None

# script represented states
def represented_state(geometry, rollouts, **kwargs):
  if hasattr(geometry, 'compute_embedded_states') and callable(geometry.compute_embedded_states):
    return geometry.compute_embedded_states(rollouts['states'])
  return None

def goal_state(rewarder, **kwargs):
  if hasattr(rewarder, 'goal'):
    return rewarder.goal
  return None

def represented_goal_state(rewarder, **kwargs):
  if hasattr(rewarder, 'embedded_goal') and callable(rewarder.embedded_goal):
    return rewarder.embedded_goal()
  return None

def graph_state(graph, **kwargs):
  if hasattr(graph, 'existing_states') and hasattr(graph, 'existing_weights'):
    return str({'states':graph.existing_states,'weights':np.array2string(graph.existing_weights, threshold=np.inf) if graph.existing_weights is not None else None})
  return None

def space_sample(rollouts, geometry, **kwargs):
  if hasattr(geometry, 'distance_function') and callable(geometry.distance_function):
    states = rollouts['states']
    n_dim = states.shape[1]
    starting_points = states[0,:]
    num_sample = 1000
    sampled_states = np.zeros((num_sample, n_dim), dtype= 'float')
    for i in range(n_dim):    
      sampled_states[:,i] = np.random.uniform(torch.min(states[:,i]),torch.max(states[:,i]),num_sample)

    sampled_tensor = torch.tensor(sampled_states).type(torch.FloatTensor)
    starting_tensor = torch.tensor(np.repeat([starting_points], num_sample, axis = 0))
    distances_tensor = geometry.distance_function(starting_tensor, sampled_tensor)
    sampled_data = torch.cat((sampled_tensor, distances_tensor.unsqueeze_(1)), 1)    
  return sampled_data
   

def save_model_checkpoint(geometry, **kwargs):
    checkpoint = {
        'network_state_dict': geometry.network.state_dict(),
        'embedding_dim': geometry.embedding_dim,
        'mahalanobis': geometry.mahalanobis,
        'dim': geometry.dim,
        'hidden_dims': geometry.hidden_dims
    }
    if geometry.mahalanobis:
        checkpoint['L_matrix'] = geometry.L
    temp_path = 'temp_checkpoint.pt'
    torch.save(checkpoint, temp_path)
    artifact = wandb.Artifact(
        name=f"model_checkpoint_{wandb.run.id}",
        type="model",
        description="Checkpoint"
    )
    artifact.add_file(temp_path); os.remove(temp_path)
    return artifact

def embedding_visualization(geometry, manifold, n_samples=1000, **kwargs):
    samples = manifold.sample(n_samples)
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    
    with torch.no_grad():
        embedded = geometry.network(samples_tensor)
        
    return {
        'original_points': samples.tolist(),
        'embedded_points': embedded.cpu().numpy().tolist()
    }

def distance_matrix_snapshot(geometry, manifold, n_points=20, **kwargs):
    """Compute pairwise distance matrix for visualization"""
    samples = manifold.sample(n_points)
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    
    distances = torch.zeros((n_points, n_points))
    with torch.no_grad():
        for i in range(n_points):
            for j in range(n_points):
                distances[i,j] = geometry._distance_function(
                    samples_tensor[i], 
                    samples_tensor[j]
                )
    
    return {
        'points': samples.tolist(),
        'distances': distances.cpu().numpy().tolist()
    }
