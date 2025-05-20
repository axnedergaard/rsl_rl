import h5py
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import PPO
import os
import numpy as np

class Data:
  def __init__(self, run):
    self.config = run.config
    self.df = run.history()

  def __getitem__(self, key):
    values = self.df[key]
    non_nan_values = values[values.notna()]
    return non_nan_values.to_numpy()

def get_plot_path(fn):
  return f"outputs/plots/{fn}.png"

def load_experiment_data(experiment_name, local=False, partial_local=[]):
  if local:
    # local_path = 
    raise NotImplementedError
  elif partial_local != []:
    import re
    def numeric_sort_key(filename):
      # Extracts the number between '-' and '.h5' in the filename.
      # Returns an integer for sorting.
      match = re.search(r'-(\d+)\.h5$', filename)
      return int(match.group(1)) if match else float('inf')  

    saved_local = {}
    for local_script in partial_local:
      local_path = f"outputs/{experiment_name}/data/{local_script}"
      h5_files = [f for f in os.listdir(local_path) if f.endswith(".h5")]
      h5_files.sort(key=numeric_sort_key)
      
      data_list = [] # for data access

      for fname in h5_files:
        path = os.path.join(local_path, fname)
        with h5py.File(path, "r") as f:
          keys = list(f.keys())
          if not keys:
            raise ValueError(f"No dataset found in {fname}")
          dataset = f[keys[0]]  # assume the first (and only) dataset
          if dataset.shape == (): # if it's scalar
            dataset = dataset[()]
          else:
            dataset = dataset[:]
          data_list.append(dataset)
      data_array = np.stack(data_list)
      saved_local[local_script] = data_array
    return saved_local

  else:
    api = wandb.Api()
    run = api.run(f"inirum/test/{experiment_name}")
    data = Data(run)
    return data

def save_plot(fig, fn):
  path = get_plot_path(fn)
  fig.savefig(path)

def load_data(path):
  raise NotImplementedError

def save_data(data, path):
  with h5py.File(path, 'a') as f: 
    f.create_dataset(path, data=data) 

def load_policy(path, env):
  return PPO.load(path, env)

def save_policy(policy, path):
  policy.save(path)
