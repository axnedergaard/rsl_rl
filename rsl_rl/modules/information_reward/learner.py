import torch
from torch import Tensor

class _Buffer():

    def __init__(self, dim_states: int, buffer_size: int):
        self.memory = torch.zeros((buffer_size, dim_states))
        self.dim_state = dim_states
        self.max_size = buffer_size
        self.size = 0

    def read(self, num_states):
        return self.memory[:num_states]

    def append(self, states):
        num_states = states.size(0)
        self.memory[self.size:self.size+num_states] = states
        self.size += num_states

    def flush(self):
        self.memory = torch.zeros((self.max_size, self.dim_state))
        self.size = 0


class _ReplayBuffer():

    def __init__(self, dim_states: int, buffer_size: int, replay_size: int): # buffer_size = retrieve size
        self.memory = torch.zeros((replay_size, dim_states))
        self.dim_state = dim_states
        self.max_replay_size = replay_size
        self.replay_size = 0
        # Virtual sizes to support delayed learning using Learner.
        self.max_size = buffer_size
        self.size = 0         

    def read(self, num_states):
        permuted_indices = torch.randperm(self.replay_size)
        read_indices = permuted_indices[:num_states]
        return self.memory[read_indices]

    def append(self, states):
        free_space = self.max_replay_size - self.replay_size

        # Append states.
        states_append = states[:free_space]
        num_append = states_append.size()[0]
        self.memory[self.max_replay_size - free_space : self.max_replay_size - free_space + num_append] = states_append
        self.replay_size += num_append

        # Replace states.
        states_replace = states[free_space:]
        num_replace = states_replace.size()[0]
        permuted_indices = torch.randperm(self.max_replay_size)
        replace_indices = permuted_indices[:num_replace]
        self.memory[replace_indices] = states_replace

        self.size = min(self.size + states.size()[0], self.max_size) # Increment virtual size.

    def flush(self):
        self.size = 0 # Reset virtual size.


class Learner():
    
    def __init__(self, dim_states: int, buffer_size: int, replay_size: int = 0):
        if replay_size > 0:  # Use replay buffer.
            self.buffer = _ReplayBuffer(dim_states, buffer_size, replay_size)
        else:
            self.buffer = _Buffer(dim_states, buffer_size)
        self.dim_states = dim_states
        self.buffer_size = buffer_size

    def learn(self, states: Tensor):
        if not isinstance(states, Tensor) or states.dim() != 2:
            raise ValueError("States must be a 2D tensor.")
        
        num_states = states.size(0)
        while num_states > 0:
            available_space = self.buffer.max_size - self.buffer.size
            if available_space == 0:
                learn_states = self.buffer.read(self.buffer_size)
                self._learn(learn_states)
                self.buffer.flush()
                continue

            add_states_count = min(available_space, num_states)
            self.buffer.append(states[:add_states_count])
            states = states[add_states_count:]
            num_states -= add_states_count


    def _learn(self, states: Tensor):
        raise NotImplementedError("Learner must implement _learn method")
