from gymnasium import spaces
import numpy as np
import torch
from typing import Union



class Transition():
    def __init__(self, obs: torch.Tensor, 
                 obs_next: torch.Tensor, 
                 act: torch.Tensor, 
                 reward: torch.Tensor, 
                 done: torch.Tensor):
        self.obs = obs
        self.obs_next = obs_next
        self.act = act
        self.reward = reward
        self.done = done



class TransitionAI(Transition):
    def __init__(self, obs: torch.Tensor, 
                 obs_next: torch.Tensor, 
                 act: torch.Tensor, 
                 act_model: torch.Tensor, 
                 reward: torch.Tensor, 
                 done: torch.Tensor):
        super().__init__(obs, obs_next, act, reward, done)
        self.act_model = act_model



class ReplayBuffer():
    def __init__(self, buffer_size: int, obs_space: spaces.Box, act_space: spaces.Box, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = (1,) if isinstance(obs_space, spaces.Discrete) else obs_space.shape
        self.act_shape = (1,) if isinstance(act_space, spaces.Discrete) else act_space.shape

        self.pos = 0
        self.full = False
        self.device = device

        self.obs_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=obs_space.dtype)
        self.obs_next_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=obs_space.dtype)
        self.act_array = np.zeros((self.buffer_size, *self.act_shape), dtype=act_space.dtype)
        self.rew_array = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_array = np.zeros((self.buffer_size, 1), dtype=np.float32)


    def add(self, obs: np.ndarray, obs_next: np.ndarray, act: np.ndarray, rew: float, done: int):
        self.obs_array[self.pos] = np.array(obs)
        self.obs_next_array[self.pos] = np.array(obs_next)
        self.act_array[self.pos] = np.array(act)
        self.rew_array[self.pos] = np.array(rew)
        self.done_array[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    

    def sample(self, batch_size: int) -> Union[Transition, TransitionAI]:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)
    

    def _get_samples(self, batch_inds):
        data = (
            self.index(self.obs_array, batch_inds),
            self.index(self.obs_next_array, batch_inds),
            self.index(self.act_array, batch_inds),
            self.index(self.rew_array, batch_inds),
            self.index(self.done_array, batch_inds),
        )
        return Transition(*tuple(map(self.to_torch, data)))


    def to_torch(self, array: np.ndarray):
        return torch.tensor(array, device=self.device)
    

    def index(self, x, inds):
        return x[inds] if x.ndim==1 else x[inds, :]



class ModelBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_space, act_space, device):
        super().__init__(buffer_size, obs_space, act_space, device)
        self.act_model_array = np.zeros((self.buffer_size, *self.act_shape), dtype=act_space.dtype)
    

    def add(self, obs: np.ndarray, obs_next: np.ndarray, act: np.ndarray, act_model: np.ndarray, rew: float, done: int):
        self.obs_array[self.pos] = np.array(obs)
        self.obs_next_array[self.pos] = np.array(obs_next)
        self.act_array[self.pos] = np.array(act)
        self.act_model_array[self.pos] = np.array(act_model)
        self.rew_array[self.pos] = np.array(rew)
        self.done_array[self.pos] = np.array(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    

    def _get_samples(self, batch_inds):
        data = (
            self.index(self.obs_array, batch_inds),
            self.index(self.obs_next_array, batch_inds),
            self.index(self.act_array, batch_inds),
            self.index(self.act_model_array, batch_inds),
            self.index(self.rew_array, batch_inds),
            self.index(self.done_array, batch_inds),
        )
        return TransitionAI(*tuple(map(self.to_torch, data)))