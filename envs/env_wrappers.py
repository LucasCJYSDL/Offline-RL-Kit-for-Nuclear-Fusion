from copy import deepcopy
from envs.base_env import NFBaseEnv
import argparse
import random
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym

class PlanningWrapper:
    def __init__(self, core_env: NFBaseEnv):
        self.core_env = deepcopy(core_env) # for memory safety
    
    def get_shot_length(self):
        # only run this after reset
        cur_shot_time_limit = getattr(self.core_env, 'cur_shot_time_limit', None)
        assert cur_shot_time_limit is not None

        return cur_shot_time_limit - self.core_env.cur_time
    
    def get_reference_shots(self):
        return self.core_env.get_eval_shot_list()
    
    def reset(self, shot_id):        
        return self.core_env.reset(shot_id)

    def step(self, cur_action):
        return self.core_env.step(cur_action)
    
    def seed(self, seed):
        self.core_env.seed(seed)

#ppo
class GymnasiumWrapper(gym.Env):
    """
    self environment -> Gymnasium compatiable
    """
    def __init__(self, custom_env,action_dim, obs_dim):
        super().__init__()
        self.env = custom_env

        # Set up the observation space 

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        
        obs = self.env.reset()
        info = {}

        # Handle torch.Tensor observations and flatten them to 1D
        obs = obs.cpu().numpy().flatten()

        return obs, info

    def step(self, action):
        if len(action.shape) == 1:
            action = action.reshape(1, -1)  # Add batch dimension (1, action_dim)
        result = self.env.step(action)
        # Old format: (obs, reward, done, info)
        obs, reward, done, info = result

        # Handle torch.Tensor observations and flatten them to 1D
        obs = obs.cpu().numpy().flatten()


        return obs, reward, done, False, info  # add truncated