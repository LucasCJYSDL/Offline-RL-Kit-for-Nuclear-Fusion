import numpy as np
import torch
from typing import Dict, Union, Optional, Any
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from offlinerlkit.policy.base_policy import BasePolicy
from offlinerlkit.utils.convert_sb3_to_offlinerl_format import convert_sb3_to_offlinerl_format




class FusionPPOEnv(gym.Env):
    """
    Dynamics-model-based nuclear fusion environment for PPO training.
    """
    def __init__(self, dynamics, sa_processor, offline_data, state_idxs, action_idxs, device='cpu'):
        super().__init__()
        
        self.dynamics = dynamics
        self.sa_processor = sa_processor
        self.offline_data = offline_data
        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.device = device
        
        # Define observation and action spaces
        obs_dim = offline_data['observations'].shape[1]
        action_dim = len(action_idxs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_full_state = None
        self.previous_action = None
        self.time_step = 0
        self.max_episode_length = 200
        self.episode_length = 0

        # Used for logging episode statistics (for SB3 rollout metrics)
        self.episode_reward = 0.0
        
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        
        # Sample a completely random initial state from the offline data
        # idx = np.random.randint(0, self.offline_data['observations'].shape[0]-1)
        traj_idx = np.random.randint(0, len(self.offline_data['traj_start_indices']))
        start_idx = self.offline_data['traj_start_indices'][traj_idx]
        idx = start_idx
        # Use the full state data as input for the dynamics model
        self.current_full_state = self.offline_data['full_observations'][idx].copy()
        self.previous_action = self.offline_data['pre_actions'][idx].copy()
        self.time_step = self.offline_data['time_step'][idx]
        self.episode_length = 0
        #save all
        self.current_global_idx = idx
        self.dynamics.reset()
        obs = self.offline_data['observations'][idx].copy()
        self.episode_reward = 0.0
        
        return obs.astype(np.float32), {}
    
    def step(self, action):
        
        # Convert action format
        step_action = self.sa_processor.get_step_action(action.reshape(1, -1))[0]

        # Construct the full action
        full_action = self.offline_data['full_actions'][0].copy()  
        full_action[self.action_idxs] = step_action
 
        max_time_step=150
        valid_time_step = min(self.time_step, max_time_step)
        max_data_idx = len(self.offline_data['terminals']) - 1
        current_terminal_idx = min(self.current_global_idx, max_data_idx)
        current_time_terminals = self.offline_data['terminals'][current_terminal_idx]
        
        # Predict the next state using the dynamics model
        next_obs, reward, terminal, info = self.dynamics.step(
            cur_state=self.current_full_state.reshape(1, -1),
            pre_action=self.previous_action.reshape(1, -1),
            cur_action=full_action.reshape(1, -1),
            time_steps=np.array([self.time_step]),
            time_terminals=np.array([current_time_terminals]),
            state_idxs=self.state_idxs,
            #batch_idxs=np.array([valid_time_step])
            batch_idxs=np.array([self.current_global_idx])
        )

        # Get the complete next state from info
        self.current_full_state = info["next_full_observations"][0]
        self.previous_action = full_action
        self.time_step += 1
        self.episode_length += 1
        self.current_global_idx +=1

        reward_scalar = float(reward[0, 0])
        self.episode_reward += reward_scalar

        # Get the next observation - extract selected state dimensions from the full state, then process through sa_processor
        selected_state = self.current_full_state[self.state_idxs]
        next_obs = self.sa_processor.get_rl_state(
            selected_state.reshape(1, -1),
            np.array([self.current_global_idx])
            #np.array([self.time_step])
        )[0]
        
        done = terminal[0] or self.episode_length >= self.max_episode_length

        if done:
            import time
            ep_info = {
                "r": round(self.episode_reward, 6),  
                "l": self.episode_length,             
            }
            info["episode"] = ep_info  
        
        return next_obs.astype(np.float32), float(reward[0]), done, False, info



class PPOPolicy(BasePolicy):
    """
    PPO policy based on Stable Baselines3
    """
    
    def __init__(
        self,
        dynamics,
        state_idxs,
        action_idxs,
        sa_processor,
        offline_data,
        device: str = 'cpu',
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        pol_hidden_dims: list = [250, 250],  
        val_hidden_dims: list = [250, 250],  
        tensorboard_log: str = None,
        **kwargs
    ):
        super().__init__()

        self.dynamics = dynamics
        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.sa_processor = sa_processor
        self.offline_data = offline_data
        self.device = device
        self.tensorboard_log = tensorboard_log

        # Store network configuration for weight conversion
        self.pol_hidden_dims = pol_hidden_dims
        self.val_hidden_dims = val_hidden_dims

        # create environment
        self.env = FusionPPOEnv(
            dynamics=dynamics,
            sa_processor=sa_processor,
            offline_data=offline_data,
            state_idxs=state_idxs,
            action_idxs=action_idxs,
            device=device
        )
        self.vec_env = DummyVecEnv([lambda: self.env])

        # Configure policy_kwargs
        policy_kwargs = {
            'net_arch': dict(pi=pol_hidden_dims, vf=val_hidden_dims),  
            'activation_fn': torch.nn.ReLU,                            
            'share_features_extractor': False,                         
            'squash_output': True, 
            'log_std_init': 0.0,                                      
        }
        policy_kwargs.update(kwargs.get('policy_kwargs', {}))

        # Create the PPO model
        self.model = PPO(
            policy="MlpPolicy", 
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=True,           
            sde_sample_freq=-1,     
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=1
        )
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action[0] if action.ndim > 1 else action
    
    def learn(self, total_timesteps: int, **kwargs) -> Dict[str, float]:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        
        return {
            "total_timesteps": total_timesteps,
            "learning_rate": self.model.learning_rate,
        }
    
    def save(self, path: str):
        self.model.save(path)
        sb3_state_dict = self.model.policy.state_dict()
        converted_state_dict = convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,
            self.val_hidden_dims
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pth_path = path + ".pth" if not path.endswith(".pth") else path
        torch.save(converted_state_dict, pth_path)

    def load(self, path: str):
        """load model"""
        self.model = PPO.load(path, env=self.vec_env)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        sb3_state_dict = self.model.policy.state_dict()
        return convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,
            self.val_hidden_dims
        )

    def get_actor(self):
        return self.model.policy

    def evaluate_policy(self, env, num_episodes: int = 5) -> Dict[str, float]:
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

                if episode_length >= 200: #modification 
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "eval/episode_reward": episode_rewards,
            "eval/episode_length": episode_lengths
        }






