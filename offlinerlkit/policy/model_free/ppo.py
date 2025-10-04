import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import time
from offlinerlkit.policy.base_policy import BasePolicy





def convert_sb3_to_offlinerl_format(sb3_state_dict: Dict[str, torch.Tensor], hidden_dims: list) -> Dict[str, torch.Tensor]:
    """
    将SB3的state_dict转换为与MOPO/COMBO兼容的格式

    Args:
        sb3_state_dict: SB3 PPO模型的state_dict
        hidden_dims: 网络隐藏层维度，用于确定层数

    Returns:
        转换后的state_dict，兼容MOPO/COMBO格式
    """
    converted = {}

    # 转换Actor网络 - 使用pi_features_extractor和mlp_extractor.policy_net
    # 第一层：pi_features_extractor.net.0 -> actor.backbone.model.0
    if "pi_features_extractor.net.0.weight" in sb3_state_dict:
        converted["actor.backbone.model.0.weight"] = sb3_state_dict["pi_features_extractor.net.0.weight"]
    if "pi_features_extractor.net.0.bias" in sb3_state_dict:
        converted["actor.backbone.model.0.bias"] = sb3_state_dict["pi_features_extractor.net.0.bias"]

    # 第二层：pi_features_extractor.net.2 -> actor.backbone.model.2
    if "pi_features_extractor.net.2.weight" in sb3_state_dict:
        converted["actor.backbone.model.2.weight"] = sb3_state_dict["pi_features_extractor.net.2.weight"]
    if "pi_features_extractor.net.2.bias" in sb3_state_dict:
        converted["actor.backbone.model.2.bias"] = sb3_state_dict["pi_features_extractor.net.2.bias"]

    # 转换Actor输出层（action_net）
    # SB3: action_net.weight/bias -> OfflineRL: actor.dist_net.mu.weight/bias
    if "action_net.weight" in sb3_state_dict:
        converted["actor.dist_net.mu.weight"] = sb3_state_dict["action_net.weight"]
    if "action_net.bias" in sb3_state_dict:
        converted["actor.dist_net.mu.bias"] = sb3_state_dict["action_net.bias"]

    # 转换log_std为sigma参数
    if "log_std" in sb3_state_dict:
        log_std = sb3_state_dict["log_std"]
        # log_std是1维的[action_dim]，需要转换为sigma网络的权重和偏置
        action_dim = log_std.shape[0]
        latent_dim = sb3_state_dict["action_net.weight"].shape[1]  # 从action_net获取输入维度

        # 创建sigma网络：Linear(latent_dim, action_dim)
        # 权重形状应该是[action_dim, latent_dim]
        converted["actor.dist_net.sigma.weight"] = torch.zeros(action_dim, latent_dim)
        # 偏置使用log_std的值
        converted["actor.dist_net.sigma.bias"] = log_std
    else:
        # 如果没有log_std，创建默认值
        action_dim = sb3_state_dict["action_net.weight"].shape[0]
        latent_dim = sb3_state_dict["action_net.weight"].shape[1]
        converted["actor.dist_net.sigma.weight"] = torch.zeros(action_dim, latent_dim)
        converted["actor.dist_net.sigma.bias"] = torch.zeros(action_dim)

    # 转换Critic网络 - 使用vf_features_extractor
    # 第一层：vf_features_extractor.net.0 -> critic1.backbone.model.0
    if "vf_features_extractor.net.0.weight" in sb3_state_dict:
        converted["critic1.backbone.model.0.weight"] = sb3_state_dict["vf_features_extractor.net.0.weight"]
    if "vf_features_extractor.net.0.bias" in sb3_state_dict:
        converted["critic1.backbone.model.0.bias"] = sb3_state_dict["vf_features_extractor.net.0.bias"]

    # 第二层：vf_features_extractor.net.2 -> critic1.backbone.model.2
    if "vf_features_extractor.net.2.weight" in sb3_state_dict:
        converted["critic1.backbone.model.2.weight"] = sb3_state_dict["vf_features_extractor.net.2.weight"]
    if "vf_features_extractor.net.2.bias" in sb3_state_dict:
        converted["critic1.backbone.model.2.bias"] = sb3_state_dict["vf_features_extractor.net.2.bias"]

    # 转换Critic输出层（value_net）
    # SB3: value_net.weight/bias -> OfflineRL: critic1.last.weight/bias
    if "value_net.weight" in sb3_state_dict:
        converted["critic1.last.weight"] = sb3_state_dict["value_net.weight"]
    if "value_net.bias" in sb3_state_dict:
        converted["critic1.last.bias"] = sb3_state_dict["value_net.bias"]

    # 为了兼容性，也创建critic2（通常与critic1相同）
    for key in list(converted.keys()):
        if key.startswith("critic1."):
            critic2_key = key.replace("critic1.", "critic2.")
            converted[critic2_key] = converted[key].clone()

    return converted


class ConvertAndSaveCallback(BaseCallback):
    """
    自定义Callback，在保存时自动进行权重转换
    """
    def __init__(self, save_path: str, hidden_dims: list, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.hidden_dims = hidden_dims
        self.save_freq = save_freq

    def _init_callback(self) -> None:
        # 创建保存目录
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_converted_model()
        return True

    def _save_converted_model(self) -> None:
        """保存转换后的模型"""
        if self.save_path is None:
            return

        # 获取SB3模型的state_dict
        sb3_state_dict = self.model.policy.state_dict()

        # 转换为OfflineRL格式
        converted_state_dict = convert_sb3_to_offlinerl_format(sb3_state_dict, self.hidden_dims)

        # 保存转换后的模型
        save_file = os.path.join(self.save_path, "policy.pth")
        torch.save(converted_state_dict, save_file)

        if self.verbose > 0:
            print(f"Converted model saved to {save_file}")


class FusionPPOEnv(gym.Env):
    """
    基于动力学模型的核聚变环境，用于PPO训练
    """
    def __init__(self, dynamics, sa_processor, offline_data, state_idxs, action_idxs, device='cpu'):
        super().__init__()
        
        self.dynamics = dynamics
        self.sa_processor = sa_processor
        self.offline_data = offline_data
        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.device = device
        
        # 定义观测和动作空间
        # 使用实际的观测维度，而不是state_idxs的长度
        obs_dim = offline_data['observations'].shape[1]
        action_dim = len(action_idxs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # 环境状态
        self.current_full_state = None
        self.previous_action = None
        self.time_step = 0
        self.max_episode_length = 200
        self.episode_length = 0
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 从离线数据中随机选择初始状态  完全随机
        idx = np.random.randint(0, self.offline_data['observations'].shape[0]-1)
        # 使用完整状态数据作为动力学模型的输入
        self.current_full_state = self.offline_data['full_observations'][idx].copy()
        self.previous_action = self.offline_data['pre_actions'][idx].copy()
        self.time_step = self.offline_data['time_step'][idx]
        self.episode_length = 0
        #save all
        self.current_global_idx = idx
        # 重置动力学模型
        self.dynamics.reset()

        # 获取观测 - 直接使用已经处理过的观测数据
        obs = self.offline_data['observations'][idx].copy()
        
        return obs.astype(np.float32), {}
    
    def step(self, action):
        """执行一步动作"""
        # 确保动作在正确范围内
        # print("action")
        # print(action)
        action=action
        action = np.clip(action, -1.0, 1.0)
        
        # 转换动作格式
        step_action = self.sa_processor.get_step_action(action.reshape(1, -1))[0]

        # 构建完整动作
        full_action = self.offline_data['full_actions'][0].copy()  # 获取动作模板
        full_action[self.action_idxs] = step_action

        #add 
        max_time_step=150
        valid_time_step = min(self.time_step, max_time_step)

        #add
        max_data_idx = len(self.offline_data['terminals']) - 1
        current_terminal_idx = min(self.current_global_idx, max_data_idx)
        current_time_terminals = self.offline_data['terminals'][current_terminal_idx]
        
        # 使用动力学模型预测下一状态
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

        # 从info中获取完整的下一状态
        self.current_full_state = info["next_full_observations"][0]
        self.previous_action = full_action
        self.time_step += 1
        self.episode_length += 1
        self.current_global_idx +=1
        # 获取下一观测 - 从完整状态中提取选定的状态维度，然后通过sa_processor处理
        selected_state = self.current_full_state[self.state_idxs]
        next_obs = self.sa_processor.get_rl_state(
            selected_state.reshape(1, -1),
            np.array([self.current_global_idx])
            #np.array([self.time_step])
        )[0]
        
        # 检查是否结束
        done = terminal[0] or self.episode_length >= self.max_episode_length
        
        return next_obs.astype(np.float32), float(reward[0]), done, False, info


# 全局变量用于传递hidden_dims配置
_GLOBAL_HIDDEN_DIMS = [256, 256]

class FusionFeatureExtractor(BaseFeaturesExtractor):
    """
    核聚变专用的特征提取器
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, **kwargs):
        # 从kwargs中获取hidden_dims，如果没有提供则使用全局变量
        hidden_dims = kwargs.get('hidden_dims', _GLOBAL_HIDDEN_DIMS)

        # 如果提供了hidden_dims，使用最后一层作为features_dim
        if 'hidden_dims' in kwargs:
            features_dim = hidden_dims[-1]
        else:
            # 使用全局变量的最后一层作为features_dim
            features_dim = _GLOBAL_HIDDEN_DIMS[-1]

        super().__init__(observation_space, features_dim)

        n_input = observation_space.shape[0]

        # 构建动态网络结构
        layers = []
        prev_dim = n_input

        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        # 添加输出层
        layers.extend([
            nn.Linear(prev_dim, features_dim),
            nn.ReLU()
        ])

        self.net = nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class FusionPPOPolicy(ActorCriticPolicy):
    """
    核聚变专用的PPO策略
    """
    def __init__(self, *args, **kwargs):
        # 设置自定义特征提取器
        kwargs['features_extractor_class'] = FusionFeatureExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        super().__init__(*args, **kwargs)


class PPOPolicy(BasePolicy):
    """
    基于Stable Baselines3的PPO策略，用于核聚变控制
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
        tensorboard_log: str = None,  # 添加这个参数
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

        # 存储网络配置用于权重转换
        self.hidden_dims = kwargs.get('hidden_dims', [256, 256])

        # 创建环境
        self.env = FusionPPOEnv(
            dynamics=dynamics,
            sa_processor=sa_processor,
            offline_data=offline_data,
            state_idxs=state_idxs,
            action_idxs=action_idxs,
            device=device
        )

        # 包装为向量化环境
        self.vec_env = DummyVecEnv([lambda: self.env])

        # 配置网络架构以匹配MOPO/COMBO
        policy_kwargs = {
            "net_arch": self.hidden_dims,
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": FusionFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": self.hidden_dims[-1],
                "hidden_dims": self.hidden_dims
            }
        }
        policy_kwargs.update(kwargs.get('policy_kwargs', {}))

        # 设置全局变量
        global _GLOBAL_HIDDEN_DIMS
        _GLOBAL_HIDDEN_DIMS = self.hidden_dims

        # 创建PPO模型 - 添加tensorboard_log参数
        self.model = PPO(
            policy=FusionPPOPolicy,
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
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,  # 启用TensorBoard日志
            verbose=1
        )
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action[0] if action.ndim > 1 else action
    
    def learn(self, total_timesteps: int, **kwargs) -> Dict[str, float]:
        """训练策略"""
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        
        # 返回训练统计信息
        return {
            "total_timesteps": total_timesteps,
            "learning_rate": self.model.learning_rate,
        }
    
    def save(self, path: str):
        """保存模型（同时保存SB3格式和转换后的格式）"""
        # 保存SB3原始格式
        self.model.save(path)

        # 保存转换后的格式
        sb3_state_dict = self.model.policy.state_dict()
        converted_state_dict = convert_sb3_to_offlinerl_format(sb3_state_dict, self.hidden_dims)

        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存为.pth格式
        pth_path = path + ".pth" if not path.endswith(".pth") else path
        torch.save(converted_state_dict, pth_path)

    def load(self, path: str):
        """加载模型"""
        self.model = PPO.load(path, env=self.vec_env)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """返回转换后的完整policy状态"""
        sb3_state_dict = self.model.policy.state_dict()
        return convert_sb3_to_offlinerl_format(sb3_state_dict, self.hidden_dims)

    def get_actor(self):
        """提取actor用于评估"""
        return self.model.policy

    def evaluate_policy(self, env, num_episodes: int = 5) -> Dict[str, float]:
        """
        评估策略性能，返回与其他算法兼容的格式
        """
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

                if episode_length >= 200:  # 防止无限循环
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "eval/episode_reward": episode_rewards,
            "eval/episode_length": episode_lengths
        }






