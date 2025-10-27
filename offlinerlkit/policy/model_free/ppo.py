import numpy as np
import torch
from typing import Dict
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from offlinerlkit.policy.base_policy import BasePolicy





def convert_sb3_to_offlinerl_format(sb3_state_dict: Dict[str, torch.Tensor], pol_hidden_dims: list, val_hidden_dims: list) -> Dict[str, torch.Tensor]:
    """
    将 SB3 的 state_dict 转换为 offlinerlkit 格式

    Args:
        sb3_state_dict: SB3 模型的 state_dict
        pol_hidden_dims: Policy network 的隐藏层维度列表
        val_hidden_dims: Value network 的隐藏层维度列表

    Returns:
        offlinerl_state_dict: 转换后的 state_dict
    """
    offlinerl_state_dict = {}

    # 1. 转换 policy network (mlp_extractor.policy_net -> actor_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(pol_hidden_dims):
        # SB3: mlp_extractor.policy_net.{layer_idx}.weight/bias
        # OfflineRL: actor.backbone.model.{i*2}.weight/bias
        weight_key_sb3 = f'mlp_extractor.policy_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.policy_net.{layer_idx}.bias'

        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'actor.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'actor.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]

        layer_idx += 2  # Linear + ReLU

    # 2. 转换 action_net (mean 输出层)
    if 'action_net.weight' in sb3_state_dict:
        offlinerl_state_dict['actor.dist_net.mu.weight'] = sb3_state_dict[f'action_net.weight']
        offlinerl_state_dict['actor.dist_net.mu.bias'] = sb3_state_dict[f'action_net.bias']

    # 3. 转换 log_std (独立参数)
    if 'log_std' in sb3_state_dict:
        # SB3 的 log_std 是 (latent_dim, action_dim) 对于 gSDE
        # OfflineRL 的 sigma 是 Linear 层
        log_std = sb3_state_dict['log_std']

        # 如果是 gSDE (2D tensor)
        if log_std.dim() == 2:
            # gSDE: log_std 形状是 (latent_dim, action_dim)
            # 转换为固定的 sigma_param: (action_dim, 1)
            # 策略：取所有 latent 维度的平均值
            avg_log_std = log_std.mean(dim=0, keepdim=True).T  # (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = avg_log_std
        else:
            # 标准 PPO (1D tensor): log_std 形状是 (action_dim,)
            # 转换为 sigma_param: (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = log_std.unsqueeze(-1)


    # 4. 转换 value network (mlp_extractor.value_net -> critic_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(val_hidden_dims):
        weight_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.bias'

        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]

        layer_idx += 2  # Linear + ReLU

    # 5. 转换 value_net (value 输出层)
    if 'value_net.weight' in sb3_state_dict:
        offlinerl_state_dict['critic1.last.weight'] = sb3_state_dict['value_net.weight']
        offlinerl_state_dict['critic1.last.bias'] = sb3_state_dict['value_net.bias']

    # 为了兼容性，也创建critic2（通常与critic1相同）
    for key in list(offlinerl_state_dict.keys()):
        if key.startswith("critic1."):
            critic2_key = key.replace("critic1.", "critic2.")
            offlinerl_state_dict[critic2_key] = offlinerl_state_dict[key].clone()

    return offlinerl_state_dict


class ConvertAndSaveCallback(BaseCallback):
    """
    自定义Callback，在保存时自动进行权重转换
    """
    def __init__(self, save_path: str, pol_hidden_dims: list, val_hidden_dims: list, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.pol_hidden_dims = pol_hidden_dims  # 新参数
        self.val_hidden_dims = val_hidden_dims  # 新参数
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
        converted_state_dict = convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,  # 传入新参数
            self.val_hidden_dims   # 传入新参数
        )

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

        # 用于记录 episode 统计信息（供 SB3 的 rollout 指标使用）
        self.episode_reward = 0.0
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # # 从离线数据中随机选择初始状态  完全随机
        # idx = np.random.randint(0, self.offline_data['observations'].shape[0]-1)
        traj_idx = np.random.randint(0, len(self.offline_data['traj_start_indices']))
        start_idx = self.offline_data['traj_start_indices'][traj_idx]
        idx = start_idx
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
        
        # 重置 episode 统计信息（供 SB3 记录 rollout 指标）
        self.episode_reward = 0.0
        
        return obs.astype(np.float32), {}
    
    def step(self, action):
        """执行一步动作"""
        # 确保动作在正确范围内
        # print("action")
        # print(action)
        #action=action
        #action = np.clip(action, -1.0, 1.0)
        
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

        reward_scalar = float(reward[0, 0])
        self.episode_reward += reward_scalar

        # 获取下一观测 - 从完整状态中提取选定的状态维度，然后通过sa_processor处理
        selected_state = self.current_full_state[self.state_idxs]
        next_obs = self.sa_processor.get_rl_state(
            selected_state.reshape(1, -1),
            np.array([self.current_global_idx])
            #np.array([self.time_step])
        )[0]
        
        # 检查是否结束
        done = terminal[0] or self.episode_length >= self.max_episode_length

        if done:
            import time
            ep_info = {
                "r": round(self.episode_reward, 6),  # episode 总奖励
                "l": self.episode_length,             # episode 长度
            }
            info["episode"] = ep_info  
        
        return next_obs.astype(np.float32), float(reward[0]), done, False, info


# 删除 FusionFeatureExtractor 和 FusionPPOPolicy 类
# 使用 SB3 标准的 MlpPolicy


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
        pol_hidden_dims: list = [250, 250],  # 新参数
        val_hidden_dims: list = [250, 250],  # 新参数
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

        # 存储网络配置用于权重转换
        self.pol_hidden_dims = pol_hidden_dims
        self.val_hidden_dims = val_hidden_dims

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

        # 配置 policy_kwargs - 分离网络架构
        policy_kwargs = {
            'net_arch': dict(pi=pol_hidden_dims, vf=val_hidden_dims),  # 分离的网络架构
            'activation_fn': torch.nn.ReLU,                            # ReLU 激活
            'share_features_extractor': False,                         # 不共享 features extractor
            'squash_output': True, 
            'log_std_init': 0.0,                                       # log_std 初始值
        }
        policy_kwargs.update(kwargs.get('policy_kwargs', {}))

        # 创建 PPO 模型 - 使用标准 MlpPolicy + gSDE + tanh
        self.model = PPO(
            policy="MlpPolicy",  # 使用标准策略字符串
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
            use_sde=True,           # 启用 gSDE (作为 PPO 的直接参数)
            sde_sample_freq=-1,     # 每次都重新采样噪声
            device=device,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
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
        converted_state_dict = convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,
            self.val_hidden_dims
        )

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
        return convert_sb3_to_offlinerl_format(
            sb3_state_dict,
            self.pol_hidden_dims,
            self.val_hidden_dims
        )

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






