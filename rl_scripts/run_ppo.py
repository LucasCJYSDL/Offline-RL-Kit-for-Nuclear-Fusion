import argparse
import random
import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.policy.model_free.ppo import PPOPolicy, ConvertAndSaveCallback
from offlinerlkit.utils.logger import Logger, make_log_dirs
from rl_preparation.get_rl_data_envs import get_rl_data_envs


def get_args():
    parser = argparse.ArgumentParser(description="PPO for Nuclear Fusion Control")
    
    # 
    parser.add_argument("--algo-name", type=str, default="ppo")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048 )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10 )
    parser.add_argument("--gamma", type=float, default=0.99 )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    
    # training parameters
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--eval-freq", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-freq", type=int, default=100000)
    
    # environment parameter
    parser.add_argument("--max-episode-length", type=int, default=200)
    #!!! what you need to specify
    parser.add_argument("--env", type=str, default="profile_control") # one of [base, profile_control]
    parser.add_argument("--task", type=str, default="betan_EFIT01") # betan_EFIT01
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda_id", type=int, default=4)
    return parser.parse_args()

class GymnasiumWrapper(gym.Env):
    """
    self environment -> Gymnasium compatiable
    """
    def __init__(self, custom_env):
        super().__init__()
        self.env = custom_env

        # 设置观测和动作空间
        # 先获取一个观测样本来确定正确的观测空间
        obs_sample = custom_env.reset()
        if isinstance(obs_sample, tuple):
            obs_sample = obs_sample[0]  # 处理新版gym返回(obs, info)的情况

        # 确保观测是一维的
        if hasattr(obs_sample, 'shape') and len(obs_sample.shape) > 1:
            obs_sample = obs_sample.flatten()

        obs_shape = obs_sample.shape if hasattr(obs_sample, 'shape') else (len(obs_sample),)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        if hasattr(custom_env, 'action_space'):
            self.action_space = custom_env.action_space
        else:
            # 假设连续动作空间
            action_dim = 4  # 根据你的设置
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

    def reset(self, seed=None, options=None):
        """重置环境"""
        if hasattr(self.env, 'seed') and seed is not None:
            self.env.seed(seed)

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}

        # 处理torch.Tensor观测
        if hasattr(obs, 'cpu'):  # 是torch.Tensor
            obs = obs.cpu().numpy()

        # 确保观测形状正确 - 展平为1维
        if hasattr(obs, 'shape') and len(obs.shape) > 1:
            obs = obs.flatten()

        return obs, info

    def step(self, action):
        """执行动作"""
        # 确保动作有正确的形状 - NFBaseEnv期望批次维度
        if len(action.shape) == 1:
            action = action.reshape(1, -1)  # 添加批次维度

        result = self.env.step(action)
        if len(result) == 4:
            # 旧格式: (obs, reward, done, info)
            obs, reward, done, info = result

            # 处理torch.Tensor观测
            if hasattr(obs, 'cpu'):  # 是torch.Tensor
                obs = obs.cpu().numpy()

            # 确保观测形状正确
            if hasattr(obs, 'shape') and len(obs.shape) > 1:
                obs = obs.flatten()  # 展平多维观测

            # 确保reward是标量
            if hasattr(reward, '__len__') and len(reward) == 1:
                reward = float(reward[0])
            elif hasattr(reward, '__len__'):
                reward = float(reward.sum())  # 如果是多维，求和
            else:
                reward = float(reward)

            # 确保done是标量
            if hasattr(done, '__len__'):
                done = bool(done.any())  # 如果任何一个为True，则为True
            else:
                done = bool(done)

            return obs, reward, done, False, info  # 添加truncated
        else:
            # 新格式: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result

            # 处理torch.Tensor观测
            if hasattr(obs, 'cpu'):  # 是torch.Tensor
                obs = obs.cpu().numpy()

            # 确保观测形状正确
            if hasattr(obs, 'shape') and len(obs.shape) > 1:
                obs = obs.flatten()  # 展平多维观测

            # 确保reward是标量
            if hasattr(reward, '__len__') and len(reward) == 1:
                reward = float(reward[0])
            elif hasattr(reward, '__len__'):
                reward = float(reward.sum())  # 如果是多维，求和
            else:
                reward = float(reward)

            # 确保terminated和truncated是标量
            if hasattr(terminated, '__len__'):
                terminated = bool(terminated.any())
            else:
                terminated = bool(terminated)

            if hasattr(truncated, '__len__'):
                truncated = bool(truncated.any())
            else:
                truncated = bool(truncated)

            return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()




def train(args=get_args()):
    # 设置设备
    args.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {args.device}")

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载数据和环境
    print("正在加载数据和环境...")
    offline_data, sa_processor, env, training_dyn_model_dir = get_rl_data_envs(
        args.env, args.task, args.device
    )
    
    # 获取状态和动作维度
    args.obs_shape = (offline_data['observations'].shape[1],)
    args.action_dim = offline_data['actions'].shape[1]
    args.state_dim = len(offline_data['state_idxs'])
    args.control_dim = len(offline_data['action_idxs'])
    
    print(f"观测维度: {args.obs_shape}")
    print(f"状态维度: {args.state_dim}")
    print(f"动作维度: {args.action_dim}")
    print(f"控制维度: {args.control_dim}")
    
    # 创建动力学模型
    print("正在加载动力学模型...")
    dynamics_model = EnsembleDynamicsModel(
        model_path=training_dyn_model_dir,
        device=args.device
    )
    
    # 创建动力学包装器
    termination_fn = env.is_done
    reward_fn = sa_processor.get_reward
    dynamics = EnsembleDynamics(
        dynamics_model,
        termination_fn,
        reward_fn,
        penalty_coef=0.5  # PPO不需要不确定性惩罚
    )
    
    # 创建PPO策略
    print("正在创建PPO策略...")

    # 调整PPO参数以确保callbacks能正常工作
    # 减小n_steps以便在较短的训练中能触发多次callbacks
    adjusted_n_steps = min(args.n_steps, args.total_timesteps // 4)  # 确保至少有4次rollout
    adjusted_n_steps = max(adjusted_n_steps, 256)  # 但不能太小

    print(f"调整n_steps从{args.n_steps}到{adjusted_n_steps}以适应总步数{args.total_timesteps}")

    policy = PPOPolicy(
        dynamics=dynamics,
        state_idxs=offline_data['state_idxs'],
        action_idxs=offline_data['action_idxs'],
        sa_processor=sa_processor,
        offline_data=offline_data,
        device=str(args.device),
        learning_rate=args.learning_rate,
        n_steps=adjusted_n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dims=args.hidden_dims  # 网络隐藏层维度
    )

    # 设置环境的最大episode长度
    policy.env.max_episode_length = args.max_episode_length
    
    # 创建日志目录
    output_dir = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))

    # 创建logger配置
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(output_dir, output_config)
    
    print(f"日志目录: {output_dir}")

    print("开始训练...")
    print(f"总训练步数: {args.total_timesteps}")
    print(f"评估频率: {args.eval_freq}")
    print(f"保存频率: {args.save_freq}")

    # 创建评估环境（使用真实环境base_env或profile_control_env）
    # 根据修改PPO方案.md的要求，评估环境必须是base_env或profile_control_env
    print("正在创建评估环境...")
    eval_env_raw = env  # 使用真实环境进行评估
    eval_env = GymnasiumWrapper(eval_env_raw)

    # 使用Monitor包装器来避免SB3的警告，并确保观测空间一致性
    from stable_baselines3.common.monitor import Monitor
    eval_env = Monitor(eval_env)

    # 创建callbacks
    callbacks = []

    # 评估回调 - 使用真实环境进行评估
    if args.eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(output_dir, "best_model"),
            log_path=os.path.join(output_dir, "evaluations"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        print(f"评估回调已配置: 每{args.eval_freq}步评估{args.eval_episodes}个episodes")

    # 检查点保存回调 - 保存SB3原始格式
    if args.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="ppo_model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        print(f"检查点回调已配置: 每{args.save_freq}步保存模型")

    # 权重转换和保存回调 - 保存兼容格式
    convert_save_callback = ConvertAndSaveCallback(
        save_path=os.path.join(output_dir, "checkpoint"),
        save_freq=args.save_freq,
        hidden_dims=args.hidden_dims,
        verbose=1
    )
    callbacks.append(convert_save_callback)
    print("权重转换回调已配置")

    # 组合所有callbacks
    callback_list = CallbackList(callbacks) if callbacks else None
    print(f"总共配置了{len(callbacks)}个callbacks")

    # 训练循环
    start_time = time.time()
    print("开始PPO训练...")

    policy.model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        progress_bar=True
    )

    # 训练完成
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}秒")

    # 保存最终模型
    final_model_path = os.path.join(output_dir, "model", "policy")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    policy.save(final_model_path)

    logger.close()
    print("训练完成！")


if __name__ == "__main__":
    train()
