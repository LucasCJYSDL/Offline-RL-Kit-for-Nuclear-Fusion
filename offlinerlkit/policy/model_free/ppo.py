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





def convert_sb3_to_offlinerl_format(sb3_state_dict: Dict[str, torch.Tensor], pol_hidden_dims: list, val_hidden_dims: list) -> Dict[str, torch.Tensor]:
    """
    Converts an SB3 state_dict to the offlinerlkit format.

    Args:
        sb3_state_dict: The state_dict from the SB3 model.
        pol_hidden_dims: A list of hidden layer dimensions for the policy network.
        val_hidden_dims: A list of hidden layer dimensions for the value network.

    Returns:
        offlinerl_state_dict: The converted state_dict.
    """
    offlinerl_state_dict = {}

    # 1. Convert policy network (mlp_extractor.policy_net -> actor_backbone)
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

    # 2. Convert action_net (the mean output layer)
    if 'action_net.weight' in sb3_state_dict:
        offlinerl_state_dict['actor.dist_net.mu.weight'] = sb3_state_dict[f'action_net.weight']
        offlinerl_state_dict['actor.dist_net.mu.bias'] = sb3_state_dict[f'action_net.bias']

    # 3. Convert log_std (independent parameter), not needed for deterministic evaluation
    if 'log_std' in sb3_state_dict:
        log_std = sb3_state_dict['log_std']
        if log_std.dim() == 2:
            avg_log_std = log_std.mean(dim=0, keepdim=True).T  # (action_dim, 1)
            offlinerl_state_dict['actor.dist_net.sigma_param'] = avg_log_std
        else:
            offlinerl_state_dict['actor.dist_net.sigma_param'] = log_std.unsqueeze(-1)


    # 4. Convert value network (mlp_extractor.value_net -> critic_backbone)
    layer_idx = 0
    for i, hidden_dim in enumerate(val_hidden_dims):
        weight_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.weight'
        bias_key_sb3 = f'mlp_extractor.value_net.{layer_idx}.bias'

        if weight_key_sb3 in sb3_state_dict:
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.weight'] = sb3_state_dict[weight_key_sb3]
            offlinerl_state_dict[f'critic1.backbone.model.{i*2}.bias'] = sb3_state_dict[bias_key_sb3]

        layer_idx += 2  # Linear + ReLU

    # 5. Convert value_net (the value output layer)
    if 'value_net.weight' in sb3_state_dict:
        offlinerl_state_dict['critic1.last.weight'] = sb3_state_dict['value_net.weight']
        offlinerl_state_dict['critic1.last.bias'] = sb3_state_dict['value_net.bias']

    # For compatibility, also create critic2 (usually the same as critic1)
    for key in list(offlinerl_state_dict.keys()):
        if key.startswith("critic1."):
            critic2_key = key.replace("critic1.", "critic2.")
            offlinerl_state_dict[critic2_key] = offlinerl_state_dict[key].clone()

    return offlinerl_state_dict




class BestModelConvertCallback(EventCallback):
    """
    Combines the best model saving logic of EvalCallback with the format conversion 
    functionality of ConvertAndSaveCallback.

    It only converts the model to the OfflineRL format and saves it to the checkpoint 
    folder when a new best model is found.
    """
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        save_path: str,
        pol_hidden_dims: list,
        val_hidden_dims: list,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(verbose=verbose)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.save_path = save_path
        self.pol_hidden_dims = pol_hidden_dims
        self.val_hidden_dims = val_hidden_dims

        # log path
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: list = []
        self.evaluations_timesteps: list = []
        self.evaluations_length: list = []
        self._is_success_buffer: list = []
        self.evaluations_successes: list = []

    def _init_callback(self) -> None:
        # Create the save directory
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Callback for logging the success rate"""
        info = locals_["info"]
        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            self._is_success_buffer = []

            
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Log to the logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Log the timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # If it is a new best model, save the converted weights
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")

                if self.save_path is not None:
                    # Get the state_dict of the SB3 model
                    sb3_state_dict = self.model.policy.state_dict()

                    # Convert to OfflineRL format
                    converted_state_dict = convert_sb3_to_offlinerl_format(
                        sb3_state_dict,
                        self.pol_hidden_dims,
                        self.val_hidden_dims
                    )

                    # Save the converted model to checkpoint/policy.pth
                    save_file = os.path.join(self.save_path, "policy.pth")
                    torch.save(converted_state_dict, save_file)

                    if self.verbose >= 1:
                        print(f"Best model saved to {save_file}")

                self.best_mean_reward = float(mean_reward)

        return continue_training



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






