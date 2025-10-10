
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import os

class TensorBoardLoggingCallback(BaseCallback):
    """
    Enhanced TensorBoard monitoring callback - designed specifically for nuclear fusion PPO
    """
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_mean_reward = -np.inf
        self.start_time = time.time()
        
    def _on_training_start(self) -> None:
        """Initialization at the start of training"""
        self.start_time = time.time()
        if self.verbose > 0:
            print(" TensorBoard monitoring started")
        
    def _on_step(self) -> bool:
        """Monitoring logic called at each step"""
        # Record detailed information every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Record training progress
            self.logger.record("train/timesteps", self.num_timesteps)
            self.logger.record("train/iterations", self.n_calls // self.log_freq)
            self.logger.record("train/fps", self.num_timesteps / (time.time() - self.start_time))
            
            # Record environment statistics
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Attempt to retrieve environment statistics
                    env_stats = self.training_env.get_attr('get_episode_rewards')
                    if env_stats and env_stats[0]:
                        recent_rewards = env_stats[0][-10:]  # Last 10 episodes
                        self.logger.record("env/recent_mean_reward", np.mean(recent_rewards))
                        self.logger.record("env/recent_reward_std", np.std(recent_rewards))
                except:
                    pass
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Record information at the end of each rollout"""
        # Record rollout statistics
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            # 收集所有episode的奖励信息
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info and ep_info['r'] not in self.episode_rewards:
                    self.episode_rewards.append(ep_info['r'])
                if 'l' in ep_info and ep_info['l'] not in self.episode_lengths:
                    self.episode_lengths.append(ep_info['l'])

            # 记录 episode_reward_mean (这是关键指标!)
            if len(self.episode_rewards) > 0:
                episode_reward_mean = np.mean(self.episode_rewards[-100:])  # 最近100个episodes
                self.logger.record("rollout/episode_reward_mean", episode_reward_mean)

                # 也记录标准差
                if len(self.episode_rewards) > 1:
                    episode_reward_std = np.std(self.episode_rewards[-100:])
                    self.logger.record("rollout/episode_reward_std", episode_reward_std)

            # 记录episode长度的平均值
            if len(self.episode_lengths) > 0:
                episode_length_mean = np.mean(self.episode_lengths[-100:])
                self.logger.record("rollout/episode_length_mean", episode_length_mean)

            # 记录最后一个episode的信息（保持原有功能）
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.logger.record("rollout/ep_rew_mean", ep_info['r'])
            if 'l' in ep_info:
                self.logger.record("rollout/ep_len_mean", ep_info['l'])

        # Record reward trends
        if len(self.episode_rewards) > 10:
            recent_mean = np.mean(self.episode_rewards[-10:])
            self.logger.record("rollout/recent_reward_trend", recent_mean - self.last_mean_reward)
            self.last_mean_reward = recent_mean

        # Record policy statistics
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            for key, value in self.model.logger.name_to_value.items():
                if 'train/' in key:
                    self.logger.record(key, value)


class FusionSpecificCallback(BaseCallback):
    """
    Monitoring callback specifically for nuclear fusion
    """
    def __init__(self, eval_env, verbose=0, eval_freq=10000, n_eval_episodes=10):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_policy()
        return True
    
    def _evaluate_policy(self):
        """Evaluate the policy and record nuclear fusion-specific metrics"""
        episode_rewards = []
        episode_lengths = []
        tracking_errors = []
        control_smoothness = []
        plasma_disruptions = 0
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            actions_history = []
            states_history = []
            
            while not done and episode_length < 200:
                action, _ = self.model.predict(obs, deterministic=True)
                actions_history.append(action.copy())
                states_history.append(obs.copy())
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                # Detect plasma disruptions (based on sharp reward drops)
                if reward < -10:  # Threshold can be adjusted based on actual conditions
                    plasma_disruptions += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate control smoothness
            if len(actions_history) > 1:
                actions_array = np.array(actions_history)
                action_changes = np.diff(actions_array, axis=0)
                smoothness = np.mean(np.std(action_changes, axis=0))
                control_smoothness.append(smoothness)
            
            # Calculate state stability as a proxy for tracking error
            if len(states_history) > 1:
                states_array = np.array(states_history)
                state_stability = np.mean(np.std(states_array, axis=0))
                tracking_errors.append(state_stability)
        
        # Record to TensorBoard
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/mean_ep_length", mean_length)
        
        # Nuclear fusion-specific metrics
        if control_smoothness:
            mean_smoothness = np.mean(control_smoothness)
            self.logger.record("fusion/control_smoothness", mean_smoothness)
            self.logger.record("fusion/control_quality_score", 1.0 / (1.0 + mean_smoothness))
        
        if tracking_errors:
            mean_tracking_error = np.mean(tracking_errors)
            self.logger.record("fusion/state_stability", mean_tracking_error)
            self.logger.record("fusion/stability_score", 1.0 / (1.0 + mean_tracking_error))
        
        disruption_rate = plasma_disruptions / self.n_eval_episodes
        self.logger.record("fusion/disruption_rate", disruption_rate)
        self.logger.record("fusion/stability_percentage", (1.0 - disruption_rate) * 100)
        
        # Record performance improvement
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.logger.record("eval/best_mean_reward", self.best_mean_reward)
            if self.verbose > 0:
                print(f"New best evaluation reward: {mean_reward:.3f}")
        
        # Save evaluation history
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(mean_reward)
        
        if self.verbose > 0:
            print(f"Evaluation@{self.num_timesteps} steps: Reward={mean_reward:.3f}±{std_reward:.3f}, "
                  f"Length={mean_length:.1f}, Disruption Rate={disruption_rate*100:.1f}%")


class TrainingProgressCallback(BaseCallback):
    """
    Training progress monitoring callback
    """
    def __init__(self, save_path, verbose=0, log_freq=5000):
        super().__init__(verbose)
        self.save_path = save_path
        self.log_freq = log_freq
        self.training_progress = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            progress_data = {
                'timestep': self.num_timesteps,
                'iteration': self.n_calls // self.log_freq,
                'time': time.time(),
                'fps': self.num_timesteps / (time.time() - getattr(self, 'start_time', time.time()))
            }
            
            # Save progress data
            self.training_progress.append(progress_data)
            
            # Periodically save to file
            if len(self.training_progress) % 10 == 0:
                import json
                progress_file = os.path.join(self.save_path, "training_progress.json")
                os.makedirs(os.path.dirname(progress_file), exist_ok=True)
                with open(progress_file, 'w') as f:
                    json.dump(self.training_progress, f, indent=2)
        
        return True
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()