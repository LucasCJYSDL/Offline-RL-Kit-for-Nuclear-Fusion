from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time
import os

class TensorBoardLoggingCallback(BaseCallback):
    """
    Simplified TensorBoard monitoring callback
    Only uses SB3 standard rollout/ep_rew_mean and rollout/ep_len_mean
    """
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = time.time()

    def _on_training_start(self) -> None:
        """Initialization at the start of training"""
        self.start_time = time.time()
        if self.verbose > 0:
            print("TensorBoard monitoring started")

    def _on_step(self) -> bool:
        """Monitoring logic called at each step"""
         # Only record basic training progress information
        if self.n_calls % self.log_freq == 0:
            self.logger.record("train/timesteps", self.num_timesteps)
            self.logger.record("train/fps", self.num_timesteps / (time.time() - self.start_time))

        return True


class FusionSpecificCallback(BaseCallback):
    """
    Monitoring callback specifically for nuclear fusion
    """
    def __init__(self, eval_env, logger=None, verbose=0, eval_freq=10000, n_eval_episodes=10):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.logger_custom = logger  
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
                
                if reward < -10:
                    plasma_disruptions += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if len(actions_history) > 1:
                actions_array = np.array(actions_history)
                action_changes = np.diff(actions_array, axis=0)
                smoothness = np.mean(np.std(action_changes, axis=0))
                control_smoothness.append(smoothness)
            
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
        
        
        if self.logger_custom is not None:
            self.logger_custom.logkv("eval/mean_reward", mean_reward)
            self.logger_custom.logkv("eval/std_reward", std_reward)
            self.logger_custom.logkv("eval/mean_ep_length", mean_length)
        
        # Nuclear fusion-specific metrics
        if control_smoothness:
            mean_smoothness = np.mean(control_smoothness)
            self.logger.record("fusion/control_smoothness", mean_smoothness)
            self.logger.record("fusion/control_quality_score", 1.0 / (1.0 + mean_smoothness))
            if self.logger_custom is not None:
                self.logger_custom.logkv("fusion/control_smoothness", mean_smoothness)
                self.logger_custom.logkv("fusion/control_quality_score", 1.0 / (1.0 + mean_smoothness))
        
        if tracking_errors:
            mean_tracking_error = np.mean(tracking_errors)
            self.logger.record("fusion/state_stability", mean_tracking_error)
            self.logger.record("fusion/stability_score", 1.0 / (1.0 + mean_tracking_error))
            if self.logger_custom is not None:
                self.logger_custom.logkv("fusion/state_stability", mean_tracking_error)
                self.logger_custom.logkv("fusion/stability_score", 1.0 / (1.0 + mean_tracking_error))
        
        disruption_rate = plasma_disruptions / self.n_eval_episodes
        self.logger.record("fusion/disruption_rate", disruption_rate)
        self.logger.record("fusion/stability_percentage", (1.0 - disruption_rate) * 100)
        if self.logger_custom is not None:
            self.logger_custom.logkv("fusion/disruption_rate", disruption_rate)
            self.logger_custom.logkv("fusion/stability_percentage", (1.0 - disruption_rate) * 100)
        
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.logger.record("eval/best_mean_reward", self.best_mean_reward)
            if self.logger_custom is not None:
                self.logger_custom.logkv("eval/best_mean_reward", self.best_mean_reward)
            if self.verbose > 0:
                print(f"New best evaluation reward: {mean_reward:.3f}")
        
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(mean_reward)
        
        # 在自定义logger中dump数据
        if self.logger_custom is not None:
            self.logger_custom.set_timestep(self.num_timesteps)
            self.logger_custom.dumpkvs()
        
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