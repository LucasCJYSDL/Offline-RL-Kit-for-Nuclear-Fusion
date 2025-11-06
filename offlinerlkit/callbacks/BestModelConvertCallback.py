import numpy as np
import torch
from typing import Dict, Union, Optional, Any
import gymnasium as gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from ..utils.convert_sb3_to_offlinerl_format import convert_sb3_to_offlinerl_format





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
