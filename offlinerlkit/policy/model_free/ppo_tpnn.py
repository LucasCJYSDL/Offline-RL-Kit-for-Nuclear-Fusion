"""PPO trained in an RPNN-compatible zero-prefix TPNN environment."""

import os
from typing import Dict, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from offlinerlkit.dynamics.tpnn_full_history_dynamics import (
    TPNNFullHistoryDynamics,
)
from offlinerlkit.policy.base_policy import BasePolicy
from offlinerlkit.utils.convert_sb3_to_offlinerl_format import (
    convert_sb3_to_offlinerl_format,
)


class TPNNFusionPPOEnv(gym.Env):
    """Single-environment PPO rollout driven by a trained TPNN."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dynamics: TPNNFullHistoryDynamics,
        sa_processor,
        offline_data: Dict[str, np.ndarray],
        state_idxs: Sequence[int],
        action_idxs: Sequence[int],
        device: str = "cpu",
        max_episode_length: int = 150,
    ) -> None:
        super().__init__()
        if not isinstance(dynamics, TPNNFullHistoryDynamics):
            raise TypeError(
                "TPNNFusionPPOEnv requires TPNNFullHistoryDynamics."
            )
        if max_episode_length < 1:
            raise ValueError("max_episode_length must be positive.")

        self.dynamics = dynamics
        self.sa_processor = sa_processor
        self.offline_data = offline_data
        self.state_idxs = np.asarray(state_idxs, dtype=np.int64)
        self.action_idxs = np.asarray(action_idxs, dtype=np.int64)
        self.device = device
        self.max_episode_length = int(max_episode_length)

        self.dynamics.history_store.validate_offline_data(offline_data)
        observation_dim = int(offline_data["observations"].shape[1])
        action_dim = int(len(self.action_idxs))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        candidates = np.asarray(
            offline_data["traj_start_indices"], dtype=np.int64
        )
        candidates = self.dynamics.history_store.validate_anchor_indices(
            candidates
        )
        if self.max_episode_length > self.dynamics.model.block_size:
            raise ValueError(
                "The requested zero-prefix PPO episode length "
                f"{self.max_episode_length} exceeds TPNN block_size="
                f"{self.dynamics.model.block_size}."
            )
        # Match the original RPNN PPO start distribution exactly: sample
        # uniformly from every stored traj_start_indices entry.  The original
        # warm_start_amount/min_start_idx/max_start_idx CLI arguments were
        # parsed but never applied by FusionPPOEnv.
        self.valid_start_indices = candidates
        self._valid_start_set = set(self.valid_start_indices.tolist())

        self.current_full_state = None
        self.previous_action = None
        self.time_step = 0
        self.episode_length = 0
        self.episode_reward = 0.0
        self.current_global_idx = 0
        self.current_model_idx = 0

    def _choose_anchor(self, options: Optional[dict]) -> int:
        if options is not None and "anchor_index" in options:
            anchor = int(options["anchor_index"])
            if anchor not in self._valid_start_set:
                raise ValueError(
                    f"Requested PPO anchor {anchor} is not a valid start."
                )
            return anchor
        position = int(np.random.randint(0, len(self.valid_start_indices)))
        return int(self.valid_start_indices[position])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        anchor = self._choose_anchor(options)
        self.current_full_state = np.asarray(
            self.offline_data["full_observations"][anchor],
            dtype=np.float32,
        ).copy()
        self.previous_action = np.asarray(
            self.offline_data["pre_actions"][anchor], dtype=np.float32
        ).copy()
        self.time_step = int(self.offline_data["time_step"][anchor])
        self.episode_length = 0
        self.episode_reward = 0.0
        self.current_global_idx = anchor
        self.current_model_idx = int(
            np.random.randint(0, self.dynamics.model.num_ensemble)
        )

        # Match the original RPNN PPO reset semantics: the model memory starts
        # empty at every sampled anchor.  x[anchor] is created below from the
        # action selected by PPO; no offline prefix or future token is read.
        self.dynamics.reset(
            anchor_indices=np.asarray([anchor], dtype=np.int64),
            rollout_length=self.max_episode_length,
            restore_offline_prefix=False,
        )
        initial_lengths = self.dynamics.current_sequence_lengths
        if (
            initial_lengths is None
            or initial_lengths.shape != (1,)
            or int(initial_lengths[0]) != 0
        ):
            raise RuntimeError(
                "TPNN PPO reset must start with an empty model context."
            )
        observation = np.asarray(
            self.offline_data["observations"][anchor], dtype=np.float32
        ).copy()
        return observation, {
            "anchor_index": anchor,
            "initial_prefix_length": 0,
            "offline_anchor_history_length": int(
                self.dynamics.history_store.history_lengths[anchor]
            ),
            "dynamics_member_idx": self.current_model_idx,
        }

    def step(self, action):
        policy_action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        controlled_action = self.sa_processor.get_step_action(
            policy_action
        )[0]

        # Match the original RPNN FusionPPOEnv exactly: dimensions outside the
        # PPO action space are copied from the first offline full-action row
        # on every step. Controlled dimensions use the current PPO action.
        full_action = np.asarray(
            self.offline_data["full_actions"][0],
            dtype=np.float32,
        ).copy()
        full_action[self.action_idxs] = controlled_action

        max_data_idx = len(self.offline_data["terminals"]) - 1
        terminal_index = min(self.current_global_idx, max_data_idx)
        data_terminal = bool(
            self.offline_data["terminals"][terminal_index]
        )
        next_observation, reward, terminal, info = self.dynamics.step(
            cur_state=self.current_full_state.reshape(1, -1),
            pre_action=self.previous_action.reshape(1, -1),
            cur_action=full_action.reshape(1, -1),
            time_steps=np.asarray([self.time_step], dtype=np.int64),
            time_terminals=np.asarray([data_terminal], dtype=np.bool_),
            state_idxs=self.state_idxs,
            batch_idxs=np.asarray(
                [self.current_global_idx], dtype=np.int64
            ),
            fixed_model_idx=self.current_model_idx,
        )
        selected_member_indices = np.asarray(
            info["model_idxs"], dtype=np.int64
        ).reshape(-1)
        if (
            selected_member_indices.shape != (1,)
            or int(selected_member_indices[0]) != self.current_model_idx
        ):
            raise RuntimeError(
                "TPNN dynamics changed member inside a PPO episode."
            )
        context_lengths = np.asarray(
            info["context_sequence_lengths"], dtype=np.int64
        ).reshape(-1)
        expected_context_length = self.episode_length + 1
        if (
            bool(info.get("restored_offline_prefix", True))
            or context_lengths.shape != (1,)
            or int(context_lengths[0]) != expected_context_length
        ):
            raise RuntimeError(
                "TPNN PPO context must contain only tokens generated in the "
                "current synthetic episode."
            )
        info["dynamics_member_idx"] = self.current_model_idx

        self.current_full_state = info["next_full_observations"][0]
        self.previous_action = full_action
        self.time_step = int(info["next_time_steps"][0])
        self.episode_length += 1
        self.current_global_idx = min(
            self.current_global_idx + 1, max_data_idx
        )

        reward_scalar = float(reward[0, 0])
        self.episode_reward += reward_scalar
        selected_state = self.current_full_state[self.state_idxs]
        next_rl_observation = self.sa_processor.get_rl_state(
            selected_state.reshape(1, -1),
            np.asarray([self.current_global_idx], dtype=np.int64),
        )[0]

        # Match FusionPPOEnv: reaching max_episode_length is reported as a
        # terminal transition (not a Gymnasium time-limit truncation).  SB3
        # bootstraps truncated transitions differently, so this distinction
        # affects PPO targets.
        terminated = bool(
            terminal[0]
            or self.episode_length >= self.max_episode_length
        )
        truncated = False
        if terminated:
            info["episode"] = {
                "r": round(self.episode_reward, 6),
                "l": self.episode_length,
            }
        return (
            np.asarray(next_rl_observation, dtype=np.float32),
            reward_scalar,
            terminated,
            truncated,
            info,
        )


class TPNNPPOPolicy(BasePolicy):
    """Stable-Baselines3 PPO using zero-prefix TPNN training rollouts."""

    def __init__(
        self,
        dynamics: TPNNFullHistoryDynamics,
        state_idxs,
        action_idxs,
        sa_processor,
        offline_data,
        device: str = "cpu",
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
        pol_hidden_dims=None,
        val_hidden_dims=None,
        tensorboard_log: Optional[str] = None,
        max_episode_length: int = 150,
        **kwargs,
    ) -> None:
        super().__init__()
        pol_hidden_dims = (
            [256, 256]
            if pol_hidden_dims is None
            else list(pol_hidden_dims)
        )
        val_hidden_dims = (
            [256, 256]
            if val_hidden_dims is None
            else list(val_hidden_dims)
        )
        self.dynamics = dynamics
        self.state_idxs = state_idxs
        self.action_idxs = action_idxs
        self.sa_processor = sa_processor
        self.offline_data = offline_data
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.pol_hidden_dims = pol_hidden_dims
        self.val_hidden_dims = val_hidden_dims

        self.env = TPNNFusionPPOEnv(
            dynamics=dynamics,
            sa_processor=sa_processor,
            offline_data=offline_data,
            state_idxs=state_idxs,
            action_idxs=action_idxs,
            device=device,
            max_episode_length=max_episode_length,
        )
        self.vec_env = DummyVecEnv([lambda: self.env])

        policy_kwargs = {
            "net_arch": dict(
                pi=pol_hidden_dims, vf=val_hidden_dims
            ),
            "activation_fn": torch.nn.ReLU,
            "share_features_extractor": False,
            "squash_output": True,
            "log_std_init": 0.0,
        }
        policy_kwargs.update(kwargs.get("policy_kwargs", {}))
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
            verbose=1,
        )

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        observation = np.asarray(obs)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        action, _ = self.model.predict(
            observation, deterministic=deterministic
        )
        return action[0] if action.ndim > 1 else action

    def learn(self, total_timesteps: int, **kwargs) -> Dict[str, float]:
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        return {
            "total_timesteps": total_timesteps,
            "learning_rate": self.model.learning_rate,
        }

    def save(self, path: str) -> None:
        self.model.save(path)
        converted = convert_sb3_to_offlinerl_format(
            self.model.policy.state_dict(),
            self.pol_hidden_dims,
            self.val_hidden_dims,
        )
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        pth_path = path if path.endswith(".pth") else path + ".pth"
        torch.save(converted, pth_path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.vec_env)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return convert_sb3_to_offlinerl_format(
            self.model.policy.state_dict(),
            self.pol_hidden_dims,
            self.val_hidden_dims,
        )

    def get_actor(self):
        return self.model.policy
