"""Full-history TPNN dynamics for PPO/COMBO/MOPO model rollouts."""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from offlinerlkit.modules.tpnn_full_history_model import (
    TPNNFullHistoryModel,
)
from rl_preparation.tpnn_history_store import TPNNHistoryStore


class TPNNFullHistoryDynamics:
    """Manage per-trajectory Transformer histories outside the TPNN model."""

    def __init__(
        self,
        model: TPNNFullHistoryModel,
        history_store: TPNNHistoryStore,
        terminal_fn,
        reward_fn,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        prediction_mode: str = "selected",
        training_member_indices: Optional[
            Union[np.ndarray, Sequence[int]]
        ] = None,
        seed: Optional[int] = None,
        use_global_numpy_noise: bool = False,
    ) -> None:
        if model.input_dim != history_store.input_dim:
            raise ValueError(
                f"TPNN input_dim={model.input_dim} does not match history "
                f"input_dim={history_store.input_dim}."
            )
        if model.output_dim != history_store.output_dim:
            raise ValueError(
                f"TPNN output_dim={model.output_dim} does not match history "
                f"output_dim={history_store.output_dim}."
            )
        if model.block_size != history_store.block_size:
            raise ValueError(
                f"TPNN block_size={model.block_size} does not match history "
                f"block_size={history_store.block_size}."
            )
        if uncertainty_mode not in {
            "aleatoric",
            "pairwise-diff",
            "ensemble_std",
        }:
            raise ValueError(
                f"Unknown uncertainty mode: {uncertainty_mode!r}."
            )
        if prediction_mode not in {"selected", "all_for_uncertainty"}:
            raise ValueError(
                f"Unknown prediction mode: {prediction_mode!r}."
            )
        if not np.isfinite(penalty_coef) or penalty_coef < 0:
            raise ValueError(
                "penalty_coef must be finite and non-negative."
            )
        if prediction_mode == "selected" and penalty_coef:
            raise ValueError(
                "A non-zero penalty_coef requires "
                "prediction_mode='all_for_uncertainty'."
            )
        active_member_indices = model.validate_member_indices(
            training_member_indices
        )
        if (
            active_member_indices.size == 1
            and uncertainty_mode != "aleatoric"
        ):
            raise ValueError(
                f"{uncertainty_mode} requires multiple dynamics members; "
                "the active TPNN ensemble has one member."
            )
        if not isinstance(use_global_numpy_noise, (bool, np.bool_)):
            raise TypeError("use_global_numpy_noise must be boolean.")

        self.model = model
        self.history_store = history_store
        self.terminal_fn = terminal_fn
        self.reward_fn = reward_fn
        self._penalty_coef = float(penalty_coef)
        self._uncertainty_mode = uncertainty_mode
        self.prediction_mode = prediction_mode
        self.training_member_indices = active_member_indices
        self.num_training_members = int(active_member_indices.size)
        self.seed = None if seed is None else int(seed)
        self.use_global_numpy_noise = bool(use_global_numpy_noise)
        self.member_rng = self._make_rng(self.seed, stream_id=101)
        self.noise_rng = self._make_rng(self.seed, stream_id=211)
        self.action_rng = self._make_rng(self.seed, stream_id=307)

        self._anchor_indices: Optional[np.ndarray] = None
        self._synthetic_suffixes: Optional[np.ndarray] = None
        self._restore_offline_prefix: Optional[bool] = None
        self._last_prediction_stats: Dict = {}

    @staticmethod
    def _make_rng(
        seed: Optional[int], stream_id: int
    ) -> np.random.Generator:
        if seed is None:
            return np.random.default_rng()
        seed_sequence = np.random.SeedSequence(
            [int(seed), int(stream_id)]
        )
        return np.random.default_rng(seed_sequence)

    @property
    def penalty_coef(self) -> float:
        return self._penalty_coef

    @property
    def uncertainty_mode(self) -> str:
        return self._uncertainty_mode

    @property
    def last_prediction_stats(self) -> Dict:
        return dict(self._last_prediction_stats)

    def sample_member_indices(self, batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        positions = self.member_rng.integers(
            0,
            self.num_training_members,
            size=int(batch_size),
            dtype=np.int64,
        )
        return self.training_member_indices[positions]

    @property
    def context_size(self) -> int:
        if self._anchor_indices is None:
            return 0
        return int(self._anchor_indices.shape[0])

    @property
    def synthetic_steps(self) -> int:
        if self._synthetic_suffixes is None:
            return 0
        return int(self._synthetic_suffixes.shape[1])

    @property
    def current_sequence_lengths(self) -> Optional[np.ndarray]:
        if self._anchor_indices is None:
            return None
        if self._restore_offline_prefix:
            prefix_lengths = self.history_store.history_lengths[
                self._anchor_indices
            ]
        else:
            prefix_lengths = np.zeros(
                self._anchor_indices.shape[0], dtype=np.int64
            )
        return prefix_lengths + self.synthetic_steps

    def reset(
        self,
        anchor_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
        rollout_length: Optional[int] = None,
        restore_offline_prefix: bool = True,
    ) -> None:
        """Clear context or bind it to offline anchors.

        With ``restore_offline_prefix=True``, an anchor ``i`` restores
        ``x[episode_start:i]``.  With it disabled, the context starts empty,
        matching an RPNN whose hidden state is reset to zero at the anchor.
        In both cases, the first call to :meth:`step` constructs a fresh
        synthetic token for transition ``i`` using the policy action.
        """
        if not isinstance(restore_offline_prefix, (bool, np.bool_)):
            raise TypeError("restore_offline_prefix must be boolean.")
        self._anchor_indices = None
        self._synthetic_suffixes = None
        self._restore_offline_prefix = None
        if anchor_indices is None:
            return

        anchors = self.history_store.validate_anchor_indices(anchor_indices)
        if rollout_length is not None:
            rollout_length = int(rollout_length)
            if restore_offline_prefix:
                self.history_store.validate_rollout_capacity(
                    anchors,
                    rollout_length,
                    block_size=self.model.block_size,
                )
            elif rollout_length < 1:
                raise ValueError("rollout_length must be positive.")
            elif rollout_length > self.model.block_size:
                raise ValueError(
                    "A zero-prefix rollout of length "
                    f"{rollout_length} exceeds TPNN block_size="
                    f"{self.model.block_size}."
                )
        self._anchor_indices = anchors.copy()
        self._synthetic_suffixes = np.empty(
            (anchors.shape[0], 0, self.model.input_dim), dtype=np.float32
        )
        self._restore_offline_prefix = bool(restore_offline_prefix)

    def select_active(
        self, active_mask: Union[np.ndarray, Sequence[bool]]
    ) -> None:
        """Compact all context arrays with the rollout's active mask."""
        if self._anchor_indices is None or self._synthetic_suffixes is None:
            raise RuntimeError("TPNN context has not been initialized.")
        mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        if mask.shape[0] != self.context_size:
            raise ValueError(
                f"Active mask has {mask.shape[0]} rows, expected "
                f"{self.context_size}."
            )
        self._anchor_indices = self._anchor_indices[mask]
        self._synthetic_suffixes = self._synthetic_suffixes[mask]

    def _ensure_context(
        self,
        batch_size: int,
        batch_indices: Union[np.ndarray, Sequence[int]],
    ) -> None:
        if self._anchor_indices is None:
            # Lazy initialization is used by PPO.  batch_indices is the
            # current offline anchor only on the first synthetic step.
            self.reset(batch_indices)
        if self.context_size != batch_size:
            raise ValueError(
                f"TPNN context contains {self.context_size} trajectories but "
                f"step received {batch_size}."
            )

    def _validate_model_indices(
        self, model_indices: np.ndarray, batch_size: int
    ) -> np.ndarray:
        indices = np.asarray(model_indices, dtype=np.int64).reshape(-1)
        if indices.shape[0] != batch_size:
            raise ValueError(
                "Dynamics member indices do not match the rollout batch."
            )
        active = np.isin(indices, self.training_member_indices)
        if not np.all(active):
            raise ValueError(
                "TPNN member indices are outside the active training "
                f"ensemble {self.training_member_indices.tolist()}: "
                f"{indices[~active][:10].tolist()}."
            )
        return indices

    def _resolve_model_indices(
        self,
        batch_size: int,
        fixed_model_idx: Optional[int],
        fixed_model_idxs: Optional[np.ndarray],
    ) -> np.ndarray:
        if fixed_model_idx is not None and fixed_model_idxs is not None:
            raise ValueError(
                "Provide fixed_model_idx or fixed_model_idxs, not both."
            )
        if fixed_model_idxs is not None:
            indices = fixed_model_idxs
        elif fixed_model_idx is not None:
            indices = np.full(
                batch_size, int(fixed_model_idx), dtype=np.int64
            )
        else:
            indices = self.sample_member_indices(batch_size)
        return self._validate_model_indices(indices, batch_size)

    def _build_history_microbatch(
        self,
        rows: np.ndarray,
        current_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._anchor_indices is None or self._synthetic_suffixes is None:
            raise RuntimeError("TPNN context has not been initialized.")
        if not self._restore_offline_prefix:
            suffixes = self._synthetic_suffixes[rows]
            current = current_tokens[rows, np.newaxis, :]
            padded = np.concatenate([suffixes, current], axis=1)
            lengths = np.full(
                rows.shape[0], padded.shape[1], dtype=np.int64
            )
            return padded, lengths
        return self.history_store.build_padded_batch(
            self._anchor_indices[rows],
            self._synthetic_suffixes[rows],
            current_tokens[rows],
            block_size=self.model.block_size,
        )

    def _prediction_sequence_lengths(self) -> np.ndarray:
        """Lengths after appending the current synthetic token once."""
        current_lengths = self.current_sequence_lengths
        if current_lengths is None:
            raise RuntimeError("TPNN context has not been initialized.")
        prediction_lengths = np.asarray(
            current_lengths + 1, dtype=np.int64
        )
        if np.any(prediction_lengths > self.model.block_size):
            raise ValueError(
                "Current synthetic token would exceed TPNN block_size."
            )
        return prediction_lengths

    @staticmethod
    def _microbatch_diagnostics(
        batch_rows: Sequence[np.ndarray],
        sequence_lengths: np.ndarray,
    ) -> Dict:
        batch_sizes = [int(rows.size) for rows in batch_rows]
        max_lengths = [
            int(sequence_lengths[rows].max()) for rows in batch_rows
        ]
        padded_tokens = [
            batch_size * max_length
            for batch_size, max_length in zip(batch_sizes, max_lengths)
        ]
        attention_cells = [
            batch_size * max_length * max_length
            for batch_size, max_length in zip(batch_sizes, max_lengths)
        ]
        return {
            "microbatch_count": len(batch_rows),
            "microbatch_sizes": batch_sizes,
            "microbatch_max_lengths": max_lengths,
            "padded_tokens_total": int(sum(padded_tokens)),
            "attention_cells_total": int(sum(attention_cells)),
            "largest_microbatch": max(batch_sizes, default=0),
            "longest_sequence": max(max_lengths, default=0),
        }

    def _predict_current_selected(
        self,
        current_tokens: np.ndarray,
        model_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict only assigned members, grouped over the logical batch."""
        batch_size = int(current_tokens.shape[0])
        means = np.empty(
            (batch_size, self.model.output_dim), dtype=np.float32
        )
        stds = np.empty_like(means)
        member_counts = np.bincount(
            model_indices, minlength=self.model.num_ensemble
        )
        forward_calls = np.zeros(
            self.model.num_ensemble, dtype=np.int64
        )
        sequence_lengths = self._prediction_sequence_lengths()
        all_batch_rows = []
        for member_idx in self.training_member_indices:
            member_idx = int(member_idx)
            member_rows = np.flatnonzero(model_indices == member_idx)
            member_batches = self.model.plan_microbatches(
                sequence_lengths[member_rows]
            )
            for local_rows in member_batches:
                rows = member_rows[local_rows]
                padded, lengths = self._build_history_microbatch(
                    rows, current_tokens
                )
                member_means, member_stds = (
                    self.model.predict_member_padded(
                        padded,
                        lengths,
                        member_idx=member_idx,
                    )
                )
                means[rows] = member_means
                stds[rows] = member_stds
                forward_calls[member_idx] += 1
                all_batch_rows.append(rows)
        self._last_prediction_stats = {
            "prediction_mode": "selected",
            "logical_batch_size": batch_size,
            "member_counts": member_counts.tolist(),
            "forward_calls_by_member": forward_calls.tolist(),
            "forward_calls_total": int(forward_calls.sum()),
            "max_microbatch_rows": self.model.microbatch_size,
            "token_budget": self.model.token_budget,
            "attention_budget": self.model.attention_budget,
            "training_member_indices": (
                self.training_member_indices.tolist()
            ),
            **self._microbatch_diagnostics(
                all_batch_rows, sequence_lengths
            ),
        }
        return means, stds

    @staticmethod
    def _uncertainty_from_all_members(
        next_state_means: np.ndarray,
        delta_stds: np.ndarray,
        uncertainty_mode: str,
    ) -> np.ndarray:
        if uncertainty_mode == "aleatoric":
            uncertainty = np.amax(
                np.linalg.norm(delta_stds, axis=2), axis=0
            )
        elif uncertainty_mode == "pairwise-diff":
            centered = next_state_means - next_state_means.mean(
                axis=0, keepdims=True
            )
            uncertainty = np.amax(
                np.linalg.norm(centered, axis=2), axis=0
            )
        else:
            uncertainty = np.sqrt(
                next_state_means.var(axis=0).mean(axis=1)
            )
        return uncertainty.astype(np.float32, copy=False)

    def _predict_current_all_streaming(
        self,
        current_tokens: np.ndarray,
        states: np.ndarray,
        model_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ensemble uncertainty without retaining full-batch M×B×D."""
        batch_size = int(current_tokens.shape[0])
        selected_means = np.empty(
            (batch_size, self.model.output_dim), dtype=np.float32
        )
        selected_stds = np.empty_like(selected_means)
        uncertainty = np.empty(batch_size, dtype=np.float32)
        member_counts = np.bincount(
            model_indices, minlength=self.model.num_ensemble
        )
        sequence_lengths = self._prediction_sequence_lengths()
        batch_rows = self.model.plan_microbatches(sequence_lengths)
        forward_calls = 0
        for rows in batch_rows:
            padded, lengths = self._build_history_microbatch(
                rows, current_tokens
            )
            (
                local_selected_means,
                local_selected_stds,
                local_uncertainty,
            ) = self.model.predict_selected_with_uncertainty(
                padded_inputs=padded,
                sequence_lengths=lengths,
                states=states[rows],
                member_indices=model_indices[rows],
                uncertainty_mode=self._uncertainty_mode,
                ensemble_member_indices=self.training_member_indices,
            )
            selected_means[rows] = local_selected_means
            selected_stds[rows] = local_selected_stds
            uncertainty[rows] = local_uncertainty
            forward_calls += self.num_training_members
        microbatch_stats = self._microbatch_diagnostics(
            batch_rows, sequence_lengths
        )
        self._last_prediction_stats = {
            "prediction_mode": "all_for_uncertainty",
            "logical_batch_size": batch_size,
            "member_counts": member_counts.tolist(),
            "forward_calls_by_member": [
                int(len(batch_rows))
                if member_idx in self.training_member_indices
                else 0
                for member_idx in range(self.model.num_ensemble)
            ],
            "forward_calls_total": int(forward_calls),
            "max_microbatch_rows": self.model.microbatch_size,
            "token_budget": self.model.token_budget,
            "attention_budget": self.model.attention_budget,
            "training_member_indices": (
                self.training_member_indices.tolist()
            ),
            **microbatch_stats,
        }
        return selected_means, selected_stds, uncertainty

    def _predict_current(
        self,
        current_tokens: np.ndarray,
        states: np.ndarray,
        model_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.prediction_mode == "selected":
            means, stds = self._predict_current_selected(
                current_tokens, model_indices
            )
            return means, stds, None
        means, stds, uncertainty = self._predict_current_all_streaming(
            current_tokens,
            states,
            model_indices,
        )
        return means, stds, uncertainty

    @staticmethod
    def _gaussian_log_probability(
        samples: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        safe_stds = np.maximum(stds, np.finfo(np.float32).tiny)
        normalized = (samples - means) / safe_stds
        elementwise = (
            -0.5 * np.square(normalized)
            - np.log(safe_stds)
            - 0.5 * np.log(2.0 * np.pi)
        )
        return elementwise.sum(axis=-1)

    def step(
        self,
        cur_state: np.ndarray,
        pre_action: np.ndarray,
        cur_action: np.ndarray,
        time_steps: np.ndarray,
        time_terminals: np.ndarray,
        state_idxs: Union[np.ndarray, Sequence[int]],
        batch_idxs: Union[np.ndarray, Sequence[int]],
        fixed_model_idx: Optional[int] = None,
        fixed_model_idxs: Optional[np.ndarray] = None,
        epsilon: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Append one policy-generated token and predict the next state."""
        states = np.asarray(cur_state, dtype=np.float32)
        previous_actions = np.asarray(pre_action, dtype=np.float32)
        current_actions = np.asarray(cur_action, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError("cur_state must have shape [batch, state_dim].")
        batch_size = int(states.shape[0])
        if states.shape[1] != self.model.output_dim:
            raise ValueError(
                f"cur_state has width {states.shape[1]}, expected "
                f"{self.model.output_dim}."
            )
        if previous_actions.shape != current_actions.shape:
            raise ValueError(
                "pre_action and cur_action must have identical shapes."
            )
        if previous_actions.shape[0] != batch_size:
            raise ValueError("Action batch does not match state batch.")

        batch_indices = np.asarray(batch_idxs, dtype=np.int64).reshape(-1)
        if batch_indices.shape[0] != batch_size:
            raise ValueError("batch_idxs does not match the rollout batch.")
        self._ensure_context(batch_size, batch_indices)
        model_indices = self._resolve_model_indices(
            batch_size,
            fixed_model_idx=fixed_model_idx,
            fixed_model_idxs=fixed_model_idxs,
        )

        current_tokens = np.concatenate(
            [
                states,
                previous_actions,
                current_actions - previous_actions,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        if current_tokens.shape[1] != self.model.input_dim:
            raise ValueError(
                f"Constructed token width is {current_tokens.shape[1]}, "
                f"expected {self.model.input_dim}."
            )

        delta_means, delta_stds, uncertainty = self._predict_current(
            current_tokens,
            states,
            model_indices,
        )

        next_state_means = delta_means + states
        # Draw noise only after all microbatches.  The default path draws
        # [B,D]; PPO compatibility mode draws [M,B,D] and gathers [B,D].
        # Either path is independent of inference grouping/microbatching.
        if epsilon is None:
            if self.use_global_numpy_noise:
                # PPO compatibility mode: the original RPNN first samples
                # noise for every ensemble member with the global NumPy
                # stream, then gathers the episode's fixed member.  Preserve
                # that RNG consumption while retaining selected-member TPNN
                # inference.
                ensemble_noise = np.random.normal(
                    size=(
                        self.model.num_ensemble,
                        *next_state_means.shape,
                    )
                )
                noise = ensemble_noise[
                    model_indices, np.arange(batch_size)
                ].astype(np.float32)
            else:
                noise = self.noise_rng.standard_normal(
                    size=next_state_means.shape
                ).astype(np.float32)
        else:
            noise = np.asarray(epsilon, dtype=np.float32)
            if noise.shape != next_state_means.shape:
                raise ValueError(
                    f"epsilon has shape {noise.shape}; expected "
                    f"{next_state_means.shape}."
                )
        samples = (
            next_state_means + noise * delta_stds
        ).astype(
            np.float32,
            copy=False,
        )
        next_time_steps = (
            np.asarray(time_steps).reshape(-1) + 1
        )

        state_indices = np.asarray(state_idxs, dtype=np.int64)
        next_observations = samples[:, state_indices]
        rewards = np.asarray(
            self.reward_fn(next_observations, batch_indices),
            dtype=np.float32,
        ).reshape(-1, 1)

        terminals = np.asarray(
            self.terminal_fn(next_time_steps), dtype=np.bool_
        ).reshape(-1)
        data_terminals = np.asarray(
            time_terminals, dtype=np.bool_
        ).reshape(-1)
        if terminals.shape[0] != batch_size:
            raise ValueError("terminal_fn returned a wrong batch size.")
        if data_terminals.shape[0] != batch_size:
            raise ValueError("time_terminals has a wrong batch size.")
        terminals = terminals | data_terminals

        # Commit the newly generated token only after all fallible downstream
        # validation has succeeded.  A caller can safely retry a failed step
        # without duplicating the current token in its history.
        self._synthetic_suffixes = np.concatenate(
            [self._synthetic_suffixes, current_tokens[:, np.newaxis, :]],
            axis=1,
        )

        info = {
            "next_full_observations": samples.copy(),
            "next_time_steps": next_time_steps.copy(),
            "dyn_net_input": current_tokens.copy(),
            "model_idxs": model_indices.copy(),
            "dyn_samples": samples - states,
            "raw_reward": rewards.copy(),
            "prediction_mode": self.prediction_mode,
            "prediction_stats": self.last_prediction_stats,
            "context_sequence_lengths": self.current_sequence_lengths.copy(),
            "restored_offline_prefix": bool(
                self._restore_offline_prefix
            ),
        }
        info["log_dyn_probs"] = self._gaussian_log_probability(
            samples, next_state_means, delta_stds
        )

        if uncertainty is not None:
            penalty = uncertainty.reshape(-1, 1)
            info["uncertainty"] = penalty.copy()
        if self._penalty_coef:
            rewards = rewards - self._penalty_coef * penalty
            info["penalty"] = penalty

        return next_observations, rewards, terminals, info
