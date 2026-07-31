"""Replay buffer that seeds full-history TPNN model rollouts."""

from typing import Dict, Optional

import numpy as np

from offlinerlkit.buffer.buffer import ReplayBuffer
from rl_preparation.tpnn_history_store import TPNNHistoryStore


class TPNNReplayBuffer(ReplayBuffer):
    """Real-data buffer that returns history anchors instead of hidden states."""

    def __init__(
        self,
        *args,
        history_store: TPNNHistoryStore,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ):
        if seed is not None and rng is not None:
            raise ValueError("Provide seed or rng, not both.")
        super().__init__(*args, **kwargs)
        self.history_store = history_store
        self.rng = (
            np.random.default_rng(seed)
            if rng is None
            else rng
        )
        self._valid_anchor_cache = {}

    def load_dataset(
        self, dataset: Dict[str, np.ndarray], hidden: bool = False
    ) -> None:
        if hidden:
            raise ValueError(
                "TPNNReplayBuffer does not load RPNN hidden_states."
            )
        super().load_dataset(dataset, hidden=False)
        if "full_next_observations" in dataset:
            self.full_next_observations = np.asarray(
                dataset["full_next_observations"], dtype=self.obs_dtype
            )
        self.history_store.validate_offline_data(dataset)

    def sample_rollouts(
        self, batch_size: int, rollout_length: int
    ) -> Dict[str, np.ndarray]:
        """Sample exactly ``batch_size`` logical rollout anchors."""
        if (
            self.full_observations is None
            or self.full_actions is None
            or self.pre_actions is None
        ):
            raise RuntimeError("A real offline dataset must be loaded first.")
        if batch_size < 1 or rollout_length < 1:
            raise ValueError(
                "batch_size and rollout_length must both be positive."
            )
        if self._size != len(self.history_store):
            raise ValueError(
                "Replay buffer and TPNN history store have different sizes."
            )

        if rollout_length not in self._valid_anchor_cache:
            self._valid_anchor_cache[rollout_length] = np.flatnonzero(
                self.history_store.history_lengths[: self._size]
                + int(rollout_length)
                <= self.history_store.block_size
            )
        valid_anchors = self._valid_anchor_cache[rollout_length]
        if valid_anchors.size == 0:
            raise ValueError(
                "No offline anchors can fit the requested TPNN rollout."
            )
        anchor_indices = self.rng.choice(
            valid_anchors, size=int(batch_size), replace=True
        ).astype(np.int64)
        self.history_store.validate_rollout_capacity(
            anchor_indices, rollout_length
        )

        offsets = np.arange(rollout_length + 1, dtype=np.int64)
        index_matrix = anchor_indices[:, np.newaxis] + offsets[np.newaxis, :]
        index_matrix = np.minimum(index_matrix, self._size - 1)

        full_observations = self.full_observations[
            index_matrix[:, :-1]
        ].copy()
        full_actions = self.full_actions[index_matrix[:, :-1]].copy()
        pre_actions = self.pre_actions[index_matrix[:, :-1]].copy()
        time_steps_with_next = self.time_steps[index_matrix].copy()
        time_steps = time_steps_with_next[:, :-1]

        crossed_boundary = (
            time_steps_with_next[:, 1:]
            <= time_steps_with_next[:, :-1]
        )
        # Once an offline boundary is crossed, all later placeholders for that
        # logical trajectory are terminal as well.
        terminals = np.maximum.accumulate(
            crossed_boundary.astype(np.bool_), axis=1
        )

        return {
            "full_observations": full_observations,
            "full_actions": full_actions,
            "pre_actions": pre_actions,
            "time_steps": time_steps,
            "terminals": terminals,
            "context_anchor_indices": anchor_indices.copy(),
            "batch_idx_list": index_matrix.T.copy(),
        }
