"""Shared full-history TPNN rollout implementation for COMBO and MOPO."""

from collections import defaultdict
import time
from typing import Dict, Tuple

import numpy as np

from offlinerlkit.dynamics.tpnn_full_history_dynamics import (
    TPNNFullHistoryDynamics,
)


class TPNNRolloutMixin:
    """Replace RPNN hidden-state rollouts with explicit TPNN histories.

    Classes using this mixin are expected to expose ``dynamics``,
    ``state_idxs``, ``action_idxs``, ``sa_processor`` and ``select_action``.
    COMBO's optional uniform rollout is honored when the corresponding
    attributes are present.
    """

    def _tpnn_select_rollout_actions(
        self, observations: np.ndarray
    ) -> np.ndarray:
        if getattr(self, "_uniform_rollout", False):
            action_space = getattr(self, "action_space", None)
            if action_space is None:
                raise AttributeError(
                    "Uniform TPNN rollout requires self.action_space."
                )
            return self.dynamics.action_rng.uniform(
                action_space.low,
                action_space.high,
                size=(len(observations), action_space.shape[0]),
            ).astype(np.float32)
        return np.asarray(self.select_action(observations), dtype=np.float32)

    def rollout(
        self, init_samples: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if not isinstance(self.dynamics, TPNNFullHistoryDynamics):
            raise TypeError(
                "TPNNRolloutMixin requires TPNNFullHistoryDynamics."
            )
        required = {
            "full_observations",
            "full_actions",
            "pre_actions",
            "time_steps",
            "terminals",
            "context_anchor_indices",
            "batch_idx_list",
        }
        missing = sorted(required - set(init_samples.keys()))
        if missing:
            raise KeyError(f"TPNN rollout input is missing: {missing}")

        rollout_length = int(init_samples["full_observations"].shape[1])
        logical_batch_size = int(
            init_samples["full_observations"].shape[0]
        )
        anchor_indices = np.asarray(
            init_samples["context_anchor_indices"], dtype=np.int64
        ).reshape(-1)
        if anchor_indices.shape[0] != logical_batch_size:
            raise ValueError(
                "context_anchor_indices does not match logical rollout batch."
            )

        self.dynamics.reset(
            anchor_indices=anchor_indices,
            rollout_length=rollout_length,
        )
        active_rows = np.arange(logical_batch_size, dtype=np.int64)
        full_observations = np.asarray(
            init_samples["full_observations"][:, 0], dtype=np.float32
        )
        pre_actions = np.asarray(
            init_samples["pre_actions"][:, 0], dtype=np.float32
        )
        offline_full_actions = np.asarray(
            init_samples["full_actions"], dtype=np.float32
        )
        if offline_full_actions.shape[:2] != (
            logical_batch_size,
            rollout_length,
        ):
            raise ValueError(
                "full_actions does not match the logical rollout batch and "
                "rollout length."
            )
        if offline_full_actions.shape[2:] != pre_actions.shape[1:]:
            raise ValueError(
                "full_actions and pre_actions have different action widths."
            )
        controlled_action_indices = np.asarray(
            self.action_idxs, dtype=np.int64
        ).reshape(-1)
        full_action_width = int(pre_actions.shape[1])
        if (
            np.any(controlled_action_indices < 0)
            or np.any(controlled_action_indices >= full_action_width)
        ):
            raise IndexError(
                "A controlled action index is outside the full action width."
            )
        uncontrolled_action_mask = np.ones(
            full_action_width, dtype=np.bool_
        )
        uncontrolled_action_mask[controlled_action_indices] = False
        time_steps = np.asarray(
            init_samples["time_steps"][:, 0]
        ).reshape(-1)

        fixed_model_idxs = self.dynamics.sample_member_indices(
            logical_batch_size
        )
        transitions = defaultdict(list)
        rewards_seen = []
        active_trajectories_by_step = []
        prediction_stats_by_step = []
        dynamics_seconds_by_step = []
        rollout_started_at = time.perf_counter()

        for step in range(rollout_length):
            active_trajectories_by_step.append(int(active_rows.shape[0]))
            current_batch_indices = np.asarray(
                init_samples["batch_idx_list"][step, active_rows],
                dtype=np.int64,
            )
            next_batch_indices = np.asarray(
                init_samples["batch_idx_list"][step + 1, active_rows],
                dtype=np.int64,
            )
            observations = full_observations[:, self.state_idxs]
            observations = self.sa_processor.get_rl_state(
                observations, current_batch_indices
            )
            actions = self._tpnn_select_rollout_actions(observations)
            step_actions = self.sa_processor.get_step_action(actions)

            # Build a hybrid synthetic action exactly as in the original
            # fusion MOPO/COMBO rollouts and the RPNN evaluation env:
            # policy-controlled dimensions come from the current policy,
            # while non-controlled dimensions follow the shot's exogenous
            # offline actuator schedule.
            full_actions = pre_actions.copy()
            if np.any(uncontrolled_action_mask):
                full_actions[:, uncontrolled_action_mask] = (
                    offline_full_actions[
                        active_rows, step
                    ][:, uncontrolled_action_mask]
                )
            full_actions[:, controlled_action_indices] = step_actions
            data_terminals = np.asarray(
                init_samples["terminals"][active_rows, step],
                dtype=np.bool_,
            ).reshape(-1)

            dynamics_started_at = time.perf_counter()
            next_observations, rewards, terminals, info = self.dynamics.step(
                    full_observations,
                    pre_actions,
                    full_actions,
                    time_steps,
                    data_terminals,
                    self.state_idxs,
                    current_batch_indices,
                    fixed_model_idxs=fixed_model_idxs,
                )
            dynamics_seconds_by_step.append(
                time.perf_counter() - dynamics_started_at
            )
            prediction_stats_by_step.append(
                dict(info["prediction_stats"])
            )
            next_rl_observations = self.sa_processor.get_rl_state(
                next_observations, next_batch_indices
            )

            transitions["obss"].append(np.asarray(observations))
            transitions["next_obss"].append(
                np.asarray(next_rl_observations)
            )
            transitions["actions"].append(actions)
            transitions["rewards"].append(rewards)
            transitions["terminals"].append(
                np.asarray(terminals).reshape(-1, 1)
            )
            rewards_seen.append(rewards.reshape(-1))

            active_mask = ~np.asarray(terminals, dtype=np.bool_).reshape(-1)
            if not np.any(active_mask):
                break

            self.dynamics.select_active(active_mask)
            active_rows = active_rows[active_mask]
            fixed_model_idxs = fixed_model_idxs[active_mask]
            full_observations = info["next_full_observations"][active_mask]
            pre_actions = full_actions[active_mask]
            time_steps = np.asarray(info["next_time_steps"])[active_mask]

        rollout_transitions = {
            key: np.concatenate(values, axis=0)
            for key, values in transitions.items()
        }
        reward_values = (
            np.concatenate(rewards_seen)
            if rewards_seen
            else np.asarray([], dtype=np.float32)
        )
        total_forward_calls = sum(
            int(stats["forward_calls_total"])
            for stats in prediction_stats_by_step
        )
        full_rollout_seconds = time.perf_counter() - rollout_started_at
        dynamics_seconds = float(sum(dynamics_seconds_by_step))
        num_transitions = int(
            rollout_transitions.get("obss", np.empty((0,))).shape[0]
        )
        # Detailed lists are kept off rollout_info because MBPolicyTrainer
        # sends every rollout_info value to a scalar logger.
        self.last_tpnn_rollout_diagnostics = {
            "logical_rollout_batch": logical_batch_size,
            "requested_rollout_length": rollout_length,
            "executed_rollout_steps": len(active_trajectories_by_step),
            "active_trajectories_by_step": (
                active_trajectories_by_step
            ),
            "prediction_stats_by_step": prediction_stats_by_step,
            "forward_calls_total": total_forward_calls,
            "dynamics_seconds_by_step": dynamics_seconds_by_step,
            "dynamics_seconds": dynamics_seconds,
            "full_rollout_seconds": full_rollout_seconds,
            "transitions_per_second": (
                num_transitions / full_rollout_seconds
                if full_rollout_seconds > 0
                else float("inf")
            ),
        }
        return rollout_transitions, {
            "num_transitions": num_transitions,
            "reward_mean": (
                float(reward_values.mean())
                if reward_values.size
                else float("nan")
            ),
            "logical_rollout_batch": logical_batch_size,
            "tpnn_forward_calls": total_forward_calls,
            "tpnn_dynamics_seconds": dynamics_seconds,
            "model_rollout_seconds": full_rollout_seconds,
            "model_rollout_transitions_per_second": (
                num_transitions / full_rollout_seconds
                if full_rollout_seconds > 0
                else float("inf")
            ),
        }
