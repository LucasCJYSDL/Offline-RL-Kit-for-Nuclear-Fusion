"""Benchmark one full-scale Ensemble TPNN model rollout without training."""

import argparse
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from offlinerlkit.buffer.tpnn_replay_buffer import TPNNReplayBuffer  # noqa: E402
from offlinerlkit.dynamics.tpnn_full_history_dynamics import (  # noqa: E402
    TPNNFullHistoryDynamics,
)
from offlinerlkit.modules.tpnn_full_history_model import (  # noqa: E402
    TPNNFullHistoryModel,
)
from offlinerlkit.policy.model_based.tpnn_rollout_mixin import (  # noqa: E402
    TPNNRolloutMixin,
)
from rl_preparation.get_rl_data_envs import get_rl_data_envs  # noqa: E402
from rl_preparation.tpnn_history_store import (  # noqa: E402
    DEFAULT_TPNN_DATA_DIR,
    TPNNHistoryStore,
)


DEFAULT_TPNN_MODEL_DIR = (
    "/data/models/tpnn_noshape_gas_benchmark_ensemble25_step_two_logvar"
)


class _ZeroActionRolloutPolicy(TPNNRolloutMixin):
    """Exercise production rollout plumbing without RL gradient updates."""

    def __init__(
        self,
        dynamics,
        state_idxs,
        action_idxs,
        sa_processor,
        action_dim,
    ):
        self.dynamics = dynamics
        self.state_idxs = np.asarray(state_idxs, dtype=np.int64)
        self.action_idxs = np.asarray(action_idxs, dtype=np.int64)
        self.sa_processor = sa_processor
        self.action_dim = int(action_dim)

    def select_action(self, observations):
        return np.zeros(
            (len(observations), self.action_dim), dtype=np.float32
        )


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a production-size full-history Ensemble TPNN rollout."
        )
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["selected", "all_for_uncertainty"],
        default="selected",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=7)
    parser.add_argument("--tpnn-microbatch-size", type=int, default=4096)
    parser.add_argument("--tpnn-token-budget", type=int, default=None)
    parser.add_argument("--tpnn-attention-budget", type=int, default=None)
    parser.add_argument("--warmup-batch-size", type=int, default=1024)
    parser.add_argument("--warmup-rollout-length", type=int, default=1)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["aleatoric", "pairwise-diff", "ensemble_std"],
        default="aleatoric",
    )
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--expected-tpnn-members", type=int, default=25)
    parser.add_argument(
        "--training-member-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional fixed source-member subset used for generation and "
            "uncertainty."
        ),
    )
    parser.add_argument(
        "--tpnn-model-dir", type=str, default=DEFAULT_TPNN_MODEL_DIR
    )
    parser.add_argument(
        "--tpnn-data-dir",
        type=str,
        default=str(DEFAULT_TPNN_DATA_DIR),
    )
    parser.add_argument("--env", type=str, default="profile_control_new")
    parser.add_argument("--task", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda-id", type=int, default=8)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _device(cuda_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if cuda_id < 0 or cuda_id >= torch.cuda.device_count():
        raise ValueError(
            f"cuda_id={cuda_id} is invalid; visible CUDA devices="
            f"{torch.cuda.device_count()}."
        )
    return torch.device(f"cuda:{cuda_id}")


def _write_result(path_value, payload):
    if path_value is None:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main():
    args = get_args()
    if args.rollout_batch_size < 1 or args.rollout_length < 1:
        raise ValueError("Rollout batch size and length must be positive.")
    if args.warmup_batch_size < 1 or args.warmup_rollout_length < 1:
        raise ValueError("Warm-up batch size and length must be positive.")
    if args.expected_tpnn_members < 1:
        raise ValueError("expected_tpnn_members must be positive.")
    if args.penalty_coef < 0:
        raise ValueError("penalty_coef must be non-negative.")

    device = _device(args.cuda_id)
    offline_data, sa_processor, evaluation_env, _ = get_rl_data_envs(
        args.env,
        args.task,
        device,
        is_val=True,
        load_hidden_states=False,
    )
    history_store = TPNNHistoryStore(args.tpnn_data_dir)
    dynamics_model = TPNNFullHistoryModel(
        args.tpnn_model_dir,
        device=device,
        microbatch_size=args.tpnn_microbatch_size,
        token_budget=args.tpnn_token_budget,
        attention_budget=args.tpnn_attention_budget,
        expected_member_count=args.expected_tpnn_members,
    )
    penalty_coef = (
        0.0
        if args.prediction_mode == "selected"
        else args.penalty_coef
    )
    dynamics = TPNNFullHistoryDynamics(
        model=dynamics_model,
        history_store=history_store,
        terminal_fn=evaluation_env.is_done,
        reward_fn=sa_processor.get_reward_new,
        penalty_coef=penalty_coef,
        uncertainty_mode=args.uncertainty_mode,
        prediction_mode=args.prediction_mode,
        training_member_indices=args.training_member_indices,
        seed=args.seed,
    )
    policy = _ZeroActionRolloutPolicy(
        dynamics=dynamics,
        state_idxs=offline_data["state_idxs"],
        action_idxs=offline_data["action_idxs"],
        sa_processor=sa_processor,
        action_dim=offline_data["actions"].shape[1],
    )
    buffer = TPNNReplayBuffer(
        buffer_size=len(offline_data["observations"]),
        obs_shape=(offline_data["observations"].shape[1],),
        obs_dtype=np.float32,
        action_dim=offline_data["actions"].shape[1],
        action_dtype=np.float32,
        device=device,
        history_store=history_store,
        seed=args.seed,
    )
    buffer.load_dataset(offline_data, hidden=False)

    warmup_batch = min(
        args.warmup_batch_size, args.rollout_batch_size
    )
    warmup_length = min(
        args.warmup_rollout_length, args.rollout_length
    )
    policy.rollout(
        buffer.sample_rollouts(warmup_batch, warmup_length)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    init_samples = buffer.sample_rollouts(
        args.rollout_batch_size, args.rollout_length
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    transitions, rollout_info = policy.rollout(init_samples)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    diagnostics = policy.last_tpnn_rollout_diagnostics
    member_counts_by_step = [
        stats["member_counts"]
        for stats in diagnostics["prediction_stats_by_step"]
    ]
    result = {
        "prediction_mode": args.prediction_mode,
        "uncertainty_mode": args.uncertainty_mode,
        "penalty_coef": penalty_coef,
        "logical_rollout_batch": args.rollout_batch_size,
        "rollout_length": args.rollout_length,
        "tpnn_microbatch_size": args.tpnn_microbatch_size,
        "tpnn_token_budget": dynamics_model.token_budget,
        "tpnn_attention_budget": dynamics_model.attention_budget,
        "num_members": dynamics.num_training_members,
        "source_num_members": dynamics_model.num_ensemble,
        "training_member_indices": (
            dynamics.training_member_indices.tolist()
        ),
        "num_transitions": int(rollout_info["num_transitions"]),
        "dynamics_seconds": diagnostics["dynamics_seconds"],
        "full_model_rollout_seconds": diagnostics[
            "full_rollout_seconds"
        ],
        "transitions_per_second": diagnostics[
            "transitions_per_second"
        ],
        "active_trajectories_by_step": diagnostics[
            "active_trajectories_by_step"
        ],
        "member_counts_by_step": member_counts_by_step,
        "forward_calls_total": diagnostics["forward_calls_total"],
        "forward_calls_by_step": [
            stats["forward_calls_total"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "forward_calls_by_member_by_step": [
            stats["forward_calls_by_member"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "microbatch_count_by_step": [
            stats["microbatch_count"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "microbatch_sizes_by_step": [
            stats["microbatch_sizes"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "microbatch_max_lengths_by_step": [
            stats["microbatch_max_lengths"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "padded_tokens_by_step": [
            stats["padded_tokens_total"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "attention_cells_by_step": [
            stats["attention_cells_total"]
            for stats in diagnostics["prediction_stats_by_step"]
        ],
        "host_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "gpu_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "gpu_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
        "transition_shapes": {
            key: list(value.shape)
            for key, value in transitions.items()
        },
    }
    _write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
