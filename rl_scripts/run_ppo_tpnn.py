"""Train PPO with zero-prefix TPNN rollouts and RPNN evaluation."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3.common.callbacks import (  # noqa: E402
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from envs.env_wrappers import GymnasiumWrapper  # noqa: E402
from envs.utils.arg_utils import normalize_args  # noqa: E402
from offlinerlkit.callbacks import (  # noqa: E402
    FusionSpecificCallback,
    TensorBoardLoggingCallback,
    TrainingProgressCallback,
)
from offlinerlkit.callbacks.ConvertAndSaveCallback import (  # noqa: E402
    ConvertAndSaveCallback,
)
from offlinerlkit.dynamics.tpnn_full_history_dynamics import (  # noqa: E402
    TPNNFullHistoryDynamics,
)
from offlinerlkit.modules.tpnn_full_history_model import (  # noqa: E402
    TPNNFullHistoryModel,
)
from offlinerlkit.policy.model_free.ppo_tpnn import (  # noqa: E402
    TPNNPPOPolicy,
)
from offlinerlkit.utils.experiment_identity import (  # noqa: E402
    build_numeric_ensemble_manifest,
    manifest_identity_sha256,
    module_state_sha256,
)
import offlinerlkit.utils.logger as logger_utils  # noqa: E402
from offlinerlkit.utils.logger import Logger, make_log_dirs  # noqa: E402
from rl_preparation.get_rl_data_envs import get_rl_data_envs  # noqa: E402
from rl_preparation.process_raw_data import evaluation_model_dir  # noqa: E402
from rl_preparation.tpnn_history_store import (  # noqa: E402
    DEFAULT_TPNN_DATA_DIR,
    TPNNHistoryStore,
)


DEFAULT_TPNN_MODEL_DIR = (
    "/data/models/tpnn_noshape_gas_benchmark_ensemble25_step_two_logvar"
)
DEFAULT_LOG_DIR = "/export/ra/baohaoming/rebuttal/exp"


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "PPO with RPNN-compatible zero-prefix TPNN training dynamics "
            "and RPNN evaluation"
        )
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="ppo_tpnn_ensemble25_zero_prefix",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.952)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-range", type=float, default=0.148)
    parser.add_argument("--ent-coef", type=float, default=0.0067)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--pol-hidden-dims", type=int, nargs="*", default=[256, 256]
    )
    parser.add_argument(
        "--val-hidden-dims", type=int, nargs="*", default=[256, 265]
    )

    parser.add_argument("--total-timesteps", type=int, default=800000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--save-freq", type=int, default=10000)

    parser.add_argument("--max-episode-length", type=int, default=150)
    parser.add_argument("--warm-start-amount", type=int, default=4)
    parser.add_argument("--min-start-idx", type=int, default=4)
    parser.add_argument("--max-start-idx", type=int, default=20)
    parser.add_argument("--env", type=str, default="profile_control_new")
    parser.add_argument("--task", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda-id", type=int, default=3)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)

    parser.add_argument(
        "--tpnn-model-dir", type=str, default=DEFAULT_TPNN_MODEL_DIR
    )
    parser.add_argument(
        "--tpnn-data-dir",
        type=str,
        default=str(DEFAULT_TPNN_DATA_DIR),
    )
    parser.add_argument(
        "--tpnn-microbatch-size", type=int, default=4096
    )
    parser.add_argument(
        "--tpnn-token-budget",
        type=int,
        default=None,
        help="Maximum padded tokens per TPNN microbatch.",
    )
    parser.add_argument(
        "--tpnn-attention-budget",
        type=int,
        default=None,
        help="Maximum rows*max_history_length^2 per microbatch.",
    )
    parser.add_argument(
        "--expected-tpnn-members", type=int, default=25
    )
    parser.add_argument(
        "--expected-evaluation-members", type=int, default=25
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default=None,
        help=(
            "Optional canonical RPNN experiment config (or experiment "
            "directory) used to verify architecture-only invariants."
        ),
    )
    return parser.parse_args()


def _get_device(cuda_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if cuda_id < 0 or cuda_id >= torch.cuda.device_count():
        raise ValueError(
            f"cuda_id={cuda_id} is invalid; visible CUDA devices="
            f"{torch.cuda.device_count()}."
        )
    return torch.device(f"cuda:{cuda_id}")


def _seed_training_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_evaluation_rpnn(
    evaluation_env,
    expected_members: int,
    expected_input_dim: int,
    expected_output_dim: int,
):
    """Validate the models that were actually loaded by the evaluation env."""
    members = getattr(evaluation_env, "all_models", None)
    if members is None:
        raise TypeError(
            "Evaluation environment does not expose loaded dynamics members."
        )
    members = list(members)
    if len(members) != int(expected_members):
        raise ValueError(
            "Evaluation dynamics member count is "
            f"{len(members)}, expected {expected_members}."
        )

    model_types = [type(member).__name__ for member in members]
    if any(model_type != "RPNN" for model_type in model_types):
        raise TypeError(
            "Evaluation dynamics must contain only RPNN members, got "
            f"{sorted(set(model_types))}."
        )
    input_dims = [int(member.input_dim) for member in members]
    output_dims = [int(member.output_dim) for member in members]
    if any(dim != int(expected_input_dim) for dim in input_dims):
        raise ValueError(
            "Evaluation RPNN input dimensions do not match the training "
            f"dynamics input dimension {expected_input_dim}: "
            f"{sorted(set(input_dims))}."
        )
    if any(dim != int(expected_output_dim) for dim in output_dims):
        raise ValueError(
            "Evaluation RPNN output dimensions do not match the training "
            f"dynamics output dimension {expected_output_dim}: "
            f"{sorted(set(output_dims))}."
        )

    resolved_path = str(
        Path(evaluation_model_dir).expanduser().resolve(strict=True)
    )
    return {
        "architecture": "RPNN",
        "member_count": len(members),
        "model_types": sorted(set(model_types)),
        "input_dim": input_dims[0],
        "output_dim": output_dims[0],
        "resolved_model_path": resolved_path,
    }


def _json_ready(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_ready(payload),
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _canonical_experiment_config(args, dynamics_model, evaluation_metadata):
    """Separate architecture-only settings from runtime bookkeeping."""
    return {
        "schema_version": 1,
        "algorithm_config": {
            "algorithm": "PPO",
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "effective_n_steps": args.effective_n_steps,
            "batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "policy_network": args.pol_hidden_dims,
            "value_network": args.val_hidden_dims,
            "total_timesteps": args.total_timesteps,
            "max_episode_length": args.max_episode_length,
            "warm_start_amount": args.warm_start_amount,
            "min_start_idx": args.min_start_idx,
            "max_start_idx": args.max_start_idx,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "save_freq": args.save_freq,
            "training_seed": args.seed,
            "initial_policy_sha256": args.initial_policy_sha256,
            "rollout_semantics_version": (
                "original_rpnn_fusion_ppo_compat_v2_zero_prefix"
            ),
            "anchor_sampling": "all_traj_start_indices_global_numpy",
            "anchor_rng": "global_numpy",
            "dynamics_member_rng": "global_numpy",
            "external_actuator_schedule": (
                "offline_full_actions_row_zero_uncontrolled_dimensions"
            ),
            "controlled_action_source": "current_policy",
            "warm_start_parameters_applied": False,
            "sb3_seed_argument": None,
            "policy_reseed_after_dynamics_load": False,
            "max_episode_length_signal": "terminated",
            "sb3_use_sde": True,
            "sb3_sde_sample_freq": -1,
            "sb3_squash_output": True,
            "observation_dim": args.obs_shape[0],
            "policy_action_dim": args.action_dim,
            "full_state_dim": args.state_dim,
            "controlled_action_dim": args.control_dim,
        },
        "dynamics_config": {
            "training": {
                "architecture": "TPNN",
                "ensemble_members": dynamics_model.num_ensemble,
                "input_dim": dynamics_model.input_dim,
                "output_dim": dynamics_model.output_dim,
                "block_size": dynamics_model.block_size,
                "resolved_model_path": dynamics_model.model_path,
                "resolved_history_data_path": str(
                    Path(args.tpnn_data_dir)
                    .expanduser()
                    .resolve(strict=True)
                ),
                "history_representation": (
                    "empty_offline_prefix_synthetic_rollout_only"
                ),
                "prediction_mode": "selected",
                "noise_rng": (
                    "global_numpy_all_members_then_fixed_member_gather"
                ),
                "member_manifest": "record/tpnn_member_manifest.json",
                "checkpoint_manifest_sha256": (
                    manifest_identity_sha256(
                        dynamics_model.get_member_manifest()
                    )
                ),
            },
            "evaluation": evaluation_metadata,
        },
        "runtime_config": {
            "task": args.task,
            "env": args.env,
            "device": str(args.device),
            "cuda_id": args.cuda_id,
            "tpnn_inference_microbatch_size": (
                args.tpnn_microbatch_size
            ),
            "tpnn_token_budget": dynamics_model.token_budget,
            "tpnn_attention_budget": dynamics_model.attention_budget,
        },
        "output_config": {
            "algo_name": args.algo_name,
            "log_root": args.log_dir,
        },
    }


def _resolve_baseline_config(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "record" / "canonical_experiment_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline canonical config does not exist: {path}"
        )
    return path


def _assert_architecture_only_config(
    current_config: dict, baseline_path: str
) -> str:
    """Require every non-architecture PPO setting to match the baseline."""
    path = _resolve_baseline_config(baseline_path)
    with path.open("r", encoding="utf-8") as file:
        baseline = json.load(file)
    if current_config.get("schema_version") != 1:
        raise ValueError("Current canonical config schema_version must be 1.")
    if baseline.get("schema_version") != 1:
        raise ValueError(
            f"{path} uses unsupported canonical schema_version="
            f"{baseline.get('schema_version')!r}."
        )
    try:
        baseline_algorithm = baseline["algorithm_config"]
        baseline_training = baseline["dynamics_config"]["training"]
        baseline_evaluation = baseline["dynamics_config"]["evaluation"]
    except KeyError as error:
        raise ValueError(
            f"{path} is missing canonical comparison metadata: {error}."
        ) from error

    mismatches = []
    current_training = current_config["dynamics_config"]["training"]
    if current_training.get("architecture") != "TPNN":
        mismatches.append(
            "dynamics_config.training.architecture: current must be TPNN"
        )
    if baseline_training.get("architecture") != "RPNN":
        mismatches.append(
            "dynamics_config.training.architecture: baseline must be RPNN"
        )
    current_algorithm = _json_ready(current_config["algorithm_config"])
    baseline_algorithm = _json_ready(baseline_algorithm)
    for key in sorted(set(current_algorithm) | set(baseline_algorithm)):
        if current_algorithm.get(key) != baseline_algorithm.get(key):
            mismatches.append(
                f"algorithm_config.{key}: "
                f"current={current_algorithm.get(key)!r}, "
                f"baseline={baseline_algorithm.get(key)!r}"
            )

    for key in ("ensemble_members", "input_dim", "output_dim"):
        if current_training.get(key) != baseline_training.get(key):
            mismatches.append(
                f"dynamics_config.training.{key}: "
                f"current={current_training.get(key)!r}, "
                f"baseline={baseline_training.get(key)!r}"
            )

    current_evaluation = current_config["dynamics_config"]["evaluation"]
    for key in (
        "architecture",
        "member_count",
        "model_types",
        "input_dim",
        "output_dim",
        "checkpoint_manifest_sha256",
    ):
        if current_evaluation.get(key) != baseline_evaluation.get(key):
            mismatches.append(
                f"dynamics_config.evaluation.{key}: "
                f"current={current_evaluation.get(key)!r}, "
                f"baseline={baseline_evaluation.get(key)!r}"
            )

    current_runtime = current_config.get("runtime_config", {})
    baseline_runtime = baseline.get("runtime_config", {})
    for key in ("task", "env"):
        if current_runtime.get(key) != baseline_runtime.get(key):
            mismatches.append(
                f"runtime_config.{key}: "
                f"current={current_runtime.get(key)!r}, "
                f"baseline={baseline_runtime.get(key)!r}"
            )

    if mismatches:
        raise ValueError(
            "Architecture-only comparison failed because invariant "
            "configuration fields differ:\n  "
            + "\n  ".join(mismatches)
        )
    return str(path)


def _log_hyperparameters(logger: Logger, args) -> None:
    logger.log_hyperparameters(_json_ready(vars(args).copy()))


def train(args=None):
    args = get_args() if args is None else args
    args = normalize_args(args)
    args.expected_tpnn_members = int(
        getattr(args, "expected_tpnn_members", 25)
    )
    args.expected_evaluation_members = int(
        getattr(args, "expected_evaluation_members", 25)
    )
    args.baseline_config = getattr(args, "baseline_config", None)
    if (
        args.expected_tpnn_members < 1
        or args.expected_evaluation_members < 1
    ):
        raise ValueError("Expected ensemble member counts must be positive.")
    args.log_dir = os.path.abspath(
        getattr(args, "log_dir", DEFAULT_LOG_DIR)
    )
    logger_utils.ROOT_DIR = args.log_dir
    args.device = _get_device(int(args.cuda_id))

    _seed_training_rngs(args.seed)

    offline_data, sa_processor, evaluation_env, _ = get_rl_data_envs(
        args.env,
        args.task,
        args.device,
        is_val=True,
        load_hidden_states=False,
    )
    history_store = TPNNHistoryStore(args.tpnn_data_dir)
    history_store.validate_offline_data(offline_data)
    dynamics_model = TPNNFullHistoryModel(
        model_path=args.tpnn_model_dir,
        device=args.device,
        microbatch_size=args.tpnn_microbatch_size,
        token_budget=args.tpnn_token_budget,
        attention_budget=args.tpnn_attention_budget,
        expected_member_count=args.expected_tpnn_members,
    )
    args.tpnn_token_budget = dynamics_model.token_budget
    args.tpnn_attention_budget = dynamics_model.attention_budget
    evaluation_metadata = _assert_evaluation_rpnn(
        evaluation_env=evaluation_env,
        expected_members=args.expected_evaluation_members,
        expected_input_dim=dynamics_model.input_dim,
        expected_output_dim=dynamics_model.output_dim,
    )
    evaluation_checkpoint_manifest = build_numeric_ensemble_manifest(
        evaluation_model_dir,
        expected_member_count=args.expected_evaluation_members,
    )
    evaluation_metadata["checkpoint_manifest_sha256"] = (
        evaluation_checkpoint_manifest["ensemble_identity_sha256"]
    )
    evaluation_metadata["member_manifest"] = (
        "record/rpnn_evaluation_manifest.json"
    )
    dynamics = TPNNFullHistoryDynamics(
        model=dynamics_model,
        history_store=history_store,
        terminal_fn=evaluation_env.is_done,
        reward_fn=sa_processor.get_reward_new,
        penalty_coef=0.0,
        prediction_mode="selected",
        use_global_numpy_noise=True,
    )

    print(
        "Training dynamics: Ensemble TPNN "
        f"({dynamics_model.model_path})"
    )
    print(
        "Training history: empty at every PPO anchor; only synthetic "
        "rollout tokens are accumulated"
    )
    print(
        f"TPNN members={dynamics_model.num_ensemble}, "
        f"block_size={dynamics_model.block_size}, "
        f"microbatch={dynamics_model.microbatch_size}"
    )
    print(
        "Evaluation dynamics: "
        f"RPNN ({evaluation_metadata['member_count']} members, "
        f"{evaluation_metadata['resolved_model_path']})"
    )

    adjusted_n_steps = min(
        int(args.n_steps), max(1, int(args.total_timesteps) // 4)
    )
    adjusted_n_steps = max(adjusted_n_steps, 256)
    effective_batch_size = min(int(args.batch_size), adjusted_n_steps)
    args.effective_n_steps = adjusted_n_steps
    args.effective_batch_size = effective_batch_size
    args.obs_shape = (offline_data["observations"].shape[1],)
    args.action_dim = offline_data["actions"].shape[1]
    args.state_dim = len(offline_data["state_idxs"])
    args.control_dim = len(offline_data["action_idxs"])
    args.training_dynamics_architecture = "Ensemble TPNN"
    args.training_dynamics_members = dynamics_model.num_ensemble
    args.evaluation_dynamics_architecture = "RPNN"
    args.evaluation_dynamics_members = evaluation_metadata["member_count"]
    args.evaluation_dynamics_input_dim = evaluation_metadata["input_dim"]
    args.evaluation_dynamics_output_dim = evaluation_metadata["output_dim"]
    args.history_mode = "empty_offline_prefix_synthetic_rollout_only"
    args.tpnn_prediction_mode = "selected"
    args.ppo_rng_mode = "global_numpy_rpnn_compatibility"
    args.warm_start_parameters_applied = False
    args.sb3_seed_argument = None
    args.policy_reseed_after_dynamics_load = False
    args.max_episode_length_signal = "terminated"
    args.external_actuator_schedule = (
        "offline_full_actions_row_zero_uncontrolled_dimensions"
    )
    args.tpnn_model_dir = dynamics_model.model_path
    args.evaluation_model_dir = evaluation_metadata[
        "resolved_model_path"
    ]

    record_params = [
        "clip_range",
        "total_timesteps",
        "learning_rate",
        "gae_lambda",
        "gamma",
        "batch_size",
        "n_steps",
        "tpnn_microbatch_size",
    ]
    output_dir = make_log_dirs(
        args.task,
        args.algo_name,
        args.seed,
        vars(args),
        record_params,
    )
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    policy = TPNNPPOPolicy(
        dynamics=dynamics,
        state_idxs=offline_data["state_idxs"],
        action_idxs=offline_data["action_idxs"],
        sa_processor=sa_processor,
        offline_data=offline_data,
        device=str(args.device),
        learning_rate=args.learning_rate,
        n_steps=adjusted_n_steps,
        batch_size=effective_batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        pol_hidden_dims=args.pol_hidden_dims,
        val_hidden_dims=args.val_hidden_dims,
        tensorboard_log=tensorboard_dir,
        max_episode_length=args.max_episode_length,
    )
    args.initial_policy_sha256 = module_state_sha256(
        policy.model.policy
    )
    canonical_config = _canonical_experiment_config(
        args, dynamics_model, evaluation_metadata
    )
    if args.baseline_config is not None:
        args.baseline_config_resolved = _assert_architecture_only_config(
            canonical_config, args.baseline_config
        )
    else:
        args.baseline_config_resolved = None

    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(output_dir, output_config)
    _log_hyperparameters(logger, args)
    record_dir = Path(output_dir) / "record"
    dynamics_model.save_member_manifest(
        record_dir / "tpnn_member_manifest.json"
    )
    _write_json(
        record_dir / "rpnn_evaluation_manifest.json",
        evaluation_checkpoint_manifest,
    )
    _write_json(
        record_dir / "canonical_experiment_config.json",
        canonical_config,
    )
    wrapped_evaluation_env = Monitor(
        GymnasiumWrapper(
            evaluation_env,
            action_dim=offline_data["act_dim"],
            obs_dim=offline_data["observations"].shape[1],
        )
    )

    callbacks = [
        TensorBoardLoggingCallback(verbose=1, log_freq=1000),
        TrainingProgressCallback(
            save_path=output_dir, verbose=1, log_freq=5000
        ),
    ]
    if args.eval_freq > 0:
        callbacks.extend(
            [
                FusionSpecificCallback(
                    eval_env=wrapped_evaluation_env,
                    logger=logger,
                    verbose=1,
                    eval_freq=args.eval_freq,
                    n_eval_episodes=5,
                ),
                EvalCallback(
                    wrapped_evaluation_env,
                    best_model_save_path=os.path.join(
                        output_dir, "best_model"
                    ),
                    log_path=os.path.join(output_dir, "evaluations"),
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                    render=False,
                    verbose=1,
                ),
            ]
        )
    if args.save_freq > 0:
        callbacks.extend(
            [
                CheckpointCallback(
                    save_freq=args.save_freq,
                    save_path=os.path.join(output_dir, "checkpoints"),
                    name_prefix="ppo_tpnn_model",
                    verbose=1,
                ),
                ConvertAndSaveCallback(
                    save_path=os.path.join(output_dir, "checkpoint"),
                    save_freq=args.save_freq,
                    pol_hidden_dims=args.pol_hidden_dims,
                    val_hidden_dims=args.val_hidden_dims,
                    verbose=1,
                ),
            ]
        )
    callback_list = CallbackList(callbacks)

    start_time = time.time()
    try:
        policy.model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
        final_model_path = os.path.join(output_dir, "model", "policy")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        policy.save(final_model_path)
    finally:
        logger.close()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    train()
