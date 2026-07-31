"""Train COMBO with full-history TPNN rollouts and RPNN evaluation."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from gym.spaces import Box

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.utils.arg_utils import normalize_args  # noqa: E402
from offlinerlkit.buffer import ReplayBuffer  # noqa: E402
from offlinerlkit.buffer.tpnn_replay_buffer import (  # noqa: E402
    TPNNReplayBuffer,
)
from offlinerlkit.dynamics.tpnn_full_history_dynamics import (  # noqa: E402
    TPNNFullHistoryDynamics,
)
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian  # noqa: E402
from offlinerlkit.modules.tpnn_full_history_model import (  # noqa: E402
    TPNNFullHistoryModel,
)
from offlinerlkit.nets import MLP  # noqa: E402
from offlinerlkit.policy.model_based.combo_tpnn import (  # noqa: E402
    TPNNCOMBOPolicy,
)
from offlinerlkit.policy_trainer import MBPolicyTrainer  # noqa: E402
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
    "/data/models/"
    "tpnn_noshape_gas_benchmark_ensemble25_step_two_logvar"
)
DEFAULT_LOG_DIR = "/export/ra/baohaoming/rebuttal/exp"
EXPECTED_EVALUATION_MEMBERS = 25
PREDICTION_MODE = "selected"


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}.")


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "COMBO with 25-member Ensemble TPNN full-history training "
            "rollouts and "
            "RPNN evaluation"
        )
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="combo_tpnn_ensemble25_full_history",
    )
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[256, 256, 256],
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=_str_to_bool, default=True)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-q-backup", type=_str_to_bool, default=False
    )
    parser.add_argument(
        "--deterministic-backup", type=_str_to_bool, default=True
    )
    parser.add_argument(
        "--with-lagrange", type=_str_to_bool, default=False
    )
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument(
        "--uniform-rollout", type=_str_to_bool, default=False
    )
    parser.add_argument(
        "--rho-s", type=str, default="mix", choices=["model", "mix"]
    )

    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=7)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument(
        "--eval-episodes",
        "--eval_episodes",
        dest="eval_episodes",
        type=int,
        default=5,
    )
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--env", type=str, default="profile_control_new")
    parser.add_argument("--task", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help="Evaluation RNG seed; defaults to --seed.",
    )
    parser.add_argument(
        "--cuda-id", "--cuda_id", dest="cuda_id", type=int, default=8
    )
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
        "--baseline-config",
        type=str,
        default=None,
        help=(
            "Optional canonical RPNN experiment config (or experiment "
            "directory). Its algorithm_config must match exactly."
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


def _build_alpha(args, action_dim):
    if not args.auto_alpha:
        return args.alpha
    target_entropy = (
        args.target_entropy
        if args.target_entropy is not None
        else -action_dim
    )
    args.target_entropy = target_entropy
    log_alpha = torch.zeros(
        1, requires_grad=True, device=args.device
    )
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    return target_entropy, log_alpha, alpha_optim


def _log_hyperparameters(logger: Logger, args) -> None:
    values = vars(args).copy()
    values["device"] = str(values["device"])
    logger.log_hyperparameters(values)


def _json_compatible(value):
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            _json_compatible(payload),
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _assert_rpnn_evaluation_env(
    evaluation_env,
    expected_input_dim: int,
    expected_output_dim: int,
) -> dict:
    models = getattr(evaluation_env, "all_models", None)
    if models is None:
        raise TypeError(
            "Evaluation environment does not expose loaded dynamics through "
            "all_models; cannot verify the RPNN evaluation architecture."
        )
    if len(models) != EXPECTED_EVALUATION_MEMBERS:
        raise ValueError(
            "Evaluation dynamics must contain exactly "
            f"{EXPECTED_EVALUATION_MEMBERS} RPNN members, found "
            f"{len(models)}."
        )

    model_types = [type(model).__name__ for model in models]
    if any(model_type != "RPNN" for model_type in model_types):
        raise TypeError(
            "Evaluation dynamics must contain only RPNN models, found "
            f"{sorted(set(model_types))}."
        )

    input_dims = [int(model.input_dim) for model in models]
    output_dims = [int(model.output_dim) for model in models]
    if any(dim != expected_input_dim for dim in input_dims):
        raise ValueError(
            "Evaluation RPNN input dimensions do not match the dataset: "
            f"expected {expected_input_dim}, found "
            f"{sorted(set(input_dims))}."
        )
    if any(dim != expected_output_dim for dim in output_dims):
        raise ValueError(
            "Evaluation RPNN output dimensions do not match the dataset: "
            f"expected {expected_output_dim}, found "
            f"{sorted(set(output_dims))}."
        )

    return {
        "architecture": "RPNN",
        "member_count": len(models),
        "model_types": sorted(set(model_types)),
        "input_dim": expected_input_dim,
        "output_dim": expected_output_dim,
        "resolved_model_path": str(
            Path(evaluation_model_dir).expanduser().resolve()
        ),
    }


def _algorithm_config(args) -> dict:
    """Fields that must match an architecture-only RPNN baseline."""
    return {
        "algorithm": "COMBO",
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "hidden_dims": list(args.hidden_dims),
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": args.alpha,
        "auto_alpha": bool(args.auto_alpha),
        "target_entropy": args.target_entropy,
        "alpha_lr": args.alpha_lr,
        "cql_weight": args.cql_weight,
        "temperature": args.temperature,
        "max_q_backup": bool(args.max_q_backup),
        "deterministic_backup": bool(args.deterministic_backup),
        "with_lagrange": bool(args.with_lagrange),
        "lagrange_threshold": args.lagrange_threshold,
        "cql_alpha_lr": args.cql_alpha_lr,
        "num_repeat_actions": args.num_repeat_actions,
        "uniform_rollout": bool(args.uniform_rollout),
        "rho_s": args.rho_s,
        "rollout_freq": args.rollout_freq,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_length": args.rollout_length,
        "model_retain_epochs": args.model_retain_epochs,
        "real_ratio": args.real_ratio,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "gradient_batch_size": args.batch_size,
        "eval_episodes": args.eval_episodes,
        "training_seed": args.seed,
        "evaluation_seed": args.evaluation_seed,
        "initial_actor_sha256": args.initial_actor_sha256,
        "initial_critic1_sha256": args.initial_critic1_sha256,
        "initial_critic2_sha256": args.initial_critic2_sha256,
        "rollout_semantics_version": (
            "fusion_anchor_hybrid_action_v1"
        ),
        "anchor_sampling": "full_history_capacity_seeded_v1",
        "external_actuator_schedule": (
            "offline_anchor_plus_t_uncontrolled_dimensions"
        ),
        "controlled_action_source": "current_policy",
        "dynamics_member_assignment": "fixed_per_trajectory",
        "observation_dim": args.obs_shape[0],
        "policy_action_dim": args.action_dim,
        "full_state_dim": args.full_state_dim,
        "full_action_dim": args.full_action_dim,
        "controlled_action_dim": args.controlled_action_dim,
    }


def _canonical_config(args, dynamics_model, evaluation_metadata) -> dict:
    return {
        "schema_version": 1,
        "algorithm_config": _algorithm_config(args),
        "dynamics_config": {
            "training": {
                "architecture": "TPNN",
                "ensemble_members": dynamics_model.num_ensemble,
                "resolved_model_path": args.tpnn_model_dir,
                "input_dim": dynamics_model.input_dim,
                "output_dim": dynamics_model.output_dim,
                "block_size": dynamics_model.block_size,
                "history_representation": (
                    "full_raw_history_exclusive_anchor"
                ),
                "resolved_history_data_path": args.tpnn_data_dir,
                "prediction_mode": PREDICTION_MODE,
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
            "task": args.task,
            "algo_name": args.algo_name,
            "log_root": args.log_dir,
            "experiment_dir": None,
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


def _assert_algorithm_config_matches(
    current_config: dict, baseline_path: str
) -> str:
    path = _resolve_baseline_config(baseline_path)
    with open(path, "r", encoding="utf-8") as file:
        baseline = json.load(file)
    if current_config.get("schema_version") != 1:
        raise ValueError("Current canonical config schema_version must be 1.")
    if baseline.get("schema_version") != 1:
        raise ValueError(
            f"{path} uses unsupported canonical schema_version="
            f"{baseline.get('schema_version')!r}."
        )
    if "algorithm_config" not in baseline:
        raise ValueError(
            f"{path} is not a canonical experiment config: "
            "algorithm_config is missing."
        )
    current_algorithm = _json_compatible(
        current_config["algorithm_config"]
    )
    reference_algorithm = _json_compatible(
        baseline["algorithm_config"]
    )
    mismatches = []
    for key in sorted(
        set(current_algorithm) | set(reference_algorithm)
    ):
        if current_algorithm.get(key) != reference_algorithm.get(key):
            mismatches.append(
                f"algorithm_config.{key}: "
                f"current={current_algorithm.get(key)!r}, "
                f"baseline={reference_algorithm.get(key)!r}"
            )

    try:
        current_training = current_config["dynamics_config"]["training"]
        baseline_training = baseline["dynamics_config"]["training"]
        current_evaluation = current_config["dynamics_config"][
            "evaluation"
        ]
        baseline_evaluation = baseline["dynamics_config"]["evaluation"]
    except KeyError as error:
        raise ValueError(
            f"{path} is missing canonical dynamics metadata: {error}."
        ) from error

    if current_training.get("architecture") != "TPNN":
        mismatches.append(
            "dynamics_config.training.architecture: current must be TPNN"
        )
    if baseline_training.get("architecture") != "RPNN":
        mismatches.append(
            "dynamics_config.training.architecture: baseline must be RPNN"
        )

    # Architecture, training model identity/history, and the TPNN
    # microbatch implementation may differ. Ensemble size and tensor
    # contracts must remain identical.
    for key in ("ensemble_members", "input_dim", "output_dim"):
        if current_training.get(key) != baseline_training.get(key):
            mismatches.append(
                f"dynamics_config.training.{key}: "
                f"current={current_training.get(key)!r}, "
                f"baseline={baseline_training.get(key)!r}"
            )

    # Evaluation is not part of the architecture intervention.  The actual
    # loaded RPNN contract is the invariant; the resolved path remains an
    # audit field only because equivalent mounts may expose the same
    # checkpoints through different paths.
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


def train(args=None):
    args = get_args() if args is None else normalize_args(args)
    args.expected_tpnn_members = int(
        getattr(args, "expected_tpnn_members", 25)
    )
    if args.expected_tpnn_members < 1:
        raise ValueError("expected_tpnn_members must be positive.")
    args.evaluation_seed = getattr(args, "evaluation_seed", None)
    if args.evaluation_seed is None:
        args.evaluation_seed = int(args.seed)
    args.baseline_config = getattr(args, "baseline_config", None)
    args.log_dir = os.path.abspath(
        getattr(args, "log_dir", DEFAULT_LOG_DIR)
    )
    args.tpnn_model_dir = str(
        Path(
            getattr(args, "tpnn_model_dir", DEFAULT_TPNN_MODEL_DIR)
        )
        .expanduser()
        .resolve()
    )
    args.tpnn_data_dir = str(
        Path(
            getattr(args, "tpnn_data_dir", DEFAULT_TPNN_DATA_DIR)
        )
        .expanduser()
        .resolve()
    )
    logger_utils.ROOT_DIR = args.log_dir
    args.device = _get_device(int(args.cuda_id))

    _seed_training_rngs(args.seed)
    torch.backends.cudnn.deterministic = True

    offline_data, sa_processor, evaluation_env, _ = get_rl_data_envs(
        args.env,
        args.task,
        args.device,
        is_val=True,
        load_hidden_states=False,
    )
    evaluation_env.seed(args.evaluation_seed)
    expected_evaluation_input_dim = int(
        offline_data["full_observations"].shape[1]
        + 2 * offline_data["full_actions"].shape[1]
    )
    expected_evaluation_output_dim = int(
        offline_data["full_observations"].shape[1]
    )
    evaluation_metadata = _assert_rpnn_evaluation_env(
        evaluation_env,
        expected_input_dim=expected_evaluation_input_dim,
        expected_output_dim=expected_evaluation_output_dim,
    )
    evaluation_checkpoint_manifest = build_numeric_ensemble_manifest(
        evaluation_model_dir,
        expected_member_count=EXPECTED_EVALUATION_MEMBERS,
    )
    evaluation_metadata["checkpoint_manifest_sha256"] = (
        evaluation_checkpoint_manifest["ensemble_identity_sha256"]
    )
    evaluation_metadata["member_manifest"] = (
        "rpnn_evaluation_manifest.json"
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
    if dynamics_model.num_ensemble != args.expected_tpnn_members:
        raise ValueError(
            "Unexpected TPNN ensemble size: expected "
            f"{args.expected_tpnn_members}, found "
            f"{dynamics_model.num_ensemble}."
        )
    dynamics = TPNNFullHistoryDynamics(
        model=dynamics_model,
        history_store=history_store,
        terminal_fn=evaluation_env.is_done,
        reward_fn=sa_processor.get_reward_new,
        penalty_coef=0.0,
        prediction_mode=PREDICTION_MODE,
        seed=args.seed,
    )

    args.obs_shape = (offline_data["observations"].shape[1],)
    args.action_dim = int(offline_data["actions"].shape[1])
    args.full_state_dim = int(
        offline_data["full_observations"].shape[1]
    )
    args.full_action_dim = int(offline_data["full_actions"].shape[1])
    args.controlled_action_dim = int(len(offline_data["action_idxs"]))
    args.max_action = 1.0
    args.training_dynamics_architecture = "Ensemble TPNN"
    args.training_dynamics_members = dynamics_model.num_ensemble
    args.evaluation_dynamics_architecture = "Ensemble RPNN"
    args.evaluation_dynamics_members = evaluation_metadata["member_count"]
    args.evaluation_model_dir = evaluation_metadata[
        "resolved_model_path"
    ]
    args.history_mode = "full_raw_history_exclusive_anchor"
    args.prediction_mode = PREDICTION_MODE

    # Frozen model loading consumes Torch RNG.  Reseed immediately before
    # policy construction so the same experiment seed gives the same
    # actor/critic initialization for RPNN and TPNN training dynamics.
    _seed_training_rngs(args.seed)
    action_space = Box(
        low=-args.max_action,
        high=args.max_action,
        shape=(args.action_dim,),
        dtype=np.float32,
    )
    actor_backbone = MLP(
        input_dim=np.prod(args.obs_shape),
        hidden_dims=args.hidden_dims,
    )
    critic1_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims,
    )
    critic2_backbone = MLP(
        input_dim=np.prod(args.obs_shape) + args.action_dim,
        hidden_dims=args.hidden_dims,
    )
    dist = TanhDiagGaussian(
        latent_dim=actor_backbone.output_dim,
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action,
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=args.actor_lr
    )
    critic1_optim = torch.optim.Adam(
        critic1.parameters(), lr=args.critic_lr
    )
    critic2_optim = torch.optim.Adam(
        critic2.parameters(), lr=args.critic_lr
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_optim, args.epoch
    )

    policy = TPNNCOMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        offline_data["state_idxs"],
        offline_data["action_idxs"],
        sa_processor,
        action_space=action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=_build_alpha(args, args.action_dim),
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s,
    )
    args.initial_actor_sha256 = module_state_sha256(actor)
    args.initial_critic1_sha256 = module_state_sha256(critic1)
    args.initial_critic2_sha256 = module_state_sha256(critic2)

    canonical_config = _canonical_config(
        args, dynamics_model, evaluation_metadata
    )
    if args.baseline_config is not None:
        args.baseline_config_resolved = (
            _assert_algorithm_config_matches(
                canonical_config, args.baseline_config
            )
        )
    else:
        args.baseline_config_resolved = None

    real_buffer = TPNNReplayBuffer(
        buffer_size=len(offline_data["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device,
        history_store=history_store,
        seed=args.seed,
    )
    real_buffer.load_dataset(offline_data, hidden=False)
    fake_buffer = ReplayBuffer(
        buffer_size=(
            args.rollout_batch_size
            * args.rollout_length
            * args.model_retain_epochs
        ),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device,
    )

    log_dirs = make_log_dirs(
        args.task,
        args.algo_name,
        args.seed,
        vars(args),
        record_params=[
            "cql_weight",
            "rollout_length",
            "tpnn_microbatch_size",
        ],
    )
    logger = Logger(
        log_dirs,
        {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "tb": "tensorboard",
        },
    )
    canonical_config["output_config"]["experiment_dir"] = str(
        Path(log_dirs).resolve()
    )
    args.tpnn_member_manifest_path = os.path.join(
        logger.record_dir, "tpnn_member_manifest.json"
    )
    args.canonical_config_path = os.path.join(
        logger.record_dir, "canonical_experiment_config.json"
    )
    args.rpnn_evaluation_manifest_path = os.path.join(
        logger.record_dir, "rpnn_evaluation_manifest.json"
    )
    canonical_config["dynamics_config"]["training"][
        "member_manifest_path"
    ] = args.tpnn_member_manifest_path
    _log_hyperparameters(logger, args)
    dynamics_model.save_member_manifest(
        args.tpnn_member_manifest_path
    )
    _write_json(
        args.rpnn_evaluation_manifest_path,
        evaluation_checkpoint_manifest,
    )
    _write_json(args.canonical_config_path, canonical_config)

    print(
        "Training dynamics: Ensemble TPNN "
        f"({dynamics_model.num_ensemble} members; "
        f"{args.tpnn_model_dir})"
    )
    print(
        "Training history: full raw history, exclusive anchor "
        f"({args.tpnn_data_dir})"
    )
    print(
        "Evaluation dynamics: Ensemble RPNN "
        f"({evaluation_metadata['member_count']} members; "
        f"{evaluation_metadata['resolved_model_path']})"
    )
    print(f"TPNN prediction mode: {PREDICTION_MODE}")
    print(
        f"Logical rollout batch={args.rollout_batch_size}, "
        f"rollout length={args.rollout_length}, "
        f"TPNN microbatch={args.tpnn_microbatch_size}, "
        f"RL gradient batch={args.batch_size}"
    )

    trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=evaluation_env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(
            args.rollout_freq,
            args.rollout_batch_size,
            args.rollout_length,
        ),
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        eval_episodes=args.eval_episodes,
        lr_scheduler=lr_scheduler,
    )
    try:
        return trainer.train()
    except Exception:
        logger.close()
        raise


if __name__ == "__main__":
    train()
