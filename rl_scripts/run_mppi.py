"""
Run MPPI planning only and save the planned rollouts.

This script intentionally does not run GCIL policy distillation. It writes a
root-level D4RL-style dataset plus per-shot/per-episode groups used by
visualization/profile_tracking_mppi.py.
"""

import argparse
import copy
import os
import random
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.env_wrappers import PlanningWrapper
from offlinerlkit.planner import MPPI
import offlinerlkit.planner.mppi as mppi_module
from offlinerlkit.utils.logger import Logger, make_log_dirs_new
from rl_preparation.get_rl_data_envs import get_rl_data_dual_envs


def get_args():
    parser = argparse.ArgumentParser(description="Run MPPI planning without GCIL distillation")
    parser.add_argument("--algo-name", type=str, default="mppi")

    # MPPI planning hyperparameters
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--penalty-coef", type=float, default=2.5)
    parser.add_argument("--episodes_per_shot", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--lam", type=float, default=2.0)

    # Environment and runtime settings
    parser.add_argument("--env", type=str, default="profile_control_new")
    parser.add_argument("--task", type=str, default="rotation")
    parser.add_argument("--test", action="store_true", help="Use test split instead of validation split")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--planner-seed", type=int, default=1, help="Random seed for MPPI action-sequence sampling")
    parser.add_argument("--eval-seed", type=int, default=None, help="Random seed for the planning/evaluation environments")
    parser.add_argument("--eval-seed-start", type=int, default=None, help="Start eval seed for batched MPPI runs")
    parser.add_argument("--num-eval-seeds", type=int, default=None, help="Number of eval seeds to run in one invocation")
    parser.add_argument("--eval-seed-list", type=int, nargs="*", default=None, help="Explicit eval seed list for batched MPPI runs")
    parser.add_argument("--cuda-id", "--cuda_id", dest="cuda_id", type=int, default=0)

    # Output settings
    parser.add_argument("--base-dir", type=str, default="/home/scratch/jiayuc2/bao/optuna_last_results_bao/log")
    parser.add_argument("--planning-data-path", type=str, default=None)
    parser.add_argument("--trial-id", type=str, default=None)
    return parser.parse_args()


def set_seed(seed, *envs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    for env in envs:
        env.seed(seed)


def get_device(cuda_id):
    if cuda_id == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")


def save_planning_data(path, dataset, rollouts, args):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    shot_ids = []
    episode_ids = []
    for shot_id, shot_rollouts in rollouts.items():
        for episode_id, rollout in enumerate(shot_rollouts):
            n_steps = len(rollout["actions"])
            shot_ids.extend([shot_id] * n_steps)
            episode_ids.extend([episode_id] * n_steps)

    with h5py.File(path, "w") as hdf:
        for key, value in dataset.items():
            value = np.asarray(value)
            print(key, value.shape)
            hdf.create_dataset(key, data=value)

        hdf.create_dataset("shot_ids", data=np.asarray(shot_ids, dtype=np.int64))
        hdf.create_dataset("episode_ids", data=np.asarray(episode_ids, dtype=np.int64))

        shots_group = hdf.create_group("shots")
        for shot_id, shot_rollouts in rollouts.items():
            shot_group = shots_group.create_group(str(shot_id))
            for episode_id, rollout in enumerate(shot_rollouts):
                episode_group = shot_group.create_group(f"episode_{episode_id}")
                for key, value in rollout.items():
                    episode_group.create_dataset(key, data=np.asarray(value))

        hdf.attrs["env"] = args.env
        hdf.attrs["task"] = args.task
        hdf.attrs["split"] = "test" if args.test else "val"
        hdf.attrs["seed"] = args.seed
        hdf.attrs["planner_seed"] = args.planner_seed
        hdf.attrs["eval_seed"] = args.eval_seed
        hdf.attrs["algo_name"] = args.algo_name
        hdf.attrs["log_dirs"] = args.log_dirs
        hdf.attrs["episodes_per_shot"] = args.episodes_per_shot
        if hasattr(args, "planning_model_dir"):
            hdf.attrs["planning_model_dir"] = args.planning_model_dir
        if hasattr(args, "evaluation_model_dir"):
            hdf.attrs["evaluation_model_dir"] = args.evaluation_model_dir


def resolve_eval_seeds(args):
    if args.eval_seed_list:
        return args.eval_seed_list
    if args.num_eval_seeds is not None:
        start_seed = args.eval_seed_start
        if start_seed is None:
            start_seed = args.eval_seed if args.eval_seed is not None else args.seed
        return list(range(start_seed, start_seed + args.num_eval_seeds))
    return [args.eval_seed if args.eval_seed is not None else args.seed]


def materialize_planning_data_path(path_template, eval_seed):
    if path_template is None:
        return None
    if "{seed}" in path_template or "{eval_seed}" in path_template:
        return path_template.format(seed=eval_seed, eval_seed=eval_seed)
    return path_template


def train_single_seed(args):
    args = copy.deepcopy(args)
    if args.planner_seed is None:
        args.planner_seed = args.seed
    if args.eval_seed is None:
        args.eval_seed = args.seed
    args.seed = args.eval_seed

    args.device = get_device(args.cuda_id)
    print(f"Using MPPI from: {mppi_module.__file__}")
    print(f"Planner seed: {args.planner_seed} | Eval seed: {args.eval_seed}")
    offline_data, _, train_env, eval_env, training_model_dir, evaluation_model_dir = get_rl_data_dual_envs(
        args.env,
        args.task,
        args.device,
        is_val=not args.test,
    )
    args.planning_model_dir = training_model_dir
    args.evaluation_model_dir = evaluation_model_dir

    args.obs_shape = (offline_data["observations"].shape[1],)
    args.action_dim = offline_data["actions"].shape[1]
    args.max_action, args.min_action = 1.0, -1.0

    set_seed(args.eval_seed, train_env, eval_env)

    record_params = ["horizon", "lam", "num_samples", "planner_seed"]
    if getattr(args, "trial_id", None):
        record_params.append("trial_id")
    log_dirs = make_log_dirs_new(
        args.task,
        args.algo_name,
        args.seed,
        vars(args),
        record_params=record_params,
        base_dir=args.base_dir,
    )
    args.log_dirs = log_dirs
    if args.planning_data_path is None:
        args.planning_data_path = os.path.join(log_dirs, "planning_data.h5")

    output_config = {
        "consoleout_backup": "stdout",
        "mppi_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(log_dirs, output_config)

    plan_env = PlanningWrapper(train_env)
    eval_plan_env = PlanningWrapper(eval_env)
    plan_env.seed(args.eval_seed)
    eval_plan_env.seed(args.eval_seed)
    planner = MPPI(plan_env, args, logger, args.device, eval_env=eval_plan_env)
    rollouts, planning_dataset = planner.run_rollouts()

    save_planning_data(args.planning_data_path, planning_dataset, rollouts, args)
    print(f"Planning data saved to: {args.planning_data_path}")
    return log_dirs


def train(args=None):
    if args is None:
        args = get_args()

    if args.eval_seed_start is not None and args.num_eval_seeds is None:
        raise ValueError("--eval-seed-start requires --num-eval-seeds")

    eval_seeds = resolve_eval_seeds(args)
    if len(eval_seeds) > 1 and args.planning_data_path is not None:
        if "{seed}" not in args.planning_data_path and "{eval_seed}" not in args.planning_data_path:
            raise ValueError(
                "Batched eval seeds with explicit --planning-data-path require a "
                "'{seed}' or '{eval_seed}' placeholder to avoid overwriting files."
            )
    log_dirs_by_seed = {}
    for eval_seed in eval_seeds:
        current_args = copy.deepcopy(args)
        current_args.eval_seed = eval_seed
        current_args.seed = eval_seed
        current_args.planning_data_path = materialize_planning_data_path(args.planning_data_path, eval_seed)
        log_dirs_by_seed[eval_seed] = train_single_seed(current_args)
    return log_dirs_by_seed


if __name__ == "__main__":
    train()
