import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from envs.utils.profile_util import reconstruct_profile_from_state
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from visualization.controller_new import Controller as ProfileTrackingController
from visualization.plotter import (
    make_paper_quality_prof_fig,
    plot_tracking_quantities_prof,
)


DEFAULT_ACTUATORS_TO_PLOT = ["pinj", "tinj", "gasA", "ech_pwr_total"]
SCRIPT_DEFAULT_ACTOR_SETTINGS = {
    "il_actor": False,
    "stochastic_actor": True,
    "hidden_dims": [256, 256, 256],
    "deterministic_mode": True,
    "use_diag_gaussian": False,
    "dropout_rate": None,
}

ALGO_ACTOR_DEFAULTS = {
    "bambrl": {"hidden_dims": [256, 256]},
    "combo": {"hidden_dims": [256, 256, 256]},
    "cql": {"hidden_dims": [256, 256, 256]},
    "edac": {"hidden_dims": [256, 256, 256]},
    "gcil": {"hidden_dims": [256, 256, 256], "il_actor": True},
    "iql": {"hidden_dims": [256, 256], "use_diag_gaussian": True, "dropout_rate": None},
    "mcq": {"hidden_dims": [400, 400], "dropout_rate": 0.1},
    "mobile": {"hidden_dims": [256, 256]},
    "mopo": {"hidden_dims": [256, 256]},
    "mppi": {"hidden_dims": [256, 256, 256]},
    "ppo": {"hidden_dims": [256, 256], "use_diag_gaussian": True,},
    "rambo": {"hidden_dims": [256, 256]},
    "rombrl": {"hidden_dims": [256, 256]},
    "td3bc": {"stochastic_actor": False, "hidden_dims": [256, 256]},
}


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def get_args():
    parser = argparse.ArgumentParser(description="Unified visualization and tracking-error utilities")

    parser.add_argument(
        "--mode",
        type=str,
        choices=("profile", "error"),
        default="profile",
        help="profile: save profile-tracking outputs; error: summarize saved-shot RMSE from saved results",
    )

    # Shared evaluation settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(10)), help="Seeds used by profile mode")
    parser.add_argument("--cuda_id", type=int, default=5, help="CUDA device ID")
    parser.add_argument("--env", type=str, default="profile_control_new")
    parser.add_argument("--task", type=str, default="temp", help="Targets to track")
    parser.add_argument("--actor_path", type=str, default="/export/pgs/fuyang/log_syn/temp/combo&cql_weight=10&rollout_length=7/seed_1&timestamp_26-0410-225124")
    parser.add_argument("--run_name", type=str, default=None, help="Optional human-readable name for a run or algorithm")
    parser.add_argument("--il_actor", type=_str2bool, default=False, help="Is this an imitation-learning actor?")
    parser.add_argument("--stochastic_actor", type=_str2bool, default=True, help="Is this a stochastic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[256, 256, 256], help="Hidden dimensions of the actor network")
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    parser.add_argument("--use_diag_gaussian", action="store_true", help="Use DiagGaussian instead of TanhDiagGaussian")
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate for actor backbone")
    parser.add_argument("--test", action="store_true", default=False, help="Use the test split; default is validation split")

    # Output base directories
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory for outputs")

    # Tracking-error summary settings
    return parser.parse_args()


def _resolve_output_base_dir(args):
    """Resolve the single output root used by profile and error modes."""

    default_dir = Path(__file__).resolve().parents[1] / "results"
    resolved = args.output_base_dir or str(default_dir)
    args.output_base_dir = resolved
    return resolved


def _build_profile_log_folder(args):
    actor_path = Path(args.actor_path)
    split_name = "val" if not args.test else "test"
    return os.path.join(
        _resolve_output_base_dir(args),
        args.task,
        actor_path.parent.name,
        split_name,
        actor_path.name,
        str(args.seed),
    )


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _set_model_dimensions(args, offline_data):
    if args.env == "fusion_env":
        args.obs_dim = offline_data["obs_dim"]
        args.action_dim = offline_data["act_dim"]
    else:
        args.obs_dim = offline_data["observations"].shape[1]
        args.action_dim = offline_data["actions"].shape[1]
    args.max_action = 1.0


def _apply_actor_defaults_from_path(args):
    algo_name = extract_algorithm_name(args.actor_path)
    algo_defaults = ALGO_ACTOR_DEFAULTS.get(algo_name, {})

    if algo_name not in ALGO_ACTOR_DEFAULTS:
        print(f"Warning: no explicit actor defaults registered for '{algo_name}', keeping current settings.")

    args.algo_name = algo_name

    for key, value in algo_defaults.items():
        if hasattr(args, key) and getattr(args, key) == SCRIPT_DEFAULT_ACTOR_SETTINGS.get(key):
            setattr(args, key, value)

    print(
        f"Auto-configured actor from '{algo_name}': "
        f"il_actor={args.il_actor}, stochastic_actor={args.stochastic_actor}, "
        f"hidden_dims={args.hidden_dims}, deterministic_mode={args.deterministic_mode}, "
        f"use_diag_gaussian={args.use_diag_gaussian}, dropout_rate={args.dropout_rate}, "
        f"test={args.test}"
    )

    return algo_name


def calculate_tracking_metrics(target_array, current_array):
    """Calculate tracking performance metrics."""

    target_array = np.asarray(target_array)
    current_array = np.asarray(current_array)

    error = target_array - current_array
    abs_error = np.abs(error)

    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "cumulative_absolute_error": float(np.sum(abs_error)),
        "cumulative_squared_error": float(np.sum(error**2)),
    }

    if target_array.ndim > 1:
        metrics["per_component"] = {}
        for dim in range(target_array.shape[1]):
            component_name = f"component{dim + 1}"
            error_dim = error[:, dim]
            abs_error_dim = abs_error[:, dim]
            metrics["per_component"][component_name] = {
                "rmse": float(np.sqrt(np.mean(error_dim**2))),
                "mae": float(np.mean(abs_error_dim)),
                "cumulative_absolute_error": float(np.sum(abs_error_dim)),
                "cumulative_squared_error": float(np.sum(error_dim**2)),
            }
    else:
        metrics["per_component"] = {
            "component1": {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "cumulative_absolute_error": metrics["cumulative_absolute_error"],
                "cumulative_squared_error": metrics["cumulative_squared_error"],
            }
        }

    return metrics


def calculate_reconstruct_tracking_metrics(target_array, current_array):
    target_array = np.asarray(target_array)
    current_array = np.asarray(current_array)
    error = target_array - current_array
    return np.mean(error**2, axis=0)


def print_tracking_metrics(shot_id, metrics, episode_reward, episode_length):
    print(f"\nShot #{shot_id} - Episode Reward: {episode_reward:.3f}, Length: {episode_length}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Cumulative Absolute Error: {metrics['cumulative_absolute_error']:.2f}")
    print(f"  Cumulative Squared Error: {metrics['cumulative_squared_error']:.2f}")


def save_tracking_results(all_results, log_folder):
    """Save tracking results to JSON and CSV files."""

    shot_results = [v for k, v in all_results.items() if k.startswith("shot_")]
    num_shots = len(shot_results)

    if num_shots > 0:
        summary_metrics = {
            "total_shots": num_shots,
            "avg_rmse": sum(shot["tracking_metrics"]["rmse"] for shot in shot_results) / num_shots,
            "avg_mae": sum(shot["tracking_metrics"]["mae"] for shot in shot_results) / num_shots,
            "avg_cumulative_absolute_error": sum(shot["tracking_metrics"]["cumulative_absolute_error"] for shot in shot_results) / num_shots,
            "avg_reward": sum(shot["episode_reward"] for shot in shot_results) / num_shots,
            "avg_length": sum(shot["episode_length"] for shot in shot_results) / num_shots,
        }

        if "per_component" in shot_results[0]["tracking_metrics"]:
            component_names = list(shot_results[0]["tracking_metrics"]["per_component"].keys())
            summary_metrics["per_component"] = {}
            for comp_name in component_names:
                summary_metrics["per_component"][comp_name] = {
                    "avg_rmse": sum(shot["tracking_metrics"]["per_component"][comp_name]["rmse"] for shot in shot_results) / num_shots,
                    "avg_mae": sum(shot["tracking_metrics"]["per_component"][comp_name]["mae"] for shot in shot_results) / num_shots,
                    "avg_cumulative_absolute_error": sum(
                        shot["tracking_metrics"]["per_component"][comp_name]["cumulative_absolute_error"] for shot in shot_results
                    )
                    / num_shots,
                }

        all_results["summary"] = summary_metrics

    results_file = os.path.join(log_folder, "tracking_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    try:
        import pandas as pd

        df_data = []
        for shot_key, shot_data in all_results.items():
            if shot_key == "summary":
                continue
            row = {
                "shot_id": shot_data["shot_id"],
                "episode_reward": shot_data["episode_reward"],
                "episode_length": shot_data["episode_length"],
                "rmse": shot_data["tracking_metrics"]["rmse"],
                "mae": shot_data["tracking_metrics"]["mae"],
                "cumulative_absolute_error": shot_data["tracking_metrics"]["cumulative_absolute_error"],
                "cumulative_squared_error": shot_data["tracking_metrics"]["cumulative_squared_error"],
            }

            if "per_component" in shot_data["tracking_metrics"]:
                for comp_name, comp_metrics in shot_data["tracking_metrics"]["per_component"].items():
                    row[f"{comp_name}_rmse"] = comp_metrics["rmse"]
                    row[f"{comp_name}_mae"] = comp_metrics["mae"]
                    row[f"{comp_name}_cumulative_absolute_error"] = comp_metrics["cumulative_absolute_error"]

            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_file = os.path.join(log_folder, "tracking_results.csv")
        df.to_csv(csv_file, index=False)

        print("\nResults saved to:")
        print(f"JSON format: {results_file}")
        print(f"CSV format: {csv_file}")
    except ImportError:
        print("\nResults saved to:")
        print(f"JSON format: {results_file}")
        print("Note: pandas not installed, cannot save CSV format")

    if "summary" in all_results:
        summary = all_results["summary"]
        print("\nOverall Statistics:")
        print(f"Average RMSE: {summary['avg_rmse']:.4f}")
        print(f"Average MAE: {summary['avg_mae']:.4f}")
        print(f"Average Cumulative Absolute Error: {summary['avg_cumulative_absolute_error']:.4f}")
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(f"Average Length: {summary['avg_length']:.1f}")

        if "per_component" in summary:
            print("\nPer-Component Average Statistics:")
            for comp_name, comp_metrics in summary["per_component"].items():
                print(f"  {comp_name}:")
                print(f"    Average RMSE: {comp_metrics['avg_rmse']:.4f}")
                print(f"    Average MAE: {comp_metrics['avg_mae']:.4f}")
                print(f"    Average Cumulative Absolute Error: {comp_metrics['avg_cumulative_absolute_error']:.4f}")


def save_shot_data(shot_data_dict, save_folder, task_name, actor_info, run_name=None):
    """Save shot data to an npz file."""

    data_save_dir = os.path.join(save_folder, "saved_data")
    os.makedirs(data_save_dir, exist_ok=True)

    npz_file = os.path.join(data_save_dir, "shot_data.npz")
    np.savez(npz_file, **shot_data_dict)

    metadata = {
        "task_name": task_name,
        "actor_info": actor_info,
        "run_name": run_name,
        "algorithm_name": extract_algorithm_name(actor_info),
        "experiment_name": Path(actor_info).parent.name,
        "num_shots": len([k for k in shot_data_dict.keys() if k.startswith("shot_")]),
        "shot_ids": [shot_data_dict[k]["shot_id"] for k in shot_data_dict.keys() if k.startswith("shot_")],
        "save_time": str(np.datetime64("now")),
    }

    metadata_file = os.path.join(data_save_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nShot data saved to: {npz_file}")
    print(f"Metadata saved to: {metadata_file}")

    return npz_file, metadata_file


def extract_algorithm_name(actor_path: str) -> str:
    """Extract a compact algorithm name from an actor checkpoint path."""

    normalized_path = Path(os.path.normpath(actor_path))
    special_names = {"policy", "actor", "model", "checkpoint", "checkpoints"}

    for part in reversed(normalized_path.parts):
        stem = part.split("&", 1)[0].split(".", 1)[0]
        if not stem:
            continue
        lower_stem = stem.lower()
        if lower_stem.startswith("seed_") or lower_stem in special_names:
            continue
        return lower_stem

    return normalized_path.parent.name.split("&", 1)[0].lower()


def _rollout_shot(args, env, sa_processor, controller, shot):
    obs = env.reset(shot_id=shot)
    episode_reward = 0.0
    episode_length = 0

    time_array = []
    target_quan_array = []
    real_quan_array = []
    cur_quan_array = []
    real_act_array = []
    cur_act_array = []

    while True:
        action = controller.act(obs)
        if args.env == "fusion_env":
            next_obs, _, reward, terminal, info = env.step(action, obs)
        else:
            next_obs, reward, terminal, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        cur_time = info["time_step"] - 1
        target_quan, real_quan, cur_quan, real_act, cur_act = sa_processor.get_plot_quantities(shot, cur_time, obs, action)
        time_array.append(cur_time)
        target_quan_array.append(target_quan)
        real_quan_array.append(real_quan)
        cur_quan_array.append(cur_quan)
        real_act_array.append(real_act)
        cur_act_array.append(cur_act)

        if terminal:
            break

        obs = next_obs

    return {
        "time_array": np.asarray(time_array),
        "target_quan_array": np.asarray(target_quan_array),
        "real_quan_array": np.asarray(real_quan_array),
        "cur_quan_array": np.asarray(cur_quan_array),
        "real_act_array": np.asarray(real_act_array),
        "cur_act_array": np.asarray(cur_act_array),
        "episode_reward": episode_reward,
        "episode_length": episode_length,
    }


def _reconstruct_profile_save_profiles(args, env, sa_processor, target_quan_array, real_quan_array, cur_quan_array):
    if args.env != "fusion_env":
        return target_quan_array, real_quan_array, cur_quan_array

    try:
        reconstruct_real_quan_array = reconstruct_profile_from_state(
            args.task, np.asarray(real_quan_array), env.info, sa_processor.idx_list, unnormalize=False
        )
        reconstruct_target_quan_array = reconstruct_profile_from_state(
            args.task, np.asarray(target_quan_array), env.info, sa_processor.idx_list, unnormalize=True
        )
        reconstruct_cur_quan_array = reconstruct_profile_from_state(
            args.task, np.asarray(cur_quan_array), env.info, sa_processor.idx_list, unnormalize=True
        )
        return reconstruct_target_quan_array, reconstruct_real_quan_array, reconstruct_cur_quan_array
    except Exception as exc:
        print(f"Warning: failed to reconstruct profile-save quantities: {exc}")
        return target_quan_array, real_quan_array, cur_quan_array


def run_profile_mode(args):
    args.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    _apply_actor_defaults_from_path(args)
    offline_data, sa_processor, env, _ = get_rl_data_envs(
        args.env,
        args.task,
        args.device,
        is_il=args.il_actor,
        is_val=not args.test,
    )
    _set_model_dimensions(args, offline_data)

    shot_list = env.get_eval_shot_list()
    _, act_names = sa_processor.get_plot_names()
    algo_name = extract_algorithm_name(args.actor_path)
    actor_path = Path(args.actor_path)

    log_folder = _build_profile_log_folder(args)
    os.makedirs(log_folder, exist_ok=True)

    _set_seed(args.seed)
    env.seed(args.seed)
    controller = ProfileTrackingController(args)

    all_results = {}
    shot_data_dict = {}
    for shot in shot_list:
        rollout = _rollout_shot(args, env, sa_processor, controller, shot)
        time_array = rollout["time_array"]
        target_quan_array = rollout["target_quan_array"]
        real_quan_array = rollout["real_quan_array"]
        cur_quan_array = rollout["cur_quan_array"]
        real_act_array = rollout["real_act_array"]
        cur_act_array = rollout["cur_act_array"]

        reconstruct_target_quan_array, reconstruct_real_quan_array, reconstruct_cur_quan_array = _reconstruct_profile_save_profiles(
            args, env, sa_processor, target_quan_array, real_quan_array, cur_quan_array
        )

        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        mse_reconstruct = calculate_reconstruct_tracking_metrics(reconstruct_target_quan_array, reconstruct_cur_quan_array)

        all_results[f"shot_{shot}"] = {
            "shot_id": shot,
            "episode_reward": rollout["episode_reward"],
            "episode_length": rollout["episode_length"],
            "tracking_metrics": tracking_metrics,
        }

        shot_data_dict[f"shot_{shot}"] = {
            "shot_id": shot,
            "cur_quan_array": cur_quan_array,
            "cur_act_array": cur_act_array,
            "reconstruct_cur_quan_array": reconstruct_cur_quan_array,
            "reconstruct_target_quan_array": reconstruct_target_quan_array,
            "episode_reward": rollout["episode_reward"],
            "episode_length": rollout["episode_length"],
            "mse_reconstruct": mse_reconstruct,
        }

        plot_tracking_quantities_prof(
            rollout["time_array"],
            reconstruct_target_quan_array,
            real_quan_array,
            reconstruct_cur_quan_array,
            args.task,
            algo_name.upper(),
            shot,
            log_folder,
        )
        make_paper_quality_prof_fig(
            rollout["time_array"],
            reconstruct_target_quan_array,
            reconstruct_cur_quan_array,
            cur_act_array,
            args.task,
            algo_name.upper(),
            shot,
            log_folder,
            env.info,
            DEFAULT_ACTUATORS_TO_PLOT,
        )

        print_tracking_metrics(shot, tracking_metrics, rollout["episode_reward"], rollout["episode_length"])

    save_tracking_results(all_results, log_folder)
    save_shot_data(shot_data_dict, log_folder, args.task, str(actor_path), getattr(args, "run_name", None))
    summarize_tracking_error(log_folder, args.task, os.path.join(log_folder, "tracking_error_summary.csv"))
    return log_folder


def summarize_tracking_error(search_root: str, task_name: str, output_path: str):
    import pandas as pd

    results = {}
    root_path = Path(search_root)
    candidate_files = sorted(root_path.rglob("saved_data/shot_data.npz"))
    if not candidate_files:
        print(f"No saved_data/shot_data.npz found under: {root_path}")
        return

    grouped_mse = {}
    grouped_file_counts = {}
    for seed_path in candidate_files:
        try:
            data = np.load(seed_path, allow_pickle=True)
            metadata = {}
            meta_path = seed_path.with_name("metadata.json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            if metadata.get("task_name") and metadata["task_name"] != task_name:
                continue

            run_name = metadata.get("run_name")
            actor_path = metadata.get("actor_info") or metadata.get("actor_path") or ""
            algorithm_name = metadata.get("algorithm_name") or (extract_algorithm_name(actor_path) if actor_path else seed_path.parents[3].name)
            experiment_name = metadata.get("experiment_name") or (Path(actor_path).parent.name if actor_path else seed_path.parents[2].name)
            group_key = (run_name or f"{algorithm_name}/{experiment_name}").replace("\\", "/")

            file_has_mse = False
            for key in data.files:
                if not key.startswith("shot_"):
                    continue
                shot_data = data[key].item()
                mse_reconstruct = shot_data.get("mse_reconstruct")
                if mse_reconstruct is None:
                    print(f"'mse_reconstruct' not found in {seed_path} ({key})")
                    continue
                file_has_mse = True
                grouped_mse.setdefault(group_key, []).append(np.asarray(mse_reconstruct))
            if file_has_mse:
                grouped_file_counts[group_key] = grouped_file_counts.get(group_key, 0) + 1
        except Exception as exc:
            print(f"Error loading {seed_path}: {exc}")

    if not grouped_mse:
        print("No valid mse data were found.")
        return

    print(f"Discovered {len(candidate_files)} shot_data file(s) under {root_path}")

    for group_key, mse_list in sorted(grouped_mse.items()):
        mse_array = np.stack(mse_list, axis=0)
        rmse_array = np.sqrt(mse_array)

        mean = rmse_array.mean(axis=0)
        se = rmse_array.std(axis=0, ddof=1) / np.sqrt(rmse_array.shape[0]) if rmse_array.shape[0] > 1 else np.zeros_like(mean)

        per_shot_overall_rmse = np.sqrt(np.mean(mse_array, axis=1))
        mean_all = float(per_shot_overall_rmse.mean())
        se_all = float(per_shot_overall_rmse.std(ddof=1) / np.sqrt(per_shot_overall_rmse.shape[0])) if per_shot_overall_rmse.shape[0] > 1 else 0.0

        results[group_key] = {
            "mean": mean,
            "se": se,
            "mean_all": mean_all,
            "se_all": se_all,
            "num_shots": rmse_array.shape[0],
            "num_files": grouped_file_counts.get(group_key, 0),
        }

    if not results:
        print("No tracking-error results were found.")
        return

    max_components = max(result["mean"].shape[0] for result in results.values())
    columns = []
    for i in range(max_components):
        columns.append(f"component_{i + 1}_rmse_mean")
        columns.append(f"component_{i + 1}_rmse_se")
    columns.extend(["overall_rmse_mean", "overall_rmse_se", "num_shots", "num_files"])

    df = pd.DataFrame(index=results.keys(), columns=columns)
    for group_key in results.keys():
        mean = results[group_key]["mean"]
        se = results[group_key]["se"]
        for i in range(mean.shape[0]):
            df.loc[group_key, f"component_{i + 1}_rmse_mean"] = mean[i]
            df.loc[group_key, f"component_{i + 1}_rmse_se"] = se[i]
        df.loc[group_key, "overall_rmse_mean"] = results[group_key]["mean_all"]
        df.loc[group_key, "overall_rmse_se"] = results[group_key]["se_all"]
        df.loc[group_key, "num_shots"] = results[group_key]["num_shots"]
        df.loc[group_key, "num_files"] = results[group_key]["num_files"]

    df.index.name = "Algorithm"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"\nSaved summary table to:\n{output_path}")
    print(df.head(3))


def main():
    args = get_args()

    if args.mode == "error":
        for seed in args.seeds:
            print(f"\nSummarizing error for seed = {seed}")
            args.seed = seed
            log_folder = _build_profile_log_folder(args)
            summarize_tracking_error(log_folder, args.task, os.path.join(log_folder, "tracking_error_summary.csv"))
        return

    for seed in args.seeds:
        print(f"\nRunning experiment with seed = {seed}")
        args.seed = seed
        run_profile_mode(args)


if __name__ == "__main__":
    main()
