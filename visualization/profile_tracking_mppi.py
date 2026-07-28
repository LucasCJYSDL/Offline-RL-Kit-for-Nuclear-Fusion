import argparse
import json
import os
import random
import sys

import h5py
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.utils.profile_util import reconstruct_profile_from_state
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from visualization.plotter import make_paper_quality_prof_fig, plot_actions, plot_tracking_quantities_prof


def get_args():
    parser = argparse.ArgumentParser(description="Profile tracking plots for saved MPPI rollouts")
    parser.add_argument("--planning-data-path", "--planning_data_path", dest="planning_data_path", required=True)
    parser.add_argument("--save_base_dir", type=str, default="/export/ra/baohaoming/fusion/profile/rotation")
    parser.add_argument("--mppi-run-name", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda_id", type=int, default=5)
    parser.add_argument("--plot_actuators", type=bool, default=False)
    parser.add_argument("--episode-index", type=int, default=0)

    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--test", dest="test", action="store_true", default=None, help="Use test split instead of validation split")
    parser.add_argument("--val", dest="test", action="store_false", help="Use validation split")
    return parser.parse_args()


def get_device(cuda_id):
    if cuda_id == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)


def calculate_tracking_metrics(target_array, current_array):
    target_array = np.array(target_array)
    current_array = np.array(current_array)
    error = target_array - current_array
    abs_error = np.abs(error)

    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "cumulative_absolute_error": float(np.sum(abs_error)),
        "cumulative_squared_error": float(np.sum(error**2)),
    }

    num_dimensions = target_array.shape[1] if len(target_array.shape) > 1 else 1
    metrics["per_component"] = {}
    if num_dimensions > 1:
        for dim in range(num_dimensions):
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
        metrics["per_component"]["component1"] = {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "cumulative_absolute_error": metrics["cumulative_absolute_error"],
            "cumulative_squared_error": metrics["cumulative_squared_error"],
        }

    return metrics


def calculate_reconstruct_tracking_metrics(target_array, current_array):
    target_array = np.array(target_array)
    current_array = np.array(current_array)
    return np.mean((target_array - current_array) ** 2, axis=0)


def print_tracking_metrics(shot_id, metrics, episode_reward, episode_length):
    print(f"\nShot #{shot_id} - Episode Reward: {episode_reward:.3f}, Length: {episode_length}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Cumulative Absolute Error: {metrics['cumulative_absolute_error']:.2f}")
    print(f"  Cumulative Squared Error: {metrics['cumulative_squared_error']:.2f}")


def save_tracking_results(all_results, log_folder):
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
                    "avg_rmse": sum(
                        shot["tracking_metrics"]["per_component"][comp_name]["rmse"]
                        for shot in shot_results
                    ) / num_shots,
                    "avg_mae": sum(
                        shot["tracking_metrics"]["per_component"][comp_name]["mae"]
                        for shot in shot_results
                    ) / num_shots,
                    "avg_cumulative_absolute_error": sum(
                        shot["tracking_metrics"]["per_component"][comp_name]["cumulative_absolute_error"]
                        for shot in shot_results
                    ) / num_shots,
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
            for comp_name, comp_metrics in shot_data["tracking_metrics"].get("per_component", {}).items():
                row[f"{comp_name}_rmse"] = comp_metrics["rmse"]
                row[f"{comp_name}_mae"] = comp_metrics["mae"]
                row[f"{comp_name}_cumulative_absolute_error"] = comp_metrics["cumulative_absolute_error"]
            df_data.append(row)

        csv_file = os.path.join(log_folder, "tracking_results.csv")
        pd.DataFrame(df_data).to_csv(csv_file, index=False)
        print(f"\nResults saved to:")
        print(f"JSON format: {results_file}")
        print(f"CSV format: {csv_file}")
    except ImportError:
        print(f"\nResults saved to:")
        print(f"JSON format: {results_file}")
        print("Note: pandas not installed, cannot save CSV format")

    if "summary" in all_results:
        summary = all_results["summary"]
        print(f"\nOverall Statistics:")
        print(f"Average RMSE: {summary['avg_rmse']:.4f}")
        print(f"Average MAE: {summary['avg_mae']:.4f}")
        print(f"Average Cumulative Absolute Error: {summary['avg_cumulative_absolute_error']:.4f}")
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(f"Average Length: {summary['avg_length']:.1f}")


def save_shot_data(shot_data_dict, save_folder, task_name, actor_info):
    data_save_dir = os.path.join(save_folder, "saved_data")
    os.makedirs(data_save_dir, exist_ok=True)

    npz_file = os.path.join(data_save_dir, "shot_data.npz")
    np.savez(npz_file, **shot_data_dict)

    metadata = {
        "task_name": task_name,
        "actor_info": actor_info,
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


def read_planning_metadata(path):
    with h5py.File(path, "r") as hdf:
        return {key: hdf.attrs[key] for key in hdf.attrs}


def attr_to_str(value, default):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def derive_output_parts(args, metadata):
    log_dirs = metadata.get("log_dirs")
    source_dir = attr_to_str(log_dirs, None) if log_dirs else os.path.dirname(os.path.abspath(args.planning_data_path))

    mppi_run_name = args.mppi_run_name
    run_id = args.run_id
    if mppi_run_name is None:
        mppi_run_name = os.path.basename(os.path.dirname(source_dir))
    if run_id is None:
        run_id = os.path.basename(source_dir)
    return mppi_run_name, run_id


def extract_algorithm_name(mppi_run_name: str) -> str:
    candidate = os.path.basename(os.path.normpath(mppi_run_name))
    return candidate.split("&", 1)[0].upper()


def get_available_shots(path):
    with h5py.File(path, "r") as hdf:
        if "shots" not in hdf:
            raise ValueError(f"{path} does not contain per-shot MPPI rollouts under 'shots'")
        return {int(shot_id) for shot_id in hdf["shots"].keys()}


def load_rollout(path, shot_id, episode_index):
    with h5py.File(path, "r") as hdf:
        episode_key = f"episode_{episode_index}"
        shot_group = hdf["shots"][str(shot_id)]
        if episode_key not in shot_group:
            available = sorted(shot_group.keys())
            raise ValueError(
                f"Shot {shot_id} has no {episode_key}; available episodes: {available}"
            )
        episode_group = shot_group[episode_key]
        return {key: episode_group[key][:] for key in episode_group.keys()}


def as_batch_tensor(array, device):
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[None, :]
    return torch.FloatTensor(array).to(device)


def as_action_batch(array):
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[None, :]
    return array


def run(args=None):
    if args is None:
        args = get_args()

    metadata = read_planning_metadata(args.planning_data_path)
    if args.env is None:
        args.env = attr_to_str(metadata.get("env"), "profile_control_new")
    if args.task is None:
        args.task = attr_to_str(metadata.get("task"), "temp")
    if args.test is None:
        args.test = attr_to_str(metadata.get("split"), "val") == "test"
    if args.seed is None:
        args.seed = int(metadata.get("seed", 0))

    args.device = get_device(args.cuda_id)
    offline_data, sa_processor, env, _ = get_rl_data_envs(
        args.env,
        args.task,
        args.device,
        is_val=not args.test,
    )
    args.obs_dim = offline_data["obs_dim"] if args.env == "fusion_env" else offline_data["observations"].shape[1]
    args.action_dim = offline_data["act_dim"] if args.env == "fusion_env" else offline_data["actions"].shape[1]
    args.max_action = 1.0

    set_seed(args.seed, env)

    mppi_run_name, run_id = derive_output_parts(args, metadata)
    algo_name = extract_algorithm_name(mppi_run_name)
    split_name = "test" if args.test else "val"
    log_folder = os.path.join(
        args.save_base_dir,
        mppi_run_name,
        args.task,
        split_name,
        "results",
        run_id,
        str(args.seed),
    )
    os.makedirs(log_folder, exist_ok=True)

    all_results = {}
    shot_data_dict = {}
    available_shots = get_available_shots(args.planning_data_path)
    shot_list = [shot for shot in env.get_eval_shot_list() if int(shot) in available_shots]
    if not shot_list:
        raise ValueError("No environment evaluation shots were found in the MPPI planning data")

    _, act_names = sa_processor.get_plot_names()
    acts_in_use = ["pinj", "tinj", "gasA", "ech_pwr_total"]

    for shot in shot_list:
        rollout = load_rollout(args.planning_data_path, int(shot), args.episode_index)
        observations = rollout["observations"]
        actions = rollout["actions"]
        rewards = np.asarray(rollout["rewards"])
        time_steps = rollout.get("time_steps", np.arange(len(actions)))

        time_array = []
        target_quan_array = []
        real_quan_array = []
        cur_quan_array = []
        real_act_array = []
        cur_act_array = []

        for obs, action, cur_time in zip(observations, actions, time_steps):
            obs_tensor = as_batch_tensor(obs, args.device)
            action_batch = as_action_batch(action)
            cur_time = int(cur_time)
            target_quan, real_quan, cur_quan, real_act, cur_act = sa_processor.get_plot_quantities(
                int(shot),
                cur_time,
                obs_tensor,
                action_batch,
            )
            time_array.append(cur_time)
            target_quan_array.append(target_quan)
            real_quan_array.append(real_quan)
            cur_quan_array.append(cur_quan)
            real_act_array.append(real_act)
            cur_act_array.append(cur_act)

        reconstruct_real_quan_array = reconstruct_profile_from_state(
            args.task,
            np.array(real_quan_array),
            env.info,
            sa_processor.idx_list,
            unnormalize=False,
        )
        reconstruct_target_quan_array = reconstruct_profile_from_state(
            args.task,
            np.array(target_quan_array),
            env.info,
            sa_processor.idx_list,
            unnormalize=True,
        )
        reconstruct_cur_quan_array = reconstruct_profile_from_state(
            args.task,
            np.array(cur_quan_array),
            env.info,
            sa_processor.idx_list,
            unnormalize=True,
        )

        target_quan_array = np.array(target_quan_array)
        cur_quan_array = np.array(cur_quan_array)
        cur_act_array = np.array(cur_act_array)
        reconstruct_target_quan_array = np.array(reconstruct_target_quan_array)
        reconstruct_cur_quan_array = np.array(reconstruct_cur_quan_array)
        time_array = np.array(time_array)

        episode_reward = float(np.sum(rewards))
        episode_length = int(len(actions))
        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        mse_reconstruct = calculate_reconstruct_tracking_metrics(
            reconstruct_target_quan_array,
            reconstruct_cur_quan_array,
        )

        all_results[f"shot_{shot}"] = {
            "shot_id": int(shot),
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "tracking_metrics": tracking_metrics,
        }

        shot_data_dict[f"shot_{shot}"] = {
            "shot_id": int(shot),
            "cur_quan_array": cur_quan_array,
            "cur_act_array": cur_act_array,
            "reconstruct_cur_quan_array": reconstruct_cur_quan_array,
            "reconstruct_target_quan_array": reconstruct_target_quan_array,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "mse_reconstruct": mse_reconstruct,
        }

        times = sa_processor.times[int(shot)]
        plot_tracking_quantities_prof(
            times,
            reconstruct_target_quan_array,
            real_quan_array,
            reconstruct_cur_quan_array,
            args.task,
            algo_name,
            int(shot),
            log_folder,
        )
        make_paper_quality_prof_fig(
            times,
            reconstruct_target_quan_array,
            reconstruct_cur_quan_array,
            cur_act_array,
            args.task,
            algo_name,
            int(shot),
            log_folder,
            env.info,
            acts_in_use,
        )
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, int(shot), log_folder)

        print_tracking_metrics(int(shot), tracking_metrics, episode_reward, episode_length)

    save_tracking_results(all_results, log_folder)
    actor_info = f"{mppi_run_name}/{run_id}/episode_{args.episode_index}"
    save_shot_data(shot_data_dict, log_folder, args.task, actor_info)
    return log_folder


if __name__ == "__main__":
    run()
