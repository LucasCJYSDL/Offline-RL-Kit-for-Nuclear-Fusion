import os
import torch
import random
import pickle as pkl
import numpy as np
from tqdm import tqdm
import argparse
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_preparation.get_rl_data_envs import get_rl_data_envs
from visualization.controller_new import Controller
from visualization.plotter import plot_tracking_quantities, plot_actions, plot_tracking_quantities_with_units
from envs.utils.profile_util import reconstruct_profile_from_state


#!!! what you need to specify
def get_args():
    parser = argparse.ArgumentParser(description="Trajectory evaluation arguments")

    # basic settings
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=5, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="profile_control") 
    parser.add_argument("--task", type=str, default="rotation", help="Targets to track") 

    # controller settings, of which the core is an NN actor
    #parser.add_argument("--actor_path", type=str, default="log/dens/cql/seed_1&timestamp_25-1004-113803", help="Path to the actor checkpoint")
    parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/bao/Training_untuned/rotation/rombrl&grad_mode=1&sl_weight=1000.0&actor_training_epoch=10&onpolicy_rollout_batch_size=2500&onpolicy_rollout_length=10&small_traj_batch=False/seed_1&timestamp_25-1031-122233", help="Path to the actor checkpoint")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256,256], help="Hidden dimensions of the actor network") # you can get this in corresponding rl scripts
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    parser.add_argument("--use_diag_gaussian", action="store_true", help="Use DiagGaussian instead of TanhDiagGaussian (required for IQL)")

    parser.add_argument("--save_dir_name", type=str, default="test", help="Subfolder name for saving results")
    parser.add_argument("--output_base_dir", type=str, default="/home/scratch/jiayuc2/bao/Eval_untuned", help="Base directory for output files")
    return parser.parse_args()


def calculate_tracking_metrics(target_array, current_array):
    """Calculate tracking performance metrics

    Args:
        target_array: shape (time_steps, num_dimensions)
        current_array: shape (time_steps, num_dimensions)

    Returns:
        metrics: Dictionary containing overall metrics and per-dimension metrics
    """
    target_array = np.array(target_array)
    current_array = np.array(current_array)

    # Basic error calculation
    error = target_array - current_array
    abs_error = np.abs(error)

    # Overall metrics (averaged across all dimensions)
    metrics = {
        # 1. Root Mean Square Error (RMSE)
        'rmse': float(np.sqrt(np.mean(error**2))),

        # 2. Mean Absolute Error (MAE)
        'mae': float(np.mean(abs_error)),

        # 3. Cumulative Absolute Error
        'cumulative_absolute_error': float(np.sum(abs_error)),

        # 4. Cumulative Squared Error
        'cumulative_squared_error': float(np.sum(error**2))
    }

    # Per-dimension metrics
    num_dimensions = target_array.shape[1] if len(target_array.shape) > 1 else 1

    if num_dimensions > 1:
        metrics['per_component'] = {}
        for dim in range(num_dimensions):
            component_name = f'component{dim + 1}'
            error_dim = error[:, dim]
            abs_error_dim = abs_error[:, dim]

            metrics['per_component'][component_name] = {
                'rmse': float(np.sqrt(np.mean(error_dim**2))),
                'mae': float(np.mean(abs_error_dim)),
                'cumulative_absolute_error': float(np.sum(abs_error_dim)),
                'cumulative_squared_error': float(np.sum(error_dim**2))
            }
    else:
        # If only one dimension, also add component1
        metrics['per_component'] = {
            'component1': {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'cumulative_absolute_error': metrics['cumulative_absolute_error'],
                'cumulative_squared_error': metrics['cumulative_squared_error']
            }
        }

    return metrics


def print_tracking_metrics(shot_id, metrics, episode_reward, episode_length):
    """Print tracking metrics"""
    print(f"\nShot #{shot_id} - Episode Reward: {episode_reward:.3f}, Length: {episode_length}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Cumulative Absolute Error: {metrics['cumulative_absolute_error']:.2f}")
    print(f"  Cumulative Squared Error: {metrics['cumulative_squared_error']:.2f}")


def save_tracking_results(all_results, log_folder):
    """
    Save tracking results to JSON and CSV files

    Args:
        all_results: Dictionary containing all shot results
        log_folder: Directory to save files
    """

    shot_results = [v for k, v in all_results.items() if k.startswith('shot_')]
    num_shots = len(shot_results)

    if num_shots > 0:
        # Calculate overall average metrics
        summary_metrics = {
            'total_shots': num_shots,
            'avg_rmse': sum(shot['tracking_metrics']['rmse'] for shot in shot_results) / num_shots,
            'avg_mae': sum(shot['tracking_metrics']['mae'] for shot in shot_results) / num_shots,
            'avg_cumulative_absolute_error': sum(shot['tracking_metrics']['cumulative_absolute_error'] for shot in shot_results) / num_shots,
            'avg_reward': sum(shot['episode_reward'] for shot in shot_results) / num_shots,
            'avg_length': sum(shot['episode_length'] for shot in shot_results) / num_shots
        }

        # Calculate average metrics for each dimension
        if 'per_component' in shot_results[0]['tracking_metrics']:
            component_names = list(shot_results[0]['tracking_metrics']['per_component'].keys())
            summary_metrics['per_component'] = {}

            for comp_name in component_names:
                summary_metrics['per_component'][comp_name] = {
                    'avg_rmse': sum(shot['tracking_metrics']['per_component'][comp_name]['rmse']
                                   for shot in shot_results) / num_shots,
                    'avg_mae': sum(shot['tracking_metrics']['per_component'][comp_name]['mae']
                                  for shot in shot_results) / num_shots,
                    'avg_cumulative_absolute_error': sum(shot['tracking_metrics']['per_component'][comp_name]['cumulative_absolute_error']
                                                         for shot in shot_results) / num_shots,
                }

        all_results['summary'] = summary_metrics

    # Save results to JSON file
    results_file = os.path.join(log_folder, 'tracking_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save to CSV file
    try:
        import pandas as pd
        df_data = []
        for shot_key, shot_data in all_results.items():
            if shot_key != 'summary':
                row = {
                    'shot_id': shot_data['shot_id'],
                    'episode_reward': shot_data['episode_reward'],
                    'episode_length': shot_data['episode_length'],
                    'rmse': shot_data['tracking_metrics']['rmse'],
                    'mae': shot_data['tracking_metrics']['mae'],
                    'cumulative_absolute_error': shot_data['tracking_metrics']['cumulative_absolute_error'],
                    'cumulative_squared_error': shot_data['tracking_metrics']['cumulative_squared_error']
                }

                # Add per-dimension metrics
                if 'per_component' in shot_data['tracking_metrics']:
                    for comp_name, comp_metrics in shot_data['tracking_metrics']['per_component'].items():
                        row[f'{comp_name}_rmse'] = comp_metrics['rmse']
                        row[f'{comp_name}_mae'] = comp_metrics['mae']
                        row[f'{comp_name}_cumulative_absolute_error'] = comp_metrics['cumulative_absolute_error']

                df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_file = os.path.join(log_folder, 'tracking_results.csv')
        df.to_csv(csv_file, index=False)

        print(f"\nResults saved to:")
        print(f"JSON format: {results_file}")
        print(f"CSV format: {csv_file}")

    except ImportError:
        print(f"\nResults saved to:")
        print(f"JSON format: {results_file}")
        print("Note: pandas not installed, cannot save CSV format")

    if 'summary' in all_results:
        summary = all_results['summary']
        print(f"\nOverall Statistics:")
        print(f"Average RMSE: {summary['avg_rmse']:.4f}")
        print(f"Average MAE: {summary['avg_mae']:.4f}")
        print(f"Average Cumulative Absolute Error: {summary['avg_cumulative_absolute_error']:.4f}")
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(f"Average Length: {summary['avg_length']:.1f}")

        # Print average metrics for each dimension
        if 'per_component' in summary:
            print(f"\nPer-Component Average Statistics:")
            for comp_name, comp_metrics in summary['per_component'].items():
                print(f"  {comp_name}:")
                print(f"    Average RMSE: {comp_metrics['avg_rmse']:.4f}")
                print(f"    Average MAE: {comp_metrics['avg_mae']:.4f}")
                print(f"    Average Cumulative Absolute Error: {comp_metrics['avg_cumulative_absolute_error']:.4f}")


def run(args=get_args()) -> None:
    # register an env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, _ = get_rl_data_envs(args.env, args.task, args.device, is_il=args.il_actor) # these are the data and env used to train the actor
    args.obs_dim = offline_data['observations'].shape[1]
    args.action_dim = offline_data['actions'].shape[1]
    args.max_action = 1.0

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # load up the actor
    controller = Controller(args)

    all_results = {}
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names() # name of the quantities to track and actuators in control

    # Extract path information from actor_path to build save directory
    # Expected format: /home/scratch/jiayuc2/bao/Training_untuned/task/algo/seed_X&timestamp_Y
    actor_path_parts = args.actor_path.split('/')

    # Extract task, algo, and seed_timestamp from the path
    
    # Find "Training_untuned" index to extract task/algo/seed_timestamp
    training_idx = actor_path_parts.index('Training_untuned')
    task_name = actor_path_parts[training_idx + 1]
    algo_name = actor_path_parts[training_idx + 2]
    seed_timestamp = actor_path_parts[training_idx + 3]

        # Build final save path: output_base_dir/task/algo/seed_timestamp
    log_folder = os.path.join(args.output_base_dir, task_name, algo_name, seed_timestamp)
    # except (ValueError, IndexError):
    #     # Fallback: use the last three path parts as task/algo/seed_timestamp
    #     if len(actor_path_parts) >= 3:
    #         task_name = actor_path_parts[-3]
    #         algo_name = actor_path_parts[-2]
    #         seed_timestamp = actor_path_parts[-1]
    #         log_folder = os.path.join(args.output_base_dir, task_name, algo_name, seed_timestamp)
    #     else:
    #         # Last resort: use save_dir_name
    #         log_folder = os.path.join(args.output_base_dir, args.save_dir_name)

    os.makedirs(log_folder, exist_ok=True)
    print(f"\nResults will be saved to: {log_folder}")

    for shot in shot_list:
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0

        time_array = []
        target_quan_array, real_quan_array, cur_quan_array, real_act_array, cur_act_array = [], [], [], [], []
        reconstruct_target_quan_array, reconstruct_real_quan_array, reconstruct_cur_quan_array, reconstruct_real_act_array, reconstruct_cur_act_array = [], [], [], [], []
        while True:
            action = controller.act(obs)
            next_obs, reward, terminal, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # get quantities to plot
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
         # Calculate tracking metrics
        reconstruct_real_quan_array = reconstruct_profile_from_state(args.task, np.array(real_quan_array), env.info, sa_processor.idx_list, unnormalize=False)
        reconstruct_target_quan_array = reconstruct_profile_from_state(args.task, np.array(target_quan_array), env.info, sa_processor.idx_list, unnormalize=False)
        reconstruct_cur_quan_array = reconstruct_profile_from_state(args.task, np.array(cur_quan_array), env.info, sa_processor.idx_list, unnormalize=False)

        # Convert to numpy arrays
        target_quan_array = np.array(target_quan_array)
        cur_quan_array = np.array(cur_quan_array)
        cur_act_array = np.array(cur_act_array)
        reconstruct_target_quan_array = np.array(reconstruct_target_quan_array)
        reconstruct_cur_quan_array = np.array(reconstruct_cur_quan_array)
        time_array = np.array(time_array)
        #print(target_quan_array)

        # Calculate quantitative metrics
        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        
        all_results[f'shot_{shot}'] = {
            'shot_id': shot,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'tracking_metrics': tracking_metrics
        }
        
        # make plots
        plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot, log_folder)
        plot_tracking_quantities_with_units(time_array,  reconstruct_target_quan_array, real_quan_array, reconstruct_cur_quan_array, args.task, shot, log_folder)
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, shot, log_folder)

        # Print detailed metrics
        print_tracking_metrics(shot, tracking_metrics, episode_reward, episode_length)

    save_tracking_results(all_results, log_folder)

if __name__ == "__main__":
    run()