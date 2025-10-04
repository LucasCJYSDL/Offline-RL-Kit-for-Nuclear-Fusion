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
from visualization.controller import Controller
from visualization.comparison_plotter import (
    plot_comparison_tracking_quantities, 
    plot_comparison_actions,
    plot_metrics_comparison,
    plot_summary_metrics
)


def get_args():
    parser = argparse.ArgumentParser(description="PPO agents comparison arguments")

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=3, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="fusion_env") 
    parser.add_argument("--task", type=str, default="rotation", help="Targets to track") 

    # old PPO controller settings
    parser.add_argument("--old_actor_path", type=str, 
                       default="/home/scratch/jiayuc2/rl_out/prof_tracking_dens_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/policy", 
                       help="Path to the old actor checkpoint")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[250, 250], help="Hidden dimensions of the actor network")
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    
    # new PPO data settings
    parser.add_argument("--new_ppo_data_path", type=str,
                       default="saved_data/dens/dens_ppo_seed_1&timestamp_25-0929-074550/shot_data.npz",
                       help="Path to the saved new PPO data")

    # comparison settings
    parser.add_argument("--comparison_name", type=str, default="old target",
                       help="Custom name for comparison (optional)")

    return parser.parse_args()


def calculate_tracking_metrics(target_array, current_array):
    """计算跟踪性能指标"""
    target_array = np.array(target_array)
    current_array = np.array(current_array)

    # 基本误差计算
    error = target_array - current_array
    abs_error = np.abs(error)

    metrics = {
        # 1. 均方根误差 (RMSE)
        'rmse': float(np.sqrt(np.mean(error**2))),

        # 2. 平均绝对误差 (MAE)
        'mae': float(np.mean(abs_error)),

        # 3. 累计绝对误差
        'cumulative_absolute_error': float(np.sum(abs_error)),

        # 4. 累计平方误差
        'cumulative_squared_error': float(np.sum(error**2))
    }

    return metrics


def print_comparison_metrics(shot_id, old_metrics, new_metrics):
    """打印比较指标"""
    print(f"\nShot #{shot_id} - Metrics Comparison:")
    print(f"  Old PPO vs Target:")
    print(f"    RMSE: {old_metrics['rmse']:.4f}, MAE: {old_metrics['mae']:.4f}")
    print(f"  New PPO vs Target:")
    print(f"    RMSE: {new_metrics['rmse']:.4f}, MAE: {new_metrics['mae']:.4f}")
    print(f"  Improvement (New - Old):")
    print(f"    RMSE: {new_metrics['rmse'] - old_metrics['rmse']:.4f}")
    print(f"    MAE: {new_metrics['mae'] - old_metrics['mae']:.4f}")


def load_new_ppo_data(data_path):
    """
    Load new PPO data from npz file
    
    Args:
        data_path: Path to the npz file
        
    Returns:
        Dictionary containing shot data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"New PPO data file not found: {data_path}")
    
    # Load npz file
    data = np.load(data_path, allow_pickle=True)
    
    # Convert to dictionary
    shot_data = {}
    for key in data.files:
        shot_data[key] = data[key].item() if data[key].ndim == 0 else data[key]
    
    print(f"Loaded new PPO data from: {data_path}")
    print(f"Available shots: {[k for k in shot_data.keys() if k.startswith('shot_')]}")
    
    return shot_data


def save_comparison_results(comparison_results, log_folder):
    """
    Save comparison results to JSON and CSV files
    
    Args:
        comparison_results: Dictionary containing comparison results
        log_folder: Directory to save files
    """
    # Save results to JSON file
    results_file = os.path.join(log_folder, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Save to CSV file
    try:
        import pandas as pd
        df_data = []
        for shot_key, shot_data in comparison_results.items():
            if shot_key.startswith('shot_'):
                row = {
                    'shot_id': shot_data['shot_id'],
                    'old_ppo_rmse': shot_data['old_ppo_vs_target']['rmse'],
                    'old_ppo_mae': shot_data['old_ppo_vs_target']['mae'],
                    'old_ppo_cum_abs_error': shot_data['old_ppo_vs_target']['cumulative_absolute_error'],
                    'old_ppo_cum_sq_error': shot_data['old_ppo_vs_target']['cumulative_squared_error'],
                    'new_ppo_rmse': shot_data['new_ppo_vs_target']['rmse'],
                    'new_ppo_mae': shot_data['new_ppo_vs_target']['mae'],
                    'new_ppo_cum_abs_error': shot_data['new_ppo_vs_target']['cumulative_absolute_error'],
                    'new_ppo_cum_sq_error': shot_data['new_ppo_vs_target']['cumulative_squared_error'],
                    'rmse_improvement': shot_data['new_ppo_vs_target']['rmse'] - shot_data['old_ppo_vs_target']['rmse'],
                    'mae_improvement': shot_data['new_ppo_vs_target']['mae'] - shot_data['old_ppo_vs_target']['mae']
                }
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = os.path.join(log_folder, 'comparison_results.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"\nComparison results saved to:")
        print(f"JSON format: {results_file}")
        print(f"CSV format: {csv_file}")
        
    except ImportError:
        print(f"\nComparison results saved to:")
        print(f"JSON format: {results_file}")
        print("Note: pandas not installed, cannot save CSV format")


def run(args=get_args()) -> None:
    # register an env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, _ = get_rl_data_envs(args.env, args.task, args.device, is_il=args.il_actor)
    args.obs_dim = offline_data['obs_dim']
    args.action_dim = offline_data['act_dim']
    args.max_action = 1.0
    args.actor_path = args.old_actor_path  # Set for controller

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # load up the old PPO actor
    controller = Controller(args)
    
    # Load new PPO data
    new_ppo_data_path = os.path.join(os.path.dirname(__file__), args.new_ppo_data_path)
    new_ppo_data = load_new_ppo_data(new_ppo_data_path)
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names()

    # Create log folder for comparison results with seed information
    # Extract seed information from old actor path and new PPO data path
    old_actor_info = args.old_actor_path.split('/')[-1] if args.old_actor_path else "unknown_old"
    new_ppo_info = os.path.basename(os.path.dirname(args.new_ppo_data_path))

    # Create a unique comparison identifier
    if args.comparison_name:
        comparison_id = args.comparison_name
    else:
        comparison_id = f"old_{old_actor_info}_vs_new_{new_ppo_info}_seed_{args.seed}"

    log_folder = os.path.join(os.path.dirname(__file__), "results", "comparison", args.task, comparison_id)
    os.makedirs(log_folder, exist_ok=True)

    print(f"Comparison results will be saved to: {log_folder}")

    # Storage for comparison results
    comparison_results = {}
    old_ppo_summary = {'total_shots': 0, 'total_rmse': 0, 'total_mae': 0, 'total_reward': 0}
    new_ppo_summary = {'total_shots': 0, 'total_rmse': 0, 'total_mae': 0, 'total_reward': 0}

    for shot in shot_list:
        # Check if new PPO data exists for this shot
        shot_key = f'shot_{shot}'
        if shot_key not in new_ppo_data:
            print(f"Warning: No new PPO data found for shot {shot}, skipping...")
            continue
            
        # Run old PPO agent
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0

        time_array = []
        target_quan_array, real_quan_array, old_ppo_quan_array, real_act_array, old_ppo_act_array = [], [], [], [], []

        while True:
            action = controller.act(obs)
            next_obs, actuators, reward, terminal, info = env.step(action, obs)
            episode_reward += reward
            episode_length += 1

            # get quantities to plot
            cur_time = info["time_step"] - 1
            target_quan, real_quan, cur_quan, real_act, cur_act = sa_processor.get_plot_quantities(shot, cur_time, obs, action)
            time_array.append(cur_time)
            target_quan_array.append(target_quan)
            real_quan_array.append(real_quan)
            old_ppo_quan_array.append(cur_quan)
            real_act_array.append(real_act)
            old_ppo_act_array.append(cur_act)

            if terminal:
                break

            obs = next_obs
        
        # Convert to numpy arrays
        target_quan_array = np.array(target_quan_array)
        real_quan_array = np.array(real_quan_array)
        old_ppo_quan_array = np.array(old_ppo_quan_array)
        old_ppo_act_array = np.array(old_ppo_act_array)
        time_array = np.array(time_array)
        
        # Get new PPO data
        new_ppo_quan_array = new_ppo_data[shot_key]['cur_quan_array']
        new_ppo_act_array = new_ppo_data[shot_key]['cur_act_array']
        
        # Calculate metrics
        old_ppo_metrics = calculate_tracking_metrics(target_quan_array, old_ppo_quan_array)
        new_ppo_metrics = calculate_tracking_metrics(target_quan_array, new_ppo_quan_array)
        
        # Store comparison results
        comparison_results[shot_key] = {
            'shot_id': shot,
            'old_ppo_vs_target': old_ppo_metrics,
            'new_ppo_vs_target': new_ppo_metrics,
            'old_ppo_reward': episode_reward,
            'new_ppo_reward': new_ppo_data[shot_key]['episode_reward']
        }
        
        # Update summary statistics
        old_ppo_summary['total_shots'] += 1
        old_ppo_summary['total_rmse'] += old_ppo_metrics['rmse']
        old_ppo_summary['total_mae'] += old_ppo_metrics['mae']
        old_ppo_summary['total_reward'] += episode_reward
        
        new_ppo_summary['total_shots'] += 1
        new_ppo_summary['total_rmse'] += new_ppo_metrics['rmse']
        new_ppo_summary['total_mae'] += new_ppo_metrics['mae']
        new_ppo_summary['total_reward'] += new_ppo_data[shot_key]['episode_reward']
        
        # Generate comparison plots
        plot_comparison_tracking_quantities(
            time_array, target_quan_array, real_quan_array, 
            old_ppo_quan_array, new_ppo_quan_array, 
            quan_names, shot, log_folder
        )
        
        if args.plot_actuators:
            plot_comparison_actions(
                time_array, real_act_array, old_ppo_act_array, 
                new_ppo_act_array, act_names, shot, log_folder
            )
        
        # Print comparison metrics
        print_comparison_metrics(shot, old_ppo_metrics, new_ppo_metrics)

    # Calculate and save summary metrics
    if old_ppo_summary['total_shots'] > 0:
        summary_metrics = {
            'old_ppo': {
                'avg_rmse': old_ppo_summary['total_rmse'] / old_ppo_summary['total_shots'],
                'avg_mae': old_ppo_summary['total_mae'] / old_ppo_summary['total_shots'],
                'avg_reward': old_ppo_summary['total_reward'] / old_ppo_summary['total_shots']
            },
            'new_ppo': {
                'avg_rmse': new_ppo_summary['total_rmse'] / new_ppo_summary['total_shots'],
                'avg_mae': new_ppo_summary['total_mae'] / new_ppo_summary['total_shots'],
                'avg_reward': new_ppo_summary['total_reward'] / new_ppo_summary['total_shots']
            }
        }
        
        comparison_results['summary'] = summary_metrics
        
        # Generate summary plots
        plot_metrics_comparison(comparison_results, log_folder)
        plot_summary_metrics(summary_metrics, log_folder)
        
        # Print summary
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"Total shots compared: {old_ppo_summary['total_shots']}")
        print(f"\nOld PPO Agent:")
        print(f"  Average RMSE: {summary_metrics['old_ppo']['avg_rmse']:.4f}")
        print(f"  Average MAE: {summary_metrics['old_ppo']['avg_mae']:.4f}")
        print(f"  Average Reward: {summary_metrics['old_ppo']['avg_reward']:.3f}")
        print(f"\nNew PPO Agent:")
        print(f"  Average RMSE: {summary_metrics['new_ppo']['avg_rmse']:.4f}")
        print(f"  Average MAE: {summary_metrics['new_ppo']['avg_mae']:.4f}")
        print(f"  Average Reward: {summary_metrics['new_ppo']['avg_reward']:.3f}")
        print(f"\nImprovement (New - Old):")
        print(f"  RMSE: {summary_metrics['new_ppo']['avg_rmse'] - summary_metrics['old_ppo']['avg_rmse']:.4f}")
        print(f"  MAE: {summary_metrics['new_ppo']['avg_mae'] - summary_metrics['old_ppo']['avg_mae']:.4f}")
        print(f"  Reward: {summary_metrics['new_ppo']['avg_reward'] - summary_metrics['old_ppo']['avg_reward']:.3f}")

    # Save comparison results
    save_comparison_results(comparison_results, log_folder)


if __name__ == "__main__":
    run()
