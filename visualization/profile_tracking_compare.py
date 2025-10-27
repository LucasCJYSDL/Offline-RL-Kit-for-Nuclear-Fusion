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
    plot_comparison_tracking_quantities_with_units,
    plot_comparison_actions,
    plot_metrics_comparison,
    plot_summary_metrics
)
from  envs.utils.profile_util import reconstruct_profile_from_state

def get_args():
    parser = argparse.ArgumentParser(description="PPO agents comparison arguments")
    parser.add_argument("--save_base_dir", type=str, 
                       default="/home/scratch/jiayuc2",
                       help="Base directory for saving all results")

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=3, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="fusion_env") 
    parser.add_argument("--task", type=str, default="dens", help="Targets to track") 

    # old PPO controller settings
    parser.add_argument("--old_actor_path", type=str, 
                       default="/home/scratch/jiayuc2/rl_out/prof_tracking_dens_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/policy", 
                       help="Path to the old actor checkpoint")
    #dens path
    # /home/scratch/jiayuc2/rl_out/prof_tracking_dens_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/policy
    #rot path
    # /home/scratch/jiayuc2/rl_out/prof_tracking_rot_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/601/policy
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[250, 250], help="Hidden dimensions of the actor network")
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    
    # new PPO data settings
    parser.add_argument("--new_ppo_data_path", type=str,
                       default="/home/scratch/jiayuc2/temp_new/saved_data/dens/home_scratch_jiayuc2_rl_out_off_test_bao_dens_network_dens_ppo_seed1_lr0.003_steps2048_batch1024_epochs20_gamma0.952_gaelambda0.98_clip0.148_ent0.0067_vf1_maxgrad0.5_timesteps800000_pol250x250_val250x250/shot_data.npz",
                       help="Path to the saved new PPO data")

    # comparison settings
    parser.add_argument("--comparison_name", type=str, default="old target",
                       help="Custom name for comparison (optional)")

    return parser.parse_args()


# MODIFIED FUNCTION
def calculate_tracking_metrics(target_array, current_array):
    """
    计算每个物理量独立的跟踪性能指标
    """
    target_array = np.array(target_array)
    current_array = np.array(current_array)

    # 基本误差计算
    error = target_array - current_array
    abs_error = np.abs(error)

    # 沿着时间轴(axis=0)计算每个物理量(component)的MSE和MAE
    mse_per_quantity = np.mean(error**2, axis=0)
    mae_per_quantity = np.mean(abs_error, axis=0)

    # 对每个分量的MSE开方
    rmse_per_quantity = np.sqrt(mse_per_quantity)

    metrics = {
        # 返回一个列表, 每个元素对应一个物理量的RMSE
        'rmse_per_quantity': rmse_per_quantity.tolist(),

        # 返回一个列表, 每个元素对应一个物理量的MAE
        'mae_per_quantity': mae_per_quantity.tolist(),

        # 累计误差仍然是总的聚合值
        'cumulative_absolute_error': float(np.sum(abs_error)),
        'cumulative_squared_error': float(np.sum(error**2))
    }

    return metrics


# MODIFIED FUNCTION
def print_comparison_metrics(shot_id, old_metrics, new_metrics):
    """打印比较指标, 逐个分量显示"""
    print(f"\nShot #{shot_id} - Metrics Comparison:")
    
    num_quantities = len(old_metrics['rmse_per_quantity'])
    
    for i in range(num_quantities):
        print(f"  Component #{i}:")
        old_rmse = old_metrics['rmse_per_quantity'][i]
        new_rmse = new_metrics['rmse_per_quantity'][i]
        old_mae = old_metrics['mae_per_quantity'][i]
        new_mae = new_metrics['mae_per_quantity'][i]
        
        print(f"    Old PPO  | RMSE: {old_rmse:.4f}, MAE: {old_mae:.4f}")
        print(f"    New PPO  | RMSE: {new_rmse:.4f}, MAE: {new_mae:.4f}")
        print(f"    Improve. | RMSE: {new_rmse - old_rmse:+.4f}, MAE: {new_mae - old_mae:+.4f}")


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


# MODIFIED FUNCTION
def save_comparison_results(comparison_results, log_folder):
    """
    Save comparison results to JSON and CSV files
    
    Args:
        comparison_results: Dictionary containing comparison results
        log_folder: Directory to save files
    """
    # Save results to JSON file (natively supports lists for per-quantity metrics)
    results_file = os.path.join(log_folder, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    # Save to CSV file
    try:
        import pandas as pd
        df_data = []
        for shot_key, shot_data in comparison_results.items():
            if shot_key.startswith('shot_'):
                # Basic info
                row = {'shot_id': shot_data['shot_id']}

                # Extract per-quantity metrics
                old_rmses = shot_data['old_ppo_vs_target']['rmse_per_quantity']
                old_maes = shot_data['old_ppo_vs_target']['mae_per_quantity']
                new_rmses = shot_data['new_ppo_vs_target']['rmse_per_quantity']
                new_maes = shot_data['new_ppo_vs_target']['mae_per_quantity']
                
                # Add per-quantity metrics to the row with specific column names
                for i in range(len(old_rmses)):
                    row[f'old_ppo_rmse_comp{i}'] = old_rmses[i]
                    row[f'old_ppo_mae_comp{i}'] = old_maes[i]
                    row[f'new_ppo_rmse_comp{i}'] = new_rmses[i]
                    row[f'new_ppo_mae_comp{i}'] = new_maes[i]
                    row[f'rmse_improvement_comp{i}'] = new_rmses[i] - old_rmses[i]
                    row[f'mae_improvement_comp{i}'] = new_maes[i] - old_maes[i]

                # Add cumulative and reward metrics
                row['old_ppo_cum_abs_error'] = shot_data['old_ppo_vs_target']['cumulative_absolute_error']
                row['old_ppo_cum_sq_error'] = shot_data['old_ppo_vs_target']['cumulative_squared_error']
                row['new_ppo_cum_abs_error'] = shot_data['new_ppo_vs_target']['cumulative_absolute_error']
                row['new_ppo_cum_sq_error'] = shot_data['new_ppo_vs_target']['cumulative_squared_error']
                row['old_ppo_reward'] = shot_data['old_ppo_reward']
                row['new_ppo_reward'] = shot_data['new_ppo_reward']
                
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
    new_ppo_data_path = args.new_ppo_data_path
    new_ppo_data = load_new_ppo_data(new_ppo_data_path)
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names()
    # MODIFICATION: Get number of quantities to track
    num_quantities = len(quan_names)

    # Create log folder for comparison results with seed information
    old_actor_info = args.old_actor_path.split('/')[-1] if args.old_actor_path else "unknown_old"
    new_ppo_info = os.path.basename(os.path.dirname(args.new_ppo_data_path))

    if args.comparison_name:
        comparison_id = args.comparison_name
    else:
        comparison_id = f"old_{old_actor_info}_vs_new_{new_ppo_info}_seed_{args.seed}"

    log_folder = os.path.join(args.save_base_dir, "results", "comparison0", args.task, comparison_id)
    os.makedirs(log_folder, exist_ok=True)

    print(f"Comparison results will be saved to: {log_folder}")

    # MODIFICATION: Storage for comparison results updated for per-quantity metrics
    comparison_results = {}
    old_ppo_summary = {
        'total_shots': 0, 
        'total_rmse': np.zeros(num_quantities), 
        'total_mae': np.zeros(num_quantities), 
        'total_reward': 0
    }
    new_ppo_summary = {
        'total_shots': 0, 
        'total_rmse': np.zeros(num_quantities), 
        'total_mae': np.zeros(num_quantities), 
        'total_reward': 0
    }

    for shot in shot_list:
        shot_key = f'shot_{shot}'
        if shot_key not in new_ppo_data:
            print(f"Warning: No new PPO data found for shot {shot}, skipping...")
            continue
            
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0
        time_array = []
        target_quan_array, real_quan_array, old_ppo_quan_array, real_act_array, old_ppo_act_array = [], [], [], [], []

        while True:
            action = controller.act(obs)
            next_obs, actuators, reward, terminal, info = env.step(action, obs)
            episode_reward += reward
            episode_length += 1
            cur_time = info["time_step"] - 1
            target_quan, real_quan, cur_quan, real_act, cur_act = sa_processor.get_plot_quantities(shot, cur_time, obs, action)
            time_array.append(cur_time)
            target_quan_array.append(target_quan)
            real_quan_array.append(real_quan)
            old_ppo_quan_array.append(cur_quan)
            real_act_array.append(real_act)
            old_ppo_act_array.append(cur_act)
            if terminal: break
            obs = next_obs
        
        target_quan_array = np.array(target_quan_array)
        real_quan_array = np.array(real_quan_array)
        old_ppo_quan_array = np.array(old_ppo_quan_array)
        old_ppo_act_array = np.array(old_ppo_act_array)
        time_array = np.array(time_array)

        reconstruct_target_quan_array = reconstruct_profile_from_state(args.task, np.array(target_quan_array), env.info, sa_processor.idx_list, unnormalize=False)
        reconstruct_real_quan_array = reconstruct_profile_from_state(args.task, np.array(real_quan_array), env.info, sa_processor.idx_list, unnormalize=False)
        old_ppo_reconstruct_cur_quan_array = reconstruct_profile_from_state(args.task, np.array(old_ppo_quan_array), env.info, sa_processor.idx_list, unnormalize=False)
        
        new_ppo_quan_array = new_ppo_data[shot_key]['cur_quan_array']
        new_ppo_act_array = new_ppo_data[shot_key]['cur_act_array']
        new_ppo_reconstruct_cur_quan_array = new_ppo_data[shot_key]['reconstruct_cur_quan_array']
        
        old_ppo_metrics = calculate_tracking_metrics(target_quan_array, old_ppo_quan_array)
        new_ppo_metrics = calculate_tracking_metrics(target_quan_array, new_ppo_quan_array)
        
        comparison_results[shot_key] = {
            'shot_id': shot,
            'old_ppo_vs_target': old_ppo_metrics,
            'new_ppo_vs_target': new_ppo_metrics,
            'old_ppo_reward': episode_reward,
            'new_ppo_reward': new_ppo_data[shot_key]['episode_reward']
        }
        
        # MODIFICATION: Update summary statistics with per-quantity metrics (element-wise addition)
        old_ppo_summary['total_shots'] += 1
        old_ppo_summary['total_rmse'] += np.array(old_ppo_metrics['rmse_per_quantity'])
        old_ppo_summary['total_mae'] += np.array(old_ppo_metrics['mae_per_quantity'])
        old_ppo_summary['total_reward'] += episode_reward
        
        new_ppo_summary['total_shots'] += 1
        new_ppo_summary['total_rmse'] += np.array(new_ppo_metrics['rmse_per_quantity'])
        new_ppo_summary['total_mae'] += np.array(new_ppo_metrics['mae_per_quantity'])
        new_ppo_summary['total_reward'] += new_ppo_data[shot_key]['episode_reward']
        
        plot_comparison_tracking_quantities(
            time_array, target_quan_array, real_quan_array, 
            old_ppo_quan_array, new_ppo_quan_array, 
            quan_names, shot, log_folder
        )
        plot_comparison_tracking_quantities_with_units(
            time_array, reconstruct_target_quan_array, reconstruct_real_quan_array, 
            old_ppo_reconstruct_cur_quan_array, new_ppo_reconstruct_cur_quan_array, 
            args.task, shot, log_folder
        )
        if args.plot_actuators:
            plot_comparison_actions(
                time_array, real_act_array, old_ppo_act_array, 
                new_ppo_act_array, act_names, shot, log_folder
            )
        
        print_comparison_metrics(shot, old_ppo_metrics, new_ppo_metrics)

    # Calculate and save summary metrics
    if old_ppo_summary['total_shots'] > 0:
        total_shots = old_ppo_summary['total_shots']
        summary_metrics = {
            'old_ppo': {
                'avg_rmse_per_quantity': (old_ppo_summary['total_rmse'] / total_shots).tolist(),
                'avg_mae_per_quantity': (old_ppo_summary['total_mae'] / total_shots).tolist(),
                'avg_reward': old_ppo_summary['total_reward'] / total_shots
            },
            'new_ppo': {
                'avg_rmse_per_quantity': (new_ppo_summary['total_rmse'] / total_shots).tolist(),
                'avg_mae_per_quantity': (new_ppo_summary['total_mae'] / total_shots).tolist(),
                'avg_reward': new_ppo_summary['total_reward'] / total_shots
            }
        }
        
        comparison_results['summary'] = summary_metrics
        
        # Note: These plotting functions might need adjustment if they don't accept list-like metrics
        #plot_metrics_comparison(comparison_results, log_folder)
        #plot_summary_metrics(summary_metrics, log_folder)
        
        # MODIFICATION: Updated summary printing
        print(f"\n{'='*60}")
        print(" " * 20 + "COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Total shots compared: {total_shots}")
        
        old_avg_rmses = summary_metrics['old_ppo']['avg_rmse_per_quantity']
        old_avg_maes = summary_metrics['old_ppo']['avg_mae_per_quantity']
        new_avg_rmses = summary_metrics['new_ppo']['avg_rmse_per_quantity']
        new_avg_maes = summary_metrics['new_ppo']['avg_mae_per_quantity']

        print("\n--- Average Metrics Per Component Across All Shots ---")
        for i in range(num_quantities):
            print(f"  Component #{i} ({quan_names[i]}):")
            print(f"    Old PPO  | Avg RMSE: {old_avg_rmses[i]:.4f}, Avg MAE: {old_avg_maes[i]:.4f}")
            print(f"    New PPO  | Avg RMSE: {new_avg_rmses[i]:.4f}, Avg MAE: {new_avg_maes[i]:.4f}")
            print(f"    Improve. | Avg RMSE: {new_avg_rmses[i] - old_avg_rmses[i]:+.4f}, Avg MAE: {new_avg_maes[i] - old_avg_maes[i]:+.4f}")

        print("\n--- Average Reward Across All Shots ---")
        old_avg_reward = summary_metrics['old_ppo']['avg_reward']
        new_avg_reward = summary_metrics['new_ppo']['avg_reward']
        print(f"  Old PPO Average Reward: {old_avg_reward:.3f}")
        print(f"  New PPO Average Reward: {new_avg_reward:.3f}")
        print(f"  Improvement in Reward: {new_avg_reward - old_avg_reward:+.3f}")
        print(f"{'='*60}")

    # Save comparison results
    save_comparison_results(comparison_results, log_folder)


if __name__ == "__main__":
    run()