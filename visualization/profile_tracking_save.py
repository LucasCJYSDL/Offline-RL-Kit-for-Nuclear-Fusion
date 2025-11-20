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
from  envs.utils.profile_util import reconstruct_profile_from_state

#!!! what you need to specify
def get_args():
    parser = argparse.ArgumentParser(description="Trajectory evaluation and data saving arguments")

    parser.add_argument("--save_base_dir", type=str, 
                       default="/home/scratch/jiayuc2/temp_1117",
                       help="Base directory for saving all results")
    
    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=5, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="profile_control")
    parser.add_argument("--task", type=str, default="rotation", help="Targets to track") 

    # controller settings, of which the core is an NN actor
    # parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out_off/rotation_ppo_seed601/lr0.0001_steps4096_batch512_epochs20_gamma0.99_gaelambda0.95_clip0.2_ent0.0_vf1_maxgrad0.5_hidden250x250", help="Path to the actor checkpoint")
    # parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out_off/test_bao/dens_network/dens_ppo_seed1/lr0.0003_steps2048_batch1024_epochs20_gamma0.952_gaelambda0.98_clip0.148_ent0.0067_vf1_maxgrad0.5_timesteps3000000_pol250x250_val250x250") # need to change
    parser.add_argument("--actor_path",type=str, default="/home/scratch/jiayuc2/fy/log/rotation/cql&cql_weight=5.0&temperature=1.0&max_q_backup=False&deterministic_backup=True&with_lagrange=False&lagrange_threshold=10.0&cql_alpha_lr=0.0003&num_repeat_actions=10/seed_1&timestamp_25-1116-124536")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[256,256], help="Hidden dimensions of the actor network") # you can get this in corresponding rl scripts
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    parser.add_argument("--use_diag_gaussian", action="store_true", help="Use DiagGaussian instead of TanhDiagGaussian (required for IQL)")
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

def calculate_reconstruct_tracking_metrics(target_array, current_array):
    target_array = np.array(target_array)
    current_array = np.array(current_array)

    # Basic error calculation
    error = target_array - current_array
    mse_per_column = np.mean(error**2, axis=0) 
    return mse_per_column

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
    results_file = os.path.join(log_folder , 'tracking_results.json')
    
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
        csv_file = os.path.join(log_folder,  'tracking_results.csv')
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



def save_shot_data(shot_data_dict, save_folder, task_name, actor_info):
    """
    Save shot data to npz file
    
    Args:
        shot_data_dict: Dictionary containing shot data
        save_folder: Base folder for saving
        task_name: Task name for folder structure
        actor_info: Actor information for folder structure
    """
    # Create save directory structure
    data_save_dir = os.path.join(save_folder, "saved_data")
    os.makedirs(data_save_dir, exist_ok=True)
    
    # Save data to npz file
    npz_file = os.path.join(data_save_dir, "shot_data.npz")
    np.savez(npz_file, **shot_data_dict)
    
    # Save metadata
    metadata = {
        'task_name': task_name,
        'actor_info': actor_info,
        'num_shots': len([k for k in shot_data_dict.keys() if k.startswith('shot_')]),
        'shot_ids': [shot_data_dict[k]['shot_id'] for k in shot_data_dict.keys() if k.startswith('shot_')],
        'save_time': str(np.datetime64('now'))
    }
    
    metadata_file = os.path.join(data_save_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nShot data saved to: {npz_file}")
    print(f"Metadata saved to: {metadata_file}")
    
    return npz_file, metadata_file


def run(args=get_args()) -> None:
    # register an env
    args.device = torch.device("cuda".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
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
    shot_data_dict = {}  # For saving shot data
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names() # name of the quantities to track and actuators in control
    
    actor_info = args.actor_path.split('/') # create a folder to store the visualization results
    log_folder = os.path.join(args.save_base_dir, actor_info[-2], args.task, "results", actor_info[-1], str(args.seed))
    os.makedirs(log_folder, exist_ok=True)

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

        # Calculate quantitative metrics
        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        mse_reconstruct = calculate_reconstruct_tracking_metrics(reconstruct_target_quan_array, reconstruct_cur_quan_array)
        all_results[f'shot_{shot}'] = {
            'shot_id': shot,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'tracking_metrics': tracking_metrics,
        }
        
        # Save shot data for later comparison
        shot_data_dict[f'shot_{shot}'] = {
            'shot_id': shot,
            'cur_quan_array': cur_quan_array,
            'cur_act_array': cur_act_array,
            'reconstruct_cur_quan_array': reconstruct_cur_quan_array,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mse_reconstruct': mse_reconstruct 
        }
        
        # make plots
        plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot, log_folder)
        plot_tracking_quantities_with_units(time_array,  reconstruct_target_quan_array, real_quan_array, reconstruct_cur_quan_array, args.task, shot, log_folder)
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, shot, log_folder)

        print_tracking_metrics(shot, tracking_metrics, episode_reward, episode_length)
    # Save tracking results
    save_tracking_results(all_results, log_folder)
    
    # Save shot data for comparison
    actor_info_str = "_".join(actor_info[1:])  # Join actor path components
    save_shot_data(shot_data_dict, log_folder, args.task, actor_info_str)

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
    for seed in seeds:
        print(f"\nRunning experiment with seed = {seed}")
        args = get_args()  
        args.seed = seed   
        run(args)
    # run()
