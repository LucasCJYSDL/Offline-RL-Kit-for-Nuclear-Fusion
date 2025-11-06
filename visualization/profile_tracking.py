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
from visualization.plotter import plot_tracking_quantities, plot_actions, plot_tracking_quantities_with_units
from  envs.utils.profile_util import reconstruct_profile_from_state


#!!! what you need to specify
def get_args():
    parser = argparse.ArgumentParser(description="Trajectory evaluation arguments")

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=3, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="fusion_env") 
    parser.add_argument("--task", type=str, default="rotation", help="Targets to track") 

    # controller settings, of which the core is an NN actor
    # parser.add_argument("--actor_path", type=str, default="log/rotation/ppo/seed_1&timestamp_25-0911-120453", help="Path to the actor checkpoint")
    parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out/prof_tracking_rot_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/601/policy", help="Path to the actor checkpoint")
    # parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out/prof_tracking_dens_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/policy", help="Path to the actor checkpoint")
    # parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/prof_tracking_dens_kth_target_flattop_subset_boots/ppo_prof_control_zipfit_dens_optimized_lite/policy", help="Path to the actor checkpoint")
    # parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out/beta_tracking_env/ppo_prof_control_zipfit_dens_optimized_lite/policy", help="Path to the actor checkpoint")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=False, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[250, 250], help="Hidden dimensions of the actor network") # you can get this in corresponding rl scripts
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")
    return parser.parse_args()

def calculate_tracking_metrics(target_array, current_array):
    target_array = np.array(target_array)
    current_array = np.array(current_array)
    error = target_array - current_array
    abs_error = np.abs(error)

    metrics = {
        'rmse': float(np.sqrt(np.mean(error**2))),

        'mae': float(np.mean(abs_error)),

        'cumulative_absolute_error': float(np.sum(abs_error)),

        'cumulative_squared_error': float(np.sum(error**2))
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
        summary_metrics = {
            'total_shots': num_shots,
            'avg_rmse': sum(shot['tracking_metrics']['rmse'] for shot in shot_results) / num_shots,
            'avg_mae': sum(shot['tracking_metrics']['mae'] for shot in shot_results) / num_shots,
            'avg_reward': sum(shot['episode_reward'] for shot in shot_results) / num_shots,
            'avg_length': sum(shot['episode_length'] for shot in shot_results) / num_shots
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
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(f"Average Length: {summary['avg_length']:.1f}")


def run(args=get_args()) -> None:
    # register an env
    args.device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    offline_data, sa_processor, env, _ = get_rl_data_envs(args.env, args.task, args.device, is_il=args.il_actor) # these are the data and env used to train the actor
    args.obs_dim = offline_data['obs_dim']
    args.action_dim = offline_data['act_dim']
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
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names() # name of the quantities to track and actuators in control
    
    actor_info = args.actor_path.split('/') # create a folder to store the visualization results
    #log_folder = os.path.join(os.path.dirname(__file__), "results", actor_info[1], actor_info[2], actor_info[3])
    log_folder = "/home/scratch/jiayuc2/results/old1/ppo_rot"
    os.makedirs(log_folder, exist_ok=True)

    # init storage for all results
    all_results = {}

    for shot in shot_list:
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0

        time_array = []
        target_quan_array, real_quan_array, cur_quan_array, real_act_array, cur_act_array = [], [], [], [], []
        reconstruct_target_quan_array, reconstruct_real_quan_array, reconstruct_cur_quan_array, reconstruct_real_act_array, reconstruct_cur_act_array = [], [], [], [], []

        while True:
            action = controller.act(obs)
            next_obs, actuators, reward, terminal, info = env.step(action,obs)
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
        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        reconstruct_tracking_metrics = calculate_tracking_metrics(reconstruct_target_quan_array, reconstruct_cur_quan_array)
        
        all_results[f'shot_{shot}'] = {
            'shot_id': shot,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'tracking_metrics': tracking_metrics
        }
        
        if (args.env == "fusion_env"):
            reconstruct_target_quan_array = reconstruct_profile_from_state(args.task, np.array(target_quan_array), env.info, sa_processor.idx_list, unnormalize=True)
            reconstruct_real_quan_array = reconstruct_profile_from_state(args.task, np.array(real_quan_array), env.info, sa_processor.idx_list, unnormalize=True)
            reconstruct_cur_quan_array = reconstruct_profile_from_state(args.task, np.array(cur_quan_array), env.info, sa_processor.idx_list, unnormalize=True)

       
        # make plots
        plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot, log_folder)
        plot_tracking_quantities_with_units(time_array,  reconstruct_target_quan_array, reconstruct_real_quan_array, reconstruct_cur_quan_array, quan_names, shot, log_folder)
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, shot, log_folder)
        
        print("Profiles' metrics:")
        print_tracking_metrics(shot, tracking_metrics, episode_reward, episode_length)
        print("\nReconstructed profiles' metrics:")
        print_tracking_metrics(shot, reconstruct_tracking_metrics, episode_reward, episode_length)

        # summary
        print("Shot #{} with return {} and length {}".format(shot, episode_reward, episode_length))

    save_tracking_results(all_results, log_folder)

if __name__ == "__main__":
    run()