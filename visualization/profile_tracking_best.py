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
from visualization.controller_best import ControllerBest
from visualization.plotter import plot_tracking_quantities, plot_actions


#!!! what you need to specify
def get_args():
    parser = argparse.ArgumentParser(description="Trajectory evaluation arguments using best model")

    # basic settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cuda_id", type=int, default=4, help="CUDA device ID")
    parser.add_argument("--plot_actuators", type=bool, default=True, help="Whether to plot actuators")

    # env settings
    parser.add_argument("--env", type=str, default="profile_control") 
    parser.add_argument("--task", type=str, default="dens", help="Targets to track") 

    # controller settings, of which the core is an NN actor
    parser.add_argument("--actor_path", type=str, default="/home/scratch/jiayuc2/rl_out_off/test_bao/dens_network/dens_ppo_seed1/lr0.003_steps2048_batch1024_epochs20_gamma0.952_gaelambda0.98_clip0.148_ent0.0067_vf1_maxgrad0.5_timesteps800000_pol250x250_val250x250", help="Path to the actor checkpoint")
    parser.add_argument("--il_actor", type=bool, default=False, help="Is this an imitation learning actor?")
    parser.add_argument("--stochastic_actor", type=bool, default=True, help="Is this a stochatic actor?")
    parser.add_argument("--hidden_dims", type=int, nargs='*', default=[250, 250], help="Hidden dimensions of the actor network") # you can get this in corresponding rl scripts
    parser.add_argument("--deterministic_mode", action="store_true", help="Whether to make the actor deterministic")

    parser.add_argument("--save_dir_name", type=str, default="dens_network_best", help="保存结果的子文件夹名称")
    parser.add_argument("--output_base_dir", type=str, default="/home/scratch/jiayuc2/eval_bao", help="输出文件的基础目录")
    return parser.parse_args()


def calculate_tracking_metrics(target_array, current_array):
    """计算跟踪性能指标

    Args:
        target_array: shape (time_steps, num_dimensions)
        current_array: shape (time_steps, num_dimensions)

    Returns:
        metrics: 包含总体指标和每个维度指标的字典
    """
    target_array = np.array(target_array)
    current_array = np.array(current_array)

    # 基本误差计算
    error = target_array - current_array
    abs_error = np.abs(error)

    # 总体指标（所有维度平均）
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

    # 每个维度的指标
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
        # 如果只有一个维度，也添加component1
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
    """打印跟踪指标"""
    print(f"\nShot #{shot_id} - 回合奖励: {episode_reward:.3f}, 长度: {episode_length}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  累计绝对误差: {metrics['cumulative_absolute_error']:.2f}")
    print(f"  累计平方误差: {metrics['cumulative_squared_error']:.2f}")


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
        # 计算总体平均指标
        summary_metrics = {
            'total_shots': num_shots,
            'avg_rmse': sum(shot['tracking_metrics']['rmse'] for shot in shot_results) / num_shots,
            'avg_mae': sum(shot['tracking_metrics']['mae'] for shot in shot_results) / num_shots,
            'avg_cumulative_absolute_error': sum(shot['tracking_metrics']['cumulative_absolute_error'] for shot in shot_results) / num_shots,
            'avg_reward': sum(shot['episode_reward'] for shot in shot_results) / num_shots,
            'avg_length': sum(shot['episode_length'] for shot in shot_results) / num_shots
        }

        # 计算每个维度的平均指标
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

                # 添加每个维度的指标
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

        # 打印每个维度的平均指标
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

    # load up the best model actor
    print("=" * 80)
    print("Loading BEST MODEL (from best_model.zip)")
    print("=" * 80)
    controller = ControllerBest(args)

    all_results = {}
    
    # rollouts
    shot_list = env.get_eval_shot_list()
    quan_names, act_names = sa_processor.get_plot_names() # name of the quantities to track and actuators in control

    # 从 actor_path 中提取路径信息来构建保存目录
    # 提取 actor_path 中从 test_bao 之后的所有路径部分
    actor_path_parts = args.actor_path.split('/')
    # 找到包含实验信息的路径部分（从 test_bao 后开始）
    try:
        test_bao_idx = actor_path_parts.index('test_bao')
        experiment_path = '/'.join(actor_path_parts[test_bao_idx + 1:])
    except ValueError:
        # 如果没有找到 test_bao，使用最后两个路径部分
        experiment_path = '/'.join(actor_path_parts[-2:]) if len(actor_path_parts) >= 2 else actor_path_parts[-1]

    # 构建最终的保存路径
    log_folder = os.path.join(args.output_base_dir, args.save_dir_name, experiment_path)

    os.makedirs(log_folder, exist_ok=True)
    print(f"\nResults will be saved to: {log_folder}")

    for shot in shot_list:
        obs = env.reset(shot_id=shot)
        episode_reward, episode_length = 0, 0

        time_array = []
        target_quan_array, real_quan_array, cur_quan_array, real_act_array, cur_act_array = [], [], [], [], []
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
         # 计算跟踪指标
        target_quan_array = np.array(target_quan_array)
        cur_quan_array = np.array(cur_quan_array)
        time_array = np.array(time_array)
        
        # 计算量化指标
        tracking_metrics = calculate_tracking_metrics(target_quan_array, cur_quan_array)
        
        all_results[f'shot_{shot}'] = {
            'shot_id': shot,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'tracking_metrics': tracking_metrics
        }
        
        # make plots
        plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot, log_folder)
        if args.plot_actuators:
            plot_actions(time_array, real_act_array, cur_act_array, act_names, shot, log_folder)

        # 打印详细指标
        print_tracking_metrics(shot, tracking_metrics, episode_reward, episode_length)

    save_tracking_results(all_results, log_folder)

if __name__ == "__main__":
    run()

