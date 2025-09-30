import os
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_tracking_quantities(time_array, target_quan_array, real_quan_array,
                                       old_ppo_quan_array, new_ppo_quan_array,
                                       quan_names, shot_id, log_folder):
    """
    Plot comparison of tracking quantities with four curves:
    Real, Target, Old PPO Agent, New PPO Agent

    Args:
        time_array: Time steps
        target_quan_array: Target quantities
        real_quan_array: Real quantities
        old_ppo_quan_array: Old PPO agent quantities
        new_ppo_quan_array: New PPO agent quantities
        quan_names: Names of quantities
        shot_id: Shot ID
        log_folder: Folder to save plots
    """
    n = len(quan_names)
    rows = (n + 1) // 2  # calculate the number of rows needed for 2 subfigures per row
    fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows), sharex=True, sharey=False)

    # Handle axes properly for different subplot configurations
    if n == 1:
        # Single subplot case
        if rows == 1:
            axes = [axes[0]]  # Take the first subplot from the row
        else:
            axes = [axes]
    else:
        # Multiple subplots case
        axes = np.atleast_1d(axes).flatten()

    for i in range(n):
        # Plot four curves with different colors and styles
        axes[i].plot(time_array, [rq[i] for rq in real_quan_array], 
                    label="Real", color='blue', linewidth=2)
        axes[i].plot(time_array, [tq[i] for tq in target_quan_array], 
                    label="Target", color='red', linestyle='--', linewidth=2)
        axes[i].plot(time_array, [oq[i] for oq in old_ppo_quan_array], 
                    label="Old PPO Agent", color='green', linewidth=2)
        axes[i].plot(time_array, [nq[i] for nq in new_ppo_quan_array], 
                    label="New PPO Agent", color='orange', linewidth=2)
        
        axes[i].set_title(f"{quan_names[i]} - Shot {shot_id}", fontsize=16)
        axes[i].set_xlabel("Time", fontsize=14)
        axes[i].set_ylabel("Value", fontsize=14)
        axes[i].legend(fontsize=12, loc='best')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    # remove unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # save the figure 
    save_path = os.path.join(log_folder, f"{shot_id}_comparison_tracking_quantities.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def plot_comparison_actions(time_array, real_act_array, old_ppo_act_array,
                           new_ppo_act_array, act_names, shot_id, log_folder):
    """
    Plot comparison of actions with three curves:
    Real, Old PPO Agent, New PPO Agent

    Args:
        time_array: Time steps
        real_act_array: Real actions
        old_ppo_act_array: Old PPO agent actions
        new_ppo_act_array: New PPO agent actions
        act_names: Names of actions
        shot_id: Shot ID
        log_folder: Folder to save plots
    """
    m = len(act_names)
    rows = (m + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows), sharex=True, sharey=False)

    # Handle axes properly for different subplot configurations
    if m == 1:
        # Single subplot case
        if rows == 1:
            axes = [axes[0]]  # Take the first subplot from the row
        else:
            axes = [axes]
    else:
        # Multiple subplots case
        axes = np.atleast_1d(axes).flatten()

    for i in range(m):
        # Plot three curves with different colors and styles
        axes[i].plot(time_array, [ra[i] for ra in real_act_array], 
                    label="Real", color='blue', linewidth=2)
        axes[i].plot(time_array, [oa[i] for oa in old_ppo_act_array], 
                    label="Old PPO Agent", color='green', linewidth=2)
        axes[i].plot(time_array, [na[i] for na in new_ppo_act_array], 
                    label="New PPO Agent", color='orange', linewidth=2)
        
        axes[i].set_title(f"{act_names[i]} - Shot {shot_id}", fontsize=16)  
        axes[i].set_xlabel("Time", fontsize=14)  
        axes[i].set_ylabel("Value", fontsize=14) 
        axes[i].legend(fontsize=12, loc='best')  
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=12) 

    # remove unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(log_folder, f"{shot_id}_comparison_actions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def plot_metrics_comparison(metrics_comparison, log_folder):
    """
    Plot comparison of metrics between old and new PPO agents
    
    Args:
        metrics_comparison: Dictionary containing metrics comparison
        log_folder: Folder to save plots
    """
    # Extract data for plotting
    shot_ids = []
    old_rmse, new_rmse = [], []
    old_mae, new_mae = [], []
    
    for shot_key, data in metrics_comparison.items():
        if shot_key.startswith('shot_'):
            shot_ids.append(data['shot_id'])
            old_rmse.append(data['old_ppo_vs_target']['rmse'])
            new_rmse.append(data['new_ppo_vs_target']['rmse'])
            old_mae.append(data['old_ppo_vs_target']['mae'])
            new_mae.append(data['new_ppo_vs_target']['mae'])
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE comparison
    axes[0, 0].bar(np.arange(len(shot_ids)) - 0.2, old_rmse, 0.4, 
                   label='Old PPO', color='green', alpha=0.7)
    axes[0, 0].bar(np.arange(len(shot_ids)) + 0.2, new_rmse, 0.4, 
                   label='New PPO', color='orange', alpha=0.7)
    axes[0, 0].set_title('RMSE Comparison', fontsize=14)
    axes[0, 0].set_xlabel('Shot ID', fontsize=12)
    axes[0, 0].set_ylabel('RMSE', fontsize=12)
    axes[0, 0].set_xticks(range(len(shot_ids)))
    axes[0, 0].set_xticklabels(shot_ids, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[0, 1].bar(np.arange(len(shot_ids)) - 0.2, old_mae, 0.4, 
                   label='Old PPO', color='green', alpha=0.7)
    axes[0, 1].bar(np.arange(len(shot_ids)) + 0.2, new_mae, 0.4, 
                   label='New PPO', color='orange', alpha=0.7)
    axes[0, 1].set_title('MAE Comparison', fontsize=14)
    axes[0, 1].set_xlabel('Shot ID', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_xticks(range(len(shot_ids)))
    axes[0, 1].set_xticklabels(shot_ids, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE improvement (negative means new PPO is better)
    rmse_improvement = [new - old for old, new in zip(old_rmse, new_rmse)]
    colors = ['red' if x > 0 else 'green' for x in rmse_improvement]
    axes[1, 0].bar(range(len(shot_ids)), rmse_improvement, color=colors, alpha=0.7)
    axes[1, 0].set_title('RMSE Improvement (New - Old)', fontsize=14)
    axes[1, 0].set_xlabel('Shot ID', fontsize=12)
    axes[1, 0].set_ylabel('RMSE Difference', fontsize=12)
    axes[1, 0].set_xticks(range(len(shot_ids)))
    axes[1, 0].set_xticklabels(shot_ids, rotation=45)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE improvement (negative means new PPO is better)
    mae_improvement = [new - old for old, new in zip(old_mae, new_mae)]
    colors = ['red' if x > 0 else 'green' for x in mae_improvement]
    axes[1, 1].bar(range(len(shot_ids)), mae_improvement, color=colors, alpha=0.7)
    axes[1, 1].set_title('MAE Improvement (New - Old)', fontsize=14)
    axes[1, 1].set_xlabel('Shot ID', fontsize=12)
    axes[1, 1].set_ylabel('MAE Difference', fontsize=12)
    axes[1, 1].set_xticks(range(len(shot_ids)))
    axes[1, 1].set_xticklabels(shot_ids, rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(log_folder, "metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def plot_summary_metrics(summary_metrics, log_folder):
    """
    Plot summary metrics comparison
    
    Args:
        summary_metrics: Dictionary containing summary metrics
        log_folder: Folder to save plots
    """
    metrics_names = ['avg_rmse', 'avg_mae', 'avg_reward']
    old_values = [summary_metrics['old_ppo'][metric] for metric in metrics_names]
    new_values = [summary_metrics['new_ppo'][metric] for metric in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, old_values, width, label='Old PPO', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, new_values, width, label='New PPO', color='orange', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Summary Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Average RMSE', 'Average MAE', 'Average Reward'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(log_folder, "summary_metrics_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path
