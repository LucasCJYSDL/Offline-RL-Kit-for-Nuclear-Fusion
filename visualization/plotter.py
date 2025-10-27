import os
import matplotlib.pyplot as plt
import numpy as np

# TODO: the first two functions can be merged


def plot_tracking_quantities(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot_id, log_folder):
    n = len(quan_names)  
    rows = (n + 1) // 2  # calculate the number of rows needed for 2 subfigures per row
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), sharex=True, sharey=False)
    axes = axes.flatten()  # flatten the 2D array of axes for easier indexing

    for i in range(n):
        axes[i].plot(time_array, [rq[i] for rq in real_quan_array], label="Real")
        axes[i].plot(time_array, [cq[i] for cq in cur_quan_array], label="Agent")
        axes[i].plot(time_array, [tq[i] for tq in target_quan_array], label="Target", linestyle="--")
        axes[i].set_title(f"{quan_names[i]}", fontsize=15)
        axes[i].set_xlabel("Time", fontsize=15)
        axes[i].set_ylabel("Value", fontsize=15)
        axes[i].legend(fontsize=15)
        axes[i].grid()
        axes[i].tick_params(axis='both', which='major', labelsize=15)  # enlarged tick labels

    # remove unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # save the figure 
    save_path = os.path.join(log_folder, f"{shot_id}_tracking_quantities.png")
    plt.savefig(save_path)
    plt.close(fig)

def plot_actions(time_array, real_act_array, cur_act_array, act_names, shot_id, log_folder):
    m = len(act_names)  
    rows = (m + 1) // 2  
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows), sharex=True, sharey=False)
    axes = axes.flatten() 

    for i in range(m):
        axes[i].plot(time_array, [ra[i] for ra in real_act_array], label="Real")
        axes[i].plot(time_array, [ca[i] for ca in cur_act_array], label="Agent")
        axes[i].set_title(f"{act_names[i]}", fontsize=15)  
        axes[i].set_xlabel("Time", fontsize=15)  
        axes[i].set_ylabel("Value", fontsize=15) 
        axes[i].legend(fontsize=15)  
        axes[i].grid()
        axes[i].tick_params(axis='both', which='major', labelsize=15) 

    # remove unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(log_folder, f"{shot_id}_actions.png")
    plt.savefig(save_path)
    plt.close(fig)

def plot_tracking_quantities_with_units(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot_id, log_folder):
    """
    Plot tracking quantities with physical units displayed on axes
    
    Args:
        time_array: Array of time steps
        target_quan_array: Target quantity values (denormalized)
        real_quan_array: Real quantity values (denormalized) 
        cur_quan_array: Current quantity values from RL agent (denormalized)
        quan_names: List of quantity names
        shot_id: Shot identifier
        log_folder: Directory to save the plot
    """
    all_titles = {
        "rotation":['Rotation (km/s)'],
        "pinj": ['Power Injected (kW)'],
        "tinj": ['Torque Injected (Nm)'],
        "ech_pwr_total":[ 'ECH Power (W)'],
        "dstdenp":[ 'Density (10^19)'],
        'gasA':[ 'Gas A (Voltage)'],
        'dens': ['Density'],
    }
    n = len([quan_names])
    rows = (n + 1) // 2  # Calculate rows needed for 2 subplots per row
    fig, axes = plt.subplots(rows, 1, figsize=(12, 5 * rows), sharex=True, sharey=False)
    quan_names = [quan_names]
    # Handle single subplot case
    if n == 1:
        axes = [axes] if rows == 1 else axes.flatten()
    else:
        axes = axes.flatten()
    
    tidx = len(time_array)-1
    for i in range(n):
        quan_name = quan_names[i]
        if quan_name == 'q_EFIT01':
            axes[i].plot(1/real_quan_array[tidx], ls='--', color='black', label='True')
            axes[i].plot(1/target_quan_array[0, tidx], ls='-', color='green', label='Target')
            axes[i].plot(1/np.mean(cur_quan_array, axis=0)[tidx], color='red', label='RL')
            axes[i].plot(1/cur_quan_array[tidx, :].T, color='red', alpha=0.1)
            axes[i].set_ylim([0.75, 6])
            axes[i].set_title('q')
            axes[i].set_xlabel('Normalized Radius')
            axes[i].legend()
            axes[i].grid()
        else:
            axes[i].plot(target_quan_array[tidx], ls='-', color='green', label='Target')
            axes[i].plot(cur_quan_array[tidx], color='red', label='RL')
            axes[i].plot( cur_quan_array[tidx, :].T, color='red', alpha=0.1)

            # axes[i].set_ylim(limits[pname])
            axes[i].set_title(all_titles[quan_name][0])
            x_values = np.linspace(0.0, len(target_quan_array[tidx]), 7)
            new_x_values = np.linspace(0.0, 1, 7)
            axes[i].set_xticks(x_values)
            axes[i].set_xticklabels([f"{label:.2f}" for label in new_x_values])
            axes[i].set_xlabel('Normalized Radius')
            axes[i].legend()
            axes[i].grid()
       

    # Remove unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Shot #{shot_id} - Physical Quantities Tracking (with Units)', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure with high resolution
    save_path = os.path.join(log_folder, f"{shot_id}_tracking_quantities_with_units.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved tracking plot with units: {save_path}")


def plot_actions_with_units(time_array, real_act_array, cur_act_array, act_names, shot_id, log_folder):
    """
    Plot actuator actions with physical units displayed on axes
    
    Args:
        time_array: Array of time steps
        real_act_array: Real actuator values (denormalized)
        cur_act_array: Current actuator values from RL agent (denormalized)
        act_names: List of actuator names
        shot_id: Shot identifier
        log_folder: Directory to save the plot
    """
    m = len(act_names)
    rows = (m + 1) // 2  # Calculate rows needed for 2 subplots per row
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), sharex=True, sharey=False)
    
    # Handle single subplot case
    if m == 1:
        axes = [axes] if rows == 1 else axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(m):
        act_name = act_names[i]
        
        # Get physical unit for this actuator
        unit = PHYSICAL_UNITS.get(act_name, '')
        ylabel = f"{act_name} ({unit})" if unit else act_name
        
        # Plot the data with enhanced styling
        if len(real_act_array) > 0:
            axes[i].plot(time_array, [ra[i] for ra in real_act_array], 
                        label="Real", color='blue', linewidth=2, alpha=0.8)
        
        axes[i].plot(time_array, [ca[i] for ca in cur_act_array], 
                    label="RL Agent", color='red', linewidth=2)
        
        # Set labels and formatting
        axes[i].set_title(f"{act_name} Control", fontsize=16, fontweight='bold')
        axes[i].set_xlabel("Time Step", fontsize=14)
        axes[i].set_ylabel(ylabel, fontsize=14)
        axes[i].legend(fontsize=12, loc='best')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        
        # Calculate and display control range statistics
        if len(cur_act_array) > 0:
            current_vals = [ca[i] for ca in cur_act_array]
            min_val, max_val = min(current_vals), max(current_vals)
            range_val = max_val - min_val
            
            # Add range information text box
            axes[i].text(0.02, 0.02, f'Range: {range_val:.3f} {unit}', 
                        transform=axes[i].transAxes, fontsize=10,
                        verticalalignment='bottom', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Remove unused subplots
    for j in range(m, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Shot #{shot_id} - Actuator Actions (with Units)', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save the figure with high resolution
    save_path = os.path.join(log_folder, f"{shot_id}_actions_with_units.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved action plot with units: {save_path}")