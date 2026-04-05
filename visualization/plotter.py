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

def plot_tracking_quantities_prof(time_array, target_quan_array, real_quan_array, cur_quan_array, quan_names, shot_id, log_folder):
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
        'temp': ['Temperature'],
        'pres_EFIT01': ['Pressure'],
        "q_EFIT01": ['q'],
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
            axes[i].plot(1/target_quan_array[tidx], ls='-', color='green', label='Target')
            axes[i].plot(1/cur_quan_array[tidx], color='red', label='RL')
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

def plot_actions_prof(time_array, real_act_array, cur_act_array, act_names, shot_id, log_folder):
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

def make_paper_quality_prof_fig(
    time_array,
    target_profiles,
    rl_profiles,
    quan_names,
    shot_id,
    log_folder,
    limits=None,
):
    """
    Prettier plotting for profile evolution (time-based at selected rho values)
    Produces two saved figures:
      - profile_evolution_shot_{shot_id}.png
      - actuator_evolution_shot_{shot_id}.png
    """
    import seaborn as sns

    all_titles = {
        "rotation":['Rotation (km/s)'],
        "pinj": ['Power Injected (kW)'],
        "tinj": ['Torque Injected (Nm)'],
        "ech_pwr_total":[ 'ECH Power (W)'],
        "dstdenp":[ 'Density (10^19)'],
        'gasA':[ 'Gas A (Voltage)'],
        'dens': ['Density'],
        'temp': ['Temperature'],
        'pres_EFIT01': ['Pressure'],
        "q_EFIT01": ['q'],
    }

    # Styling (paper-friendly)
    sns.set_style("white")
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
    })

    rolling_avg = 1
    req_rho = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    shot_id = int(shot_id)

    # times may be shape (1, T) or (T,) etc.
    if hasattr(time_array, "flatten"):
        plotting_times = np.array(time_array).flatten()
    else:
        plotting_times = np.array(time_array)

    # number of radial points
    n_rad = rl_profiles.shape[1]
    rho_indices = [min(int(r * n_rad), n_rad - 1)-1 for r in req_rho]

    # Figure: 2 x 3 grid for selected rho values
    ncols = 3
    nrows = 2
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5), constrained_layout=True)
    axs = axarr.flatten()
    # palette = sns.color_palette()

    for i, ridx in enumerate(rho_indices):
        ax = axs[i]

        # ensure time dim alignment
        max_t = min(plotting_times.shape[0], rl_profiles.shape[0])
        times_plot = plotting_times[:max_t]

        # RL present traces (many unrolls) and mean
        rl_traces = rl_profiles[ :max_t, ridx]  # shape (n_unrolls, time)
        
       
        ax.plot(times_plot, rl_traces, color="blue", label="RL Present")

        # plot translucent individual traces
        # ax.plot(times_plot, rl_traces.T, color="blue", alpha=0.12, linewidth=0.8)


        # Target profile (if provided)
        if target_profiles is not None :
            tgt = target_profiles[ :max_t, ridx]
            ax.plot(times_plot, tgt, color="red", linestyle="--", label="Target")

        # Small details & limits
        rho_val = np.round((ridx + 1) / float(n_rad), 2)
        ax.set_title(fr"$\psi_n$ = {rho_val}", pad=6)
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=0.25)
        # ax.set_xlim(times_plot[0], times_plot[-1])
        ax.set_xlim(1200, times_plot[-1])

        # y-labels and sensible y-limits (fall back to limits if available)
        if quan_names == "rotation":
            ax.set_ylabel("Rotation (km/s)")
        else:
            ax.set_ylabel(all_titles.get(quan_names, [quan_names.capitalize()])[0] if 'all_titles' in globals() else quan_names)

    # Create a single legend for the whole figure
    from collections import OrderedDict
    all_handles = []
    all_labels = []
    for ax in axs[: len(rho_indices) ]:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    legend_dict = OrderedDict()
    for h, lab in zip(all_handles, all_labels):
        if lab not in legend_dict:
            legend_dict[lab] = h

    fig.legend(legend_dict.values(), legend_dict.keys(),
               loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2,
               frameon=True, fontsize=15)

    plt.suptitle(fr"Simulated Shot {shot_id} {quan_names.capitalize()} Evolution at Selected $\psi_n$ Values", fontsize=15)
    save_path = os.path.join(log_folder, f"{shot_id}_tracking_quantities_profile.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    