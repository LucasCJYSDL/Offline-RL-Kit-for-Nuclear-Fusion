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

def plot_tracking_quantities_prof(
    time_array,
    target_quan_array,
    real_quan_array,
    cur_quan_array,
    quan_names,
    algo_name,
    shot_id,
    log_folder,
    snapshot_times=None,
):
    """
    Plot six profile snapshots across time.

    Args:
        time_array: Array of time steps
        target_quan_array: Target profile values (time, radius)
        real_quan_array: Real profile values (time, radius)
        cur_quan_array: Current profile values from RL agent (time, radius)
        quan_names: Quantity name
        shot_id: Shot identifier
        log_folder: Directory to save the plot
        snapshot_times: Optional list of time values to plot. If omitted,
            six evenly spaced times are selected automatically.
    """
    all_titles = {
        "rotation": ['Rotation (km/s)'],
        "pinj": ['Power Injected (kW)'],
        "tinj": ['Torque Injected (Nm)'],
        "ech_pwr_total": ['ECH Power (W)'],
        "dstdenp": ['Density (10^19)'],
        'gasA': ['Gas A (Voltage)'],
        'dens': ['Density'],
        'temp': ['Temperature'],
        'pres_EFIT01': ['Pressure'],
        "q_EFIT01": ['q'],
    }

    quan_name = quan_names[0] if isinstance(quan_names, (list, tuple)) else quan_names
    profile_title = all_titles.get(quan_name, [quan_name])[0]

    time_array = np.asarray(time_array).flatten()
    target_quan_array = np.asarray(target_quan_array)
    real_quan_array = np.asarray(real_quan_array)
    cur_quan_array = np.asarray(cur_quan_array)

    max_t = min(len(time_array), target_quan_array.shape[0], cur_quan_array.shape[0])
    if max_t <= 0:
        raise ValueError("time_array and profile arrays must contain at least one timestep")

    if snapshot_times is None:
        snapshot_indices = np.linspace(0, max_t - 1, min(6, max_t), dtype=int)
    else:
        snapshot_times = np.asarray(snapshot_times).flatten()
        snapshot_indices = np.array([
            int(np.argmin(np.abs(time_array[:max_t] - t)))
            for t in snapshot_times
        ], dtype=int)

    snapshot_indices = list(dict.fromkeys(snapshot_indices.tolist()))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    rho = np.linspace(0.0, 1.0, target_quan_array.shape[1])

    for i, tidx in enumerate(snapshot_indices[:6]):
        ax = axes[i]
        if quan_name == 'q_EFIT01':
            present = 1.0 / cur_quan_array[tidx]
            target = 1.0 / target_quan_array[tidx]
        else:
            present = cur_quan_array[tidx]
            target = target_quan_array[tidx]

        ax.plot(rho, present, color='steelblue', linewidth=1.6, label='Present')
        ax.plot(rho, target, color='indianred', linestyle='--', linewidth=1.6, label='Target')
        ax.set_title(f"t = {int(time_array[tidx])} ms", fontsize=14)
        ax.set_xlabel(r'$\psi_n$')
        ax.set_ylabel(profile_title)
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, 1.0)
        xticks = np.linspace(0.0, 1.0, 6)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    for j in range(len(snapshot_indices), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True, fontsize=13)

    plt.suptitle(
        f'Shot {shot_id} {algo_name} {profile_title} Snapshots',
        fontsize=18,
        fontweight='bold'
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.98, wspace=0.22, hspace=0.32)

    save_path = os.path.join(log_folder, f"{shot_id}_tracking_quantities_with_units.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


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
    act_array,
    quan_names,
    algo_name,
    shot_id,
    log_folder,
    info,
    actuators_to_plot=None,
    limits=None
):
    """
    Prettier plotting for profile evolution (time-based at selected rho values)
    Produces one combined figure with:
      - profile panels on the left
      - actuator panels on the right
    """
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from collections import OrderedDict

    if actuators_to_plot is None:
        actuators_to_plot = []

    all_titles = {
        "rotation": ['Rotation (km/s)'],
        "pinj": ['Power Injected (kW)'],
        "tinj": ['Torque Injected (Nm)'],
        "ech_pwr_total": ['ECH Power (W)'],
        "dstdenp": ['Density (10^19)'],
        'gasA': ['Gas A (Voltage)'],
        'dens': ['Density'],
        'temp': ['Temperature'],
        'pres_EFIT01': ['Pressure'],
        "q_EFIT01": ['q'],
    }

    sns.set_style("white")
    sns.set_palette("colorblind")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,
    })

    req_rho = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    shot_id = int(shot_id)
    plotting_times = np.array(time_array).flatten()

    n_rad = rl_profiles.shape[1]
    rho_indices = [min(int(r * n_rad), n_rad - 1) for r in req_rho]

    # One figure: left block for profiles, right block for actuators.
    fig = plt.figure(figsize=(24, 7.2), constrained_layout=False)
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1], wspace=0.35, hspace=0.42)

    profile_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    act_axes = [fig.add_subplot(gs[r, 3 + c]) for r in range(2) for c in range(2)]

    controller_start = info.get("controller_start_ms", 1200)

    # Left block: selected profile traces.
    for i, ridx in enumerate(rho_indices):
        ax = profile_axes[i]

        max_t = min(plotting_times.shape[0], rl_profiles.shape[0])
        times_plot = plotting_times[:max_t]
        rl_traces = rl_profiles[:max_t, ridx]

        ax.plot(times_plot, rl_traces, color="blue", label=f"{algo_name} Present")

        if target_profiles is not None:
            tgt = target_profiles[:max_t, ridx]
            ax.plot(times_plot, tgt, color="red", linestyle="--", label="Target")

        rho_val = np.round((ridx + 1) / float(n_rad), 2)
        ax.set_title(fr"$\psi_n$ = {rho_val}", pad=6)
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=0.25)
        ax.set_xlim(1200, times_plot[-1])

        if quan_names == "rotation":
            ax.set_ylabel("Rotation (km/s)")
        else:
            ax.set_ylabel(all_titles.get(quan_names, [quan_names.capitalize()])[0])

        ax.axvline(controller_start, color="0.35", linestyle="--", linewidth=1.1, alpha=0.9)

    for j in range(len(rho_indices), len(profile_axes)):
        fig.delaxes(profile_axes[j])

    # Right block: actuators.
    multipliers = {
        'pinj': 1.0e-3,
        'tinj': 1.0,
        'gasA': 1.0,
        'ech_pwr_total': 1e-6,
    }

    ylabel = {
        'pinj': 'MW',
        'tinj': 'MNm',
        'gasA': 'Volts',
        'ech_pwr_total': 'MW',
    }

    for i, actname in enumerate(actuators_to_plot[:4]):
        ax = act_axes[i]

        norm_cfg = info['normalization_dict'][actname]
        if norm_cfg['method'] == 'RobustScaler':
            mu = norm_cfg['median']
            sigma = norm_cfg['iqr']
        elif norm_cfg['method'] == 'MinMax':
            mu = norm_cfg['armin']
            sigma = norm_cfg['armax'] - mu
        else:
            raise ValueError(f"Unknown normalization method: {norm_cfg['method']}")

        rl_acts = act_array[..., i] * sigma + mu
        max_t = min(plotting_times.shape[0], rl_acts.shape[0])
        tplot = plotting_times[:max_t]
        rl_acts = rl_acts * multipliers.get(actname, 1.0)

        lower_perc = np.percentile(rl_acts[:max_t], 5, axis=0)
        upper_perc = np.percentile(rl_acts[:max_t], 95, axis=0)
        ax.fill_between(tplot, lower_perc, upper_perc, color="blue", alpha=0.1)
        ax.plot(tplot, rl_acts[:max_t], color="blue", linewidth=1.6, label="Actuator Value")

        ax.set_title(all_titles.get(actname, [actname])[0])
        ax.set_xlim(1200, tplot[-1])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(ylabel.get(actname, actname))
        ax.grid(alpha=0.25)

    for j in range(len(actuators_to_plot[:4]), len(act_axes)):
        fig.delaxes(act_axes[j])

    # Block titles.
    fig.text(0.31, 0.985,
             fr"Simulated Shot {shot_id} {quan_names.capitalize()} Evolution at Selected $\psi_n$ Values",
             ha="center", va="top", fontsize=16)
    fig.text(0.79, 0.985,
             f"Shot {shot_id} Actuation Signals",
             ha="center", va="top", fontsize=16)

    # Separate legends for each block.
    prof_handles, prof_labels = profile_axes[0].get_legend_handles_labels()
    prof_dict = OrderedDict()
    for h, l in zip(prof_handles, prof_labels):
        if l not in prof_dict:
            prof_dict[l] = h

    act_handles, act_labels = act_axes[0].get_legend_handles_labels() if actuators_to_plot else ([], [])
    act_dict = OrderedDict()
    for h, l in zip(act_handles, act_labels):
        if l not in act_dict:
            act_dict[l] = h

    if prof_dict:
        fig.legend(prof_dict.values(), prof_dict.keys(),
                   loc='lower center', bbox_to_anchor=(0.29, 0.035),
                   ncol=2, frameon=True, fontsize=13)
    if act_dict:
        fig.legend(act_dict.values(), act_dict.keys(),
                   loc='lower center', bbox_to_anchor=(0.79, 0.035),
                   ncol=1, frameon=True, fontsize=13)

    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.05, right=0.98)

    save_path = os.path.join(log_folder, f"{shot_id}_profile_actuator_combined.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined plot: {save_path}")
