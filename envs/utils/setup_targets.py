"""
Extract tracking targets from given trajectories.
"""
import numpy as np


def step_function_targets(obs_seq, target_idxs, terminal_seq, change_every=50):
    """
    Generate a sequence of targets that change every `change_every` steps.
    """
    tot_len = len(obs_seq)
    target_dim = len(target_idxs)

    if terminal_seq is None: # a trick to handle the difference between general data and tracking data
        terminal_seq = np.zeros((tot_len, ), dtype=bool)
        terminal_seq[-1] = True

    targets = np.zeros((tot_len, target_dim), dtype=np.float32)
    s_id = 0
    for i in range(tot_len):
        if (i+1) % change_every == 0 or terminal_seq[i] or i == (tot_len - 1):
            tmp_target = obs_seq[i][target_idxs]
            targets[s_id:(i+1)] = tmp_target # TODO: add some randomness here
            s_id = i + 1

    return targets

def fixed_ref_shot_targets(ref_obs_seq, target_idxs, terminal_seq):
    """
    Use tracking targets in a reference shot as the targets for each shot in the offline dataset.
    """
    ref_shot_len = len(ref_obs_seq)
    ref_targets = ref_obs_seq[:, target_idxs]
    padding_target = ref_targets[-1]

    if terminal_seq is None: # a trick to handle the difference between general data and tracking data
        terminal_seq = np.zeros((ref_shot_len, ), dtype=bool)
        terminal_seq[-1] = True

    tot_len = len(terminal_seq)
    # for shots longer than the reference shot, we fill in the remaining time steps with the final state
    targets = np.array([padding_target for _ in range(tot_len)]) 
    s_id = 0
    for i in range(tot_len):
        if terminal_seq[i] or i == (tot_len - 1):
            tmp_shot_len = i - s_id + 1
            fill_len = min(tmp_shot_len, ref_shot_len)
            targets[s_id:(s_id+fill_len)] = ref_targets[:fill_len]
            s_id = i + 1
    
    return targets

def uniform_targets(target_lows, target_highs, horizon):
    """
    Generate uniform random targets within specified bounds.
    
    Args:
        target_lows: List of lower bounds for each target dimension
        target_highs: List of upper bounds for each target dimension  
        num_targets: Number of target sequences to generate
        horizon: Length of each target sequence
        
    Returns:
        targets: Array of shape (horizon, target_dimensions)
    """
    assert len(target_lows) == len(target_highs), \
        f'Length of target_lows must match length of target_highs but received {len(target_lows)} and {len(target_highs)}'
    
    targets = np.random.uniform(target_lows, target_highs,
                                size=len(target_lows))
    return np.repeat(targets[np.newaxis,:], horizon, axis=0)

def original_trajectory_targets(obs_seq, target_idxs,horizon, terminal_seq, 
                               near_reference_target=False, eval_mode=False, 
                               fixed_profile_target=True):
    """
    Generate targets based on observation sequence with profile processing.
    Mimics OriginalTrajectoryTarget behavior.
    
    Args:
        obs_seq: Observation sequence array, shape (total_steps, obs_dim)
        target_idxs: Indices of target dimensions to extract  
        terminal_seq: Terminal flags for episode boundaries
        near_reference_target: Whether to use nearby episodes as reference
        eval_mode: Whether in evaluation mode
        fixed_profile_target: Whether to use fixed profile targets
    
    Returns:
        targets: Array of shape (total_steps, len(target_idxs))
    """
    tot_len = len(obs_seq)
    target_dim = len(target_idxs)
    
    if terminal_seq is None:
        terminal_seq = np.zeros((tot_len), dtype=bool)
        terminal_seq[-1] = True
    
    targets = np.zeros((horizon, obs_seq.shape[1]), dtype=np.float32)
    
    # Split data into episodes based on terminal_seq
    episodes = _split_into_episodes(obs_seq, terminal_seq)
    
    # Generate targets for each episode
    s_id = 0
    for episode_idx, episode_obs in enumerate(episodes):
        # Select reference episode
        # 与原环境逻辑不太一致需要修改
        if near_reference_target and len(episodes) > 1:
            available_episodes = [i for i in range(len(episodes)) if i != episode_idx]
            if available_episodes:
                if eval_mode:
                    ref_episode_idx = available_episodes[0]
                else:
                    ref_episode_idx = np.random.choice(available_episodes)
                ref_episode = episodes[ref_episode_idx]
            else:
                ref_episode = episode_obs
        else:
            # Use current episode as reference
            ref_episode = episode_obs
        
        # Generate targets for this episode
        episode_targets = _generate_original_trajectory_targets(
            ref_episode, target_idxs, horizon, eval_mode, fixed_profile_target
        )
        
        targets = episode_targets
        s_id += horizon
    
    return targets[:, target_idxs]


def _generate_original_trajectory_targets(ref_episode, target_idxs, horizon, 
                                        eval_mode, fixed_profile_target):
    """
    Generate targets for a single episode, exactly like OriginalTrajectoryTarget.
    
    Args:
        ref_episode: Reference episode observations
        target_idxs: Target dimension indices
        horizon: Current episode length
        eval_mode: Whether in evaluation mode
        fixed_profile_target: Whether to use fixed profile
    
    Returns:
        target: Array of shape (horizon, len(target_idxs))
    """
    target_dim = len(target_idxs)
    
    # Extract target dimensions from reference episode (模拟原代码的 target = trajectories[r_]['states'][start_idx:horizon+start_idx, :])
    if len(ref_episode) > 0:
        target = ref_episode[:150,:]
        
        # Handle length mismatch 
        if len(target) < horizon:
            if len(target) == 0:
                # special case for len(target) == 0 -- no data picked, just repeat whatever we have
                target = ref_episode[-1, target_idxs]
                target = np.tile(target, (horizon, 1))
            else:
                # repeat last value
                last_value = target[-1]  # Get the last value in the target array
                repetitions = np.tile(last_value, (horizon - len(target), 1))  # Repeat the last value
                target = np.vstack((target, repetitions))
    else:
        # Fallback
        target = np.zeros((horizon, target_dim))
    
    # Apply fixed profile processing 
    if fixed_profile_target:
        # pick two samples from given trajectory and repeat them
        if eval_mode:
            midpoint = horizon // 2 
            t1 = midpoint - 35
            t2 = midpoint + 35
        else:
            midpoint = horizon // 2
            quaterpoint = int(horizon * 0.65)
            if midpoint >= quaterpoint:
                t1 = midpoint 
            else:
                t1 = np.random.randint(midpoint, quaterpoint)

            t2 = np.random.randint(quaterpoint, horizon)
            
            #t2=quaterpoint+3
            # flip t1 and t2 by some probability
            if np.random.rand() > 0.5:
                t1, t2 = t2, t1
            # print(f"t1: {t1}")
            # print(f"t2: {t2}")
  
        # repeat t1 timepoint till midpoint and t2 timepoint till end
        new_target = np.vstack((
            np.tile(target[t1, :], (midpoint, 1)),
            np.tile(target[t2, :], (horizon - midpoint, 1))
        ))
        target = new_target
    
    # 如果 fixed_profile_target=False，直接return原始目标序列
    return target

# new original_trajectory_targets
def original_trajectory_targets_new(obs_seq, target_idxs, horizon, terminal_seq,
                                   near_reference_target=False, eval_mode=False,
                                   fixed_profile_target=True):
    """
    Generate targets based on observation sequence with profile processing.
    Returns targets with length equal to len(terminal_seq) instead of fixed horizon.

    Args:
        obs_seq: Observation sequence array, shape (total_steps, obs_dim)
        target_idxs: Indices of target dimensions to extract
        horizon: Horizon length for generating base targets (e.g., 150)
        terminal_seq: Terminal flags for episode boundaries
        near_reference_target: Whether to use nearby episodes as reference
        eval_mode: Whether in evaluation mode
        fixed_profile_target: Whether to use fixed profile targets

    Returns:
        targets: Array of shape (len(terminal_seq), len(target_idxs))
    """
    tot_len = len(obs_seq)
    target_dim = len(target_idxs)

    if terminal_seq is None: # a trick to handle the difference between general data and tracking data
        terminal_seq = np.zeros((tot_len, ), dtype=bool)
        terminal_seq[-1] = True

    # First generate the base targets using original function logic
    base_targets = np.zeros((horizon, obs_seq.shape[1]), dtype=np.float32)

    # Split data into episodes based on terminal_seq
    episodes = _split_into_episodes(obs_seq, terminal_seq)

    # Generate base targets for the first episode (or reference episode)
    for episode_idx, episode_obs in enumerate(episodes):
        # Select reference episode
        # 与原环境逻辑不太一致需要修改
        if near_reference_target and len(episodes) > 1:
            available_episodes = [i for i in range(len(episodes)) if i != episode_idx]
            if available_episodes:
                if eval_mode:
                    ref_episode_idx = available_episodes[0]
                else:
                    ref_episode_idx = np.random.choice(available_episodes)
                ref_episode = episodes[ref_episode_idx]
            else:
                ref_episode = episode_obs
        else:
            # Use current episode as reference
            ref_episode = episode_obs

        # Generate base targets (horizon length)
        base_targets = _generate_original_trajectory_targets(
            ref_episode, target_idxs, horizon, eval_mode, fixed_profile_target
        )

    # Now create new_targets with length tot_len
    new_targets = np.zeros((tot_len, target_dim), dtype=np.float32)

    e_id = 0
    for i in range(tot_len):
        if terminal_seq[i]:
            # Episode ends at position i
            if i - e_id < horizon:
                # Within base targets range
                new_targets[i] = base_targets[i - e_id, target_idxs]
            else:
                # Beyond base targets range, use last target
                new_targets[i] = base_targets[horizon - 1, target_idxs]
            e_id = i + 1
        else:
            # Within episode
            if i - e_id < horizon:
                # Within base targets range
                new_targets[i] = base_targets[i - e_id, target_idxs]
            else:
                # Beyond base targets range, use last target
                new_targets[i] = base_targets[horizon - 1, target_idxs]

    return new_targets

def _split_into_episodes(obs_seq, terminal_seq):
    """Split observation sequence into episodes based on terminal flags."""
    episodes = []
    s_id = 0
    
    for i in range(len(terminal_seq)):
        if terminal_seq[i] or i == (len(terminal_seq) - 1):
            episode = obs_seq[s_id:(i+1)]
            episodes.append(episode)
            s_id = i + 1
    
    return episodes

