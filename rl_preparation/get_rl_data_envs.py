import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_preparation.state_actuator_spaces_new import ( 
    state_names_to_idxs, 
    actuator_names_to_idxs, 
    get_target_indices, 
    acts_in_use, 
    action_space,
    computed_obs_in_use,
    discrete_k_target_idx_in_obs,
    k_step_targets_in_obs,
    add_tm_probs_to_obs,
    targets_in_obs,
    track_signals,
    vel_act_and_posn_act_idxs,  
    actuator_act_and_next_act_idxs, 
    pinj_tinj_idxs,
    actuator_posn_and_vel_idxs,
    state_posn_and_vel_idxs,
    pred_vel_idx,
    target_lows,
    target_highs,
    horizon
)
from rl_preparation.process_raw_data import raw_data_dir, rl_data_path, il_data_path, tracking_data_path, reference_shot, training_model_dir, evaluation_model_dir, change_every
from envs.utils.setup_targets import fixed_ref_shot_targets, step_function_targets, original_trajectory_targets, uniform_targets,original_trajectory_targets_new

# load the offline dataset from the disk
def load_offline_data(env, tracking_target, is_il):
    # get general data 
    offline_data = {}

    if is_il:
        general_data_path = il_data_path
    else:
        general_data_path = rl_data_path
    
    hdf = h5py.File(general_data_path, 'r')
    offline_data['observations'] = hdf['observations'][:]
    offline_data['actions'] = hdf['actions'][:]
    offline_data['pre_actions'] = hdf['pre_actions'][:]
    offline_data['next_observations'] = hdf['next_observations'][:]
    offline_data['terminals'] = hdf['terminals'][:]
    offline_data['time_step'] = hdf['time_step'][:]
    offline_data['action_lower_bounds'] = hdf['action_lower_bounds'][:]
    offline_data['action_upper_bounds'] = hdf['action_upper_bounds'][:]
    offline_data['action_positions_lower_bounds'] = hdf['action_positions_lower_bounds'][:]
    offline_data['action_positions_upper_bounds'] = hdf['action_positions_upper_bounds'][:]
    offline_data['action_velocity_lower_bounds'] = hdf['action_velocity_lower_bounds'][:]
    offline_data['action_velocity_upper_bounds'] = hdf['action_velocity_upper_bounds'][:]
    offline_data['states_positions_lower_bounds'] = hdf['states_positions_lower_bounds'][:]
    offline_data['states_positions_upper_bounds'] = hdf['states_positions_upper_bounds'][:]
    offline_data['states_velocity_lower_bounds'] = hdf['states_velocity_lower_bounds'][:]
    offline_data['states_velocity_upper_bounds'] = hdf['states_velocity_upper_bounds'][:]
    if not is_il:
        offline_data['hidden_states'] = hdf['hidden_states'][:]
    hdf.close()
    
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['actions'].shape[1]

    # if only a subset of the full state space is used, obs, act, and next_obs will go through a post process;
    # so we additionally store full versions of them.
    offline_data['full_observations'] = offline_data['observations'].copy()
    offline_data['full_actions'] = offline_data['actions'].copy()
    offline_data['full_next_observations'] = offline_data['next_observations'].copy()

    # get indices of the tracking target in the state space
    with open(raw_data_dir + '/info.pkl', 'rb') as file:
        data_info = pickle.load(file)

    offline_data['index_list'] = []
    offline_data['tracking_target_names'] = []
    keyword = tracking_target
    for i in range(offline_data['obs_dim']):
        if data_info['state_space'][i].startswith(keyword):
            offline_data['index_list'].append(i)
            offline_data['tracking_target_names'].append(data_info['state_space'][i])

    # get the tracking data
    tracking_data = {} 

    with h5py.File(tracking_data_path, 'r') as hdf:
        ref_shot = hdf[str(reference_shot)]['tracking_states'][:]
        ref_shot_next = hdf[str(reference_shot)]['tracking_next_states'][:]

        for shot_id in hdf:
            tracking_data[int(shot_id)] = {}
            shot = hdf[shot_id]
            for key in shot:
                tracking_data[int(shot_id)][key] = shot[key][:]

            # get targets for the tracking data (evaluation)
            if env == "base":
                tracking_data[int(shot_id)]['tracking_ref'] = fixed_ref_shot_targets(ref_shot_next, offline_data['index_list'], None)
            elif env == "profile_control": # TODO: use the first option (commented for now)
                #change evaluation targets
                if tracking_target in ['dens', 'rotation']:
                    tracking_data[int(shot_id)]['tracking_ref'] = original_trajectory_targets_new(tracking_data[int(shot_id)]['tracking_states'], offline_data['index_list'], horizon, None, eval_mode=True)
                else:
                    tracking_data[int(shot_id)]['tracking_ref'] = uniform_targets(target_lows, target_highs, horizon)                
                # tracking_data[int(shot_id)]['tracking_ref'] = step_function_targets(ref_shot, offline_data['index_list'], None, change_every)
                #tracking_data[int(shot_id)]['tracking_ref'] = step_function_targets(tracking_data[int(shot_id)]['tracking_states'], offline_data['index_list'], None, change_every)
            elif env == "fusion_env":
                if tracking_target in ['dens', 'rotation']:
                    tracking_data[int(shot_id)]['tracking_ref'] = original_trajectory_targets(tracking_data[int(shot_id)]['tracking_states'], offline_data['index_list'], horizon, None, eval_mode=True)
                else:
                    tracking_data[int(shot_id)]['tracking_ref'] = uniform_targets(target_lows, target_highs, horizon)
            else:
                raise NotImplementedError

    # get targets for the general data (training) 
    if env == "base": # TODO: decouple the target setting menthod with the env type
        offline_data['tracking_ref'] = fixed_ref_shot_targets(ref_shot_next, offline_data['index_list'], offline_data['terminals'])
    elif env == "profile_control":
        #change tariningtargets
        if tracking_target in ['dens', 'rotation']:
            print("x----x----x--")
            offline_data['tracking_ref'] = original_trajectory_targets_new(offline_data['observations'], offline_data['index_list'],150, offline_data['terminals'])
            print("x-----x----x---")
        else:
            offline_data['tracking_ref'] = uniform_targets(target_lows, target_highs, horizon)
            #offline_data['tracking_ref'] = step_function_targets(offline_data['observations'], offline_data['index_list'], offline_data['terminals'], change_every)
        # TODO: usig 'next_observations'
    elif env == "fusion_env":
        if tracking_target in ['dens', 'rotation']:
            offline_data['tracking_ref'] = original_trajectory_targets(offline_data['observations'], offline_data['index_list'],150, offline_data['terminals'])
        else:
            offline_data['tracking_ref'] = uniform_targets(target_lows, target_highs, horizon)
    else:
        raise NotImplementedError
    
    _labels_are_velocities = 'velocity' in data_info['next_state_space'][0]
    # if only using part of the full state space, we need to update some quantities.
    state_idxs = state_names_to_idxs(raw_data_dir)
    action_idxs, next_ob_idxs = actuator_names_to_idxs(raw_data_dir)
    vel_act_idxs, posn_act_idxs = vel_act_and_posn_act_idxs()
    actuator_act_idxs, nxts_act_idxs = actuator_act_and_next_act_idxs(raw_data_dir)
    pinj_actuator_idx, pinj_next_actuator_idx, tinj_actuator_idx, tinj_next_actuator_idx = pinj_tinj_idxs(raw_data_dir)
    actuator_posn_idxs, actuator_vel_idxs = actuator_posn_and_vel_idxs(raw_data_dir)
    state_posn_idxs, state_vel_idxs = state_posn_and_vel_idxs(raw_data_dir)
    pred_vel_idxs = pred_vel_idx(raw_data_dir, _labels_are_velocities)

    offline_data['state_idxs'] = state_idxs
    offline_data['action_idxs'] = action_idxs
    offline_data['action_names'] = action_space
    offline_data['vel_act_idxs'] = vel_act_idxs
    offline_data['posn_act_idxs'] = posn_act_idxs
    offline_data['actuator_act_idxs'] = actuator_act_idxs
    offline_data['nxts_act_idxs'] = nxts_act_idxs
    offline_data['pinj_actuator_idx'] = pinj_actuator_idx
    offline_data['pinj_next_actuator_idx'] = pinj_next_actuator_idx
    offline_data['tinj_actuator_idx'] = tinj_actuator_idx
    offline_data['tinj_next_actuator_idx'] = tinj_next_actuator_idx
    offline_data['actuator_posn_idxs'] = actuator_posn_idxs
    offline_data['actuator_vel_idxs'] = actuator_vel_idxs
    offline_data['state_posn_idxs'] = state_posn_idxs
    offline_data['state_vel_idxs'] = state_vel_idxs
    offline_data['pred_vel_idxs'] = pred_vel_idxs
    offline_data['next_ob_idxs'] = next_ob_idxs
    offline_data['_labels_are_velocities'] = _labels_are_velocities

    offline_data['observations'] = offline_data['observations'][:, state_idxs]
    offline_data['next_observations'] = offline_data['next_observations'][:, state_idxs]
    offline_data['actions'] = offline_data['actions'][:, action_idxs]
    offline_data['action_lower_bounds'] = offline_data['action_lower_bounds'][action_idxs]
    offline_data['action_upper_bounds'] = offline_data['action_upper_bounds'][action_idxs]
    offline_data['obs_dim'] = offline_data['observations'].shape[1]
    offline_data['act_dim'] = offline_data['actions'].shape[1]
    offline_data['index_list'] = get_target_indices(tracking_target, offline_data['obs_dim'])

    return offline_data, tracking_data

# get the offline rl data (in d4rl format) and training
def get_rl_data_envs(env_id, task, device, is_il=False):
    offline_data, tracking_data = load_offline_data(env_id, task, is_il)

    if env_id == 'base':
        from envs.base_env import NFBaseEnv, SA_processor
        sa_processor = SA_processor(offline_data, tracking_data, device)
        env = NFBaseEnv(evaluation_model_dir, sa_processor, offline_data, tracking_data[reference_shot], reference_shot, device) # this is the env for evaluation

    elif env_id == 'profile_control':
        from envs.profile_control_env import ProfileControlEnv
        from envs.base_env import SA_processor
        sa_processor = SA_processor(offline_data, tracking_data, device)
        env = ProfileControlEnv(evaluation_model_dir, sa_processor, offline_data, tracking_data, reference_shot, device) # this is the env for evaluation
    
    elif env_id == 'fusion_env':
        from envs.fusion_env import FusionEnv, SA_processor
        sa_processor = SA_processor(offline_data, tracking_data, device)
        env = FusionEnv(evaluation_model_dir, sa_processor, offline_data, tracking_data, reference_shot, device)
    
    else:
        raise NotImplementedError
    
    # collect the data for rl training: (s, a, r, s', d), where d denotes the termination signal
    # For fusion_env, the rewards and actions are processed in the env step function.
    if(env_id != "fusion_env"):
        offline_data['rewards'] = sa_processor.get_reward(offline_data['next_observations'], offline_data['time_step'])
        offline_data['actions'] = sa_processor.normalize_action(offline_data['actions'])
        offline_data['observations'] = sa_processor.get_rl_state(offline_data['observations'], batch_idx=np.arange(0, offline_data['observations'].shape[0]))
        offline_data['next_observations'] = sa_processor.get_rl_state(offline_data['next_observations'], batch_idx=np.arange(1, offline_data['observations'].shape[0]+1))
    # For next_obs where termination is True, the tracking targets for them might be problematic. 
    # However, this doesn't affect training since the bootstrapping from those next_obs will be masked out.
    else:
        if discrete_k_target_idx_in_obs is not None:
            num_targets_in_obs = len(discrete_k_target_idx_in_obs)
        else:
            num_targets_in_obs = k_step_targets_in_obs
        offline_data['obs_dim'] = (
                len(offline_data['state_idxs'])
                + len(offline_data['action_idxs'])
                + len(offline_data['next_ob_idxs'])
                + int(np.sum([ob.num_obs_computed * num_targets_in_obs
                                if 'PTerm' in str(ob) else ob.num_obs_computed for ob in computed_obs_in_use ]))
                + targets_in_obs * len(track_signals) * num_targets_in_obs
                + add_tm_probs_to_obs)
        offline_data['act_dim'] = len(action_space)
    return offline_data, sa_processor, env, training_model_dir
    

if __name__ == "__main__":
    # dummy test only
    # offline_data, tracking_data = load_offline_data(env="profile_control", tracking_target='betan_EFIT01')
    # print(offline_data['index_list'])
    # print(tracking_data.keys())
    # for k, v in tracking_data[reference_shot].items():
    #     print(k, v.shape)
    import torch
    offline_data, sa_processor, env, training_model_dir = get_rl_data_envs("base", "betan_EFIT01", torch.device("cuda"))
    print(offline_data['observations'].shape, offline_data['next_observations'].shape, offline_data['rewards'].shape)