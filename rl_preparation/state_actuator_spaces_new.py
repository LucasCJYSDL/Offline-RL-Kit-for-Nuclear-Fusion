import pickle
import os
import sys

import numpy as np
repo_root = os.path.expanduser("/zfsauton2/home/jiayuc2/Proj_8/Offline-RL-Kit-for-Nuclear-Fusion")
sys.path.append(repo_root)
from envs.utils.observation_calculators import PTerm
from envs.utils.actuator_bounding.beam_bounder import D3dTotalPowerTorqueBounding
from envs.utils.actuator_bounding.actuator_bounder import CompositeActuatorBounder
from envs.utils.rewards import ProfileTrackingReward, TrackingReward


track_signals = [
            #   "rotation_component1", 
            #   "rotation_component2", 
            #   "rotation_component3", 
            #   "rotation_component4",
            #   "dens_component1", 
            #   "dens_component2", 
            #   "dens_component3", 
            #   "dens_component4",
              "betan_EFIT01"]
#!!! what you need to specify
# which states/actuators are actually used
# obs_in_use = ["betan_EFIT01",
#                 "temp_component1", 
#                 "temp_component2", 
#                 "temp_component3", 
#                 "temp_component4", 
#                 "itemp_component1", 
#                 "itemp_component2", 
#                 "itemp_component3", 
#                 "itemp_component4", 
#                 "dens_component1", 
#                 "dens_component2", 
#                 "dens_component3", 
#                 "dens_component4", 
#                 "rotation_component1", 
#                 "rotation_component2", 
#                 "rotation_component3", 
#                 "rotation_component4", 
#                 "pres_EFIT01_component1", 
#                 "pres_EFIT01_component2", 
#                 "q_EFIT01_component1", 
#                 "q_EFIT01_component2"]

obs_in_use = ["rotation_component1", 
              "rotation_component2", 
              "rotation_component3", 
              "rotation_component4"]
#obs_in_use =["betan_EFIT01"]
# obs_in_use=[ "dens_component1", 
#                  "dens_component2", 
#                  "dens_component3", 
#                  "dens_component4" ]
# obs_in_use = [
#     'betan_EFIT01',
#     "temp_component1",
#     "temp_component2",
#     "temp_component3",
#     "temp_component4",
#     "itemp_component1",
#     "itemp_component2",
#     "itemp_component3",
#     "itemp_component4",
#     "dens_component1",
#     "dens_component2",
#     "dens_component3",
#     "dens_component4",
#     "rotation_component1",
#     "rotation_component2",
#     "rotation_component3",
#     "rotation_component4",
#     "pres_EFIT01_component1",
#     "pres_EFIT01_component2",
#     "q_EFIT01_component1",
#     "q_EFIT01_component2"
# ]

# acts_in_use= ['pinj','tinj', 'bt_magnitude', 'bt_is_positive','ech_pwr_total']

acts_in_use= ['pinj','tinj', 'gasA','ech_pwr_total']

action_space = [
    'pinj_velocity',
    'tinj_velocity',
    'gasA_velocity',
    # 'ech_pwr_total_velocity'
    ]

# computed_obs_in_use = [
#     PTerm(signal_name="rotation_component1", target_idx=0),
#     PTerm(signal_name="rotation_component2", target_idx=0),
#     PTerm(signal_name="rotation_component3", target_idx=0),
#     PTerm(signal_name="rotation_component4", target_idx=0),
# ]

computed_obs_in_use = [
    # PTerm(signal_name="dens_component1", target_idx=0),
    # PTerm(signal_name="dens_component2", target_idx=0),
    # PTerm(signal_name="dens_component3", target_idx=0),
    # PTerm(signal_name="dens_component4", target_idx=0),
    PTerm(signal_name="betan_EFIT01", target_idx=0),
]

beams = ['30L', '30R', '150L', '150R', '210L', '210R', '330L', '330R']

voltages = {beam: 75000 for beam in beams}
perveances = {beam: 2.55 for beam in beams}

rtans = {
    '30L': 1.149,   '30R': 0.749,
    '150L': 1.149,  '150R': 0.749,
    '210L': -0.749, '210R': -1.149,
    '330L': 1.149,  '330R': 0.749,
}

min_duty_cycle = {
    '30L': 0.5,   '30R': 0.0,    
    '150L': 0.0,  '150R': 0.0,
    '210L': 0.0,  '210R': 0.0,
    '330L': 0.5,  '330R': 0.0,   
}

targets_in_obs = True
k_step_targets_in_obs = 10 # default 1 which is current step only
discrete_k_target_idx_in_obs = [0,9]
add_tm_probs_to_obs = False

beam_bounder = D3dTotalPowerTorqueBounding(
    voltages=voltages,
    perveances=perveances,
    rtans=rtans,
    min_duty_cycle=min_duty_cycle
)

target_lows = [-0.17593358763413988] # 1.5
target_highs= [1.0847411701303655] # 2.5

actuator_bounder = CompositeActuatorBounder(bounders=[beam_bounder])

reward_function = ProfileTrackingReward(
        unnormalize=False,
        profile_name=["rotation"],
        square_costs=True,
        track_signals=track_signals,
    )

# reward_function = TrackingReward(
#     track_signals=track_signals,
#     track_coefficients = [1],
# )

horizon = 150

# functions that you do not need to modify
def state_names_to_idxs(data_path):
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        all_states = info['state_space']
        idxs = []
        for s in obs_in_use:
            idxs.append(all_states.index(s))
    return idxs

# def actuator_names_to_idxs(data_path):
#     with open(data_path + '/info.pkl', 'rb') as f:
#         info = pickle.load(f)
#         all_acts = info['actuator_space']
#         idxs = []
#         for a in acts_in_use:
#             idxs.append(all_acts.index(a))
#     return idxs

# get the indices of tracking targets in the observation space
def get_target_indices(tracking_target, obs_dim):
    indices = []
    for i in range(obs_dim):
        if obs_in_use[i].startswith(tracking_target):
            indices.append(i)
    
    return indices

def actuator_names_to_idxs(data_path):
    """Return indices of actuators to be used in observation, considering velocity logic."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        if np.sum(['velocity' in a for a in info['actuator_space']]) == 0:
            actuator_ob_idxs = [info['actuator_space'].index(a)
                                for a in acts_in_use if 'velocity' not in a]
            next_ob_idxs = [info['next_actuator_space'].index(a)
                            for a in acts_in_use if 'velocity' in a]
        else:
            actuator_ob_idxs = [info['actuator_space'].index(a) for a in acts_in_use]
            next_ob_idxs = []
    return actuator_ob_idxs, next_ob_idxs


def vel_act_and_posn_act_idxs():
    """Return indices of velocity actions and position (absolute) actions."""
    vel_act_idxs = [i for i, a in enumerate(action_space) if 'velocity' in a]
    posn_act_idxs = [i for i, a in enumerate(action_space) if 'velocity' not in a]
    return vel_act_idxs, posn_act_idxs


def actuator_act_and_next_act_idxs(data_path):
    """Return mapping from actions to actuator indices and next_actuator indices."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        actuator_act_idxs = [info['actuator_space'].index(a[:-len('_velocity')])
                             if 'velocity' in a else info['actuator_space'].index(a)
                             for a in action_space]
        nxts_act_idxs = [info['next_actuator_space'].index(a)
                         if 'velocity' in a else info['next_actuator_space'].index(a + '_velocity')
                         for a in action_space]
    return actuator_act_idxs, nxts_act_idxs


def pinj_tinj_idxs(data_path):
    """Return pinj and tinj actuator indices if they exist."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        if 'pinj' in info['actuator_space']:
            pinj_actuator_idx = info['actuator_space'].index('pinj')
            pinj_next_actuator_idx = info['next_actuator_space'].index('pinj_velocity')
        else:
            pinj_actuator_idx = None
            pinj_next_actuator_idx = None

        if 'tinj' in info['actuator_space']:
            tinj_actuator_idx = info['actuator_space'].index('tinj')
            tinj_next_actuator_idx = info['next_actuator_space'].index('tinj_velocity')
        else:
            tinj_actuator_idx = None
            tinj_next_actuator_idx = None
    return pinj_actuator_idx, pinj_next_actuator_idx, tinj_actuator_idx, tinj_next_actuator_idx


def actuator_posn_and_vel_idxs(data_path):
    """Return indices separating actuator positions and velocities."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        actuator_posn_idxs, actuator_vel_idxs = [], []
        for i, a in enumerate(info['actuator_space']):
            if 'velocity' in a:
                actuator_vel_idxs.append(i)
            else:
                actuator_posn_idxs.append(i)
    return actuator_posn_idxs, actuator_vel_idxs


def state_posn_and_vel_idxs(data_path):
    """Return indices separating state positions and velocities."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        state_posn_idxs, state_vel_idxs = [], []
        for i, s in enumerate(info['state_space']):
            if 'velocity' in s:
                state_vel_idxs.append(i)
            else:
                state_posn_idxs.append(i)
    return state_posn_idxs, state_vel_idxs


def pred_vel_idx(data_path, labels_are_velocities):
    """Return indices of next states that are velocities."""
    with open(data_path + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        if labels_are_velocities:
            pred_vel_idxs = [i for i, s in enumerate(info['next_state_space'])
                             if s in info['state_space']]
        else:
            pred_vel_idxs = [i for i, s in enumerate(info['next_state_space'])
                             if (s + '_velocity') in info['state_space']]
    return pred_vel_idxs