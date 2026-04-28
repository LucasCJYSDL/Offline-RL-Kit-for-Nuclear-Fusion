import os
import pickle

import numpy as np
from envs.utils.observation_calculators import PTerm, ITermCalculator, SimpleDTerm
from envs.utils.actuator_bounding.beam_bounder import D3dTotalPowerTorqueBounding
from envs.utils.actuator_bounding.actuator_bounder import CompositeActuatorBounder
from envs.utils.rewards import ProfileTrackingReward, TrackingReward, ProfileTrackingWithBetanReward

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

beam_bounder = D3dTotalPowerTorqueBounding(
    voltages=voltages,
    perveances=perveances,
    rtans=rtans,
    min_duty_cycle=min_duty_cycle
)

actuator_bounder = CompositeActuatorBounder(bounders=[beam_bounder])

AVAILABLE_TASKS = ('temp', 'rotation', 'dens', 'pres', 'q', 'betan')
DEFAULT_TASK = os.getenv('OFFLINERLKIT_TASK', 'temp')
CURRENT_TASK = None


def _profile_obs(prefix):
    return [f'{prefix}_component{i}' for i in range(1, 5)]


def _profile_terms(prefix, count):
    return [PTerm(signal_name=f'{prefix}_component{i}', target_idx=0) for i in range(1, count + 1)]


def _normalize_task_name(task_name):
    task_name = task_name.lower().strip()
    if task_name in AVAILABLE_TASKS:
        return task_name
    if task_name.endswith('_efit01'):
        stripped = task_name[:-len('_efit01')]
        if stripped in AVAILABLE_TASKS:
            return stripped
    return task_name


def normalize_task_name(task_name):
    """Normalize legacy task names to the canonical registry task name."""
    return _normalize_task_name(task_name)


def _task_spec(task_name):
    task_name = _normalize_task_name(task_name)
    if task_name == 'temp':
        obs = _profile_obs('temp')
        track = obs[:]
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA', 'ech_pwr_total'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity', 'ech_pwr_total_velocity'],
            computed_obs_in_use=_profile_terms('temp', 4),
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=10,
            discrete_k_target_idx_in_obs=[0, 9],
            add_tm_probs_to_obs=False,
            reward_function=ProfileTrackingReward(
                unnormalize=False,
                profile_name=['temp'],
                square_costs=True,
                track_signals=track,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    if task_name == 'rotation':
        obs = _profile_obs('rotation')
        track = obs[:]
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA', 'ech_pwr_total'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity', 'ech_pwr_total_velocity'],
            computed_obs_in_use=_profile_terms('rotation', 4),
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=10,
            discrete_k_target_idx_in_obs=[0, 9],
            add_tm_probs_to_obs=False,
            reward_function=ProfileTrackingReward(
                unnormalize=False,
                profile_name=['rotation'],
                square_costs=True,
                track_signals=track,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    if task_name == 'dens':
        obs = _profile_obs('dens')
        track = obs[:]
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA', 'ech_pwr_total'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity', 'ech_pwr_total_velocity'],
            computed_obs_in_use=_profile_terms('dens', 4),
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=10,
            discrete_k_target_idx_in_obs=[0, 9],
            add_tm_probs_to_obs=False,
            reward_function=ProfileTrackingReward(
                unnormalize=False,
                profile_name=['dens'],
                square_costs=True,
                track_signals=track,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    if task_name == 'pres':
        obs = ['pres_EFIT01_component1', 'pres_EFIT01_component2']
        track = obs[:]
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA', 'ech_pwr_total'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity', 'ech_pwr_total_velocity'],
            computed_obs_in_use=[PTerm(signal_name='pres_EFIT01_component1', target_idx=0),
                                 PTerm(signal_name='pres_EFIT01_component2', target_idx=0)],
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=10,
            discrete_k_target_idx_in_obs=[0, 9],
            add_tm_probs_to_obs=False,
            reward_function=ProfileTrackingReward(
                unnormalize=False,
                profile_name=['pres_EFIT01'],
                square_costs=True,
                track_signals=track,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    if task_name == 'q':
        obs = ['q_EFIT01_component1', 'q_EFIT01_component2']
        track = obs[:]
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA', 'ech_pwr_total'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity', 'ech_pwr_total_velocity'],
            computed_obs_in_use=[PTerm(signal_name='q_EFIT01_component1', target_idx=0),
                                 PTerm(signal_name='q_EFIT01_component2', target_idx=0)],
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=10,
            discrete_k_target_idx_in_obs=[0, 9],
            add_tm_probs_to_obs=False,
            reward_function=ProfileTrackingReward(
                unnormalize=False,
                profile_name=['q_EFIT01'],
                square_costs=True,
                track_signals=track,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    if task_name == 'betan':
        obs = [
            'betan_EFIT01',
            'temp_component1',
            'temp_component2',
            'temp_component3',
            'temp_component4',
            'itemp_component1',
            'itemp_component2',
            'itemp_component3',
            'itemp_component4',
            'dens_component1',
            'dens_component2',
            'dens_component3',
            'dens_component4',
            'rotation_component1',
            'rotation_component2',
            'rotation_component3',
            'rotation_component4',
            'pres_EFIT01_component1',
            'pres_EFIT01_component2',
            'q_EFIT01_component1',
            'q_EFIT01_component2',
        ]
        track = ['betan_EFIT01']
        return dict(
            obs_in_use=obs,
            acts_in_use=['pinj', 'tinj', 'gasA'],
            action_space=['pinj_velocity', 'tinj_velocity', 'gasA_velocity'],
            computed_obs_in_use=[],
            track_signals=track,
            targets_in_obs=True,
            k_step_targets_in_obs=1,
            discrete_k_target_idx_in_obs=None,
            add_tm_probs_to_obs=False,
            reward_function=TrackingReward(
                track_signals=track,
                track_coefficients=[1],
                square_costs=True,
            ),
            target_lows=[-0.17593358763413988],
            target_highs=[1.0847411701303655],
            horizon=150,
        )
    raise ValueError('Unknown task: {}'.format(task_name))


def configure_task(task_name=None):
    """Configure the module-level task-dependent constants."""
    global CURRENT_TASK
    global obs_in_use
    global acts_in_use
    global action_space
    global computed_obs_in_use
    global track_signals
    global targets_in_obs
    global k_step_targets_in_obs
    global discrete_k_target_idx_in_obs
    global add_tm_probs_to_obs
    global reward_function
    global target_lows
    global target_highs
    global horizon

    if task_name is None:
        task_name = DEFAULT_TASK
    task_name = _normalize_task_name(task_name)
    spec = _task_spec(task_name)
    CURRENT_TASK = task_name
    obs_in_use = spec['obs_in_use']
    acts_in_use = spec['acts_in_use']
    action_space = spec['action_space']
    computed_obs_in_use = spec['computed_obs_in_use']
    track_signals = spec['track_signals']
    targets_in_obs = spec['targets_in_obs']
    k_step_targets_in_obs = spec['k_step_targets_in_obs']
    discrete_k_target_idx_in_obs = spec['discrete_k_target_idx_in_obs']
    add_tm_probs_to_obs = spec['add_tm_probs_to_obs']
    reward_function = spec['reward_function']
    target_lows = spec['target_lows']
    target_highs = spec['target_highs']
    horizon = spec['horizon']
    return spec


TASK_SPECS = {task: _task_spec(task) for task in AVAILABLE_TASKS}
configure_task(DEFAULT_TASK)

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
