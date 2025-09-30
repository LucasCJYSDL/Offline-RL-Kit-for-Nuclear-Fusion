import torch
import random
import numpy as np
import pickle
from typing import Dict, Sequence, List, Tuple, Optional
from  envs.utils.profile_util import reconstruct_profile_from_state

from envs.base_env import NFBaseEnv
from rl_preparation.state_actuator_spaces import ( 
    acts_in_use, 
    action_space,
    computed_obs_in_use,
    discrete_k_target_idx_in_obs,
    k_step_targets_in_obs,
    add_tm_probs_to_obs,
    targets_in_obs,
    track_signals,
    actuator_bounder,
    reward_function,
    target_lows,
    target_highs,
    horizon
)
from rl_preparation.process_raw_data import raw_data_dir

class SA_processor:  
    def __init__(self, offline_data, tracking_data, device):
        bounds = (offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
        self.range = torch.FloatTensor((bounds[1] - bounds[0]) / 2.0).unsqueeze(0).to(device) # actuator bounds
        self.mid = torch.FloatTensor((bounds[1] + bounds[0]) / 2.0).unsqueeze(0).to(device)

        self.np_range = ((bounds[1] - bounds[0]) / 2.0)[np.newaxis, :]
        self.np_mid = ((bounds[1] + bounds[0]) / 2.0)[np.newaxis, :]

        # store the tracking taregts for both the training and evaluation data
        training_data_size = offline_data['tracking_ref'].shape[0]
        self.training_tracking_targets = np.array([offline_data['tracking_ref'][-1] for _ in range(training_data_size+1)])
        self.training_tracking_targets[:training_data_size] = offline_data['tracking_ref']
        self.device = device

        self.eval_tracking_targets = {}
        self.eval_traj_states = {}
        self.eval_traj_actions = {}
        for key in tracking_data: # to avoid indexing issue, we use a padding trick here
            tracking_data_size = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_tracking_targets[key] = np.array([tracking_data[key]['tracking_ref'][-1] for _ in range(tracking_data_size+1)])
            self.eval_tracking_targets[key][:tracking_data_size] = tracking_data[key]['tracking_ref']
            # self.eval_data_sizes[key] = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_traj_states[key] = tracking_data[key]["tracking_states"]
            self.eval_traj_actions[key] = tracking_data[key]["tracking_pre_actions"][1:]
        
        # store idxs of the tracking targets to query them from the state
        # there could be multiple dimensions of the tracking target, so we may assign different coefficients to different dimensions.
        self.state_idxs = offline_data['state_idxs'] # selected state dimensions 
        self.action_idxs = offline_data["action_idxs"]
        self.vel_act_idxs = offline_data['vel_act_idxs']
        self.posn_act_idxs = offline_data['posn_act_idxs']
        self.actuator_act_idxs = np.array(offline_data['actuator_act_idxs'])
        self.nxts_act_idxs = np.array(offline_data['nxts_act_idxs'])
        self.pinj_actuator_idx = offline_data['pinj_actuator_idx']
        self.pinj_next_actuator_idx = offline_data['pinj_next_actuator_idx']
        self.tinj_actuator_idx = offline_data['tinj_actuator_idx']
        self.tinj_next_actuator_idx = offline_data['tinj_next_actuator_idx']
        self.actuator_posn_idxs = offline_data['actuator_posn_idxs']
        self.actuator_vel_idxs = offline_data['actuator_vel_idxs']
        self.state_posn_idxs = offline_data['state_posn_idxs']
        self.state_vel_idxs = offline_data['state_vel_idxs']
        self.pred_vel_idxs = offline_data['pred_vel_idxs']
        self.next_ob_idxs = offline_data['next_ob_idxs']
        self._labels_are_velocities = offline_data['_labels_are_velocities']
        self.idx_list = offline_data['index_list'] # indices corresponding to the target
        self.track_coefficients = np.array([1.0 for _ in range(len(self.idx_list))])
        coe = sum(self.track_coefficients)
        self.track_coefficients = self.track_coefficients / coe

        self.action_positions_lower_bounds = offline_data['action_positions_lower_bounds']
        self.action_positions_upper_bounds = offline_data['action_positions_upper_bounds']
        self.action_velocity_lower_bounds = offline_data['action_velocity_lower_bounds']
        self.action_velocity_upper_bounds = offline_data['action_velocity_upper_bounds']
        self.states_positions_lower_bounds = offline_data['states_positions_lower_bounds']
        self.states_positions_upper_bounds = offline_data['states_positions_upper_bounds']
        self.states_velocity_lower_bounds = offline_data['states_velocity_lower_bounds']
        self.states_velocity_upper_bounds = offline_data['states_velocity_upper_bounds']

        # get names of the tracking quantities and actuators in control
        self.tracking_target_names = offline_data['tracking_target_names']
        self.action_names = offline_data['action_names']
        self.track_signals = track_signals
        self.targets_in_obs = targets_in_obs
        self.add_tm_probs_to_obs = add_tm_probs_to_obs
        self.actuator_bounders = actuator_bounder
        self.reward_function = reward_function

        self.obs_noise = 0.
        self.k_step_targets_in_obs = k_step_targets_in_obs
        self.discrete_k_target_idx_in_obs = discrete_k_target_idx_in_obs
        if computed_obs_in_use is None:
            self.computed_observations = []
        else:
            self.computed_observations = computed_obs_in_use
            print(f'Using {len(self.computed_observations)} computed observations.')
        
        with open(raw_data_dir + '/info.pkl', 'rb') as file:
          self.info = pickle.load(file)

    def get_rl_state(self, obs, state, actuator, next_actuator, targets, batch_idx, shot_id=None, tm_probs=None):
        """Form observation from the underlying state and actuator.

        Args:
            obs: The previous observations with state
                (num_obs, self.cur_time + 1, obs dim)
            states: The states seen up to this point with shape
                (num_states, self.cur_time + 2, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, self.cur_time + 2, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, self.cur_time + 2, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.

        Returns:
            The observation from the state and actuator.
        """
        is_np = True if type(state) == np.ndarray else False

        if np.isscalar(batch_idx):
            batch_idx = np.array([batch_idx for _ in range(state.shape[0])])
        else:
            batch_idx = np.array(batch_idx)

        if not is_np:
            targets = torch.FloatTensor(targets).to(state.device)
        
        if len(self.next_ob_idxs) == 0:
            new_obs = np.hstack([state[-1, self.state_idxs],
                                 actuator[-1, self.action_idxs]])
        else:
            new_obs = np.hstack([state[-1, self.state_idxs],
                                 actuator[-1,  self.actuator_ob_idxs],
                                 next_actuator[-1, self.next_ob_idxs]])
        if len(self.computed_observations):
            posns, vels = [], []
            additional_obs = {}
            for co in self.computed_observations:
                # Pterm is computed for all timestep targets in the observation
                # Iterm is computed for current timestep target only
                # Dterm is computed for current timestep target only
                # this filtering of target will happen in below function calls for I and D
                posn, vel, ad_obs = co.compute_observation(obs, state, actuator,
                                                           next_actuator,
                                                           targets,
                                                           self.info, additional_obs)
                for k, v in ad_obs.items():
                    additional_obs[k] = v
                if self.discrete_k_target_idx_in_obs is not None and 'pterm' in list(ad_obs.keys())[0]:
                    posn = posn[:, self.discrete_k_target_idx_in_obs]
                posns.append(posn)
            if vel is not None:
                vels.append(vel)
            new_obs = np.hstack([np.atleast_2d(new_obs)] + posns + vels)
        if targets is not None and self.targets_in_obs:
            # new_obs = np.hstack([new_obs, targets[:, -1, self.state_ob_idxs]])
            # target_idx_to_use = [self.info['state_space'].index(o) for o in self.track_signals]
            target_idx_to_use = self.idx_list
            if self.discrete_k_target_idx_in_obs is not None:
                idx_to_use = np.array([k + state.shape[0] - 1 for k in self.discrete_k_target_idx_in_obs])
                idx_to_use2 = np.array(target_idx_to_use)
                target_subset = targets[idx_to_use[:, None], idx_to_use2]
                new_obs = np.hstack([new_obs, np.atleast_2d(target_subset.reshape(-1))])
            else:
                new_obs = np.hstack([new_obs, np.atleast_2d(targets[(state.shape[0]-1):, target_idx_to_use].reshape( -1))])
        if self.add_tm_probs_to_obs and tm_probs is not None:
            new_obs = np.hstack([new_obs, tm_probs[:, -1]])
        new_obs = torch.FloatTensor(new_obs).to(self.device) 
        return new_obs

    def get_step_action(self, action):
        """
        The action from the rl model is from [-1, 1], we need to transform it back to the original range before inputing it to the dynamics model.
        """
        act_num = action.shape[0]
        if type(action) == np.ndarray:
            return action * np.repeat(self.np_range, repeats=act_num, axis=0) + np.repeat(self.np_mid, repeats=act_num, axis=0)
        
        return action * self.range.repeat(act_num, 1) + self.mid.repeat(act_num, 1)

    def get_plot_quantities(self, shot_id, time_step, state, action):
        """
        According to the shot id and time step, query the quantity target, real quantity, achieved quantity, real actuators, and actuators adopted by the controller.
        """
        target_quan = self.eval_tracking_targets[shot_id][time_step]
        real_quan = self.eval_traj_states[shot_id][time_step][self.state_idxs][self.idx_list]
        cur_quan = state[0, self.idx_list].cpu().numpy() # Note that this is correct only because state is the first component of the rl_state. 
        real_act = self.eval_traj_actions[shot_id][time_step][self.action_idxs]
        cur_act = action[0]

        return target_quan, real_quan, cur_quan, real_act, cur_act
    
    def get_plot_names(self):
        """
        Return the list of tracking targets and controllable actuators.
        """
        return self.tracking_target_names, self.action_names
    

    def add_obs_noise(
            self,
            obs: np.ndarray
    ) -> np.ndarray:
        """Add noise to observation based on obs_noise parameter"""
        noise = np.random.normal(0, self.obs_noise, size=obs.shape)
        return obs + noise

    def unscale_actions(self, actions: np.ndarray) -> np.ndarray:
        """Transform actions from [-1, 1] back into original amount.
        Args:
            actions: The actions in [-1, 1].
        Returns: The unscaled actions.
        """
        lows  = np.atleast_2d(self.action_velocity_lower_bounds)
        highs = np.atleast_2d(self.action_velocity_upper_bounds)
        lows[:, self.posn_act_idxs] =\
            np.atleast_2d(self.action_positions_lower_bounds)[:, self.posn_act_idxs]
        highs[:, self.posn_act_idxs] =\
            np.atleast_2d(self.action_positions_upper_bounds)[:, self.posn_act_idxs]
        unscaled = (actions + 1) / 2
        return unscaled * (highs - lows) + lows

    def _update_actuators(
                self,
                actuators: np.ndarray,
                next_actuators: np.ndarray,
                act: np.ndarray,
                time_idx: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Update the actuator information to include the policy's actions.

            Args:
                actuators: The history of actuators as shape
                    (horizon + 1, actuator_dim)
                next_actuators: The history of next actutor change to be applied.
                    Has shape (horizon, num actuator signals).
                act: The actions the policy give at this time step with shape
                    (num_unrolls, action dimension)
                time_idx: The index of the time step this action was made.

            Returns:
                actuators but also altered in place.
            """
            # if self.copy_pinj_to_tinj:
            #     act = act.squeeze()
            #     act[self.info['actuator_space'].index('tinj')] = act[self.info['actuator_space'].index('pinj')]
            
            # Scale appropriately and conver all actions to deltas
            deltas = self.unscale_actions(act)
            deltas[..., self.posn_act_idxs] -=\
                 np.atleast_2d(actuators[time_idx, self.actuator_act_idxs[self.posn_act_idxs]])
            # Set the next actuators and bound them.
            next_actuators[time_idx, self.nxts_act_idxs] = deltas
            next_actuators[time_idx, self.nxts_act_idxs] = np.clip(
                    next_actuators[time_idx, self.nxts_act_idxs],
                    np.maximum(self.action_velocity_lower_bounds,
                            self.action_positions_lower_bounds
                            - actuators[time_idx, self.actuator_act_idxs]),
                    np.minimum(self.action_velocity_upper_bounds,
                            self.action_positions_upper_bounds
                            - actuators[time_idx, self.actuator_act_idxs])
            )
            self.actuator_bounders.bound_actuators(actuators, next_actuators,
                                                time_idx, self.info)
            # Propagate these updates to the actuators.
            if len(self.actuator_vel_idxs):
                actuators[time_idx + 1, self.actuator_vel_idxs] = \
                    next_actuators[time_idx]
            actuators[time_idx + 1, self.actuator_posn_idxs] = \
                actuators[time_idx, self.actuator_posn_idxs] \
                + next_actuators[time_idx]
            
            # if self.zero_tinj:
            #     actuators[:, time_idx + 1, self.info['actuator_space'].index('tinj')] = 0
            #     next_actuators[:, time_idx, self.info['actuator_space'].index('tinj')] = 0

            return actuators, next_actuators

    def _update_states(
            self,
            states: np.ndarray,
            preds: np.ndarray,
            time_idx: int,
        ) -> np.ndarray:
            """Update the states to reflect the model predictions.

            Args:
                states: The history of all the states so far and to come.
                    Has shape (num_unrolls, horizon + 1, state_dim)
                preds: The predictions made by the model should have shape
                    (num_unrolls, number of state signals). In most cases
                    last dimension will be half of states because states
                    will often have position + velocity.
                time_idx: The current time index.
            """
            preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
            if len(preds.shape) == 1:
                preds = preds[..., np.newaxis]
            # If the predictions are absolute states, make into the form of velocities.
            if not self._labels_are_velocities:
                preds = preds - states[time_idx, self.state_posn_idxs]
            # Optionally make bounds on the predictions..
            if self.states_positions_lower_bounds is not None :
                # Form the minimum bound.
                min_bound = self.states_velocity_lower_bounds.reshape(1, -1)
                min_bound = np.repeat(min_bound, 1, axis=0)
                posn_min_bd = self.states_positions_lower_bounds.reshape(1, -1)
                posn_min_bd = np.repeat(posn_min_bd, 1, axis=0)
                posn_diffs = posn_min_bd - states[time_idx, self.state_posn_idxs]
                # We only count if the signal is above the low bound. However, if this
                # is the case don't allow the signal to drop anymore.
                invalid_idxs = np.argwhere(posn_diffs >= 0)
                posn_diffs[invalid_idxs[:, 0], invalid_idxs[:, -1]] = 0
                min_bound = np.maximum(min_bound, posn_diffs)
                # Form the maximum bound.
                max_bound = self.states_velocity_upper_bounds.reshape(1, -1)
                max_bound = np.repeat(max_bound, 1, axis=0)
                posn_max_bd = self.states_positions_upper_bounds.reshape(1, -1)
                posn_max_bd = np.repeat(posn_max_bd, 1, axis=0)
                posn_diffs = posn_max_bd - np.atleast_2d(states[time_idx, self.state_posn_idxs])
                # We only count if the signal is below the high bound. We don't allow the
                # signal to grow anymore if this is the case however.
                invalid_idxs = np.argwhere(posn_diffs <= 0)
                posn_diffs[invalid_idxs[:, 0], invalid_idxs[:, -1]] = 0
                max_bound = np.minimum(max_bound, posn_diffs)
                preds = np.clip(preds, min_bound, max_bound)
            new_vels = preds[:, self.pred_vel_idxs]
            new_posns = np.atleast_2d(states[time_idx, self.state_posn_idxs]) + preds
            if len(self.state_vel_idxs):
                states[time_idx + 1, self.state_vel_idxs] = new_vels
            states[time_idx + 1, self.state_posn_idxs] = new_posns
            return states
    
class FusionEnv(NFBaseEnv):  # env for evaluation
    def __init__(self, model_dir, sa_processor, general_data, tracking_data, ref_shot_id, device):
        super().__init__(model_dir, sa_processor, general_data, tracking_data[ref_shot_id], ref_shot_id, device)
        # Initialize variables specific to FusionEnv
        self.ref_shot_id = None
        self.tracking_states, self.tracking_pre_actions, self.tracking_actions = None, None, None
        self.eval_shot_list = list(tracking_data.keys())
        self.tracking_data = tracking_data
        self.add_tm_probs_to_obs = add_tm_probs_to_obs
        self.k_step_targets_in_obs = k_step_targets_in_obs
        self.targets_in_obs = targets_in_obs
        self.reward_delay = 0  # no delay in reward
        self.reward_function = reward_function
        
        with open(raw_data_dir + '/info.pkl', 'rb') as file:
          self.info = pickle.load(file)
        self.idx_list = general_data['index_list']
    def get_eval_shot_list(self):
        """
        Return the list of shots for evaluation.
        """
        return self.eval_shot_list

    def reset(self, shot_id=None):
        """
        Reset the environment for a new evaluation episode.
        """
        # randomly sample a shot for evaluation
        if shot_id is None:
            self.ref_shot_id = random.choice(self.eval_shot_list)
        else:
            self.ref_shot_id = shot_id

        self.tracking_states = self.tracking_data[self.ref_shot_id]['tracking_states'][1:].copy()
        self.tracking_pre_actions = self.tracking_data[self.ref_shot_id]['tracking_pre_actions'][1:].copy()
        self.tracking_actions = self.tracking_data[self.ref_shot_id]['tracking_actions'][1:].copy()
        self.tracking_next_actuator = self.tracking_data[self.ref_shot_id]['tracking_actions'] -  self.tracking_data[self.ref_shot_id]['tracking_pre_actions']
        #self.cur_shot_time_limit = self.tracking_states.shape[0]
        self.cur_shot_time_limit =  150

        # randomly sample an initial time step
        # self.cur_time = random.randint(0, 9) # TODO
        self.cur_time = 0
        # self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        # self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)
        # self.cur_action = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).to(self.device)
        self.cur_state = np.expand_dims(self.tracking_states[self.cur_time], axis=0)       # shape: (1, state_dim)
        self.pre_action = np.expand_dims(self.tracking_pre_actions[self.cur_time], axis=0) # shape: (1, action_dim)
        self.cur_action = np.expand_dims(self.tracking_actions[self.cur_time], axis=0)    # shape: (1, action_dim)

        self.next_actuator = np.expand_dims(self.tracking_next_actuator[self.cur_time], axis=0)

        if shot_id is None:
            self.targets = self.sa_processor.training_tracking_targets
        else:
            self.targets = self.sa_processor.eval_tracking_targets[self.ref_shot_id]

        # reset the model
        for memb in self.all_models:
            memb.reset()
        new_state = self.sa_processor.get_rl_state(None, self.cur_state, self.pre_action, self.next_actuator,
                                                   self.targets[:self.k_step_targets_in_obs] if self.targets_in_obs else None, 
                                                   self.cur_time, 
                                                   shot_id=self.ref_shot_id, tm_probs=np.zeros(shape = (1, 1)) if self.add_tm_probs_to_obs else None)
        
        self.reward = np.zeros((self.tracking_states.shape[0])) 
        self.tm_probs = np.zeros((self.tracking_states.shape[0]))
        self.targets = np.pad(self.targets, ((0, self.k_step_targets_in_obs), (0, 0)), 'edge')
        return new_state
    
    def step(self, cur_action, obs):
         # prepare the input for the dymamics model
        batch_size = cur_action.shape[0] # step with a batch of actions
       
        self.tracking_pre_actions, self.tracking_next_actuator[1:, :] = self.sa_processor._update_actuators(
                self.tracking_pre_actions,
                # Chop off the fist index of the next actuators since that happened
                # before the start.
                self.tracking_next_actuator[1:, :],
                np.atleast_2d(cur_action),
                self.cur_time,
            )

        net_input = np.hstack([np.expand_dims(self.tracking_states[self.cur_time], axis=0), 
                               np.expand_dims(self.tracking_pre_actions[self.cur_time], axis=0), 
                               np.expand_dims(self.tracking_next_actuator[self.cur_time+1], axis=0)])
        net_input = torch.FloatTensor(net_input).to(self.device) 
      
        # get the ensemble output
        ensemble_preds = 0.
        means, stds = [], []
        with torch.no_grad():
            for memb in self.all_models:
                net_input_n = memb.normalizer.normalize(net_input, 0)
                net_output_n, info = memb.single_sample_output_from_torch(net_input_n) # torch.Size([1, 27])
                net_output = memb.normalizer.unnormalize(net_output_n, 1)
                ensemble_preds += net_output
                # collect the means and stds of predictions
                mean = memb.normalizer.unnormalize(info["mean_predictions"], 1)
                std = getattr(memb.normalizer, f'{1}_scaling') * info["std_predictions"] # danger
                means.append(mean)
                stds.append(std)

        ensemble_preds = ensemble_preds / float(len(self.all_models)) # delta of the state, which is the mean of the ensemble outputs
        means, stds = torch.stack(means).cpu().numpy(), torch.stack(stds).cpu().numpy()

        # proceed to the next time step
        # self.cur_state = self.cur_state + ensemble_preds # the next state, TODO: use the true value for the unselected dimensions
        self.tracking_states = self.sa_processor._update_states(self.tracking_states, ensemble_preds, self.cur_time)
        # return_state = self.cur_state[:, self.state_idxs]
        
        self.curr_target =  np.expand_dims(self.targets[self.cur_time], axis=0)             

        if getattr(self.reward_function, "get_tm_probability", None) is not None:
            if getattr(self.reward_function, "provide_full_action", False):
                full_actuator = self.tracking_pre_actions[self.cur_time] + self.tracking_next_actuator[ self.cur_time+1]
                self.tm_probs[:,self.cur_time] = self.reward_function.get_tm_probability(
                    self.tracking_states[self.cur_time + 1], full_actuator, self.info
                )
            else:
                self.tm_probs[self.cur_time] = self.reward_function.get_tm_probability(
                    self.tracking_pre_actions[:, self.cur_time + 1], self.tracking_next_actuator[:, self.cur_time+1], self.info
                )
        else:
            self.tm_probs[self.cur_time] = np.nan
        
         # reward = self.get_reward(return_state.cpu().numpy(), self.cur_time, shot_id=self.ref_shot_id) # next state and current time step
        if self.cur_time >= self.reward_delay:
            if getattr(self.reward_function, "provide_full_action", False):
                full_actuator = self.tracking_pre_actions[self.cur_time] + self.tracking_next_actuator[self.cur_time+1]
                self.reward[self.cur_time] = self.reward_function.get_reward(
                    np.atleast_2d(self.tracking_states[self.cur_time + 1]), np.atleast_2d(full_actuator), self.info, self.curr_target)
            else:
                self.reward[self.cur_time] = self.reward_function.get_reward(
                    np.atleast_2d(self.tracking_states[self.cur_time + 1]), np.atleast_2d(self.tracking_next_actuator[self.cur_time + 1]), self.info, self.idx_list, self.curr_target)
        else:
            self.reward[self.cur_time] = 0
        
        
        self.cur_time += 1
        self.pre_action = cur_action.copy()

        # new to this env
        done = self.is_done(self.cur_time)
        done = done | (self.cur_time >= self.cur_shot_time_limit)

        if batch_size > 1:
            done = np.array([done for _ in range(batch_size)])
        else:
            self.reward_return = self.reward[self.cur_time-1]

        return self.sa_processor.get_rl_state(obs[:self.cur_time], 
                                              self.tracking_states[:self.cur_time+1], 
                                              self.tracking_pre_actions[:self.cur_time+1], 
                                              self.tracking_next_actuator[:self.cur_time+1], 
                                              self.targets[:self.cur_time + 1 + self.k_step_targets_in_obs-1] if self.targets_in_obs else None,
                                              self.cur_time, 
                                              shot_id=self.ref_shot_id, tm_probs=self.tm_probs[:self.cur_time, np.newaxis] if self.add_tm_probs_to_obs else None),self.tracking_pre_actions.copy() , self.reward_return, done, {'means': means, 'stds': stds, "time_step": self.cur_time}


