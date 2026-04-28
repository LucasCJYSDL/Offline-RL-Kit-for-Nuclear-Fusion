import torch
import random
import numpy as np
import pickle
from envs.base_env import NFBaseEnv
from rl_preparation.process_raw_data import raw_data_dir
from rl_preparation import state_actuator_spaces as sas

class SA_processor: # used for both training and evaluation
    def __init__(self, offline_data, tracking_data, device):
        # for normalization or denormalization of the actuators
        bounds = (offline_data['action_lower_bounds'], offline_data['action_upper_bounds'])
        self.range = torch.FloatTensor((bounds[1] - bounds[0]) / 2.0).unsqueeze(0).to(device) # actuator bounds
        self.mid = torch.FloatTensor((bounds[1] + bounds[0]) / 2.0).unsqueeze(0).to(device)

        self.np_range = ((bounds[1] - bounds[0]) / 2.0)[np.newaxis, :]
        self.np_mid = ((bounds[1] + bounds[0]) / 2.0)[np.newaxis, :]

        self.k_step_targets_in_obs = sas.k_step_targets_in_obs
        self.discrete_k_target_idx_in_obs = sas.discrete_k_target_idx_in_obs
        if sas.computed_obs_in_use is None or len(sas.computed_obs_in_use) == 0:
            self.computed_observations = []
        else:
            self.computed_observations = list(sas.computed_obs_in_use)
            print(f'Using {len(self.computed_observations)} computed observations.')
        self.track_signals = sas.track_signals
        self.targets_in_obs = sas.targets_in_obs
        self.add_tm_probs_to_obs = sas.add_tm_probs_to_obs
        self.actuator_bounders = sas.actuator_bounder
        self.reward_function = sas.reward_function
        self.target_lows = sas.target_lows
        self.target_highs = sas.target_highs

        with open(raw_data_dir + '/info.pkl', 'rb') as file:
          self.info = pickle.load(file)

        # store the tracking taregts for both the training and evaluation data
        self.training_data_size = offline_data['tracking_ref'].shape[0] #151
        self.training_tracking_targets = np.array([offline_data['tracking_ref'][-1] for _ in range(self.training_data_size + self.k_step_targets_in_obs)])
        self.training_tracking_targets[:self.training_data_size] = offline_data['tracking_ref']
        terminals = offline_data['terminals'].reshape(-1).astype(bool)
        self.training_episode_last_idx = np.zeros(self.training_data_size, dtype=np.int32)
        episode_start = 0
        for terminal_idx in np.flatnonzero(terminals):
            self.training_episode_last_idx[episode_start:terminal_idx + 1] = terminal_idx
            episode_start = terminal_idx + 1
        if episode_start < self.training_data_size:
            self.training_episode_last_idx[episode_start:] = self.training_data_size - 1
        
        

        self.eval_tracking_targets = {}
        self.eval_traj_states = {}
        self.eval_traj_actions = {}
        self.times = {}
        for key in tracking_data: # to avoid indexing issue, we use a padding trick here
            tracking_data_size = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_tracking_targets[key] = np.array([tracking_data[key]['tracking_ref'][-1] for _ in range(tracking_data_size+self.k_step_targets_in_obs)])
            self.eval_tracking_targets[key][:tracking_data_size] = tracking_data[key]['tracking_ref']
            # self.eval_data_sizes[key] = tracking_data[key]['tracking_ref'].shape[0]
            self.eval_traj_states[key] = tracking_data[key]["tracking_states"]
            self.eval_traj_actions[key] = tracking_data[key]["tracking_actions"]
            self.times[key] =  tracking_data[key]["time"]
        
        # store idxs of the tracking targets to query them from the state
        # there could be multiple dimensions of the tracking target, so we may assign different coefficients to different dimensions.
        self.state_idxs = offline_data['state_idxs'] # selected state dimensions 
        self.action_idxs = offline_data["action_idxs"]
        self.idx_list = offline_data['index_list'] # indices corresponding to the target
        self.track_coefficients = np.array([1.0 for _ in range(len(self.idx_list))])
        coe = sum(self.track_coefficients)
        self.track_coefficients = self.track_coefficients / coe

        # get names of the tracking quantities and actuators in control
        self.tracking_target_names = offline_data['tracking_target_names']
        self.action_names = offline_data['action_names']

    def build_rl_state_from_next_state(
        self,
        next_state,      # (B, state_dim_selected)
        batch_idx,
        shot_id=None
    ):
        return self.get_rl_state(
            next_state,
            batch_idx,
            shot_id=shot_id
        )
    
    def _get_concatenated_targets(self, targets_source, batch_idx, offsets=None):
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        max_valid_idx = min(self.training_data_size, targets_source.shape[0]) - 1
        batch_idx = np.clip(batch_idx, 0, max_valid_idx)
        episode_last_idx = np.minimum(self.training_episode_last_idx[batch_idx], max_valid_idx)

        if offsets is None:
            target_indices = batch_idx
            return targets_source[target_indices]

        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
        target_indices = np.minimum(batch_idx[np.newaxis, :] + offsets[:, np.newaxis], episode_last_idx[np.newaxis, :])
        return targets_source[target_indices]

    def _gather_targets(self, targets_source, batch_idx, offsets):
        batch_idx = np.asarray(batch_idx, dtype=np.int64).reshape(-1)
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
        max_valid_idx = targets_source.shape[0] - 1
        target_indices = np.clip(batch_idx[np.newaxis, :] + offsets[:, np.newaxis], 0, max_valid_idx)
        return targets_source[target_indices]
        
    
    def get_rl_state(self, state, batch_idx, shot_id=None, target=None):
        """
        Design of the state space (used for the rl policy).
        """
        is_np = True if type(state) == np.ndarray else False

        if np.isscalar(batch_idx):
            batch_idx = np.array([batch_idx for _ in range(state.shape[0])])
        else:
            batch_idx = np.array(batch_idx)
        
        if target is not None:
            targets_source = target
        else:
            if shot_id is None:
                targets_source = self.training_tracking_targets
            else:
                targets_source = self.eval_tracking_targets[shot_id]
        
        if self.discrete_k_target_idx_in_obs is not None:
            offsets = np.array(self.discrete_k_target_idx_in_obs)
        else:
            offsets = np.arange(self.k_step_targets_in_obs)

        # if self.discrete_k_target_idx_in_obs is not None:

        #     idx_to_use = np.array([batch_idx + k for k in self.discrete_k_target_idx_in_obs])
        #     targets = targets_source[idx_to_use[:], :]
                
        # else:
        #     idx_to_use = np.array([batch_idx + k for k in range(self.k_step_targets_in_obs)])
        #     targets = targets_source[idx_to_use[:], :]
        targets = self._get_concatenated_targets(targets_source, batch_idx, offsets)   


        if not is_np:
            targets = torch.FloatTensor(targets).to(state.device)
        difference = targets - state[:, self.idx_list]

        targets_list = [targets[i] for i in range(targets.shape[0])]
        difference_list = [difference[i] for i in range(difference.shape[0])]

        if is_np:
            rl_state = np.hstack([state] + targets_list + difference_list) # the rl state contains the current state, tracking targets, and distance to the tracking targets
        else:
            rl_state = torch.cat(
                [state] + targets_list + difference_list,
                dim=-1
            )
        

        return rl_state
    
    def get_step_action(self, action):
        """
        The action from the rl model is from [-1, 1], we need to transform it back to the original range before inputing it to the dynamics model.
        """
        act_num = action.shape[0]
        if type(action) == np.ndarray:
            return action * np.repeat(self.np_range, repeats=act_num, axis=0) + np.repeat(self.np_mid, repeats=act_num, axis=0)
        
        return action * self.range.repeat(act_num, 1) + self.mid.repeat(act_num, 1)
    
    def normalize_action(self, action):
        """
        Normalize actions (based on the actuator bounds) in the offline dataset for rl training.
        """
        act_num = action.shape[0]
        return (action - self.mid.repeat(act_num, 1).cpu().numpy()) / (self.range.repeat(act_num, 1).cpu().numpy()+1e-6)
    
    def get_reward(self, next_state, time_step, shot_id=None): 
        """
        Design of the reward function.
        The reward function is defined based on the distance between the actual next state and the target next state (i.e., the target specified at the current time step).
        """
        if np.isscalar(time_step): # for evaluation
            assert shot_id is not None
            time_step = np.array([time_step])
            #time_step = np.clip(time_step, 0, len(self.eval_tracking_targets[shot_id]) - 1)
            targets = self.eval_tracking_targets[shot_id][time_step]
        else:
            #boundary checkout
            targets = self.training_tracking_targets[time_step] # for training
        # this design is flexible - we are using "-mse" as the reaward
        # reward = reward_function.get_reward(next_state[:, self.idx_list],None,self.info,self.idx_list,targets)
        return -1.0 * (np.square(next_state[:, self.idx_list] - targets) * self.track_coefficients[np.newaxis, :]).sum(axis=1) 
        # return reward
    def get_reward_new(self, next_state, idx, shot_id=None): 
        """
        Design of the reward function.
        The reward function is defined based on the distance between the actual next state and the target next state (i.e., the target specified at the current time step).
        """
        if np.isscalar(idx): # for evaluation
            assert shot_id is not None
            idx = np.array([idx])
            #time_step = np.clip(time_step, 0, len(self.eval_tracking_targets[shot_id]) - 1)
            targets = self.eval_tracking_targets[shot_id][idx]
        else:
            #boundary checkout
            targets = self.training_tracking_targets[idx] # for training
        reward = self.reward_function.get_reward(next_state[:, self.idx_list], None, self.info, self.idx_list, targets)
        return reward
        # this design is flexible - we are using "-mse" as the reaward
        #return -1.0 * (np.square(next_state[:, self.idx_list] - targets) * self.track_coefficients[np.newaxis, :]).sum(axis=1) 

    def get_plot_quantities(self, shot_id, time_step, state, action):
        """
        According to the shot id and time step, query the quantity target, real quantity, achieved quantity, real actuators, and actuators adopted by the controller.
        """
        target_quan = self.eval_tracking_targets[shot_id][time_step]
        real_quan = self.eval_traj_states[shot_id][time_step][self.state_idxs][self.idx_list]
        cur_quan = state[0, self.idx_list].cpu().numpy() # Note that this is correct only because state is the first component of the rl_state. 
        real_act = self.eval_traj_actions[shot_id][time_step][self.action_idxs]
        cur_act = self.get_step_action(action)[0]

        return target_quan, real_quan, cur_quan, real_act, cur_act
    
    def get_plot_names(self):
        """
        Return the list of tracking targets and controllable actuators.
        """
        return self.tracking_target_names, self.action_names


class ProfileControlEnv(NFBaseEnv): # env for evaluation
    def __init__(self, model_dir, sa_processor, general_data, tracking_data, ref_shot_id, device):
        super().__init__(model_dir, sa_processor, general_data, tracking_data[ref_shot_id], ref_shot_id, device)
        # these variables are from the base env but we don't need them
        self.ref_shot_id = None
        self.tracking_states, self.tracking_pre_actions, self.tracking_actions = None, None, None
        self.eval_shot_list = list(tracking_data.keys())
        self.tracking_data = tracking_data
          
        with open(raw_data_dir + '/info.pkl', 'rb') as file:
          self.info = pickle.load(file)

    def get_eval_shot_list(self):
        """
        return the list of shots for evaluation
        """
        return self.eval_shot_list

    def reset(self, shot_id=None):
        # randomly sample a shot for evaluation
        if shot_id is None:
            self.ref_shot_id = random.choice(self.eval_shot_list)
        else:
            self.ref_shot_id = shot_id

        self.tracking_states, self.tracking_pre_actions, self.tracking_actions = self.tracking_data[self.ref_shot_id]['tracking_states'], \
                                                                                 self.tracking_data[self.ref_shot_id]['tracking_pre_actions'], \
                                                                                 self.tracking_data[self.ref_shot_id]['tracking_actions']
        self.cur_shot_time_limit = self.tracking_states.shape[0]
        
        # randomly sample an initial time step
        # self.cur_time = random.randint(0, 9) # TODO
        self.cur_time = 0
        self.cur_state = torch.FloatTensor(self.tracking_states[self.cur_time]).unsqueeze(0).to(self.device)
        self.pre_action = torch.FloatTensor(self.tracking_pre_actions[self.cur_time]).unsqueeze(0).to(self.device)

        # reset the model
        for memb in self.all_models:
            memb.reset()
        
        return_state = self.cur_state[:, self.state_idxs]
        
        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id=self.ref_shot_id)

    def step(self, cur_action):
        # prepare the input for the dymamics model
        cur_action = torch.Tensor(cur_action).to(self.device)
        batch_size = cur_action.shape[0] # step with a batch of actions  #？？？
        cur_action = self.sa_processor.get_step_action(cur_action)
        cur_action_pad = torch.FloatTensor(self.tracking_actions[self.cur_time]).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        cur_action_pad[:, self.action_idxs] = cur_action
        cur_action = cur_action_pad

        if self.cur_state.shape[0] < batch_size:
            self.cur_state = self.cur_state.repeat(batch_size, 1)
            self.pre_action = self.pre_action.repeat(batch_size, 1)
        net_input = torch.cat([self.cur_state, self.pre_action, cur_action-self.pre_action], dim=-1)

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
        self.cur_state = self.cur_state + ensemble_preds # the next state, TODO: use the true value for the unselected dimensions
        return_state = self.cur_state[:, self.state_idxs]
        reward = self.get_reward(return_state.cpu().numpy(), self.cur_time, shot_id=self.ref_shot_id) # next state and current time step
        self.cur_time += 1
        self.pre_action = cur_action.clone()

        # new to this env
        done = self.is_done(self.cur_time)
        # done = done | (self.cur_time >= self.cur_shot_time_limit)

        if batch_size > 1:
            done = np.array([done for _ in range(batch_size)])
        else:
            reward = reward[0]

        return self.sa_processor.get_rl_state(return_state, self.cur_time, shot_id=self.ref_shot_id), reward, done, {'means': means, 'stds': stds, "time_step": self.cur_time}
