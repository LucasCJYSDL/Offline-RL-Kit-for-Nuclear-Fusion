
import abc
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import math

from  envs.utils.profile_util import reconstruct_profile_from_state


import re


class RewardFunction(metaclass=abc.ABCMeta):

    def reset(self) -> None:
        """Reset the reward function."""
        pass

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return 0

    @abc.abstractmethod
    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """


class NoGoal(RewardFunction):
    """Returns 0 reward for everything."""

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        return 0


class TrackingReward(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        if track_coefficients is None:
            self.track_coefficients = np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.track_coefficients = np.array(track_coefficients)
        self.track_coefficients = self.track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.square_costs = square_costs

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals)

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            index_list: list,
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[1] == len(self.track_idxs), 'Invalid number of targets.'
        if self.square_costs:
            return -1 * np.dot(np.square(nxts[:, self.track_idxs] - targets),
                               self.track_coefficients).flatten()
        else:
            return -1 * np.dot(np.abs(nxts[:, self.track_idxs] - targets),
                               self.track_coefficients).flatten()


class TrackingIndepActPenalties(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            act_signals: Sequence[str],
            act_penalties: Sequence[float],
            track_coefficients: Optional[Sequence[float]] = None,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            act_penalty: The penalty for the size of change in the actuator.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        assert len(act_signals) == len(act_penalties)
        self.track_signals = track_signals
        self.act_signals = act_signals
        self.act_penalties = np.array([ap for ap in act_penalties])
        if track_coefficients is None:
            self.track_coefficients = np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.track_coefficients = np.array(track_coefficients)
        self.track_coefficients = self.track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.act_idxs = None

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals)

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        if self.act_idxs is None:
            self.act_idxs = []
            for sig in self.act_signals:
                self.act_idxs.append(info['next_actuator_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[-1] == len(self.track_idxs), 'Invalid number of targets.'
        rewards = -1 * np.dot(np.square(nxts[..., self.track_idxs] - targets),
                              self.track_coefficients).flatten()
        rewards -= np.dot(np.abs(actions[..., self.act_idxs]),
                          self.act_penalties).flatten()
        return rewards


class VarTrackingIndepActPenalties(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            act_signals: Sequence[str],
            track_coefficients: Optional[Sequence[float]] = None,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            act_penalty: The penalty for the size of change in the actuator.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        self.act_signals = act_signals
        if track_coefficients is None:
            self.track_coefficients = np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.track_coefficients = np.array(track_coefficients)
        self.track_coefficients = self.track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.act_idxs = None

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals) + len(self.act_signals)

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        if self.act_idxs is None:
            self.act_idxs = []
            for sig in self.act_signals:
                self.act_idxs.append(info['next_actuator_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[1] == self.num_targets, 'Invalid number of targets.'
        rewards = -1 * np.dot(np.square(nxts[:, self.track_idxs]
                                        - targets[:, :len(self.track_idxs)]),
                              self.track_coefficients).flatten()
        rewards -= np.sum(np.abs(actions[:, self.act_idxs])
                          * targets[:, len(self.track_idxs):], axis=1)
        return rewards


class TrackingRewardWActPenalty(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            act_penalty: float,
            track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            act_penalty: The penalty for the size of change in the actuator.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        self.act_penalty = act_penalty
        if track_coefficients is None:
            self.track_coefficients = np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.track_coefficients = np.array(track_coefficients)
        self.track_coefficients = self.track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.square_costs = square_costs

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals)

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[1] == len(self.track_idxs), 'Invalid number of targets.'
        penalty = -self.act_penalty * np.linalg.norm(actions, axis=1).flatten()
        if self.square_costs:
            return -1 * np.dot(np.square(nxts[:, self.track_idxs] - targets),
                               self.track_coefficients).flatten() + penalty
        else:
            return -1 * np.dot(np.abs(nxts[:, self.track_idxs] - targets),
                               self.track_coefficients).flatten() + penalty


class AsymTrackingRewardWActPenalty(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            act_penalty: float,
            under_track_coefficients: Optional[Sequence[float]] = None,
            above_track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            act_penalty: The penalty for the size of change in the actuator.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        self.act_penalty = act_penalty
        if under_track_coefficients is None:
            self.under_track_coefficients =\
                np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(under_track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.under_track_coefficients = np.array(under_track_coefficients)
        if above_track_coefficients is None:
            self.above_track_coefficients =\
                np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(above_track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.above_track_coefficients = np.array(under_track_coefficients)
        self.under_track_coefficients = self.under_track_coefficients.reshape(-1, 1)
        self.above_track_coefficients = self.above_track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.square_costs = square_costs

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals)

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[1] == len(self.track_idxs), 'Invalid number of targets.'
        penalty = -self.act_penalty * np.linalg.norm(actions, axis=1).flatten()
        # coefs = (self.under_track_coefficients * (nxts[:, self.track_idxs] <= targets)
        #          + self.above_track_coefficients * (nxts[:, self.track_idxs] > targets))
        # if self.square_costs:
        #     return -1 * np.sum(np.square(nxts[:, self.track_idxs] - targets)
        #                        * coefs, axis=-1).flatten() + penalty
        # else:
        #     return -1 * np.sum(np.abs(nxts[:, self.track_idxs] - targets)
        #                        * coefs, axis=-1).flatten() + penalty

        # Rohit - changed to dot product to prevent broadcasting errors - above only works for a single datapoint not batch
        # coefs must be (2,100)
        # this is not dot product
        coefs_under = np.vstack([(nxts[:, self.track_idxs] <= targets)[:,i] * self.under_track_coefficients[i] for i in range(self.num_targets)])
        coefs_over = np.vstack([(nxts[:, self.track_idxs] > targets)[:,i] * self.above_track_coefficients[i] for i in range(self.num_targets)])
        coefs = coefs_under + coefs_over
        if self.square_costs:
            return -1 * np.sum(np.dot(np.square(nxts[:, self.track_idxs] - targets), coefs), axis=-1).flatten() + penalty
        else:
            return -1 * np.sum(np.dot((nxts[:, self.track_idxs] - targets), coefs), axis=-1).flatten() + penalty


class AsymTrackingRewardWVarActPenalty(RewardFunction):

    def __init__(
            self,
            track_signals: Sequence[str],
            under_track_coefficients: Optional[Sequence[float]] = None,
            above_track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to track.
            act_penalty: The penalty for the size of change in the actuator.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        if under_track_coefficients is None:
            self.under_track_coefficients =\
                np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(under_track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.under_track_coefficients = np.array(under_track_coefficients)
        if above_track_coefficients is None:
            self.above_track_coefficients =\
                np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(above_track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.above_track_coefficients = np.array(under_track_coefficients)
        self.under_track_coefficients = self.under_track_coefficients.reshape(-1, 1)
        self.above_track_coefficients = self.above_track_coefficients.reshape(-1, 1)
        self.track_idxs = None
        self.square_costs = square_costs

    @property
    def num_targets(self) -> int:
        """Number of targets for this reward."""
        return len(self.track_signals) + 1

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        assert targets is not None, 'Targets must be provided.'
        assert targets.shape[1] == len(self.track_idxs), 'Invalid number of targets.'
        penalty =  -1 * targets[:, -1] * np.linalg.norm(actions, axis=1).flatten()
        coefs = (self.under_track_coefficients * (nxts[:, self.track_idxs]
                 <= targets[:, :-1])
                 + self.above_track_coefficients * (nxts[:, self.track_idxs]
                                                    > targets[:, :-1]))
        if self.square_costs:
            return -1 * np.sum(np.square(nxts[:, self.track_idxs] - targets[:, :-1])
                               * coefs, axis=-1).flatten() + penalty
        else:
            return -1 * np.sum(np.abs(nxts[:, self.track_idxs] - targets[:, :-1])
                               * coefs, axis=-1).flatten() + penalty


class HighValueReward(RewardFunction):
    """Reward function for achieving a high values for signals"""

    def __init__(
            self,
            track_signals: Sequence[str],
            track_coefficients: Optional[Sequence[float]] = None,
    ):
        """Constructor

        Args:
            track_signals: The names of the signals to try to achieve high values.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.track_signals = track_signals
        if track_coefficients is None:
            self.track_coefficients = np.array([1 for _ in range(len(track_signals))])
        else:
            assert len(track_coefficients) == len(track_signals), \
                    ('Number of coefficients must match number of coefficients '
                     f'{len(track_coefficients)} does not equal {len(track_signals)}')
            self.track_coefficients = np.array(track_coefficients)
        self.track_coefficients = self.track_coefficients.reshape(-1, 1)
        self.track_idxs = None

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        return np.dot(nxts[:, self.track_idxs], self.track_coefficients).flatten()



class ProfileTrackingReward(RewardFunction):
    
    def __init__(
            self,
            profile_name: str,
            track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
            track_signals: Optional[Sequence[int]] = None,
            unnormalize: bool = False,
    ):
        """Constructor.

        Args:
            profile_name: The name of the profile to track.
            track_coefficients: How to weight the linear combination of the tracking
                rewards.
        """
        self.profile_name = profile_name
        self.track_coefficients = track_coefficients
        self.square_costs = square_costs
        self.profile_idxs = {}
        self.normalizers = {}
        self.track_idxs = None
        self.track_signals = track_signals
        self.unnormalize = unnormalize

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            index_list: list,
            targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the reward.

        Args:
            nxts: The next states.
            actions: The actions applied.
            info: The info dictionary.

        Returns:
            The rewards.
        """
        total_reward = 0
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))

        for pname in self.profile_name:
            profile = reconstruct_profile_from_state(profile_name=pname, states=nxts, info=info,target_index=None, unnormalize=self.unnormalize)
            # create an array called targets which has zeros filled out indices not track_idxs
            # full_targets = np.zeros((nxts.shape[0], nxts.shape[1]))
            # if targets is not None:
            #     full_targets[:, self.track_idxs] = targets
            # target_profile = self._reconstruct_profile(targets, info, pname, unnormalize=self.unnormalize)
            target_profile = reconstruct_profile_from_state(profile_name=pname, states=targets, info=info,target_index=index_list, unnormalize=self.unnormalize)
            if self.track_coefficients is None:
                self.track_coefficients = np.array([1 for _ in range(profile.shape[1])])
            if self.square_costs:
                total_reward = total_reward + -1 * np.dot(np.square(profile - target_profile), self.track_coefficients).flatten()
            else:
                total_reward = total_reward + -1 * np.dot(np.abs(profile - target_profile), self.track_coefficients).flatten()

        
        return total_reward


class ProfileTrackingWithBetanReward(RewardFunction):
    def __init__(
            self,
            profile_name: str,
            track_coefficients: Optional[Sequence[float]] = None,
            square_costs: bool = False,
            track_signals: Optional[Sequence[str]] = None,
            unnormalize: bool = False,
            betan_penalty_wt : float = 1.0,
            rewards_on_component: bool = False,
            integral_term: bool = False,
            integral_term_wt: float = 1.0,
    ):
        """
        Profile tracking reward with BetaN constraint.
        """
        self.profile_name = profile_name
        self.track_coefficients = track_coefficients
        self.square_costs = square_costs
        self.profile_idxs = {}
        self.normalizers = {}
        self.track_idxs = None
        self.track_signals = track_signals
        self.unnormalize = unnormalize
        self.betan_penalty_wt = betan_penalty_wt
        self.integral_term = integral_term
        self.rewards_on_component = rewards_on_component
        self.integral_term_wt = integral_term_wt

    def get_reward(
            self,
            nxts: np.ndarray,
            actions: np.ndarray,
            info: Dict[str, Any],
            targets: Optional[np.ndarray] = None,
            env_merge: bool = False,
            env_type: int = None,
            get_indv_rewards: bool = False,
            **kwargs
    ) -> np.ndarray:
        total_reward = 0
        if self.track_idxs is None:
            self.track_idxs = []
            for sig in self.track_signals:
                self.track_idxs.append(info['state_space'].index(sig))
        assert env_type in (0, 1) if env_merge else True, "env_type must be 0 (ZIPFIT) or 1 (CAKENN) when env_merge is True"
        if not env_merge or (env_merge and env_type == 0):
            if self.rewards_on_component:
                for idx in self.track_idxs:
                    if self.square_costs:
                        costs = np.square(nxts[:, idx] - targets[:, idx])
                    else:
                        costs = np.abs(nxts[:, idx] - targets[:, idx])
                    total_reward = total_reward + -1 * costs
            else:
                for pname in self.profile_name:
                    profile = reconstruct_profile_from_state(profile_name=pname, states=nxts, info=info, unnormalize=self.unnormalize)
                    target_profile = reconstruct_profile_from_state(profile_name=pname, states=targets, info=info, unnormalize=self.unnormalize)
                    if self.track_coefficients is None:
                        self.track_coefficients = np.array([1 for _ in range(profile.shape[1])])
                    if self.square_costs:
                        total_reward = total_reward + -1 * np.dot(np.square(profile - target_profile), self.track_coefficients).flatten()
                    else:
                        total_reward = total_reward + -1 * np.dot(np.abs(profile - target_profile), self.track_coefficients).flatten()
            beta_idx = info['state_space'].index('betan_EFIT01')
        else:  # env_type == 1 and CAKENN rewards
            if self.rewards_on_component:
                for idx in self.track_idxs:
                    if self.square_costs:
                        costs = np.square(nxts[:, idx] - targets[:, idx])
                    else:
                        costs = np.abs(nxts[:, idx] - targets[:, idx])
                    total_reward = total_reward + -1 * costs
            else:
                for pname in self.profile_name:
                    profile = reconstruct_obs_profile_from_obs(profile_name=pname, states=nxts, info=info, unnormalize=self.unnormalize, state_names=info['state_space'])
                    target_profile = reconstruct_obs_profile_from_obs(profile_name=pname, states=targets, info=info, unnormalize=self.unnormalize, state_names=info['state_space'])
                    if self.track_coefficients is None:
                        self.track_coefficients = np.array([1 for _ in range(profile.shape[1])])
                    if self.square_costs:
                        total_reward = total_reward + -1 * np.dot(np.square(profile - target_profile), self.track_coefficients).flatten()
                    else:
                        total_reward = total_reward + -1 * np.dot(np.abs(profile - target_profile), self.track_coefficients).flatten()
            beta_idx = info['state_space'].index('betan_EFITRT')
        beta_error = np.square(targets[:, beta_idx] - nxts[:, beta_idx])
        if self.integral_term:
            additional_obs = kwargs.get('additional_obs', {})
            current_cum_error = 0
            for key in additional_obs.keys():
                if 'component1_iterm' in key:
                    if self.square_costs:
                        current_cum_error += np.square(additional_obs[key][:,0])
                    else:
                        current_cum_error += np.abs(additional_obs[key][:,0])
            
        total = total_reward - self.betan_penalty_wt * beta_error - self.integral_term_wt*current_cum_error
        
        if not get_indv_rewards:
            return total
        
        # Return dictionary with 'total' key for compatibility with fusion_env.py
        return {
            'total': total,
            'profile_error': total_reward,
            'beta_error': -self.betan_penalty_wt * beta_error,
            'integral_error': -self.integral_term_wt*current_cum_error,
        }
