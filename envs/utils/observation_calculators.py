"""
Helpers to form observations that are not clearly just in the state.

Author: Ian Char
Date: July 15, 2022
"""
import abc
from copy import deepcopy
from typing import Any, Dict, Tuple, Optional

import numpy as np


class ObservationCalculator(metaclass=abc.ABCMeta):

    def reset(self):
        """Reset the calculator if there is state."""
        pass

    @abc.abstractmethod
    def compute_observation(
            self,
            obs: Optional[np.ndarray],
            states: np.ndarray,
            actuators: np.ndarray,
            next_actuators: np.ndarray,
            targets: Optional[np.ndarray],
            info: Dict[str, Any],
            additional_obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Compute the observation.

        Args:
            obs: The previous observations with state
                (num_obs, h + 1, obs dim). This is None if there are no previous obs.
            states: The states seen up to this point with shape
                (num_states, h + 2, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.
            info: The data information dict.
            additional_obs: The additional observations made so far.

        Returns:
            The computed array of observations and the velocity for those observations
            if applicable. Each have shape (num_obs, 1). Also return dictionary of
            named outputs.
        """

    @abc.abstractproperty
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""


class PTerm(ObservationCalculator):
    """Calculate the P term, i.e. error"""

    def __init__(
        self,
        signal_name: str,
        target_idx: int,
        additional_obs_name: Optional[int] = None,
        invert_sign=False,
    ):
        """Constructor.

        Args:
            signal_name: The signal of the P, I, and D terms.
            target_idx: The index of the target corresponding to the signal name.
            obs_idx: the index in the observation to track. If this is provided we
                will look at this rather than the signal_name.
        """
        self.signal_name = signal_name
        self.target_idx = target_idx
        self.additional_obs_name = additional_obs_name
        self.signal_idx = None
        self.invert_sign = invert_sign

    def compute_observation(
            self,
            obs: Optional[np.ndarray],
            states: np.ndarray,
            actuators: np.ndarray,
            next_actuators: np.ndarray,
            targets: Optional[np.ndarray],
            info: Dict[str, Any],
            additional_obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Compute the observation.

        Args:
            obs: The previous observations with state
                (num_obs, obs dim). This is None if there are no previous obs.
            states: The states seen up to this point with shape
                (num_states, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.
            info: The data information dict.
            additional_obs: The additional observations made so far.

        Returns:
            The computed array of observations and the velocity for those observations
            if applicable. Each have shape (num_obs, 1). Also return dictionary of
            named outputs.
        """
        if self.additional_obs_name is not None:
            curr = additional_obs[self.additional_obs_name]
        else:
            if self.signal_idx is None:
                self.state_idx = info['state_space'].index(self.signal_name)
            curr = states[-1, self.state_idx]
        # pterm = curr - targets[:, -1, self.target_idx]
        curr = np.expand_dims(curr, axis=0)  # (1,)
        target = targets[states.shape[0]-1, self.target_idx]  # scalar
        pterm = curr - target
        # curr = np.expand_dims(curr, axis=0)
        # pterm = curr[:, np.newaxis] - targets[(states.shape[0]-1):, self.target_idx] #pterm is calculated for all target terms
        if self.invert_sign:
            pterm *= -1
        return (
            pterm.reshape(-1, 1),
            None,
            # {f'{self.signal_name}_pterm': pterm.reshape(-1, 1)},
            {f'{self.signal_name}_pterm': pterm},
        )

    @property
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""
        return 1

class DTerm(ObservationCalculator):
    """Calculate the D term, i.e. error"""

    def __init__(
        self,
        signal_name: str,
        target_idx: int,
        lookback: int = 1,
        dt: float = 1,
        obs_idx: Optional[int] = None,
        additional_obs_name: Optional[int] = None,
        invert_sign=False,
    ):
        """Constructor.

        Args:
            signal_name: The signal of the P, I, and D terms.
            target_idx: The index of the target corresponding to the signal name.
            lookback
            obs_idx: the index in the observation to track. If this is provided we
                will look at this rather than the signal_name.
            additional_obs_idx: If the signal is an additional observation, then
                specify the name.
        """
        self.signal_name = signal_name
        self.target_idx = target_idx
        self.lookback = lookback
        self.dt = dt
        self.additional_obs_name = additional_obs_name
        self.obs_idx = obs_idx
        if ((obs_idx is not None and additional_obs_name is None)
                or (obs_idx is None and additional_obs_name is not None)):
            raise ValueError('If observation index is sepcified the name must also be.')
        self.signal_idx = None
        self.invert_sign = invert_sign

    def compute_observation(
            self,
            obs: Optional[np.ndarray],
            states: np.ndarray,
            actuators: np.ndarray,
            next_actuators: np.ndarray,
            targets: Optional[np.ndarray],
            info: Dict[str, Any],
            additional_obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Compute the observation.

        Args:
            obs: The previous observations with state
                (num_obs, h + 1, obs dim)
            states: The states seen up to this point with shape
                (num_states, h + 2, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.
            info: The data information dict.
            additional_obs: The additional observations made so far.

        Returns:
            The computed array of observations and the velocity for those observations
            if applicable. Each have shape (num_obs, 1). Also return dictionary of
            named outputs.
        """
        if self.additional_obs_name is not None:
            curr = additional_obs[self.additional_obs_name]
            if obs is None:
                last = curr
            else:
                last_idx = -self.lookback if obs.shape[1] >= self.lookback else 0
                last = obs[:, last_idx, self.obs_idx]
        else:
            if self.signal_idx is None:
                self.state_idx = info['state_space'].index(self.signal_name)
            last_idx = -self.lookback if states.shape[1] >= self.lookback else 0
            curr = states[:, -1, self.state_idx]
            last = states[:, last_idx, self.state_idx]
        dterm = (curr - last) / self.dt
        if self.invert_sign:
            dterm *= -1
        return (
            dterm.reshape(-1, 1),
            None,
            {f'{self.signal_name}_{self.lookback}_dterm': dterm.reshape(-1, 1)},
        )

    @property
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""
        return 1

class SimpleDTerm(ObservationCalculator):
    """Calculate the D term, i.e. error
    Rohit - This is similar to above DTerm but does not use lookback and uses only current and previous timestep
    This is how we will implement D term for PCS in PACMAN
    """

    def __init__(
        self,
        signal_name: str,
        dt: float = 1,
        obs_idx: Optional[int] = None,
        additional_obs_name: Optional[int] = None,
        invert_sign=False,
    ):
        """Constructor.

        Args:
            signal_name: The signal of the P, I, and D terms.
            dt: The amount of time change.
            obs_idx: the index in the observation to track. If this is provided we
                will look at this rather than the signal_name.
            additional_obs_idx: If the signal is an additional observation, then
                specify the name.
        """
        self.signal_name = signal_name
        self.last_error = None
        self.dt = dt
        self.additional_obs_name = additional_obs_name
        self.obs_idx = obs_idx
        if ((obs_idx is not None and additional_obs_name is None)
                or (obs_idx is None and additional_obs_name is not None)):
            raise ValueError('If observation index is sepcified the name must also be.')
        self.signal_idx = None
        self.invert_sign = invert_sign
        self.name = 'SimpleDTerm'

    def reset(self):
        """Reset the last error."""
        self.last_error = None

    def compute_observation(
            self,
            obs: Optional[np.ndarray],
            states: np.ndarray,
            actuators: np.ndarray,
            next_actuators: np.ndarray,
            targets: Optional[np.ndarray],
            info: Dict[str, Any],
            additional_obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Compute the observation.

        Args:
            obs: The previous observations with state
                (num_obs, h + 1, obs dim)
            states: The states seen up to this point with shape
                (num_states, h + 2, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.
            info: The data information dict.
            additional_obs: The additional observations made so far.

        Returns:
            The computed array of observations and the velocity for those observations
            if applicable. Each have shape (num_obs, 1). Also return dictionary of
            named outputs.
        """
        if self.additional_obs_name is not None:
            curr = additional_obs[self.additional_obs_name]

        else:
            if self.signal_idx is None:
                self.state_idx = info['state_space'].index(self.signal_name)

            curr = states[:, -1, self.state_idx] - targets[:, -1, self.state_idx]
        
        # Initialize or reinitialize if batch size changed
        if self.last_error is None or len(self.last_error) != len(states):
            self.last_error = np.zeros(len(states))
        
        dterm = (curr - self.last_error) / self.dt
        self.last_error = curr

        if self.invert_sign:
            dterm *= -1
        return (
            dterm.reshape(-1, 1),
            None,
            {f'{self.signal_name}_dterm': dterm.reshape(-1, 1)},
        )

    @property
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""
        return 1


class FiniteHorizonITerm(ObservationCalculator):
    """Calculate the P, I, and D term over a fixed window"""

    def __init__(
        self,
        signal_name: str,
        obs_idx: Optional[int] = None,
        additional_obs_name: Optional[int] = None,
        window_length: int = 7,
        invert_sign=False,
    ):
        """Constructor.

        Args:
            signal_name: The signal of the P, I, and D terms.
            target_idx: The index of the target corresponding to the signal name.
            obs_idx: the index in the observation to track. If this is provided we
                will look at this rather than the signal_name.
            additional_obs_idx: If the signal is an additional observation, then
                specify the name.
            window_length: The window for how long to calculate these terms.
            d_indices: The indices of the derivatives to compute.
        """
        self.signal_name = signal_name
        self.additional_obs_name = additional_obs_name
        self.obs_idx = obs_idx
        self.window_length = window_length
        if ((obs_idx is not None and additional_obs_name is None)
                or (obs_idx is None and additional_obs_name is not None)):
            raise ValueError('If observation index is sepcified the name must also be.')
        self.signal_idx = None
        self.invert_sign = invert_sign


class ITermCalculator(ObservationCalculator):
    """Calculate the I term of a PID controller."""

    def __init__(
        self,
        signal_name: str,
        target_idx: int,
        additional_obs_name: Optional[int] = None,
        dt: float = 0.1,
        invert_sign=False,
    ):
        """Constructor.

        Args:
            signal_name: The signal to get the I statistic for.
            target_idx: The index of the target corresponding to the signal name.
            additional_obs_idx: If the signal is an additional observation, then
                specify the name.
            dt: The amount of time change.
        """
        self.signal_name = signal_name
        self.target_idx = target_idx
        self.additional_obs_name = additional_obs_name
        self.signal_idx = None
        self.dt = dt
        self.iterm = None
        self.invert_sign = invert_sign

    def reset(self):
        """Set the i term to 0."""
        self.iterm = None

    def compute_observation(
            self,
            obs: Optional[np.ndarray],
            states: np.ndarray,
            actuators: np.ndarray,
            next_actuators: np.ndarray,
            targets: Optional[np.ndarray],
            info: Dict[str, Any],
            additional_obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Compute the observation.

        Args:
            obs: The previous observations with state
                (num_obs, h + 1, obs dim)
            states: The states seen up to this point with shape
                (num_states, h + 2, state dim).
            actuators: The actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            next_actuators: The next actuators up to this point with shape
                (num_actuators, h + 2, actuator dim)
            targets: The targets up to this point. None if there is not targets in
                observations.
            info: The data information dict.
            additional_obs: The additional observations made so far.

        Returns:
            The computed array of observations and the velocity for those observations
            if applicable. Each have shape (num_obs, 1). Also return dictionary of
            named outputs.
        """
        assert targets.shape[1] < 3, 'Future targets not supported' # -- Rohit : only implemented for P term
        if self.additional_obs_name is not None:
            curr = additional_obs[self.additional_obs_name]
        else:
            if self.signal_idx is None:
                self.state_idx = info['state_space'].index(self.signal_name)
            curr = states[-1, self.state_idx]
        if self.iterm is None:
            self.iterm = np.zeros(len(states))
        self.iterm += self.dt * (curr - targets[-1, self.target_idx])
        iterm_to_return = deepcopy(self.iterm)
        if self.invert_sign:
            iterm_to_return *= -1
        return (
            iterm_to_return.reshape(-1, 1),
            None,
            {f'{self.signal_name}_iterm': self.iterm.reshape(-1, 1)},
        )

    @property
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""
        return 1
