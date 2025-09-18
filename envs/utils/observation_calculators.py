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
        curr = np.expand_dims(curr, axis=0)
        pterm = curr[:, np.newaxis] - targets[(states.shape[0]-1):, self.target_idx] #pterm is calculated for all target terms
        if self.invert_sign:
            pterm *= -1
        return (
            # pterm.reshape(-1, 1),
            pterm,
            None,
            # {f'{self.signal_name}_pterm': pterm.reshape(-1, 1)},
            {f'{self.signal_name}_pterm': pterm},
        )

    @property
    def num_obs_computed(self) -> int:
        """The number of observations computed by this maker."""
        return 1
