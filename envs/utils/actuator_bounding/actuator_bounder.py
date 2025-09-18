"""
General class for bound actuators.

Author: Ian Char
Date: July 15, 2022
"""
import abc
from typing import Any, Dict, Sequence

import numpy as np


class ActuatorBounder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def bound_actuators(
        self,
        actuators: np.ndarray,
        next_actuators: np.ndarray,
        time_idx: int,
        info: Dict[str, Any],
    ) -> None:
        """Alter the next_actuators so that it is physically viable value.

        Args:
            actuators: The history of actuators as shape
                (num_unrolls, horizon + 1, actuator_dim)
            next_actuators: The history of next actutor change to be applied.
                Has shape (num_unrolls, horizon, num actuator signals). This is
                altered in place.
            time_idx: The index in time to bound.
            info: The information dictionary.
        """


class CompositeActuatorBounder(ActuatorBounder):

    def __init__(self, bounders: Sequence[ActuatorBounder]):
        """Constructor.

        Args:
            bounders: The bounders to use.
        """
        self.bounders = bounders

    def bound_actuators(
        self,
        actuators: np.ndarray,
        next_actuators: np.ndarray,
        time_idx: int,
        info: Dict[str, Any],
    ) -> None:
        """Alter the next_actuators so that it is physically viable value.

        Args:
            actuators: The history of actuators as shape
                (num_unrolls, horizon + 1, actuator_dim)
            next_actuators: The history of next actutor change to be applied.
                Has shape (num_unrolls, horizon, num actuator signals). This is
                altered in place.
            time_idx: The index in time to bound.
            info: The information dictionary.
        """
        for bounder in self.bounders:
            bounder.bound_actuators(actuators, next_actuators, time_idx, info)


class NoActuatorBounding(ActuatorBounder):

    def bound_actuators(
        self,
        actuators: np.ndarray,
        next_actuators: np.ndarray,
        time_idx: int,
        info: Dict[str, Any],
    ) -> None:
        """Alter the next_actuators so that it is physically viable value.

        Args:
            actuators: The history of actuators as shape
                (num_unrolls, horizon + 1, actuator_dim)
            next_actuators: The history of next actutor change to be applied.
                Has shape (num_unrolls, horizon, num actuator signals). This is
                altered in place.
            time_idx: The index in time to bound.
            info: The information dictionary.
        """
        return


class CobeamBounder(ActuatorBounder):

    def bound_actuators(
        self,
        actuators: np.ndarray,
        next_actuators: np.ndarray,
        time_idx: int,
        info: Dict[str, Any],
    ) -> None:
        "Copy outputs from pinj to tinj, should be done in normed env space only"
        pinj_idx = info['actuator_space'].index('pinj')
        tinj_idx = info['actuator_space'].index('tinj')
        actuators[:, time_idx, tinj_idx] = actuators[:, time_idx, pinj_idx]
        next_actuators[:, time_idx, pinj_idx] = next_actuators[:, time_idx, tinj_idx]

        return
        
