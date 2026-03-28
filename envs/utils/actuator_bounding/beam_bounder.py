"""
Bounders for the neutral beams.

Author: Ian Char
Date: July 15, 2022
"""
from typing import Any, Dict

import numpy as np

from envs.utils.actuator_bounding.actuator_bounder import ActuatorBounder
from envs.utils.actuator_bounding.beam import prepare_d3d_beams


class D3dTotalPowerTorqueBounding(ActuatorBounder):
    """Total power and torque bounding assuming that voltages and perveances are
       kept the same and only duty cycles are changed.
    """

    def __init__(
        self,
        voltages: Dict[str, float],
        perveances: Dict[str, float],
        rtans: Dict[str, float],
        min_duty_cycle: Dict[str, float],
    ):
        """Constructor.

        Args:
            voltages: Map from beam to the fixed voltage.
            perveances: Map from the beam to the fixed perveance.
            rtans: Map from the beam to the rtan value.
            min_duty_cycle: The minimum duty cycle needed at any given time. There may
                be conditions on this such as 30L(or R) and 330L(or R) being on
                50% of the time for CER diagnostic.
        """
        beams = prepare_d3d_beams(rtans)
        self._is_info_preprocessed = False
        # Calculate max powers and torques.
        min_dcs, pinjs, tinjs = [[] for _ in range(3)]
        for bname in ('30L', '30R', '150L', '150R', '210L', '210R', '330L', '330R'):
            min_dcs.append(min_duty_cycle[bname])
            pinjs.append(beams[bname].calc_power(voltages[bname],
                                                 perveances[bname]) * 1e-3)
            tinjs.append(beams[bname].calc_torque(voltages[bname],
                                                  perveances[bname]))
        min_dcs = np.array(min_dcs)
        pinjs = np.array(pinjs)
        tinjs = np.array(tinjs)
        beam_slopes = tinjs / pinjs
        ordered_idxs = np.argsort(beam_slopes)
        self.min_total_power = np.sum(min_dcs * pinjs)
        self.max_total_power = np.sum(pinjs)
        off_tinj = np.sum(min_dcs * tinjs)
        # Create the bounding lines and the vertices of the polygon
        for bd_type, sort_order in [('lb', 1), ('ub', -1)]:
            domains, slopes, offsets, vertices = [[] for _ in range(4)]
            curr_pinj_bdry, curr_tinj_bdry = self.min_total_power, off_tinj
            domains.append(curr_pinj_bdry)
            for bidx in ordered_idxs[::sort_order]:
                slopes.append(beam_slopes[bidx])
                offsets.append(curr_tinj_bdry - beam_slopes[bidx] * curr_pinj_bdry)
                beam_pinj_avail = pinjs[bidx] * (1 - min_dcs[bidx])
                curr_pinj_bdry += beam_pinj_avail
                curr_tinj_bdry += beam_pinj_avail * beam_slopes[bidx]
                domains.append(curr_pinj_bdry)
                vertices.append([curr_pinj_bdry, curr_tinj_bdry])
            setattr(self, f'{bd_type}_domains', np.array(domains))
            setattr(self, f'{bd_type}_slopes', np.array(slopes))
            setattr(self, f'{bd_type}_offsets', np.array(offsets))
            setattr(self, f'{bd_type}_vertices', np.array(vertices))

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
        if not self._is_info_preprocessed:
            self._preprocess_info(info)
        curr_pinj = np.array([actuators[time_idx, self.pinj_actuator_idx]])
        curr_tinj = np.array([actuators[time_idx, self.tinj_actuator_idx]])
        next_pinj = curr_pinj + np.array([next_actuators[time_idx, self.pinj_next_actuator_idx]])
        # If next pinj is too big or small bound by min and max values.
        too_small_idxs = next_pinj < self.normed_min_total_power
        if too_small_idxs[0]:
            next_actuators[time_idx, self.pinj_next_actuator_idx] = \
                self.normed_min_total_power - curr_pinj[too_small_idxs] + 1e-6
        too_big_idxs = next_pinj > self.normed_max_total_power
        if too_big_idxs[0]:
            next_actuators[too_big_idxs, time_idx, self.pinj_next_actuator_idx] = \
                self.normed_max_total_power - curr_pinj[too_big_idxs] - 1e-6
        next_pinj = curr_pinj + np.array([next_actuators[time_idx, self.pinj_next_actuator_idx]])
        next_tinj = curr_tinj + np.array([next_actuators[time_idx, self.tinj_next_actuator_idx]])
        for btype in ['lb', 'ub']:
            # Figure out which of the requests violate the bounds.
            sgmts = np.argmin(
                next_pinj.reshape(-1, 1)
                >= getattr(self, f'{btype}_domains_normed').reshape(1, -1),
                axis=1) - 1
            slopes = np.array([getattr(self, f'{btype}_slopes_normed')[sg]
                               for sg in sgmts])
            offsets = np.array([getattr(self, f'{btype}_offsets_normed')[sg]
                                for sg in sgmts])
            tinj_bdry = next_pinj * slopes + offsets
            if btype == 'lb':
                violations = tinj_bdry > next_tinj
            else:
                violations = tinj_bdry < next_tinj
            if np.sum(violations):
                # Project onto each of the boundaries of the polygon.
                offsets = getattr(self, f'{btype}_offsets_normed')
                slopes = getattr(self, f'{btype}_slopes_normed')
                domains = getattr(self, f'{btype}_domains_normed')
                ortho_offsets = (next_tinj[violations].reshape(-1, 1)
                                 + 1 / slopes.reshape(1, -1)
                                 * next_pinj[violations].reshape(-1, 1))
                pinj_intersects = ((ortho_offsets - offsets.reshape(1, -1))
                                   / (slopes.reshape(1, -1) + 1
                                       / slopes.reshape(1, -1)))
                tinj_intersects = pinj_intersects * slopes + offsets
                # Figure out which of the boundaries we should project onto.
                invalid_projs = np.logical_or(
                    pinj_intersects < domains[:-1],
                    pinj_intersects > domains[1:]
                )
                pinj_intersects[invalid_projs] = np.inf
                vertices = getattr(self, f'{btype}_vertices_normed')
                pinj_intersects = np.hstack([
                    pinj_intersects,
                    vertices[:, 0].reshape(1, -1).repeat(len(pinj_intersects), axis=0)
                ])
                tinj_intersects = np.hstack([
                    tinj_intersects,
                    vertices[:, 1].reshape(1, -1).repeat(len(tinj_intersects), axis=0)
                ])
                proj_idxs = np.argmin(np.sqrt(
                    (pinj_intersects - next_pinj[violations].reshape(-1, 1)) ** 2
                    + (tinj_intersects - next_tinj[violations].reshape(-1, 1)) ** 2
                ), axis=1)
                idxs = np.arange(len(proj_idxs))
                if violations[0]:
                    next_actuators[time_idx, self.pinj_next_actuator_idx] = \
                        pinj_intersects[idxs, proj_idxs] - curr_pinj[violations]
                    next_actuators[time_idx, self.tinj_next_actuator_idx] = \
                        tinj_intersects[idxs, proj_idxs] - curr_tinj[violations]
        return next_actuators

    def _preprocess_info(self, info: Dict[str, Any]):
        """Preprocess information in the info dict.

        Args:
            info: The information dictionary.
        """
        #hacky
        if info['normalization_dict']['pinj']['method']=='StandardScaler':
            info['normalization_dict']['pinj']['median'] = info['normalization_dict']['pinj']['mean']
            info['normalization_dict']['pinj']['iqr'] = info['normalization_dict']['pinj']['std']

            
        for btype in ['lb', 'ub']:
            setattr(self, f'{btype}_domains_normed',
                    (getattr(self, f'{btype}_domains')
                     - info['normalization_dict']['pinj']['median'])
                    / info['normalization_dict']['pinj']['iqr'])
            setattr(self, f'{btype}_slopes_normed',
                    getattr(self, f'{btype}_slopes')
                    * info['normalization_dict']['pinj']['iqr']
                    / info['normalization_dict']['tinj']['iqr'])
            setattr(self, f'{btype}_offsets_normed',
                    (getattr(self, f'{btype}_offsets')
                     - info['normalization_dict']['tinj']['median'])
                    / info['normalization_dict']['tinj']['iqr']
                    + (getattr(self, f'{btype}_slopes')
                       * info['normalization_dict']['pinj']['median']
                       / info['normalization_dict']['tinj']['iqr']))
            vertices = getattr(self, f'{btype}_vertices')
            setattr(self, f'{btype}_vertices_normed',
                    np.hstack([
                        ((vertices[:, 0] - info['normalization_dict']['pinj']['median'])
                         / info['normalization_dict']['pinj']['iqr']).reshape(-1, 1),
                        ((vertices[:, 1] - info['normalization_dict']['tinj']['median'])
                         / info['normalization_dict']['tinj']['iqr']).reshape(-1, 1),
                    ]))
        self.normed_min_total_power = ((self.min_total_power
                                        - info['normalization_dict']['pinj']['median'])
                                       / info['normalization_dict']['pinj']['iqr'])
        self.normed_max_total_power = ((self.max_total_power
                                        - info['normalization_dict']['pinj']['median'])
                                       / info['normalization_dict']['pinj']['iqr'])
        self.pinj_actuator_idx = info['actuator_space'].index('pinj')
        self.pinj_next_actuator_idx = \
            info['next_actuator_space'].index('pinj_velocity')
        self.tinj_actuator_idx = info['actuator_space'].index('tinj')
        self.tinj_next_actuator_idx = \
            info['next_actuator_space'].index('tinj_velocity')
        self._is_info_preprocessed = True


if __name__ == '__main__':
    beams = ['30L', '30R', '150L', '150R', '210L', '210R', '330L', '330R']
    voltages = {beam: 75000 for beam in beams}
    perveances = {beam: 2.55 for beam in beams}
    rtans = {
        '30L': 1.149,
        '30R': 0.749,
        '150L': 1.149,
        '150R': 0.749,
        '210L': -0.749,
        '210R': -1.149,
        '330L': 1.149,
        '330R': 0.749,
    }
    min_duty = {
        '30L': 0.5,
        '30R': 0.0,
        '150L': 0.0,
        '150R': 0.0,
        '210L': 0.0,
        '210R': 0.0,
        '330L': 0.5,
        '330R': 0.0,
    }
    bounder = D3dTotalPowerTorqueBounding(voltages, perveances, rtans, min_duty)
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.rcParams.update({
        'font.size': 16,
        'legend.fontsize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    for lb_idx in range(len(bounder.lb_domains) - 1):
        xs = np.linspace(bounder.lb_domains[lb_idx], bounder.lb_domains[lb_idx + 1],
                         100)
        ys = bounder.lb_slopes[lb_idx] * xs + bounder.lb_offsets[lb_idx]
        plt.plot(xs, ys, color='blue')
    for ub_idx in range(len(bounder.ub_domains) - 1):
        xs = np.linspace(bounder.ub_domains[ub_idx], bounder.ub_domains[ub_idx + 1],
                         100)
        ys = bounder.ub_slopes[ub_idx] * xs + bounder.ub_offsets[ub_idx]
        plt.plot(xs, ys, color='blue', zorder=1)
    plt.xlabel('Total Power (kW)')
    plt.ylabel('Total Torque (Nm)')

    actuators = np.array([
        [7e3, 5],
        [9e3, 5],
        [13e3, 5],
        [7e3, 1.5],
        [3e3, 0.5],
        [5e3, 4],
        [6e3, 3],
    ])
    nexts = np.array([
        [7e3, 7],
        [13e3, 8],
        [15e3, 5.5],
        [9e3, 0.0],
        [2e3, -0.5],
        [4e3, 5],
        [7e3, 4],
    ])
    for ridx in range(len(actuators)):
        plt.plot([nexts[ridx, 0]],
                 [nexts[ridx, 1]],
                 marker='o', zorder=2)
    info = {
        'normalization_dict': {
            'pinj': {'median': 4754.914, 'iqr': 3218.9},
            'tinj': {'median': 3.446, 'iqr': 2.7268},
        },
        'actuator_space': ['pinj', 'tinj'],
        'next_actuator_space': ['pinj_velocity', 'tinj_velocity'],
    }
    next_actuators = nexts - actuators
    mu_vect = np.array([
        info['normalization_dict']['pinj']['median'],
        info['normalization_dict']['tinj']['median'],
    ]).reshape(1, -1)
    sig_vect = np.array([
        info['normalization_dict']['pinj']['iqr'],
        info['normalization_dict']['tinj']['iqr'],
    ]).reshape(1, -1)
    next_actuators = bounder.bound_actuators(
        ((actuators - mu_vect) / sig_vect).reshape(len(actuators), 1, 2),
        (next_actuators / sig_vect).reshape(len(actuators), 1, 2),
        0,
        info,
    ).squeeze(1) * sig_vect
    new_nexts = (next_actuators + actuators).reshape(len(actuators), 2)
    for ridx in range(len(actuators)):
        plt.scatter([new_nexts[ridx, 0]], [new_nexts[ridx, 1]], marker='x',
                    color='red', zorder=3, s=100, lw=3)
    # plt.show()
    plt.savefig("beam_bounds.png")
