"""
Utility functions for dynamics.

Author: Ian Char
Date: December 15, 2021
"""
from typing import Any, Dict

import numpy as np


def reconstruct_profile_from_state(
        profile_name: str,
        states: np.ndarray,
        info: Dict[str, Any],
        target_index: list,
        unnormalize: bool = True
) -> np.ndarray:
    """Reconstruct profile from PCA components.
    Args:
        profile_name: The name of the profile to reconstruct.
        states: The states to decode from.
        info: The info dictionary.
        unnormalize: Whether to unnormalize the resulting profile.
    Returns:
        The reconstruction.
    """
    if profile_name == 'temp':
        prof_idxs = [i for i in range(len(info['state_space']))
                     if (profile_name in info['state_space'][i]
                         and 'velocity' not in info['state_space'][i]
                         and 'itemp' not in info['state_space'][i])]
    else:
        prof_idxs = [i for i in range(len(info['state_space']))
                     if (profile_name in info['state_space'][i]
                         and 'velocity' not in info['state_space'][i]
                         # Hack for there being a dens and density.
                         and 'density' not in info['state_space'][i])]
    if target_index is not None:
        prof_idxs = target_index
    if f'{profile_name}_component1' in info['normalization_dict']:
        iqrs, medians = [], []
        for cnum in range(1, len(prof_idxs) + 1):
            iqrs.append(info['normalization_dict']\
                            [f'{profile_name}_component{cnum}']['iqr'])
            medians.append(info['normalization_dict']\
                            [f'{profile_name}_component{cnum}']['median'])
        iqrs = np.array(iqrs).reshape(1, -1)
        medians = np.array(medians).reshape(1, -1)
        encode = states[:, prof_idxs] * iqrs + medians
    else:
        encode = states[:, prof_idxs]
    profiles = np.dot(encode, info['pca_components'][profile_name]['components']) \
        + info['pca_components'][profile_name]['mean']
    if unnormalize:
        if info['normalization_dict'][profile_name]['method'] =='RobustScaler':
            profiles = (profiles * info['normalization_dict'][profile_name]['iqr'] + 
                info['normalization_dict'][profile_name]['median'])
        elif info['normalization_dict'][profile_name]['method'] == 'MinMax':

                armax = info['normalization_dict'][profile_name]['armax'].reshape(1, -1)
                armin = info['normalization_dict'][profile_name]['armin'].reshape(1, -1)
                profiles = profiles * (armax - armin) + armin

        else:
            raise ValueError(f'Unknown normalization method: {info["normalization_dict"][profile_name]["method"]}')
            
    return profiles
