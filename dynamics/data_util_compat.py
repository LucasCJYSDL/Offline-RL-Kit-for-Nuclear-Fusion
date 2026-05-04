"""
Compatibility helpers ported from FusionControl data utilities.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def sort_by_continuous_snippets(
    data: Dict[str, np.ndarray],
    info: Dict[str, Any],
    min_amt_needed: int = 0,
    dt: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Split flat transition data into contiguous shot snippets.

    The original FusionControl implementation segments whenever the shot id
    changes or the time gap exceeds ``dt``.
    """
    del info  # Kept for compatibility with the original signature.

    shots: List[Dict[str, np.ndarray]] = []
    time_diffs = np.abs(data["time"][1:] - data["time"][:-1])
    if dt is None:
        dt = time_diffs[0]
    cuts = np.argwhere(
        np.logical_or(
            data["shotnum"][1:] != data["shotnum"][:-1],
            time_diffs > dt,
        )
    ).flatten() + 1

    if len(cuts) == 0:
        if len(data["shotnum"]) > min_amt_needed:
            shots.append({key: value.copy() for key, value in data.items()})
        return shots

    if cuts[0] >= min_amt_needed:
        shots.append({key: value[0:cuts[0]] for key, value in data.items()})

    for idx in range(len(cuts) - 1):
        left, right = cuts[idx], cuts[idx + 1]
        if right - left > min_amt_needed:
            shots.append({key: value[left:right] for key, value in data.items()})

    last_left = cuts[-1]
    if len(data["shotnum"]) - last_left > min_amt_needed:
        shots.append({key: value[last_left:] for key, value in data.items()})

    return shots
