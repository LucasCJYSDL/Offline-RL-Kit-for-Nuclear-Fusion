from importlib import import_module
from typing import Any

import rl_preparation as rp

from .version import __version__

AVAILABLE_ENVS = ("base", "profile_control", "profile_control_new", "fusion_env")
AVAILABLE_TASKS = rp.AVAILABLE_TASKS
AVAILABLE_TASK_ALIASES = tuple(sorted(rp.TASK_ALIASES.keys()))


def available_envs():
    return AVAILABLE_ENVS


def available_tasks():
    return AVAILABLE_TASKS


def resolve_task_name(task: str):
    return rp.resolve_task_name(task)


def load_bundle(
    env_id: str = "profile_control_new",
    task: str = "temp",
    device: Any = "cpu",
    *,
    is_il: bool = False,
    is_val: bool = True,
    load_hidden_states: bool = False,
):
    """Load the offline dataset, processor, env, and model directory together."""
    if device is None:
        try:
            import torch

            device = torch.device("cpu")
        except Exception:  # pragma: no cover - torch is an optional runtime dep here
            device = "cpu"

    task = resolve_task_name(task)
    module = import_module("rl_preparation.get_rl_data_envs")
    return module.get_rl_data_envs(
        env_id,
        task,
        device,
        is_il=is_il,
        is_val=is_val,
        load_hidden_states=load_hidden_states,
    )


def load_dataset(
    env_id: str = "profile_control_new",
    task: str = "temp",
    device: Any = "cpu",
    *,
    is_il: bool = False,
    is_val: bool = True,
    load_hidden_states: bool = False,
):
    """Return the offline dataset dict only."""
    offline_data, *_ = load_bundle(
        env_id=env_id,
        task=task,
        device=device,
        is_il=is_il,
        is_val=is_val,
        load_hidden_states=load_hidden_states,
    )
    return offline_data


def make_env(
    env_id: str = "profile_control_new",
    task: str = "temp",
    device: Any = "cpu",
    *,
    is_il: bool = False,
    is_val: bool = True,
    load_hidden_states: bool = False,
):
    """Return the configured evaluation environment."""
    _, _, env, _ = load_bundle(
        env_id=env_id,
        task=task,
        device=device,
        is_il=is_il,
        is_val=is_val,
        load_hidden_states=load_hidden_states,
    )
    return env


def make(
    env_id: str = "profile_control_new",
    task: str = "temp",
    device: Any = "cpu",
    *,
    is_il: bool = False,
    is_val: bool = True,
    load_hidden_states: bool = False,
):
    """D4RL-style alias for make_env."""
    return make_env(
        env_id=env_id,
        task=task,
        device=device,
        is_il=is_il,
        is_val=is_val,
        load_hidden_states=load_hidden_states,
    )
