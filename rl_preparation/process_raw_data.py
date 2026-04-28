import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
from typing import List, Optional

import rl_preparation as rp
from rl_preparation import state_actuator_spaces as sas
from envs.utils.data_preprocess import get_raw_data, store_offlinerl_dataset
from rl_preparation.paths import (
    evaluation_model_dir as _evaluation_model_dir,
    full_data_path as _full_data_path,
    il_data_path as _il_data_path,
    processed_data_dir as _processed_data_dir,
    raw_data_dir as _raw_data_dir,
    rl_data_path as _rl_data_path,
    tracking_test_data_path as _tracking_test_data_path,
    tracking_val_data_path as _tracking_val_data_path,
    training_model_dir as _training_model_dir,
)



action_bound_file = os.getenv("OFFLINERLKIT_ACTION_BOUND_FILE", "noshape_gas_flattop.yaml")
state_bound_file = os.getenv("OFFLINERLKIT_STATE_BOUND_FILE", "noshape_gas_flattop.yaml")
reference_shot = int(os.getenv("OFFLINERLKIT_REFERENCE_SHOT", "161409"))
warmup_steps = int(os.getenv("OFFLINERLKIT_WARMUP_STEPS", "0"))
change_every = int(os.getenv("OFFLINERLKIT_CHANGE_EVERY", "50"))
code_base = os.getenv("OFFLINERLKIT_CODE_BASE", "new")

# Backward-compatible string aliases that existing scripts can import.
raw_data_dir = str(_raw_data_dir)
training_model_dir = str(_training_model_dir)
evaluation_model_dir = str(_evaluation_model_dir)
save_data_dir = str(_processed_data_dir)
full_data_path = str(_full_data_path)
rl_data_path = str(_rl_data_path)
il_data_path = str(_il_data_path)
tracking_val_data_path = str(_tracking_val_data_path)
tracking_test_data_path = str(_tracking_test_data_path)


def _resolve_tracking_data_path() -> str:
    tracking_split = os.getenv("OFFLINERLKIT_TRACKING_SPLIT", "val").strip().lower()
    if tracking_split == "test":
        return tracking_test_data_path
    return tracking_val_data_path


tracking_data_path = _resolve_tracking_data_path()


def _parse_shot_list(value: Optional[str]) -> List[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _default_shot_window(center: int, radius: int) -> List[int]:
    return list(range(center - radius, center + radius + 1))


def _resolve_shot_list(env_name: str, fallback: List[int]) -> List[int]:
    shot_list = _parse_shot_list(os.getenv(env_name))
    return shot_list if shot_list else fallback


def main() -> None:
    os.makedirs(save_data_dir, exist_ok=True)

    task_name = os.getenv("OFFLINERLKIT_TASK", "temp")
    rp.configure_task(task_name)

    shot_radius = int(os.getenv("OFFLINERLKIT_SHOT_RADIUS", "200"))
    rl_shot_list = _resolve_shot_list(
        "OFFLINERLKIT_RL_SHOTS",
        _default_shot_window(reference_shot, shot_radius),
    )
    il_shot_list = _resolve_shot_list(
        "OFFLINERLKIT_IL_SHOTS",
        _default_shot_window(reference_shot, shot_radius),
    )
    tracking_shot_list = _resolve_shot_list(
        "OFFLINERLKIT_TRACKING_SHOTS",
        [reference_shot, reference_shot + 1, reference_shot + 2],
    )

    device_name = os.getenv("OFFLINERLKIT_DEVICE")
    device = (
        torch.device(device_name)
        if device_name
        else torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    )

    all_shots = sorted(set(rl_shot_list) | set(il_shot_list) | set(tracking_shot_list))
    offline_dst = get_raw_data(
        raw_data_dir,
        action_bound_file,
        state_bound_file,
        all_shots,
        warmup_steps,
    )
    tracking_output_path = _resolve_tracking_data_path()
    store_offlinerl_dataset(
        offline_dst,
        training_model_dir,
        rl_data_path,
        il_data_path,
        tracking_output_path,
        rl_shot_list,
        il_shot_list,
        tracking_shot_list,
        device,
        code_base,
    )

if __name__ == "__main__":
    main()
